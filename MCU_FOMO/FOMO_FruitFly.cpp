/*------------------------------------------------------------
 * STM32F746 – FOMO Fruit Fly Detection (REWRITTEN VERSION)
 * Model: MobileNetV2 0.35 (96x96, INT8)
 *-----------------------------------------------------------*/

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <algorithm>

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"
#include "cmsis_os2.h"
#include "fomo_fruitfly_model_data.h"

extern int stdin_getchar(void);
int app_main(void);

#ifdef __cplusplus
}
#endif

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// --- CONFIGURATION ---
#define IMG_W 96
#define IMG_H 96
#define IMG_C 3 
#define IMG_SIZE (IMG_W * IMG_H * IMG_C)
#define GRID_W 12 
#define GRID_H 12
#define TENSOR_ARENA_SIZE (200 * 1024)
#define DETECTION_THRESHOLD 0.65f

// Communication Protocol
#define HANDSHAKE_TIMEOUT_MS 30000
#define IMAGE_RECEIVE_TIMEOUT_MS 10000
#define MAX_DETECTIONS 20

struct Detection {
    int x;
    int y;
    float confidence;
};

enum SystemState {
    STATE_INITIALIZING,
    STATE_READY,
    STATE_RECEIVING_IMAGE,
    STATE_PROCESSING,
    STATE_ERROR
};

alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;
static volatile SystemState system_state = STATE_INITIALIZING;

// Stats
static uint32_t total_frames_processed = 0;
static uint32_t total_errors = 0;

static void safe_printf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    fflush(stdout);
}

static void send_status(const char* status, const char* message = nullptr) {
    if (message) safe_printf("STATUS:%s:%s\r\n", status, message);
    else safe_printf("STATUS:%s\r\n", status);
}

/*-----------------------------------------------------------
 * Function: receive_and_quantize
 * Corrected to treat incoming bytes as unsigned 0-255.
 *-----------------------------------------------------------*/
static bool receive_and_quantize(int8_t* out_buffer, float scale, int32_t zero_point) {
    uint32_t bytes_received = 0;
    uint32_t start_tick = osKernelGetTickCount();
    
    while (bytes_received < IMG_SIZE) {
        if ((osKernelGetTickCount() - start_tick) > IMAGE_RECEIVE_TIMEOUT_MS) return false;
        
        int c = stdin_getchar();
        if (c >= 0) {
            // 1. Treat as unsigned byte (0-255)
            uint8_t pixel_val = (uint8_t)c;
            
            // 2. Normalize and Quantize immediately to save memory
            float normalized = (float)pixel_val / 255.0f;
            int32_t qval = (int32_t)round(normalized / scale) + zero_point;
            
            // 3. Clamp to INT8 range
            out_buffer[bytes_received++] = (int8_t)std::max(-128, std::min(127, (int)qval));
        }
    }
    return true;
}

static int find_detections(Detection* dets, float threshold) {
    if (!output) return 0;
    
    int count = 0;
    int8_t* data = output->data.int8;
    float out_scale = output->params.scale;
    int32_t out_zero = output->params.zero_point;
    int num_classes = output->dims->data[3]; 

    for (int y = 0; y < GRID_H; y++) {
        for (int x = 0; x < GRID_W; x++) {
            // Index for Class 1 (Fruit Fly)
            int idx = (y * GRID_W + x) * num_classes + 1;
            float score = (static_cast<float>(data[idx]) - out_zero) * out_scale;

            if (score >= threshold && count < MAX_DETECTIONS) {
                dets[count].x = (x * 8) + 4; // Center of 8x8 cell
                dets[count].y = (y * 8) + 4;
                dets[count].confidence = score;
                count++;
            }
        }
    }
    return count;
}

static void run_inference() {
    Detection detections[MAX_DETECTIONS];
    system_state = STATE_PROCESSING;
    
    uint32_t start_t = osKernelGetTickCount();
    TfLiteStatus status = interpreter->Invoke();
    uint32_t end_t = osKernelGetTickCount();
    
    if (status != kTfLiteOk) {
        send_status("ERROR", "Inference Failed");
        total_errors++;
        return;
    }
    
    int n = find_detections(detections, DETECTION_THRESHOLD);
    
    safe_printf("TIME:%lu\r\n", (end_t - start_t));
    safe_printf("COUNT:%d\r\n", n);
    
    for (int i = 0; i < n; i++) {
        safe_printf("FLY:%d:%d:%d\r\n", detections[i].x, detections[i].y, (int)(detections[i].confidence * 100));
    }
    
    total_frames_processed++;
}

static __NO_RETURN void inference_thread(void* arg) {
    (void)arg;
    float in_scale = input->params.scale;
    int32_t in_zero = input->params.zero_point;
    bool start_received = false;
    send_status("INITIALIZED");

    for (;;) {
        system_state = STATE_READY;
        safe_printf("READY\r\n");

        // Wait for START command
        char cmd[8] = {0};
        int cmd_ptr = 0;
        uint32_t start_wait = osKernelGetTickCount();


        while ((osKernelGetTickCount() - start_wait) < HANDSHAKE_TIMEOUT_MS) {
            if (start_received)break;
            int c = stdin_getchar();
            if (c < 0) { osDelay(5); continue; }
            if (c == '\n' || c == '\r') {
                if (strncmp(cmd, "START", 5) == 0) { start_received = true; break; }
                cmd_ptr = 0; memset(cmd, 0, sizeof(cmd));
            } else if (cmd_ptr < 6) {
                cmd[cmd_ptr++] = (char)c;
            }
        }

        if (!start_received) continue;

        send_status("ACK");
        system_state = STATE_RECEIVING_IMAGE;

        if (receive_and_quantize(input->data.int8, in_scale, in_zero)) {
            send_status("RECEIVED");
            run_inference();
            send_status("DONE");
            continue;
        } else {
            send_status("ERROR", "RX Timeout");
            total_errors++;
        }
        
        if (total_frames_processed % 5 == 0) {
            safe_printf("STATS:Frames=%lu,Errors=%lu\r\n", total_frames_processed, total_errors);
        }
    }
}

int app_main(void) {
    osKernelInitialize();
    osDelay(100);

    static tflite::MicroMutableOpResolver<12> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddConcatenation();
    resolver.AddPad();

    const tflite::Model* model = tflite::GetModel(g_fomo_fruitfly_model_data);
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();

    input = interpreter->input(0);
    output = interpreter->output(0);

    const osThreadAttr_t attr = {.name = "Inference", .stack_size = 8192, .priority = osPriorityNormal};
    osThreadNew(inference_thread, NULL, &attr);

    osKernelStart();
    for(;;);
}