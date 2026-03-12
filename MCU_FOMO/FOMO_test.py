import serial
import serial.tools.list_ports
import time
import json
import csv
import sys
import os
import cv2
import numpy as np
from datetime import datetime

# ============= CONFIGURATION =============
COM_PORT = "COM3"  
BAUD_RATE = 115200

IMG_W = 96
IMG_H = 96
IMG_SIZE = IMG_W * IMG_H * 3 
HANDSHAKE_TIMEOUT = 30  
IMAGE_SEND_TIMEOUT = 15  
RESPONSE_TIMEOUT = 10   

# Paths from your code
JSON_PATH = os.path.join("py", "mff", "test_labels.json")
IMG_DIR = os.path.join("py", "mff", "test")

# ============= SERIAL LOGGING =============
LOG_FILE = f"mcu_log_{int(time.time())}.txt"

def log_mcu(message, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] [{level}] {message}"
    print(line)
    # Append to log file
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_serial_port():
    global COM_PORT
    try:
        ser = serial.Serial(
            port=COM_PORT,
            baudrate=BAUD_RATE,
            timeout=0.01, # Fast polling
            dsrdtr=False, 
            rtscts=False
        )
        ser.set_buffer_size(rx_size=128000, tx_size=128000)
        time.sleep(1) # Wait for stable connection
        ser.reset_input_buffer()
        log_mcu(f"Connected to {COM_PORT}", "SYSTEM")
        return ser
    except Exception as e:
        log_mcu(f"Failed to open port: {e}", "ERROR")
        sys.exit(1)

def wait_for_line(ser, contains_text, timeout_sec):
    """Polls the UART until a specific string is found or timeout occurs."""
    start_time = time.time()
    line_buffer = ""
    
    while (time.time() - start_time) < timeout_sec:
        if ser.in_waiting:
            try:
                char = ser.read(1).decode('utf-8', errors='ignore')
                if char == '\n':
                    clean_line = line_buffer.strip()
                    if clean_line:
                        log_mcu(f"MCU >> {clean_line}", "UART")
                        if contains_text in clean_line:
                            return clean_line
                    line_buffer = ""
                else:
                    line_buffer += char
            except Exception:
                pass
    return None

# ============= GEOMETRY HELPERS (POINT CHECK) =============

def is_point_in_box(point, box):
    """
    Checks if a prediction point (x,y) is strictly inside a Ground Truth box.
    """
    px, py = point['x'], point['y']
    bx, by = box['x'], box['y']
    bw, bh = box['width'], box['height']
    
    # Check if point is inside the rectangle boundaries
    return (bx <= px <= bx + bw) and (by <= py <= by + bh)

# ============= MAIN TEST LOGIC =============

def test_single_image(ser, sample):
    img_orig = cv2.imread(sample['path'])
    if img_orig is None: return None
    
    orig_h, orig_w = img_orig.shape[:2]
    img_resized = cv2.resize(img_orig, (IMG_W, IMG_H))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    log_mcu(f"TESTING: {sample['name']}", "TEST")

    # 1. Sync with MCU
    ser.reset_input_buffer()
    if not wait_for_line(ser, "READY", HANDSHAKE_TIMEOUT):
        log_mcu("MCU timed out waiting for READY", "ERROR")
        return None
    
    # 2. Handshake
    ser.write(b"START\n")
    if not wait_for_line(ser, "ACK", 5):
        return None
    
    # 3. Send Image with Flow Control
    img_bytes = img_rgb.tobytes()
    chunk_size = 256
    for i in range(0, len(img_bytes), chunk_size):
        ser.write(img_bytes[i:i+chunk_size])
        time.sleep(0.002) # 2ms gap to prevent MCU overflow
    
    # 4. Result Collection
    predictions = []
    inference_time = 0
    collection_start = time.time()
    
    while (time.time() - collection_start) < RESPONSE_TIMEOUT:
        line = wait_for_line(ser, "", 0.5) # Read ANY line as it comes
        if not line: continue
        
        if "TIME:" in line:
            try:
                inference_time = int(line.split("TIME:")[1].strip())
            except: pass

        elif "FLY:" in line:
            # Format: FLY:x:y:conf
            parts = line.split(":")
            if len(parts) >= 3: # Expecting at least FLY:x:y
                px, py = int(parts[1]), int(parts[2])
                
                scale_x, scale_y = orig_w / IMG_W, orig_h / IMG_H
                
                # --- POINT LOGIC ---
                # Store the EXACT center point scaled to original image
                predictions.append({
                    'x': int(px * scale_x),
                    'y': int(py * scale_y)
                })

        elif "DONE" in line:
            break

    # 5. Evaluate (Point Check Logic)
    gt_boxes = sample['boxes']
    tp = 0
    used_gt = [False] * len(gt_boxes)
    
    # Check every prediction against every GT box
    for pred in predictions:
        for i, gt in enumerate(gt_boxes):
            if not used_gt[i]:
                # Use Point-In-Box instead of IoU
                if is_point_in_box(pred, gt):
                    tp += 1
                    used_gt[i] = True
                    break # This prediction matched a fly, stop checking it against others
    
    fp = len(predictions) - tp
    fn = len(gt_boxes) - tp
    
    res = {'tp': tp, 'fp': fp, 'fn': fn, 'ms': inference_time}
    log_mcu(f"RESULT: {tp} TP, {fp} FP in {inference_time}ms", "INFO")
    return {**sample, **res}

def main():
    ser = get_serial_port()
    
    # Load JSON
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    samples = [{'path': os.path.join(IMG_DIR, x['path']), 'name': x['path'], 'boxes': x['boundingBoxes']} for x in data['files']]

    filename = f"results_{int(time.time())}.csv"
    print(f"Saving results to {filename}...")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "GT_Count", "TP", "FP", "FN", "Time_ms"])

        for sample in samples:
            result = test_single_image(ser, sample)
            if result:
                writer.writerow([
                    result['name'], 
                    len(result['boxes']), 
                    result['tp'], 
                    result['fp'], 
                    result['fn'], 
                    result['ms']
                ])
            time.sleep(0.5) # Guard time between images

    ser.close()

if __name__ == "__main__":
    main()