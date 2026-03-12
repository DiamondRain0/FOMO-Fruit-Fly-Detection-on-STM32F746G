"""
FOMO TFLite Testing Script (PC Side) - FIXED
--------------------------------------------
1. Fixes int8 overflow in confidence calc.
2. Uses 'Point-in-Box' matching instead of IoU.
"""

import tensorflow as tf
import numpy as np
import cv2
import json
import os
import shutil
from pathlib import Path

# ================= CONFIGURATION =================
MODEL_PATH = "fomo_fruitfly.tflite"
TEST_DIR = Path("dataset/mff/test")
LABELS_FILE = Path("dataset/mff/test_labels.json")

IMG_W = 96
IMG_H = 96

# Thresholds
CONFIDENCE_THRESHOLD = 0.65

# Output folder for images with drawn boxes
DEBUG_OUTPUT_DIR = "debug_images"
# =================================================

def load_test_data():
    if not LABELS_FILE.exists():
        print(f"❌ Error: {LABELS_FILE} not found.")
        return []

    with open(LABELS_FILE, "r") as f:
        labels = json.load(f)

    files = labels.get("files", []) if isinstance(labels, dict) else labels
    samples = []
    
    for f in files:
        if f.get("category", "testing") == "testing":
            p = TEST_DIR / f["path"]
            if p.exists():
                samples.append({
                    "path": str(p),
                    "name": f["path"],
                    "boxes": f.get("boundingBoxes", [])
                })
    return samples

def is_point_in_box(point, box):
    """
    Checks if a prediction point (x,y) is inside a Ground Truth box.
    """
    px, py = point['x'], point['y']
    bx, by = box['x'], box['y']
    bw, bh = box['width'], box['height']
    
    # Check if point is inside the rectangle
    return (bx <= px <= bx + bw) and (by <= py <= by + bh)

def main():
    # 1. Setup
    if os.path.exists(DEBUG_OUTPUT_DIR):
        shutil.rmtree(DEBUG_OUTPUT_DIR)
    os.makedirs(DEBUG_OUTPUT_DIR)
    
    print(f"Loading TFLite model: {MODEL_PATH}")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']
    
    samples = load_test_data()
    print(f"Loaded {len(samples)} test images.")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    print(f"\n{'Image':<20} | {'Status':<10} | {'TP':<3} {'FP':<3} {'FN':<3}")
    print("-" * 55)

    for sample in samples:
        img_orig = cv2.imread(sample['path'])
        if img_orig is None: continue
        orig_h, orig_w = img_orig.shape[:2]
        
        # 2. Preprocess
        img_resized = cv2.resize(img_orig, (IMG_W, IMG_H))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        
        # Quantize Input
        if input_details['dtype'] == np.int8:
            img_input = (img_norm / input_scale) + input_zero_point
            img_input = np.clip(img_input, -128, 127).astype(np.int8)
        else:
            img_input = img_norm

        img_input = np.expand_dims(img_input, axis=0)

        # 3. Run Inference
        interpreter.set_tensor(input_details['index'], img_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])

        # 4. Decode FOMO
        grid_h, grid_w, num_classes = output_data.shape[1:]
        predictions = []

        for y in range(grid_h):
            for x in range(grid_w):
                # Class 1 is Fruit Fly
                # FIX 1: Cast to float BEFORE math to prevent Overflow
                raw_val = float(output_data[0, y, x, 1]) 
                
                confidence = (raw_val - output_zero_point) * output_scale
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Map to Original Coords
                    cx = x * 8 + 4
                    cy = y * 8 + 4
                    
                    real_x = int(cx * (orig_w / IMG_W))
                    real_y = int(cy * (orig_h / IMG_H))
                    
                    predictions.append({
                        'x': real_x,
                        'y': real_y,
                        'confidence': confidence
                    })

        # 5. Matching Logic (Point inside Box)
        gt_boxes = sample['boxes']
        tp = 0
        matched_gt_indices = set()
        
        # We need to track which predictions matched to color them
        pred_matches = [False] * len(predictions)

        for i, pred in enumerate(predictions):
            # Check this prediction against all UNMATCHED ground truths
            for j, box in enumerate(gt_boxes):
                if j in matched_gt_indices: continue # Don't double count GT
                
                if is_point_in_box(pred, box):
                    tp += 1
                    matched_gt_indices.add(j)
                    pred_matches[i] = True
                    break # This prediction is used, move to next
        
        fp = len(predictions) - tp
        fn = len(gt_boxes) - tp
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # 6. Draw Debug Image
        img_debug = img_orig.copy()
        
        # Draw GT (Green Boxes)
        for box in gt_boxes:
            cv2.rectangle(img_debug, 
                          (box['x'], box['y']), 
                          (box['x']+box['width'], box['y']+box['height']), 
                          (0, 255, 0), 2)
            
        # Draw Predictions (Dots/Circles)
        for i, pred in enumerate(predictions):
            # Yellow = Correct, Red = Wrong
            color = (0, 255, 255) if pred_matches[i] else (0, 0, 255)
            
            # Draw a circle at the predicted point
            cv2.circle(img_debug, (pred['x'], pred['y']), 5, color, -1)
            
            # Draw confidence text
            cv2.putText(img_debug, f"{pred['confidence']:.2f}", 
                        (pred['x']+8, pred['y']), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, sample['name']), img_debug)
        print(f"{sample['name'][:20]:<20} | Done       | {tp:<3} {fp:<3} {fn:<3}")

    # Final Stats
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*40)
    print(f"TFLite PC Test Results")
    print("="*40)
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")

if __name__ == "__main__":
    main()