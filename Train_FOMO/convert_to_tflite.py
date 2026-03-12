import tensorflow as tf
import numpy as np
import os
from dataloaders.mff import MFFDataset
from configs import config

# --- FORCE CONFIGURATION TO MATCH TRAINING ---
# The script was defaulting to 400 or 224. We must force it to 96.
config.DATASET.IMAGE_SIZE = [96, 96]
config.DATASET.ROOT = 'dataset'
config.DATASET.TRAIN_SET = 'train'
config.DATASET.VALIDATION_SET = 'test'
config.DATASET.NUM_CLASSES = 2
config.TRAIN.BATCH_SIZE = 1 # Batch size 1 for conversion is safer
# ---------------------------------------------

def main():
    # 1. Load the trained Keras model
    print("Loading Keras model...")
    try:
        model = tf.keras.models.load_model('best.keras', compile=False)
    except OSError:
        print("Error: 'best.keras' not found. Make sure you trained the model first!")
        return

    # 2. Create Representative Dataset (Required for Quantization)
    def representative_data_gen():
        print("Generating representative data (96x96)...")
        # Use the test set
        val_gen = MFFDataset(config, split='test', augment=False)
        
        # Take 100 samples
        count = 0
        limit = 100
        
        # MFFDataset yields (batch_x, batch_y)
        # We only need batch_x
        for i in range(len(val_gen)):
            batch_x, _ = val_gen[i]
            
            for img in batch_x:
                if count >= limit:
                    return
                
                # Ensure shape is [1, 96, 96, 3] and float32
                img = img.astype(np.float32)
                img = np.expand_dims(img, axis=0)
                
                yield [img]
                count += 1

    # 3. Configure TFLite Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable optimizations (Quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    # Force the specific hardware ops for STM32 (INT8)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # IMPORTANT: Set Input/Output to int8 for MCU
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Converting model...")
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"Conversion Failed: {e}")
        return

    # 4. Save the file
    output_path = 'fomo_fruitfly.tflite'
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Success! Model saved to {output_path}")
    print(f"Size: {len(tflite_model) / 1024:.2f} KB")

if __name__ == "__main__":
    main()