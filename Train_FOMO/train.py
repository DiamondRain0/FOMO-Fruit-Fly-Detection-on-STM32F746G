import argparse
import importlib
import tensorflow as tf
from keras import optimizers, Model
from keras.metrics import OneHotIoU, CategoricalAccuracy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from utils.callbacks import cosine_annealing_with_warmup
import numpy as np

import backbones
from configs import config, update_config
TF_ENABLE_ONEDNN_OPTS=0

# --- CUSTOM LOSS FUNCTION ---
def get_weighted_loss(weights_list):
    """
    Manual Weighted Cross Entropy.
    weights_list: [1.0, 100.0] -> Missing class 1 is 100x worse.
    """
    class_weights = tf.constant(weights_list, dtype=tf.float32)
    
    def weighted_cce(y_true, y_pred):
        # Standard loss
        cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        # Calculate weight per pixel based on Ground Truth
        # If GT is Fly, weight is 100. If GT is Bg, weight is 1.
        weight_map = tf.reduce_sum(y_true * class_weights, axis=-1)
        return cce * weight_map

    return weighted_cce

# --- DEBUG CALLBACK ---
class DebugPreds(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\n[Epoch {epoch+1}] Loss: {logs.get('loss'):.4f} | IoU: {logs.get('iou'):.4f}")
        if logs.get('iou', 0) < 0.0001:
            print(">>> WARNING: IoU is 0. Model is ignoring the flies.")
        else:
            print(">>> PROGRESS: IoU > 0. Flies are being detected!")

def parse_args():
    parser = argparse.ArgumentParser(description="Train FOMO network")
    parser.add_argument("--cfg", help="config file", default="configs/mff/mff_mobilenetv2.yaml", type=str)
    args = parser.parse_args()
    update_config(config, args)
    return args

def get_model():
    # Force MobileNetV2 with alpha 0.35 (Standard for FOMO/STM32)
    # Weights=None because we changed size to 96x96
    print(f"Building MobileNetV2 alpha=0.35 for {config.DATASET.NUM_CLASSES} classes...")
    model = backbones.MobileFOMOv2(
        config.DATASET.IMAGE_SIZE, 0.35, config.DATASET.NUM_CLASSES, None
    )
    return model

def main():
    args = parse_args()
    
    # Dynamic Import of Dataloader
    # Ensure 'dataloaders/mff.py' exists and has class 'MffDataset'
    from dataloaders.mff import MFFDataset  # <--- CHANGED
    
    print("--- Loading Data ---")
    train_gen = MFFDataset(config, split=config.DATASET.TRAIN_SET, augment=True) # <--- CHANGED
    val_gen = MFFDataset(config, split=config.DATASET.VALIDATION_SET, augment=False) # <--- CHANGED
    
    print("--- Building Model ---")
    model = get_model()

    # DEADLY WEIGHTS: [Background, Fly]
    # If it still fails, increase 100.0 to 500.0
    loss_fn = get_weighted_loss([1.0, 15.0])
    
    optim = optimizers.Adam(learning_rate=config.TRAIN.LR)
    
    model.compile(
        loss=loss_fn,
        optimizer=optim,
        metrics=[
            OneHotIoU(config.DATASET.NUM_CLASSES, range(1, config.DATASET.NUM_CLASSES), "iou"),
            CategoricalAccuracy(name="acc")
        ]
    )

    callbacks = [
        ModelCheckpoint(config.TRAIN.BEST_SAVE_PATH, monitor="val_iou", mode="max", save_best_only=True, verbose=1),
        DebugPreds()
    ]

    print("--- Starting Training ---")
    model.fit(
        train_gen,
        epochs=config.TRAIN.NUM_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )

if __name__ == "__main__":
    main()