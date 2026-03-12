import os
import json
import cv2
import numpy as np
import copy
import tensorflow as tf
from tensorflow import keras

class MFFDataset(keras.utils.Sequence):
    def __init__(self, config, split='train', augment=False, shuffle=True, workers=1, use_multiprocessing=False):
        self.config = config
        self.split = split
        self.augment = augment
        self.shuffle = shuffle
        
        # Paths
        self.root_dir = os.path.join(config.DATASET.ROOT, 'mff')
        self.img_dir = os.path.join(self.root_dir, split)
        self.json_path = os.path.join(self.root_dir, f'{split}_labels.json')
        
        # Config params
        self.input_size = config.DATASET.IMAGE_SIZE[0] # e.g. 96
        self.num_classes = config.DATASET.NUM_CLASSES
        self.batch_size = config.TRAIN.BATCH_SIZE
        
        # FOMO Output Grid Size (MobileNetV2 usually cuts by 8)
        # 96 / 8 = 12
        self.grid_size = self.input_size // 8 

        # Load Data
        self.data = self._load_annotations()
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_annotations(self):
        """Parse the JSON and link images to boxes"""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON not found: {self.json_path}")

        with open(self.json_path, 'r') as f:
            content = json.load(f)

        # Map JSON structure to a list
        data_list = []
        files_array = content.get('files', []) if isinstance(content, dict) else content
        
        for entry in files_array:
            img_name = entry.get('path', '')
            boxes = entry.get('boundingBoxes', [])
            
            data_list.append({
                'path': os.path.join(self.img_dir, img_name),
                'boxes': boxes
            })
            
        print(f"Found {len(data_list)} images in {self.split} set.")
        return data_list

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _augment(self, img, boxes):
        """Randomly augment image and adjust boxes accordingly"""
        h, w, _ = img.shape
        
        # 1. Random Horizontal Flip (50% chance)
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1) # 1 = Horizontal Flip
            
            # Adjust boxes for flip: NewX = Width - (OldX + Width)
            for box in boxes:
                box['x'] = w - (box['x'] + box['width'])
        
        # 2. Random Brightness/Contrast (Color Jitter)
        if np.random.rand() > 0.5:
            # Random brightness integer (-30 to 30)
            brightness = np.random.randint(-30, 30)
            # Random contrast float (0.8 to 1.2)
            contrast = np.random.uniform(0.8, 1.2)
            
            # Apply: new = alpha*old + beta
            img = cv2.addWeighted(img, contrast, img, 0, brightness)

        return img, boxes

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_x = []
        batch_y = []

        for i in batch_indices:
            # CRITICAL: Use deepcopy to ensure we don't modify the original dataset 
            # in memory when flipping coordinates during augmentation.
            item_data = copy.deepcopy(self.data[i])
            
            img_path = item_data['path']
            raw_boxes = item_data['boxes']
            
            # 1. Load Image
            img = cv2.imread(img_path)
            if img is None:
                continue # Skip broken images
            
            # 2. Augment (Only if augment=True, usually for training)
            if self.augment:
                img, raw_boxes = self._augment(img, raw_boxes)

            orig_h, orig_w, _ = img.shape
            
            # 3. Resize Image to Network Input (96x96)
            img_resized = cv2.resize(img, (self.input_size, self.input_size))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0 # Normalize 0-1
            
            # 4. Create FOMO Target Grid (12x12xClasses)
            # Initialize with Background (Class 0) = 1, Others = 0
            target_mask = np.zeros((self.grid_size, self.grid_size, self.num_classes), dtype=np.float32)
            target_mask[:, :, 0] = 1.0 # Default to background
            
            # 5. Scale Boxes and Map to Grid
            scale_x = self.input_size / orig_w
            scale_y = self.input_size / orig_h
            
            for box in raw_boxes:
                # Get Original Center
                x = box['x']
                y = box['y']
                w = box['width']
                h = box['height']
                
                cx = x + (w / 2)
                cy = y + (h / 2)
                
                # Scale Center to New Size (e.g. 96x96)
                cx_scaled = cx * scale_x
                cy_scaled = cy * scale_y
                
                # Map to Grid Cell (e.g. 12x12)
                # If stride is 8: pixel 48 -> grid 6
                grid_x = int(cx_scaled / 8)
                grid_y = int(cy_scaled / 8)
                
                # Boundary Check
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    # Set Class 1 (FruitFly) to 1.0
                    target_mask[grid_y, grid_x, 0] = 0.0 # Remove background label
                    target_mask[grid_y, grid_x, 1] = 1.0 # Add fly label
            
            batch_x.append(img_norm)
            batch_y.append(target_mask)

        return np.array(batch_x), np.array(batch_y)

    def get_dataset(self):
        # Compatibility wrapper
        return self