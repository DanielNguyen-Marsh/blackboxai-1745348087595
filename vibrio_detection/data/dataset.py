"""
Dataset handling for Vibrio Detection
"""
import os
import shutil
from PIL import Image
import numpy as np

class VibrioDataset:
    """Class for handling Vibrio dataset"""
    
    def __init__(self, dataset_dir="dataset"):
        """
        Initialize the dataset
        
        Args:
            dataset_dir (str): Path to the dataset directory
        """
        self.dataset_dir = dataset_dir
        self.images_train_dir = os.path.join(dataset_dir, "images", "train")
        self.images_val_dir = os.path.join(dataset_dir, "images", "val")
        self.labels_train_dir = os.path.join(dataset_dir, "labels", "train")
        self.labels_val_dir = os.path.join(dataset_dir, "labels", "val")
    
    def create_directory_structure(self):
        """Create the dataset directory structure"""
        os.makedirs(self.images_train_dir, exist_ok=True)
        os.makedirs(self.images_val_dir, exist_ok=True)
        os.makedirs(self.labels_train_dir, exist_ok=True)
        os.makedirs(self.labels_val_dir, exist_ok=True)
        
        print(f"Created dataset directory structure at {self.dataset_dir}")
    
    def create_sample_data(self, num_samples=10):
        """
        Create sample data for testing
        
        Args:
            num_samples (int): Number of sample images to create
        """
        self.create_directory_structure()
        
        # Create sample images
        for i in range(num_samples):
            # Create a white image with a black rectangle
            img = Image.new('RGB', (640, 640), color=(255, 255, 255))
            img_array = np.array(img)
            
            # Add a black rectangle (simulating an object)
            x1, y1 = 200, 200
            x2, y2 = 400, 400
            img_array[y1:y2, x1:x2] = (0, 0, 0)
            
            # Convert back to image
            img = Image.fromarray(img_array)
            
            # Save to train and val folders
            train_path = os.path.join(self.images_train_dir, f"sample_{i}.jpg")
            val_path = os.path.join(self.images_val_dir, f"sample_{i}.jpg")
            img.save(train_path)
            img.save(val_path)
            
            # Create YOLO format labels (class_id, x_center, y_center, width, height)
            # All values normalized to [0, 1]
            x_center = (x1 + x2) / 2 / 640
            y_center = (y1 + y2) / 2 / 640
            width = (x2 - x1) / 640
            height = (y2 - y1) / 640
            
            # Write labels
            with open(os.path.join(self.labels_train_dir, f"sample_{i}.txt"), 'w') as f:
                f.write(f'0 {x_center} {y_center} {width} {height}\n')
            with open(os.path.join(self.labels_val_dir, f"sample_{i}.txt"), 'w') as f:
                f.write(f'0 {x_center} {y_center} {width} {height}\n')
        
        print(f"Created {num_samples} sample images and labels")
    
    def count_images(self):
        """Count the number of images in the dataset"""
        train_count = len([f for f in os.listdir(self.images_train_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))])
        val_count = len([f for f in os.listdir(self.images_val_dir) 
                        if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        return {
            "train": train_count,
            "val": val_count,
            "total": train_count + val_count
        }
    
    def clean(self):
        """Remove all data from the dataset"""
        if os.path.exists(self.dataset_dir):
            shutil.rmtree(self.dataset_dir)
            print(f"Removed dataset directory: {self.dataset_dir}")
