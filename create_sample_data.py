import os
from PIL import Image
import numpy as np

def create_sample_data():
    """Create sample data for testing"""
    # Create directories
    os.makedirs('dataset/images/train', exist_ok=True)
    os.makedirs('dataset/images/val', exist_ok=True)
    os.makedirs('dataset/labels/train', exist_ok=True)
    os.makedirs('dataset/labels/val', exist_ok=True)
    
    print("Created directories")
    
    # Create sample images
    for i in range(10):
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
        train_path = f'dataset/images/train/sample_{i}.jpg'
        val_path = f'dataset/images/val/sample_{i}.jpg'
        img.save(train_path)
        img.save(val_path)
        
        # Create YOLO format labels (class_id, x_center, y_center, width, height)
        # All values normalized to [0, 1]
        x_center = (x1 + x2) / 2 / 640
        y_center = (y1 + y2) / 2 / 640
        width = (x2 - x1) / 640
        height = (y2 - y1) / 640
        
        # Write labels
        with open(f'dataset/labels/train/sample_{i}.txt', 'w') as f:
            f.write(f'0 {x_center} {y_center} {width} {height}\n')
        with open(f'dataset/labels/val/sample_{i}.txt', 'w') as f:
            f.write(f'0 {x_center} {y_center} {width} {height}\n')
    
    print("Created sample images and labels")
    
    # Create dataset YAML file
    yaml_content = """path: ./dataset
train: images/train
val: images/val

nc: 2
names: ['v_para', 'v_algi']
"""
    
    with open('dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("Created dataset.yaml file")

if __name__ == '__main__':
    create_sample_data()
    print("Sample data creation complete!")
