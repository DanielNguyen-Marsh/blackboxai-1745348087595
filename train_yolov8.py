import os
import sys

# Try to import ultralytics, provide helpful error if not installed
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: The 'ultralytics' package is not installed.")
    print("Please install it using: pip install -r requirements.txt")
    sys.exit(1)

# Define constants
PRETRAINED_MODEL = 'yolov8n.pt'
OUTPUT_MODEL_NAME = 'vibrio_yolov8_model'

# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CURRENT_DIR, 'dataset')

def create_sample_data():
    """Create sample data for testing if no real data is available"""
    # Create sample images directory
    os.makedirs(os.path.join(DATASET_DIR, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'labels', 'val'), exist_ok=True)

    # Create a sample image (1x1 pixel) for testing
    try:
        from PIL import Image
        # Create a small blank image
        img = Image.new('RGB', (640, 640), color=(255, 255, 255))
        img.save(os.path.join(DATASET_DIR, 'images', 'train', 'sample.jpg'))
        img.save(os.path.join(DATASET_DIR, 'images', 'val', 'sample.jpg'))

        # Create sample labels
        with open(os.path.join(DATASET_DIR, 'labels', 'train', 'sample.txt'), 'w') as f:
            f.write('0 0.5 0.5 0.1 0.1\n')  # class_id x_center y_center width height
        with open(os.path.join(DATASET_DIR, 'labels', 'val', 'sample.txt'), 'w') as f:
            f.write('0 0.5 0.5 0.1 0.1\n')  # class_id x_center y_center width height

        print("Created sample data for testing")
        return True
    except Exception as e:
        print(f"Error creating sample data: {str(e)}")
        return False

def create_dataset_yaml():
    """Create a dataset YAML file with the correct paths"""
    yaml_content = f"""path: {DATASET_DIR}
train: images/train
val: images/val

nc: 2
names: ['v_para', 'v_algi']
"""

    yaml_path = os.path.join(CURRENT_DIR, 'dataset.yaml')
    try:
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"Created dataset YAML file at {yaml_path}")
        return yaml_path
    except Exception as e:
        print(f"Error creating dataset YAML file: {str(e)}")
        return None

def train_model():
    """Train the YOLOv8 model on the Vibrio dataset"""
    # Always create sample data for testing
    print("Creating sample data for testing...")
    if not create_sample_data():
        return

    # Create dataset YAML file
    dataset_yaml = create_dataset_yaml()
    if not dataset_yaml:
        return

    try:
        # Load a pretrained YOLOv8 model (nano version for quick training)
        model = YOLO(PRETRAINED_MODEL)

        print(f"\n=== Starting training ===\n")
        print(f"This may take a while depending on your hardware...\n")

        # Train the model on the Vibrio dataset
        results = model.train(
            data=dataset_yaml,
            epochs=10,  # Reduced for testing
            imgsz=640,
            batch=16,
            name=OUTPUT_MODEL_NAME
        )

        print(f"\n=== Training completed ===\n")
        print(f"Model saved to: runs/train/{OUTPUT_MODEL_NAME}/weights/best.pt")
        print(f"You can now use inference_and_evaluation.py to test the model.")

        return results
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

if __name__ == '__main__':
    print("Starting training process...")
    print(f"Current directory: {CURRENT_DIR}")
    print(f"Dataset directory: {DATASET_DIR}")
    train_model()
