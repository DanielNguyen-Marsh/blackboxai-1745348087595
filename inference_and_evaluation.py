import os
import sys

# Try to import ultralytics, provide helpful error if not installed
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: The 'ultralytics' package is not installed.")
    print("Please install it using: pip install -r requirements.txt")
    sys.exit(1)

# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CURRENT_DIR, 'dataset')

# Define model path
MODEL_PATH = os.path.join(CURRENT_DIR, 'runs/detect/vibrio_yolov8_model5/weights/best.pt')
DATASET_CONFIG = os.path.join(CURRENT_DIR, 'dataset.yaml')

def check_model_exists():
    """Check if the model file exists and provide helpful message if not"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("You need to train the model first using train_yolov8.py")
        print("Run: python train_yolov8.py")
        return False
    return True

def check_image_exists(image_path):
    """Check if the image file exists"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return False
    return True

def run_inference(image_path):
    """Run inference on a single image"""
    if not check_model_exists() or not check_image_exists(image_path):
        return None

    try:
        model = YOLO(MODEL_PATH)
        results = model.predict(source=image_path, conf=0.25, save=True)
        return results
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None

def evaluate_model():
    """Evaluate the model on the validation dataset"""
    if not check_model_exists():
        return None

    if not os.path.exists(DATASET_CONFIG):
        print(f"Error: Dataset configuration file not found at {DATASET_CONFIG}")
        return None

    try:
        model = YOLO(MODEL_PATH)
        metrics = model.val(data=DATASET_CONFIG)
        return metrics
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None

def find_sample_image():
    """Find a sample image for inference"""
    # Check if the dataset directory exists
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        return None

    # Check for images in the validation directory
    val_dir = os.path.join(DATASET_DIR, 'images', 'val')
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found at {val_dir}")
        return None

    # Get the first image in the validation directory
    for file in os.listdir(val_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(val_dir, file)

    print(f"Error: No images found in {val_dir}")
    return None

if __name__ == '__main__':
    # Example usage:
    print("\n=== Running Inference ===\n")
    sample_image = find_sample_image()
    if sample_image:
        print(f"Using sample image: {sample_image}")
        results = run_inference(sample_image)
        if results:
            print(f"Inference results: {results}")
    else:
        print("No sample image found for inference.")

    print("\n=== Evaluating Model ===\n")
    metrics = evaluate_model()
    if metrics:
        print(f"Evaluation metrics: {metrics}")
