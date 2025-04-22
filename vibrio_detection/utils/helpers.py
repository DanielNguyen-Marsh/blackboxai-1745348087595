"""
Helper functions for Vibrio Detection
"""
import os
import sys
import shutil

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        "ultralytics",
        "cv2",  # opencv-python package name is 'cv2' when importing
        "PIL",  # pillow package name is 'PIL' when importing
        "numpy",
        "torch",
        "torchvision"
    ]

    missing_packages = []
    package_mapping = {
        "cv2": "opencv-python",
        "PIL": "pillow"
    }

    for package in required_packages:
        try:
            if package == "PIL":
                __import__("PIL.Image")
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package_mapping.get(package, package))

    if missing_packages:
        print("Error: The following required packages are not installed:")
        for package in missing_packages:
            print(f"  - {package}")
        print("Please install them using: pip install -r requirements.txt")
        return False

    return True

def find_sample_image(dataset_dir):
    """
    Find a sample image for inference

    Args:
        dataset_dir (str): Path to the dataset directory
    """
    # Check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found at {dataset_dir}")
        return None

    # Check for images in the validation directory
    val_dir = os.path.join(dataset_dir, 'images', 'val')
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found at {val_dir}")
        return None

    # Get the first image in the validation directory
    for file in os.listdir(val_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(val_dir, file)

    print(f"Error: No images found in {val_dir}")
    return None

def copy_model_to_models_dir(source_path, models_dir):
    """
    Copy a trained model to the models directory

    Args:
        source_path (str): Path to the source model file
        models_dir (str): Path to the models directory
    """
    if not os.path.exists(source_path):
        print(f"Error: Model file not found at {source_path}")
        return None

    os.makedirs(models_dir, exist_ok=True)

    # Get the filename from the source path
    filename = os.path.basename(source_path)

    # Create the destination path
    dest_path = os.path.join(models_dir, filename)

    try:
        shutil.copy2(source_path, dest_path)
        print(f"Copied model from {source_path} to {dest_path}")
        return dest_path
    except Exception as e:
        print(f"Error copying model: {str(e)}")
        return None
