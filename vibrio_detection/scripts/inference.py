"""
Script for running inference with the Vibrio detection model
"""
import os
import sys
import argparse

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vibrio_detection.configs.config import Config
from vibrio_detection.models.model import VibrioModel
from vibrio_detection.utils.helpers import check_dependencies, find_sample_image

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run inference with Vibrio detection model")
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    return parser.parse_args()

def main():
    """Main function"""
    # Check dependencies
    if not check_dependencies():
        return
    
    # Parse arguments
    args = parse_args()
    
    # Create configuration
    config = Config()
    config.confidence_threshold = args.conf
    
    # Create model
    model = VibrioModel(config)
    
    # Load trained model
    if not model.load_trained():
        print("Failed to load trained model.")
        return
    
    # Get image path
    image_path = args.image
    if image_path is None:
        print("No image specified, looking for a sample image...")
        image_path = find_sample_image(config.dataset_dir)
    
    if image_path is None:
        print("No image found for inference.")
        return
    
    print(f"Running inference on image: {image_path}")
    
    # Run inference
    results = model.predict(image_path)
    
    if results is not None:
        print("Inference completed successfully!")
        print(f"Results saved to: {os.path.dirname(results[0].save_dir)}")
    else:
        print("Inference failed.")

if __name__ == "__main__":
    main()
