"""
Script for training the Vibrio detection model
"""
import os
import sys
import argparse

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vibrio_detection.configs.config import Config
from vibrio_detection.data.dataset import VibrioDataset
from vibrio_detection.models.model import VibrioModel
from vibrio_detection.utils.helpers import check_dependencies

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Vibrio detection model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--image-size", type=int, default=640, help="Image size")
    parser.add_argument("--create-samples", action="store_true", help="Create sample data")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of sample images to create")
    
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
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.image_size = args.image_size
    
    # Create dataset
    dataset = VibrioDataset(config.dataset_dir)
    
    # Create sample data if requested
    if args.create_samples:
        dataset.create_sample_data(args.num_samples)
    
    # Create dataset YAML file
    config.create_dataset_yaml()
    
    # Create and train model
    model = VibrioModel(config)
    model.load_pretrained()
    results = model.train()
    
    if results is not None:
        print("Training completed successfully!")
    else:
        print("Training failed.")

if __name__ == "__main__":
    main()
