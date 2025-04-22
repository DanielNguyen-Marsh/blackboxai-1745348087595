"""
Script for evaluating the Vibrio detection model
"""
import os
import sys
import argparse

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vibrio_detection.configs.config import Config
from vibrio_detection.models.model import VibrioModel
from vibrio_detection.utils.helpers import check_dependencies

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate Vibrio detection model")
    
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
    
    # Create model
    model = VibrioModel(config)
    
    # Load trained model
    if not model.load_trained():
        print("Failed to load trained model.")
        return
    
    print("Evaluating model...")
    
    # Evaluate model
    metrics = model.evaluate()
    
    if metrics is not None:
        print("Evaluation completed successfully!")
        
        # Print metrics
        print("\nEvaluation Metrics:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
    else:
        print("Evaluation failed.")

if __name__ == "__main__":
    main()
