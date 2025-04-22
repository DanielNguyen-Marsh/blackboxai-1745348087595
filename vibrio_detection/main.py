"""
Main script for Vibrio Detection
"""
import os
import sys
import argparse

# Try to import required modules
try:
    from vibrio_detection.configs.config import Config
    from vibrio_detection.data.dataset import VibrioDataset
    from vibrio_detection.models.model import VibrioModel
    from vibrio_detection.utils.helpers import check_dependencies, find_sample_image
except ImportError:
    # If running as a script, adjust the path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vibrio_detection.configs.config import Config
    from vibrio_detection.data.dataset import VibrioDataset
    from vibrio_detection.models.model import VibrioModel
    from vibrio_detection.utils.helpers import check_dependencies, find_sample_image

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Vibrio Detection System")
    parser.add_argument("--action", type=str, required=True,
                        choices=["train", "inference", "evaluate", "create-samples"],
                        help="Action to perform")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--image-size", type=int, default=640, help="Image size for training")
    parser.add_argument("--image", type=str, help="Path to the image file for inference")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for inference")
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
    config.confidence_threshold = args.conf

    # Perform the requested action
    if args.action == "create-samples":
        # Create dataset
        dataset = VibrioDataset(config.dataset_dir)
        dataset.create_sample_data(args.num_samples)

    elif args.action == "train":
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

    elif args.action == "inference":
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

    elif args.action == "evaluate":
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
