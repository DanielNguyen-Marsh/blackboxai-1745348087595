"""
Script to migrate from old structure to new structure
"""
import os
import shutil
import sys

def migrate():
    """Migrate from old structure to new structure"""
    print("Migrating from old structure to new structure...")
    
    # Copy model file if it exists
    old_model_path = os.path.join("runs", "detect", "vibrio_yolov8_model5", "weights", "best.pt")
    new_model_path = os.path.join("vibrio_detection", "models", "best.pt")
    
    if os.path.exists(old_model_path):
        os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
        shutil.copy2(old_model_path, new_model_path)
        print(f"Copied model from {old_model_path} to {new_model_path}")
    
    # Copy dataset if it exists
    if os.path.exists("dataset"):
        new_dataset_path = os.path.join("vibrio_detection", "data", "dataset")
        if not os.path.exists(new_dataset_path):
            shutil.copytree("dataset", new_dataset_path)
            print(f"Copied dataset to {new_dataset_path}")
    
    # Copy dataset.yaml if it exists
    if os.path.exists("dataset.yaml"):
        new_config_path = os.path.join("vibrio_detection", "configs", "dataset.yaml")
        shutil.copy2("dataset.yaml", new_config_path)
        print(f"Copied dataset.yaml to {new_config_path}")
    
    print("Migration completed!")
    print("\nYou can now use the new structure with the following commands:")
    print("- Create sample data: python -m vibrio_detection.main --action create-samples")
    print("- Train model: python -m vibrio_detection.main --action train")
    print("- Run inference: python -m vibrio_detection.main --action inference")
    print("- Evaluate model: python -m vibrio_detection.main --action evaluate")

if __name__ == "__main__":
    migrate()
