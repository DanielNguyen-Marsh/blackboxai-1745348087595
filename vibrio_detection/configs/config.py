"""
Configuration for Vibrio Detection
"""
import os
import yaml

class Config:
    """Configuration class for Vibrio Detection"""

    def __init__(self):
        """Initialize the configuration"""
        self.current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.dataset_dir = os.path.join(self.current_dir, "dataset")
        self.pretrained_model = os.path.join(self.current_dir, "yolov8n.pt")
        self.output_model_name = "vibrio_yolov8_model"
        self.dataset_yaml = os.path.join(self.current_dir, "dataset.yaml")
        # Try multiple possible model paths
        self.model_paths = [
            os.path.join(self.current_dir, "runs", "detect", self.output_model_name, "weights", "best.pt"),
            os.path.join(self.current_dir, "runs", "detect", self.output_model_name + "5", "weights", "best.pt"),
            os.path.join(self.current_dir, "runs", "train", self.output_model_name, "weights", "best.pt"),
            os.path.join(self.current_dir, "vibrio_detection", "models", "best.pt")
        ]

        # Find the first existing model path
        self.model_path = next((path for path in self.model_paths if os.path.exists(path)), self.model_paths[0])

        # Training parameters
        self.epochs = 100
        self.batch_size = 16
        self.image_size = 640
        self.confidence_threshold = 0.25

        # Class names
        self.class_names = ['v_para', 'v_algi']
        self.num_classes = len(self.class_names)

    def create_dataset_yaml(self):
        """Create a dataset YAML file with the correct paths"""
        yaml_content = f"""path: {self.dataset_dir}
train: images/train
val: images/val

nc: {self.num_classes}
names: {self.class_names}
"""

        try:
            with open(self.dataset_yaml, 'w') as f:
                f.write(yaml_content)
            print(f"Created dataset YAML file at {self.dataset_yaml}")
            return self.dataset_yaml
        except Exception as e:
            print(f"Error creating dataset YAML file: {str(e)}")
            return None

    def load_from_yaml(self, yaml_path):
        """
        Load configuration from a YAML file

        Args:
            yaml_path (str): Path to the YAML file
        """
        try:
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # Update configuration
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)

            print(f"Loaded configuration from {yaml_path}")
            return True
        except Exception as e:
            print(f"Error loading configuration from {yaml_path}: {str(e)}")
            return False

    def save_to_yaml(self, yaml_path):
        """
        Save configuration to a YAML file

        Args:
            yaml_path (str): Path to the YAML file
        """
        config_data = {
            "dataset_dir": self.dataset_dir,
            "pretrained_model": self.pretrained_model,
            "output_model_name": self.output_model_name,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "image_size": self.image_size,
            "confidence_threshold": self.confidence_threshold,
            "class_names": self.class_names,
            "num_classes": self.num_classes
        }

        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            print(f"Saved configuration to {yaml_path}")
            return True
        except Exception as e:
            print(f"Error saving configuration to {yaml_path}: {str(e)}")
            return False
