"""
Model for Vibrio Detection
"""
import os
import sys

# Try to import ultralytics, provide helpful error if not installed
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: The 'ultralytics' package is not installed.")
    print("Please install it using: pip install -r requirements.txt")
    sys.exit(1)

class VibrioModel:
    """Class for handling Vibrio detection model"""
    
    def __init__(self, config):
        """
        Initialize the model
        
        Args:
            config (Config): Configuration object
        """
        self.config = config
        self.model = None
    
    def load_pretrained(self):
        """Load a pretrained YOLOv8 model"""
        try:
            self.model = YOLO(self.config.pretrained_model)
            print(f"Loaded pretrained model from {self.config.pretrained_model}")
            return True
        except Exception as e:
            print(f"Error loading pretrained model: {str(e)}")
            return False
    
    def load_trained(self):
        """Load a trained model"""
        if not os.path.exists(self.config.model_path):
            print(f"Error: Model file not found at {self.config.model_path}")
            print("You need to train the model first using train_model()")
            return False
        
        try:
            self.model = YOLO(self.config.model_path)
            print(f"Loaded trained model from {self.config.model_path}")
            return True
        except Exception as e:
            print(f"Error loading trained model: {str(e)}")
            return False
    
    def train(self):
        """Train the model"""
        if not os.path.exists(self.config.dataset_yaml):
            print(f"Error: Dataset configuration file not found at {self.config.dataset_yaml}")
            return None
        
        if self.model is None:
            if not self.load_pretrained():
                return None
        
        try:
            print(f"\n=== Starting training ===\n")
            print(f"This may take a while depending on your hardware...\n")
            
            # Train the model
            results = self.model.train(
                data=self.config.dataset_yaml,
                epochs=self.config.epochs,
                imgsz=self.config.image_size,
                batch=self.config.batch_size,
                name=self.config.output_model_name
            )
            
            print(f"\n=== Training completed ===\n")
            print(f"Model saved to: runs/detect/{self.config.output_model_name}/weights/best.pt")
            
            return results
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return None
    
    def predict(self, image_path):
        """
        Run inference on a single image
        
        Args:
            image_path (str): Path to the image
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return None
        
        if self.model is None:
            if not self.load_trained():
                return None
        
        try:
            results = self.model.predict(
                source=image_path, 
                conf=self.config.confidence_threshold, 
                save=True
            )
            return results
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return None
    
    def evaluate(self):
        """Evaluate the model on the validation dataset"""
        if not os.path.exists(self.config.dataset_yaml):
            print(f"Error: Dataset configuration file not found at {self.config.dataset_yaml}")
            return None
        
        if self.model is None:
            if not self.load_trained():
                return None
        
        try:
            metrics = self.model.val(data=self.config.dataset_yaml)
            return metrics
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None
