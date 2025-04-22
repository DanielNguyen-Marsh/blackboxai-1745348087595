# Vibrio Detection System

A system for detecting Vibrio bacteria in images using YOLOv8.

## Installation

1. **Create a virtual environment**:
   ```
   python -m venv .venv
   ```

2. **Activate the virtual environment**:
   - Windows:
     ```
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Creating Sample Data

To create sample data for testing:

```
python -m vibrio_detection.main --action create-samples --num-samples 10
```

This will create a dataset with the following structure:
```
dataset/
├── images/
│   ├── train/
│   │   └── ... (training images)
│   └── val/
│       └── ... (validation images)
└── labels/
    ├── train/
    │   └── ... (training labels in YOLO format)
    └── val/
        └── ... (validation labels in YOLO format)
```

### Training the Model

To train the model:

```
python -m vibrio_detection.main --action train --epochs 100 --batch-size 16 --image-size 640
```

This will train the YOLOv8 model on your dataset and save the best model.

### Running Inference

To run inference on an image:

```
python -m vibrio_detection.main --action inference --image path/to/image.jpg --conf 0.25
```

If no image is specified, the system will look for a sample image in the dataset.

### Evaluating the Model

To evaluate the model:

```
python -m vibrio_detection.main --action evaluate
```

## Troubleshooting

- If you encounter errors about missing packages, make sure you've installed all dependencies:
  ```
  pip install -r requirements.txt
  ```

- If you see errors about missing model files, make sure you've trained the model first:
  ```
  python -m vibrio_detection.main --action train
  ```

- If you see errors about missing dataset files, make sure your dataset is properly organized or create sample data:
  ```
  python -m vibrio_detection.main --action create-samples
  ```

## Project Structure

```
vibrio_detection/
├── configs/            # Configuration files
├── data/               # Data handling
├── models/             # Model definitions
├── scripts/            # Scripts for training, inference, etc.
├── utils/              # Utility functions
└── main.py             # Main script
```

## Legacy Scripts

The following legacy scripts are still available but are deprecated:

- `train_yolov8.py`: Old script to train the YOLOv8 model
- `inference_and_evaluation.py`: Old script to run inference and evaluate the model
- `vibrio_dataset.yaml`: Old configuration file for the dataset

## Authors

Vibrio Detection Team
