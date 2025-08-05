# Pokemon YOLO Dataset Usage Example
# For Google Colab training

# Install required packages
!pip install ultralytics
!pip install datasets
!pip install huggingface_hub

# Load dataset from Hugging Face
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("liuhuanjim013/pokemon-yolo-1025")

print(f"Dataset loaded: {dataset}")
print(f"Train split: {len(dataset['train'])} images")
print(f"Validation split: {len(dataset['validation'])} images")
print(f"Test split: {len(dataset['test'])} images")

# Example: Get first training image
train_dataset = dataset["train"]
example = train_dataset[0]
print(f"First image: {example}")

# For YOLO training with ultralytics
from ultralytics import YOLO

# Load YOLOv3 model
model = YOLO('yolov3.pt')

# Train on the dataset
results = model.train(
    data='liuhuanjim013/pokemon-yolo-1025',  # Dataset from Hugging Face
    epochs=100,
    imgsz=416,
    batch=16,
    lr0=0.001,
    # Original blog parameters (minimal augmentation)
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5,
    shear=0.0, perspective=0.0, flipud=0.0,
    fliplr=0.5, mosaic=0.0, mixup=0.0
)

print("Training completed!")
