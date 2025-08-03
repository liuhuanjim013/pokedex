# Pokemon Classifier

A real-time Pokemon classification system using YOLOv3, optimized for deployment on Sipeed Maix Bit RISC-V IoT device.

## Project Overview

This project reproduces and extends the baseline Pokemon classifier work described in the [blog post](https://www.cnblogs.com/xianmasamasa/p/18995912), using Google Colab for training instead of the original Mx_yolo binary approach.

### Goals
- **Baseline**: Reproduce 386-class (generations 1-3) Pokemon classifier
- **Extension**: Extend to 1025-class (all generations) Pokemon classifier
- **Advanced**: Implement VLM and newer YOLO variants for improved performance
- **Deployment**: Optimize for IoT deployment on Sipeed Maix Bit

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd pokedex

# Install dependencies
pip install -r requirements.txt

# Setup experiment
python scripts/setup_experiment.py --dataset_path /path/to/your/900mb/dataset
```

### 2. Process Dataset

```bash
# Process your 900MB gen1-3 dataset
python src/data/preprocessing.py \
    --dataset_path /path/to/your/dataset \
    --config configs/data_config.yaml \
    --create_yolo_dataset
```

### 3. Upload to Hugging Face

```bash
# Upload processed dataset to Hugging Face
python scripts/upload_dataset.py \
    --processed_dir data/processed/yolo_dataset \
    --dataset_name your-username/pokemon-gen1-3 \
    --yolo_format
```

### 4. Train Model

```bash
# Train YOLOv3 model
python src/training/yolo_trainer.py \
    --data_yaml data/processed/data.yaml \
    --config configs/training_config.yaml \
    --evaluate
```

## Dataset Preparation

### Expected Dataset Structure

Your 900MB gen1-3 dataset should have this structure:

```
dataset/
├── pokemon_name_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── pokemon_name_2/
│   ├── image1.jpg
│   └── ...
└── ...
```

### Processing Steps

1. **Download Dataset**: Place your 900MB dataset in `data/raw/gen1_3_pokemon/`
2. **Preprocess**: Run preprocessing script to resize and normalize images
3. **Create YOLO Dataset**: Generate train/val/test splits with YOLO format
4. **Upload to Hugging Face**: Make dataset accessible for Colab training

### Hugging Face Upload

The processed dataset will be uploaded to Hugging Face with:
- **Images**: Resized to 416x416 (YOLO standard)
- **Labels**: Classification labels (not detection)
- **Splits**: Train/val/test splits (70/15/15)
- **Metadata**: Pokemon names and class mappings

## Training in Google Colab

### Environment Setup

```python
# Install conda and uv
!wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!bash Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local
!conda install -c conda-forge uv -y

# Create environment
!conda create -n pokemon-classifier python=3.9 -y
!conda activate pokemon-classifier

# Install dependencies
!uv add torch torchvision torchaudio
!uv add ultralytics opencv-python
!uv add wandb pillow matplotlib seaborn
!uv add accelerate bitsandbytes
```

### Load Dataset from Hugging Face

```python
from datasets import load_dataset

# Load your uploaded dataset
dataset = load_dataset("your-username/pokemon-gen1-3")
print(f"Dataset: {dataset}")
```

### Train YOLOv3

```python
from ultralytics import YOLO
import wandb

# Initialize W&B
wandb.init(project="pokemon-classifier", name="yolov3-baseline")

# Load model
model = YOLO("yolov3.pt")

# Train
results = model.train(
    data="data.yaml",  # Path to your data.yaml
    epochs=100,
    batch=16,
    imgsz=416,
    device="0",  # GPU
    project="pokemon-classifier",
    name="yolov3-baseline"
)
```

## Project Structure

```
pokedex/
├── data/                          # Data storage
│   ├── raw/                      # Raw datasets
│   ├── processed/                # Processed data
│   └── splits/                  # Train/val/test splits
├── models/                       # Model storage
│   ├── checkpoints/             # Training checkpoints
│   ├── final/                   # Final models
│   └── compressed/              # IoT optimized models
├── src/                         # Source code
│   ├── data/                    # Data processing
│   ├── training/                # Training pipelines
│   └── evaluation/              # Evaluation code
├── configs/                     # Configuration files
├── scripts/                     # Utility scripts
├── notebooks/                   # Jupyter notebooks
└── docs/                        # Documentation
```

## Configuration

### Data Configuration (`configs/data_config.yaml`)

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  
processing:
  image_size: 416  # YOLO standard
  batch_size: 16
  
generations:
  baseline: [1, 2, 3]  # 386 Pokemon
  full: [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 1025 Pokemon
```

### Training Configuration (`configs/training_config.yaml`)

```yaml
model:
  name: "yolov3"
  classes: 386  # Start with 1-3 generations
  img_size: 416

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001

wandb:
  project: "pokemon-classifier"
  name: "yolov3-baseline"
```

## Workflow

### Phase 1: Baseline Reproduction
1. **Setup**: Environment and dependencies
2. **Data**: Process 900MB gen1-3 dataset
3. **Training**: Train YOLOv3 on 386 classes
4. **Evaluation**: Test baseline performance

### Phase 2: Extension
1. **Data Collection**: Gather data for all 1025 Pokemon
2. **Retraining**: Train model on full dataset
3. **Comparison**: Compare 386 vs 1025 class performance

### Phase 3: Advanced Approaches
1. **VLM Testing**: Implement CLIP, SMoLVM
2. **YOLO Variants**: Test YOLOv8, YOLOv9, YOLOv10
3. **Multi-frame**: Implement temporal aggregation
4. **Selection**: Choose best approach for deployment

### Phase 4: IoT Deployment
1. **Optimization**: Model compression and quantization
2. **Hardware**: Sipeed Maix Bit deployment
3. **Testing**: Real-world performance validation

## Monitoring

### Weights & Biases

Training progress is tracked in W&B:
- **Project**: `pokemon-classifier`
- **Metrics**: Loss, accuracy, mAP
- **Models**: Checkpoints and final models
- **Experiments**: Hyperparameter sweeps

### Hugging Face

- **Datasets**: Processed Pokemon datasets
- **Models**: Trained model checkpoints
- **Versioning**: Dataset and model versions

## Expected Performance

Based on the original project:
- **Baseline (386 classes)**: ~60-70% accuracy
- **Full dataset (1025 classes)**: ~50-60% accuracy
- **Real-world**: 10-20% lower than controlled conditions
- **Limitations**: Lighting, angle, background sensitivity

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Original project by [弦masamasa](https://www.cnblogs.com/xianmasamasa/p/18995912)
- YOLO by Ultralytics
- Hugging Face for dataset hosting
- Weights & Biases for experiment tracking 