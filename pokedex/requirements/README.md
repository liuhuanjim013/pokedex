# Pokemon Classifier - Dependencies

This directory contains experiment-specific dependency requirements for the Pokemon Classifier project.

## Structure

```
requirements/
├── README.md                    # This file
├── yolo_requirements.txt        # YOLO experiment dependencies
├── vlm_requirements.txt         # VLM experiment dependencies
└── hybrid_requirements.txt      # Hybrid approach dependencies
```

## Usage

### Quick Setup (Conda + uv)

1. **Create conda environment and install base requirements**:
   ```bash
   python scripts/common/setup_environment.py
   ```

2. **Set up for YOLO experiments**:
   ```bash
   python scripts/common/setup_environment.py --experiment yolo
   ```

3. **Set up for VLM experiments**:
   ```bash
   python scripts/common/setup_environment.py --experiment vlm
   ```

4. **Set up for Hybrid experiments**:
   ```bash
   python scripts/common/setup_environment.py --experiment hybrid
   ```

5. **Set up for Google Colab**:
   ```bash
   python scripts/common/setup_environment.py --experiment yolo --colab
   ```

### Manual Setup

If you prefer manual setup:

1. **Create conda environment**:
   ```bash
   conda create -n pokemon-classifier python=3.9 -y
   conda activate pokemon-classifier
   ```

2. **Install uv**:
   ```bash
   conda install -c conda-forge uv -y
   ```

3. **Install dependencies**:
   ```bash
   # Base requirements
   uv pip install -r requirements.txt
   
   # YOLO requirements
   uv pip install -r requirements/yolo_requirements.txt
   
   # VLM requirements
   uv pip install -r requirements/vlm_requirements.txt
   ```

### Automated Setup

Use the setup script for easy environment configuration:

```bash
# Set up base environment
python scripts/common/setup_environment.py

# Set up for YOLO experiments
python scripts/common/setup_environment.py --experiment yolo

# Set up for VLM experiments  
python scripts/common/setup_environment.py --experiment vlm

# Set up for Google Colab
python scripts/common/setup_environment.py --experiment yolo --colab

# Verify installation
python scripts/common/setup_environment.py --experiment yolo --verify
```

## Dependencies by Experiment Type

### Base Requirements (`requirements.txt`)
- Core Python packages (numpy, pandas, Pillow)
- Data visualization (matplotlib, seaborn, opencv)
- Machine learning (scikit-learn, tqdm)
- Experiment tracking (wandb, tensorboard)
- Hugging Face ecosystem (huggingface-hub, datasets, transformers)
- Development tools (black, flake8, pytest)

### YOLO Requirements (`yolo_requirements.txt`)
- Extends base requirements
- YOLO frameworks (ultralytics, torch, torchvision)
- YOLO training (albumentations, pycocotools)
- Deployment (onnx, onnxruntime)

### VLM Requirements (`vlm_requirements.txt`)
- Extends base requirements
- VLM frameworks (transformers, accelerate, bitsandbytes)
- CLIP specific (ftfy, regex, sentencepiece)
- Training optimization (torch, tensorboard)
- Deployment (onnx, onnxruntime)

### Hybrid Requirements (`hybrid_requirements.txt`)
- Combines YOLO and VLM dependencies
- For hybrid approaches using both YOLO and VLM

## Google Colab Setup

For Google Colab environments, use the `--colab` flag:

```bash
python scripts/common/setup_environment.py --experiment yolo --colab
```

This will:
1. Install conda if not available
2. Install uv for dependency management
3. Install experiment-specific requirements
4. Verify the installation

## Verification

To verify that all dependencies are properly installed:

```bash
python scripts/common/setup_environment.py --experiment yolo --verify
```

This will check for key packages:
- numpy, pandas, matplotlib, seaborn
- PIL (Pillow), cv2 (opencv-python)
- torch, transformers

## Troubleshooting

1. **Missing packages**: Run the setup script with `--verify` to identify missing packages
2. **Version conflicts**: Use virtual environments for each experiment type
3. **Colab issues**: Ensure you're using the correct Python version and have sufficient disk space 