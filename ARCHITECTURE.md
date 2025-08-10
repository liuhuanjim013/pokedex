# Pokemon Classifier Project Architecture

## 1. Project Overview
This project implements a Pokemon classifier using multiple approaches:
1. YOLOv3 (baseline)
2. Vision Language Models (VLM)
3. Hybrid approach

### Goals
1. Reproduce YOLOv3 blog post results
2. Improve accuracy with modern techniques
3. Compare different approaches
4. Deploy on IoT devices

### Key Requirements
1. Handle all Pokemon (1025 classes)
2. Fast inference on IoT devices
3. High accuracy (>90%)
4. Robust to real-world conditions

## 2. Data Strategy & Organization

### YOLO Dataset Format
```
processed/yolo_dataset/
├── images/
│   ├── train/           # 70% of each Pokemon's images
│   │   ├── 0001_1.jpg  # Bulbasaur image 1 (416x416)
│   │   ├── 0001_2.jpg  # Bulbasaur image 2
│   │   ├── 0002_1.jpg  # Ivysaur image 1
│   │   └── ...
│   ├── validation/      # 15% of each Pokemon's images
│   │   ├── 0001_3.jpg  # Bulbasaur image 3
│   │   ├── 0002_2.jpg  # Ivysaur image 2
│   │   └── ...
│   └── test/           # 15% of each Pokemon's images
│       ├── 0001_4.jpg  # Bulbasaur image 4
│       ├── 0002_3.jpg  # Ivysaur image 3
│       └── ...
└── labels/
    ├── train/          # Labels for training images
    │   ├── 0001_1.txt  # Contains: "0 0.5 0.5 1.0 1.0"
    │   ├── 0001_2.txt  # Contains: "0 0.5 0.5 1.0 1.0"
    │   ├── 0002_1.txt  # Contains: "1 0.5 0.5 1.0 1.0"
    │   └── ...
    ├── validation/     # Labels for validation images
    └── test/          # Labels for test images
```

**Key Format Requirements:**
1. **Images**:
   - Size: 416x416 pixels
   - Format: JPEG
   - Naming: `{pokemon_id}_{image_number}.jpg`
   - Pokemon IDs: 0001-1025 (zero-padded)

2. **Labels**:
   - Format: `{class_id} {x_center} {y_center} {width} {height}`
   - Class IDs: 0-1024 (0-based indexing)
   - Coordinates: Normalized (0.0-1.0)
   - Full-image boxes: Always "0.5 0.5 1.0 1.0"
   - Naming: Same as image but .txt extension

3. **Splits**:
   - Per-Pokemon splitting (70/15/15)
   - Each Pokemon's images split independently
   - All splits maintain class balance
   - No data leakage between splits

4. **Class Mapping**:
   - Bulbasaur: ID 0001 → class_id 0
   - Ivysaur: ID 0002 → class_id 1
   - Last Pokemon: ID 1025 → class_id 1024

### Directory Structure
```
pokedex/
├── data/                          # Local data storage (gitignored)
│   ├── raw/                      # Raw downloaded datasets
│   │   └── all_pokemon/         # All Pokemon data (1025 folders)
│   │       ├── 0001/            # Bulbasaur - all images
│   │       ├── 0002/            # Ivysaur - all images
│   │       └── ...              # Pokemon 003-1025
│   ├── processed/                # Shared preprocessed data (gitignored)
│   │   ├── images/              # Resized images for all models
│   │   ├── metadata/            # Dataset info, Pokemon mappings
│   │   └── yolo_dataset/        # YOLO format dataset (see above)
│   └── splits/                  # Train/val/test splits (gitignored)
│       ├── train/
│       ├── validation/
│       └── test/
├── models/                       # Model storage
│   ├── checkpoints/             # Training checkpoints
│   ├── final/                   # Final trained models
│   ├── compressed/              # Optimized for IoT
│   └── configs/                 # Model configurations
├── src/                         # Source code (reusable modules)
│   ├── data/                    # Data processing modules
│   │   ├── preprocessing.py     # Data preprocessing class
│   │   ├── augmentation.py      # Data augmentation utilities
│   │   └── dataset.py          # Dataset loading classes
│   ├── models/                  # Model implementations
│   │   ├── yolo/               # YOLO model classes
│   │   │   ├── trainer.py      # YOLO training class
│   │   │   ├── checkpoint_manager.py # Checkpoint utilities
│   │   │   └── wandb_integration.py # W&B integration class
│   │   ├── vlm/                # VLM model classes
│   │   └── hybrid/             # Hybrid model classes
│   ├── training/                # Training pipeline modules
│   │   ├── yolo/               # YOLO training utilities
│   │   ├── vlm/                # VLM training utilities
│   │   └── hybrid/             # Hybrid training utilities
│   ├── evaluation/              # Evaluation modules
│   │   ├── yolo/               # YOLO evaluation classes
│   │   ├── vlm/                # VLM evaluation classes
│   │   └── hybrid/             # Hybrid evaluation classes
│   └── deployment/              # IoT deployment modules
│       ├── yolo/               # YOLO deployment utilities
│       ├── vlm/                # VLM deployment utilities
│       └── hybrid/             # Hybrid deployment utilities
├── scripts/                     # Utility scripts (executable)
│   ├── yolo/                   # YOLO-specific scripts
│   │   ├── setup_colab_training.py # Colab environment setup
│   │   ├── train_yolov3_baseline.py # Baseline training script
│   │   ├── train_yolov3_improved.py # Improved training script
│   │   ├── resume_training.py  # Resume from checkpoint script
│   │   └── evaluate_model.py   # Evaluation script for trained models
│   ├── vlm/                    # VLM-specific scripts
│   ├── hybrid/                 # Hybrid-specific scripts
│   └── common/                 # Common utility scripts
│       ├── setup_environment.py # Environment setup
│       ├── dataset_analysis.py # Dataset analysis script
│       ├── upload_dataset.py   # Dataset upload script
│       ├── experiment_manager.py # Experiment management
│       └── data_processor.py   # Data processing script
├── notebooks/                   # Jupyter notebooks
│   ├── yolo_experiments/       # YOLO experiments
│   ├── vlm_experiments/        # VLM experiments
│   ├── hybrid_experiments/     # Hybrid experiments
│   └── deployment/             # Deployment testing
├── configs/                     # Configuration files
│   ├── yolov3/                 # YOLOv3 configurations
│   │   ├── data_config.yaml
│   │   ├── training_config.yaml
│   │   ├── reproduction_config.yaml
│   │   └── improvement_config.yaml
│   ├── yolov8/                 # YOLOv8 configurations
│   ├── clip/                   # CLIP configurations
│   ├── smolvm/                 # SMoLVM configurations
│   └── hybrid/                 # Hybrid configurations
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── .github/                     # GitHub workflows
│   └── workflows/
├── .gitignore
├── requirements.txt             # Base requirements
├── pyproject.toml
└── README.md
```

### Data Workflow
```python
# Shared data processing pipeline with optimizations
raw_data/ → preprocessing.py (multiprocessing) → processed_data/ → model_specific_format() → Hugging Face Hub

# Performance Optimizations
1. **Processing Stage**:
   - Multiprocessing: 8 workers for parallel image processing
   - Lookup Tables: O(1) access for raw-to-processed mapping
   - Batched Operations: Process images in batches of 100
   - Progress Tracking: Real-time progress with percentage
   - Caching: Skip already processed splits
   - Type Handling: Support both raw bytes and PIL Images
   - Format Validation: Verify image types and formats

2. **Upload Stage**:
   - Network Resilience: Exponential backoff with retries
   - Memory Management: Memory-mapped features, no image decoding
   - Git Configuration: Optimized for large file uploads
   - Progress Saving: Commit files as they are uploaded
   - Shard Size: 200MB shards for reliable uploads
   - Concurrency: 2 workers to prevent timeouts
   - Timeouts: 10-minute timeout per request
   - Buffer Sizes: Increased for large files

3. **Download & Verification Stage**:
   - Format Conversion: HF dataset → YOLO format
   - Progress Tracking: tqdm for visual feedback
   - Caching: Skip existing files
   - Class ID Handling: Convert to 0-based indexing
   - Image Processing: Handle both bytes and PIL objects
   - Error Handling: Type checks and descriptive errors
   - Config Updates: Dynamic path resolution

# Step 1: Process raw data once (shared across all models)
raw_data/all_pokemon/ → preprocessing.py → processed_data/images/

# Step 2: Create model-specific formats
processed_data/images/ → create_yolo_dataset() → yolo_dataset/
processed_data/images/ → create_clip_dataset() → clip_dataset/
processed_data/images/ → create_smolvm_dataset() → smolvm_dataset/

# Step 3: Upload to Hugging Face (not GitHub)
yolo_dataset/ → upload_to_hf.py (with retry) → Hugging Face Hub
clip_dataset/ → upload_to_hf.py (with retry) → Hugging Face Hub

# Upload Configuration
git_config = {
    "http.postBuffer": "512M",      # Large file buffer
    "http.lowSpeedLimit": "1000",   # Min bytes/sec
    "http.lowSpeedTime": "600",     # Time window
    "http.maxRequestBuffer": "100M", # Max request size
    "core.compression": "0",        # No compression
    "http.timeout": "600"           # 10-min timeout
}
```

### Version Control Strategy
- **GitHub**: Code, configs, documentation, workflows (NO DATA)
- **Hugging Face**: Datasets, model checkpoints, experiment results
- **W&B**: Training logs, metrics, model comparisons
- **Local**: Raw and processed data (gitignored)

### Data Security & Storage
- **Raw Data**: Stored locally, never uploaded to GitHub
- **Processed Data**: Stored locally, uploaded to Hugging Face for Colab access
- **Gitignore**: All data directories and image files excluded from Git
- **Backup**: Raw data backed up separately (not in Git)
- **Upload Security**:
  - Network resilience with exponential backoff
  - Progress saving during upload
  - Automatic retry on network errors
  - Configurable timeouts and limits
  - Memory-efficient data handling
  - Test coverage for upload process

### Data Augmentation Strategy
- **Geometric**: Rotation, scaling, perspective transforms
- **Photometric**: Brightness, contrast, color jittering
- **Environmental**: Different lighting conditions, backgrounds
- **Real-world**: Motion blur, focus variations, partial occlusions
- **Multi-scale**: Various distances and viewing angles

### Data Collection Pipeline
```python
# Automated data collection
- Web scraping Pokemon images
- Community-contributed photos
- Synthetic data generation
- Cross-validation with multiple sources
```

## 3. Model Architecture

### YOLOv3 Baseline
1. **Architecture**:
   - Darknet-53 backbone
   - FPN neck
   - Detection head modified for classification
   - Input size: 416x416
   - Output: 1025 class probabilities

2. **Training**:
    - Learning rate: 0.0001
    - Scheduler: cosine with 5 warmup epochs
    - Batch size: 16
    - Epochs: 100
    - Augmentation: Standard YOLO augmentations

3. **Evaluation**:
   - Top-1 accuracy
   - Top-5 accuracy
   - Confusion matrix
   - Per-class accuracy
   - Inference speed

### Vision Language Models
1. **CLIP**:
   - Zero-shot classification
   - Fine-tuning experiments
   - Text prompt engineering

2. **SMoLVM**:
   - Lightweight VLM for IoT
   - Distilled from CLIP
   - Optimized inference

### Hybrid Approach
1. **Architecture**:
   - YOLO feature extractor
   - VLM semantic embeddings
   - Fusion module
   - Classification head

2. **Training**:
   - Two-phase training
   - Knowledge distillation
   - Multi-task learning

## 4. Training Infrastructure

### Compute Resources
1. **Development**:
   - Local: RTX 3080
   - Cloud: Colab Pro+ (A100/L4)

2. **Google Colab Setup**:
   - Conda installation: `/content/miniconda3/` (ephemeral, no Drive persistence)
   - Environment: `pokemon-classifier` with Python 3.9
   - Dependencies: ultralytics, torch, wandb, datasets
   - Storage: Repository files in `/content/pokedex/pokedex/`
   - Models: Saved to `/content/models/` directories

3. **Production Training**:
   - Cloud: AWS g4dn.xlarge
   - Multi-GPU support
   - Distributed training

### Google Colab Training Configuration
1. **Environment Setup**:
   - Conda installation: `/content/miniconda3/` (ephemeral, reinstalled per session)
   - Environment activation: `source conda.sh && conda activate pokemon-classifier`
   - PATH management: Dynamic conda path detection (`/content/miniconda3/bin/conda`)
   - Shell script: `activate_env.sh` for easy activation

2. **Data Persistence**:
   - Dataset: Downloaded from Hugging Face to `/content/pokedex/pokedex/data/yolo_dataset/`
   - Models: Saved to `/content/models/` directories
   - Checkpoints: Local storage (ephemeral per session)
   - Configurations: Local configs with dynamic path updates

3. **Model Loading Strategy**:
    - **Primary**: Load official YOLOv3 weights via Ultralytics (downloads on first use)
    - **Fallback**: Load from Ultralytics hub (`YOLO("yolov3")`)
    - Local YAML fallback removed (previous `models/configs/yolov3.yaml` deleted)
   - **Cache management**: Auto-cleanup of corrupted weights
   - **Path resolution**: Dynamic working directory detection

### Experiment Tracking
1. **W&B Integration**:
    - Metrics logging with Ultralytics built-in integration
    - Real-time training monitoring (loss, mAP, precision, recall)
    - Live metric callbacks from Ultralytics events to W&B (epoch-end and val-end)
   - Artifact storage for model checkpoints
   - Experiment comparison between runs
   - Hyperparameter tuning with sweeps
   - Run resumption with persistent run IDs

2. **Checkpointing**:
   - Regular saves (every 10 epochs, configurable)
   - Best model saves with metadata
   - Resume capability with W&B run ID matching
   - Model export for deployment
   - Local storage in Colab (ephemeral per session)
    - Automatic cleanup of old checkpoints
    - Auto-backup of training outputs to Google Drive every 30 minutes

3. **Training Pipeline Integration**:
   - Dataset verification and preprocessing
   - Automatic directory creation
   - Progress tracking with visual feedback
   - Error handling for I/O issues (Google Drive)
   - Dynamic configuration updates

### Testing Strategy
1. **Unit Tests**:
   - Data processing
   - Model components
   - Training pipeline
   - Evaluation metrics

2. **Integration Tests**:
   - End-to-end training
   - Multi-GPU training
   - Checkpoint loading
   - Model export

3. **Performance Tests**:
   - Inference speed
   - Memory usage
   - GPU utilization
   - Training throughput

## 5. Deployment Strategy

### IoT Deployment
1. **Model Optimization**:
   - Quantization
   - Pruning
   - Knowledge distillation
   - TensorRT conversion

2. **Hardware Targets**:
   - Jetson Nano
   - Raspberry Pi 4
   - Edge TPU
   - Mobile devices

3. **Deployment Pipeline**:
   - Model conversion
   - Performance testing
   - Deployment packaging
   - Update mechanism

#### K210 (Maix Bit) Export & Deployment
- **Training assumption**: YOLO detection with full-image bounding boxes (one box per image), compatible with K210 YOLO runtime.
- **Export**: Convert trained `.pt` to ONNX with fixed input (e.g., 320x320), static shape, opset 12, simplified graph.
- **Compile**: Use nncase (ncc, target k210) with INT8/UINT8 quantization and a calibration dataset to generate `.kmodel`.
- **Artifacts**: Ship `model.kmodel`, `classes.txt`, and (if needed) `anchors.txt` to the device; configure thresholds/NMS in firmware.
- **Runtime**: MaixPy/C uses KPU YOLO runner; since labels are full-image boxes, take top detection’s class as prediction.
- **Helper script**: `scripts/yolo/export_k210.py` automates ONNX export, nncase compilation, and packaging of artifacts.

### Production Infrastructure
1. **Model Serving**:
   - TensorFlow Serving
   - ONNX Runtime
   - TensorRT
   - Custom inference server

2. **Monitoring**:
   - Inference latency
   - Throughput
   - Error rates
   - Resource usage

3. **Maintenance**:
   - Model updates
   - A/B testing
   - Performance monitoring
   - Error tracking

## 6. Future Improvements

### Model Improvements
1. **Architecture**:
   - Try YOLOv8
   - Experiment with ViT
   - Test EfficientNet
   - Custom architectures

2. **Training**:
   - Advanced augmentations
   - Curriculum learning
   - Self-supervised pretraining
   - Multi-task learning

### Infrastructure Improvements
1. **Training**:
   - Multi-node training
   - Mixed precision
   - Gradient accumulation
   - Dynamic batching

2. **Deployment**:
   - Edge optimization
   - Battery optimization
   - Compression techniques
   - Update strategies

### Dataset Improvements
1. **Data Quality**:
   - Better cleaning
   - Balanced augmentation
   - Hard negative mining
   - Active learning

2. **Data Sources**:
   - More real photos
   - Synthetic generation
   - Domain adaptation
   - Cross-dataset validation

## YOLO Training Rules

### Data Configuration
1. **YOLO Dataset Format Requirements**
   - Directory: `processed/yolo_dataset/`
   - Images: 416x416 JPEG in `images/{split}/`
   - Labels: YOLO format in `labels/{split}/`
   - Naming: `{pokemon_id}_{image_number:03d}.{ext}` (3-digit padding)
   - Class IDs: 0-based (0-1024)
   - Full-image boxes: "0.5 0.5 1.0 1.0"
   - Multiprocessing: 8 workers, 100 images per batch
   - Progress Tracking: Percentage complete per split
   - Lookup Tables: O(1) access for raw-to-processed mapping

2. **Dataset Verification**
   - Check Hugging Face access first
   - Verify YOLO config format
   - Validate class count (1025)
   - Check split names match
   - Verify per-Pokemon 70/15/15 splits
   - Validate label format and class IDs (0-based)
   - Check image sizes (416x416)
   - Verify image data types (bytes or PIL)
   - Skip already processed splits
   - Show progress with tqdm
   - Handle errors with descriptive messages
   - Update config paths dynamically

### Model Loading Strategy
1. **YOLOv3 Loading Priority**
   - **Primary**: Load official YOLOv3 weights (Ultralytics auto-download)
   - **Fallback**: Load from Ultralytics hub (`YOLO("yolov3")`)
   - YAML fallback removed (file `models/configs/yolov3.yaml` deleted)
   - **Cache Management**: Auto-cleanup corrupted weights
   - **Path Resolution**: Dynamic working directory detection

2. **Training Configuration**
   - Classes: 1025 (all Pokemon generations)
   - Input size: 416x416 pixels
   - Learning rate: 1e-4, cosine schedule with 5 warmup epochs

### Training Process
1. **Initialization Order**
   - Verify environment setup and conda activation
   - Check dataset access from Hugging Face
   - Initialize W&B tracking with run resumption
   - Set up model with fallback loading strategy
   - Create required directories automatically

2. **Checkpoint Management**
   - Save metadata with checkpoints (W&B run ID, epoch, metrics)
   - Track W&B run ID for resumption
   - Record actual progress vs saved epochs
   - Handle mid-epoch interruptions
    - Local storage (ephemeral per Colab session)
    - Auto-backup of training outputs to Google Drive every 30 minutes
    - Resume scans multiple default paths for latest checkpoint:
      - Custom checkpoints directory
      - `pokemon-classifier/<run-name>/weights`
      - `pokemon-yolo-training/<run-name>/weights`

3. **Error Handling**
   - File I/O errors: Retry with exponential backoff for large operations
   - Missing conda: Dynamic path detection and helpful messages
   - Model loading failures: Multiple fallback strategies
   - Dataset access: Hugging Face authentication and caching

### Baseline Training Results & Learnings
1. **Performance Metrics (48 epochs completed)**
   - **Best mAP50**: 0.9746 (epoch 44)
   - **Best mAP50-95**: 0.803 (epoch 44)
   - **Training Loss**: Steady decrease until epoch 45
   - **Validation Loss**: Increasing trend after epoch 30 (overfitting)

2. **Critical Issues Identified**
   - **Training Instability**: Dramatic performance drop at epoch 45
     - mAP50 dropped from 0.9746 to 0.00041
     - mAP50-95 dropped from 0.803 to 0.00033
   - **Overfitting**: Validation loss increasing while training loss decreasing
   - **Learning Rate**: 1e-4 may be too high for 1025 classes
   - **Augmentation**: Minimal augmentation insufficient for generalization

3. **Improvement Opportunities**
   - **Early Stopping**: Implement to prevent overfitting (patience=10)
   - **Learning Rate**: Reduce to 5e-5 or implement adaptive scheduling
   - **Augmentation**: Add rotation, shear, mosaic, mixup
   - **Regularization**: Add dropout, weight decay, label smoothing
   - **Batch Size**: Consider increasing to 32 for better gradient estimates
   - **Monitoring**: Add validation metrics monitoring for early detection

4. **Baseline Configuration (Current)**
   - Learning rate: 1e-4 (cosine schedule)
   - Batch size: 16
   - Warmup epochs: 5
   - Augmentation: Horizontal flip only (0.5 probability)
   - No early stopping
   - No additional regularization

### W&B Integration
1. **Configuration**
   - Use singleton pattern for initialization
   - Enable offline fallback with environment variables
   - Disable code/git tracking for Colab
   - Use personal account (liuhuanjim013-self)
   - Project: "pokemon-classifier"

2. **Metrics Logging**
   - **Built-in Ultralytics**: Automatic loss, mAP, precision, recall
   - **Real-time Monitoring**: Live training progress (callbacks on epoch/val end)
   - **System Metrics**: GPU usage, memory consumption
   - **Custom Logging**: Configuration parameters and experiment metadata
   - **Run Persistence**: Save run ID to disk for resumption

3. **Resume Strategy**
   - Try specified checkpoint with matching W&B run ID
   - Match W&B run ID from saved checkpoint metadata
   - Fall back to latest checkpoint if run ID missing
   - Support forcing new run with --force-new-run flag
   - Maintain metrics continuity during resume

4. **Error Handling**
   - Clean up W&B runs on training failure
   - Proper exception handling with informative messages
   - Resource cleanup and session management
   - Offline mode fallback for network issues

### Testing Requirements
1. **Training Tests**
   - Verify setup works
   - Check arguments
   - Test data loading
   - Validate progress

2. **W&B Tests**
   - Test initialization
   - Verify resumption
   - Check offline mode
   - Validate cleanup

3. **Checkpoint Tests**
   - Test metadata saving
   - Verify loading
   - Check progress tracking
   - Test interruption recovery

4. **Dataset Format Tests**
   - Verify image sizes (416x416)
   - Check label format
   - Validate class IDs (0-based)
   - Verify per-Pokemon splits
   - Test image-label pairs match