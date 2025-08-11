# Pokemon Classifier - Task Breakdown

## Project Overview
Build a real-time Pokemon classifier that can identify 1025 Pokemon species from camera input, optimized for deployment on Sipeed Maix Bit RISC-V IoT device. Start by reproducing the baseline work from the blog post using Google Colab (instead of binary training software) to establish workflows and understand the challenges, then extend to full 1025 Pokemon classification with improved approaches.

## Current Status & Next Steps

### ✅ COMPLETED TASKS
1. **Task 1.1**: Environment setup with conda + uv
2. **Task 1.2**: Comprehensive dataset analysis (ALL 128,768 images)
   - ✅ Quality assessment (100% validity rate)
   - ✅ Within-class splitting strategy
   - ✅ Dataset statistics and visualizations
   - ✅ Bias analysis and augmentation planning
3. **Task 2.1**: Data Processing & Validation (COMPLETED)
   - ✅ Multiprocessing preprocessing pipeline implemented
   - ✅ All 128,768 images processed in ~92 seconds
   - ✅ Correct Pokemon names mapping (all 1025 Pokemon)
   - ✅ YOLO dataset created with proper format
   - ✅ Within-class splitting (70/15/15) implemented
4. **Task 2.1.5**: Enhanced Dataset Verification (COMPLETED)
   - ✅ Image content verification (102 sample images)
   - ✅ Processing quality validation (100% properly processed)
   - ✅ YOLO format verification (perfect detection format)
   - ✅ Statistical quality analysis (excellent diversity)
   - ✅ Dataset integrity validation (14,458 image-label pairs)
5. **Task 2.2**: Hugging Face Dataset Upload (COMPLETED)
   - ✅ YOLO dataset uploaded to Hugging Face Hub
   - ✅ Dataset URL: https://huggingface.co/datasets/liuhuanjim013/pokemon-yolo-1025
   - ✅ Comprehensive dataset card with usage examples
   - ✅ Proper attribution to original author 弦masamasa (xianmasamasa)
   - ✅ License: CC BY-NC-SA 4.0 (Creative Commons)
   - ✅ Citation requirements for both original and this dataset
   - ✅ Usage example created for Google Colab training
   - ✅ Optimized upload process with multiprocessing (8 workers)
   - ✅ Added O(1) lookup tables for raw-to-processed mapping
   - ✅ Implemented batched processing (100 images per batch)
   - ✅ Added detailed progress tracking with percentages
   - ✅ Fixed Hugging Face API changes for dataset card handling
   - ✅ Improved error handling and validation
6. **Task 2.3**: Dataset Download & Verification (COMPLETED)
   - ✅ Implemented robust HF dataset loading
   - ✅ Added type-safe image processing (bytes & PIL)
   - ✅ Created local YOLO format extraction
   - ✅ Added progress tracking with tqdm
   - ✅ Implemented caching for processed splits
   - ✅ Fixed class ID indexing (0-based)
   - ✅ Added comprehensive error handling
   - ✅ Implemented dynamic config updates

### 🎯 CURRENT PRIORITY: K210 Model Size Optimization (CRITICAL)
**Priority**: CRITICAL  
**Status**: Training Completed Successfully - Model Too Large for K210  
**Next Action**: Implement YOLOv5n with class reduction strategy

**Current Status:**
1. ✅ **K210 Training Pipeline** - FULLY IMPLEMENTED & WORKING
   - YOLOv3-tiny-ultralytics training successfully started
   - Model: 12.66M parameters (88% reduction from full YOLOv3)
   - Architecture: 53 layers, 20.1 GFLOPs
   - Training: 90,126 train + 19,316 validation images loaded
   - Configuration: 224x224 input, 8 batch size, 5e-5 learning rate
2. ✅ **Training Infrastructure** - FULLY OPERATIONAL
   - W&B integration working (project: pokemon-classifier)
   - Resume pattern implemented (follows baseline/improved scripts)
   - Auto-backup functionality via YOLOTrainer integration
   - Command line compatibility (--resume, --fresh, --checkpoint)
3. ✅ **Dataset Strategy** - SUCCESSFULLY VALIDATED
   - Uses existing `liuhuanjim013/pokemon-yolo-1025` dataset
   - Runtime resizing from 416x416 to 224x224 working
   - All 1025 Pokemon classes maintained (not reduced)
   - No new dataset creation required
4. ✅ **K210 Configuration** - OPTIMIZED & APPLIED
   - Conservative augmentation (mosaic=0.0, mixup=0.0 for K210 stability)
   - Extended patience (20 epochs) for K210 convergence
   - SGD optimizer forced to prevent learning rate override
   - Early stopping and weight decay applied
5. 🔄 **Training In Progress** - CURRENTLY RUNNING
   - Training started successfully without resume conflicts
   - 200 epochs total with early stopping patience=20
   - Real-time monitoring via W&B dashboard
   - Checkpoint saving every epoch with metadata

**K210 Optimization Achievements:**
- **Model Size**: 88% parameter reduction (12.66M vs 104.45M)
- **Memory Efficiency**: 71% input buffer reduction (224x224 vs 416x416)
- **Architecture**: YOLOv3-tiny-ultralytics proven working
- **Full Coverage**: All 1025 Pokemon classes maintained
- **Infrastructure**: Complete training and backup pipeline

**Training Progress Monitoring:**
- **W&B Run**: yolov3-tinyu-k210-optimized (rwcl26gk)
- **Project**: pokemon-classifier (for comparison with other models)
- **Checkpoints**: Saved to pokemon-classifier/yolov3n_k210_optimized/
- **Auto-backup**: Every 30 minutes to Google Drive
- **Resume Ready**: Can resume from any checkpoint

**Critical Issue Identified:**
1. ✅ **Training Success** - COMPLETED
   - Achieved 91.7% mAP50 (exceeded targets)
   - Stable training with aggressive parameters
   - Perfect convergence without overfitting
2. ✅ **Export Pipeline** - WORKING
   - ONNX export successful (48.4MB)
   - nncase compilation successful (49MB kmodel)
   - Complete export infrastructure validated
3. 🚨 **Deployment Blocker** - MODEL TOO LARGE
   - Model Size: 49MB vs 16MB K210 Flash limit (3x over)
   - Runtime Memory: 59MB vs 6MB K210 RAM limit (10x over)
   - Architecture: Even "tiny" YOLO with 1025 classes exceeds constraints

**Immediate Next Steps:**
1. 🚨 **YOLOv5n Implementation** - HIGH PRIORITY
   - Switch to YOLOv5n (1.9M vs 12.66M parameters - 6.7x reduction)
   - Test with 151 classes (Gen 1) for size validation
   - Apply knowledge distillation from 91.7% teacher model
2. 📋 **Class Reduction Strategy** - PARALLEL TASK
   - Evaluate hierarchical classification (generation → specific)
   - Test class grouping approaches (similar Pokemon)
   - Maintain accuracy through advanced techniques
3. 🎯 **K210 Deployment Validation** - FINAL PHASE
   - Target: <2MB model size after quantization
   - Verify <6MB runtime memory usage
   - Real hardware testing on Sipeed Maix Bit

### 🎯 CURRENT PRIORITY: YOLOv3 Training Optimization & W&B Integration (COMPLETED - BASELINE)
**Priority**: CRITICAL  
**Status**: Baseline Training Completed - Issues Identified  
**Next Action**: Implement improved training with fixes for identified issues

**Current Status:**
1. ✅ **Google Colab environment setup** - COMPLETED
2. ✅ **Dataset loading from Hugging Face** - COMPLETED (128,768 files)
3. ✅ **YOLOv3 training pipeline** - COMPLETED (1025 classes)
4. ✅ **Baseline training completed** - 48 epochs, critical issues identified
5. ✅ **W&B integration** - Live training metrics via Ultralytics callbacks
6. ✅ **Baseline performance documented** - Training completed with findings

**Baseline Training Results (48 epochs):**
- **Best mAP50**: 0.9746 (epoch 44)
- **Best mAP50-95**: 0.803 (epoch 44)
- **Critical Issue**: Training instability at epoch 45
  - mAP50 dropped from 0.9746 to 0.00041
  - mAP50-95 dropped from 0.803 to 0.00033
- **Overfitting**: Validation loss increasing after epoch 30
- **Configuration**: LR 1e-4, batch 16, minimal augmentation

**Immediate Next Steps:**
1. **Implement improved training** with fixes for identified issues
2. **Add early stopping** to prevent overfitting (patience=10)
3. **Reduce learning rate** to 5e-5 for better stability
4. **Enhance augmentation** (rotation, shear, mosaic, mixup)
5. **Add regularization** (dropout, weight decay, label smoothing)
6. **Increase batch size** to 32 for better gradient estimates

### 📋 IMMEDIATE TODO LIST

#### Authentication & Environment (COMPLETED):
- [x] **Hugging Face Authentication**:
  - [x] Test different auth methods (env var, token file, login)
  - [x] Verify dataset access works (21,689 training examples)
  - [x] Document token priority and storage
  - [x] Create test script for auth verification
  - [x] Set up git credential helper (store)
  - [x] Test push access to Hugging Face
  - [x] Document git credential setup
- [x] **W&B Authentication & Integration**:
  - [x] Set up W&B project and API key
  - [x] Test experiment tracking (run-20250806_101644-lnoe0hkj)
  - [x] Configure project structure
  - [x] Set up run resumption with run ID persistence
  - [x] Configure offline mode fallback with environment variable
  - [x] Verify metrics-only logging works
  - [x] Implement W&B resume from saved run ID
  - [x] Add run ID persistence to disk for resume
  - [x] Test W&B resume functionality
  - [x] Implement singleton pattern for W&B integration
  - [x] Add proper environment variable handling
  - [x] Ensure string conversion for run IDs
  - [x] Add comprehensive error handling
  - [x] Test all W&B resume scenarios
  - [x] Document W&B integration patterns
  - [x] Implement accurate step counting during resume
  - [x] Add checkpoint metadata with W&B run ID
  - [x] Track both saved and actual epochs
  - [x] Handle mid-epoch interruptions
  - [x] Support forcing new run with --force-new-run
  - [x] Add checkpoint matching by W&B run ID
  - [x] Maintain metrics continuity during resume
  - [x] Test all resume scenarios (checkpoint, run ID, latest)
- [x] **Dataset Access**:
  - [x] Verify dataset loading works
  - [x] List dataset files successfully
  - [x] Confirm training examples count
  - [x] Document dataset structure

#### Model Loading & Training Resume (COMPLETED):
- [x] **Model Loading Strategy**:
  - [x] Primary: Load official YOLOv3 weights (auto-download)
  - [x] Fallback: Ultralytics hub (`YOLO("yolov3")`)
  - [x] Removed YAML fallback (deleted `models/configs/yolov3.yaml`)
  - [x] Add comprehensive error handling
  - [x] Document model loading strategy
- [x] **Training Resume**:
  - [x] Implement checkpoint management
  - [x] Add W&B run ID persistence
  - [x] Support resuming from specific checkpoints
  - [x] Add automatic latest checkpoint detection
  - [x] Implement W&B run resumption
  - [x] Test resume functionality
  - [x] Document resume strategy

#### Critical Priority (COMPLETED/IN PROGRESS):
- [x] **Google Colab environment setup** - COMPLETED
  - [x] Conda installation with Google Drive persistence
  - [x] Environment: `pokemon-classifier` with all dependencies
  - [x] Dynamic conda path detection and shell activation
  - [x] Google Drive mounting and persistence strategy
- [x] **Dataset loading and processing** - COMPLETED
  - [x] Hugging Face dataset access (liuhuanjim013/pokemon-yolo-1025)
  - [x] YOLO format conversion with proper class IDs (0-based)
  - [x] Dataset verification (128,768 files: 90,126 train + 19,316 val + 19,326 test)
  - [x] Dynamic path configuration for Colab environment
- [x] **YOLOv3 model setup** - COMPLETED
  - [x] Model loading with fallback strategy (official → hub)
  - [x] 1025 class configuration for all Pokemon generations
  - [x] Training pipeline integration with Ultralytics
- [x] **Training infrastructure** - COMPLETED
  - [x] W&B integration with experiment tracking
  - [x] Checkpoint management with metadata
  - [x] Directory auto-creation for models/logs/checkpoints
  - [x] Progress tracking and error handling
- [x] **Configuration files** - COMPLETED
  - [x] `configs/yolov3/baseline_config.yaml` - baseline training parameters (LR 1e-4, cosine scheduler, 5 warmup epochs)
  - [x] `configs/yolov3/yolo_data.yaml` - dataset configuration (1025 classes)
  - [x] Dynamic path updates for Colab environment
- [x] **Training scripts** - COMPLETED
  - [x] `scripts/yolo/train_yolov3_baseline.py` - main training script
  - [x] `scripts/yolo/setup_colab_training.py` - environment setup
  - [x] `scripts/yolo/activate_env.sh` - conda environment activation
  - [x] Error handling for Google Drive I/O issues
- [x] **Source code modules** - COMPLETED
  - [x] `src/training/yolo/trainer.py` - core training class with model loading
  - [x] Enhanced W&B integration (built-in Ultralytics logging + callbacks)
  - [x] Robust error handling and fallback mechanisms
  - [x] Auto-backup worker to Google Drive (30 min interval)
- [x] **YOLOv3 baseline training** - COMPLETED (48 epochs)
  - [x] Training completed with baseline configuration
  - [x] Performance metrics documented (mAP50: 0.9746, mAP50-95: 0.803)
  - [x] Critical issues identified (training instability, overfitting)
  - [x] Improvement opportunities documented

#### Current Priority (Next 1-2 days):
- [ ] **Implement YOLOv3 improvements** - HIGH PRIORITY
  - [ ] Create improved training configuration with fixes
  - [ ] Add early stopping (patience=10) to prevent overfitting
  - [ ] Reduce learning rate to 5e-5 for better stability
  - [ ] Enhance augmentation (rotation, shear, mosaic, mixup)
  - [ ] Add regularization (dropout, weight decay, label smoothing)
  - [ ] Increase batch size to 32 for better gradient estimates
- [ ] **Test improved training** with enhanced parameters
- [ ] **Compare improved vs baseline** performance
- [ ] **Document improvement results** and lessons learned

#### High Priority (Next 3-5 days):
- [ ] **Implement YOLOv3 improvements** (augmentation, scheduling)
- [ ] **Compare improved vs baseline** performance
- [ ] **Create CLIP dataset format** (text prompts + images)
- [ ] **Create SMoLVM dataset format** (text prompts + images)
- [ ] **Upload additional datasets to Hugging Face** for Colab access
- [ ] **Set up W&B sweeps** for hyperparameter optimization
- [ ] **Implement automatic checkpoint cleanup** and backup

#### Medium Priority (Next week):
- [ ] **Test newer YOLO variants** (v8, v9, v10)
- [ ] **Implement VLM training pipelines** (CLIP, SMoLVM)
- [ ] **Create model comparison framework**
- [ ] **Implement multi-frame aggregation**
- [ ] **Create comprehensive W&B dashboard** for experiment comparison
- [ ] **Upload final models to Hugging Face** for sharing

### 🎯 SUCCESS METRICS FOR NEXT PHASE
- **Colab Setup**: YOLOv3 training environment ready in Google Colab
- **Dataset Loading**: Successfully load from Hugging Face Hub
- **Baseline Training**: YOLOv3 model trained on 1025 classes (original blog reproduction)
- **Performance Baseline**: Documented accuracy and limitations
- **Improvement Ready**: Enhanced YOLOv3 training pipeline implemented
- **Experiment Tracking**: W&B project with baseline and improvement experiments
- **Checkpoint Management**: Automatic save/resume functionality working
- **W&B Integration**: Real-time monitoring and visualization active
 - **Evaluation**: `scripts/yolo/evaluate_model.py` provides mAP and top-k metrics

### 🚨 BLOCKERS & RISKS
1. **Colab GPU Access**: May need Colab Pro for large-scale training
2. **Training Time**: 1025 classes may require significant training time
3. **Memory Constraints**: Large dataset may require batch size optimization
4. **Hugging Face Access**: Ensure dataset is publicly accessible
5. **W&B API Limits**: Monitor API usage for experiment tracking
6. **Google Drive Storage**: Ensure sufficient space for checkpoints
7. **Existing File Conflicts**: Need to merge/update existing configs and scripts

### 📊 PROGRESS SUMMARY
- **Phase 1**: 100% complete (environment + data analysis + preprocessing + verification + upload done)
- **Phase 2**: 100% complete (data processing done, dataset uploaded to Hugging Face)
- **Phase 3**: 60% complete (baseline training completed, improvements ready to implement)
- **Overall Project**: 45% complete

**Estimated Timeline**: 6 weeks remaining (original 10-week timeline)

**CURRENT FOCUS**: YOLO training improvements based on baseline analysis

### 🎯 File Organization & Cleanup Tasks

#### Existing Files to Update/Merge:
- [ ] **Configs**: 
  - [ ] Merge `configs/yolov3/original_blog_config.yaml` → `configs/yolov3/baseline_config.yaml`
  - [ ] Update `configs/yolov3/reproduction_config.yaml` with exact blog parameters
  - [ ] Create `configs/yolov3/improved_config.yaml` for enhanced training
- [ ] **Scripts**:
  - [ ] Merge `scripts/yolo/reproduce_original_blog.py` → `scripts/yolo/train_yolov3_baseline.py`
  - [ ] Update `scripts/yolo/setup_yolov3_experiment.py` → `scripts/yolo/setup_colab_training.py`
  - [ ] Create `scripts/yolo/train_yolov3_improved.py` for enhanced training
  - [ ] Create `scripts/yolo/resume_training.py` for checkpoint resume
- [ ] **Source Code**:
  - [ ] Update `src/training/yolo_trainer.py` → `src/training/yolo/trainer.py`
  - [ ] Update `src/training/yolo/yolov3_trainer.py` → merge into `src/training/yolo/trainer.py`
  - [ ] Create `src/training/yolo/checkpoint_manager.py`
  - [ ] Create `src/training/yolo/wandb_integration.py`
  - [ ] Add `scripts/yolo/evaluate_model.py` and wire into docs

#### Files to Remove (Wrong Design):
- [ ] Remove any files that don't follow the new architecture
- [ ] Clean up redundant configs and scripts
- [ ] Ensure all files are in correct locations according to `.cursorrules`

#### Files to Create (Missing):
- [ ] **Evaluation**: `src/evaluation/yolo/evaluator.py`
- [ ] **Evaluation**: `src/evaluation/yolo/metrics.py`
- [ ] **Notebooks**: `notebooks/yolo_experiments/baseline_training.ipynb`
- [ ] **Notebooks**: `notebooks/yolo_experiments/improved_training.ipynb`
- [ ] **Requirements**: `requirements/yolo_requirements.txt`

### 🎯 W&B Integration & Checkpoint Management

#### W&B Setup Strategy
**Project Configuration:**
- **Project Name**: "pokemon-classifier"
- **Experiment Names**: 
  - "yolov3-baseline-reproduction" (original blog parameters)
  - "yolov3-improved-training" (enhanced parameters)
  - "yolov3-hyperparameter-sweep" (optimization runs)
- **Metrics Tracking**: Loss curves, accuracy, mAP, learning rate, validation metrics
- **Artifacts**: Model checkpoints, training logs, evaluation results
- **Dashboard**: Real-time training progress and experiment comparison

**W&B Integration Features:**
- **Real-time Monitoring**: Live loss curves and accuracy plots
- **Hyperparameter Logging**: Track all training parameters and configurations
- **Model Comparison**: Side-by-side comparison of baseline vs improved models
- **Artifact Management**: Automatic upload of checkpoints and logs
- **Sweep Configuration**: Automated hyperparameter optimization
- **Alert System**: Notifications for training completion or issues

#### Checkpoint Management Strategy
**Save Strategy:**
- **Frequency**: Every 10 epochs (configurable)
- **Storage Location**: Google Drive for persistence across Colab sessions
- **Metadata**: Training configuration, epoch number, performance metrics
- **Version Control**: Checkpoint naming with timestamp and epoch

**Resume Strategy:**
- **Auto-detection**: Automatically find latest checkpoint on startup
- **Validation**: Verify checkpoint integrity before loading
- **Configuration**: Resume with same training parameters
- **Progress Tracking**: Continue from exact epoch and step

**Storage Management:**
- **Google Drive Integration**: Mount drive for persistent storage
- **Cleanup Policy**: Keep last 5 checkpoints, delete older ones
- **Backup Strategy**: Critical checkpoints uploaded to Hugging Face Hub
- **Space Optimization**: Compress old checkpoints to save space

#### Training Pipeline with W&B & Checkpoints
```python
# W&B Integration
import wandb
wandb.init(
    project="pokemon-classifier",
    name="yolov3-baseline-reproduction",
    config={
        "model": "yolov3",
        "dataset": "liuhuanjim013/pokemon-yolo-1025",
        "classes": 1025,
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.001,
        "original_blog": "https://www.cnblogs.com/xianmasamasa/p/18995912"
    }
)

# Checkpoint Management
def save_checkpoint(model, optimizer, epoch, metrics):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': wandb.config
    }
    torch.save(checkpoint, f'/content/drive/checkpoints/yolov3_epoch_{epoch}.pt')
    wandb.save(f'/content/drive/checkpoints/yolov3_epoch_{epoch}.pt')

def load_latest_checkpoint(model, optimizer):
    # Find latest checkpoint
    checkpoint_files = glob.glob('/content/drive/checkpoints/yolov3_epoch_*.pt')
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
    return 0, {}

# Training Loop with W&B Logging
for epoch in range(start_epoch, total_epochs):
    # Training
    train_loss = train_epoch(model, train_loader, optimizer)
    
    # Validation
    val_metrics = validate_epoch(model, val_loader)
    
    # Log to W&B
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_metrics['loss'],
        'val_accuracy': val_metrics['accuracy'],
        'val_map': val_metrics['mAP'],
        'learning_rate': optimizer.param_groups[0]['lr']
    })
    
    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        save_checkpoint(model, optimizer, epoch, val_metrics)
```

#### Success Metrics for W&B & Checkpoints
- **W&B Integration**: Real-time monitoring dashboard active
- **Checkpoint Saving**: Automatic save every 10 epochs working
- **Resume Functionality**: Training can resume from any checkpoint
- **Storage Management**: Google Drive integration working
- **Performance Tracking**: All metrics logged and visualized
- **Experiment Comparison**: Baseline vs improved models comparable

## Baseline Training Analysis & Improvement Plan

### Baseline Training Results (48 epochs completed)
**Training Configuration:**
- Learning rate: 1e-4 (cosine schedule)
- Batch size: 16
- Warmup epochs: 5
- Augmentation: Horizontal flip only (0.5 probability)
- No early stopping
- No additional regularization

**Performance Metrics:**
- **Best mAP50**: 0.9746 (epoch 44)
- **Best mAP50-95**: 0.803 (epoch 44)
- **Training Loss**: Steady decrease until epoch 45
- **Validation Loss**: Increasing trend after epoch 30 (overfitting)

### Critical Issues Identified
1. **Training Instability (Epoch 45)**
   - mAP50 dropped from 0.9746 to 0.00041
   - mAP50-95 dropped from 0.803 to 0.00033
   - Suggests learning rate too high for 1025 classes

2. **Overfitting (After Epoch 30)**
   - Validation loss increasing while training loss decreasing
   - Insufficient regularization
   - No early stopping mechanism

3. **Insufficient Augmentation**
   - Only horizontal flip used
   - Poor generalization to real-world conditions
   - Limited robustness to lighting, angle, and background variations

### Improvement Strategy
1. **Training Stability**
   - Reduce learning rate to 5e-5 (from 1e-4)
   - Add early stopping with patience=10
   - Implement validation monitoring

2. **Regularization**
   - Add dropout (0.1)
   - Increase weight decay to 0.001
   - Add label smoothing (0.1)

3. **Augmentation**
   - Add rotation (±10°)
   - Add translation (±20%)
   - Add shear (±2°)
   - Add mosaic (prob=1.0)
   - Add mixup (prob=0.1)

4. **Training Configuration**
   - Increase batch size to 32
   - Extend training to 200 epochs
   - Add proper checkpoint management

### Expected Improvements
- **Training Stability**: Prevent catastrophic performance drops
- **Generalization**: Better real-world performance
- **Robustness**: Improved handling of lighting, angle, and background variations
- **Consistency**: More stable training curves
- **Final Performance**: Higher overall accuracy and mAP scores

## Phase 1: Research & Setup (Weeks 1-2)

### Task 1.1: Google Colab Environment Setup
**Priority**: High  
**Duration**: 2-3 days  
**Dependencies**: None

- [ ] Set up Google Colab environment with conda
- [ ] Install uv for Python dependency management
- [ ] Install required packages: PyTorch, OpenCV, ultralytics (YOLO)
- [ ] Set up GPU access for training (Google Colab Pro recommended)
- [ ] Create project structure and version control
- [ ] Set up Weights & Biases project tracking
- [ ] Configure environment for YOLOv3 training (replacing Mx_yolo binary)
- [ ] Set up YOLO training pipeline in Colab (not binary)

### Task 1.2: Dataset Preparation & Analysis
**Priority**: High  
**Duration**: 3-4 days  
**Dependencies**: Task 1.1

- [x] Copy full 1025 Pokemon dataset to `data/raw/all_pokemon/`
- [x] Run `organize_raw_data.py` to analyze dataset structure and statistics
- [x] Create dataset statistics and visualization
- [x] Identify data quality issues and biases
- [x] Plan data augmentation strategy
- [x] Design train/validation/test splits (70/15/15) - **UPDATED: Within-class splitting**
- [x] **COMPLETED**: Comprehensive dataset analysis with all 128,768 images
- [x] **COMPLETED**: Quality assessment (100% validity rate)
- [x] **COMPLETED**: Within-class splitting strategy implemented
- [ ] Process raw data once for all models (shared pipeline)
- [ ] Create model-specific dataset formats (YOLO, CLIP, SMoLVM)
- [ ] Upload processed datasets to Hugging Face Hub
- [ ] Create comprehensive dataset cards with metadata
- [ ] Verify all data files are gitignored

**COMPLETED DELIVERABLES:**
- ✅ Dataset analysis report: `data/raw/analysis_report.md`
- ✅ Quality assessment: `data/raw/quality_assessment.json`
- ✅ Image distribution visualization: `data/raw/image_distribution_analysis.png`
- ✅ Bias analysis: `data/raw/bias_analysis.json`
- ✅ Augmentation plan: `data/raw/augmentation_plan.json`
- ✅ Dataset splits: `data/raw/dataset_splits.json` (within-class splitting)
- ✅ Quality visualization: `data/raw/quality_assessment.png`

**KEY FINDINGS:**
- **Total Images**: 128,768 across all 1025 Pokemon
- **Quality**: 100% validity rate (no corrupted images)
- **Distribution**: Good balance (37-284 images per Pokemon)
- **Splitting**: Within-class approach ensures all Pokemon seen in training

### Task 1.3: Literature Review
**Priority**: Medium  
**Duration**: 2-3 days  
**Dependencies**: None

- [ ] Research existing Pokemon classification projects
- [ ] Review VLM fine-tuning papers (CLIP, BLIP-2)
- [ ] Study IoT model optimization techniques
- [ ] Research multi-frame aggregation methods
- [ ] Document findings and best practices

### Task 1.4: Original Blog Reproduction Setup
**Priority**: High  
**Duration**: 3-4 days  
**Dependencies**: Tasks 1.1, 1.2

- [ ] Study original blog's approach and methodology
- [ ] Configure YOLOv3 for exact reproduction (1025 classes for all generations)
- [ ] Set up YOLO training pipeline in Colab (replacing Mx_yolo binary)
- [ ] Create evaluation pipeline for Pokemon classification
- [ ] Set up Weights & Biases tracking for experiments
- [ ] Establish baseline performance metrics
- [ ] Test original project's approach with Colab-based training
- [ ] Document original limitations and constraints

## Phase 2: Data Collection & Preparation (Weeks 2-3)

### Task 2.1: Data Processing & Validation
**Priority**: High  
**Duration**: 2-3 days  
**Dependencies**: Task 1.2

- [x] Process full 1025 Pokemon dataset with shared preprocessing pipeline
- [x] Validate image quality and format consistency
- [x] Create Pokemon name mappings (numbered directories → names)
- [x] Generate dataset statistics and quality report
- [x] Create balanced train/validation/test splits
- [x] Verify data integrity across all images
- [x] Document data processing pipeline for reproducibility

**COMPLETED DELIVERABLES:**
- ✅ Multiprocessing preprocessing pipeline (`src/data/preprocessing.py`)
- ✅ All 128,768 images processed in ~92 seconds
- ✅ Correct Pokemon names mapping (all 1025 Pokemon)
- ✅ YOLO dataset created with proper format
- ✅ Within-class splitting (70/15/15) implemented
- ✅ Processing speed: ~1400 images/second with 8 workers

### Task 2.1.5: Fix YOLO Dataset Format Issues
**Priority**: High  
**Duration**: 1-2 days  
**Dependencies**: Task 2.1

- [ ] Fix class indices (convert from 1-based to 0-based indexing)
- [ ] Add full-image bounding boxes to all label files
- [ ] Convert to YOLO detection format with proper coordinates
- [ ] Verify dataset format is correct for YOLO training
- [ ] Test with YOLO training pipeline
- [ ] Document the format conversion process

**ISSUES IDENTIFIED:**
- **Class Index Problem**: Grimmsnarl at line 365 (0-based = 364), but label shows `364` (should be `363`)
- **Missing Bounding Boxes**: Need full-image bounding box coordinates for YOLO detection
- **Label Format**: Currently classification format, need detection format
- **Required Format**: `<class_id> <x_center> <y_center> <width> <height>`

**SOLUTION APPROACH:**
- **Fix Indices**: Subtract 1 from all class IDs for 0-based indexing
- **Add Bounding Boxes**: Use `0.5 0.5 1.0 1.0` for full-image coverage
- **Example**: `93 0.5 0.5 1.0 1.0` for bulbasaur (class 93)
- **Verify**: Test with YOLO training to ensure format is correct

### Task 2.2: Model-Specific Dataset Creation
**Priority**: High  
**Duration**: 2-3 days  
**Dependencies**: Task 2.1

- [x] Create YOLO format dataset from processed images
- [ ] Create CLIP format dataset from processed images
- [ ] Create SMoLVM format dataset from processed images
- [ ] Implement model-specific data loading pipelines
- [ ] Create data quality checks and filtering
- [ ] Prepare Pokemon name mappings and labels for each format
- [ ] Upload datasets to Hugging Face Hub

**CURRENT STATUS:**
- 🔄 YOLO dataset created but needs format fixes (class indices + bounding boxes)
- 🔄 CLIP dataset creation pending (text prompts + images)
- 🔄 SMoLVM dataset creation pending (text prompts + images)
- 🔄 Hugging Face upload pending

### Task 2.3: Dataset Validation
**Priority**: Medium  
**Duration**: 2-3 days  
**Dependencies**: Task 2.2

- [ ] Validate data quality across all sources
- [ ] Check for class imbalance and bias
- [ ] Create balanced train/validation/test splits
- [ ] Document dataset statistics
- [ ] Plan data collection strategy for missing Pokemon

## Phase 3: Model Development (Weeks 3-6)

### Task 3.1: Original Blog Reproduction Training
**Priority**: High  
**Duration**: 5-7 days  
**Dependencies**: Tasks 1.4, 2.2

- [ ] Set up YOLOv3 training environment in Colab
- [ ] Configure YOLOv3 for exact reproduction (1025 classes, original parameters)
- [ ] Implement training pipeline with W&B tracking (replacing Mx_yolo binary)
- [ ] Set up data loading for Pokemon classification
- [ ] Use original training parameters (no scheduling, minimal augmentation)
- [ ] Implement basic checkpointing (no early stopping)
- [ ] Set up experiment tracking in W&B
- [ ] Create training dashboard in W&B
- [ ] Train with exact original blog parameters (`reproduction_config.yaml`)
- [ ] Document original performance baseline and limitations

### Task 3.2: Original Blog Evaluation & Improvement Planning
**Priority**: High  
**Duration**: 3-5 days  
**Dependencies**: Task 3.1

- [ ] Evaluate reproduced YOLOv3 model performance on test set
- [ ] Test model on real Pokemon images (cards, toys, figurines)
- [ ] Evaluate performance under different lighting conditions
- [ ] Test model robustness and accuracy
- [ ] Create comprehensive evaluation metrics
- [ ] Compare results with original blog performance
- [ ] Document original limitations and improvement opportunities
- [ ] Plan improvement strategies based on identified limitations

### Task 3.3: YOLOv3 Improvement Implementation
**Priority**: High  
**Duration**: 7-10 days  
**Dependencies**: Task 3.2

- [ ] Implement enhanced YOLOv3 training with improved parameters
- [ ] Add advanced data augmentation (rotation, shear, mosaic, mixup)
- [ ] Implement cosine learning rate scheduling with warmup
- [ ] Add early stopping and regularization techniques
- [ ] Optimize for IoT deployment (INT8 quantization)
- [ ] Train improved model with `improvement_config.yaml`
- [ ] Compare improved vs reproduced model performance
- [ ] Create detailed improvement report for original author
- [ ] Prepare improved model for Sipeed Maix Bit testing

### Task 3.4: Advanced Model Development & Comparison
**Priority**: High  
**Duration**: 7-10 days  
**Dependencies**: Tasks 3.1, 3.2, 3.3

- [x] Full 1025 Pokemon dataset available (generations 1-9)
- [ ] Train improved YOLOv3 on full 1025 Pokemon dataset
- [ ] Test performance on extended dataset
- [ ] Optimize model for larger class count
- [ ] Document performance differences between 386 vs 1025 classes
- [ ] Implement VLM approaches (CLIP, SMoLVM) for comparison
- [ ] Test newer YOLO variants (v8, v9, v10) vs improved YOLOv3
- [ ] Implement multi-frame aggregation for improved accuracy
- [ ] Analyze performance metrics and trade-offs across all approaches
- [ ] Evaluate real-world robustness for each approach
- [ ] Create comprehensive comparison report in W&B
- [ ] Select best performing approach for final deployment
- [ ] Document model selection rationale and trade-offs
- [ ] Plan final optimization for IoT deployment

## Phase 4: Model Optimization (Weeks 6-8)

### Task 4.1: Model Compression
**Priority**: High  
**Duration**: 5-7 days  
**Dependencies**: Task 3.4

- [ ] Implement model pruning
- [ ] Apply knowledge distillation
- [ ] Implement quantization (INT8)
- [ ] Optimize model architecture
- [ ] Validate compressed model performance

### Task 4.2: Hardware Optimization
**Priority**: High  
**Duration**: 5-7 days  
**Dependencies**: Task 4.1

- [ ] Profile model on target hardware
- [ ] Optimize for memory constraints
- [ ] Implement power-efficient inference
- [ ] Optimize for real-time performance
- [ ] Test on actual Sipeed Maix Bit

### Task 4.3: Real-world Testing
**Priority**: High  
**Duration**: 3-4 days  
**Dependencies**: Task 4.2

- [ ] Test with real camera input
- [ ] Evaluate performance in varying lighting
- [ ] Test with different Pokemon objects
- [ ] Measure latency and accuracy
- [ ] Document real-world performance

## Phase 5: Deployment & Testing (Weeks 8-10)

### Task 5.1: IoT Deployment Pipeline
**Priority**: High  
**Duration**: 5-7 days  
**Dependencies**: Task 4.2

- [ ] Create deployment package
- [ ] Implement OTA update mechanism
- [ ] Add error handling and logging
- [ ] Optimize for battery life
- [ ] Create deployment documentation

### Task 5.2: Comprehensive Testing
**Priority**: High  
**Duration**: 4-5 days  
**Dependencies**: Task 5.1

- [ ] Test across different scenarios
- [ ] Evaluate edge cases and failures
- [ ] Measure system performance
- [ ] Test multi-frame aggregation
- [ ] Validate confidence calibration

### Task 5.3: Performance Optimization
**Priority**: Medium  
**Duration**: 3-4 days  
**Dependencies**: Task 5.2

- [ ] Optimize inference pipeline
- [ ] Fine-tune hyperparameters
- [ ] Implement caching strategies
- [ ] Optimize memory usage
- [ ] Final performance validation

## Phase 6: Documentation & Maintenance (Week 10)

### Task 6.1: Documentation
**Priority**: Medium  
**Duration**: 2-3 days  
**Dependencies**: All previous tasks

- [ ] Write technical documentation
- [ ] Create user guide
- [ ] Document model architecture
- [ ] Create maintenance procedures
- [ ] Write deployment guide

### Task 6.2: Future Improvements
**Priority**: Low  
**Duration**: 1-2 days  
**Dependencies**: Task 6.1

- [ ] Plan model updates
- [ ] Design data collection pipeline
- [ ] Plan feature enhancements
- [ ] Document research directions
- [ ] Create maintenance schedule

## Detailed Task Descriptions

### Task 1.2: Dataset Analysis
**Specific Subtasks:**
- [ ] Download Kaggle dataset and extract images
- [ ] Create dataset statistics (class distribution, image sizes, etc.)
- [ ] Visualize sample images from each class
- [ ] Identify data quality issues (blurry images, wrong labels, etc.)
- [ ] Analyze class imbalance and plan mitigation strategies
- [ ] Design appropriate train/validation/test splits (70/15/15)

**Deliverables:**
- Dataset analysis report
- Data quality assessment
- Split strategy documentation

### Task 3.1: VLM Fine-tuning Implementation
**Specific Subtasks:**
- [ ] Set up CLIP model with Pokemon text prompts and W&B tracking
- [ ] Set up SMoLVM model with Pokemon text prompts and W&B tracking
- [ ] Set up MobileVLM model with Pokemon text prompts and W&B tracking
- [ ] Implement contrastive learning loss function for each VLM
- [ ] Add data augmentation pipeline optimized for each model
- [ ] Implement learning rate scheduling and optimization
- [ ] Add model checkpointing and early stopping
- [ ] Create evaluation metrics (top-1, top-5 accuracy)
- [ ] Set up W&B sweeps for hyperparameter optimization
- [ ] Create model comparison dashboard

**Deliverables:**
- Fine-tuned CLIP, SMoLVM, and MobileVLM models
- Training pipeline code with W&B integration
- Performance metrics and comparison reports
- W&B project with all experiments tracked

### Task 4.1: Model Compression
**Specific Subtasks:**
- [ ] Implement structured pruning (remove less important layers)
- [ ] Apply quantization-aware training
- [ ] Implement INT8 quantization
- [ ] Test compressed model accuracy
- [ ] Optimize for target hardware constraints

**Deliverables:**
- Compressed model (<50MB)
- Performance comparison report
- Hardware compatibility validation

## Risk Mitigation Tasks

### High-Risk Tasks
1. **Task 3.1**: VLM fine-tuning complexity
   - Mitigation: Start with simpler approaches, iterate
   
2. **Task 4.2**: Hardware optimization challenges
   - Mitigation: Early hardware testing, fallback strategies
   
3. **Task 5.1**: IoT deployment complexity
   - Mitigation: Use proven deployment frameworks

### Contingency Plans
- If VLM approach fails: Fall back to YOLO classification
- If hardware constraints too strict: Use cloud-based inference
- If accuracy too low: Implement ensemble methods

## Resource Requirements

### Hardware
- **Training**: GPU with 8GB+ VRAM (recommended)
- **Testing**: Sipeed Maix Bit RISC-V board
- **Development**: Standard laptop/desktop

### Software
- **Python 3.8+**
- **Conda for environment management**
- **uv for Python dependency management**
- **PyTorch 1.12+**
- **OpenCV 4.5+**
- **Transformers library**
- **Ultralytics (YOLO)**
- **Weights & Biases for experiment tracking**
- **Hugging Face datasets and hub**
- **Google Colab for development and training**

### Data
- **Kaggle Pokemon dataset**: 11,945 images
- **Additional sources**: TCG cards, 3D renders, real photos
- **Target**: 50+ images per Pokemon (51,250+ total)

## Success Criteria

### Technical Metrics
- **Accuracy**: >80% top-1 accuracy on test set
- **Latency**: <500ms inference time
- **Memory**: <50MB model size
- **Power**: <2W power consumption

### Project Metrics
- **Timeline**: Complete within 10 weeks
- **Documentation**: Comprehensive technical docs
- **Deployment**: Working IoT deployment
- **Maintainability**: Clear update procedures

## Weekly Milestones

### Week 1-2: Foundation
- [ ] Environment setup complete
- [ ] Dataset analysis finished
- [ ] Baseline models implemented

### Week 3-4: Data & Models
- [ ] Data collection and preprocessing complete
- [ ] VLM fine-tuning pipeline working
- [ ] Initial model training started

### Week 5-6: Development
- [ ] Model comparison complete
- [ ] Best approach selected
- [ ] Multi-frame aggregation implemented

### Week 7-8: Optimization
- [ ] Model compression complete
- [ ] Hardware optimization finished
- [ ] Real-world testing started

### Week 9-10: Deployment
- [ ] IoT deployment complete
- [ ] Comprehensive testing finished
- [ ] Documentation complete

## Reproduction Guide for Original Project

### Original Project Analysis
Based on the [blog post](https://www.cnblogs.com/xianmasamasa/p/18995912), the original project:
- Used **Mx_yolo 3.0.0 binary** for training (we'll replace with Colab-based training)
- Achieved working results with **1-3 generations (386 Pokemon)**
- Used **YOLOv3** for classification (not detection)
- Deployed on **Sipeed Maix Bit** with RISC-V processor
- Had **limited recognition accuracy** due to lighting, angle, and background factors

### Key Insights from Original Project
1. **Dataset Quality**: Original author noted dataset was "garbage" with poor quality images
2. **Recognition Limitations**: "识别受光线、目标大小因素、背景等因素影响很大"
3. **Model Performance**: "识别的局限性太大了，很多时候识别不出来"
4. **Hardware Constraints**: Used Sipeed Maix Bit with limited resources

### Google Colab Setup Commands
```python
# Install conda in Google Colab
!wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!bash Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local
!conda install -c conda-forge uv -y

# Create conda environment
!conda create -n pokemon-classifier python=3.9 -y
!conda activate pokemon-classifier

# Install dependencies for YOLO training
!uv add torch torchvision torchaudio
!uv add ultralytics opencv-python
!uv add wandb pillow matplotlib seaborn
!uv add accelerate bitsandbytes

# Install YOLO training dependencies (replacing Mx_yolo binary)
# We'll use ultralytics or custom YOLO implementation in Colab
```

### Data Preparation Steps
```python
# 1. Download original dataset (1-3 generations)
# Link: https://pan.baidu.com/s/1K8QMzebb1QpxGWurGE4jyg?pwd=n4zw

# 2. Download Kaggle dataset (1st generation)
# https://www.kaggle.com/datasets/unexpectedscepticism/11945-pokemon-from-first-gen

# 3. Web scraping for additional data
# Sites: 52poke.com, pokemondb.net, pokemon.fandom.com

# 4. Convert to YOLO detection format with full-image boxes
def prepare_yolo_dataset():
    # Convert images to YOLO detection format
    # Use full-image bounding boxes (0.5 0.5 1.0 1.0)
    # Class IDs are 0-based (0-1024)
    # Example: "0 0.5 0.5 1.0 1.0" for Bulbasaur
    pass
```

### Training Setup (Colab-based)
```python
# Configure YOLOv3 for Pokemon classification in Colab
# Original project used 386 classes (1-3 generations)
# Target: 1025 classes (all generations)

# Training configuration for Colab
config = {
    "model": "yolov3",
    "classes": 386,  # Start with 1-3 generations (baseline)
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.001,
    "img_size": 416,
    "training_platform": "google_colab",  # Instead of Mx_yolo binary
    "framework": "ultralytics"  # Or custom PyTorch implementation
}

# Workflow: Baseline → Extension → Advanced Approaches
# 1. Reproduce baseline (386 classes) in Colab
# 2. Extend to full dataset (1025 classes)
# 3. Implement VLM and newer YOLO variants
# 4. Select best approach for IoT deployment
```

### Evaluation Process
```python
# Test on real-world scenarios
# - Pokemon cards
# - Toys and figurines  
# - Different lighting conditions
# - Various angles and backgrounds

# Expected limitations (from original project):
# - Poor performance in low light
# - Difficulty with similar Pokemon
# - Background interference
# - Size and angle sensitivity
```

### Weights & Biases Setup
```python
# Login to W&B
import wandb
wandb.login()

# Initialize project
wandb.init(
    project="pokemon-classifier",
    name="baseline-comparison",
    config={
        "model_type": "clip",
        "dataset": "pokemon-1025",
        "learning_rate": 1e-5,
        "batch_size": 32
    }
)
```

### Hugging Face Dataset Upload
```python
# Upload dataset to Hugging Face
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

# Create dataset
dataset = Dataset.from_dict({
    "image": image_paths,
    "label": labels,
    "pokemon_name": pokemon_names
})

# Upload to Hub
dataset.push_to_hub("your-username/pokemon-dataset")
```

## Dependencies Graph

```
Task 1.1 → Task 1.2 → Task 2.1 → Task 2.2 → Task 3.1 → Task 3.4 → Task 4.1 → Task 4.2 → Task 5.1 → Task 5.2
    ↓         ↓         ↓         ↓         ↓         ↓         ↓         ↓         ↓         ↓
Task 1.3 → Task 1.4 → Task 2.3 → Task 3.2 → Task 3.3 → Task 4.3 → Task 5.3 → Task 6.1 → Task 6.2
```

This task breakdown provides a clear roadmap for implementing the Pokemon classifier project with realistic timelines and dependencies. 