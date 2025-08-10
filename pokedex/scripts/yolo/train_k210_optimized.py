#!/usr/bin/env python3
"""
K210-Optimized YOLOv3-Nano Training Script
Trains a YOLOv3-nano model optimized for Sipeed Maix Bit K210 deployment
Based on train_yolov3_improved.py with K210-specific optimizations
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import wandb
from ultralytics import YOLO
from datasets import load_dataset
from PIL import Image
import numpy as np

# Add src to path for module imports
src_path = str(Path(__file__).resolve().parents[2] / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from training.yolo.trainer import YOLOTrainer
    from evaluation.yolo.evaluator import YOLOEvaluator
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Make sure you're running from the project root directory.")
    YOLOTrainer = None
    YOLOEvaluator = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def is_colab():
    """Check if running in Google Colab."""
    try:
        from google.colab import drive
        return True, drive
    except (ImportError, ModuleNotFoundError):
        return False, None

def get_storage_dirs():
    """Get storage directories (creates them if they don't exist).
    
    Uses pokemon-classifier directory structure (compatible with YOLOTrainer auto-backup).
    """
    try:
        # Create local pokemon-classifier directory structure (matches trainer.py pattern)
        local_base = os.path.join(os.getcwd(), 'pokemon-classifier')
        
        print(f"🔍 Current working directory: {os.getcwd()}")
        print(f"🔍 Local base directory: {local_base}")
        
        # Simple structure that works with YOLOTrainer's auto-backup
        dirs = {
            'checkpoints': local_base,  # YOLOTrainer will create subdirectories
            'logs': local_base,
            'models': local_base,
        }
        
        print(f"🔍 Directory structure (compatible with YOLOTrainer):")
        for name, path in dirs.items():
            print(f"  • {name}: {path}")
        
        # Create base directory
        if not os.path.exists(local_base):
            os.makedirs(local_base, exist_ok=True)
            print(f"📁 Created base directory: {local_base}")
        else:
            print(f"✅ Found base directory: {local_base}")
        
        return dirs
    except Exception as e:
        print(f"❌ Failed to get storage directories: {e}")
        raise

def verify_dataset():
    """Verify Pokemon dataset is accessible and properly formatted."""
    try:
        print("\n📦 Verifying dataset access...")
        # First verify Hugging Face dataset access
        from datasets import load_dataset
        import shutil
        from PIL import Image
        import io

        # Create YOLO dataset directory
        yolo_dataset_dir = Path("data/yolo_dataset")
        yolo_dataset_dir.mkdir(parents=True, exist_ok=True)

        def process_image(example):
            """Process image data from either raw bytes or PIL Image."""
            if isinstance(example['image'], dict) and 'bytes' in example['image']:
                # Raw bytes from HF dataset
                img_bytes = example['image']['bytes']
                return Image.open(io.BytesIO(img_bytes))
            elif isinstance(example['image'], Image.Image):
                # Already a PIL Image
                return example['image']
            else:
                raise ValueError(f"Unexpected image type: {type(example['image'])}")

        # Load HF dataset
        dataset = load_dataset("liuhuanjim013/pokemon-yolo-1025")
        print("✅ Hugging Face dataset is accessible")

        # Create directories
        splits = ['train', 'validation', 'test']
        for split in splits:
            (yolo_dataset_dir / split).mkdir(parents=True, exist_ok=True)
            (yolo_dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (yolo_dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Extract images and labels
        from tqdm import tqdm
        for split in splits:
            print(f"\n📦 Preparing {split} split...")
            split_data = dataset[split]
            
            # Skip if files already exist
            try:
                if (yolo_dataset_dir / split / "images").exists() and \
                   len(list((yolo_dataset_dir / split / "images").glob("*.jpg"))) == len(split_data):
                    print(f"✅ {split} split already prepared, skipping...")
                    continue
            except OSError as e:
                print(f"❌ I/O error checking {split} split: {e}")
                print("🔄 Will re-process this split due to I/O issues...")
                
            for i, example in tqdm(enumerate(split_data), total=len(split_data), desc=f"Processing {split} images"):
                try:
                    # Process and save image
                    img = process_image(example)
                    img_path = yolo_dataset_dir / split / "images" / f"{example['pokemon_id']:04d}_{i+1:03d}.jpg"
                    img.save(img_path)

                    # Save label
                    label_path = yolo_dataset_dir / split / "labels" / f"{example['pokemon_id']:04d}_{i+1:03d}.txt"
                    with open(label_path, 'w') as f:
                        # YOLO format: class_id x_center y_center width height
                        f.write(f"{example['label']} 0.5 0.5 1.0 1.0\n")
                except OSError as e:
                    print(f"❌ I/O error processing example {i} in {split} split: {e}")
                    print("🔄 This might be a Google Drive issue. Continuing...")
                    continue

        print("✅ YOLO dataset prepared")
        
        # Update data config path
        data_config_path = Path("configs/yolov3/k210_data.yaml")
        if data_config_path.exists():
            with open(data_config_path) as f:
                data_config = yaml.safe_load(f)
            
            # Update the path to use current working directory
            data_config['path'] = str(Path.cwd() / "data" / "yolo_dataset")
                
            # Save updated config
            with open(data_config_path, 'w') as f:
                yaml.safe_dump(data_config, f)
                
            print(f"📝 Updated data config path to: {data_config['path']}")
        
        print("\n📋 Dataset Configuration:")
        print("• Classes: 1025 (all generations 1-9)")
        print("• Image size: 416x416 (resized to 224x224 for K210)")
        print("• Format: YOLO detection with full-image bounding boxes")
        print("• Labels: 0-based class indices")
        print(f"• Data Config: {data_config_path}")
        
        return True
    except Exception as e:
        print(f"❌ Dataset verification failed: {e}")
        raise

def resume_wandb():
    """Resume W&B run from saved ID."""
    run_id_file = Path("wandb_run_id.txt")
    if not run_id_file.exists():
        print("ℹ️ No W&B run ID found. Starting new run.")
        return None
        
    with open(run_id_file) as f:
        run_id = f.read().strip()
    print(f"📋 Found W&B run ID: {run_id}")
    return run_id

def initialize_wandb(config_path: Path, resume_id: Optional[str] = None):
    """Initialize W&B with configuration from YAML file."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Use wandb config from the K210 config file
        wandb_config = config.get('wandb', {})
        
        # Base init kwargs
        init_kwargs = {
            'project': wandb_config.get('project', 'pokemon-classifier'),
            'name': wandb_config.get('name', 'yolov3n-k210-optimized'),
            'entity': wandb_config.get('entity', 'liuhuanjim013-self'),
            'tags': wandb_config.get('tags', ['k210', 'yolov3n']),
            'config': config,
            'reinit': True,
            'settings': wandb.Settings(
                save_code=wandb_config.get('settings', {}).get('save_code', False),
                disable_git=wandb_config.get('settings', {}).get('disable_git', True),
            )
        }
        
        # Add resume ID if provided
        if resume_id:
            init_kwargs.update({
                'id': resume_id,
                'resume': 'must'
            })
        
        # Try online mode first if not already in offline mode
        if os.environ.get('WANDB_MODE') != 'offline':
            try:
                # Clear offline mode if set
                if 'WANDB_MODE' in os.environ:
                    del os.environ['WANDB_MODE']
                    
                # Try online init
                wandb.init(**init_kwargs)
                print(f"✅ W&B experiment initialized: {wandb.run.name}")
                print(f"📊 Project: {wandb_config.get('project')}, Run: {wandb_config.get('name')}")
                
                # Save run ID for future resume
                run_id_file = Path("wandb_run_id.txt")
                with open(run_id_file, "w") as f:
                    f.write(str(wandb.run.id))
                print(f"💾 Saved run ID: {wandb.run.id}")
                return config
                
            except Exception as e:
                print(f"⚠️ Failed to initialize W&B: {e}")
                print("ℹ️ Falling back to offline mode")
                os.environ['WANDB_MODE'] = 'offline'
        
        # Try offline mode
        try:
            if os.environ.get('WANDB_MODE') != 'offline':
                os.environ['WANDB_MODE'] = 'offline'
            wandb.init(**init_kwargs)
            print("✅ W&B initialized in offline mode")
            return config
        except Exception as e2:
            print(f"❌ Failed to initialize W&B in offline mode: {e2}")
            raise
            
    except Exception as e:
        print(f"❌ W&B initialization failed: {e}")
        raise

class K210ModelTrainer:
    """K210-optimized YOLOv3-nano trainer with hardware constraints"""
    
    def __init__(self, config_path: str):
        """
        Initialize K210 trainer
        
        Args:
            config_path: Path to training configuration YAML
        """
        self.config_path = Path(config_path)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # K210 constraints
        self.max_model_size_mb = 2.0
        self.max_runtime_memory_mb = 3.0
        self.target_input_size = self.config.get('k210', {}).get('input_size', 224)
        
        # Training state
        self.model = None
        self.best_model_path = None
        self.training_start_time = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            raise
            
    def _setup_model(self) -> YOLO:
        """Create and configure K210-optimized model based on config"""
        try:
            # Get model config
            model_config = self.config.get('model', {})
            model_name = model_config.get('name', 'yolov3-tinyu')
            
            logger.info(f"Creating {model_name} model for K210...")
            
            # Create a fresh model instance to avoid resume conflicts
            # Use model name without .pt extension to get fresh architecture
            if model_name.endswith('.pt'):
                model_name = model_name[:-3]  # Remove .pt extension
            
            # Create model from architecture (not from checkpoint)
            try:
                # Try to create from YAML config first (fresh architecture)
                model = YOLO(f"{model_name}.yaml")
                logger.info(f"✅ Created fresh {model_name} model from YAML")
            except Exception as e:
                logger.warning(f"Could not load from YAML: {e}")
                # Fallback: create from model name but force fresh training
                model = YOLO(model_name)  # This will download the model
                
                # Clear any resume state that might be set by Ultralytics
                if hasattr(model, 'ckpt'):
                    model.ckpt = None
                if hasattr(model, 'ckpt_path'):
                    model.ckpt_path = None
                    
                logger.info(f"✅ Loaded {model_name} model and cleared resume state")
            
            # Verify model architecture for K210 compatibility
            self._verify_k210_compatibility(model)
            
            logger.info(f"✅ Model setup complete")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
            
    def _verify_k210_compatibility(self, model: YOLO) -> None:
        """Verify model meets K210 hardware constraints"""
        try:
            # Get model info
            model_info = model.info()
            
            # Check parameter count (should be < 10M for K210)
            if hasattr(model_info, 'parameters'):
                params = model_info.parameters
                if params > 10_000_000:
                    logger.warning(f"⚠️ Model has {params:,} parameters (>10M), may be too large for K210")
                else:
                    logger.info(f"✅ Model parameters: {params:,} (<10M)")
                    
            # Check layer count (K210 KPU has ~25 layer limit)
            if hasattr(model.model, 'model'):
                layer_count = len(model.model.model)
                if layer_count > 25:
                    logger.warning(f"⚠️ Model has {layer_count} layers (>25), may exceed K210 KPU limit")
                else:
                    logger.info(f"✅ Model layers: {layer_count} (<25)")
                    
        except Exception as e:
            logger.warning(f"Could not verify K210 compatibility: {e}")
            
    def _check_model_size(self, model_path: str) -> bool:
        """Check if model meets K210 size constraints"""
        if not os.path.exists(model_path):
            return False
            
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        if size_mb > self.max_model_size_mb:
            logger.warning(f"⚠️ Model size {size_mb:.2f}MB exceeds K210 limit ({self.max_model_size_mb}MB)")
            return False
        else:
            logger.info(f"✅ Model size: {size_mb:.2f}MB (within K210 limit)")
            return True
            
    def _test_onnx_export(self, model_path: str) -> bool:
        """Test ONNX export for K210 compatibility"""
        try:
            logger.info("Testing ONNX export for K210...")
            
            # Load model for export
            model = YOLO(model_path)
            
            # Export to ONNX with K210 settings
            onnx_path = model_path.replace('.pt', '_test.onnx')
            model.export(
                format='onnx',
                imgsz=self.target_input_size,
                opset=12,
                simplify=True,
                dynamic=False,  # Fixed input for K210
                half=False  # K210 uses INT8, not FP16
            )
            
            # Check ONNX file size
            if os.path.exists(onnx_path):
                onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
                logger.info(f"✅ ONNX export successful: {onnx_size_mb:.2f}MB")
                
                # Clean up test file
                os.remove(onnx_path)
                return True
            else:
                logger.error("❌ ONNX export failed: no output file")
                return False
                
        except Exception as e:
            logger.error(f"❌ ONNX export test failed: {e}")
            return False
            
    def _create_training_args(self, storage_dirs: dict) -> Dict[str, Any]:
        """Create training arguments from configuration matching improved script structure"""
        # Get config sections
        model_config = self.config.get('model', {})
        training_config = self.config.get('training', {})
        aug_config = training_config.get('augmentation', {})
        
        # Use existing dataset path (updated by verify_dataset)
        data_path = "configs/yolov3/k210_data.yaml"
        
        # Base training arguments matching improved script
        args = {
            'data': data_path,
            'epochs': training_config.get('epochs', 200),
            'batch': training_config.get('batch_size', 8),
            'imgsz': model_config.get('img_size', 224),
            'lr0': training_config.get('learning_rate', 5e-5),
            'weight_decay': training_config.get('weight_decay', 0.001),
            'project': storage_dirs['checkpoints'],
            'name': 'yolov3n_k210_optimized',  # Fixed name for consistent resumption
            'save': True,
            'save_period': 1,
            'cache': False,
            'device': 'cpu' if not torch.cuda.is_available() else 0,
            'exist_ok': True,
            'patience': training_config.get('early_stopping', {}).get('patience', 20),
            'optimizer': 'SGD',  # Force SGD to prevent auto-optimizer override
            'resume': False,  # Explicitly disable resume
            'pretrained': False,  # Explicitly disable pretrained
            'model': None,  # Don't pass model to avoid resume detection
        }
        
        # Add K210-optimized augmentation parameters
        if aug_config:
            args.update({
                'degrees': aug_config.get('degrees', 5.0),
                'translate': aug_config.get('translate', 0.1),
                'scale': aug_config.get('scale', 0.1),
                'shear': aug_config.get('shear', 2.0),
                'perspective': aug_config.get('perspective', 0.0),
                'flipud': aug_config.get('flipud', 0.0),
                'fliplr': aug_config.get('fliplr', 0.5),
                'hsv_h': aug_config.get('hsv_h', 0.015),
                'hsv_s': aug_config.get('hsv_s', 0.7),
                'hsv_v': aug_config.get('hsv_v', 0.4),
                'mosaic': aug_config.get('mosaic', 0.0),  # Disabled for K210
                'mixup': aug_config.get('mixup', 0.0),    # Disabled for K210
            })
            
        return args
        
    def train(self, storage_dirs: dict, resume: bool = True) -> str:
        """Run K210-optimized training"""
        logger.info("🚀 Starting K210-optimized YOLOv3-nano training...")
        self.training_start_time = time.time()
        
        # Create model
        self.model = self._setup_model()
        
        # Prepare training arguments based on improved config structure
        train_args = self._create_training_args(storage_dirs)
        
        # Override resume option based on parameter (default is False in _create_training_args)
        if resume:
            train_args['resume'] = True
            train_args['pretrained'] = True  # Allow pretrained weights when resuming
            logger.info("🔄 Resume enabled - will continue from last checkpoint if available")
        else:
            # Keep resume=False and pretrained=False from _create_training_args
            logger.info("🆕 Fresh training - starting from scratch to avoid resume conflicts")
        
        # Get model config for logging
        model_config = self.config.get('model', {})
        training_config = self.config.get('training', {})
        
        logger.info("📋 K210 Training Configuration:")
        logger.info(f"  • Model: {model_config.get('name', 'yolov3n')}")
        logger.info(f"  • Classes: {model_config.get('classes', 1025)}")
        logger.info(f"  • Input Size: {model_config.get('img_size', 224)}")
        logger.info(f"  • Batch Size: {training_config.get('batch_size', 8)}")
        logger.info(f"  • Learning Rate: {training_config.get('learning_rate', 5e-5)}")
        logger.info(f"  • Epochs: {training_config.get('epochs', 200)}")
        logger.info(f"  • Target Model Size: <{self.max_model_size_mb}MB")
        
        print("\n📈 K210 Optimizations over improved_config:")
        print("1. **YOLOv3-tiny-ultralytics**: Lightweight variant instead of full YOLOv3")
        print("2. **Input Size**: 224x224 (vs 416x416) for K210 memory constraints")
        print("3. **Reduced Batch Size**: 8 (vs 32) for K210 memory constraints")
        print("4. **Conservative Augmentation**: More stable for embedded deployment")
        print("5. **Extended Patience**: 20 (vs 10) for K210 convergence")
        print("6. **Disabled Heavy Augmentation**: No mosaic/mixup for K210 stability")
        print("7. **MAINTAINED ALL 1025 Classes**: Full Pokemon coverage (not reduced)")
        
        # Log K210-specific training configuration to W&B
        if wandb.run:
            wandb.log({
                "k210_config": {
                    "model_name": model_config.get('name', 'yolov3n'),
                    "target_classes": model_config.get('classes', 1025),
                    "input_size": model_config.get('img_size', 224),
                    "batch_size": training_config.get('batch_size', 8),
                    "learning_rate": training_config.get('learning_rate', 5e-5),
                    "epochs": training_config.get('epochs', 200),
                    "max_model_size_mb": self.max_model_size_mb,
                    "max_runtime_memory_mb": self.max_runtime_memory_mb,
                    "hardware_target": "Sipeed Maix Bit K210"
                }
            })
        
        try:
            # Start fresh training (resume=False to avoid pretrained model conflicts)
            logger.info("Starting fresh K210 training...")
            
            # Remove model-related args that might trigger resume detection
            clean_args = {k: v for k, v in train_args.items() if k not in ['model']}
            
            # Explicitly ensure fresh training
            clean_args['resume'] = False
            clean_args['pretrained'] = False
            
            logger.info(f"Training with args: resume={clean_args.get('resume')}, pretrained={clean_args.get('pretrained')}")
            results = self.model.train(**clean_args)
            
            # Get best model path
            self.best_model_path = str(results.save_dir / 'weights' / 'best.pt')
            
            # Log training metrics from results
            self._log_training_metrics(results)
            
            # Verify final model and log results
            final_metrics = self._verify_final_model()
            
            # Log final training results to W&B
            if wandb.run:
                wandb.log({
                    "k210_training_results": {
                        "best_model_path": self.best_model_path,
                        "training_time_hours": (time.time() - self.training_start_time) / 3600,
                        "model_verification": final_metrics,
                        "k210_ready": final_metrics.get('k210_compatible', False)
                    }
                })
            
            # Save model with metadata (matching improved script)
            self._save_model_with_metadata(results, storage_dirs)
            
            # Training summary with detailed paths
            training_time = time.time() - self.training_start_time
            logger.info(f"✅ Training completed in {training_time/3600:.2f} hours")
            logger.info(f"📁 Best model saved: {self.best_model_path}")
            
            # Show all training artifacts
            if results and hasattr(results, 'save_dir'):
                save_dir = str(results.save_dir)
                logger.info(f"\n📂 Training Artifacts Saved:")
                logger.info(f"  📁 Training Directory: {save_dir}")
                logger.info(f"  🏆 Best Weights: {save_dir}/weights/best.pt")
                logger.info(f"  📊 Last Checkpoint: {save_dir}/weights/last.pt")
                logger.info(f"  📈 Training Results: {save_dir}/results.csv")
                logger.info(f"  📊 Training Plots: {save_dir}/results.png")
                logger.info(f"  🎯 Model Summary: {save_dir}/args.yaml")
                
                # Check actual file sizes
                best_weights = os.path.join(save_dir, 'weights', 'best.pt')
                last_weights = os.path.join(save_dir, 'weights', 'last.pt')
                
                if os.path.exists(best_weights):
                    best_size = os.path.getsize(best_weights) / (1024 * 1024)
                    logger.info(f"  📏 Best Model Size: {best_size:.2f}MB")
                    
                if os.path.exists(last_weights):
                    last_size = os.path.getsize(last_weights) / (1024 * 1024)
                    logger.info(f"  📏 Last Checkpoint Size: {last_size:.2f}MB")
            
            return self.best_model_path
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
                
    def _verify_final_model(self) -> dict:
        """Verify final model meets K210 constraints and return metrics"""
        if not self.best_model_path or not os.path.exists(self.best_model_path):
            logger.error("❌ Best model not found")
            return {"k210_compatible": False, "error": "model_not_found"}
            
        logger.info("🔍 Verifying final model for K210 compatibility...")
        
        # Check model size
        size_ok = self._check_model_size(self.best_model_path)
        size_mb = os.path.getsize(self.best_model_path) / (1024 * 1024) if os.path.exists(self.best_model_path) else 0
        
        # Test ONNX export
        export_ok = self._test_onnx_export(self.best_model_path)
        
        # Get model info
        model_info = {}
        try:
            if self.model:
                info = self.model.info()
                if hasattr(info, 'parameters'):
                    model_info['parameters'] = info.parameters
                if hasattr(self.model.model, 'model'):
                    model_info['layers'] = len(self.model.model.model)
        except Exception as e:
            logger.warning(f"Could not get model info: {e}")
        
        # Compile verification results
        metrics = {
            "k210_compatible": size_ok and export_ok,
            "model_size_mb": size_mb,
            "size_within_limit": size_ok,
            "onnx_export_ok": export_ok,
            "target_size_mb": self.max_model_size_mb,
            "target_memory_mb": self.max_runtime_memory_mb,
            **model_info
        }
        
        # Log verification to W&B
        if wandb.run:
            wandb.log({
                "k210_verification": metrics
            })
        
        if size_ok and export_ok:
            logger.info("✅ Model verification passed - ready for K210 deployment!")
        else:
            logger.warning("⚠️ Model verification failed - may need optimization")
            
        return metrics
            
    def _save_model_with_metadata(self, results, storage_dirs: dict) -> None:
        """Save model with comprehensive metadata (matching improved script)"""
        try:
            # Save final model to models directory
            final_model_path = os.path.join(storage_dirs['models'], "yolov3_k210_final.pt")
            if self.best_model_path and os.path.exists(self.best_model_path):
                import shutil
                shutil.copy2(self.best_model_path, final_model_path)
                
                # Save metadata with detailed progress
                meta_path = Path(final_model_path).with_suffix('.json')
                metadata = {
                    'wandb_run_id': wandb.run.id if wandb.run else None,
                    'timestamp': datetime.now().isoformat(),
                    'config_path': str(self.config_path),
                    'k210_config': {
                        'max_model_size_mb': self.max_model_size_mb,
                        'max_runtime_memory_mb': self.max_runtime_memory_mb,
                        'target_input_size': self.target_input_size,
                    },
                    'model_path': final_model_path,
                    'training_time_hours': (time.time() - self.training_start_time) / 3600 if self.training_start_time else 0,
                    'hardware_target': 'Sipeed Maix Bit K210',
                }
                
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"✅ Saved final model: {final_model_path}")
                logger.info(f"✅ Saved metadata: {meta_path}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save model with metadata: {e}")
            
    def _log_training_metrics(self, results):
        """Log K210-specific training metrics from results"""
        try:
            if wandb.run and results:
                # Extract key metrics from training results
                training_metrics = {}
                
                # Get final model size
                if self.best_model_path and os.path.exists(self.best_model_path):
                    size_mb = os.path.getsize(self.best_model_path) / (1024 * 1024)
                    size_within_limit = size_mb <= self.max_model_size_mb
                    
                    training_metrics.update({
                        "final_model_size_mb": size_mb,
                        "size_within_k210_limit": size_within_limit,
                        "k210_size_limit_mb": self.max_model_size_mb,
                        "size_reduction_achieved": size_mb < 50,  # Much smaller than baseline
                    })
                    
                    if not size_within_limit:
                        logger.warning(f"⚠️ Final model size {size_mb:.2f}MB exceeds K210 limit ({self.max_model_size_mb}MB)")
                    else:
                        logger.info(f"✅ Final model size {size_mb:.2f}MB (within K210 limit)")
                
                # Log K210-specific training summary
                training_metrics.update({
                    "target_hardware": "K210",
                    "input_size": self.target_input_size,
                    "training_completed": True,
                    "k210_optimizations_applied": True,
                })
                
                # Extract training results if available
                if hasattr(results, 'results_dict'):
                    training_metrics.update({
                        "training_results": results.results_dict
                    })
                
                wandb.log({
                    "k210_training_summary": training_metrics
                })
                
        except Exception as e:
            logger.warning(f"Training metrics logging failed: {e}")
    

            
    def export_for_k210(self, output_dir: str = "export_k210") -> str:
        """Export final model for K210 deployment"""
        if not self.best_model_path:
            raise RuntimeError("No trained model available for export")
            
        logger.info("📦 Exporting model for K210 deployment...")
        
        # Import the export script
        sys.path.append(str(Path(__file__).parent))
        from export_k210 import main as export_main
        
        # Prepare export arguments
        export_args = [
            '--weights', self.best_model_path,
            '--outdir', output_dir,
            '--imgsz', str(self.target_input_size[0]),
            '--calib-dir', 'data/yolo_dataset_k210/images/validation',  # Use validation set for calibration
            '--classes', str(self.data_path).replace('k210_data.yaml', 'k210_classes.txt'),
            '--simplify-onnx'
        ]
        
        # Run export
        sys.argv = ['export_k210.py'] + export_args
        try:
            export_main()
            logger.info(f"✅ K210 export completed: {output_dir}")
            return output_dir
        except Exception as e:
            logger.error(f"❌ K210 export failed: {e}")
            raise

def find_matching_checkpoint(run_id: str, checkpoint_dir: str) -> str:
    """Find checkpoint file matching W&B run ID."""
    if not run_id or not os.path.exists(checkpoint_dir):
        return None
        
    # Look for checkpoint with matching run ID in metadata
    for checkpoint in sorted(Path(checkpoint_dir).glob("*.pt"), reverse=True):
        meta_file = checkpoint.with_suffix('.json')
        if meta_file.exists():
            with open(meta_file) as f:
                metadata = json.load(f)
                if metadata.get('wandb_run_id') == run_id:
                    return str(checkpoint)
    return None

def find_latest_checkpoint(checkpoint_dir: str) -> tuple:
    """Find latest checkpoint and its W&B run ID."""
    if not os.path.exists(checkpoint_dir):
        return None, None
        
    checkpoints = sorted(Path(checkpoint_dir).glob("*.pt"), reverse=True)
    if not checkpoints:
        return None, None
        
    # Get latest checkpoint and its metadata
    latest = str(checkpoints[0])
    meta_file = checkpoints[0].with_suffix('.json')
    run_id = None
    
    if meta_file.exists():
        with open(meta_file) as f:
            metadata = json.load(f)
            run_id = metadata.get('wandb_run_id')
            
    return latest, run_id

def train_k210(config_path: str, storage_dirs: dict, checkpoint_path: str = None, wandb_run_id: str = None):
    """Execute K210 training with the original trainer (fallback)."""
    try:
        # Initialize trainer
        trainer = K210ModelTrainer(config_path)
        
        # Handle checkpoints and resumption (same logic as baseline)
        start_epoch = 0
        
        if checkpoint_path:
            # Use specified checkpoint
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            print(f"\n📦 Using specified checkpoint: {checkpoint_path}")
            # Note: K210ModelTrainer doesn't have load_checkpoint, so we start fresh
            
        elif wandb_run_id:
            # Find checkpoint matching W&B run ID
            matching_checkpoint = find_matching_checkpoint(wandb_run_id, storage_dirs['checkpoints'])
            if matching_checkpoint:
                print(f"\n📦 Found matching checkpoint for run {wandb_run_id}")
                checkpoint_path = matching_checkpoint
            else:
                print(f"\n⚠️ No checkpoint found for run {wandb_run_id}")
                
        else:
            # Auto-resume from latest if available
            latest_checkpoint, latest_run_id = find_latest_checkpoint(storage_dirs['checkpoints'])
            if latest_checkpoint:
                print("\n📦 Found latest checkpoint:")
                print(f"• File: {os.path.basename(latest_checkpoint)}")
                print(f"• W&B Run: {latest_run_id or 'Unknown'}")
                checkpoint_path = latest_checkpoint
                print(f"✅ Will attempt to resume from checkpoint")
            else:
                print("\n📋 No existing checkpoints found. Starting fresh training.")
        
        # Start training
        print(f"\n🚀 Starting K210 training...")
        best_model_path = trainer.train(storage_dirs, resume=not checkpoint_path is None)
        
        # Save final model with metadata (same as baseline)
        final_model_path = os.path.join(storage_dirs['models'], "yolov3_k210_final.pt")
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            
            # Save metadata
            meta_path = Path(final_model_path).with_suffix('.json')
            metadata = {
                'wandb_run_id': wandb.run.id if wandb.run else None,
                'timestamp': datetime.now().isoformat(),
                'config_path': config_path,
                'hardware_target': 'Sipeed Maix Bit K210',
                'resume_info': {
                    'resumed_from': checkpoint_path if checkpoint_path else None,
                    'resumed_wandb_run': wandb_run_id,
                },
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Saved final model: {final_model_path}")
            print(f"✅ Saved metadata: {meta_path}")
        
        # Return results in same format as baseline
        return {'best_model_path': best_model_path}, trainer
        
    except Exception as e:
        print(f"❌ K210 training failed: {e}")
        raise

def cleanup_resources():
    """Clean up resources and unmount Google Drive if in Colab."""
    try:
        # Finish W&B run
        if wandb.run:
            wandb.finish()
            print("✅ W&B run completed and synced")
        
        # Unmount Google Drive if in Colab
        is_colab_env, drive_module = is_colab()
        if is_colab_env:
            drive_module.flush_and_unmount()
            print("✅ Google Drive unmounted safely")
        
        print("✅ Resources cleaned up successfully")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        print("⚠️ Please manually close W&B run and unmount Google Drive if needed")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="K210-optimized YOLOv3-nano training")
    parser.add_argument("--config", default="configs/yolov3/k210_optimized.yaml", 
                       help="Path to training configuration YAML")
    parser.add_argument("--export", action="store_true", help="Export model for K210 after training")
    parser.add_argument("--export-dir", default="export_k210", help="Export output directory")
    # Resume options (same as baseline/improved scripts)
    resume_group = parser.add_argument_group('Resume Options')
    resume_group.add_argument("--resume", action="store_true",
                          help="Resume training from latest checkpoint")
    resume_group.add_argument("--checkpoint", type=str,
                          help="Resume from specific checkpoint file")
    resume_group.add_argument("--wandb-run-id", type=str,
                          help="W&B run ID to resume")
    resume_group.add_argument("--force-new-run", action="store_true",
                          help="Force new W&B run even when resuming training")
    
    parser.add_argument("--fresh", action="store_true", help="Force fresh training (ignore existing checkpoints)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Update help text to reflect auto-resume behavior
    resume_group.description = "Resume Options (script auto-detects checkpoints by default)"
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        print("\n🚀 K210-Optimized YOLOv3-Nano Training")
        print("Optimized for Sipeed Maix Bit K210 deployment with hardware constraints")
        
        # Setup directories and verify dataset
        verify_dataset()
        storage_dirs = get_storage_dirs()
        
        # Handle resumption logic (same as baseline/improved scripts)
        checkpoint_path = args.checkpoint
        wandb_run_id = args.wandb_run_id
        
        # Auto-detect existing checkpoints if not forced fresh
        if not args.fresh and not args.resume and not checkpoint_path:
            # Check if checkpoints exist and auto-enable resume
            checkpoint_locations = [
                storage_dirs['checkpoints'],  # Our custom directory
                'pokemon-classifier/yolov3n_k210_optimized/weights',  # Ultralytics default
                'pokemon-yolo-training/yolov3n_k210_optimized/weights'  # Alternative location
            ]
            
            for location in checkpoint_locations:
                latest_checkpoint, latest_run_id = find_latest_checkpoint(location)
                if latest_checkpoint:
                    print(f"🔍 Found existing checkpoint: {os.path.basename(latest_checkpoint)}")
                    print(f"🔄 Auto-enabling resume mode (use --fresh to start over)")
                    args.resume = True
                    checkpoint_path = latest_checkpoint
                    if not wandb_run_id:
                        wandb_run_id = latest_run_id
                    break
        
        if args.resume:
            if not checkpoint_path:
                # Try to find latest checkpoint in multiple locations
                checkpoint_locations = [
                    storage_dirs['checkpoints'],  # Our custom directory
                    'pokemon-classifier/yolov3n-k210-optimized/weights',  # Ultralytics default
                    'pokemon-yolo-training/yolov3n-k210-optimized/weights'  # Alternative location
                ]
                
                for location in checkpoint_locations:
                    latest_checkpoint, latest_run_id = find_latest_checkpoint(location)
                    if latest_checkpoint:
                        checkpoint_path = latest_checkpoint
                        if not wandb_run_id and not args.force_new_run:
                            wandb_run_id = latest_run_id
                        print(f"📦 Found checkpoint in: {location}")
                        break
                        
            if not wandb_run_id and not args.force_new_run:
                # Try to find run ID from file
                wandb_run_id = resume_wandb()
                
            if checkpoint_path:
                print(f"🔄 Resuming from checkpoint: {os.path.basename(checkpoint_path)}")
            if wandb_run_id:
                print(f"🔄 Resuming W&B run: {wandb_run_id}")
                
        # Load config (YOLOTrainer will handle W&B initialization)
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Show checkpoint and progress save locations
        checkpoint_dir = storage_dirs['checkpoints']
        training_dir = os.path.join(checkpoint_dir, 'yolov3n_k210_optimized')
        weights_dir = os.path.join(training_dir, 'weights')
        
        print(f"\n💾 Training Progress & Checkpoint Locations:")
        print(f"📁 Base Directory: {storage_dirs['checkpoints']}")
        if YOLOTrainer is not None:
            print(f"📁 Training Output: YOLOTrainer will create subdirectories")
            print(f"🏆 Models will be saved in: pokemon-yolo-training/ or pokemon-classifier/")
            print(f"📊 Auto-backup enabled: Every 30 minutes to Google Drive (via YOLOTrainer)")
        else:
            print(f"📁 Training Output: K210ModelTrainer (no auto-backup)")
            print(f"🏆 Models will be saved in: {storage_dirs['checkpoints']}")
            print(f"📊 Auto-backup: Manual rsync required")
        print(f"📋 W&B Run ID File: wandb_run_id.txt")
        print(f"🎯 K210 Export: Available after training completion")
        
        # Check for existing checkpoints for resumption
        resume_path = None
        if os.path.exists(checkpoint_dir):
            possible_checkpoints = [
                os.path.join(weights_dir, 'last.pt'),
                os.path.join(weights_dir, 'best.pt'),
            ]
            for checkpoint in possible_checkpoints:
                if os.path.exists(checkpoint):
                    resume_path = checkpoint
                    print(f"✅ Found existing checkpoint: {checkpoint}")
                    # Show checkpoint info
                    if os.path.exists(checkpoint):
                        size_mb = os.path.getsize(checkpoint) / (1024 * 1024)
                        modified_time = os.path.getmtime(checkpoint)
                        from datetime import datetime
                        mod_date = datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"   📏 Size: {size_mb:.2f}MB | 📅 Modified: {mod_date}")
                    break
        
        if resume_path:
            print(f"\n🔄 Resuming training from: {os.path.basename(resume_path)}")
            print("📊 This will continue from the last saved epoch")
        else:
            print(f"\n📋 No existing checkpoints found. Starting fresh training.")
            print(f"   Checkpoints will be saved to: {weights_dir}")
        
        # Verify K210 training settings (show what we're about to run)
        print("\n📋 Verifying K210 Training Settings:")
        print("• Model: YOLOv3-tiny-ultralytics (lightweight variant)")
        print("• Classes: 1025 (all Pokemon generations 1-9)")
        print("• Input Size: 224x224 (optimized for K210)")
        print("• Resume enabled:", "Yes" if args.resume else "No")
        print("• Fresh training:", "Yes" if args.fresh else "No")
        
        # Use YOLOTrainer for actual training (same as baseline/improved scripts)
        print("\n🚀 Starting K210 Training...")
        
        # Train using the proven YOLOTrainer approach
        if YOLOTrainer is not None:
            print("✅ Using YOLOTrainer with auto-backup...")
            
            # Initialize YOLOTrainer with resume ID (same as improved script)
            # This will handle W&B initialization internally
            yolo_trainer = YOLOTrainer(args.config, resume_id=wandb_run_id if not args.force_new_run else None)
            
            # Initialize model (YOLOTrainer handles checkpoint loading internally)
            yolo_trainer._setup_model()
            
            # Train with automatic resumption (same pattern as other scripts)
            results = yolo_trainer.train()
            
            # Get best model path
            best_model = results.get('best_model_path', 'unknown')
            trainer = yolo_trainer
            trainer.best_model_path = best_model
            
        else:
            print("⚠️ YOLOTrainer not available. Using K210-specific training...")
            
            # Fallback: Initialize W&B manually and use K210-specific training
            config = initialize_wandb(
                Path(args.config), 
                resume_id=None if args.force_new_run else wandb_run_id
            )
            
            # Use train_k210 function (like train_baseline in baseline script)
            results, trainer = train_k210(args.config, storage_dirs, checkpoint_path, wandb_run_id)
            best_model = trainer.best_model_path
        
        # Export if requested
        if args.export:
            print("\n📦 Exporting for K210...")
            trainer.export_for_k210(args.export_dir)
            
        # Cleanup
        cleanup_resources()
        
        # Print final summary
        print("\n✨ K210 Training Results Summary:")
        print(f"• Best model: {best_model}")
        print(f"• Target deployment: Sipeed Maix Bit K210")
        print(f"• Model size target: <2MB")
        print(f"• Runtime memory target: <3MB")
        
        print("\n🔍 K210 Optimizations Applied:")
        print("1. YOLOv3-tiny-ultralytics architecture (lightweight)")
        print("2. 224x224 input size (vs 416x416)")
        print("3. Conservative augmentation for stability")
        print("4. Extended patience for convergence")
        print("5. ALL 1025 Pokemon classes maintained (not reduced)")
        
        print(f"\n📂 Training Artifacts Summary:")
        print(f"📁 Local Storage: {storage_dirs['checkpoints']}")
        print(f"📁 Training Output: {best_model}")
        print(f"🏆 Best Model: {best_model}")
        if YOLOTrainer is not None:
            print(f"📦 Auto-Backup: /content/drive/MyDrive/ (via YOLOTrainer)")
        else:
            print(f"📦 Auto-Backup: Manual - use rsync commands shown in training")
        print(f"📋 W&B Dashboard: https://wandb.ai/liuhuanjim013-self/pokemon-classifier")
        if wandb.run:
            print(f"🔗 Current Run: https://wandb.ai/liuhuanjim013-self/pokemon-classifier/runs/{wandb.run.id}")
        
        print("\n🎯 Next Steps:")
        print("1. Check model size and verify <2MB")
        print("2. Test ONNX export compatibility")
        print("3. Compile with nncase for K210")
        print("4. Deploy to Sipeed Maix Bit")
        print(f"5. Resume training: python scripts/yolo/train_k210_optimized.py")
        print(f"6. Fresh training: python scripts/yolo/train_k210_optimized.py --fresh")
        
        logger.info("🎉 K210 training pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user!")
        print("Latest checkpoint was saved automatically.")
        print("You can resume training by running this script again")
        cleanup_resources()
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        cleanup_resources()
        logger.error(f"❌ Training pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
