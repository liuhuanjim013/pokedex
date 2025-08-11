#!/usr/bin/env python3
"""
Core YOLO Training Class
Imported by training scripts for YOLOv3 training with W&B and checkpoint management
"""

import os
import yaml
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from ultralytics import YOLO
import wandb

from .checkpoint_manager import CheckpointManager
from .wandb_integration import WandBIntegration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOTrainer:
    """Core YOLO training class with W&B integration and checkpoint management."""
    
    def __init__(self, config_path: str, resume_id: Optional[str] = None):
        """Initialize trainer with configuration."""
        try:
            self.config = self._load_config(config_path)
            self.model = None
            self.checkpoint_manager = None
            self.wandb_integration = None
            
            # Initialize components in order of least likely to fail to most likely
            self._setup_checkpoint_manager()  # Just creates directories
            self._setup_wandb(resume_id)  # Just sets up logging
            # Don't initialize model here - let caller do it explicitly
            
            logger.info(f"YOLOTrainer initialized with config: {config_path}")
            logger.info(f"Model: {self.config['model']['name']}, Classes: {self.config['model']['classes']}")
        except Exception as e:
            # Clean up W&B on any initialization error
            if hasattr(self, 'wandb_integration') and self.wandb_integration:
                self.wandb_integration.finish()
            raise
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_model(self):
        """Initialize YOLOv3 model."""
        model_name = self.config['model']['name']
        model_weights = self.config['model']['weights']
        classes = self.config['model']['classes']
        cache_dir = Path.home() / '.cache' / 'ultralytics'
        
        try:
            # Create cache directory if it doesn't exist
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Clean up potentially corrupted weights
            weights_file = cache_dir / model_weights
            if weights_file.exists():
                logger.info(f"Removing potentially corrupted weights: {weights_file}")
                weights_file.unlink()
            
            # Initialize model with pretrained weights
            logger.info(f"Initializing {model_name} model...")
            
            # Try loading official YOLOv3 weights directly
            try:
                logger.info("Loading official YOLOv3 model...")
                self.model = YOLO(model_weights)  # This will download official weights
                logger.info("Successfully loaded official YOLOv3 model")
                        
            except Exception as e1:
                logger.warning(f"Official weights loading failed: {e1}")
                try:
                    # Fallback - try loading from hub
                    logger.info("Attempting to load from hub...")
                    self.model = YOLO("yolov3")  # This will download from hub if needed
                    logger.info("Successfully loaded model from hub")
                    
                except Exception as e2:
                    # Final fallback - try creating from YAML
                    try:
                        logger.info("Falling back to YAML model creation...")
                        
                        # First try user-specified YAML if provided
                        if 'yaml' in self.config['model']:
                            yaml_path = Path(self.config['model']['yaml'])
                            if not yaml_path.exists():
                                raise FileNotFoundError(f"Specified YAML file not found: {yaml_path}")
                        else:
                            # Try default location
                            yaml_path = Path("models/configs/yolov3.yaml")
                            if not yaml_path.exists():
                                # Try relative to current directory
                                yaml_path = Path.cwd() / "models" / "configs" / "yolov3.yaml"
                                if not yaml_path.exists():
                                    raise FileNotFoundError(f"Default YAML file not found: {yaml_path}")
                        
                        self.model = YOLO(str(yaml_path))
                        logger.info("Successfully created model from YAML")
                    except Exception as e3:
                        raise RuntimeError(f"Model loading failed: Official error: {e1}, Hub error: {e2}, YAML error: {e3}")
                    
                            
            # Configure for 1025 classes
            logger.info(f"Configuring model for {classes} classes...")
            
            # Update the model to support our number of classes
            if hasattr(self.model.model, 'model') and hasattr(self.model.model.model[-1], 'nc'):
                # Standard YOLO model structure
                self.model.model.model[-1].nc = classes
                logger.info(f"Updated detection head to {classes} classes")
            elif hasattr(self.model.model, 'nc'):
                # Direct model attribute
                self.model.model.nc = classes
                logger.info(f"Updated model to {classes} classes")
            else:
                logger.warning("Could not automatically update model classes - will be handled during training")
            
            logger.info(f"Model setup complete: {model_name} with {classes} classes")
            
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise
    
    def _setup_checkpoint_manager(self):
        """Initialize checkpoint manager."""
        # Use a Colab-friendly default if the configured path is not suitable
        checkpoint_cfg = dict(self.config['checkpoint'])
        try:
            from copy import deepcopy
            checkpoint_cfg = deepcopy(self.config['checkpoint'])
        except Exception:
            pass

        configured_save_dir = Path(checkpoint_cfg.get('save_dir', ''))

        # Detect Colab and prefer /content paths to avoid /home mismatches
        in_colab_env = Path('/content').exists()
        if in_colab_env:
            colab_default = Path('/content/models/checkpoints')
            # If configured path is empty, under /home, or clearly not writable, switch to Colab default
            if (not configured_save_dir or str(configured_save_dir).startswith('/home') or
                    (configured_save_dir.exists() and not os.access(configured_save_dir, os.W_OK))):
                checkpoint_cfg['save_dir'] = str(colab_default)

        self.checkpoint_manager = CheckpointManager(checkpoint_cfg)
    
    def _setup_wandb(self, resume_id: Optional[str] = None):
        """Initialize W&B integration."""
        self.wandb_integration = WandBIntegration(self.config['wandb'], resume_id=resume_id)
    
    def train(self, start_epoch: int = 0) -> Dict[str, Any]:
        """
        Main training loop with W&B logging and checkpoint management.
        
        Args:
            start_epoch: Epoch to start training from (for resume)
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training from epoch {start_epoch}")
        
        # Prepare training arguments
        train_args = self._prepare_training_args()
        
        # Start training
        try:
            # Set up automatic backup
            self._setup_auto_backup()
            
            # Add custom logging during training
            if self.wandb_integration and wandb.run:
                logger.info("Enhanced W&B logging enabled")
                # Log initial configuration
                wandb.log({
                    'config/model': self.config['model'],
                    'config/training': self.config['training'],
                    'config/data': self.config['data'],
                })
                
                # Attach a callback to stream Ultralytics metrics to W&B
                def _wandb_on_epoch_end(trainer_obj):
                    try:
                        payload = {}
                        # Trainer metrics dict (Ultralytics)
                        if hasattr(trainer_obj, 'metrics') and isinstance(trainer_obj.metrics, dict):
                            for key, value in trainer_obj.metrics.items():
                                try:
                                    payload[key] = float(value)
                                except Exception:
                                    pass
                        # Learning rate (first param group)
                        try:
                            if hasattr(trainer_obj, 'optimizer') and trainer_obj.optimizer:
                                payload['lr'] = float(trainer_obj.optimizer.param_groups[0].get('lr', 0.0))
                        except Exception:
                            pass
                        # Epoch index
                        try:
                            payload['epoch'] = int(getattr(trainer_obj, 'epoch', 0))
                        except Exception:
                            pass
                        if payload:
                            wandb.log(payload)
                    except Exception as _e:
                        logger.debug(f"W&B epoch-end logging skipped: {_e}")
                
                # Register for both train and val epoch ends to capture full set
                try:
                    self.model.add_callback('on_fit_epoch_end', _wandb_on_epoch_end)
                    self.model.add_callback('on_val_end', _wandb_on_epoch_end)
                    logger.info("W&B live metric callback attached")
                except Exception as _e:
                    logger.debug(f"Failed to attach W&B callbacks: {_e}")
            
            results = self.model.train(**train_args)
            
            # Track training progress
            results['start_epoch'] = start_epoch
            results['end_epoch'] = start_epoch + results.get('epochs', 0)
            results['global_step'] = wandb.run.step if wandb.run else 0
            
            # Log final metrics
            self._log_final_metrics(results)
            
            logger.info("Training completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _prepare_training_args(self) -> Dict[str, Any]:
        """Prepare training arguments from config."""
        train_config = self.config['training']
        model_config = self.config['model']
        data_config = self.config['data']
        
        # Detect available device
        if torch.cuda.is_available():
            device = '0'  # Use first GPU
            logger.info("🚀 CUDA GPU detected - using GPU acceleration")
        else:
            device = 'cpu'
            logger.info("💻 No GPU detected - using CPU training")
        
        # Auto-detect data config based on model type
        if 'k210' in self.config['wandb']['name'].lower():
            data_config_path = 'configs/yolov3/k210_data.yaml'
            logger.info("📊 Using K210 data configuration")
        else:
            data_config_path = 'configs/yolov3/yolo_data.yaml'
            logger.info("📊 Using standard data configuration")
        
        # Base training arguments
        train_args = {
            'data': str(Path(data_config_path)),  # Auto-detected data config
            'task': 'detect',  # Detection task (not classification)
            'epochs': train_config['epochs'],
            'batch': train_config['batch_size'],
            'imgsz': model_config['img_size'],
            'device': device,  # Use detected device
            'project': 'pokemon-yolo-training',
            'name': f"{self.config['wandb']['name']}",
            'save_period': self.config['checkpoint']['save_frequency'],
            'save_dir': self.config['checkpoint']['save_dir'],  # Save checkpoints here
            'lr0': train_config['learning_rate'],
            'weight_decay': train_config['weight_decay'],
            'pretrained': model_config['pretrained'],
            'plots': True,  # Enable plotting
            'save_period': 1,  # Save every 1 epoch
            'verbose': True,  # Enable verbose logging
            'exist_ok': True,  # Overwrite existing runs
            'patience': 100,  # Early stopping patience
            'save': True,  # Save checkpoints
            'save_txt': False,  # Don't save text predictions
            'save_conf': False,  # Don't save confidence scores
            # W&B integration for Ultralytics
            'project': self.config['wandb']['project'],
            'name': self.config['wandb']['name'],
            
            # Augmentation parameters
            'hsv_h': train_config['augmentation']['hsv_h'],
            'hsv_s': train_config['augmentation']['hsv_s'],
            'hsv_v': train_config['augmentation']['hsv_v'],
            'degrees': train_config['augmentation']['degrees'],
            'translate': train_config['augmentation']['translate'],
            'scale': train_config['augmentation']['scale'],
            'shear': train_config['augmentation']['shear'],
            'perspective': train_config['augmentation']['perspective'],
            'flipud': train_config['augmentation']['flipud'],
            'fliplr': train_config['augmentation']['fliplr'],
            'mosaic': train_config['augmentation']['mosaic'],
            'mixup': train_config['augmentation']['mixup'],
        }
        
        # Auto-detect resume from Ultralytics run directories
        try:
            run_name = self.config['wandb']['name']
            project_dir = Path(self.config['wandb']['project'])
            
            # Multiple possible checkpoint locations
            candidate_run_dirs = [
                project_dir / run_name,
                Path('pokemon-yolo-training') / run_name,
                # K210-specific paths (handle naming differences)
                project_dir / 'yolov3n_k210_optimized',
                project_dir / run_name.replace('-', '_'),  # Handle dash/underscore differences
                project_dir / run_name.replace('yolov3-tinyu', 'yolov3n'),  # Handle model name differences
            ]
            resume_path = None
            latest_mtime = -1.0
            logger.info(f"🔍 Searching for checkpoints in {len(candidate_run_dirs)} locations...")
            for run_dir in candidate_run_dirs:
                weights_dir = run_dir / 'weights'
                last_pt = weights_dir / 'last.pt'
                logger.debug(f"Checking: {last_pt}")
                if last_pt.exists():
                    mtime = last_pt.stat().st_mtime
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        resume_path = last_pt
                        logger.info(f"✅ Found checkpoint: {last_pt}")
                    else:
                        logger.debug(f"Found older checkpoint: {last_pt}")
                else:
                    logger.debug(f"No checkpoint at: {last_pt}")

            if resume_path is not None:
                # Reinitialize model from checkpoint to restore weights seamlessly
                try:
                    self.model = YOLO(str(resume_path))
                    logger.info(f"Loaded model from checkpoint: {resume_path}")
                except Exception as _load_e:
                    logger.warning(f"Failed to load model from checkpoint, will rely on Ultralytics resume: {_load_e}")
                train_args['resume'] = True
                logger.info(f"Resuming Ultralytics training from checkpoint: {resume_path}")
            else:
                train_args['resume'] = False
        except Exception as _e:
            # If anything goes wrong, don't block training
            train_args['resume'] = False
            logger.debug(f"Resume auto-detection failed, starting fresh: {_e}")

        # Add scheduler if specified
        if train_config['scheduler'] == 'cosine':
            train_args['cos_lr'] = True
            train_args['warmup_epochs'] = train_config.get('warmup_epochs', 5)
        
        # Add early stopping if specified
        if train_config['early_stopping'] != 'none':
            train_args['patience'] = train_config['early_stopping']['patience']
        
        # Add optimizer settings if specified
        if 'optimizer' in train_config:
            train_args['optimizer'] = train_config['optimizer']
        if 'momentum' in train_config:
            train_args['momentum'] = train_config['momentum']
        
        return train_args
    
    def _setup_auto_backup(self):
        """Set up automatic backup to Google Drive."""
        import subprocess
        import threading
        import time
        
        def backup_worker():
            """Background worker to backup training outputs."""
            while True:
                try:
                    # Check if training directory exists - look for actual training dirs
                    training_dirs = ['pokemon-classifier', 'pokemon-yolo-training']
                    backup_dir = '/content/drive/MyDrive/pokemon-yolo-training/'
                    
                    for training_dir in training_dirs:
                        if Path(training_dir).exists():
                            # Create backup directory if it doesn't exist
                            Path(backup_dir).mkdir(parents=True, exist_ok=True)
                            
                            # Backup to Google Drive
                            subprocess.run([
                                'rsync', '-ravz',
                                f'{training_dir}/',
                                backup_dir
                            ], check=False)  # Don't fail if Google Drive not mounted
                            logger.info(f"📁 Auto-backup completed: {training_dir} -> {backup_dir}")
                    
                    # Wait 30 minutes before next backup
                    time.sleep(1800)
                except Exception as e:
                    logger.warning(f"Auto-backup failed: {e}")
                    time.sleep(1800)  # Wait before retrying
        
        # Start backup worker in background
        backup_thread = threading.Thread(target=backup_worker, daemon=True)
        backup_thread.start()
        logger.info("🔄 Auto-backup to Google Drive enabled (every 30 minutes)")
    
    def _calculate_classification_metrics(self, model, val_dataloader):
        """Calculate top-1 and top-5 accuracy for classification."""
        try:
            import torch
            import numpy as np
            from collections import defaultdict
            
            model.eval()
            correct_top1 = 0
            correct_top5 = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    images, targets = batch
                    if torch.cuda.is_available():
                        images = images.cuda()
                    
                    # Run inference
                    results = model(images)
                    
                    # For YOLO, extract class predictions
                    for i, result in enumerate(results):
                        if hasattr(result, 'boxes') and len(result.boxes) > 0:
                            # Get the highest confidence detection
                            confidences = result.boxes.conf
                            classes = result.boxes.cls
                            
                            # Get ground truth class
                            gt_class = int(targets[i]['cls'][0]) if 'cls' in targets[i] else 0
                            
                            # Get top predictions
                            top_indices = torch.argsort(confidences, descending=True)[:5]
                            top_classes = classes[top_indices].cpu().numpy()
                            
                            # Check top-1 accuracy
                            if len(top_classes) > 0 and int(top_classes[0]) == gt_class:
                                correct_top1 += 1
                            
                            # Check top-5 accuracy
                            if gt_class in [int(c) for c in top_classes]:
                                correct_top5 += 1
                            
                            total += 1
            
            top1_acc = correct_top1 / total if total > 0 else 0
            top5_acc = correct_top5 / total if total > 0 else 0
            
            return {
                'top1_accuracy': top1_acc,
                'top5_accuracy': top5_acc,
                'total_samples': total
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate classification metrics: {e}")
            return {'top1_accuracy': 0, 'top5_accuracy': 0, 'total_samples': 0}
    
    def _log_final_metrics(self, results: Dict[str, Any]):
        """Log final training metrics to W&B."""
        if self.wandb_integration:
            final_metrics = {
                'final_map': results.get('metrics/mAP50(B)', 0),
                'final_accuracy': results.get('metrics/accuracy', 0),
                'final_loss': results.get('train/box_loss', 0),
                'training_time': results.get('train/epoch', 0),
            }
            self.wandb_integration.log_final_metrics(final_metrics)
    
    def load_checkpoint(self) -> int:
        """
        Load latest checkpoint and return starting epoch.
        
        Returns:
            Starting epoch number
        """
        if self.checkpoint_manager:
            start_epoch = self.checkpoint_manager.load_latest_checkpoint(self.model)
            logger.info(f"Loaded checkpoint, starting from epoch {start_epoch}")
            return start_epoch
        return 0
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, Any]):
        """Save checkpoint with current state."""
        if self.checkpoint_manager:
            self.checkpoint_manager.save_checkpoint(
                self.model, epoch, metrics, self.config
            )
    
    def evaluate(self, test_data: str) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Path to test data
            
        Returns:
            Evaluation metrics
        """
        logger.info("Starting model evaluation...")
        
        # Run validation
        results = self.model.val(data=test_data)
        
        # Extract metrics
        metrics = {
            'mAP50': results.get('metrics/mAP50(B)', 0),
            'mAP50-95': results.get('metrics/mAP50-95(B)', 0),
            'precision': results.get('metrics/precision(B)', 0),
            'recall': results.get('metrics/recall(B)', 0),
        }
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics 