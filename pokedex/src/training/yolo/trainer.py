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
            
            # Try loading from local cache first
            try:
                weights_file = cache_dir / model_weights
                if weights_file.exists():
                    logger.info(f"Loading from cache: {weights_file}")
                    self.model = YOLO(str(weights_file))
                    logger.info("Successfully loaded model from cache")
                else:
                    # Try downloading from Ultralytics assets
                    logger.info("Downloading from Ultralytics assets...")
                    from ultralytics.utils.downloads import download
                    
                    # Try different asset URLs
                    urls = [
                        f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_weights}",
                        f"https://github.com/ultralytics/yolov3/releases/download/v9.0/{model_weights}",
                        f"https://ultralytics.com/assets/{model_weights}"
                    ]
                    
                    for url in urls:
                        try:
                            logger.info(f"Trying URL: {url}")
                            weights_path = download(
                                url=url,
                                dir=str(cache_dir),
                                unzip=False,
                                curl=True,
                                retry=3
                            )
                            if weights_path is not None and Path(weights_path).exists():
                                weights_file = Path(weights_path)
                                self.model = YOLO(str(weights_file))
                                logger.info("Successfully loaded downloaded model")
                                break
                        except Exception as e:
                            logger.warning(f"Download failed from {url}: {e}")
                    else:
                        raise FileNotFoundError(f"Failed to download weights from any URL")
                        
            except Exception as e1:
                logger.warning(f"Local and download attempts failed: {e1}")
                try:
                    # Try loading directly from hub
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
                        raise RuntimeError(f"Model loading failed: Local error: {e1}, Hub error: {e2}, YAML error: {e3}")
                    
                            
            # Configure for 1025 classes
            logger.info(f"Configuring model for {classes} classes...")
            if not hasattr(self.model.model, 'model') or not hasattr(self.model.model.model[-1], 'nc'):
                raise AttributeError("Model structure not as expected")
            
            self.model.model.model[-1].nc = classes
            logger.info(f"Model setup complete: {model_name} with {classes} classes")
            
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise
    
    def _setup_checkpoint_manager(self):
        """Initialize checkpoint manager."""
        self.checkpoint_manager = CheckpointManager(self.config['checkpoint'])
    
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
            # Add custom logging during training
            if self.wandb_integration and wandb.run:
                logger.info("Enhanced W&B logging enabled")
                # Log initial configuration
                wandb.log({
                    'config/model': self.config['model'],
                    'config/training': self.config['training'],
                    'config/data': self.config['data'],
                })
            
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
        
        # Base training arguments
        train_args = {
            'data': str(Path('configs/yolov3/yolo_data.yaml')),  # YOLO format data config
            'epochs': train_config['epochs'],
            'batch': train_config['batch_size'],
            'imgsz': model_config['img_size'],
            'device': 'auto',  # Auto-detect GPU/CPU
            'project': 'pokemon-yolo-training',
            'name': f"{self.config['wandb']['name']}",
            'save_period': self.config['checkpoint']['save_frequency'],
            'save_dir': self.config['checkpoint']['save_dir'],  # Save checkpoints here
            'lr0': train_config['learning_rate'],
            'weight_decay': train_config['weight_decay'],
            'pretrained': model_config['pretrained'],
            'plots': True,  # Enable plotting
            'save_period': 1,  # Save every epoch for better tracking
            'verbose': True,  # Enable verbose logging
            'exist_ok': True,  # Overwrite existing runs
            'patience': 100,  # Early stopping patience
            'save': True,  # Save checkpoints
            'save_txt': False,  # Don't save text predictions
            'save_conf': False,  # Don't save confidence scores
            
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
        
        # Add scheduler if specified
        if train_config['scheduler'] == 'cosine':
            train_args['cos_lr'] = True
            train_args['warmup_epochs'] = train_config.get('warmup_epochs', 5)
        
        # Add early stopping if specified
        if train_config['early_stopping'] != 'none':
            train_args['patience'] = train_config['early_stopping']['patience']
        
        return train_args
    
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