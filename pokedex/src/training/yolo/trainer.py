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
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.model = None
        self.checkpoint_manager = None
        self.wandb_integration = None
        
        # Initialize components
        self._setup_model()
        self._setup_checkpoint_manager()
        self._setup_wandb()
        
        logger.info(f"YOLOTrainer initialized with config: {config_path}")
        logger.info(f"Model: {self.config['model']['name']}, Classes: {self.config['model']['classes']}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_model(self):
        """Initialize YOLOv3 model."""
        model_name = self.config['model']['name']
        classes = self.config['model']['classes']
        
        # Load YOLOv3 model
        self.model = YOLO(f"{model_name}.pt")
        
        # Configure for 1025 classes
        self.model.model.model[-1].nc = classes
        
        logger.info(f"Model setup: {model_name} with {classes} classes")
    
    def _setup_checkpoint_manager(self):
        """Initialize checkpoint manager."""
        self.checkpoint_manager = CheckpointManager(self.config['checkpoint'])
    
    def _setup_wandb(self):
        """Initialize W&B integration."""
        self.wandb_integration = WandBIntegration(self.config['wandb'])
    
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
            results = self.model.train(**train_args)
            
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
            'data': data_config['dataset'],
            'epochs': train_config['epochs'],
            'batch': train_config['batch_size'],
            'imgsz': model_config['img_size'],
            'device': 'auto',  # Auto-detect GPU/CPU
            'project': 'pokemon-yolo-training',
            'name': f"{self.config['wandb']['name']}",
            'save_period': self.config['checkpoint']['save_frequency'],
            'lr0': train_config['learning_rate'],
            'weight_decay': train_config['weight_decay'],
            'pretrained': model_config['pretrained'],
            
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