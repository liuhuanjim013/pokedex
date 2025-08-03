#!/usr/bin/env python3
"""
YOLOv3 training script for Pokemon classifier.
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any
import logging
import wandb
from ultralytics import YOLO
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOv3PokemonTrainer:
    """Train YOLOv3 model for Pokemon classification."""
    
    def __init__(self, config_path: str = "configs/yolov3/training_config.yaml"):
        """Initialize trainer with YOLOv3-specific configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize W&B
        self._setup_wandb()
        
        # Model settings
        self.model_name = self.config['model']['name']
        self.num_classes = self.config['model']['classes']
        self.img_size = self.config['model']['img_size']
        
        # Training settings
        self.epochs = self.config['training']['epochs']
        self.batch_size = self.config['training']['batch_size']
        self.learning_rate = self.config['training']['learning_rate']
        
        # Hardware settings
        self.device = self.config['hardware']['device']
        
        logger.info(f"Initialized YOLOv3 trainer for {self.num_classes} classes")
    
    def _setup_wandb(self):
        """Set up Weights & Biases tracking."""
        wandb_config = self.config['wandb']
        
        wandb.init(
            project=wandb_config['project'],
            name=wandb_config['name'],
            tags=wandb_config['tags'],
            config={
                'model': self.config['model'],
                'training': self.config['training'],
                'data': self.config['data'],
                'yolov3': self.config['yolov3']
            }
        )
        
        logger.info(f"W&B initialized: {wandb.run.name}")
    
    def train(self, data_yaml_path: str, model_save_dir: str = "models/checkpoints/yolov3"):
        """
        Train YOLOv3 model for Pokemon classification.
        
        Args:
            data_yaml_path: Path to data.yaml file
            model_save_dir: Directory to save trained models
        """
        logger.info("Starting YOLOv3 training for Pokemon classification")
        
        # Create model save directory
        model_save_dir = Path(model_save_dir)
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load YOLOv3 model
        if self.config['model']['pretrained']:
            model = YOLO("yolov3.pt")  # Load pretrained YOLOv3
        else:
            model = YOLO("yolov3.yaml")  # Load YOLOv3 config
        
        # Configure model for classification
        model.model.model[-1].nc = self.num_classes  # Set number of classes
        
        # YOLOv3-specific training arguments
        train_args = {
            'data': data_yaml_path,
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.img_size,
            'device': self.device,
            'project': str(model_save_dir),
            'name': f"{self.model_name}_pokemon",
            'save_period': self.config['checkpointing']['save_freq'],
            'patience': self.config['training']['early_stopping']['patience'],
            'lr0': self.learning_rate,
            'weight_decay': self.config['training']['weight_decay'],
            'warmup_epochs': self.config['training']['scheduler']['warmup_epochs'],
            'cos_lr': True,  # Cosine learning rate scheduling
            # YOLOv3-specific augmentations
            'hsv_h': self.config['data']['augmentation']['hsv_h'],
            'hsv_s': self.config['data']['augmentation']['hsv_s'],
            'hsv_v': self.config['data']['augmentation']['hsv_v'],
            'degrees': self.config['data']['augmentation']['degrees'],
            'translate': self.config['data']['augmentation']['translate'],
            'scale': self.config['data']['augmentation']['scale'],
            'shear': self.config['data']['augmentation']['shear'],
            'perspective': self.config['data']['augmentation']['perspective'],
            'flipud': self.config['data']['augmentation']['flipud'],
            'fliplr': self.config['data']['augmentation']['fliplr'],
            'mosaic': self.config['data']['augmentation']['mosaic'],
            'mixup': self.config['data']['augmentation']['mixup'],
            # YOLOv3-specific settings
            'anchor_t': self.config['yolov3']['anchor_t'],
            'fl_gamma': self.config['yolov3']['fl_gamma'],
            'label_smoothing': self.config['yolov3']['label_smoothing'],
            'overlap_mask': self.config['yolov3']['overlap_mask'],
            'mask_ratio': self.config['yolov3']['mask_ratio'],
            'dropout': self.config['yolov3']['dropout'],
        }
        
        # Start training
        logger.info(f"Training YOLOv3 with {self.epochs} epochs, batch size {self.batch_size}")
        results = model.train(**train_args)
        
        # Log final metrics
        self._log_final_metrics(results)
        
        # Save best model
        best_model_path = model_save_dir / f"{self.model_name}_pokemon" / "weights" / "best.pt"
        if best_model_path.exists():
            final_model_path = model_save_dir / f"{self.model_name}_pokemon_best.pt"
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            logger.info(f"Best YOLOv3 model saved to {final_model_path}")
        
        wandb.finish()
        return results
    
    def _log_final_metrics(self, results):
        """Log final training metrics to W&B."""
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            wandb.log({
                'final/train_loss': metrics.get('train/box_loss', 0),
                'final/val_loss': metrics.get('val/box_loss', 0),
                'final/mAP50': metrics.get('metrics/mAP50', 0),
                'final/mAP50-95': metrics.get('metrics/mAP50-95', 0),
                'final/precision': metrics.get('metrics/precision', 0),
                'final/recall': metrics.get('metrics/recall', 0)
            })
    
    def evaluate(self, model_path: str, data_yaml_path: str):
        """
        Evaluate trained YOLOv3 model.
        
        Args:
            model_path: Path to trained model
            data_yaml_path: Path to data.yaml file
        """
        logger.info(f"Evaluating YOLOv3 model: {model_path}")
        
        model = YOLO(model_path)
        results = model.val(data=data_yaml_path)
        
        # Log evaluation metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            wandb.log({
                'eval/mAP50': metrics.get('metrics/mAP50', 0),
                'eval/mAP50-95': metrics.get('metrics/mAP50-95', 0),
                'eval/precision': metrics.get('metrics/precision', 0),
                'eval/recall': metrics.get('metrics/recall', 0)
            })
        
        return results

def create_data_yaml(dataset_path: str, output_path: str, num_classes: int):
    """
    Create data.yaml file for YOLOv3 training.
    
    Args:
        dataset_path: Path to YOLO dataset
        output_path: Output path for data.yaml
        num_classes: Number of classes
    """
    dataset_path = Path(dataset_path)
    
    yaml_content = {
        'path': str(dataset_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': num_classes,
        'names': []
    }
    
    # Load class names
    classes_file = dataset_path / "classes.txt"
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            yaml_content['names'] = [line.strip() for line in f.readlines()]
    
    # Write data.yaml
    with open(output_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    logger.info(f"Created data.yaml at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv3 model for Pokemon classification")
    parser.add_argument("--data_yaml", required=True, help="Path to data.yaml file")
    parser.add_argument("--config", default="configs/yolov3/training_config.yaml", help="Config file path")
    parser.add_argument("--model_save_dir", default="models/checkpoints/yolov3", help="Model save directory")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model after training")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = YOLOv3PokemonTrainer(args.config)
    
    # Train model
    results = trainer.train(args.data_yaml, args.model_save_dir)
    
    # Evaluate if requested
    if args.evaluate:
        best_model_path = Path(args.model_save_dir) / f"{trainer.model_name}_pokemon_best.pt"
        if best_model_path.exists():
            trainer.evaluate(str(best_model_path), args.data_yaml)
    
    print("YOLOv3 training completed!")

if __name__ == "__main__":
    main() 