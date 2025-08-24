#!/usr/bin/env python3
"""
YOLOv11 Training Script for Maix Cam
Optimized for modern hardware with full 1025 Pokemon classes
Based on latest YOLOv11 recommendations for classification tasks
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.yolo.trainer import YOLOTrainer
from src.training.yolo.wandb_integration import WandBIntegration
import wandb

def setup_logging():
    """Configure logging for Maix Cam YOLOv11 training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('maixcam_yolov11_training.log')
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for Maix Cam YOLOv11 training"""
    parser = argparse.ArgumentParser(description='Train YOLOv11 for Maix Cam')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume from latest checkpoint')
    parser.add_argument('--fresh', action='store_true',
                       help='Start fresh training (ignore checkpoints)')
    parser.add_argument('--checkpoint', type=str,
                       help='Resume from specific checkpoint path')
    parser.add_argument('--model', choices=['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'],
                       default='yolo11m', help='YOLOv11 model variant')
    parser.add_argument('--resolution', type=int, choices=[256, 320],
                       default=256, help='Input resolution')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--force-new-run', action='store_true',
                       help='Force new W&B run even if checkpoint exists')
    return parser.parse_args()

def main():
    """Main training function for Maix Cam YOLOv11"""
    args = parse_args()
    logger = setup_logging()
    
    logger.info("=== Maix Cam YOLOv11 Training ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Resolution: {args.resolution}x{args.resolution}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    
    # Maix Cam specific configuration for YOLOv11
    maixcam_config = {
        'model_name': f'yolo11-{args.model}-maixcam',
        'hardware_target': 'Maix Cam',
        'model_variant': args.model,
        'input_resolution': args.resolution,
        'classes': 1025,  # Full Pokemon coverage
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': 0.01,  # YOLOv11 default
        'optimizer': 'auto',  # YOLOv11 auto-selects best optimizer
        'task': 'classify',  # Classification task
        'augmentation': {
            # Resize-based pipeline (avoid aggressive random crops)
            'resize': True,  # Use resize instead of random crops
            'randaugment': True,  # RandAugment for fine-grained classification
            'random_erasing': 0.1,  # RandomErasing for robustness
            'color_jitter': True,  # Color jittering
            'degrees': 15.0,  # Rotation
            'translate': 0.2,  # Translation
            'scale': 0.5,  # Scale
            'shear': 2.0,  # Shear
            'perspective': 0.001,  # Perspective
            'flipud': 0.5,  # Vertical flip
            'fliplr': 0.5,  # Horizontal flip
            # Disable mosaic/mixup for classification
            'mosaic': 0.0,  # Disabled for classification
            'mixup': 0.0,  # Disabled for classification
        },
        'scheduler': 'cosine',
        'warmup_epochs': 3,
        'early_stopping': {
            'patience': 15,
            'min_delta': 0.001,
            'monitor': 'val_loss'
        },
        'class_balancing': True,  # Class-balanced sampling for 1025 classes
        'metrics': ['top1', 'top5'],  # Track top-1 and top-5 accuracy
    }
    
    # YOLOTrainer will handle W&B initialization using the config file
    # The config file already has the correct entity: liuhuanjim013-self
    
    # Create YOLO trainer with Maix Cam optimizations for YOLOv11
    # Use the correct constructor signature
    trainer = YOLOTrainer(
        config_path="configs/yolov11/maixcam_data.yaml",
        resume_id=None if args.force_new_run else None
    )
    
    # Initialize the model (required before training)
    trainer._setup_model()
    
    # Storage directories for Maix Cam (use relative paths for local development)
    storage_dirs = {
        'models': 'models/maixcam',
        'checkpoints': 'models/maixcam/checkpoints',
        'logs': 'models/maixcam/logs',
        'exports': 'models/maixcam/exports'
    }
    
    # Create directories
    for dir_path in storage_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    try:
        # Start training
        logger.info("Starting Maix Cam YOLOv11 training...")
        logger.info("Key YOLOv11 features:")
        logger.info("- Resize-based pipeline (avoid aggressive random crops)")
        logger.info("- RandAugment + RandomErasing for fine-grained classification")
        logger.info("- Class-balanced sampling for 1025 classes")
        logger.info("- top-1/top-5 accuracy tracking")
        logger.info("- Optimized for classification tasks")
        
        # YOLOTrainer handles its own checkpoint management
        results = trainer.train(start_epoch=0)
        
        # Get the best model path from results
        checkpoint_path = results.get('best_model_path', 'unknown')
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Best checkpoint: {checkpoint_path}")
        
        # Export for Maix Cam (manual export after training)
        logger.info("Training completed! Model ready for Maix Cam deployment.")
        logger.info(f"Best model saved at: {checkpoint_path}")
        logger.info("To export for Maix Cam, use the export script after training.")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        wandb.finish(exit_code=1)
        sys.exit(1)
    
    wandb.finish()
    logger.info("Maix Cam YOLOv11 training completed!")

if __name__ == "__main__":
    main()
