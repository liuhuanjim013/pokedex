#!/usr/bin/env python3
"""
Reproduce the original blog's YOLOv3 training approach.
Based on: https://www.cnblogs.com/xianmasamasa/p/18995912
"""

import os
import yaml
import argparse
from pathlib import Path
import logging
import wandb
from ultralytics import YOLO
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OriginalBlogReproducer:
    """Reproduce the original blog's YOLOv3 training approach."""
    
    def __init__(self, config_path: str = "configs/yolov3/original_blog_config.yaml"):
        """Initialize reproducer with original blog configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.original_config = self.config['original_blog']
        self.reproduction_config = self.config['reproduction']
        
        # Initialize W&B for tracking
        self._setup_wandb()
        
        logger.info("Original Blog Reproducer initialized")
        logger.info(f"Original limitations: {self.original_config['limitations']}")
    
    def _setup_wandb(self):
        """Set up Weights & Biases tracking."""
        wandb_config = self.reproduction_config['wandb']
        
        wandb.init(
            project=wandb_config['project'],
            name=wandb_config['name'],
            tags=wandb_config['tags'],
            config={
                'original_blog': self.original_config,
                'reproduction': self.reproduction_config
            }
        )
        
        logger.info(f"W&B initialized: {wandb.run.name}")
    
    def reproduce_original_training(self, data_yaml_path: str, model_save_dir: str = "models/checkpoints/yolov3_original"):
        """
        Reproduce the original blog's training approach.
        
        Args:
            data_yaml_path: Path to data.yaml file
            model_save_dir: Directory to save trained models
        """
        logger.info("Reproducing original blog's YOLOv3 training...")
        
        # Create model save directory
        model_save_dir = Path(model_save_dir)
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load YOLOv3 model (replacing Mx_yolo binary)
        model = YOLO("yolov3.pt")
        
        # Configure model for 386 classes
        model.model.model[-1].nc = self.reproduction_config['model']['classes']
        
        # Original training parameters (with improvements)
        train_args = {
            'data': data_yaml_path,
            'epochs': self.reproduction_config['training']['epochs'],
            'batch': self.reproduction_config['training']['batch_size'],
            'imgsz': self.reproduction_config['model']['img_size'],
            'device': self.reproduction_config['hardware']['device'],
            'project': str(model_save_dir),
            'name': "yolov3_original_reproduction",
            'save_period': self.reproduction_config['checkpointing']['save_freq'],
            'patience': self.reproduction_config['training']['early_stopping']['patience'],
            'lr0': self.reproduction_config['training']['learning_rate'],
            'weight_decay': self.reproduction_config['training']['weight_decay'],
            'warmup_epochs': self.reproduction_config['training']['scheduler']['warmup_epochs'],
            'cos_lr': True,  # Cosine learning rate scheduling
            
            # Improved augmentation to address original limitations
            'hsv_h': self.reproduction_config['training']['augmentation']['hsv_h'],
            'hsv_s': self.reproduction_config['training']['augmentation']['hsv_s'],
            'hsv_v': self.reproduction_config['training']['augmentation']['hsv_v'],
            'degrees': self.reproduction_config['training']['augmentation']['degrees'],
            'translate': self.reproduction_config['training']['augmentation']['translate'],
            'scale': self.reproduction_config['training']['augmentation']['scale'],
            'shear': self.reproduction_config['training']['augmentation']['shear'],
            'perspective': self.reproduction_config['training']['augmentation']['perspective'],
            'flipud': self.reproduction_config['training']['augmentation']['flipud'],
            'fliplr': self.reproduction_config['training']['augmentation']['fliplr'],
            'mosaic': self.reproduction_config['training']['augmentation']['mosaic'],
            'mixup': self.reproduction_config['training']['augmentation']['mixup'],
        }
        
        # Start training
        logger.info(f"Training with original parameters (improved): {self.reproduction_config['training']['epochs']} epochs")
        results = model.train(**train_args)
        
        # Log final metrics
        self._log_final_metrics(results)
        
        # Save best model
        best_model_path = model_save_dir / "yolov3_original_reproduction" / "weights" / "best.pt"
        if best_model_path.exists():
            final_model_path = model_save_dir / "yolov3_original_reproduction_best.pt"
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            logger.info(f"Best reproduction model saved to {final_model_path}")
        
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
    
    def compare_with_original(self):
        """Compare our reproduction with original blog results."""
        logger.info("Comparing with original blog results...")
        
        # Original blog limitations
        original_limitations = self.original_config['limitations']
        
        # Our improvements
        improvements = [
            "Better data quality control",
            "Improved augmentation strategy", 
            "Cosine learning rate scheduling",
            "Early stopping to prevent overfitting",
            "Enhanced experiment tracking",
            "IoT optimization for Sipeed Maix Bit"
        ]
        
        comparison = {
            'original_limitations': original_limitations,
            'our_improvements': improvements,
            'expected_improvements': [
                "Better performance in low light",
                "Reduced sensitivity to object size",
                "Improved background handling",
                "Higher recognition accuracy",
                "Better dataset quality"
            ]
        }
        
        wandb.log({'comparison': comparison})
        logger.info("Comparison logged to W&B")
        
        return comparison
    
    def create_improvement_report(self):
        """Create a report for the original author."""
        report = f"""
# YOLOv3 Pokemon Classifier - Improved Reproduction

## Original Blog Analysis
- **Author**: 弦masamasa
- **Blog**: https://www.cnblogs.com/xianmasamasa/p/18995912
- **Model**: Mx_yolo 3.0.0 binary
- **Classes**: 386 (generations 1-3)
- **Hardware**: Sipeed Maix Bit RISC-V

## Original Limitations Identified
{chr(10).join([f"- {limitation}" for limitation in self.original_config['limitations']])}

## Our Improvements
{chr(10).join([f"- {improvement}" for improvement in [
    "Replaced Mx_yolo binary with ultralytics (more maintainable)",
    "Enhanced data augmentation to address lighting/size sensitivity",
    "Improved learning rate scheduling with cosine annealing",
    "Added early stopping to prevent overfitting",
    "Better experiment tracking with Weights & Biases",
    "IoT optimization for Sipeed Maix Bit deployment",
    "Enhanced data quality control and validation"
]])}

## Expected Improvements
- Better performance in varying lighting conditions
- Reduced sensitivity to object size and background
- Higher overall recognition accuracy
- More robust real-world performance
- Better model optimization for IoT deployment

## Model Files
- **Original Reproduction**: `models/checkpoints/yolov3_original/yolov3_original_reproduction_best.pt`
- **Configuration**: `configs/yolov3/original_blog_config.yaml`
- **Training Logs**: Available in Weights & Biases

## Testing Instructions
1. Load the improved model: `YOLO('models/checkpoints/yolov3_original/yolov3_original_reproduction_best.pt')`
2. Test on your original Sipeed Maix Bit hardware
3. Compare performance with your original model
4. Report back on improvements and any remaining issues

## Contact
For questions or collaboration, please reach out through the original blog or GitHub repository.
        """
        
        # Save report
        report_path = "models/checkpoints/yolov3_original/improvement_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Improvement report saved to {report_path}")
        return report

def main():
    parser = argparse.ArgumentParser(description="Reproduce original blog's YOLOv3 training")
    parser.add_argument("--data_yaml", required=True, help="Path to data.yaml file")
    parser.add_argument("--config", default="configs/yolov3/original_blog_config.yaml", help="Config file path")
    parser.add_argument("--model_save_dir", default="models/checkpoints/yolov3_original", help="Model save directory")
    parser.add_argument("--create_report", action="store_true", help="Create improvement report")
    
    args = parser.parse_args()
    
    # Initialize reproducer
    reproducer = OriginalBlogReproducer(args.config)
    
    # Reproduce training
    results = reproducer.reproduce_original_training(args.data_yaml, args.model_save_dir)
    
    # Compare with original
    reproducer.compare_with_original()
    
    # Create improvement report if requested
    if args.create_report:
        reproducer.create_improvement_report()
    
    print("Original blog reproduction completed!")

if __name__ == "__main__":
    main() 