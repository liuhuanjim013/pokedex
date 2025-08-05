#!/usr/bin/env python3
"""
YOLOv3 Improved Training Script (Enhanced Parameters)

This script implements enhanced YOLOv3 training to address the original blog's limitations.
It uses Google Colab for training and follows the centralized environment setup.

Improvements:
- Enhanced augmentation (rotation, shear, mosaic, mixup)
- Cosine learning rate scheduling with warmup
- Early stopping to prevent overfitting
- Larger batch size (32 vs 16)
- Longer training (200 epochs vs 100)
- Better regularization techniques
"""

import os
import sys
import argparse
from pathlib import Path
from google.colab import drive
import wandb
import yaml

# Add src to path for module imports
sys.path.append(str(Path(__file__).resolve().parents[2] / 'src'))
from training.yolo.trainer import YOLOTrainer
from evaluation.yolo.evaluator import YOLOEvaluator

def setup_google_drive():
    """Mount Google Drive and create directories for checkpoints and logs."""
    try:
        drive.mount('/content/drive')
        
        # Create directories for checkpoints and logs
        checkpoint_dir = '/content/drive/MyDrive/pokemon_yolo/checkpoints/improved'
        log_dir = '/content/drive/MyDrive/pokemon_yolo/logs/improved'
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        print("✅ Google Drive mounted successfully!")
        print(f"📁 Checkpoint directory: {checkpoint_dir}")
        print(f"📁 Log directory: {log_dir}")
        
        return checkpoint_dir, log_dir
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        raise

def verify_environment():
    """Verify GPU availability and dependencies."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available! Please enable GPU runtime in Colab.")
        
        print("🎯 System Check:")
        print(f"• Python version: {sys.version.split()[0]}")
        print(f"• PyTorch version: {torch.__version__}")
        print(f"• GPU available: {torch.cuda.get_device_name(0)}")
        print(f"• CUDA version: {torch.version.cuda}")
        
        print("\n✅ Environment verified successfully!")
    except Exception as e:
        print(f"❌ Environment verification failed: {e}")
        raise

def initialize_wandb(config_path: Path):
    """Initialize W&B with configuration from YAML file."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        wandb.init(
            project=config['wandb']['project'],
            name=config['wandb']['name'],
            entity=config['wandb']['entity'],
            tags=config['wandb']['tags'],
            config=config,
            resume=True  # Enable run resumption
        )
        
        print("✅ W&B initialized successfully!")
        print(f"📊 Dashboard: https://wandb.ai/{config['wandb']['entity']}/{config['wandb']['project']}")
        
        return config
    except Exception as e:
        print(f"❌ W&B initialization failed: {e}")
        raise

def train_improved(config_path: str, checkpoint_dir: str):
    """Execute improved training with enhanced parameters."""
    try:
        # Initialize trainer
        trainer = YOLOTrainer(config_path)
        
        # Check for existing checkpoints
        start_epoch = 0
        if os.path.exists(checkpoint_dir):
            checkpoint_files = sorted(os.listdir(checkpoint_dir))
            if checkpoint_files:
                latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
                print(f"\n📦 Found checkpoint: {latest_checkpoint}")
                start_epoch = trainer.load_checkpoint()
                print(f"✅ Resuming from epoch {start_epoch}")
            else:
                print("\n📋 No existing checkpoints found. Starting fresh training.")
        
        print("\n📈 Improvements over baseline:")
        print("• Enhanced augmentation (rotation, shear, mosaic, mixup)")
        print("• Cosine learning rate scheduling with warmup")
        print("• Early stopping (patience=10)")
        print("• Larger batch size (32 vs 16)")
        print("• Longer training (200 epochs vs 100)")
        print("• Better regularization techniques")
        
        # Start training
        print("\n🚀 Starting improved training...")
        results = trainer.train(start_epoch=start_epoch)
        
        # Save final model
        final_model_path = os.path.join(checkpoint_dir, "yolov3_improved_final.pt")
        trainer.save_model(final_model_path)
        
        # Log final artifacts to W&B
        wandb.save(final_model_path)
        
        return results, trainer
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

def evaluate_model(trainer, results):
    """Evaluate the trained model and compare with baseline."""
    try:
        # Initialize evaluator
        evaluator = YOLOEvaluator(trainer.model, trainer.config)
        
        # Run evaluation
        test_data = "liuhuanjim013/pokemon-yolo-1025"
        evaluation_results = evaluator.evaluate_model(test_data)
        
        print("\n📊 Final Results:")
        for metric, value in results.items():
            print(f"• {metric}: {value:.4f}")
        
        print("\n📈 Improvements Over Baseline:")
        print("1. Enhanced Augmentation:")
        print("   • Added rotation (±10°)")
        print("   • Added translation (±20%)")
        print("   • Added shear (±2°)")
        print("   • Added mosaic (prob=1.0)")
        print("   • Added mixup (prob=0.1)")
        
        print("\n2. Training Enhancements:")
        print("   • Cosine learning rate scheduling")
        print("   • 5 epochs warmup")
        print("   • Early stopping (patience=10)")
        print("   • Larger batch size (32)")
        print("   • Longer training (200 epochs)")
        
        print("\n3. Expected Benefits:")
        print("   • Better handling of lighting variations")
        print("   • Improved size/scale robustness")
        print("   • Reduced background interference")
        print("   • Higher overall accuracy")
        print("   • Better generalization")
        
        # Log evaluation results to W&B
        wandb.log({"final_evaluation": evaluation_results})
        
        return evaluation_results
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise

def cleanup_resources():
    """Clean up resources and unmount Google Drive."""
    try:
        # Finish W&B run
        wandb.finish()
        print("✅ W&B run completed and synced")
        
        # Unmount Google Drive
        drive.flush_and_unmount()
        print("✅ Google Drive unmounted safely")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        print("⚠️ Please manually unmount Google Drive and close W&B run")

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv3 with improved parameters")
    parser.add_argument("--config", type=str, default="configs/yolov3/improved_config.yaml",
                      help="Path to configuration file")
    args = parser.parse_args()
    
    try:
        # Setup and verification
        checkpoint_dir, log_dir = setup_google_drive()
        verify_environment()
        
        # Initialize W&B and load config
        config = initialize_wandb(Path(args.config))
        
        # Train model
        results, trainer = train_improved(args.config, checkpoint_dir)
        
        # Evaluate model
        evaluation_results = evaluate_model(trainer, results)
        
        # Cleanup
        cleanup_resources()
        
        print("\n✨ Improved training completed successfully!")
        print("\n🎯 Next Steps:")
        print("1. Check W&B dashboard for training visualizations")
        print("2. Review saved checkpoints in Google Drive")
        print("3. Compare performance with baseline results")
        print("4. Consider further improvements based on results")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user!")
        print("Latest checkpoint was saved automatically.")
        print("You can resume training by running this script again.")
        cleanup_resources()
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        cleanup_resources()
        raise

if __name__ == "__main__":
    main()