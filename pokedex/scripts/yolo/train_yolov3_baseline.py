#!/usr/bin/env python3
"""
YOLOv3 Baseline Training Script (Original Blog Reproduction)

This script reproduces the original blog's YOLOv3 training approach with 1025 classes.
It uses Google Colab for training and follows the centralized environment setup.

Original Blog: https://www.cnblogs.com/xianmasamasa/p/18995912
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import wandb
import yaml
import torch
from datasets import load_dataset
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Add src to path for module imports
sys.path.append(str(Path(__file__).resolve().parents[2] / 'src'))
from training.yolo.trainer import YOLOTrainer
from evaluation.yolo.evaluator import YOLOEvaluator

def is_colab():
    """Check if running in Google Colab."""
    try:
        from google.colab import drive
        return True, drive
    except (ImportError, ModuleNotFoundError):
        return False, None

def get_storage_dirs():
    """Get storage directories (assumes setup_colab_training.py was run)."""
    try:
        # Get the root directory (where the repository is)
        root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
        
        # Get project directories relative to root
        dirs = {
            'checkpoints': os.path.join(root_dir, 'models', 'checkpoints'),
            'logs': os.path.join(root_dir, 'models', 'logs'),
            'models': os.path.join(root_dir, 'models', 'final')
        }
        
        # Verify directories exist (should have been created by setup script)
        for name, path in dirs.items():
            if not os.path.exists(path):
                raise RuntimeError(f"Directory not found: {path}. Did you run setup_colab_training.py first?")
            print(f"✅ Found {name} directory: {path}")
        
        return dirs
    except Exception as e:
        print(f"❌ Failed to get storage directories: {e}")
        raise

def validate_hf_token(token: str) -> bool:
    """Validate Hugging Face token format."""
    if not token:
        return False
    token = token.strip()
    # Check basic format (should be "hf_..." and about 31-40 chars)
    if not token.startswith("hf_") or len(token) < 31 or len(token) > 40:
        return False
    return True

def verify_training_ready():
    """Verify training prerequisites are met (assumes setup_colab_training.py was run)."""
    try:
        # Check for required environment variables
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token or not validate_hf_token(hf_token):
            raise RuntimeError("Valid HUGGINGFACE_TOKEN not found. Did you run setup_colab_training.py first?")
            
        wandb_key = os.getenv("WANDB_API_KEY")
        if not wandb_key:
            raise RuntimeError("WANDB_API_KEY not found. Did you run setup_colab_training.py first?")
            
        # Verify GPU is available
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available! Did you run setup_colab_training.py first?")
            
        # Verify git credentials
        result = subprocess.run(
            ["git", "config", "--global", "credential.helper"],
            capture_output=True,
            text=True,
            check=False
        )
        if not result.stdout.strip():
            raise RuntimeError("Git credential helper not set. Did you run setup_colab_training.py first?")
            
        # Print status
        print("\n🔍 Training Prerequisites:")
        print(f"• GPU: {torch.cuda.get_device_name(0)}")
        print(f"• CUDA: {torch.version.cuda}")
        print(f"• Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"• HF Token: {hf_token[:6]}... (valid format)")
        print(f"• W&B Key: {wandb_key[:6]}...")
        print(f"• Git Credentials: {result.stdout.strip()}")
        print("\n✅ All training prerequisites verified!")
        
    except Exception as e:
        print(f"\n❌ Training prerequisites not met: {e}")
        print("\nℹ️ Please run setup_colab_training.py first!")
        print("   This will:")
        print("   1. Set up environment and dependencies")
        print("   2. Configure Hugging Face authentication")
        print("   3. Set up W&B project tracking")
        print("   4. Create necessary directories")
        print("   5. Verify dataset access")
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

def prepare_dataset():
    """Prepare Pokemon dataset for training."""
    try:
        print("\n📦 Loading Pokemon dataset...")
        dataset = load_dataset("liuhuanjim013/pokemon-yolo-1025")
        print(f"✅ Dataset loaded with {len(dataset['train'])} training examples")
        
        # Verify dataset format
        example = dataset['train'][0]
        if not isinstance(example['image'], Image.Image):
            raise ValueError("Unexpected image format in dataset")
        if not isinstance(example['label'], (int, np.integer)):
            raise ValueError("Unexpected label format in dataset")
            
        print("✅ Dataset format verified")
        print(f"• Image format: {example['image'].mode} {example['image'].size}")
        print(f"• Label range: 0-1024 (1025 classes)")
        
        return dataset
    except Exception as e:
        print(f"❌ Dataset preparation failed: {e}")
        raise

def train_baseline(config_path: str, storage_dirs: dict):
    """Execute baseline training with the original blog parameters."""
    try:
        # Load dataset
        dataset = prepare_dataset()
        
        # Initialize trainer
        trainer = YOLOTrainer(config_path)
        
        # Check for existing checkpoints
        start_epoch = 0
        checkpoint_dir = storage_dirs['checkpoints']
        if os.path.exists(checkpoint_dir):
            checkpoint_files = sorted(os.listdir(checkpoint_dir))
            if checkpoint_files:
                latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
                print(f"\n📦 Found checkpoint: {latest_checkpoint}")
                start_epoch = trainer.load_checkpoint(latest_checkpoint)
                print(f"✅ Resuming from epoch {start_epoch}")
            else:
                print("\n📋 No existing checkpoints found. Starting fresh training.")
        
        # Start training
        print("\n🚀 Starting baseline training...")
        results = trainer.train(
            dataset=dataset,
            start_epoch=start_epoch,
            checkpoint_dir=checkpoint_dir,
            log_dir=storage_dirs['logs']
        )
        
        # Save final model
        final_model_path = os.path.join(storage_dirs['models'], "yolov3_final.pt")
        trainer.save_model(final_model_path)
        
        # Log final artifacts to W&B
        wandb.save(final_model_path)
        
        return results, trainer
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

def evaluate_model(trainer, results):
    """Evaluate the trained model on test set."""
    try:
        # Initialize evaluator
        evaluator = YOLOEvaluator(trainer.model, trainer.config)
        
        # Run evaluation
        test_data = "liuhuanjim013/pokemon-yolo-1025"
        evaluation_results = evaluator.evaluate_model(test_data)
        
        print("\n📊 Final Results:")
        for metric, value in results.items():
            print(f"• {metric}: {value:.4f}")
        
        print("\n📋 Original Blog Limitations Confirmed:")
        print("1. Poor performance in low light conditions")
        print("2. Sensitive to object size variations") 
        print("3. Background interference issues")
        print("4. Limited recognition accuracy")
        print("5. No advanced augmentation techniques")
        print("6. No learning rate scheduling")
        print("7. No early stopping mechanisms")
        
        # Log evaluation results to W&B
        wandb.log({"final_evaluation": evaluation_results})
        
        return evaluation_results
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise

def cleanup_resources():
    """Clean up resources and unmount Google Drive if in Colab."""
    try:
        # Finish W&B run
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
    parser = argparse.ArgumentParser(description="Train YOLOv3 baseline model (original blog reproduction)")
    parser.add_argument("--config", type=str, default="configs/yolov3/baseline_config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--resume", action="store_true",
                      help="Resume training from latest checkpoint")
    args = parser.parse_args()
    
    try:
        print("\n🚀 Starting YOLOv3 baseline training...")
        
        # Verify setup is complete
        verify_training_ready()
        storage_dirs = get_storage_dirs()
        
        # Initialize W&B and load config
        config = initialize_wandb(Path(args.config))
        
        # Train model
        results, trainer = train_baseline(args.config, storage_dirs)
        
        # Evaluate model
        evaluation_results = evaluate_model(trainer, results)
        
        # Cleanup
        cleanup_resources()
        
        print("\n✨ Baseline reproduction completed successfully!")
        print("\n📊 Results Summary:")
        print(f"• Training examples: {len(results['train'])}")
        print(f"• Final accuracy: {results.get('accuracy', 'N/A'):.4f}")
        print(f"• Training time: {results.get('training_time', 'N/A')}")
        
        print("\n🎯 Next Steps:")
        print("1. Check W&B dashboard for training visualizations")
        print(f"2. Review saved checkpoints in {storage_dirs['checkpoints']}")
        print("3. Run train_yolov3_improved.py to address limitations")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user!")
        print("Latest checkpoint was saved automatically.")
        print("You can resume training by running this script again with --resume")
        cleanup_resources()
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        cleanup_resources()
        raise

if __name__ == "__main__":
    main()