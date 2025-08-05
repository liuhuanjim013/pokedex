#!/usr/bin/env python3
"""
Resume YOLO Training Script

This script allows resuming YOLO training from the latest saved checkpoint.
It supports both baseline and improved training configurations.
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

def setup_google_drive():
    """Mount Google Drive and locate checkpoint directory."""
    try:
        drive.mount('/content/drive')
        base_dir = '/content/drive/MyDrive/pokemon_yolo/checkpoints'
        return base_dir
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        raise

def find_latest_checkpoint(checkpoint_dir: str):
    """Find the latest checkpoint in the specified directory."""
    try:
        if not os.path.exists(checkpoint_dir):
            print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
            return None
        
        checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
        if not checkpoint_files:
            print("❌ No checkpoints found")
            return None
        
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        print(f"✅ Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    except Exception as e:
        print(f"❌ Error finding checkpoint: {e}")
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
            tags=[*config['wandb']['tags'], 'resumed'],
            config=config,
            resume=True  # Enable run resumption
        )
        
        print("✅ W&B initialized successfully!")
        print(f"📊 Dashboard: https://wandb.ai/{config['wandb']['entity']}/{config['wandb']['project']}")
        
        return config
    except Exception as e:
        print(f"❌ W&B initialization failed: {e}")
        raise

def resume_training(config_path: str, checkpoint_dir: str):
    """Resume training from the latest checkpoint."""
    try:
        # Initialize trainer
        trainer = YOLOTrainer(config_path)
        
        # Find and load latest checkpoint
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is None:
            print("❌ Cannot resume training without checkpoint")
            return
        
        # Load checkpoint and resume training
        start_epoch = trainer.load_checkpoint()
        print(f"✅ Resuming from epoch {start_epoch}")
        
        # Continue training
        print("\n🚀 Resuming training...")
        results = trainer.train(start_epoch=start_epoch)
        
        # Save final model
        final_model_path = os.path.join(checkpoint_dir, "yolov3_final.pt")
        trainer.save_model(final_model_path)
        
        # Log final artifacts to W&B
        wandb.save(final_model_path)
        
        return results
    except Exception as e:
        print(f"❌ Training resumption failed: {e}")
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
    parser = argparse.ArgumentParser(description="Resume YOLO training from checkpoint")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to configuration file (baseline or improved)")
    parser.add_argument("--experiment", type=str, choices=['baseline', 'improved'],
                      required=True, help="Which experiment to resume")
    args = parser.parse_args()
    
    try:
        # Setup
        base_checkpoint_dir = setup_google_drive()
        checkpoint_dir = os.path.join(base_checkpoint_dir, args.experiment)
        
        # Initialize W&B and load config
        config = initialize_wandb(Path(args.config))
        
        # Resume training
        results = resume_training(args.config, checkpoint_dir)
        
        # Cleanup
        cleanup_resources()
        
        print("\n✨ Training resumed and completed successfully!")
        print("\n🎯 Next Steps:")
        print("1. Check W&B dashboard for continued training progress")
        print("2. Review updated checkpoints in Google Drive")
        print("3. Evaluate final model performance")
        
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