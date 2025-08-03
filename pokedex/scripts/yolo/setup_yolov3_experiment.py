#!/usr/bin/env python3
"""
Setup script for Pokemon classifier YOLOv3 experiments.
"""

import os
import yaml
import argparse
from pathlib import Path
import logging
import subprocess
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PokemonExperimentSetup:
    """Setup environment and data for Pokemon classifier experiments."""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """Initialize setup with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path.cwd()
        logger.info(f"Project root: {self.project_root}")
    
    def setup_environment(self):
        """Set up Python environment and install dependencies."""
        logger.info("Setting up Python environment...")
        
        # Install dependencies
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True, capture_output=True)
            logger.info("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
        
        return True
    
    def setup_directories(self):
        """Create necessary directories."""
        logger.info("Creating project directories...")
        
        directories = [
            "data/raw",
            "data/processed",
            "data/splits",
            "models/checkpoints",
            "models/final",
            "models/compressed",
            "models/configs",
            "notebooks/exploration",
            "notebooks/experiments",
            "notebooks/deployment",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created: {directory}")
    
    def process_dataset(self, dataset_path: str):
        """
        Process the 900MB gen1-3 dataset.
        
        Args:
            dataset_path: Path to the downloaded dataset
        """
        logger.info(f"Processing dataset: {dataset_path}")
        
        # Run preprocessing
        cmd = [
            sys.executable, "src/data/preprocessing.py",
            "--dataset_path", dataset_path,
            "--config", "configs/data_config.yaml",
            "--create_yolo_dataset"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Dataset processing completed successfully")
            logger.info(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Dataset processing failed: {e}")
            logger.error(e.stderr)
            return False
    
    def create_data_yaml(self, dataset_path: str = "data/processed/yolo_dataset"):
        """Create data.yaml file for YOLO training."""
        from src.training.yolo_trainer import create_data_yaml
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path}")
            return False
        
        # Count classes
        classes_file = dataset_path / "classes.txt"
        if not classes_file.exists():
            logger.error("Classes file not found")
            return False
        
        with open(classes_file, 'r') as f:
            num_classes = len(f.readlines())
        
        # Create data.yaml
        output_path = "data/processed/data.yaml"
        create_data_yaml(str(dataset_path), output_path, num_classes)
        
        logger.info(f"Created data.yaml with {num_classes} classes")
        return True
    
    def upload_to_huggingface(self, dataset_name: str, token: str = None):
        """Upload processed dataset to Hugging Face."""
        logger.info(f"Uploading dataset to Hugging Face: {dataset_name}")
        
        cmd = [
            sys.executable, "scripts/upload_dataset.py",
            "--processed_dir", "data/processed/yolo_dataset",
            "--dataset_name", dataset_name,
            "--yolo_format"
        ]
        
        if token:
            cmd.extend(["--token", token])
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Dataset uploaded successfully")
            logger.info(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Upload failed: {e}")
            logger.error(e.stderr)
            return False
    
    def setup_wandb(self):
        """Set up Weights & Biases."""
        logger.info("Setting up Weights & Biases...")
        
        try:
            import wandb
            # This will prompt for login if not already logged in
            wandb.login()
            logger.info("W&B setup completed")
            return True
        except Exception as e:
            logger.error(f"W&B setup failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Setup Pokemon classifier experiment")
    parser.add_argument("--dataset_path", help="Path to raw dataset")
    parser.add_argument("--dataset_name", help="Hugging Face dataset name")
    parser.add_argument("--hf_token", help="Hugging Face token")
    parser.add_argument("--skip_upload", action="store_true", help="Skip Hugging Face upload")
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = PokemonExperimentSetup()
    
    # Setup environment
    if not setup.setup_environment():
        logger.error("Environment setup failed")
        return
    
    # Create directories
    setup.setup_directories()
    
    # Setup W&B
    setup.setup_wandb()
    
    # Process dataset if provided
    if args.dataset_path:
        if not setup.process_dataset(args.dataset_path):
            logger.error("Dataset processing failed")
            return
        
        # Create data.yaml
        if not setup.create_data_yaml():
            logger.error("Failed to create data.yaml")
            return
        
        # Upload to Hugging Face if requested
        if not args.skip_upload and args.dataset_name:
            if not setup.upload_to_huggingface(args.dataset_name, args.hf_token):
                logger.error("Dataset upload failed")
                return
    
    logger.info("Setup completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Train model: python src/training/yolo_trainer.py --data_yaml data/processed/data.yaml")
    logger.info("2. Monitor training: wandb login && wandb dashboard")

if __name__ == "__main__":
    main() 