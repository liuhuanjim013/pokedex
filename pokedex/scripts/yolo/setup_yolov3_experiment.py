#!/usr/bin/env python3
"""
Setup script for YOLOv3 Pokemon classifier experiment.
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

class YOLOv3ExperimentSetup:
    """Setup environment and data for YOLOv3 Pokemon classifier experiment."""
    
    def __init__(self, config_path: str = "configs/yolov3/data_config.yaml"):
        """Initialize setup with YOLOv3-specific configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path.cwd()
        logger.info(f"Project root: {self.project_root}")
    
    def setup_environment(self):
        """Set up Python environment and install YOLO dependencies."""
        logger.info("Setting up YOLO environment...")
        
        # Install YOLO-specific dependencies
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements/yolo_requirements.txt"], 
                         check=True, capture_output=True)
            logger.info("YOLO dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install YOLO dependencies: {e}")
            return False
        
        return True
    
    def setup_directories(self):
        """Create YOLOv3-specific directories."""
        logger.info("Creating YOLOv3 project directories...")
        
        directories = [
            "data/raw",
            "data/processed/yolov3",
            "data/splits/yolov3",
            "models/checkpoints/yolov3",
            "models/final/yolov3",
            "models/compressed/yolov3",
            "notebooks/yolo_experiments",
            "logs/yolov3"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created: {directory}")
    
    def process_dataset(self, dataset_path: str):
        """
        Process the 900MB gen1-3 dataset for YOLOv3.
        
        Args:
            dataset_path: Path to the downloaded dataset
        """
        logger.info(f"Processing dataset for YOLOv3: {dataset_path}")
        
        # Run YOLOv3-specific preprocessing
        cmd = [
            sys.executable, "src/data/preprocessing.py",
            "--dataset_path", dataset_path,
            "--config", "configs/yolov3/data_config.yaml",
            "--create_yolo_dataset"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("YOLOv3 dataset processing completed successfully")
            logger.info(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"YOLOv3 dataset processing failed: {e}")
            logger.error(e.stderr)
            return False
    
    def create_data_yaml(self, dataset_path: str = "data/processed/yolov3/yolo_dataset"):
        """Create data.yaml file for YOLOv3 training."""
        from src.training.yolo.yolov3_trainer import create_data_yaml
        
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
        output_path = "data/processed/yolov3/data.yaml"
        create_data_yaml(str(dataset_path), output_path, num_classes)
        
        logger.info(f"Created YOLOv3 data.yaml with {num_classes} classes")
        return True
    
    def upload_to_huggingface(self, dataset_name: str, token: str = None):
        """Upload processed dataset to Hugging Face."""
        logger.info(f"Uploading YOLOv3 dataset to Hugging Face: {dataset_name}")
        
        cmd = [
            sys.executable, "scripts/upload_dataset.py",
            "--processed_dir", "data/processed/yolov3/yolo_dataset",
            "--dataset_name", dataset_name,
            "--yolo_format"
        ]
        
        if token:
            cmd.extend(["--token", token])
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("YOLOv3 dataset uploaded successfully")
            logger.info(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"YOLOv3 upload failed: {e}")
            logger.error(e.stderr)
            return False
    
    def setup_wandb(self):
        """Set up Weights & Biases for YOLOv3 experiment."""
        logger.info("Setting up Weights & Biases for YOLOv3...")
        
        try:
            import wandb
            # This will prompt for login if not already logged in
            wandb.login()
            logger.info("W&B setup completed for YOLOv3")
            return True
        except Exception as e:
            logger.error(f"W&B setup failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Setup YOLOv3 Pokemon classifier experiment")
    parser.add_argument("--dataset_path", help="Path to raw dataset")
    parser.add_argument("--dataset_name", help="Hugging Face dataset name")
    parser.add_argument("--hf_token", help="Hugging Face token")
    parser.add_argument("--skip_upload", action="store_true", help="Skip Hugging Face upload")
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = YOLOv3ExperimentSetup()
    
    # Setup environment
    if not setup.setup_environment():
        logger.error("YOLOv3 environment setup failed")
        return
    
    # Create directories
    setup.setup_directories()
    
    # Setup W&B
    setup.setup_wandb()
    
    # Process dataset if provided
    if args.dataset_path:
        if not setup.process_dataset(args.dataset_path):
            logger.error("YOLOv3 dataset processing failed")
            return
        
        # Create data.yaml
        if not setup.create_data_yaml():
            logger.error("Failed to create YOLOv3 data.yaml")
            return
        
        # Upload to Hugging Face if requested
        if not args.skip_upload and args.dataset_name:
            if not setup.upload_to_huggingface(args.dataset_name, args.hf_token):
                logger.error("YOLOv3 dataset upload failed")
                return
    
    logger.info("YOLOv3 experiment setup completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Train YOLOv3: python src/training/yolo/yolov3_trainer.py --data_yaml data/processed/yolov3/data.yaml")
    logger.info("2. Monitor training: wandb login && wandb dashboard")

if __name__ == "__main__":
    main() 