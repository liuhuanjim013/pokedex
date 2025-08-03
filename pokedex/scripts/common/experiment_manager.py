#!/usr/bin/env python3
"""
Experiment manager for Pokemon classifier project.
Manages different experiment types (YOLO, VLM, Hybrid).
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging
import subprocess
import sys
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentType(Enum):
    """Supported experiment types."""
    YOLOV3 = "yolov3"
    YOLOV8 = "yolov8"
    YOLOV9 = "yolov9"
    YOLOV10 = "yolov10"
    YOLO_NAS = "yolo_nas"
    CLIP = "clip"
    SMOLVM = "smolvm"
    MOBILEVLM = "mobilevlm"
    HYBRID = "hybrid"

class ExperimentManager:
    """Manage different experiment types for Pokemon classifier."""
    
    def __init__(self):
        """Initialize experiment manager."""
        self.project_root = Path.cwd()
        self.experiments = {
            ExperimentType.YOLOV3: {
                "config_dir": "configs/yolov3",
                "data_dir": "data/processed/yolov3",
                "model_dir": "models/checkpoints/yolov3",
                "trainer": "src/training/yolo/yolov3_trainer.py",
                "setup_script": "scripts/yolo/setup_yolov3_experiment.py",
                "requirements": "requirements/yolo_requirements.txt"
            },
            ExperimentType.YOLOV8: {
                "config_dir": "configs/yolov8",
                "data_dir": "data/processed/yolov8",
                "model_dir": "models/checkpoints/yolov8",
                "trainer": "src/training/yolo/yolov8_trainer.py",
                "setup_script": "scripts/yolo/setup_yolov8_experiment.py",
                "requirements": "requirements/yolo_requirements.txt"
            },
            ExperimentType.CLIP: {
                "config_dir": "configs/clip",
                "data_dir": "data/processed/clip",
                "model_dir": "models/checkpoints/clip",
                "trainer": "src/training/vlm/clip_trainer.py",
                "setup_script": "scripts/vlm/setup_clip_experiment.py",
                "requirements": "requirements/vlm_requirements.txt"
            },
            ExperimentType.SMOLVM: {
                "config_dir": "configs/smolvm",
                "data_dir": "data/processed/smolvm",
                "model_dir": "models/checkpoints/smolvm",
                "trainer": "src/training/vlm/smolvm_trainer.py",
                "setup_script": "scripts/vlm/setup_smolvm_experiment.py",
                "requirements": "requirements/vlm_requirements.txt"
            }
        }
    
    def list_experiments(self):
        """List all available experiments."""
        logger.info("Available experiments:")
        for exp_type in ExperimentType:
            if exp_type in self.experiments:
                config = self.experiments[exp_type]
                logger.info(f"  {exp_type.value}:")
                logger.info(f"    Config: {config['config_dir']}")
                logger.info(f"    Data: {config['data_dir']}")
                logger.info(f"    Model: {config['model_dir']}")
                logger.info(f"    Trainer: {config['trainer']}")
    
    def setup_experiment(self, experiment_type: ExperimentType, dataset_path: str = None, 
                        dataset_name: str = None, hf_token: str = None):
        """
        Set up a specific experiment.
        
        Args:
            experiment_type: Type of experiment to set up
            dataset_path: Path to raw dataset
            dataset_name: Hugging Face dataset name
            hf_token: Hugging Face token
        """
        if experiment_type not in self.experiments:
            logger.error(f"Experiment type {experiment_type.value} not supported")
            return False
        
        config = self.experiments[experiment_type]
        logger.info(f"Setting up {experiment_type.value} experiment...")
        
        # Install dependencies
        if not self._install_dependencies(config['requirements']):
            return False
        
        # Create directories
        self._create_directories(experiment_type)
        
        # Setup W&B
        if not self._setup_wandb():
            return False
        
        # Process dataset if provided
        if dataset_path:
            if not self._process_dataset(experiment_type, dataset_path):
                return False
            
            # Upload to Hugging Face if requested
            if dataset_name:
                if not self._upload_dataset(experiment_type, dataset_name, hf_token):
                    return False
        
        logger.info(f"{experiment_type.value} experiment setup completed!")
        return True
    
    def train_experiment(self, experiment_type: ExperimentType, data_path: str, 
                        config_path: str = None, model_save_dir: str = None):
        """
        Train a specific experiment.
        
        Args:
            experiment_type: Type of experiment to train
            data_path: Path to data (YAML for YOLO, dataset for VLM)
            config_path: Path to config file
            model_save_dir: Directory to save models
        """
        if experiment_type not in self.experiments:
            logger.error(f"Experiment type {experiment_type.value} not supported")
            return False
        
        config = self.experiments[experiment_type]
        logger.info(f"Training {experiment_type.value} experiment...")
        
        # Set default paths
        if not config_path:
            config_path = f"{config['config_dir']}/training_config.yaml"
        if not model_save_dir:
            model_save_dir = config['model_dir']
        
        # Run training
        cmd = [
            sys.executable, config['trainer'],
            "--data_yaml" if experiment_type.value.startswith('yolo') else "--dataset_path", data_path,
            "--config", config_path,
            "--model_save_dir", model_save_dir,
            "--evaluate"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"{experiment_type.value} training completed successfully")
            logger.info(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"{experiment_type.value} training failed: {e}")
            logger.error(e.stderr)
            return False
    
    def compare_experiments(self, experiment_types: List[ExperimentType]):
        """
        Compare multiple experiments.
        
        Args:
            experiment_types: List of experiment types to compare
        """
        logger.info("Comparing experiments...")
        
        results = {}
        for exp_type in experiment_types:
            if exp_type in self.experiments:
                config = self.experiments[exp_type]
                model_path = Path(config['model_dir']) / f"{exp_type.value}_pokemon_best.pt"
                
                if model_path.exists():
                    # Run evaluation
                    results[exp_type.value] = self._evaluate_model(exp_type, str(model_path))
                else:
                    logger.warning(f"No trained model found for {exp_type.value}")
        
        # Create comparison report
        self._create_comparison_report(results)
        
        return results
    
    def _install_dependencies(self, requirements_file: str):
        """Install experiment-specific dependencies."""
        logger.info(f"Installing dependencies from {requirements_file}")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], 
                         check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def _create_directories(self, experiment_type: ExperimentType):
        """Create experiment-specific directories."""
        config = self.experiments[experiment_type]
        
        directories = [
            config['data_dir'],
            config['model_dir'],
            f"logs/{experiment_type.value}",
            f"notebooks/{experiment_type.value}_experiments"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created: {directory}")
    
    def _setup_wandb(self):
        """Set up Weights & Biases."""
        logger.info("Setting up Weights & Biases...")
        
        try:
            import wandb
            wandb.login()
            return True
        except Exception as e:
            logger.error(f"W&B setup failed: {e}")
            return False
    
    def _process_dataset(self, experiment_type: ExperimentType, dataset_path: str):
        """Process dataset for specific experiment."""
        config = self.experiments[experiment_type]
        
        # Run experiment-specific setup script
        setup_script = config['setup_script']
        if Path(setup_script).exists():
            cmd = [
                sys.executable, setup_script,
                "--dataset_path", dataset_path,
                "--skip_upload"
            ]
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"Dataset processed for {experiment_type.value}")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Dataset processing failed: {e}")
                return False
        else:
            logger.warning(f"Setup script not found: {setup_script}")
            return False
    
    def _upload_dataset(self, experiment_type: ExperimentType, dataset_name: str, token: str = None):
        """Upload dataset to Hugging Face."""
        config = self.experiments[experiment_type]
        
        cmd = [
            sys.executable, "scripts/upload_dataset.py",
            "--processed_dir", config['data_dir'],
            "--dataset_name", dataset_name,
            "--yolo_format" if experiment_type.value.startswith('yolo') else ""
        ]
        
        if token:
            cmd.extend(["--token", token])
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Dataset uploaded for {experiment_type.value}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Dataset upload failed: {e}")
            return False
    
    def _evaluate_model(self, experiment_type: ExperimentType, model_path: str):
        """Evaluate a trained model."""
        # This would implement model-specific evaluation
        # For now, return placeholder results
        return {
            "accuracy": 0.75,
            "precision": 0.72,
            "recall": 0.78,
            "f1_score": 0.75
        }
    
    def _create_comparison_report(self, results: Dict[str, Any]):
        """Create comparison report for experiments."""
        report_path = "logs/experiment_comparison.md"
        
        with open(report_path, 'w') as f:
            f.write("# Pokemon Classifier Experiment Comparison\n\n")
            f.write("## Results Summary\n\n")
            
            for exp_name, metrics in results.items():
                f.write(f"### {exp_name.upper()}\n")
                for metric, value in metrics.items():
                    f.write(f"- {metric}: {value:.3f}\n")
                f.write("\n")
        
        logger.info(f"Comparison report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Manage Pokemon classifier experiments")
    parser.add_argument("--action", choices=["list", "setup", "train", "compare"], required=True,
                       help="Action to perform")
    parser.add_argument("--experiment", choices=[e.value for e in ExperimentType],
                       help="Experiment type")
    parser.add_argument("--dataset_path", help="Path to raw dataset")
    parser.add_argument("--dataset_name", help="Hugging Face dataset name")
    parser.add_argument("--hf_token", help="Hugging Face token")
    parser.add_argument("--data_path", help="Path to data for training")
    parser.add_argument("--config_path", help="Path to config file")
    parser.add_argument("--model_save_dir", help="Directory to save models")
    parser.add_argument("--experiments", nargs="+", choices=[e.value for e in ExperimentType],
                       help="Experiment types for comparison")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = ExperimentManager()
    
    if args.action == "list":
        manager.list_experiments()
    
    elif args.action == "setup":
        if not args.experiment:
            logger.error("Experiment type required for setup")
            return
        
        exp_type = ExperimentType(args.experiment)
        manager.setup_experiment(exp_type, args.dataset_path, args.dataset_name, args.hf_token)
    
    elif args.action == "train":
        if not args.experiment or not args.data_path:
            logger.error("Experiment type and data path required for training")
            return
        
        exp_type = ExperimentType(args.experiment)
        manager.train_experiment(exp_type, args.data_path, args.config_path, args.model_save_dir)
    
    elif args.action == "compare":
        if not args.experiments:
            logger.error("Experiment types required for comparison")
            return
        
        exp_types = [ExperimentType(exp) for exp in args.experiments]
        manager.compare_experiments(exp_types)

if __name__ == "__main__":
    main() 