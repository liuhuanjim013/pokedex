#!/usr/bin/env python3
"""
W&B Integration for YOLO Training
Handles experiment tracking, metrics logging, and artifact management
"""

import logging
import wandb
from typing import Dict, Any, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WandBIntegration:
    """W&B integration for experiment tracking and visualization."""
    _instance = None
    
    def __new__(cls, config: Dict[str, Any], resume_id: Optional[str] = None):
        """Create or return singleton instance."""
        if cls._instance is None:
            cls._instance = super(WandBIntegration, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self, config: Dict[str, Any], resume_id: Optional[str] = None):
        """Initialize W&B integration with configuration."""
        if self.initialized:
            return
            
        self.config = config
        self.run = None
        try:
            self._init_experiment(resume_id)
            self.initialized = True
        except Exception as e:
            self.finish()  # Clean up on initialization error
            raise
    
    def _init_experiment(self, resume_id: Optional[str] = None):
        """Initialize W&B experiment."""
        # Base init kwargs
        init_kwargs = {
            'project': self.config['project'],
            'name': self.config['name'],
            'entity': self.config.get('entity'),
            'tags': self.config.get('tags', []),
            'config': self.config,
            'reinit': True
        }
        
        # Add resume ID if provided
        if resume_id:
            init_kwargs.update({
                'id': resume_id,
                'resume': 'must'
            })
        
        import os
        
        def try_init(mode: str = None) -> bool:
            """Try to initialize W&B with given mode."""
            try:
                # Clean up any existing run
                if wandb.run is not None:
                    wandb.finish()
                
                # Set or clear mode
                if mode:
                    os.environ['WANDB_MODE'] = mode
                elif 'WANDB_MODE' in os.environ:
                    del os.environ['WANDB_MODE']
                    
                # Try to initialize
                self.run = wandb.init(**init_kwargs)
                
                if self.run:
                    # Save run ID for future resume
                    run_id_file = Path("wandb_run_id.txt")
                    with open(run_id_file, "w") as f:
                        f.write(str(self.run.id))
                    logger.info(f"W&B initialized in {mode or 'online'} mode")
                    return True
                    
            except Exception as e:
                # Check for permission error
                if "permission denied" in str(e).lower() or "403" in str(e):
                    logger.error("W&B permission error. Please check your API key and entity access.")
                    logger.error("Try running: wandb login --relogin")
                    # Clean up run on permission error
                    if wandb.run is not None:
                        wandb.finish()
                else:
                    logger.error(f"Failed to initialize W&B in {mode or 'online'} mode: {e}")
                return False
            
            return False
        
        # First try online mode
        if try_init():
            return
            
        # If online fails, try offline mode
        if try_init('offline'):
            return
            
        # If both fail, raise error
        raise RuntimeError("Failed to initialize W&B in both online and offline modes")
    
    def log_metrics(self, epoch: int, train_loss: float, val_metrics: Dict[str, Any]):
        """
        Log training and validation metrics.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_metrics: Validation metrics dictionary
        """
        if not self.run:
            return
        
        try:
            # Prepare metrics for logging
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
            }
            
            # Add validation metrics
            if val_metrics:
                metrics.update({
                    'val_loss': val_metrics.get('loss', 0),
                    'val_map': val_metrics.get('mAP', 0),
                    'val_accuracy': val_metrics.get('accuracy', 0),
                    'val_precision': val_metrics.get('precision', 0),
                    'val_recall': val_metrics.get('recall', 0),
                })
            
            # Log to W&B
            self.run.log(metrics)
            
            logger.debug(f"Logged metrics for epoch {epoch}")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_artifacts(self, file_path: Path, artifact_name: str = None):
        """
        Upload file as artifact to W&B.
        
        Args:
            file_path: Path to file to upload
            artifact_name: Name for the artifact
        """
        if not self.run or not file_path.exists():
            return
        
        try:
            if artifact_name is None:
                artifact_name = file_path.name
            
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"Model checkpoint from {self.config['name']}"
            )
            
            artifact.add_file(str(file_path))
            self.run.log_artifact(artifact)
            
            logger.info(f"Uploaded artifact: {artifact_name}")
            
        except Exception as e:
            logger.error(f"Failed to upload artifact: {e}")
    
    def log_final_metrics(self, final_metrics: Dict[str, Any]):
        """
        Log final training metrics.
        
        Args:
            final_metrics: Final training metrics
        """
        if not self.run:
            return
        
        try:
            # Log final metrics
            self.run.log(final_metrics)
            
            # Mark run as finished
            self.run.finish()
            
            logger.info("Final metrics logged and run finished")
            
        except Exception as e:
            logger.error(f"Failed to log final metrics: {e}")
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log configuration to W&B.
        
        Args:
            config: Configuration dictionary
        """
        if not self.run:
            return
        
        try:
            self.run.config.update(config)
            logger.info("Configuration logged to W&B")
            
        except Exception as e:
            logger.error(f"Failed to log configuration: {e}")
    
    def log_model_summary(self, model_info: Dict[str, Any]):
        """
        Log model architecture summary.
        
        Args:
            model_info: Model information dictionary
        """
        if not self.run:
            return
        
        try:
            # Log model information
            self.run.log({
                'model/name': model_info.get('name', 'yolov3'),
                'model/classes': model_info.get('classes', 1025),
                'model/img_size': model_info.get('img_size', 416),
                'model/parameters': model_info.get('parameters', 0),
            })
            
            logger.info("Model summary logged to W&B")
            
        except Exception as e:
            logger.error(f"Failed to log model summary: {e}")
    
    def create_dashboard(self, baseline_run_id: str = None, improved_run_id: str = None):
        """
        Create comparison dashboard for baseline vs improved runs.
        
        Args:
            baseline_run_id: W&B run ID for baseline experiment
            improved_run_id: W&B run ID for improved experiment
        """
        try:
            # Create dashboard configuration
            dashboard_config = {
                'baseline_run': baseline_run_id,
                'improved_run': improved_run_id,
                'comparison_metrics': [
                    'train_loss',
                    'val_loss', 
                    'val_map',
                    'val_accuracy'
                ]
            }
            
            # Log dashboard configuration
            if self.run:
                self.run.log({'dashboard_config': dashboard_config})
            
            logger.info("Dashboard configuration created")
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
    
    def log_training_progress(self, epoch: int, total_epochs: int, current_loss: float):
        """
        Log training progress for monitoring.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            current_loss: Current training loss
        """
        if not self.run:
            return
        
        try:
            progress = epoch / total_epochs
            self.run.log({
                'training/progress': progress,
                'training/epoch': epoch,
                'training/total_epochs': total_epochs,
                'training/current_loss': current_loss,
            })
            
        except Exception as e:
            logger.error(f"Failed to log training progress: {e}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters for experiment tracking.
        
        Args:
            hyperparams: Hyperparameters dictionary
        """
        if not self.run:
            return
        
        try:
            self.run.config.update(hyperparams)
            logger.info("Hyperparameters logged to W&B")
            
        except Exception as e:
            logger.error(f"Failed to log hyperparameters: {e}")
    
    def finish(self):
        """Finish the W&B run."""
        if self.run:
            try:
                self.run.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.error(f"Failed to finish W&B run: {e}")
    
    def get_run_url(self) -> Optional[str]:
        """Get the URL for the current W&B run."""
        if self.run:
            return self.run.url
        return None 