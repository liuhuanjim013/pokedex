#!/usr/bin/env python3
"""
Checkpoint Manager for YOLO Training
Handles saving, loading, and managing training checkpoints
"""

import os
import glob
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages checkpoint saving, loading, and cleanup for YOLO training."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize checkpoint manager with configuration."""
        self.save_dir = Path(config['save_dir'])
        self.save_frequency = config['save_frequency']
        self.max_checkpoints = config['max_checkpoints']
        
        # Create save directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CheckpointManager initialized: {self.save_dir}")
        logger.info(f"Save frequency: {self.save_frequency}, Max checkpoints: {self.max_checkpoints}")
    
    def save_checkpoint(self, model: YOLO, epoch: int, metrics: Dict[str, Any], config: Dict[str, Any]):
        """
        Save checkpoint with model state and metadata.
        
        Args:
            model: YOLO model to save
            epoch: Current epoch number
            metrics: Training metrics
            config: Training configuration
        """
        try:
            # Create checkpoint data
            checkpoint_data = {
                'epoch': epoch,
                'model_state': model.model.state_dict(),
                'metrics': metrics,
                'config': config,
                'model_info': {
                    'name': model.model.name,
                    'classes': model.model.model[-1].nc,
                    'img_size': model.model.model[0].img_size
                }
            }
            
            # Save checkpoint
            checkpoint_path = self.save_dir / f"yolov3_epoch_{epoch:03d}.pt"
            torch.save(checkpoint_data, checkpoint_path)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_latest_checkpoint(self, model: YOLO) -> int:
        """
        Load the latest checkpoint and return the epoch number.
        
        Args:
            model: YOLO model to load checkpoint into
            
        Returns:
            Epoch number of loaded checkpoint, 0 if no checkpoint found
        """
        try:
            # Find all checkpoint files
            checkpoint_pattern = self.save_dir / "yolov3_epoch_*.pt"
            checkpoint_files = list(self.save_dir.glob("yolov3_epoch_*.pt"))
            
            if not checkpoint_files:
                logger.info("No checkpoints found, starting from epoch 0")
                return 0
            
            # Find the latest checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            
            # Load checkpoint
            checkpoint_data = torch.load(latest_checkpoint, map_location='cpu')
            
            # Load model state
            model.model.load_state_dict(checkpoint_data['model_state'])
            
            epoch = checkpoint_data['epoch']
            logger.info(f"Loaded checkpoint from epoch {epoch}: {latest_checkpoint}")
            
            return epoch + 1  # Start from next epoch
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return 0
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space."""
        try:
            # Find all checkpoint files
            checkpoint_files = list(self.save_dir.glob("yolov3_epoch_*.pt"))
            
            if len(checkpoint_files) <= self.max_checkpoints:
                return
            
            # Sort by modification time (oldest first)
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest checkpoints
            files_to_remove = checkpoint_files[:-self.max_checkpoints]
            
            for checkpoint_file in files_to_remove:
                checkpoint_file.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint_file}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about available checkpoints."""
        try:
            checkpoint_files = list(self.save_dir.glob("yolov3_epoch_*.pt"))
            
            if not checkpoint_files:
                return {'checkpoints': [], 'latest_epoch': 0}
            
            # Get checkpoint information
            checkpoint_info = []
            for checkpoint_file in checkpoint_files:
                try:
                    checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
                    info = {
                        'file': checkpoint_file.name,
                        'epoch': checkpoint_data['epoch'],
                        'size_mb': checkpoint_file.stat().st_size / (1024 * 1024),
                        'modified': checkpoint_file.stat().st_mtime
                    }
                    checkpoint_info.append(info)
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint info for {checkpoint_file}: {e}")
            
            # Sort by epoch
            checkpoint_info.sort(key=lambda x: x['epoch'])
            
            latest_epoch = max([info['epoch'] for info in checkpoint_info]) if checkpoint_info else 0
            
            return {
                'checkpoints': checkpoint_info,
                'latest_epoch': latest_epoch,
                'total_checkpoints': len(checkpoint_info)
            }
            
        except Exception as e:
            logger.error(f"Failed to get checkpoint info: {e}")
            return {'checkpoints': [], 'latest_epoch': 0}
    
    def backup_to_huggingface(self, checkpoint_path: Path, repo_name: str):
        """
        Backup checkpoint to Hugging Face Hub.
        
        Args:
            checkpoint_path: Path to checkpoint file
            repo_name: Hugging Face repository name
        """
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            
            # Upload checkpoint
            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=f"checkpoints/{checkpoint_path.name}",
                repo_id=repo_name,
                repo_type="model"
            )
            
            logger.info(f"Checkpoint backed up to Hugging Face: {checkpoint_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to backup checkpoint to Hugging Face: {e}")
    
    def validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Validate checkpoint file integrity.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            # Try to load checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Check required keys
            required_keys = ['epoch', 'model_state', 'metrics', 'config']
            for key in required_keys:
                if key not in checkpoint_data:
                    logger.error(f"Checkpoint missing required key: {key}")
                    return False
            
            # Check model state
            if not isinstance(checkpoint_data['model_state'], dict):
                logger.error("Invalid model state in checkpoint")
                return False
            
            logger.info(f"Checkpoint validation passed: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
            return False 