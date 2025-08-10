#!/usr/bin/env python3
"""Test W&B step counting behavior during training resumption."""

import os
import json
import yaml
import shutil
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
import wandb
from datetime import datetime

# Add src to path for module imports
import sys
from pathlib import Path
src_path = str(Path(__file__).resolve().parents[1] / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)
print(f"Added to Python path: {src_path}")

from training.yolo.trainer import YOLOTrainer
from training.yolo.wandb_integration import WandBIntegration

class TestWandBStepCounting(unittest.TestCase):
    """Test W&B step counting during training resumption."""
    
    def setUp(self):
        """Set up test environment."""
        # Reset singleton instance
        WandBIntegration._instance = None
        
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test config
        self.config = {
            'wandb': {
                'project': 'test-project',
                'name': 'test-run',
                'entity': 'test-entity'
            },
            'training': {
                'epochs': 10,
                'batch_size': 16,
                'learning_rate': 0.001
            },
            'checkpoint': {
                'save_dir': os.path.join(self.test_dir, 'checkpoints'),
                'save_frequency': 5,  # How often to save checkpoints (epochs)
                'max_checkpoints': 3  # Maximum number of checkpoints to keep
            },
            'model': {
                'name': 'yolov3',
                'weights': 'yolov3.pt',
                'classes': 1025,
                'img_size': 416,
                'pretrained': True,
                'yaml': 'models/configs/yolov3.yaml'
            },
            'data': {
                'path': 'test_dataset'
            },
            'training': {
                'epochs': 10,
                'batch_size': 16,
                'learning_rate': 0.001,
                'weight_decay': 0.0005,
                'scheduler': 'cosine',
                'warmup_epochs': 3,
                'early_stopping': 'none',
                'augmentation': {
                    'hsv_h': 0.015,
                    'hsv_s': 0.7,
                    'hsv_v': 0.4,
                    'degrees': 0.0,
                    'translate': 0.1,
                    'scale': 0.5,
                    'shear': 0.0,
                    'perspective': 0.0,
                    'flipud': 0.0,
                    'fliplr': 0.5,
                    'mosaic': 1.0,
                    'mixup': 0.0
                }
            }
        }
        
        # Write config to YAML file
        self.config_path = os.path.join(self.test_dir, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        # Create test directories
        self.checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock wandb.init and run
        self.mock_run = MagicMock()
        self.mock_run.id = 'test_run_id'
        self.mock_run.step = 0
        
        # Create test checkpoint and metadata
        self.checkpoint_path = self.checkpoint_dir / 'checkpoint_epoch_5.pt'
        self.meta_path = self.checkpoint_path.with_suffix('.json')
        self.checkpoint_metadata = {
            'wandb_run_id': self.mock_run.id,
            'saved_epoch': 5,
            'actual_epoch': 7,  # Simulates crash at epoch 7
            'last_wandb_step': 700,  # 100 steps per epoch
            'timestamp': datetime.now().isoformat()
        }
        
        # Write test metadata
        with open(self.meta_path, 'w') as f:
            json.dump(self.checkpoint_metadata, f)
            
        # Create empty checkpoint file
        self.checkpoint_path.touch()
    
    def tearDown(self):
        """Clean up test environment."""
        # Clean up test directory and all its contents
        shutil.rmtree(self.test_dir, ignore_errors=True)
            
        # Clean up wandb run ID file
        run_id_file = Path('wandb_run_id.txt')
        if run_id_file.exists():
            run_id_file.unlink()
            
        # Reset environment
        if 'WANDB_MODE' in os.environ:
            del os.environ['WANDB_MODE']
    
    @patch('wandb.init')
    @patch('training.yolo.trainer.YOLOTrainer._setup_model')
    def test_resume_step_counting(self, mock_setup_model, mock_init):
        """Test that W&B step counting resumes correctly."""
        # Set up mock wandb.init to return our mock run
        mock_init.return_value = self.mock_run
        
        # Initialize trainer with resume ID
        trainer = YOLOTrainer(self.config_path, resume_id=self.mock_run.id)
        
        # Mock model and its train method
        mock_model = MagicMock()
        mock_model.train.return_value = {
            'epochs': 10,
            'metrics/mAP50(B)': 0.85,
            'metrics/accuracy': 0.90,
            'train/box_loss': 0.15
        }
        trainer.model = mock_model
        
        # Verify wandb.init was called with correct resume args
        mock_init.assert_called_once()
        init_kwargs = mock_init.call_args[1]
        self.assertEqual(init_kwargs.get('id'), self.mock_run.id)
        self.assertEqual(init_kwargs.get('resume'), 'must')
        
        # Load checkpoint and verify step counting
        results = trainer.train(start_epoch=5)  # Resume from epoch 5
        
        # Verify results contain correct epoch info
        self.assertEqual(results['start_epoch'], 5)
        self.assertEqual(results['end_epoch'], 15)  # 5 + 10 epochs
        self.assertEqual(results['global_step'], self.mock_run.step)
    
    @patch('wandb.init')
    @patch('training.yolo.trainer.YOLOTrainer._setup_model')
    def test_resume_with_step_offset(self, mock_setup_model, mock_init):
        """Test that W&B logging continues from last step."""
        # Set up mock wandb run with existing steps
        self.mock_run.step = self.checkpoint_metadata['last_wandb_step']
        mock_init.return_value = self.mock_run
        
        # Initialize trainer and resume
        trainer = YOLOTrainer(self.config_path, resume_id=self.mock_run.id)
        
        # Mock model and its train method
        mock_model = MagicMock()
        mock_model.train.return_value = {
            'epochs': 3,  # Train for 3 more epochs
            'metrics/mAP50(B)': 0.85,
            'metrics/accuracy': 0.90,
            'train/box_loss': 0.15
        }
        trainer.model = mock_model
        
        # Train and verify step counting
        results = trainer.train(start_epoch=5)  # Resume from checkpoint epoch
        
        # Verify W&B logging starts after last logged step
        self.mock_run.log.assert_called()
        first_log = self.mock_run.log.call_args_list[0]
        step = first_log[1].get('step', None)
        if step is not None:
            self.assertGreaterEqual(step, self.checkpoint_metadata['last_wandb_step'])
        results = trainer.train(start_epoch=5)
        
        # Verify wandb.log was called with correct step offset
        log_calls = self.mock_run.log.call_args_list
        for call in log_calls:
            step = call[1].get('step', None)
            if step is not None:
                self.assertGreaterEqual(step, self.checkpoint_metadata['last_wandb_step'])
    
    @patch('wandb.init')
    @patch('training.yolo.trainer.YOLOTrainer._setup_model')
    def test_resume_with_crash_recovery(self, mock_setup_model, mock_init):
        """Test resuming when crash happened after last checkpoint."""
        # Set up mock wandb run
        mock_init.return_value = self.mock_run
        
        # Initialize trainer
        trainer = YOLOTrainer(self.config_path, resume_id=self.mock_run.id)
        
        # Mock model and its train method
        mock_model = MagicMock()
        mock_model.train.return_value = {
            'epochs': 10,
            'metrics/mAP50(B)': 0.85,
            'metrics/accuracy': 0.90,
            'train/box_loss': 0.15
        }
        trainer.model = mock_model
        
        # Load checkpoint that crashed at epoch 7
        results = trainer.train(start_epoch=7)  # Resume from actual progress
        
        # Verify we resume from actual progress
        self.assertEqual(results['start_epoch'], 7)
        self.assertEqual(results['end_epoch'], 17)  # 7 + 10 epochs
        
        # Verify W&B logging starts after last logged step
        self.mock_run.log.assert_called()
        first_log = self.mock_run.log.call_args_list[0]
        step = first_log[1].get('step', None)
        if step is not None:
            self.assertGreaterEqual(step, self.checkpoint_metadata['last_wandb_step'])
    
    @patch('wandb.init')
    @patch('training.yolo.trainer.YOLOTrainer._setup_model')
    def test_new_run_step_counting(self, mock_setup_model, mock_init):
        """Test step counting for a new run (no resume)."""
        # Set up mock wandb run
        mock_init.return_value = self.mock_run
        
        # Initialize trainer without resume
        trainer = YOLOTrainer(self.config_path)
        
        # Mock model and its train method
        mock_model = MagicMock()
        mock_model.train.return_value = {
            'epochs': 10,
            'metrics/mAP50(B)': 0.85,
            'metrics/accuracy': 0.90,
            'train/box_loss': 0.15
        }
        trainer.model = mock_model
        trainer._setup_wandb()
        
        # Verify wandb.init was called without resume args
        mock_init.assert_called_once()
        init_kwargs = mock_init.call_args[1]
        self.assertNotIn('id', init_kwargs)
        self.assertNotIn('resume', init_kwargs)
        
        # Train and verify step counting starts from 0
        results = trainer.train()
        self.assertEqual(results['start_epoch'], 0)
        self.assertEqual(results['end_epoch'], 10)  # 10 epochs
        self.assertEqual(results['global_step'], self.mock_run.step)

if __name__ == '__main__':
    unittest.main()