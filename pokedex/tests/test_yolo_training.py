#!/usr/bin/env python3
"""Test YOLO training setup and data loading."""

import os
import yaml
import json
import shutil
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import wandb

# Add src to path for module imports
import sys
src_path = str(Path(__file__).resolve().parents[1] / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from training.yolo.trainer import YOLOTrainer

class TestYOLOTraining(unittest.TestCase):
    """Test YOLO training setup and data loading."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test dataset structure
        self.dataset_dir = Path(self.test_dir) / 'datasets' / 'liuhuanjim013' / 'pokemon-yolo-1025'
        for split in ['train', 'validation', 'test']:
            split_dir = self.dataset_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            # Add a test image and label
            (split_dir / 'images').mkdir(exist_ok=True)
            (split_dir / 'labels').mkdir(exist_ok=True)
            with open(split_dir / 'images' / 'test.jpg', 'w') as f:
                f.write('test image')
            with open(split_dir / 'labels' / 'test.txt', 'w') as f:
                f.write('0 0.5 0.5 1.0 1.0')  # Full image bounding box
        
        # Create test YOLO data config
        self.data_config = {
            'path': 'liuhuanjim013/pokemon-yolo-1025',  # HF dataset path
            'train': 'train',  # Training split
            'val': 'validation',  # Validation split
            'test': 'test',  # Test split
            'nc': 1025,  # Number of classes
            'names': []  # Empty list for class names (not needed for training)
        }
        self.data_config_path = Path(self.test_dir) / 'yolo_data.yaml'
        with open(self.data_config_path, 'w') as f:
            yaml.dump(self.data_config, f)
        
        # Create test training config
        self.config = {
            'wandb': {
                'project': 'test-project',
                'name': 'test-run',
                'entity': 'test-entity'
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
            },
            'model': {
                'name': 'yolov3',
                'weights': 'yolov3.pt',
                'classes': 1025,
                'img_size': 416,
                'pretrained': True,
                'yaml': str(Path(self.test_dir) / 'yolov3.yaml')
            },
            'data': {
                'path': str(self.data_config_path)
            },
            'checkpoint': {
                'save_dir': str(Path(self.test_dir) / 'checkpoints'),
                'save_frequency': 5,
                'max_checkpoints': 3
            }
        }
        
        # Write config to file
        self.config_path = Path(self.test_dir) / 'test_config.yaml'
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
            
        # Create test YOLO model config
        self.model_config = {
            'nc': 1025,
            'depth_multiple': 1.0,
            'width_multiple': 1.0,
            'anchors': 3,
            'backbone': [
                [-1, 1, 'Conv', [32, 3, 1]],  # Minimal backbone for testing
                [-1, 1, 'Conv', [64, 3, 2]],
            ],
            'head': [
                [-1, 1, 'Conv', [1025, 1, 1]],  # Minimal head for testing
            ]
        }
        with open(self.config['model']['yaml'], 'w') as f:
            yaml.dump(self.model_config, f)
            
        # Create directories
        Path(self.config['checkpoint']['save_dir']).mkdir(parents=True, exist_ok=True)
        
        # Update Ultralytics settings to use our test dataset directory
        os.makedirs(os.path.expanduser('~/.config/Ultralytics'), exist_ok=True)
        settings_path = os.path.expanduser('~/.config/Ultralytics/settings.json')
        settings = {
            'datasets_dir': str(Path(self.test_dir) / 'datasets'),
            'weights_dir': str(Path(self.test_dir) / 'weights'),
            'runs_dir': str(Path(self.test_dir) / 'runs')
        }
        with open(settings_path, 'w') as f:
            json.dump(settings, f)
    
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
    def test_training_setup(self, mock_setup_model, mock_init):
        """Test that training setup works correctly."""
        # Set up mock wandb run
        mock_run = MagicMock()
        mock_run.id = 'test_run_id'
        mock_init.return_value = mock_run
        
        # Initialize trainer
        trainer = YOLOTrainer(self.config_path)
        
        # Verify config loaded correctly
        self.assertEqual(trainer.config['model']['classes'], 1025)
        self.assertEqual(trainer.config['training']['batch_size'], 16)
        
        # Verify data config exists and is correct
        self.assertTrue(self.data_config_path.exists())
        with open(self.data_config_path) as f:
            data_config = yaml.safe_load(f)
        self.assertEqual(data_config['nc'], 1025)
        self.assertEqual(data_config['train'], 'train')
        
        # Verify dataset structure exists
        self.assertTrue((self.dataset_dir / 'train').exists())
        self.assertTrue((self.dataset_dir / 'validation').exists())
        self.assertTrue((self.dataset_dir / 'test').exists())
        
        # Verify test image and label exist
        self.assertTrue((self.dataset_dir / 'train' / 'images' / 'test.jpg').exists())
        self.assertTrue((self.dataset_dir / 'train' / 'labels' / 'test.txt').exists())
    
    @patch('wandb.init')
    @patch('training.yolo.trainer.YOLOTrainer._setup_model')
    def test_training_args(self, mock_setup_model, mock_init):
        """Test that training arguments are prepared correctly."""
        # Set up mock wandb run
        mock_run = MagicMock()
        mock_run.id = 'test_run_id'
        mock_init.return_value = mock_run
        
        # Initialize trainer
        trainer = YOLOTrainer(self.config_path)
        
        # Get training arguments
        train_args = trainer._prepare_training_args()
        
        # Verify training arguments
        self.assertEqual(train_args['data'], str(self.data_config_path))
        self.assertEqual(train_args['epochs'], 10)
        self.assertEqual(train_args['batch'], 16)
        self.assertEqual(train_args['imgsz'], 416)
        self.assertEqual(train_args['save_period'], 5)
        
        # Verify augmentation settings
        self.assertEqual(train_args['fliplr'], 0.5)  # Only horizontal flip
        self.assertEqual(train_args['mosaic'], 1.0)
        self.assertEqual(train_args['mixup'], 0.0)
    
    @patch('wandb.init')
    @patch('training.yolo.trainer.YOLOTrainer._setup_model')
    def test_data_loading(self, mock_setup_model, mock_init):
        """Test that data loading works correctly."""
        # Set up mock wandb run
        mock_run = MagicMock()
        mock_run.id = 'test_run_id'
        mock_init.return_value = mock_run
        
        # Initialize trainer
        trainer = YOLOTrainer(self.config_path)
        
        # Mock the YOLO model
        mock_model = MagicMock()
        mock_model.train.return_value = {
            'epochs': 10,
            'metrics/mAP50(B)': 0.85,
            'metrics/accuracy': 0.90,
            'train/box_loss': 0.15
        }
        trainer.model = mock_model
        
        # Start training
        results = trainer.train(start_epoch=0)
        
        # Verify training results
        self.assertEqual(results['epochs'], 10)
        self.assertGreater(results.get('metrics/mAP50(B)', 0), 0)
        
        # Verify model was called with correct data path
        train_args = mock_model.train.call_args[1]
        self.assertEqual(train_args['data'], str(self.data_config_path))

if __name__ == '__main__':
    unittest.main()