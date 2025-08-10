"""Test W&B resume functionality."""

import unittest
import os
import sys
from pathlib import Path
import wandb
import yaml
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add project root to path for module imports
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import trainer and checkpoint manager
from src.training.yolo.trainer import YOLOTrainer
from src.training.yolo.checkpoint_manager import CheckpointManager

class TestWandbResume(unittest.TestCase):
    """Test W&B resume functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        # Import here to avoid circular imports
        from src.training.yolo.wandb_integration import WandBIntegration
        cls.WandBIntegration = WandBIntegration
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset WandBIntegration singleton
        self.WandBIntegration._instance = None
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create temporary config file
        self.config = {
            'wandb': {
                'project': 'pokemon-classifier-test',
                'name': 'test-run',
                'entity': 'liuhuanjim013',
                'settings': {
                    'save_code': False,
                    'disable_git': True
                }
            },
            'model': {
                'name': 'yolov3',
                'weights': 'yolov3.pt',
                'classes': 1025
            },
            'checkpoint': {
                'save_frequency': 10,
                'save_dir': str(Path(self.test_dir) / 'checkpoints'),
                'max_checkpoints': 5
            },
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'learning_rate': 0.001,
                'weight_decay': 0.0005,
                'augmentation': {
                    'hsv_h': 0.015,
                    'hsv_s': 0.7,
                    'hsv_v': 0.4,
                    'degrees': 0.0,
                    'translate': 0.0,
                    'scale': 0.5,
                    'shear': 0.0,
                    'perspective': 0.0,
                    'flipud': 0.0,
                    'fliplr': 0.5,
                    'mosaic': 0.0,
                    'mixup': 0.0
                },
                'scheduler': 'none',
                'early_stopping': 'none'
            },
            'data': {
                'dataset': 'liuhuanjim013/pokemon-yolo-1025',
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15
            }
        }
        
        # Create checkpoint directory
        Path(self.config['checkpoint']['save_dir']).mkdir(parents=True, exist_ok=True)
        self.config_path = Path(self.test_dir) / 'test_config.yaml'
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(self.config, f)
            
        # Create temporary run ID file
        self.run_id = 'test_run_123'
        self.run_id_file = Path("wandb_run_id.txt")
        if self.run_id_file.exists():
            self.run_id_file.unlink()
        with open(self.run_id_file, 'w') as f:
            f.write(self.run_id)
            
        # Clear any existing W&B environment variables
        for key in list(os.environ.keys()):
            if key.startswith('WANDB_'):
                del os.environ[key]
            
        # Store original working directory
        self.original_cwd = os.getcwd()
        # Change to test directory
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Restore original working directory
        os.chdir(self.original_cwd)
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
        # Clean up run ID file
        if self.run_id_file.exists():
            self.run_id_file.unlink()
        # Finish any active W&B run
        if wandb.run is not None:
            wandb.finish()
        # Clear any W&B environment variables
        for key in list(os.environ.keys()):
            if key.startswith('WANDB_'):
                del os.environ[key]
    
    def test_run_id_persistence(self):
        """Test that run ID is correctly saved and loaded."""
        # Mock W&B run and checkpoint manager
        mock_run = MagicMock()
        mock_run.id = 'new_run_456'
        mock_checkpoint_manager = MagicMock()
        
        with patch('wandb.init', return_value=mock_run), \
             patch('src.training.yolo.trainer.CheckpointManager', return_value=mock_checkpoint_manager):
            # Initialize W&B
            trainer = YOLOTrainer(self.config_path)
            trainer._setup_wandb()
            
            # Verify run ID was saved
            self.assertTrue(self.run_id_file.exists())
            with open(self.run_id_file) as f:
                saved_id = f.read().strip()
            self.assertEqual(saved_id, 'new_run_456')
    
    def test_resume_with_valid_id(self):
        """Test resuming with a valid run ID."""
        # Mock W&B run and checkpoint manager
        mock_run = MagicMock()
        mock_run.id = self.run_id
        mock_checkpoint_manager = MagicMock()
        
        with patch('wandb.init', return_value=mock_run) as mock_init, \
             patch('src.training.yolo.trainer.CheckpointManager', return_value=mock_checkpoint_manager):
            # Resume training
            trainer = YOLOTrainer(self.config_path, resume_id=self.run_id)
            
            # Verify W&B was initialized with resume
            mock_init.assert_called_once()
            init_kwargs = mock_init.call_args[1]
            self.assertEqual(init_kwargs.get('id'), self.run_id)
            self.assertEqual(init_kwargs.get('resume'), 'must')
    
    def test_resume_with_missing_id(self):
        """Test behavior when run ID file is missing."""
        # Remove run ID file if it exists
        if self.run_id_file.exists():
            self.run_id_file.unlink()
        
        # Mock W&B run and checkpoint manager
        mock_run = MagicMock()
        mock_run.id = 'new_run_789'
        mock_checkpoint_manager = MagicMock()
        
        # Ensure we start in online mode
        if 'WANDB_MODE' in os.environ:
            del os.environ['WANDB_MODE']
        
        with patch('wandb.init', return_value=mock_run) as mock_init, \
             patch('src.training.yolo.trainer.CheckpointManager', return_value=mock_checkpoint_manager):
            # Try to resume training
            trainer = YOLOTrainer(self.config_path)
            trainer._setup_wandb()
            
            # Verify W&B was initialized without resume
            mock_init.assert_called_once()
            init_kwargs = mock_init.call_args[1]
            self.assertNotIn('id', init_kwargs)
            self.assertNotIn('resume', init_kwargs)
    
    def test_resume_with_invalid_id(self):
        """Test behavior with invalid run ID."""
        # Write invalid run ID
        with open(self.run_id_file, 'w') as f:
            f.write('invalid_run_id')
        
        # Mock W&B init to fail on resume but succeed in offline mode
        def mock_init(**kwargs):
            if 'id' in kwargs:
                raise wandb.errors.CommError("Run not found")
            if os.environ.get('WANDB_MODE') != 'offline':
                raise wandb.errors.CommError("Connection failed")
            mock_run = MagicMock()
            mock_run.id = 'fallback_run_123'
            return mock_run
        
        mock_checkpoint_manager = MagicMock()
        
        # Ensure we start in online mode
        if 'WANDB_MODE' in os.environ:
            del os.environ['WANDB_MODE']
        
        with patch('wandb.init', side_effect=mock_init) as mock_init, \
             patch('src.training.yolo.trainer.CheckpointManager', return_value=mock_checkpoint_manager):
            # Try to resume training
            trainer = YOLOTrainer(self.config_path)
            trainer._setup_wandb()
            
            # Verify fallback to offline mode
            self.assertEqual(os.environ.get('WANDB_MODE'), 'offline')
            # Verify one resume attempt and one offline attempt
            self.assertEqual(mock_init.call_count, 2)
    
    def test_offline_mode_fallback(self):
        """Test fallback to offline mode on connection error."""
        # Mock W&B init to fail online but succeed offline
        def mock_init(**kwargs):
            if os.environ.get('WANDB_MODE') != 'offline':
                raise wandb.errors.CommError("Connection failed")
            mock_run = MagicMock()
            mock_run.id = 'offline_run_123'
            return mock_run
        
        mock_checkpoint_manager = MagicMock()
        
        # Ensure we start in online mode
        if 'WANDB_MODE' in os.environ:
            del os.environ['WANDB_MODE']
        
        with patch('wandb.init', side_effect=mock_init) as mock_init, \
             patch('src.training.yolo.trainer.CheckpointManager', return_value=mock_checkpoint_manager):
            # Try to initialize W&B
            trainer = YOLOTrainer(self.config_path)
            trainer._setup_wandb()
            
            # Verify fallback to offline mode
            self.assertEqual(os.environ.get('WANDB_MODE'), 'offline')
            # Verify one online attempt and one offline attempt
            self.assertEqual(mock_init.call_count, 2)
    
    def test_cleanup_on_error(self):
        """Test that W&B is cleaned up on error."""
        # Mock W&B run and checkpoint manager
        mock_run = MagicMock()
        mock_checkpoint_manager = MagicMock()
        
        # Mock wandb.init to fail
        def mock_init(**kwargs):
            raise wandb.errors.CommError("Test error")
        
        with patch('wandb.init', side_effect=mock_init), \
             patch('src.training.yolo.wandb_integration.WandBIntegration.finish') as mock_finish, \
             patch('src.training.yolo.trainer.CheckpointManager', return_value=mock_checkpoint_manager):
            try:
                trainer = YOLOTrainer(self.config_path)
                trainer._setup_wandb()
            except Exception:
                pass
            
            # Verify W&B was finished
            mock_finish.assert_called_once()

if __name__ == '__main__':
    unittest.main()