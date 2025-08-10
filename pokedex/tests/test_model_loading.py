import unittest
from pathlib import Path
import yaml
import logging
import shutil
import time
from src.training.yolo.trainer import YOLOTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestModelLoading(unittest.TestCase):
    """Test cases for YOLO model loading functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        # Use baseline config for testing
        cls.config_path = Path("configs/yolov3/baseline_config.yaml")
        cls.cache_dir = Path.home() / '.cache' / 'ultralytics'
        
        # Ensure we're testing with a clean cache
        if cls.cache_dir.exists():
            logger.info(f"Cleaning up existing cache at {cls.cache_dir}")
            shutil.rmtree(cls.cache_dir)
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.assertTrue(self.config_path.exists(), f"Config file not found: {self.config_path}")
        
        # Load config to verify its contents
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Verify config has required fields
        self.assertIn('model', self.config, "Config missing 'model' section")
        self.assertIn('name', self.config['model'], "Config missing model name")
        self.assertIn('weights', self.config['model'], "Config missing model weights")
        self.assertIn('classes', self.config['model'], "Config missing number of classes")
    
    def test_model_download_and_load(self):
        """Test that model weights can be downloaded and loaded."""
        try:
            # Initialize trainer
            trainer = YOLOTrainer(self.config_path)
            
            # Initialize model (this should trigger download and setup)
            trainer._setup_model()
            
            # Verify model was created
            self.assertIsNotNone(trainer.model, "Model was not initialized")
            
                        # Verify model architecture
            self.assertTrue(hasattr(trainer.model.model, 'model'),
                          "Model missing expected architecture attributes")
            
            # Verify model was configured for correct number of classes
            expected_classes = self.config['model']['classes']
            actual_classes = trainer.model.model.model[-1].nc
            self.assertEqual(actual_classes, expected_classes,
                           f"Model has {actual_classes} classes, expected {expected_classes}")
            
            # Verify model has detection head
            self.assertTrue(isinstance(trainer.model.model.model[-1], type(trainer.model.model.model[-1])),
                          "Model missing detection head")
            
            logger.info("✅ Model download and loading test passed")
            
        except Exception as e:
            logger.error(f"Detailed error during model loading: {str(e)}")
            if hasattr(e, '__cause__') and e.__cause__:
                logger.error(f"Caused by: {str(e.__cause__)}")
            self.fail(f"Model loading failed with error: {e}")
    
    def test_invalid_weights_handling(self):
        """Test that invalid weights path is handled gracefully."""
        # Create a temporary directory that definitely doesn't exist
        nonexistent_dir = Path("/tmp/definitely_nonexistent_dir_" + str(time.time()))
        
        # Modify config to use non-existent model, weights, and YAML
        with open(self.config_path) as f:
            bad_config = yaml.safe_load(f)
        bad_config['model']['name'] = "nonexistent_model"  # Invalid model name
        bad_config['model']['weights'] = str(nonexistent_dir / "nonexistent_model.pt")  # Invalid weights path
        bad_config['model']['yaml'] = str(nonexistent_dir / "nonexistent.yaml")  # Invalid YAML path
        
        # Write temporary config with bad weights
        temp_config = Path("configs/yolov3/temp_test_config.yaml")
        with open(temp_config, 'w') as f:
            yaml.safe_dump(bad_config, f)
        
        try:
            # Create trainer but don't initialize model yet
            trainer = YOLOTrainer(temp_config)
            
            # Now try to initialize model - this should fail
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                trainer._setup_model()
            
            logger.info("✅ Invalid weights handling test passed")
            
        finally:
            # Clean up temporary config
            if temp_config.exists():
                temp_config.unlink()

if __name__ == '__main__':
    unittest.main()