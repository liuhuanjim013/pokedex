#!/usr/bin/env python3
"""
Test W&B permissions and verify logging capabilities.
Only attempts to log metrics without any model artifacts or code.
"""

import os
import wandb
import time
import random
import unittest

class TestWandbPermissions(unittest.TestCase):
    """Test suite for W&B permissions and basic logging functionality."""
    
    def setUp(self):
        """Set up test case."""
        self.project = "pokemon-classifier"
        self.run_name = "permission-test"
    
    def test_basic_logging(self):
        """Test basic W&B logging functionality."""
        try:
            # Initialize run with minimal settings
            run = wandb.init(
                project=self.project,
                name=self.run_name,
                settings=wandb.Settings(
                    save_code=False,  # Don't try to save source code
                    disable_git=True,  # Don't try to track git
                )
            )
            
            print("✅ W&B initialization successful")
            print(f"📊 Run URL: {run.get_url()}")
            
            # Log some simple metrics
            for i in range(5):
                metrics = {
                    "test_metric": random.random(),
                    "step": i
                }
                wandb.log(metrics)
                print(f"✅ Logged metrics for step {i}")
                time.sleep(1)  # Small delay to simulate work
                
            print("\n✨ Test completed successfully!")
            print("If you see this message and no errors, basic logging is working.")
            
        except wandb.errors.CommError as e:
            print(f"\n❌ Communication Error: {e}")
            print("\nTrying offline mode...")
            os.environ['WANDB_MODE'] = 'offline'
            self.test_basic_logging()  # Retry in offline mode
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nTroubleshooting steps:")
            print("1. Check your W&B API key: wandb login")
            print("2. Verify project permissions")
            print("3. Try running in offline mode: export WANDB_MODE=offline")
            raise  # Re-raise the exception for unittest to catch
        
        finally:
            wandb.finish()

if __name__ == "__main__":
    print("\n🔍 Testing W&B Permissions\n")
    unittest.main()