#!/usr/bin/env python3
"""
Google Colab Environment Setup for YOLO Training

This script automates the setup of the Google Colab environment for YOLOv3 training.
It uses the centralized setup_environment.py script for consistency with local development.
"""

import os
import sys
import subprocess
from pathlib import Path
def is_colab():
    """Check if running in Google Colab."""
    try:
        from google.colab import drive
        return True, drive
    except (ImportError, ModuleNotFoundError):
        return False, None
import wandb

def setup_storage():
    """Set up storage directories."""
    try:
        is_colab_env, drive_module = is_colab()
        
        # Get the repository root (pokedex/pokedex)
        repo_root = Path(__file__).resolve().parents[2]
        
        if is_colab_env:
            # Mount drive and use Colab paths
            drive_module.mount('/content/drive')
            base_dir = Path('/content/drive/MyDrive/pokemon_yolo')
            print("✅ Google Drive mounted successfully!")
            print("✅ Using Google Drive storage")
        else:
            # Use local paths inside the repository
            base_dir = repo_root
            print("✅ Using local repository storage")
        
        # Create project directories
        dirs = {
            'checkpoints': base_dir / 'models' / 'checkpoints',
            'logs': base_dir / 'models' / 'logs',
            'models': base_dir / 'models' / 'final'
        }
        
        # Create directories and verify
        for name, path in dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                raise RuntimeError(f"Failed to create {name} directory at {path}")
            print(f"📁 {name.capitalize()} directory: {path}")
        
        return dirs, is_colab_env
    except Exception as e:
        print(f"❌ Failed to set up storage: {e}")
        raise

def verify_repository():
    """Verify we're in a valid repository."""
    try:
        # Check if we're in a git repository
        result = subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'],
                              capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError("Not in a git repository")
        
        # Check if we're in the pokedex directory structure
        cwd = os.getcwd()
        if not any(p in cwd for p in ['/pokedex/pokedex', '\\pokedex\\pokedex']):
            raise RuntimeError("Not in the pokedex project directory structure")
            
        print("✅ Valid repository structure detected")
        print(f"📂 Working directory: {cwd}")
    except Exception as e:
        print(f"❌ Repository verification failed: {e}")
        print("⚠️ Please ensure you're in the pokedex/pokedex directory")
        raise

def setup_environment():
    """Set up environment using centralized setup script."""
    try:
        # First run centralized setup script to set up conda and uv
        subprocess.run(['python', 'scripts/common/setup_environment.py',
                       '--experiment', 'yolo',
                       '--colab',
                       '--verify'],
                      check=True)
        print("✅ Environment setup completed using centralized script")
        
        # Then install critical packages using uv in the conda environment
        subprocess.check_call([
            "conda", "run", "-n", "pokemon-classifier",
            "uv", "pip", "install",
            # Core ML packages
            "ultralytics", "wandb", "huggingface_hub",
            # Network resilience packages
            "backoff", "requests", "urllib3",
            # Progress tracking
            "tqdm", "rich"
        ])
        print("✅ Critical packages installed with uv")
    except subprocess.CalledProcessError as e:
        print(f"❌ Environment setup failed: {e}")
        raise

def verify_gpu(is_colab: bool):
    """Verify GPU availability and CUDA setup."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            msg = "No GPU available! Please enable GPU runtime in Colab." if is_colab else "No GPU available!"
            raise RuntimeError(msg)
        
        print("🎯 GPU Check:")
        print(f"• GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"• CUDA Version: {torch.version.cuda}")
        print(f"• GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test CUDA memory allocation
        print("\n🧪 Testing CUDA memory...")
        test_tensor = torch.randn(1000, 1000).cuda()  # 4MB test tensor
        del test_tensor
        torch.cuda.empty_cache()
        print("✅ CUDA memory allocation test passed")
        
        # Test CUDA computation
        print("\n🔢 Testing CUDA computation...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x.t())
        del x, y
        torch.cuda.empty_cache()
        print("✅ CUDA computation test passed")
        
        print("\n✅ GPU verification successful")
    except Exception as e:
        print(f"❌ GPU verification failed: {e}")
        if is_colab:
            print("⚠️ Make sure to select GPU runtime in Colab!")
            print("⚠️ Go to Runtime > Change runtime type > Hardware accelerator > GPU")
        raise

def setup_wandb():
    """Set up Weights & Biases for experiment tracking."""
    try:
        # Check for WANDB_API_KEY in environment
        wandb_token = os.getenv("WANDB_API_KEY")
        if not wandb_token:
            print("⚠️ WANDB_API_KEY not found in environment")
            print("ℹ️ Will prompt for login...")
            wandb.login()
        else:
            print("✅ Found WANDB_API_KEY in environment")
            wandb.login(key=wandb_token)
        
        print("✅ W&B login successful")
        
        # Test W&B setup
        test_run = wandb.init(project="pokemon-classifier", name="colab-setup-test")
        test_run.finish()
        print("✅ W&B setup verified")
    except Exception as e:
        print(f"❌ W&B setup failed: {e}")
        print("ℹ️ Set WANDB_API_KEY environment variable or run wandb login")
        raise

def validate_hf_token(token: str) -> bool:
    """Validate Hugging Face token format."""
    if not token:
        return False
    token = token.strip()
    # Check basic format (should be "hf_..." and about 31-40 chars)
    if not token.startswith("hf_") or len(token) < 31 or len(token) > 40:
        return False
    return True

def setup_git_credentials():
    """Set up git credential helper."""
    try:
        # Check current git credential helper
        result = subprocess.run(
            ["git", "config", "--global", "credential.helper"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if not result.stdout.strip():
            print("\n🔧 Setting up git credential helper...")
            subprocess.run(
                ["git", "config", "--global", "credential.helper", "store"],
                check=True
            )
            print("✅ Git credential helper set to 'store'")
        else:
            print(f"✅ Git credential helper already set to: {result.stdout.strip()}")
            
        return True
    except Exception as e:
        print(f"⚠️ Could not set up git credentials: {e}")
        return False

def verify_dataset_access():
    """Verify access to the Hugging Face dataset."""
    try:
        # First try different environment variables
        token = None
        for var in ["HF_TOKEN", "HUGGINGFACE_TOKEN"]:
            if var in os.environ:
                token = os.environ[var].strip()
                if validate_hf_token(token):
                    print(f"✅ Found valid token in {var}")
                    break
                else:
                    print(f"⚠️ Invalid token format in {var}")
        
        if not token:
            print("❌ No valid token found in environment")
            print("ℹ️ Token should:")
            print("  • Start with 'hf_'")
            print("  • Be about 35 characters long")
            print("  • Example: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            return False
            
        # Set up git credentials first
        setup_git_credentials()
        
        # Clear any existing tokens
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
        token_file = os.path.expanduser("~/.huggingface/token")
        if os.path.exists(token_file):
            os.remove(token_file)
            
        # Try CLI login with git credentials
        subprocess.run(["hf", "auth", "login", "--token", token, "--add-to-git-credential"], check=True)
        print("✅ Logged in to Hugging Face with git credentials")
        
        # Verify login worked
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']} ({user['fullname']})")
        
        # Test dataset access
        print("\n🐉 Testing Pokemon dataset access...")
        from datasets import load_dataset
        
        # Try to access dataset directly
        dataset = load_dataset("liuhuanjim013/pokemon-yolo-1025", token=token)
        print("✅ Successfully loaded Pokemon dataset!")
        print(f"• Training examples: {len(dataset['train'])}")
        
        # Try to list files
        files = api.list_repo_files(
            repo_id="liuhuanjim013/pokemon-yolo-1025",
            repo_type="dataset",
            token=token
        )
        print("\n📂 Files in dataset:")
        for f in sorted(files)[:5]:  # Show first 5 files
            print(f"  • {f}")
        if len(files) > 5:
            print(f"  • ... and {len(files)-5} more files")
            
        return True
            
    except Exception as e:
        print(f"\n❌ Dataset verification failed: {e}")
        print("\nℹ️ Common issues:")
        print("1. Token might be expired - generate a new one")
        print("2. Dataset might be private - check permissions")
        print("3. Network connectivity issues")
        return False

def print_setup_summary(dirs: dict, is_colab: bool):
    """Print setup summary and next steps."""
    print("\n" + "="*60)
    print("🎯 Environment Setup Summary")
    print("="*60)
    
    print("\n📁 Project Directories:")
    for name, path in dirs.items():
        print(f"• {name}: {path}")
    
    print("\n🔧 Environment Status:")
    if is_colab:
        print("• Environment: Google Colab")
        print("• Storage: Google Drive (mounted)")
    else:
        print("• Environment: Local Development")
        print("• Storage: Local filesystem")
    print("• GPU: Available and verified")
    print("• Dependencies: Installed via setup_environment.py")
    print("• W&B: Configured and tested")
    print("• Dataset: Access verified")
    
    print("\n📋 Next Steps:")
    print("1. Run baseline training:")
    print("   python scripts/yolo/train_yolov3_baseline.py")
    print("\n2. Run improved training:")
    print("   python scripts/yolo/train_yolov3_improved.py")
    
    print("\n" + "="*60)

def main():
    """Main setup function."""
    try:
        print("🚀 Setting up training environment...")
        
        # First verify we're in the right place
        verify_repository()
        
        # Then proceed with other setup steps
        dirs, is_colab = setup_storage()
        setup_environment()
        verify_gpu(is_colab)
        setup_wandb()
        
        # Try dataset access but don't fail if not available
        dataset_available = verify_dataset_access()
        
        # Print summary
        print_setup_summary(dirs, is_colab)
        
        if not dataset_available:
            print("\n⚠️ Setup completed with warnings (dataset not available)")
            return True
        
        return True
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        if isinstance(e, ModuleNotFoundError) and 'google.colab' in str(e):
            print("ℹ️ Not running in Google Colab - this is expected in local development")
        elif isinstance(e, RuntimeError) and "Not in the pokedex project directory structure" in str(e):
            print("⚠️ Please run this script from the pokedex/pokedex directory")
        return False

if __name__ == "__main__":
    main()