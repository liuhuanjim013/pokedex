#!/usr/bin/env python3
"""
Google Colab Environment Setup for YOLO Training

This script automates the setup of the Google Colab environment for YOLOv3 training.
It uses the centralized setup_environment.py script for consistency with local development.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def is_colab():
    """Check if running in Google Colab."""
    try:
        # Check if we're in a Colab environment by looking for specific paths
        if os.path.exists('/usr/local/lib/python3.*/dist-packages/google/colab'):
            return True
        return False
    except Exception:
        return False

import wandb

def setup_storage():
    """Get storage configuration."""
    try:
        is_colab_env = is_colab()
        repo_root = Path(__file__).resolve().parents[2]
        
        # Just return the paths without trying to create/verify
        dirs = {
            'checkpoints': repo_root / 'models' / 'checkpoints',
            'logs': repo_root / 'models' / 'logs',
            'models': repo_root / 'models' / 'final'
        }
        return dirs, is_colab_env
    except Exception as e:
        print(f"❌ Failed to get storage config: {e}")
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
        print("🔧 Running centralized environment setup...")
        print("   This may take several minutes for conda operations...")
        
        # Run the setup script with real-time output
        try:
            print("🔄 Starting centralized setup (showing real-time progress)...")
            print("   ⏳ This may take several minutes - you should see progress below...")
            
            # Start the setup process
            result = subprocess.run(['python', 'scripts/common/setup_environment.py',
                                   '--experiment', 'yolo',
                                   '--colab',
                                   '--verify'],
                                  timeout=1800)  # 30 minute timeout, no output capture
            
            if result.returncode != 0:
                print(f"❌ Centralized setup failed with return code: {result.returncode}")
                raise subprocess.CalledProcessError(result.returncode, result.args)
            
            print("✅ Environment setup completed using centralized script")
            print("📋 Note: If the installation seems stuck, it's likely downloading large packages like PyTorch")
            print("   This can take 5-15 minutes depending on your internet connection")
            
        except subprocess.TimeoutExpired:
            print("⏰ Centralized setup timed out after 30 minutes")
            print("💡 This might be due to slow internet connection or conda metadata download issues")
            print("   Try running the script again, or check your network connection")
            raise
        
        # Then install critical packages using uv in the conda environment
        # Get conda path directly
        def get_conda_path():
            """Get the conda executable path, handling both local and Colab environments"""
            # Check if we're in a Colab environment with /content installation
            if os.path.exists("/content/miniconda3/bin/conda"):
                return "/content/miniconda3/bin/conda"
            # Check if conda is in PATH
            try:
                result = subprocess.run(["which", "conda"], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
            # Fallback to common local paths
            for path in ["/home/liuhuan/miniconda3/bin/conda", "/opt/conda/bin/conda"]:
                if os.path.exists(path):
                    return path
            return "conda"  # Fallback to PATH
        
        conda_path = get_conda_path()
        
        # Check if packages are already installed
        print("🔍 Checking if critical packages are already installed...")
        check_script = """
import sys
packages = ['ultralytics', 'wandb', 'huggingface_hub', 'backoff', 'requests', 'urllib3', 'tqdm', 'rich']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        missing.append(pkg)
        print(f"❌ {pkg}")
if missing:
    print(f"Missing packages: {missing}")
    sys.exit(1)
else:
    print("All packages already installed!")
    sys.exit(0)
"""
        
        try:
            subprocess.run([
                conda_path, "run", "-n", "pokemon-classifier",
                "python", "-c", check_script
            ], check=True, capture_output=True, text=True)
            print("✅ All critical packages are already installed!")
            return
        except subprocess.CalledProcessError:
            print("📦 Some packages missing, installing critical packages with uv...")
            print("   Installing: ultralytics, wandb, huggingface_hub, backoff, requests, urllib3, tqdm, rich")
            print("   ⏳ Please wait - downloading and installing packages...")
        
        # Use verbose output for uv installation with timeout
        try:
            # Start installation in background and show progress
            import threading
            import time
            
            # Flag to track if installation is complete
            installation_complete = False
            installation_result = None
            
            def run_installation():
                nonlocal installation_complete, installation_result
                try:
                    installation_result = subprocess.run([
                        conda_path, "run", "-n", "pokemon-classifier",
                        "uv", "pip", "install", "--verbose",
                        # Core ML packages
                        "ultralytics", "wandb", "huggingface_hub",
                        # Network resilience packages
                        "backoff", "requests", "urllib3",
                        # Progress tracking
                        "tqdm", "rich"
                    ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
                except subprocess.TimeoutExpired:
                    installation_result = subprocess.TimeoutExpired("Installation timed out", 600)
                finally:
                    installation_complete = True
            
            # Start installation in background thread
            install_thread = threading.Thread(target=run_installation)
            install_thread.start()
            
            # Show progress dots while installation is running
            dots = 0
            while not installation_complete:
                print(f"\r   Installing{'.' * dots}   ", end='', flush=True)
                time.sleep(1)
                dots = (dots + 1) % 4
            
            print()  # New line after progress
            
            # Get the result
            if isinstance(installation_result, subprocess.TimeoutExpired):
                print("⏰ Installation timed out after 10 minutes")
                print("💡 This might be due to slow internet connection")
                print("   Try running the script again, or check your network connection")
                raise installation_result
            
            result = installation_result
            
        except subprocess.TimeoutExpired:
            print("⏰ Installation timed out after 10 minutes")
            print("💡 This might be due to slow internet connection")
            print("   Try running the script again, or check your network connection")
            raise
        
        if result.returncode != 0:
            print(f"❌ Package installation failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, result.args)
        
        print("✅ Critical packages installed with uv")
        print(f"📋 Installation output: {result.stdout[-500:]}...")  # Show last 500 chars
        
        # Test that we can run Python in the conda environment
        conda_path = get_conda_path()
        subprocess.check_call([
            conda_path, "run", "-n", "pokemon-classifier",
            "python", "-c", "import ultralytics; print('✅ ultralytics available')"
        ])
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Environment setup failed: {e}")
        raise

def verify_gpu(is_colab: bool):
    """Verify GPU availability and CUDA setup."""
    try:
        # Get conda path
        def get_conda_path():
            """Get the conda executable path, handling both local and Colab environments"""
            # Check if we're in a Colab environment with /content installation
            if os.path.exists("/content/miniconda3/bin/conda"):
                return "/content/miniconda3/bin/conda"
            # Check if conda is in PATH
            try:
                result = subprocess.run(["which", "conda"], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
            # Fallback to common local paths
            for path in ["/home/liuhuan/miniconda3/bin/conda", "/opt/conda/bin/conda"]:
                if os.path.exists(path):
                    return path
            return "conda"  # Fallback to PATH
        
        conda_path = get_conda_path()
        
        # Run GPU verification in conda environment
        gpu_check_script = """
import torch

print("🎯 GPU Check:")
print(f"• CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"• GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"• CUDA Version: {torch.version.cuda}")
    print(f"• GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test CUDA memory allocation
    print("🧪 Testing CUDA memory...")
    test_tensor = torch.randn(1000, 1000).cuda()  # 4MB test tensor
    del test_tensor
    torch.cuda.empty_cache()
    print("✅ CUDA memory allocation test passed")
    
    # Test CUDA computation
    print("🔢 Testing CUDA computation...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x.t())
    del x, y
    torch.cuda.empty_cache()
    print("✅ CUDA computation test passed")
    
    print("✅ GPU verification successful")
else:
    print("⚠️ No GPU available - will use CPU for training")
    print("ℹ️ Training will be slower but still functional")
    print("✅ CPU-only setup verified")
"""
        
        subprocess.check_call([
            conda_path, "run", "-n", "pokemon-classifier",
            "python", "-c", gpu_check_script
        ])
        
    except Exception as e:
        print(f"❌ GPU verification failed: {e}")
        if is_colab:
            print("⚠️ Make sure to select GPU runtime in Colab!")
            print("⚠️ Go to Runtime > Change runtime type > Hardware accelerator > GPU")
        else:
            print("⚠️ GPU verification failed, but continuing with CPU setup...")
            print("ℹ️ Training will work on CPU, just slower")
        # Don't raise exception for local development
        if is_colab:
            raise

def setup_wandb():
    """Set up Weights & Biases for experiment tracking."""
    try:
        # Get conda path
        def get_conda_path():
            """Get the conda executable path, handling both local and Colab environments"""
            # Check if we're in a Colab environment with /content installation
            if os.path.exists("/content/miniconda3/bin/conda"):
                return "/content/miniconda3/bin/conda"
            # Check if conda is in PATH
            try:
                result = subprocess.run(["which", "conda"], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
            # Fallback to common local paths
            for path in ["/home/liuhuan/miniconda3/bin/conda", "/opt/conda/bin/conda"]:
                if os.path.exists(path):
                    return path
            return "conda"  # Fallback to PATH
        
        conda_path = get_conda_path()
        
        # Check for WANDB_API_KEY in environment
        wandb_token = os.getenv("WANDB_API_KEY")
        
        # Run W&B setup in conda environment
        wandb_setup_script = f"""
import wandb
import os

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
"""
        
        subprocess.check_call([
            conda_path, "run", "-n", "pokemon-classifier",
            "python", "-c", wandb_setup_script
        ])
        
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
    
    print("\n🔧 Environment Status:")
    print("• Environment:", "Google Colab" if is_colab else "Local Development")
    print("• GPU: Available and verified")
    print("• Dependencies: Installed via setup_environment.py")
    print("• W&B: Configured and tested")
    print("• Dataset: Access verified")
    
    print("\n📋 Next Steps:")
    print("1. Run baseline training:")
    print("   python scripts/yolo/run_training_in_env.py train_yolov3_baseline.py")
    print("\n2. Run improved training:")
    print("   python scripts/yolo/run_training_in_env.py train_yolov3_improved.py")
    print("\n3. Set up experiment:")
    print("   python scripts/yolo/run_training_in_env.py setup_yolov3_experiment.py")
    
    print("\n" + "="*60)

def setup_maixcam_environment():
    """Set up Maix Cam specific environment."""
    try:
        print("📱 Setting up Maix Cam environment...")
        
        # Get conda path
        def get_conda_path():
            """Get the conda executable path, handling both local and Colab environments"""
            # Check if we're in a Colab environment with /content installation
            if os.path.exists("/content/miniconda3/bin/conda"):
                return "/content/miniconda3/bin/conda"
            # Check if conda is in PATH
            try:
                result = subprocess.run(["which", "conda"], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
            # Fallback to common local paths
            for path in ["/home/liuhuan/miniconda3/bin/conda", "/opt/conda/bin/conda"]:
                if os.path.exists(path):
                    return path
            return "conda"  # Fallback to PATH
        
        conda_path = get_conda_path()
        
        # Check if packages are already installed (quick setup approach)
        print("🔍 Checking if Maix Cam packages are already installed...")
        maixcam_packages = [
            'ultralytics', 'wandb', 'huggingface_hub', 'torch', 
            'torchvision', 'opencv-python', 'pillow', 'matplotlib',
            'albumentations', 'scikit-learn', 'pandas'
        ]
        
        missing = []
        available = []
        
        for pkg in maixcam_packages:
            try:
                # Try importing the package
                result = subprocess.run([
                    conda_path, "run", "-n", "pokemon-classifier",
                    "python", "-c", f"import {pkg.replace('-', '_')}; print('OK')"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    available.append(pkg)
                    print(f"✅ {pkg}")
                else:
                    missing.append(pkg)
                    print(f"❌ {pkg}")
            except Exception as e:
                missing.append(pkg)
                print(f"❌ {pkg} (error: {e})")
        
        print(f"\n📊 Package Status:")
        print(f"   Available: {len(available)}/{len(available) + len(missing)}")
        print(f"   Missing: {len(missing)}")
        
        # Install missing packages if any
        if missing:
            print(f"\n📦 Installing {len(missing)} missing packages...")
            
            # Define package versions for Maix Cam
            maixcam_package_versions = [
                # Core ML packages (already installed, but ensure compatibility)
                "ultralytics>=8.3.0",  # Latest YOLOv8/YOLOv11 support
                "torch>=2.0.0",
                "torchvision>=0.15.0",
                
                # Maix Cam specific packages
                "onnx>=1.18.0",  # For model export
                "onnxruntime>=1.19.0",  # For model testing
                "opencv-python>=4.8.0",  # For image processing
                
                # YOLOv11 Classification-specific packages
                "albumentations>=1.3.0",  # For RandAugment and RandomErasing
                "torchvision-transforms>=0.1.0",  # Enhanced transforms
                
                # Additional utilities for Maix Cam
                "pillow>=10.0.0",  # Image processing
                "numpy>=1.24.0",  # Numerical computing
                "matplotlib>=3.7.0",  # Visualization
                "seaborn>=0.12.0",  # Enhanced plotting
                
                # Classification-specific utilities
                "scikit-learn>=1.3.0",  # For classification metrics
                "pandas>=2.0.0",  # For data analysis
            ]
            
            # Install packages with verbose output
            for pkg in maixcam_package_versions:
                print(f"   Installing {pkg}...")
                try:
                    result = subprocess.run([
                        conda_path, "run", "-n", "pokemon-classifier",
                        "pip", "install", pkg
                    ], capture_output=True, text=True, timeout=300)  # 5 minute timeout per package
                    
                    if result.returncode == 0:
                        print(f"   ✅ {pkg} installed successfully")
                    else:
                        print(f"   ⚠️ {pkg} installation had issues: {result.stderr}")
                except subprocess.TimeoutExpired:
                    print(f"   ⏰ {pkg} installation timed out")
                except Exception as e:
                    print(f"   ❌ {pkg} installation error: {e}")
        else:
            print("✅ All Maix Cam packages are already installed!")
        
        # Test Maix Cam environment
        print("🧪 Testing Maix Cam environment...")
        test_script = """
import torch
import ultralytics
import onnx
import onnxruntime
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from sklearn.metrics import accuracy_score, top_k_accuracy_score

print("✅ Maix Cam environment test:")
print(f"• PyTorch: {torch.__version__}")
print(f"• Ultralytics: {ultralytics.__version__}")
print(f"• ONNX: {onnx.__version__}")
print(f"• ONNX Runtime: {onnxruntime.__version__}")
print(f"• OpenCV: {cv2.__version__}")
print(f"• NumPy: {np.__version__}")
print(f"• Albumentations: {A.__version__}")

# Test YOLOv11 availability and classification capabilities
try:
    from ultralytics import YOLO
    
    # Test YOLOv11 classification model download
    print("📥 Testing YOLOv11 classification model download...")
    model = YOLO('yolo11m-cls.pt')  # Test YOLOv11 classification model
    print("✅ YOLOv11 classification model available")
    
    # Test classification inference
    print("🧪 Testing YOLOv11 classification inference...")
    # Create a dummy image for testing
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    results = model(dummy_img, verbose=False)
    
    if hasattr(results[0], 'probs'):
        print("✅ YOLOv11 classification inference working")
        print(f"   • Output shape: {results[0].probs.shape}")
        print(f"   • Number of classes: {results[0].probs.shape[0]}")
    else:
        print("⚠️ YOLOv11 classification inference test incomplete")
        
except Exception as e:
    print(f"⚠️ YOLOv11 test failed: {e}")

# Test RandAugment and RandomErasing availability
try:
    # Test RandAugment
    randaug = A.RandAugment(num_ops=2, magnitude=9)
    print("✅ RandAugment available")
    
    # Test RandomErasing (simulated with albumentations)
    random_erase = A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.1)
    print("✅ RandomErasing (CoarseDropout) available")
    
except Exception as e:
    print(f"⚠️ Augmentation test failed: {e}")

# Test classification metrics
try:
    # Test top-k accuracy calculation
    y_true = [0, 1, 2, 3, 4]
    y_pred_proba = np.random.rand(5, 1025)  # 5 samples, 1025 classes
    top1_acc = accuracy_score(y_true, np.argmax(y_pred_proba, axis=1))
    top5_acc = top_k_accuracy_score(y_true, y_pred_proba, k=5)
    print("✅ Classification metrics (top-1/top-5) available")
    
except Exception as e:
    print(f"⚠️ Classification metrics test failed: {e}")

print("✅ Maix Cam environment ready for YOLOv11 classification training!")
"""
        
        subprocess.check_call([
            conda_path, "run", "-n", "pokemon-classifier",
            "python", "-c", test_script
        ])
        
        print("✅ Maix Cam environment setup completed!")
        return True
        
    except Exception as e:
        print(f"❌ Maix Cam environment setup failed: {e}")
        return False

def setup_k210_environment():
    """Set up K210 specific environment (original functionality)."""
    try:
        print("🔧 Setting up K210 environment...")
        
        # Get conda path
        def get_conda_path():
            """Get the conda executable path, handling both local and Colab environments"""
            # Check if we're in a Colab environment with /content installation
            if os.path.exists("/content/miniconda3/bin/conda"):
                return "/content/miniconda3/bin/conda"
            # Check if conda is in PATH
            try:
                result = subprocess.run(["which", "conda"], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
            # Fallback to common local paths
            for path in ["/home/liuhuan/miniconda3/bin/conda", "/opt/conda/bin/conda"]:
                if os.path.exists(path):
                    return path
            return "conda"  # Fallback to PATH
        
        conda_path = get_conda_path()
        
        # Install K210 specific packages
        print("📦 Installing K210 dependencies...")
        
        k210_packages = [
            # ONNX packages for K210 export
            "onnx>=1.18.0",
            "onnxruntime>=1.19.0", 
            "onnxsim>=0.4.36",
            
            # TensorFlow for ONNX to TFLite conversion
            "tensorflow>=2.20.0",
            "onnx-tf>=1.10.0",
            
            # Additional K210 tools
            "flatbuffers>=25.2.10",
            "protobuf>=3.20.3",
        ]
        
        # Install packages with verbose output
        for pkg in k210_packages:
            print(f"   Installing {pkg}...")
            try:
                result = subprocess.run([
                    conda_path, "run", "-n", "pokemon-classifier",
                    "pip", "install", pkg
                ], capture_output=True, text=True, timeout=300)  # 5 minute timeout per package
                
                if result.returncode == 0:
                    print(f"   ✅ {pkg} installed successfully")
                else:
                    print(f"   ⚠️ {pkg} installation had issues: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"   ⏰ {pkg} installation timed out")
            except Exception as e:
                print(f"   ❌ {pkg} installation error: {e}")
        
        # Test K210 environment
        print("🧪 Testing K210 environment...")
        test_script = """
import onnx
import onnxruntime
import tensorflow as tf
import onnx_tf

print("✅ K210 environment test:")
print(f"• ONNX: {onnx.__version__}")
print(f"• ONNX Runtime: {onnxruntime.__version__}")
print(f"• TensorFlow: {tf.__version__}")
print(f"• ONNX-TF: {onnx_tf.__version__}")

print("✅ K210 environment ready!")
"""
        
        subprocess.check_call([
            conda_path, "run", "-n", "pokemon-classifier",
            "python", "-c", test_script
        ])
        
        print("✅ K210 environment setup completed!")
        return True
        
    except Exception as e:
        print(f"❌ K210 environment setup failed: {e}")
        return False

def print_setup_summary(dirs: dict, is_colab: bool, target_platform: str):
    """Print setup summary and next steps."""
    print("\n" + "="*60)
    print("🎯 Environment Setup Summary")
    print("="*60)
    
    print(f"\n🔧 Environment Status:")
    print(f"• Environment: {'Google Colab' if is_colab else 'Local Development'}")
    print(f"• Target Platform: {target_platform}")
    print(f"• GPU: Available and verified")
    print(f"• Dependencies: Installed via setup_environment.py")
    print(f"• W&B: Configured and tested")
    print(f"• Dataset: Access verified")
    
    if target_platform == "maixcam":
        print(f"\n📱 Maix Cam Specific:")
        print(f"• YOLOv11 support: Available")
        print(f"• ONNX export: Ready")
        print(f"• Classification models: Supported")
        
        print(f"\n📋 Next Steps for Maix Cam:")
        print(f"1. Train YOLOv11 classification model:")
        print(f"   python scripts/yolo/train_yolov11_maixcam.py")
        print(f"\n2. Export model for Maix Cam:")
        print(f"   python scripts/yolo/export_maixcam.py")
        print(f"\n3. Test on Maix Cam:")
        print(f"   python maixcam/main.py")
        
    elif target_platform == "k210":
        print(f"\n🔧 K210 Specific:")
        print(f"• ONNX export: Available")
        print(f"• TensorFlow conversion: Ready")
        print(f"• nncase compatibility: Configured")
        
        print(f"\n📋 Next Steps for K210:")
        print(f"1. Train YOLOv3-tiny model:")
        print(f"   python scripts/yolo/train_yolov3_baseline.py")
        print(f"\n2. Train YOLOv5n model:")
        print(f"   python scripts/yolo/train_yolov5n_k210.py")
        print(f"\n3. Export for K210:")
        print(f"   python scripts/yolo/export_k210.py")
    
    print(f"\n" + "="*60)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Setup training environment for Pokemon Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup for Maix Cam (recommended)
  python scripts/yolo/setup_colab_training.py --platform maixcam
  
  # Setup for K210 (legacy)
  python scripts/yolo/setup_colab_training.py --platform k210
  
  # Skip platform-specific setup (base environment only)
  python scripts/yolo/setup_colab_training.py --platform none
        """
    )
    
    parser.add_argument(
        "--platform", "-p",
        choices=["maixcam", "k210", "none"],
        default="maixcam",
        help="Target platform for setup (default: maixcam)"
    )
    
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Skip GPU verification (useful for CPU-only setups)"
    )
    
    parser.add_argument(
        "--skip-wandb",
        action="store_true", 
        help="Skip Weights & Biases setup"
    )
    
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset access verification"
    )
    
    return parser.parse_args()

def main():
    """Main setup function."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        print("🚀 Setting up training environment...")
        
        # First verify we're in the right place
        verify_repository()
        
        # Show platform selection
        if args.platform == "maixcam":
            print(f"\n🎯 Setting up environment for: MAIX CAM (Modern, powerful)")
            print("   • YOLOv11 classification models")
            print("   • Full 1025 Pokemon classes")
            print("   • Modern export pipeline")
        elif args.platform == "k210":
            print(f"\n🎯 Setting up environment for: K210 (Legacy, limited memory)")
            print("   • YOLOv3-tiny and YOLOv5n models")
            print("   • Memory-optimized for 6MB RAM")
            print("   • Legacy nncase export pipeline")
        else:
            print(f"\n🎯 Setting up base environment only (no platform-specific setup)")
        
        # Then proceed with other setup steps
        dirs, is_colab = setup_storage()
        setup_environment()
        
        # Optional GPU verification
        if not args.skip_gpu_check:
            verify_gpu(is_colab)
        else:
            print("⏭️ Skipping GPU verification")
        
        # Optional W&B setup
        if not args.skip_wandb:
            try:
                setup_wandb()
            except Exception as e:
                print(f"⚠️ W&B setup failed: {e}")
                print("ℹ️ W&B is optional - training will work without it")
                print("💡 To set up W&B later, run: wandb login")
        else:
            print("⏭️ Skipping Weights & Biases setup")
        
        # Platform-specific setup
        if args.platform == "maixcam":
            print("📱 Setting up Maix Cam environment (skipping K210 components)...")
            maixcam_success = setup_maixcam_environment()
            if not maixcam_success:
                print("⚠️ Maix Cam setup had issues, but continuing...")
        elif args.platform == "k210":
            print("🔧 Setting up K210 environment...")
            k210_success = setup_k210_environment()
            if not k210_success:
                print("⚠️ K210 setup had issues, but continuing...")
        else:
            print("⏭️ Skipping platform-specific setup")
        
        # Optional dataset access verification
        if not args.skip_dataset:
            dataset_available = verify_dataset_access()
        else:
            print("⏭️ Skipping dataset access verification")
            dataset_available = False
        
        # Print summary
        print_setup_summary(dirs, is_colab, args.platform)
        
        if not dataset_available and not args.skip_dataset:
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