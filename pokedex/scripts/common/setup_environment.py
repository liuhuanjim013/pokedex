#!/usr/bin/env python3
"""
Environment Setup Script
Sets up dependencies using conda for environment management and uv for Python dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def get_project_root():
    """Get the project root directory (pokedex/)"""
    current_dir = Path.cwd()
    # Navigate up to find the pokedex directory
    while current_dir.name != "pokedex" and current_dir.parent != current_dir:
        current_dir = current_dir.parent
    return current_dir

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

def create_conda_environment(env_name="pokemon-classifier", python_version="3.9"):
    """Create conda environment for the project"""
    print(f"🐍 Creating conda environment: {env_name}")
    try:
        conda_path = get_conda_path()
        # Check if environment already exists
        result = subprocess.run([conda_path, "env", "list"], capture_output=True, text=True)
        if env_name in result.stdout:
            print(f"✅ Environment {env_name} already exists")
            return True
        
        # Create new environment
        subprocess.check_call([
            conda_path, "create", "-n", env_name, f"python={python_version}", "-y"
        ])
        print(f"✅ Conda environment {env_name} created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create conda environment: {e}")
        return False

def install_uv():
    """Install uv using conda"""
    print("📦 Installing uv...")
    try:
        conda_path = get_conda_path()
        subprocess.check_call([conda_path, "install", "-c", "conda-forge", "uv", "-y"])
        print("✅ uv installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install uv: {e}")
        return False

def install_dependencies_with_uv(requirements_file, env_name="pokemon-classifier"):
    """Install dependencies using uv in the specified conda environment"""
    print(f"📦 Installing dependencies from {requirements_file}...")
    try:
        conda_path = get_conda_path()
        # Use conda run to execute uv in the specified environment
        subprocess.check_call([
            conda_path, "run", "-n", env_name, "uv", "pip", "install", "-r", requirements_file
        ])
        print(f"✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def accept_conda_tos():
    """Accept conda Terms of Service for required channels."""
    print("📜 Accepting conda Terms of Service...")
    conda_path = get_conda_path()
    channels = [
        "https://repo.anaconda.com/pkgs/main",
        "https://repo.anaconda.com/pkgs/r",
        "conda-forge"  # Also accept for conda-forge which we'll use
    ]
    
    for channel in channels:
        try:
            subprocess.check_call([conda_path, "tos", "accept", "--override-channels", "--channel", channel])
            print(f"✅ Accepted ToS for {channel}")
        except subprocess.CalledProcessError as e:
            if "Unknown command" in str(e):
                # Older conda versions don't have tos command, try to proceed
                print("ℹ️ Older conda version detected, attempting to proceed without ToS acceptance")
                return True
            print(f"⚠️ Failed to accept ToS for {channel}: {e}")
            return False
    return True

def setup_colab_environment():
    """Set up environment for Google Colab"""
    print("☁️  Setting up Google Colab environment...")
    
    # Check if conda is already available
    conda_path = get_conda_path()
    try:
        subprocess.check_call([conda_path, "--version"])
        print("✅ Conda is already available")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("🔍 Conda not found, will install...")
    
    # Check if installer is already downloaded
    installer = "Miniconda3-latest-Linux-x86_64.sh"
    if not os.path.exists(installer):
        print("📥 Downloading Miniconda installer...")
        subprocess.check_call([
            "wget", f"https://repo.anaconda.com/miniconda/{installer}"
        ])
    else:
        print("✅ Miniconda installer already downloaded")
    
    # Install Miniconda
    print("📦 Installing conda...")
    
    # Check if we're in a Colab environment and use Google Drive for persistence
    conda_install_path = "/content/miniconda3"
    print("📁 Installing conda in /content for local environment...")
    
    # Check if conda directory already exists
    if os.path.exists(conda_install_path):
        print(f"✅ Conda already installed at {conda_install_path}")
        # Add conda to PATH if not already there
        if f"{conda_install_path}/bin" not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{conda_install_path}/bin:" + os.environ.get("PATH", "")
    else:
        # Install Miniconda only if directory doesn't exist
        subprocess.check_call([
            "bash", installer, "-b", "-p", conda_install_path
        ])
        
        # Add conda to PATH
        os.environ["PATH"] = f"{conda_install_path}/bin:" + os.environ.get("PATH", "")
        
        # Initialize conda for shell
        subprocess.check_call([conda_path, "init"])
        
        print("✅ Conda installed successfully")
    
    # Ensure conda is in PATH for all subsequent operations
    if f"{conda_install_path}/bin" not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{conda_install_path}/bin:" + os.environ.get("PATH", "")
    
    # Clean up installer
    if os.path.exists(installer):
        os.remove(installer)
    
    # Accept Terms of Service
    if not accept_conda_tos():
        raise RuntimeError("Failed to accept conda Terms of Service")
    
    # Install uv
    install_uv()

def verify_installation(env_name="pokemon-classifier"):
    """Verify that key packages are installed in the conda environment"""
    print("🔍 Verifying installation...")
    
    conda_path = get_conda_path()
    required_packages = [
        # Core ML packages
        "numpy", "pandas", "matplotlib", "seaborn", 
        "PIL", "cv2", "torch", "transformers",
        "datasets", "huggingface_hub", "ultralytics",
        # Network resilience packages
        "backoff", "requests", "urllib3",
        # Progress tracking
        "tqdm", "rich"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Use conda run to check imports in the specified environment
            result = subprocess.run([
                conda_path, "run", "-n", env_name, "python", "-c", 
                f"import {package}; print('OK')"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  ✅ {package}")
            else:
                print(f"  ❌ {package}")
                missing_packages.append(package)
        except Exception:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True

def install_huggingface_dependencies(env_name="pokemon-classifier"):
    """Install Hugging Face dependencies for dataset upload"""
    print("📦 Installing Hugging Face dependencies...")
    try:
        conda_path = get_conda_path()
        # Install datasets and huggingface_hub
        subprocess.check_call([
            conda_path, "run", "-n", env_name, "uv", "pip", "install", 
            "datasets", "huggingface_hub",
            # Network resilience packages
            "backoff", "requests", "urllib3",
            # Progress tracking
            "tqdm", "rich"
        ])
        print("✅ Hugging Face dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Hugging Face dependencies: {e}")
        return False

def install_yolo_dependencies(env_name="pokemon-classifier"):
    """Install YOLO training dependencies"""
    print("📦 Installing YOLO training dependencies...")
    try:
        conda_path = get_conda_path()
        # Install ultralytics for YOLO training
        subprocess.check_call([
            conda_path, "run", "-n", env_name, "uv", "pip", "install", 
            "ultralytics", "wandb"
        ])
        print("✅ YOLO training dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install YOLO dependencies: {e}")
        return False

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up Pokemon Classifier environment")
    parser.add_argument("--env-name", default="pokemon-classifier", 
                       help="Conda environment name")
    parser.add_argument("--python-version", default="3.9", 
                       help="Python version for conda environment")
    parser.add_argument("--experiment", choices=["yolo", "vlm", "hybrid"], 
                       help="Experiment type to set up")
    parser.add_argument("--colab", action="store_true", 
                       help="Set up for Google Colab environment")
    parser.add_argument("--verify", action="store_true", 
                       help="Verify installation after setup")
    parser.add_argument("--skip-env", action="store_true", 
                       help="Skip conda environment creation (use existing)")
    
    args = parser.parse_args()
    
    # Get project root and ensure we're in the right directory
    project_root = get_project_root()
    if not project_root.exists():
        print("❌ Error: Could not find pokedex project root directory")
        return
    
    print("🚀 Setting up Pokemon Classifier environment...")
    print(f"📋 Configuration:")
    print(f"  • Project root: {project_root}")
    print(f"  • Environment: {args.env_name}")
    print(f"  • Python version: {args.python_version}")
    print(f"  • Experiment: {args.experiment}")
    print(f"  • Colab setup: {args.colab}")
    
    # Set up Colab environment if requested
    if args.colab:
        setup_colab_environment()
    
    # Create conda environment (unless skipped)
    if not args.skip_env:
        if not create_conda_environment(args.env_name, args.python_version):
            return
    
    # Install uv if not already installed
    if not install_uv():
        return
    
    # Install base requirements
    print("\n📦 Installing base requirements...")
    base_requirements = project_root / "requirements.txt"
    if not base_requirements.exists():
        print(f"❌ Base requirements file not found: {base_requirements}")
        return
    
    if not install_dependencies_with_uv(str(base_requirements), args.env_name):
        return
    
    # Install experiment-specific requirements
    if args.experiment:
        exp_requirements = project_root / "requirements" / f"{args.experiment}_requirements.txt"
        if not exp_requirements.exists():
            print(f"❌ Experiment requirements file not found: {exp_requirements}")
            return
        
        if not install_dependencies_with_uv(str(exp_requirements), args.env_name):
            return
    
    # Install additional dependencies based on experiment type
    if args.experiment == "yolo":
        print("\n📦 Installing YOLO training dependencies...")
        if not install_yolo_dependencies(args.env_name):
            return
    
    # Install Hugging Face dependencies for dataset upload
    print("\n📦 Installing Hugging Face dependencies...")
    if not install_huggingface_dependencies(args.env_name):
        return
    
    # Verify installation
    if args.verify or args.experiment:
        verify_installation(args.env_name)
    
    print("\n🎉 Environment setup complete!")
    print(f"\n📋 Next steps:")
    print(f"  1. Activate environment: conda activate {args.env_name}")
    print(f"  2. Run dataset analysis: python scripts/common/dataset_analysis.py")
    print(f"  3. Upload dataset to Hugging Face: python scripts/common/upload_yolo_dataset.py")
    print(f"  4. Set up experiment: python scripts/common/experiment_manager.py")
    print(f"  5. Start training: python scripts/yolo/setup_yolov3_experiment.py")
    
    print(f"\n💡 For Google Colab:")
    print(f"  • Use the --colab flag for automatic conda/uv setup")
    print(f"  • Environment will be ready for immediate use")
    print(f"  • Dataset upload script ready for Hugging Face integration")

if __name__ == "__main__":
    main() 