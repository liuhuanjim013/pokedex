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

def create_conda_environment(env_name="pokemon-classifier", python_version="3.9"):
    """Create conda environment for the project"""
    print(f"🐍 Creating conda environment: {env_name}")
    try:
        # Check if environment already exists
        result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
        if env_name in result.stdout:
            print(f"✅ Environment {env_name} already exists")
            return True
        
        # Create new environment
        subprocess.check_call([
            "conda", "create", "-n", env_name, f"python={python_version}", "-y"
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
        subprocess.check_call(["conda", "install", "-c", "conda-forge", "uv", "-y"])
        print("✅ uv installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install uv: {e}")
        return False

def install_dependencies_with_uv(requirements_file, env_name="pokemon-classifier"):
    """Install dependencies using uv in the specified conda environment"""
    print(f"📦 Installing dependencies from {requirements_file}...")
    try:
        # Use conda run to execute uv in the specified environment
        subprocess.check_call([
            "conda", "run", "-n", env_name, "uv", "pip", "install", "-r", requirements_file
        ])
        print(f"✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_colab_environment():
    """Set up environment for Google Colab"""
    print("☁️  Setting up Google Colab environment...")
    
    # Install conda if not available
    try:
        subprocess.check_call(["conda", "--version"])
        print("✅ Conda is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 Installing conda...")
        subprocess.check_call([
            "wget", "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        ])
        subprocess.check_call([
            "bash", "Miniconda3-latest-Linux-x86_64.sh", "-b", "-p", "/usr/local"
        ])
        # Add conda to PATH
        os.environ["PATH"] = "/usr/local/bin:" + os.environ.get("PATH", "")
    
    # Install uv
    install_uv()

def verify_installation(env_name="pokemon-classifier"):
    """Verify that key packages are installed in the conda environment"""
    print("🔍 Verifying installation...")
    
    required_packages = [
        "numpy", "pandas", "matplotlib", "seaborn", 
        "PIL", "cv2", "torch", "transformers"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Use conda run to check imports in the specified environment
            result = subprocess.run([
                "conda", "run", "-n", env_name, "python", "-c", 
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
    
    # Verify installation
    if args.verify or args.experiment:
        verify_installation(args.env_name)
    
    print("\n🎉 Environment setup complete!")
    print(f"\n📋 Next steps:")
    print(f"  1. Activate environment: conda activate {args.env_name}")
    print(f"  2. Run dataset analysis: python scripts/common/dataset_analysis.py")
    print(f"  3. Set up experiment: python scripts/common/experiment_manager.py")
    print(f"  4. Start training: python scripts/yolo/setup_yolov3_experiment.py")
    
    print(f"\n💡 For Google Colab:")
    print(f"  • Use the --colab flag for automatic conda/uv setup")
    print(f"  • Environment will be ready for immediate use")

if __name__ == "__main__":
    main() 