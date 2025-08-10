#!/usr/bin/env python3
"""
Environment Setup Script
Sets up dependencies using conda for environment management and uv for Python dependencies
"""

import subprocess
import sys
import os
import shutil
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

def verify_k210_installation(env_name="pokemon-classifier"):
    """Verify that K210-specific packages and tools are installed correctly"""
    print("🔍 Verifying K210 tooling installation...")
    
    conda_path = get_conda_path()
    success = True
    
    # Check Python packages
    k210_packages = ["onnx", "onnxruntime", "onnxsim", "nncase"]
    for package in k210_packages:
        try:
            result = subprocess.run([
                conda_path, "run", "-n", env_name, "python", "-c", 
                f"import {package}; print('OK')"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  ✅ {package}")
            else:
                print(f"  ❌ {package}")
                success = False
        except Exception:
            print(f"  ❌ {package}")
            success = False
    
    # Check nncase Python API with actual compilation test
    try:
        # Create a tiny test ONNX model
        test_dir = Path.home() / ".k210_test"
        test_dir.mkdir(exist_ok=True)
        test_onnx = test_dir / "test.onnx"
        test_kmodel = test_dir / "test.kmodel"
        
        # Use onnx to create a tiny model
        import onnx
        from onnx import helper, TensorProto
        X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 2, 2])
        Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 2, 2])
        node = helper.make_node('Identity', ['input'], ['output'])
        graph = helper.make_graph([node], 'test', [X], [Y])
        model = helper.make_model(graph)
        onnx.save(model, str(test_onnx))
        
        # Try to compile with nncase Python API
        import nncase
        
        # Create compile options
        compile_options = nncase.CompileOptions()
        compile_options.target = "k210"
        compile_options.quant_type = "int8"
        compile_options.input_shape = [1, 1, 2, 2]
        compile_options.input_layout = "NCHW"
        compile_options.output_layout = "NCHW"
        compile_options.mean = [0.0]
        compile_options.std = [255.0]
        
        # Create compiler
        compiler = nncase.Compiler(compile_options)
        
        # Import ONNX
        import_options = nncase.ImportOptions()
        with open(test_onnx, 'rb') as f:
            onnx_bytes = f.read()
        compiler.import_onnx(onnx_bytes, import_options)
        
        # Compile
        compiler.compile()
        
        # Generate kmodel
        with open(test_kmodel, 'wb') as f:
            compiler.gencode(f)
        
        # Clean up test files
        shutil.rmtree(test_dir)
        
        print("  ✅ nncase Python API (compilation test passed)")
    except Exception as e:
        print(f"  ❌ nncase Python API test failed: {e}")
        success = False
    
    # Check K210 runtime
    runtime_dir = Path.home() / "nncaseruntime-k210"
    if runtime_dir.exists() and runtime_dir.is_dir():
        print("  ✅ K210 runtime")
    else:
        print("  ❌ K210 runtime not found")
        success = False
    
    if success:
        print("\n✅ All K210 dependencies are installed correctly!")
    else:
        print("\n⚠️ Some K210 dependencies are missing or not working correctly")
    
    return success

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

def install_k210_dependencies(env_name="pokemon-classifier"):
    """Install K210 export dependencies including nncase"""
    print("📦 Installing K210 export dependencies...")
    try:
        conda_path = get_conda_path()
        # Install nncase and related packages for K210 compilation
        subprocess.check_call([
            conda_path, "run", "-n", env_name, "uv", "pip", "install", 
            "onnx", "onnxruntime", "onnxsim", "nncase"
        ])
        print("✅ K210 export dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install K210 dependencies: {e}")
        return False

def install_dotnet_sdk_if_missing():
    """Install .NET SDK if missing (required for nncase CLI)"""
    print("🔧 Checking for .NET SDK...")
    try:
        result = subprocess.run(["dotnet", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ .NET SDK already available (version {result.stdout.strip()})")
            return True
    except FileNotFoundError:
        pass
    
    print("📦 Installing .NET SDK...")
    try:
        # Detect Ubuntu version and add Microsoft repository
        ubuntu_version = None
        try:
            with open("/etc/os-release", "r") as f:
                for line in f:
                    if line.startswith("VERSION_ID="):
                        ubuntu_version = line.split("=")[1].strip().strip('"')
                        break
        except:
            pass
        
        # Add Microsoft repository
        subprocess.check_call([
            "wget", "https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb", 
            "-O", "packages-microsoft-prod.deb"
        ])
        subprocess.check_call(["sudo", "dpkg", "-i", "packages-microsoft-prod.deb"])
        subprocess.check_call(["rm", "packages-microsoft-prod.deb"])
        
        # Update package list
        subprocess.check_call(["sudo", "apt-get", "update"])
        
        # Install .NET SDK (try 7.0 first, fallback to 8.0)
        try:
            subprocess.check_call(["sudo", "apt-get", "install", "-y", "dotnet-sdk-7.0"])
        except subprocess.CalledProcessError:
            subprocess.check_call(["sudo", "apt-get", "install", "-y", "dotnet-sdk-8.0"])
        
        # Set environment variables
        os.environ["DOTNET_ROOT"] = "/usr/share/dotnet"
        if "/usr/share/dotnet" not in os.environ.get("PATH", ""):
            os.environ["PATH"] = "/usr/share/dotnet:" + os.environ.get("PATH", "")
        
        print("✅ .NET SDK installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install .NET SDK: {e}")
        return False

def install_k210_tooling(env_name="pokemon-classifier"):
    """Install K210 tooling including nncase compiler"""
    print("🔧 Installing K210 tooling...")
    
    # Install nncase Python package
    if not install_k210_dependencies(env_name):
        return False
    
    # Try to install .NET SDK for nncase CLI
    install_dotnet_sdk_if_missing()
    
    # Try to find existing ncc in common locations first
    print("🔍 Checking for existing nncase compiler...")
    ncc_candidates = [
        "ncc",
        os.path.expanduser("~/.local/bin/ncc"),
        os.path.expanduser("~/.dotnet/tools/ncc"),
        "/usr/local/bin/ncc",
        "/usr/bin/ncc",
        os.path.expanduser("~/miniconda3/bin/ncc"),
        os.path.expanduser("~/anaconda3/bin/ncc"),
    ]
    
    for candidate in ncc_candidates:
        try:
            result = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and "nncase" in result.stdout.lower():
                print(f"✅ Found nncase compiler at {candidate}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            continue
    
    # Install K210 runtime and nncase v1.6.0 first
    print("📦 Installing K210 runtime and nncase v1.6.0...")
    try:
        conda_path = get_conda_path()
        # First uninstall any existing nncase to avoid conflicts
        subprocess.run([
            conda_path, "run", "-n", env_name, "uv", "pip", "uninstall", "nncase", "-y"
        ], capture_output=True)
        
        # Download and extract K210 runtime
        runtime_url = "https://github.com/kendryte/nncase/releases/download/v1.6.0/nncaseruntime-k210.zip"
        runtime_zip = Path.home() / "nncaseruntime-k210.zip"
        runtime_dir = Path.home() / "nncaseruntime-k210"
        
        if not runtime_dir.exists():
            print("📥 Downloading K210 runtime...")
            subprocess.check_call(["wget", "-O", str(runtime_zip), runtime_url])
            
            # Extract runtime
            subprocess.check_call(["unzip", "-o", str(runtime_zip), "-d", str(runtime_dir)])
            runtime_zip.unlink()  # Clean up zip file
            
            # Add to PATH
            os.environ["PATH"] = f"{runtime_dir}/bin:{os.environ.get('PATH', '')}"
            
            # Create symlinks in ~/.local/bin
            local_bin = Path.home() / ".local" / "bin"
            local_bin.mkdir(parents=True, exist_ok=True)
            for binary in runtime_dir.glob("bin/*"):
                symlink = local_bin / binary.name
                if symlink.exists():
                    symlink.unlink()
                symlink.symlink_to(binary)
            
            print("✅ K210 runtime installed!")
        
        # Install nncase v1.6.0 which supports K210
        subprocess.check_call([
            conda_path, "run", "-n", env_name, "uv", "pip", "install", 
            "https://github.com/kendryte/nncase/releases/download/v1.6.0/nncase-1.6.0.20220505-cp39-cp39-manylinux_2_24_x86_64.whl"
        ])
        print("✅ nncase v1.6.0 installed!")
        
        # Check if ncc is now available
        result = subprocess.run([conda_path, "run", "-n", env_name, "which", "ncc"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Found ncc at {result.stdout.strip()}")
            return True
            
        # Test the installation
        try:
            result = subprocess.run(["ncc", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ nncase test successful: {result.stdout.strip()}")
                return True
            else:
                print(f"⚠️ nncase test failed: {result.stderr}")
        except Exception as test_e:
            print(f"⚠️ nncase test failed: {test_e}")
        
        return True
            
    except Exception as e:
        print(f"⚠️ K210 tooling installation failed: {e}")
    
    # Create Python wrapper as last resort
    print("🐍 Creating Python nncase wrapper...")
    try:
        wrapper_path = create_nncase_wrapper(env_name)
        print(f"✅ Created nncase wrapper at {wrapper_path}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to create nncase wrapper: {e}")
    
    print("\n⚠️ nncase compiler (ncc) installation failed.")
    print("\n📋 Manual Installation Options:")
    print("1. Download working nncase binary (recommended):")
    print("   cd ~")
    print("   wget https://github.com/kendryte/nncase_old/releases/download/v1.0/ncc-linux-x86_64.tar.xz")
    print("   tar -xJf ncc-linux-x86_64.tar.xz")
    print("   mv ncc-linux-x86_64 ncc_k210")
    print("   export PATH=$HOME/ncc_k210:$PATH")
    print("   # Optional: add to ~/.bashrc for permanent access")
    print("2. Install nncase Python package:")
    print("   pip install nncase==1.6.0")
    print("3. Build from source:")
    print("   git clone https://github.com/kendryte/nncase.git")
    print("   cd nncase && mkdir build && cd build")
    print("   cmake .. && make -j$(nproc)")
    print("   sudo make install")
    print("\n💡 The export script will use the CLI approach to avoid Python API issues.")
    print("   Make sure ncc is in PATH or use --ncc /path/to/ncc")
    
    return False

def create_nncase_wrapper(env_name="pokemon-classifier"):
    """Create a Python wrapper for nncase that uses the Python API"""
    wrapper_script = '''#!/usr/bin/env python3
"""
nncase Python API wrapper
Provides a CLI interface to nncase Python API for K210 compilation
"""

import sys
import argparse
import tempfile
import shutil
import os
from pathlib import Path

def compile_onnx_to_k210(
    onnx_path,
    kmodel_path,
    input_shape=[1, 3, 320, 320],
    input_layout='NCHW',
    mean_values=[0.0, 0.0, 0.0],
    std_values=[255.0, 255.0, 255.0],
    quant_type='int8',
    dataset_path=None
):
    """Compile ONNX model to K210 kmodel using nncase Python API"""
    import nncase
    
    # Create compile options
    compile_options = nncase.CompileOptions()
    compile_options.target = "k210"
    compile_options.quant_type = quant_type
    compile_options.input_shape = input_shape
    compile_options.input_layout = input_layout
    compile_options.output_layout = input_layout
    compile_options.mean = mean_values
    compile_options.std = std_values
    
    # Create compiler
    compiler = nncase.Compiler(compile_options)
    
    # Import ONNX
    import_options = nncase.ImportOptions()
    with open(onnx_path, 'rb') as f:
        onnx_bytes = f.read()
    compiler.import_onnx(onnx_bytes, import_options)
    
    # Compile
    compiler.compile()
    
    # Save kmodel
    try:
        with open(kmodel_path, 'wb') as f:
            compiler.gencode(f)
        print(f"Successfully compiled {onnx_path} to {kmodel_path}")
        return True
    except Exception as e:
        print(f"Direct gencode failed: {e}", file=sys.stderr)
        # Try alternative approach - write to temporary file first
        try:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp_f:
                compiler.gencode(tmp_f)
                tmp_f.flush()
                shutil.copy2(tmp_f.name, kmodel_path)
                os.unlink(tmp_f.name)
            print(f"Successfully compiled {onnx_path} to {kmodel_path} (using temp file)")
            return True
        except Exception as e2:
            print(f"Failed to compile using temp file: {e2}", file=sys.stderr)
            return False

def main():
    parser = argparse.ArgumentParser(description='nncase Python API wrapper')
    parser.add_argument('compile', help='compile command')
    parser.add_argument('input', help='input ONNX file')
    parser.add_argument('output', help='output kmodel file')
    parser.add_argument('--dataset', help='calibration dataset directory')
    parser.add_argument('--input-mean', help='input mean values (comma-separated)')
    parser.add_argument('--input-std', help='input std values (comma-separated)')
    parser.add_argument('--input-layout', default='NCHW', help='input layout')
    parser.add_argument('--shape', help='input shape (comma-separated)')
    parser.add_argument('--quanttype', default='int8', help='quantization type')
    
    args = parser.parse_args()
    
    if args.compile != 'compile':
        print("Only 'compile' command supported", file=sys.stderr)
        sys.exit(1)
    
    # Parse parameters
    input_shape = [1, 3, 320, 320]
    if args.shape:
        input_shape = [int(x) for x in args.shape.split(',')]
    
    mean_values = [0.0, 0.0, 0.0]
    if args.input_mean:
        mean_values = [float(x) for x in args.input_mean.split(',')]
        if len(mean_values) == 1:
            mean_values = [mean_values[0]] * 3
    
    std_values = [255.0, 255.0, 255.0]
    if args.input_std:
        std_values = [float(x) for x in args.input_std.split(',')]
        if len(std_values) == 1:
            std_values = [std_values[0]] * 3
    
    success = compile_onnx_to_k210(
        onnx_path=args.input,
        kmodel_path=args.output,
        input_shape=input_shape,
        input_layout=args.input_layout,
        mean_values=mean_values,
        std_values=std_values,
        quant_type=args.quanttype,
        dataset_path=args.dataset
    )
    
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()
'''
    
    # Create wrapper in conda environment bin directory
    conda_path = get_conda_path()
    result = subprocess.run([conda_path, "run", "-n", env_name, "python", "-c", 
                           "import sys; print(sys.prefix)"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        env_prefix = result.stdout.strip()
        wrapper_path = Path(env_prefix) / "bin" / "ncc"
    else:
        # Fallback to user local bin
        wrapper_path = Path.home() / ".local" / "bin" / "ncc"
    
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_script)
    
    wrapper_path.chmod(0o755)
    return wrapper_path

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
    parser.add_argument("--k210-only", action="store_true",
                       help="Only install K210 dependencies and verify their installation")
    
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
    print(f"  • K210 only: {args.k210_only}")
    
    # Set up Colab environment if requested
    if args.colab:
        setup_colab_environment()
    
    # Create conda environment (unless skipped)
    if not args.skip_env:
        if not create_conda_environment(args.env_name, args.python_version):
            return
    
    if args.k210_only:
        # Skip uv installation for k210-only mode, assume it's installed
        # Only install and verify K210 dependencies
        print("\n📦 Installing K210 export dependencies...")
        k210_success = install_k210_tooling(args.env_name)
        if not k210_success:
            print("\n⚠️ K210 tooling installation had issues")
            print("💡 You can:")
            print("   1. Try running this script again")
            print("   2. Install nncase manually using the instructions shown above")
            print("   3. Use the export script with --ncc /path/to/ncc once installed")
            return
        
        # Verify K210 installation
        if not verify_k210_installation(args.env_name):
            return
        
        print("\n🎉 K210 tooling setup complete!")
        print("\n📋 Next steps:")
        print("  • Export trained model: python scripts/yolo/export_k210.py --weights model.pt --calib-dir calib")
        print("  • Deploy kmodel and classes.txt to Maix Bit SD card")
        return

    # Install uv if not already installed
    if not install_uv():
        return
    
    # Regular installation flow
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
        
        # Also install K210 export dependencies for YOLO models
        print("\n📦 Installing K210 export dependencies...")
        k210_success = install_k210_tooling(args.env_name)
        if not k210_success:
            print("\n⚠️ K210 tooling installation had issues, but YOLO training will still work")
            print("💡 For K210 deployment, you can:")
            print("   1. Run the export script - it will create a Python wrapper if needed")
            print("   2. Install nncase manually using the instructions shown above")
            print("   3. Use the export script with --ncc /path/to/ncc once installed")
        else:
            print("✅ K210 tooling installed successfully!")
            verify_k210_installation(args.env_name)
    
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
    
    if args.experiment == "yolo":
        print(f"\n🚀 For K210 deployment:")
        print(f"  • Export trained model: python scripts/yolo/export_k210.py --weights model.pt --calib-dir calib")
        print(f"  • The export script will handle nncase installation automatically")
        print(f"  • Deploy kmodel and classes.txt to Maix Bit SD card")
    
    print(f"\n💡 For Google Colab:")
    print(f"  • Use the --colab flag for automatic conda/uv setup")
    print(f"  • Environment will be ready for immediate use")
    print(f"  • Dataset upload script ready for Hugging Face integration")

if __name__ == "__main__":
    main() 