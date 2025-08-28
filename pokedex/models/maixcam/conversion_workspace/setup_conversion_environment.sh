#!/bin/bash

# MaixCam Conversion Environment Setup Script
# Based on: https://wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam.html

set -e

echo "🚀 Setting up MaixCam conversion environment..."
echo "📖 Following: https://wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam.html"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is designed for Linux systems"
    print_info "For other systems, please follow the manual installation guide"
    exit 1
fi

print_info "Detected Linux system - proceeding with setup..."

# Check Python version
print_info "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    print_status "Python $python_version is compatible (>= $required_version)"
else
    print_error "Python $python_version is too old. Required: >= $required_version"
    exit 1
fi

# Install system dependencies
print_info "Installing system dependencies..."
sudo apt-get update

# Install required packages
packages=(
    "build-essential"
    "cmake"
    "git"
    "wget"
    "curl"
    "unzip"
    "python3-pip"
    "python3-dev"
    "python3-venv"
    "libopencv-dev"
    "libopencv-contrib-dev"
    "libatlas-base-dev"
    "libblas-dev"
    "liblapack-dev"
    "libhdf5-dev"
    "libhdf5-serial-dev"
    "libhdf5-103"
    "libqtgui4"
    "libqtwebkit4"
    "libqt4-test"
    "python3-pyqt5"
    "libgstreamer1.0-0"
    "libgstreamer-plugins-base1.0-0"
    "libgtk-3-0"
    "libavcodec-dev"
    "libavformat-dev"
    "libswscale-dev"
    "libv4l-dev"
    "libxvidcore-dev"
    "libx264-dev"
    "libjpeg-dev"
    "libpng-dev"
    "libtiff-dev"
    "libatlas-base-dev"
    "gfortran"
    "libgstreamer-plugins-bad1.0-0"
    "libgstreamer-plugins-good1.0-0"
    "libgstreamer-plugins-ugly1.0-0"
    "libgstreamer1.0-dev"
    "libgstreamer-plugins-base1.0-dev"
    "libgstreamer-plugins-bad1.0-dev"
    "libgstreamer-plugins-good1.0-dev"
    "libgstreamer-plugins-ugly1.0-dev"
    "libgstreamer1.0-tools"
    "libgstreamer-plugins-base1.0-tools"
    "libgstreamer-plugins-bad1.0-tools"
    "libgstreamer-plugins-good1.0-tools"
    "libgstreamer-plugins-ugly1.0-tools"
    "libgstreamer1.0-0-dbg"
    "libgstreamer-plugins-base1.0-0-dbg"
    "libgstreamer-plugins-bad1.0-0-dbg"
    "libgstreamer-plugins-good1.0-0-dbg"
    "libgstreamer-plugins-ugly1.0-0-dbg"
    "libgstreamer1.0-dev-dbg"
    "libgstreamer-plugins-base1.0-dev-dbg"
    "libgstreamer-plugins-bad1.0-dev-dbg"
    "libgstreamer-plugins-good1.0-dev-dbg"
    "libgstreamer-plugins-ugly1.0-dev-dbg"
    "libgstreamer1.0-tools-dbg"
    "libgstreamer-plugins-base1.0-tools-dbg"
    "libgstreamer-plugins-bad1.0-tools-dbg"
    "libgstreamer-plugins-good1.0-tools-dbg"
    "libgstreamer-plugins-ugly1.0-tools-dbg"
)

for package in "${packages[@]}"; do
    print_info "Installing $package..."
    sudo apt-get install -y "$package" || print_warning "Failed to install $package"
done

# Create virtual environment
print_info "Creating Python virtual environment..."
python3 -m venv maixcam_env
source maixcam_env/bin/activate

print_status "Virtual environment created and activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
print_info "Installing Python dependencies..."

# Core ML dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime onnxsim
pip install opencv-python opencv-contrib-python
pip install numpy scipy matplotlib seaborn
pip install pillow tqdm pyyaml

# MaixCam specific dependencies
pip install maixpy
pip install maixcdk

# Additional dependencies for conversion
pip install tensorflow==2.20.0
pip install keras
pip install h5py
pip install protobuf
pip install six
pip install future
pip install requests
pip install urllib3
pip install certifi
pip install chardet
pip install idna
pip install packaging
pip install setuptools
pip install wheel
pip install Cython

# Install MaixCam converter tools
print_info "Setting up MaixCam converter tools..."

# Create converter directory
mkdir -p maixcam_tools
cd maixcam_tools

# Download MaixCam converter (if available)
print_info "Attempting to download MaixCam converter tools..."
if command -v wget &> /dev/null; then
    # Try to download converter tools from Sipeed
    wget -O maixcam_converter.zip "https://dl.sipeed.com/MAIX/MaixPy/release/master/maixcam_converter.zip" || print_warning "Could not download converter tools"
    
    if [ -f maixcam_converter.zip ]; then
        unzip maixcam_converter.zip || print_warning "Could not extract converter tools"
        print_status "Converter tools downloaded and extracted"
    fi
else
    print_warning "wget not available, skipping automatic download"
fi

cd ..

# Set up environment variables
print_info "Setting up environment variables..."

# Add to .bashrc
echo "" >> ~/.bashrc
echo "# MaixCam Conversion Environment" >> ~/.bashrc
echo "export MAIXCAM_ENV_PATH=\"$(pwd)/maixcam_env\"" >> ~/.bashrc
echo "export MAIXCAM_TOOLS_PATH=\"$(pwd)/maixcam_tools\"" >> ~/.bashrc
echo "export PATH=\"\$PATH:\$(pwd)/maixcam_tools\"" >> ~/.bashrc
echo "alias activate_maixcam='source $(pwd)/maixcam_env/bin/activate'" >> ~/.bashrc

print_status "Environment variables added to ~/.bashrc"

# Create activation script
cat > activate_maixcam_env.sh << 'EOF'
#!/bin/bash
# MaixCam Environment Activation Script

echo "🚀 Activating MaixCam conversion environment..."

# Activate virtual environment
source maixcam_env/bin/activate

# Set environment variables
export MAIXCAM_ENV_PATH="$(pwd)/maixcam_env"
export MAIXCAM_TOOLS_PATH="$(pwd)/maixcam_tools"
export PATH="$PATH:$(pwd)/maixcam_tools"

echo "✅ MaixCam environment activated!"
echo "📁 Environment path: $MAIXCAM_ENV_PATH"
echo "🔧 Tools path: $MAIXCAM_TOOLS_PATH"
echo ""
echo "📋 Available commands:"
echo "  - convert_model: MaixCam model converter"
echo "  - python: Python with all dependencies"
echo "  - pip: Package manager"
echo ""
echo "🎯 Next steps:"
echo "  1. Run: ./convert_yolov11_pokemon_to_cvimodel.sh"
echo "  2. Or manually: convert_model --help"
EOF

chmod +x activate_maixcam_env.sh

# Create test script
cat > test_conversion_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify MaixCam conversion environment setup
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name}: OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: FAILED - {e}")
        return False

def main():
    print("🧪 Testing MaixCam conversion environment...")
    print("=" * 50)
    
    # Test core dependencies
    core_modules = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("onnx", "ONNX"),
        ("onnxruntime", "ONNX Runtime"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("tensorflow", "TensorFlow"),
        ("keras", "Keras"),
    ]
    
    success_count = 0
    total_count = len(core_modules)
    
    for module, name in core_modules:
        if test_import(module, name):
            success_count += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {success_count}/{total_count} modules working")
    
    if success_count == total_count:
        print("🎉 All dependencies installed successfully!")
        print("🚀 Ready for MaixCam conversion!")
    else:
        print("⚠️  Some dependencies failed to install")
        print("💡 Try running: pip install -r requirements.txt")
    
    return success_count == total_count

if __name__ == "__main__":
    main()
EOF

chmod +x test_conversion_setup.py

# Create requirements file
cat > requirements.txt << 'EOF'
# MaixCam Conversion Dependencies
# Based on: https://wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam.html

# Core ML frameworks
torch>=1.12.0
torchvision>=0.13.0
torchaudio>=0.12.0

# ONNX ecosystem
onnx>=1.12.0
onnxruntime>=1.12.0
onnxsim>=0.1.0

# Computer vision
opencv-python>=4.5.0
opencv-contrib-python>=4.5.0

# Scientific computing
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Image processing
pillow>=8.3.0

# Utilities
tqdm>=4.62.0
pyyaml>=5.4.0

# TensorFlow (for alternative conversion)
tensorflow>=2.8.0
keras>=2.8.0

# Additional dependencies
h5py>=3.6.0
protobuf>=3.19.0
six>=1.16.0
future>=0.18.0
requests>=2.27.0
urllib3>=1.26.0
certifi>=2021.10.0
chardet>=4.0.0
idna>=3.3.0
packaging>=21.0.0
setuptools>=60.0.0
wheel>=0.37.0
Cython>=0.29.0

# MaixCam specific
maixpy>=0.1.0
maixcdk>=0.1.0
EOF

print_status "Requirements file created: requirements.txt"

# Final setup summary
echo ""
echo "🎉 MaixCam Conversion Environment Setup Complete!"
echo "=" * 60
echo ""
echo "📁 Setup Location: $(pwd)"
echo "🐍 Virtual Environment: maixcam_env/"
echo "🔧 Tools Directory: maixcam_tools/"
echo ""
echo "🚀 To activate the environment:"
echo "   source activate_maixcam_env.sh"
echo "   # or"
echo "   source maixcam_env/bin/activate"
echo ""
echo "🧪 To test the setup:"
echo "   python test_conversion_setup.py"
echo ""
echo "📋 To install additional dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "🎯 Next steps:"
echo "   1. Activate environment: source activate_maixcam_env.sh"
echo "   2. Test setup: python test_conversion_setup.py"
echo "   3. Run conversion: ./convert_yolov11_pokemon_to_cvimodel.sh"
echo ""
echo "📖 Documentation: https://wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam.html"
echo ""
print_status "Setup complete! Ready for MaixCam model conversion."
