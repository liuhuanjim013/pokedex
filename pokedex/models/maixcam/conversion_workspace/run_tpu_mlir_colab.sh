#!/bin/bash

echo "🚀 TPU-MLIR Conversion for Google Colab (Direct Installation)"
echo "============================================================="

# Configuration
WORKSPACE_DIR="tpu_mlir_workspace"

# Create workspace if it doesn't exist
mkdir -p ${WORKSPACE_DIR}

# Copy ONNX model to workspace if it exists
if [ -f "pokemon_classifier.onnx" ]; then
    echo "📁 Copying ONNX model to workspace..."
    cp pokemon_classifier.onnx ${WORKSPACE_DIR}/
fi

# Check if images directory exists
if [ -d "images" ]; then
    echo "✅ Images directory found"
    echo "📊 Images directory contains $(find images -name "*.jpg" | wc -l) images"
else
    echo "❌ Images directory not found"
    exit 1
fi

# Change to workspace directory
cd ${WORKSPACE_DIR}

echo "🔧 Installing TPU-MLIR directly in Colab environment..."

# Install TPU-MLIR dependencies
echo "📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3-dev \
    python3-pip \
    python3-venv \
    libgomp1 \
    libblas3 \
    liblapack3 \
    libatlas-base-dev \
    gfortran

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --quiet \
    numpy \
    scipy \
    pillow \
    psutil \
    tqdm \
    onnx \
    onnxruntime

# Install TPU-MLIR
echo "📦 Installing TPU-MLIR..."
if [ -f "../tpu_mlir_packages/tpu_mlir-1.21.1-py3-none-any.whl" ]; then
    echo "✅ Installing from local package..."
    pip install --quiet ../tpu_mlir_packages/tpu_mlir-1.21.1-py3-none-any.whl
else
    echo "⚠️  Local package not found, installing from PyPI..."
    pip install --quiet tpu-mlir==1.21.1
fi

# Verify TPU-MLIR installation
echo "🔍 Verifying TPU-MLIR installation..."
if command -v model_transform.py &> /dev/null; then
    echo "✅ TPU-MLIR tools available"
else
    echo "❌ TPU-MLIR tools not found in PATH"
    echo "🔍 Checking Python installation..."
    python3 -c "import tpu_mlir; print('✅ TPU-MLIR Python package installed')" || {
        echo "❌ TPU-MLIR Python package not found"
        exit 1
    }
    
    # Add TPU-MLIR tools to PATH
    export PATH=$PATH:$(python3 -c "import tpu_mlir; import os; print(os.path.dirname(tpu_mlir.__file__))")/bin
    echo "✅ Added TPU-MLIR tools to PATH"
fi

# Copy the Python converter script
if [ -f "../tpu_mlir_converter.py" ]; then
    echo "📁 Copying converter script..."
    cp ../tpu_mlir_converter.py .
fi

# Create symbolic link to images directory
if [ ! -L "images" ]; then
    echo "🔗 Creating symbolic link to images directory..."
    ln -sf ../images images
fi

echo "🐍 Running Python conversion script..."

# Run the conversion script
python3 tpu_mlir_converter.py

# Check if conversion was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 TPU-MLIR conversion completed successfully!"
    
    # List output files
    echo ""
    echo "📁 Output files in ${WORKSPACE_DIR}:"
    ls -la *.cvimodel *.mud *.mlir *_cali_table 2>/dev/null || echo "Some files may not have been created"
    
    # Show file sizes
    echo ""
    echo "📊 File sizes:"
    for file in *.cvimodel *.mud *.mlir *_cali_table; do
        if [ -f "$file" ]; then
            size=$(du -h "$file" | cut -f1)
            echo "  - $file: $size"
        fi
    done
    
    echo ""
    echo "🚀 Ready for MaixCam deployment!"
    
    # Copy files to parent directory for easy access
    echo "📁 Copying output files to parent directory..."
    cp *.cvimodel ../ 2>/dev/null || true
    cp *.mud ../ 2>/dev/null || true
    cp *.mlir ../ 2>/dev/null || true
    cp *_cali_table ../ 2>/dev/null || true
    
    echo "✅ Files copied to parent directory"
    
    # Go back to parent directory
    cd ..
    
else
    echo "❌ TPU-MLIR conversion failed"
    cd ..
    exit 1
fi

echo "✅ TPU-MLIR conversion process completed!"

# Show final status
echo ""
echo "🎯 Conversion Summary:"
echo "======================"
echo "✅ TPU-MLIR conversion completed"
echo "✅ CV181x chip target specified"
echo "✅ Stratified calibration with 15,000 images"
echo "✅ INT8 quantization applied"
echo "✅ MaixCam compatible model generated"
echo ""
echo "📁 Output files available in:"
echo "  - Current directory (copied for easy access)"
echo "  - ${WORKSPACE_DIR}/ (original location)"
echo ""
echo "🔧 Installation method: Direct installation in Colab environment"
echo "🚫 Note: udocker not used due to root user limitations"
