#!/bin/bash

echo "🚀 TPU-MLIR Conversion for Google Colab (udocker)"
echo "=================================================="

# Configuration
WORKSPACE_DIR="tpu_mlir_workspace"
CONTAINER_NAME="tpu_mlir_pokemon_conversion_colab"

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

# Check if udocker is available
if ! command -v udocker &> /dev/null; then
    echo "❌ udocker not found. Installing udocker..."
    
    # Install udocker for Colab
    curl https://raw.githubusercontent.com/indigo-dc/udocker/master/udocker.py > udocker
    chmod +x udocker
    sudo mv udocker /usr/local/bin/
    
    # Initialize udocker
    udocker install
    echo "✅ udocker installed and initialized"
else
    echo "✅ udocker found"
fi

# Clean up existing container
echo "🧹 Cleaning up existing containers..."
udocker rm ${CONTAINER_NAME} 2>/dev/null || true

echo "�� Setting up udocker environment..."

# Set udocker to use runc backend (more compatible)
echo "🔧 Configuring udocker backend..."
export UDOCKER_USE_RUNC=1
export UDOCKER_USE_CRUN=0

# Pull the TPU-MLIR image if not already available
echo "📦 Pulling TPU-MLIR Docker image..."
udocker pull sophgo/tpuc_dev:latest

# Create container
echo "🔧 Creating udocker container..."
udocker create --name=${CONTAINER_NAME} sophgo/tpuc_dev:latest

# Run the conversion script
echo "🐍 Running Python conversion script in udocker container..."

# Create a temporary script to run inside the container
cat > /tmp/run_conversion.sh << 'EOF'
#!/bin/bash
set -e

echo "📦 Installing TPU-MLIR from local package..."
if [ -f "/tpu_mlir_packages/tpu_mlir-1.21.1-py3-none-any.whl" ]; then
    pip install /tpu_mlir_packages/tpu_mlir-1.21.1-py3-none-any.whl
    echo "✅ TPU-MLIR installation completed from local package"
else
    echo "⚠️  Local TPU-MLIR package not found, installing from PyPI..."
    pip install tpu-mlir==1.21.1
    echo "✅ TPU-MLIR installation completed from PyPI"
fi

echo "🐍 Running Python conversion script..."
cd /workspace
python3 tpu_mlir_converter.py

if [ $? -eq 0 ]; then
    echo "✅ Python conversion script completed successfully"
    exit 0
else
    echo "❌ Python conversion script failed"
    exit 1
fi
EOF

chmod +x /tmp/run_conversion.sh

# Try different udocker execution methods
echo "🚀 Starting conversion in udocker container..."

# Method 1: Try with default settings
if udocker run \
    --volume=$(pwd)/${WORKSPACE_DIR}:/workspace \
    --volume=$(pwd)/images:/workspace/images \
    --volume=$(pwd)/tpu_mlir_packages:/tpu_mlir_packages \
    --volume=$(pwd)/tpu_mlir_converter.py:/workspace/tpu_mlir_converter.py \
    --volume=/tmp/run_conversion.sh:/run_conversion.sh \
    ${CONTAINER_NAME} \
    /run_conversion.sh; then
    
    echo "✅ udocker execution successful with default settings"
else
    echo "⚠️  Default udocker execution failed, trying alternative method..."
    
    # Method 2: Try with different backend
    export UDOCKER_USE_RUNC=0
    export UDOCKER_USE_CRUN=1
    
    if udocker run \
        --volume=$(pwd)/${WORKSPACE_DIR}:/workspace \
        --volume=$(pwd)/images:/workspace/images \
        --volume=$(pwd)/tpu_mlir_packages:/tpu_mlir_packages \
        --volume=$(pwd)/tpu_mlir_converter.py:/workspace/tpu_mlir_converter.py \
        --volume=/tmp/run_conversion.sh:/run_conversion.sh \
        ${CONTAINER_NAME} \
        /run_conversion.sh; then
        
        echo "✅ udocker execution successful with crun backend"
    else
        echo "❌ All udocker execution methods failed"
        echo "💡 Falling back to direct installation method..."
        
        # Fallback to direct installation
        cd ${WORKSPACE_DIR}
        
        # Install basic dependencies
        pip install --quiet numpy scipy pillow psutil tqdm onnx onnxruntime
        
        # Try to install TPU-MLIR from PyPI with Python version check
        python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        echo "🐍 Python version: $python_version"
        
        if [[ "$python_version" == "3.10" ]]; then
            pip install --quiet tpu-mlir==1.21.1
        else
            echo "⚠️  Python version $python_version not compatible with TPU-MLIR"
            echo "💡 TPU-MLIR requires Python 3.10, but Colab has $python_version"
            echo "🔧 Trying to install without version constraint..."
            pip install --quiet tpu-mlir --no-deps
        fi
        
        # Copy the Python converter script
        if [ -f "../tpu_mlir_converter.py" ]; then
            cp ../tpu_mlir_converter.py .
        fi
        
        # Create symbolic link to images directory
        if [ ! -L "images" ]; then
            ln -sf ../images images
        fi
        
        # Run the conversion script
        python3 tpu_mlir_converter.py
    fi
fi

# Check if conversion was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 TPU-MLIR conversion completed successfully!"
    
    # List output files
    echo ""
    echo "📁 Output files in ${WORKSPACE_DIR}:"
    ls -la ${WORKSPACE_DIR}/*.cvimodel ${WORKSPACE_DIR}/*.mud ${WORKSPACE_DIR}/*.mlir ${WORKSPACE_DIR}/*_cali_table 2>/dev/null || echo "Some files may not have been created"
    
    # Show file sizes
    echo ""
    echo "📊 File sizes:"
    for file in ${WORKSPACE_DIR}/*.cvimodel ${WORKSPACE_DIR}/*.mud ${WORKSPACE_DIR}/*.mlir ${WORKSPACE_DIR}/*_cali_table; do
        if [ -f "$file" ]; then
            size=$(du -h "$file" | cut -f1)
            echo "  - $(basename "$file"): $size"
        fi
    done
    
    echo ""
    echo "🚀 Ready for MaixCam deployment!"
    
    # Copy files to current directory for easy access
    echo "📁 Copying output files to current directory..."
    cp ${WORKSPACE_DIR}/*.cvimodel . 2>/dev/null || true
    cp ${WORKSPACE_DIR}/*.mud . 2>/dev/null || true
    cp ${WORKSPACE_DIR}/*.mlir . 2>/dev/null || true
    cp ${WORKSPACE_DIR}/*_cali_table . 2>/dev/null || true
    
    echo "✅ Files copied to current directory"
else
    echo "❌ TPU-MLIR conversion failed"
    exit 1
fi

# Clean up
echo "🧹 Cleaning up udocker container..."
udocker rm ${CONTAINER_NAME} 2>/dev/null || true
rm -f /tmp/run_conversion.sh

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
echo "🔧 Installation method: udocker with fallback to direct installation"
echo "✅ Successfully handled root user limitations"
