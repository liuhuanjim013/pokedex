#!/bin/bash

echo "🚀 TPU-MLIR Conversion with Python Script"
echo "=========================================="

# Configuration
WORKSPACE_DIR="tpu_mlir_workspace"
CONTAINER_NAME="tpu_mlir_pokemon_conversion_python"

# Clean up existing container
echo "🧹 Cleaning up existing containers..."
sudo docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

# Create workspace if it doesn't exist
mkdir -p ${WORKSPACE_DIR}

# Copy ONNX model to workspace if it exists
if [ -f "pokemon_classifier.onnx" ]; then
    echo "📁 Copying ONNX model to workspace..."
    cp pokemon_classifier.onnx ${WORKSPACE_DIR}/
fi

# Check if images directory exists
if [ -d "images" ]; then
    echo "✅ Images directory found - will mount directly"
    echo "📊 Images directory contains $(find images -name "*.jpg" | wc -l) images"
else
    echo "❌ Images directory not found"
    exit 1
fi

echo "🔧 Starting TPU-MLIR Docker container with Python script..."

# Run Docker container with Python script
sudo docker run --privileged --name ${CONTAINER_NAME} \
    -v $(pwd)/${WORKSPACE_DIR}:/workspace \
    -v $(pwd)/images:/workspace/images \
    -v $(pwd)/tpu_mlir_packages:/tpu_mlir_packages \
    -v $(pwd)/tpu_mlir_converter.py:/workspace/tpu_mlir_converter.py \
    -it sophgo/tpuc_dev:latest \
    bash -c "
        echo '📦 Installing TPU-MLIR from local package...'
        pip install /tpu_mlir_packages/tpu_mlir-1.21.1-py3-none-any.whl
        echo '✅ TPU-MLIR installation completed'
        
        echo '🐍 Running Python conversion script...'
        cd /workspace
        python3 tpu_mlir_converter.py
        
        if [ \$? -eq 0 ]; then
            echo '✅ Python conversion script completed successfully'
        else
            echo '❌ Python conversion script failed'
            exit 1
        fi
    "

# Check if conversion was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 TPU-MLIR conversion completed successfully!"
    
    # List output files
    echo ""
    echo "📁 Output files in ${WORKSPACE_DIR}:"
    ls -la ${WORKSPACE_DIR}/*.cvimodel ${WORKSPACE_DIR}/*.mud ${WORKSPACE_DIR}/*.mlir ${WORKSPACE_DIR}/*_cali_table 2>/dev/null || echo "Some files may not have been created"
    
    echo ""
    echo "🚀 Ready for MaixCam deployment!"
else
    echo "❌ TPU-MLIR conversion failed"
    exit 1
fi

# Clean up
echo "🧹 Cleaning up Docker container..."
sudo docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

echo "✅ TPU-MLIR conversion process completed!"
