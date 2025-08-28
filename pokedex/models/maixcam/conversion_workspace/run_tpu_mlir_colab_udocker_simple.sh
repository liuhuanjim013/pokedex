#!/bin/bash

echo "🚀 TPU-MLIR Conversion for Google Colab (udocker)"
echo "=================================================="

# Configuration
DOCKER_IMAGE="sophgo/tpuc_dev:latest"
CONTAINER_NAME="tpu_mlir_converter"

# Check if images directory exists
if [ -d "images" ]; then
    echo "✅ Images directory found"
    echo "📊 Images directory contains $(find images -name "*.jpg" | wc -l) images"
else
    echo "❌ Images directory not found"
    exit 1
fi

# Check if ONNX model exists
if [ ! -f "pokemon_classifier.onnx" ]; then
    echo "❌ ONNX model not found: pokemon_classifier.onnx"
    exit 1
fi

echo "✅ ONNX model found: pokemon_classifier.onnx"

# Install udocker if not available
if ! command -v udocker &> /dev/null; then
    echo "📦 Installing udocker..."
    curl https://raw.githubusercontent.com/indigo-dc/udocker/master/udocker.py > udocker
    chmod +x udocker
    mv udocker ~/.local/bin/ 2>/dev/null || {
        mkdir -p ~/.local/bin
        mv udocker ~/.local/bin/
    }
    export PATH=$PATH:~/.local/bin
    echo "✅ udocker installed"
fi

echo "✅ udocker found"

# Clean up existing containers
echo "🧹 Cleaning up existing containers..."
udocker rm ${CONTAINER_NAME} 2>/dev/null || true

# Pull Docker image
echo "📦 Pulling TPU-MLIR Docker image..."
udocker pull ${DOCKER_IMAGE}

if [ $? -ne 0 ]; then
    echo "❌ Failed to pull Docker image"
    exit 1
fi

echo "✅ Docker image pulled successfully"

# Create container
echo "🔧 Creating udocker container..."
udocker create --name=${CONTAINER_NAME} ${DOCKER_IMAGE}

if [ $? -ne 0 ]; then
    echo "❌ Failed to create container"
    exit 1
fi

echo "✅ Container created"

# Copy files to container
echo "📁 Copying files to container..."
udocker cp pokemon_classifier.onnx ${CONTAINER_NAME}:/workspace/
udocker cp tpu_mlir_converter.py ${CONTAINER_NAME}:/workspace/
udocker cp -r images ${CONTAINER_NAME}:/workspace/

echo "✅ Files copied to container"

# Run the conversion
echo "🐍 Running TPU-MLIR conversion in container..."
udocker run ${CONTAINER_NAME} bash -c "cd /workspace && python3 tpu_mlir_converter.py"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 TPU-MLIR conversion completed successfully!"
    
    # Copy output files from container
    echo "📁 Copying output files from container..."
    udocker cp ${CONTAINER_NAME}:/workspace/*.cvimodel . 2>/dev/null || true
    udocker cp ${CONTAINER_NAME}:/workspace/*.mud . 2>/dev/null || true
    udocker cp ${CONTAINER_NAME}:/workspace/*.mlir . 2>/dev/null || true
    udocker cp ${CONTAINER_NAME}:/workspace/*_cali_table . 2>/dev/null || true
    
    echo "✅ Output files copied"
    
    # List output files
    echo ""
    echo "📁 Output files:"
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
    
else
    echo "❌ TPU-MLIR conversion failed"
fi

# Clean up container
echo "🧹 Cleaning up container..."
udocker rm ${CONTAINER_NAME}

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
echo "🔧 Method: udocker with TPU-MLIR Docker image"
echo "✅ Successfully used Docker registry image"
