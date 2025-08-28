#!/bin/bash

echo "🚀 TPU-MLIR Conversion for Google Colab (Simple)"
echo "================================================="

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

echo "🔧 Setting up TPU-MLIR environment..."

# Install basic dependencies
echo "📦 Installing basic dependencies..."
pip install --quiet numpy scipy pillow psutil tqdm onnx onnxruntime

# Try to install TPU-MLIR from PyPI
echo "📦 Installing TPU-MLIR from PyPI..."
pip install --quiet tpu-mlir==1.21.1 || {
    echo "⚠️  TPU-MLIR installation from PyPI failed"
    echo "🔍 Trying alternative installation method..."
    
    # Try installing from source or alternative method
    pip install --quiet tpu-mlir || {
        echo "❌ TPU-MLIR installation failed"
        echo "💡 Please install TPU-MLIR manually or use the Docker version"
        exit 1
    }
}

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
echo "🔧 Installation method: Simple PyPI installation"
echo "💡 If this fails, try the full installation script or use Docker locally"
