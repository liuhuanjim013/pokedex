#!/bin/bash

echo "🔧 Downloading TPU-MLIR locally..."

# Create a directory for TPU-MLIR
mkdir -p tpu_mlir_packages

# Download TPU-MLIR wheel
echo "📦 Downloading TPU-MLIR wheel..."
wget -O tpu_mlir_packages/tpu_mlir-1.21.1-py3-none-any.whl \
    "https://files.pythonhosted.org/packages/py3/t/tpu-mlir/tpu_mlir-1.21.1-py3-none-any.whl"

if [ $? -eq 0 ]; then
    echo "✅ TPU-MLIR downloaded successfully"
    echo "📁 File size: $(du -h tpu_mlir_packages/tpu_mlir-1.21.1-py3-none-any.whl | cut -f1)"
else
    echo "❌ Failed to download TPU-MLIR"
    exit 1
fi

echo "🚀 Ready for Docker installation!"
