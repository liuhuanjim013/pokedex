#!/bin/bash
# Script to activate the pokemon-classifier conda environment in Colab

echo "🔧 Activating pokemon-classifier conda environment..."

# Check if conda is installed in Google Drive for persistence
if [ -f "/content/drive/MyDrive/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "📁 Using conda from Google Drive for persistence..."
    source /content/drive/MyDrive/miniconda3/etc/profile.d/conda.sh
elif [ -f "/content/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "📁 Using conda from /content..."
    source /content/miniconda3/etc/profile.d/conda.sh
else
    echo "❌ Conda not found. Please run setup_colab_training.py first."
    exit 1
fi

# Activate the environment
conda activate pokemon-classifier

echo "✅ Environment activated!"
echo "You can now run training scripts directly:"
echo "  python scripts/yolo/train_yolov3_baseline.py"
echo "  python scripts/yolo/train_yolov3_improved.py"
echo ""
echo "Or use the wrapper script:"
echo "  python scripts/yolo/run_training_in_env.py train_yolov3_baseline.py"
