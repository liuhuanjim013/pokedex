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
