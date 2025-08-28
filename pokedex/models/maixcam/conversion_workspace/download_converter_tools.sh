#!/bin/bash

# MaixCam Converter Tools Download Script
# Attempts to download converter tools from various sources

set -e

echo "🔍 Attempting to download MaixCam converter tools..."
echo "📖 Following: https://wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam.html"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Create download directory
mkdir -p converter_download
cd converter_download

print_info "Searching for MaixCam converter tools..."

# List of possible URLs to try
urls=(
    "https://dl.sipeed.com/MAIX/MaixPy/release/master/maixcam_converter.zip"
    "https://dl.sipeed.com/MAIX/MaixPy/release/master/converter_tools.zip"
    "https://dl.sipeed.com/MAIX/MaixPy/release/master/maixcam_tools.zip"
    "https://dl.sipeed.com/MAIX/MaixPy/release/master/tools.zip"
    "https://dl.sipeed.com/MAIX/MaixPy/release/master/maixcam_sdk.zip"
    "https://dl.sipeed.com/MAIX/MaixPy/release/master/sdk.zip"
)

# Try to download from each URL
downloaded=false
for url in "${urls[@]}"; do
    print_info "Trying: $url"
    
    # Try to download with wget
    if wget -O maixcam_converter.zip "$url" 2>/dev/null; then
        if [ -f maixcam_converter.zip ] && [ -s maixcam_converter.zip ]; then
            print_status "Successfully downloaded from: $url"
            
            # Try to extract
            if unzip -t maixcam_converter.zip >/dev/null 2>&1; then
                print_info "Extracting converter tools..."
                unzip -o maixcam_converter.zip
                print_status "Converter tools extracted successfully"
                downloaded=true
                break
            else
                print_warning "Downloaded file is not a valid zip archive"
                rm -f maixcam_converter.zip
            fi
        else
            print_warning "Download failed or file is empty"
            rm -f maixcam_converter.zip
        fi
    else
        print_warning "Could not download from: $url"
    fi
done

if [ "$downloaded" = false ]; then
    print_error "Could not download converter tools automatically"
    echo ""
    print_info "Manual download required:"
    echo "1. Visit: https://wiki.sipeed.com/maixpy/"
    echo "2. Navigate to 'AI 模型转换和移植' section"
    echo "3. Download MaixCam SDK or converter tools"
    echo "4. Extract and copy to maixcam_tools/ directory"
    echo ""
    print_info "Alternative sources:"
    echo "- GitHub: https://github.com/sipeed/MaixPy"
    echo "- Downloads: https://dl.sipeed.com/MAIX/"
    echo "- Community: https://forum.sipeed.com/"
    echo ""
    print_info "After manual download:"
    echo "1. Extract the archive"
    echo "2. Copy converter tools to: ../maixcam_tools/"
    echo "3. Make executable: chmod +x ../maixcam_tools/convert_model"
    echo "4. Test: ../maixcam_tools/convert_model --help"
else
    print_status "Converter tools downloaded successfully!"
    echo ""
    print_info "Installing converter tools..."
    
    # Copy to maixcam_tools directory
    if [ -d "../maixcam_tools" ]; then
        cp -r * ../maixcam_tools/ 2>/dev/null || print_warning "Some files could not be copied"
        
        # Make executable if convert_model exists
        if [ -f "../maixcam_tools/convert_model" ]; then
            chmod +x ../maixcam_tools/convert_model
            print_status "convert_model made executable"
        fi
        
        # Test installation
        if [ -f "../maixcam_tools/convert_model" ]; then
            print_info "Testing converter installation..."
            if ../maixcam_tools/convert_model --help >/dev/null 2>&1; then
                print_status "Converter tools installed successfully!"
                echo ""
                print_info "Next steps:"
                echo "1. Activate environment: source ../activate_maixcam_env.sh"
                echo "2. Test converter: convert_model --help"
                echo "3. Run conversion: ../convert_yolov11_pokemon_to_cvimodel.sh"
            else
                print_warning "Converter tool found but may not be working correctly"
                print_info "Try running: ../maixcam_tools/convert_model --help"
            fi
        else
            print_warning "convert_model not found in extracted files"
            print_info "Check the extracted contents and copy manually"
        fi
    else
        print_error "maixcam_tools directory not found"
        print_info "Please run the setup script first"
    fi
fi

cd ..

echo ""
print_info "Download attempt completed"
print_info "Check DOWNLOAD_CONVERTER_GUIDE.md for manual instructions"
