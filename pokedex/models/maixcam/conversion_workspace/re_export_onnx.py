#!/usr/bin/env python3
"""
Re-export ONNX model from working PyTorch model
"""

import os
import sys
import logging
from pathlib import Path
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def re_export_onnx():
    """Re-export ONNX model from working PyTorch model."""
    
    # Paths
    pt_model_path = Path("../../../pokemon-classifier-maixcam/yolo11m-maixcam-classification/weights/best.pt")
    onnx_output_path = Path("tpu_mlir_workspace/pokemon_classifier_fixed.onnx")
    
    if not pt_model_path.exists():
        logger.error(f"PyTorch model not found: {pt_model_path}")
        return False
    
    try:
        logger.info(f"Loading PyTorch model: {pt_model_path}")
        model = YOLO(pt_model_path)
        
        logger.info("Model loaded successfully")
        logger.info(f"Model task: {model.task}")
        logger.info(f"Model classes: {len(model.names) if hasattr(model, 'names') else 'Unknown'}")
        
        # Create output directory
        onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export ONNX with proper parameters
        logger.info("Exporting ONNX model...")
        onnx_path = model.export(
            format='onnx',
            imgsz=256,  # Match the expected input size
            batch=1,    # Batch size 1 for inference
            dynamic=False,  # Static batch size
            simplify=True,  # Simplify model
            opset=12,   # ONNX opset version
            half=False, # FP32 precision
            int8=False, # No INT8 quantization
            verbose=True
        )
        
        # Move to our desired location
        if Path(onnx_path).exists():
            import shutil
            shutil.move(onnx_path, onnx_output_path)
            logger.info(f"ONNX model exported successfully: {onnx_output_path}")
            
            # Check file size
            size_mb = onnx_output_path.stat().st_size / (1024 * 1024)
            logger.info(f"ONNX model size: {size_mb:.2f} MB")
            
            return True
        else:
            logger.error(f"ONNX export failed - file not found: {onnx_path}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to export ONNX model: {e}")
        return False

if __name__ == "__main__":
    success = re_export_onnx()
    sys.exit(0 if success else 1)









