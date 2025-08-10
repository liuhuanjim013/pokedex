#!/usr/bin/env python3
"""
K210 Export Script for Pokemon Classifier
Exports trained YOLO models to K210-compatible .kmodel format
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

def export_onnx(weights_path: Path, imgsz: int, outdir: Path) -> Path:
    """Export YOLO model to ONNX format"""
    LOGGER.info(f"Exporting ONNX: imgsz={imgsz}")
    
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        
        onnx_path = outdir / f"{weights_path.stem}.onnx"
        model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, verbose=False)
        
        # Find exported ONNX file
        exported_onnx = weights_path.parent / f"{weights_path.stem}.onnx"
        if exported_onnx.exists():
            shutil.move(str(exported_onnx), str(onnx_path))
            LOGGER.info(f"ONNX exported: {onnx_path}")
            return onnx_path
        else:
            raise FileNotFoundError("ONNX export did not produce any .onnx file")
            
    except Exception as e:
        LOGGER.error(f"ONNX export failed: {e}")
        raise

def simplify_onnx(onnx_path: Path, outdir: Path) -> Path:
    """Simplify ONNX model for better nncase compatibility"""
    try:
        import onnx
        import onnxsim
        
        simplified_path = outdir / f"{onnx_path.stem}_simplified.onnx"
        LOGGER.info(f"Simplifying ONNX: {onnx_path} -> {simplified_path}")
        
        onnx_model = onnx.load(str(onnx_path))
        model_simplified, check = onnxsim.simplify(onnx_model)
        
        if not check:
            raise RuntimeError("Simplified ONNX model could not be validated")
        
        onnx.save(model_simplified, str(simplified_path))
        LOGGER.info("ONNX simplified successfully")
        return simplified_path
        
    except ImportError:
        LOGGER.warning("onnxsim not available, skipping simplification")
        return onnx_path
    except Exception as e:
        LOGGER.warning(f"ONNX simplification failed: {e}")
        return onnx_path

def prepare_calibration_dataset(calib_dir: Path, target_size: int, outdir: Path) -> Path:
    """Prepare calibration dataset by resizing images"""
    prepared_dir = outdir / "calib_prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from PIL import Image
        import random
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        calib_images = [f for f in calib_dir.iterdir() 
                       if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not calib_images:
            LOGGER.warning(f"No calibration images found in {calib_dir}")
            return calib_dir
        
        # Limit to 400 images
        calib_images = random.sample(calib_images, min(400, len(calib_images)))
        
        LOGGER.info(f"Preparing {len(calib_images)} calibration images...")
        
        for img_path in calib_images:
            try:
                with Image.open(img_path) as img:
                    img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                    img_resized.save(prepared_dir / img_path.name)
            except Exception as e:
                LOGGER.warning(f"Failed to process {img_path}: {e}")
        
        LOGGER.info(f"Prepared {len(calib_images)} calibration images at {prepared_dir}")
        return prepared_dir
        
    except ImportError:
        LOGGER.warning("PIL not available, using original calibration directory")
        return calib_dir
    except Exception as e:
        LOGGER.warning(f"Calibration preparation failed: {e}")
        return calib_dir

def create_ncc_wrapper():
    """Create ncc wrapper script using nncase Python API"""
    wrapper_script = '''#!/usr/bin/env python3
import sys
import argparse
import nncase
import tempfile
import shutil
import os

def main():
    parser = argparse.ArgumentParser(description='nncase compiler wrapper')
    parser.add_argument('compile', help='compile command')
    parser.add_argument('input', help='input ONNX file')
    parser.add_argument('output', help='output kmodel file')
    parser.add_argument('-t', '--target', default='k210', help='target platform')
    parser.add_argument('--dataset', help='calibration dataset directory')
    parser.add_argument('--input-mean', help='input mean values')
    parser.add_argument('--input-std', help='input std values')
    parser.add_argument('--input-layout', default='NCHW', help='input layout')
    parser.add_argument('--shape', help='input shape')
    parser.add_argument('--quanttype', default='int8', help='quantization type')
    parser.add_argument('--inference-type', default='int8', help='inference type')
    
    args = parser.parse_args()
    
    if args.compile != 'compile':
        print("Only 'compile' command supported", file=sys.stderr)
        sys.exit(1)
    
    # Parse parameters
    input_shape = [1, 3, 320, 320]
    if args.shape:
        input_shape = [int(x) for x in args.shape.split(',')]
    
    mean_val = 0.0
    std_val = 255.0
    if args.input_mean:
        mean_val = float(args.input_mean)
    if args.input_std:
        std_val = float(args.input_std)
    
    quant_type = args.quanttype if args.quanttype != 'int8' else args.inference_type
    
    # Create compile options
    compile_options = nncase.CompileOptions()
    compile_options.target = args.target
    compile_options.quant_type = quant_type
    compile_options.mean = [mean_val, mean_val, mean_val]
    compile_options.std = [std_val, std_val, std_val]
    compile_options.input_shape = input_shape
    compile_options.input_layout = args.input_layout
    compile_options.output_layout = args.input_layout
    
    # Create compiler
    compiler = nncase.Compiler(compile_options)
    
    # Import ONNX
    import_options = nncase.ImportOptions()
    with open(args.input, 'rb') as f:
        onnx_bytes = f.read()
    compiler.import_onnx(onnx_bytes, import_options)
    
    # Compile
    compiler.compile()
    
    # Save kmodel - try multiple approaches
    success = False
    
    # Approach 1: Direct file write
    try:
        with open(args.output, 'wb') as f:
            compiler.gencode(f)
        print(f"Successfully compiled {args.input} to {args.output}")
        success = True
    except Exception as e:
        print(f"Direct gencode failed: {e}", file=sys.stderr)
    
    # Approach 2: Temporary file
    if not success:
        try:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp_f:
                compiler.gencode(tmp_f)
                tmp_f.flush()
                shutil.copy2(tmp_f.name, args.output)
                os.unlink(tmp_f.name)
            print(f"Successfully compiled {args.input} to {args.output} (using temp file)")
            success = True
        except Exception as e:
            print(f"Temporary file approach failed: {e}", file=sys.stderr)
    
    # Approach 3: BytesIO buffer
    if not success:
        try:
            import io
            buffer = io.BytesIO()
            compiler.gencode(buffer)
            buffer.seek(0)
            with open(args.output, 'wb') as f:
                f.write(buffer.getvalue())
            print(f"Successfully compiled {args.input} to {args.output} (using BytesIO)")
            success = True
        except Exception as e:
            print(f"BytesIO approach failed: {e}", file=sys.stderr)
    
    if not success:
        print(f"All gencode approaches failed for {args.input}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
'''
    
    wrapper_path = Path.home() / '.local' / 'bin' / 'ncc'
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_script)
    
    wrapper_path.chmod(0o755)
    return wrapper_path

def compile_with_nncase_api(onnx_path: Path, kmodel_path: Path, calib_dir: Path, 
                        imgsz: int, mean: Tuple[float, float, float], 
                        std: Tuple[float, float, float], layout: str, quant_type: str):
    """Compile ONNX to K210 kmodel using nncase Python API"""
    LOGGER.info("Compiling with nncase Python API...")
    
    try:
        import nncase
        
        # Create compile options
        compile_options = nncase.CompileOptions()
        compile_options.target = "k210"
        compile_options.quant_type = quant_type
        compile_options.input_shape = [1, 3, imgsz, imgsz]  # NCHW format
        compile_options.input_layout = layout
        compile_options.output_layout = layout
        compile_options.mean = list(mean)  # Convert tuple to list
        compile_options.std = list(std)
        
        LOGGER.info("Creating nncase compiler...")
        compiler = nncase.Compiler(compile_options)
        
        # Import ONNX
        LOGGER.info("Importing ONNX model...")
        import_options = nncase.ImportOptions()
        with open(onnx_path, 'rb') as f:
            onnx_bytes = f.read()
        compiler.import_onnx(onnx_bytes, import_options)
        
        # Compile
        LOGGER.info("Compiling model...")
        compiler.compile()
        
        # Save kmodel - try multiple approaches
        success = False
        
        # Approach 1: BytesIO buffer (most reliable)
        try:
            LOGGER.info("Attempting BytesIO approach...")
            import io
            buffer = io.BytesIO()
            compiler.gencode_tobytes(buffer)  # Use gencode_tobytes instead of gencode
            buffer.seek(0)
            with open(kmodel_path, 'wb') as f:
                f.write(buffer.getvalue())
            LOGGER.info(f"Successfully compiled {onnx_path} to {kmodel_path} (using BytesIO)")
            success = True
        except AttributeError:
            LOGGER.warning("gencode_tobytes not available, trying alternative approaches")
        except Exception as e:
            LOGGER.warning(f"BytesIO approach failed: {e}")
        
        # Approach 2: Get bytes directly
        if not success:
            try:
                LOGGER.info("Attempting direct bytes approach...")
                kmodel_bytes = compiler.gencode_tobytes()  # Get bytes directly
                with open(kmodel_path, 'wb') as f:
                    f.write(kmodel_bytes)
                LOGGER.info(f"Successfully compiled {onnx_path} to {kmodel_path} (using direct bytes)")
                success = True
            except AttributeError:
                LOGGER.warning("gencode_tobytes not available, trying alternative approaches")
            except Exception as e:
                LOGGER.warning(f"Direct bytes approach failed: {e}")
        
        # Approach 3: Use string path (for older nncase versions)
        if not success:
            try:
                LOGGER.info("Attempting string path approach...")
                compiler.gencode(str(kmodel_path))  # Pass path as string
                LOGGER.info(f"Successfully compiled {onnx_path} to {kmodel_path} (using string path)")
                success = True
            except Exception as e:
                LOGGER.warning(f"String path approach failed: {e}")
        
        if not success:
            raise RuntimeError("All gencode approaches failed")
            
        return True
            
    except Exception as e:
        LOGGER.error(f"nncase compilation failed: {e}")
        raise

def main():
    """Main export function"""
    parser = argparse.ArgumentParser(description="Export YOLO model to K210 kmodel format")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights (.pt)")
    parser.add_argument("--outdir", default="export_k210", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size")
    parser.add_argument("--calib-dir", required=True, help="Calibration dataset directory")
    parser.add_argument("--prepare-calib", action="store_true", help="Prepare calibration dataset")
    parser.add_argument("--mean", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Input mean values")
    parser.add_argument("--std", nargs=3, type=float, default=[255.0, 255.0, 255.0], help="Input std values")
    parser.add_argument("--layout", default="NCHW", choices=["NCHW", "NHWC"], help="Input layout")
    parser.add_argument("--quant-type", default="int8", choices=["int8", "uint8"], help="Quantization type")
    parser.add_argument("--simplify-onnx", action="store_true", help="Simplify ONNX model")
    parser.add_argument("--classes", help="Path to classes.txt file")
    parser.add_argument("--anchors", help="Path to anchors.txt file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs
    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.exists():
        LOGGER.error(f"Weights file not found: {weights_path}")
        return 1
    
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Export ONNX
    onnx_path = export_onnx(weights_path, args.imgsz, outdir)
    
    # Simplify ONNX if requested
    if args.simplify_onnx:
        onnx_path = simplify_onnx(onnx_path, outdir)
    
    # Prepare calibration dataset
    calib_dir = Path(args.calib_dir).expanduser().resolve()
    if not calib_dir.exists():
        LOGGER.error(f"Calibration directory not found: {calib_dir}")
        return 1
    
    prepared_calib_dir = prepare_calibration_dataset(calib_dir, args.imgsz, outdir)
    
    # Compile to kmodel
    kmodel_path = outdir / f"{weights_path.stem}_k210.kmodel"
    
    try:
        compile_with_nncase_api(
            onnx_path=onnx_path,
            kmodel_path=kmodel_path,
            calib_dir=prepared_calib_dir,
            imgsz=args.imgsz,
            mean=tuple(args.mean),
            std=tuple(args.std),
            layout=args.layout,
            quant_type=args.quant_type,
        )
        
        # Verify the kmodel file was created and has reasonable size
        if kmodel_path.exists():
            file_size = kmodel_path.stat().st_size
            if file_size < 1024:  # Less than 1KB is suspicious
                LOGGER.warning(f"Generated kmodel file is very small ({file_size} bytes), may be incomplete")
            else:
                LOGGER.info(f"Generated kmodel file: {file_size} bytes")
        else:
            raise RuntimeError("kmodel file was not created")
            
    except Exception as e:
        LOGGER.error(f"nncase compilation failed: {e}")
        LOGGER.error("Please ensure nncase v1.6.0 is installed correctly.")
        return 1
        
    # Package artifacts
    if args.classes:
        classes_src = Path(args.classes).expanduser().resolve()
        if classes_src.exists():
            dst = outdir / "classes.txt"
            shutil.copy2(classes_src, dst)
            LOGGER.info(f"Packaged classes: {dst}")
    
    if args.anchors:
        anchors_src = Path(args.anchors).expanduser().resolve()
        if anchors_src.exists():
            dst = outdir / "anchors.txt"
            shutil.copy2(anchors_src, dst)
            LOGGER.info(f"Packaged anchors: {dst}")
    
    LOGGER.info(f"Success! kmodel generated: {kmodel_path}")
    LOGGER.info("Deploy the kmodel and classes.txt to Maix Bit SD card.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
