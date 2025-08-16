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

def convert_onnx_to_tflite_with_version_management(onnx_path: Path, outdir: Path, imgsz: int) -> Path:
    """Convert ONNX to TFLite using conda environment isolation for version compatibility"""
    import subprocess
    import sys
    
    LOGGER.info(f"Converting ONNX to TensorFlow Lite using conda environment isolation...")
    
    tflite_path = outdir / f"{onnx_path.stem}.tflite"
    tf_env_name = "tf_conversion_temp"
    
    try:
        # Check if conda is available
        conda_path = None
        try:
            result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                conda_path = "conda"
        except FileNotFoundError:
            try:
                result = subprocess.run(["/home/liuhuan/miniconda3/bin/conda", "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    conda_path = "/home/liuhuan/miniconda3/bin/conda"
            except FileNotFoundError:
                pass
        
        if not conda_path:
            LOGGER.warning("Conda not found, falling back to direct conversion")
            return _do_onnx_to_tflite_conversion(onnx_path, tflite_path, imgsz)
        
        LOGGER.info(f"Creating temporary conda environment: {tf_env_name}")
        
        # Create temporary environment with stable K210 setup
        subprocess.check_call([
            conda_path, "create", "-n", tf_env_name, "python=3.7", "-y", "--quiet"
        ])
        
        # Install stable K210 packages (recommended working combination)
        subprocess.check_call([
            conda_path, "run", "-n", tf_env_name, "pip", "install",
            "numpy==1.18.5", "tensorflow==2.3.0", "tensorflow-probability==0.11.0",
            "onnx==1.6.0", "typing_extensions==3.7.4.3",
            "--quiet"
        ])
        
        # Try onnx-tf 1.5.0 - if it fails, we'll skip TFLite conversion
        try:
            subprocess.check_call([
                conda_path, "run", "-n", tf_env_name, "pip", "install",
                "onnx-tf==1.5.0", "--quiet"
            ])
            tflite_available = True
        except subprocess.CalledProcessError:
            LOGGER.warning("onnx-tf 1.5.0 installation failed - skipping TFLite conversion")
            tflite_available = False
        
        # Create conversion script
        conversion_script = f'''
import sys
from pathlib import Path
try:
    import onnx
    import tensorflow as tf
    from onnx_tf.backend import prepare
    import shutil
    
    onnx_path = Path("{str(onnx_path)}")
    tflite_path = Path("{str(tflite_path)}")
    
    print(f"Loading ONNX model: {{onnx_path}}")
    onnx_model = onnx.load(str(onnx_path))
    
    print("Converting ONNX to TensorFlow...")
    tf_rep = prepare(onnx_model)
    tf_model_path = tflite_path.parent / f"{{onnx_path.stem}}_tf"
    tf_rep.export_graph(str(tf_model_path))
    
    # Create representative dataset for INT8 quantization
    def representative_dataset():
        import cv2
        import glob
        import numpy as np
        
        # Get sample images from calibration directory  
        calib_images = glob.glob(str(tflite_path.parent / "calib_prepared" / "*.jpg"))[:20]
        if not calib_images:
            # Fallback: create dummy data
            for _ in range(8):
                dummy_img = np.random.rand(1, 3, {imgsz}, {imgsz}).astype(np.float32)
                yield [dummy_img]
            return
            
        for img_path in calib_images:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, ({imgsz}, {imgsz}))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # HWC to CHW
                img = np.expand_dims(img, axis=0)   # Add batch dimension
                yield [img]
            except Exception as e:
                continue
    
    print("Converting TensorFlow to TensorFlow Lite with representative dataset...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TensorFlow Lite model saved: {{tflite_path}}")
    
    # Clean up
    shutil.rmtree(tf_model_path, ignore_errors=True)
    print("SUCCESS")
    
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        
        # Run conversion in isolated environment
        result = subprocess.run([
            conda_path, "run", "-n", tf_env_name, "python", "-c", conversion_script
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and "SUCCESS" in result.stdout:
            LOGGER.info("TensorFlow Lite conversion completed successfully")
            LOGGER.info(result.stdout)
            return tflite_path
        else:
            LOGGER.error(f"TFLite conversion failed: {result.stderr}")
            LOGGER.error(f"Stdout: {result.stdout}")
            raise RuntimeError(f"TFLite conversion subprocess failed")
            
    except subprocess.TimeoutExpired:
        LOGGER.error("TFLite conversion timed out")
        raise RuntimeError("TFLite conversion timed out")
    except Exception as e:
        LOGGER.error(f"TFLite environment setup failed: {e}")
        # Fall back to direct conversion
        LOGGER.warning("Falling back to direct conversion...")
        return _do_onnx_to_tflite_conversion(onnx_path, tflite_path, imgsz)
    finally:
        # Clean up temporary environment
        try:
            subprocess.run([conda_path, "env", "remove", "-n", tf_env_name, "-y", "--quiet"], 
                         capture_output=True, timeout=60)
            LOGGER.info(f"Cleaned up temporary environment: {tf_env_name}")
        except:
            LOGGER.warning(f"Could not clean up temporary environment: {tf_env_name}")

def _do_onnx_to_tflite_conversion(onnx_path: Path, tflite_path: Path, imgsz: int) -> Path:
    """Perform the actual ONNX to TFLite conversion with numpy compatibility fix"""
    try:
        # Fix numpy compatibility issue for TensorFlow 2.6.0
        import numpy as np
        if not hasattr(np, 'object'):
            # Restore deprecated numpy.object for TensorFlow 2.6.0 compatibility
            np.object = object
            np.bool = bool
            np.int = int
            np.float = float
            np.complex = complex
            np.unicode = str
            LOGGER.info("Applied numpy compatibility patch for TensorFlow 2.6.0")
        
        import onnx
        import tensorflow as tf
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        onnx_model = onnx.load(str(onnx_path))
        
        # Convert ONNX to TensorFlow
        tf_rep = prepare(onnx_model)
        tf_model_path = tflite_path.parent / f"{onnx_path.stem}_tf"
        tf_rep.export_graph(str(tf_model_path))
        
        # Create representative dataset for INT8 quantization
        def representative_dataset():
            import cv2
            import glob
            
            # Get sample images from calibration directory
            calib_images = glob.glob(str(tflite_path.parent / "calib_prepared" / "*.jpg"))[:20]
            if not calib_images:
                # Fallback: create dummy data
                for _ in range(8):
                    dummy_img = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
                    yield [dummy_img]
                return
                
            for img_path in calib_images:
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.resize(img, (imgsz, imgsz))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
                    img = np.expand_dims(img, axis=0)   # Add batch dimension
                    yield [img]
                except Exception as e:
                    continue
        
        # Convert TensorFlow model to TensorFlow Lite with representative dataset
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # Convert
        tflite_model = converter.convert()
        
        # Save TensorFlow Lite model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        LOGGER.info(f"TensorFlow Lite model saved: {tflite_path}")
        
        # Clean up temporary TensorFlow model
        import shutil
        shutil.rmtree(tf_model_path, ignore_errors=True)
        
        return tflite_path
        
    except ImportError as e:
        LOGGER.error(f"Required packages not installed: {e}")
        LOGGER.error("Please install: pip install onnx-tf tensorflow")
        raise
    except Exception as e:
        LOGGER.error(f"ONNX to TensorFlow Lite conversion failed: {e}")
        raise

def convert_onnx_to_tflite(onnx_path: Path, outdir: Path, imgsz: int) -> Path:
    """Convert ONNX model to TensorFlow Lite format for nncase v0.1.0-rc5 compatibility"""
    LOGGER.info(f"Converting ONNX to TensorFlow Lite for nncase v0.1.0-rc5...")
    
    tflite_path = outdir / f"{onnx_path.stem}.tflite"
    
    # Try direct conversion with numpy compatibility patch first
    try:
        return _do_onnx_to_tflite_conversion(onnx_path, tflite_path, imgsz)
    except Exception as e:
        LOGGER.warning(f"Direct TFLite conversion failed: {e}")
        LOGGER.info("Trying isolated environment approach...")
        return convert_onnx_to_tflite_with_version_management(onnx_path, outdir, imgsz)

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
    """Compile ONNX to K210 kmodel using nncase v0.1.0-rc5 binary for MaixPy compatibility"""
    LOGGER.info("Compiling with nncase v0.1.0-rc5 binary for MaixPy kmodel v3...")
    
    # Try to use the binary version first (preferred for v0.1.0-rc5)
    try:
        return compile_with_nncase_binary(onnx_path, kmodel_path, calib_dir, imgsz, mean, std, layout, quant_type)
    except Exception as e:
        LOGGER.warning(f"Binary compilation failed: {e}")
        LOGGER.info("Falling back to Python API...")
        
        # Fallback to Python API if available
        try:
            import nncase
            
            # Check nncase version to determine API approach
            nncase_version = getattr(nncase, '__version__', 'unknown')
            LOGGER.info(f"Using nncase version: {nncase_version}")
            
            # For v0.1.0-rc5, use the older API structure
            if 'v0.1.0' in nncase_version or 'dev5' in nncase_version:
                LOGGER.info("Detected nncase v0.1.0-rc5, using compatible API...")
                return compile_with_old_nncase_api(onnx_path, kmodel_path, calib_dir, imgsz, mean, std, layout, quant_type)
            else:
                LOGGER.warning(f"nncase version {nncase_version} may not be compatible with MaixPy kmodel v3")
                LOGGER.warning("For best compatibility, please install nncase v0.1.0-rc5")
                # Fall back to newer API approach
                return compile_with_new_nncase_api(onnx_path, kmodel_path, calib_dir, imgsz, mean, std, layout, quant_type)
                
        except ImportError:
            LOGGER.error("nncase Python API not available and binary compilation failed")
            raise RuntimeError("No working nncase installation found")

def compile_with_nncase_binary(onnx_path: Path, kmodel_path: Path, calib_dir: Path, 
                           imgsz: int, mean: Tuple[float, float, float], 
                           std: Tuple[float, float, float], layout: str, quant_type: str):
    """Compile using nncase v0.1.0-rc5 binary for kmodel v3 compatibility"""
    LOGGER.info("Using nncase v0.1.0-rc5 binary for kmodel v3 generation...")
    
    # First convert ONNX to TensorFlow Lite since nncase v0.1.0-rc5 only supports TensorFlow Lite
    LOGGER.info("nncase v0.1.0-rc5 only supports TensorFlow Lite format. Converting ONNX to TensorFlow Lite...")
    try:
        tflite_path = convert_onnx_to_tflite(onnx_path, onnx_path.parent, imgsz)
    except Exception as e:
        LOGGER.error(f"ONNX to TensorFlow Lite conversion failed: {e}")
        raise RuntimeError(f"Cannot convert ONNX to TensorFlow Lite: {e}")
    
    # Find ncc binary
    ncc_path = None
    ncc_candidates = [
        "ncc",  # in PATH
        str(Path.home() / ".local/bin/ncc"),  # symlink location
        str(Path.home() / "ncc"),  # direct binary location (v0.1.0-rc5 extracts here)
        str(Path.home() / "ncc-linux-x86_64/ncc"),  # subdirectory location (fallback)
    ]
    
    for candidate in ncc_candidates:
        try:
            result = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                ncc_path = candidate
                LOGGER.info(f"Found nncase binary at: {ncc_path}")
                break
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            continue
    
    if not ncc_path:
        raise RuntimeError("nncase binary (ncc) not found. Please install nncase v0.1.0-rc5")
    
    # Build ncc command for v0.1.0-rc5 (correct format - no subcommands)
    cmd = [
        ncc_path,
        str(tflite_path),    # input .tflite file
        str(kmodel_path),    # output model file
        "--input-format", "tflite",  # nncase v0.1.0-rc5 only supports TensorFlow Lite
        "--output-format", "kmodel", # output to kmodel format
        "--inference-type", quant_type,
    ]
    
    # Add calibration dataset if provided
    if calib_dir and calib_dir.exists():
        cmd.extend(["--dataset", str(calib_dir)])
    
    LOGGER.info(f"Running nncase command: {' '.join(cmd)}")
    
    try:
        # Run the compilation
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            LOGGER.info("nncase compilation completed successfully!")
            LOGGER.info(f"Generated kmodel: {kmodel_path}")
            
            # Verify the output file exists
            if kmodel_path.exists():
                file_size = kmodel_path.stat().st_size
                LOGGER.info(f"Output kmodel size: {file_size} bytes")
                return True
            else:
                raise RuntimeError("kmodel file was not created")
        else:
            LOGGER.error(f"nncase compilation failed with return code {result.returncode}")
            LOGGER.error(f"stdout: {result.stdout}")
            LOGGER.error(f"stderr: {result.stderr}")
            raise RuntimeError(f"nncase compilation failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        LOGGER.error("nncase compilation timed out (5 minutes)")
        raise RuntimeError("nncase compilation timed out")
    except Exception as e:
        LOGGER.error(f"nncase binary execution failed: {e}")
        raise

def compile_with_old_nncase_api(onnx_path: Path, kmodel_path: Path, calib_dir: Path, 
                            imgsz: int, mean: Tuple[float, float, float], 
                            std: Tuple[float, float, float], layout: str, quant_type: str):
    """Compile using nncase v0.1.0-rc5 API for kmodel v3 compatibility"""
    import nncase
    
    LOGGER.info("Using nncase v0.1.0-rc5 API for kmodel v3 generation...")
    
    try:
        # For older nncase, try to use the available API
        # The exact API may vary, so we'll try multiple approaches
        
        # Approach 1: Try the most common v0.1.0-rc5 API pattern
        try:
            LOGGER.info("Attempting v0.1.0-rc5 compilation approach...")
            
            # Create compile options (API may differ)
            if hasattr(nncase, 'CompileOptions'):
                compile_options = nncase.CompileOptions()
                compile_options.target = "k210"
                compile_options.input_shape = [1, 3, imgsz, imgsz]
                compile_options.mean = list(mean)
                compile_options.std = list(std)
                
                if hasattr(compile_options, 'quant_type'):
                    compile_options.quant_type = quant_type
                if hasattr(compile_options, 'input_layout'):
                    compile_options.input_layout = layout
                if hasattr(compile_options, 'output_layout'):
                    compile_options.output_layout = layout
                
                # Create compiler
                compiler = nncase.Compiler(compile_options)
                
                # Import ONNX
                with open(onnx_path, 'rb') as f:
                    onnx_bytes = f.read()
                
                if hasattr(nncase, 'ImportOptions'):
                    import_options = nncase.ImportOptions()
                    compiler.import_onnx(onnx_bytes, import_options)
                else:
                    compiler.import_onnx(onnx_bytes)
                
                # Compile
                compiler.compile()
                
                # Generate kmodel
                with open(kmodel_path, 'wb') as f:
                    compiler.gencode(f)
                
                LOGGER.info(f"Successfully compiled {onnx_path} to {kmodel_path} (v0.1.0-rc5 API)")
                return True
                
        except Exception as e:
            LOGGER.warning(f"v0.1.0-rc5 API approach failed: {e}")
        
        # Approach 2: Direct function call if available
        try:
            if hasattr(nncase, 'compile_onnx'):
                LOGGER.info("Attempting direct nncase.compile_onnx approach...")
                nncase.compile_onnx(
                    str(onnx_path),
                    str(kmodel_path),
                    target="k210",
                    input_shape=[1, 3, imgsz, imgsz],
                    mean=list(mean),
                    std=list(std)
                )
                LOGGER.info(f"Successfully compiled {onnx_path} to {kmodel_path} (direct function)")
                return True
        except Exception as e:
            LOGGER.warning(f"Direct function approach failed: {e}")
        
        raise RuntimeError("All v0.1.0-rc5 compilation approaches failed")
        
    except Exception as e:
        LOGGER.error(f"Old nncase API compilation failed: {e}")
        raise

def compile_with_new_nncase_api(onnx_path: Path, kmodel_path: Path, calib_dir: Path, 
                            imgsz: int, mean: Tuple[float, float, float], 
                            std: Tuple[float, float, float], layout: str, quant_type: str):
    """Compile using newer nncase API (fallback for compatibility)"""
    import nncase
    
    LOGGER.warning("Using newer nncase API - may not generate kmodel v3 compatible with MaixPy")
    
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
        LOGGER.error("Please ensure nncase v0.1.0-rc5 is installed correctly for MaixPy kmodel v3 compatibility.")
        LOGGER.error("Install with: conda run -n pokemon-classifier python scripts/common/setup_environment.py --k210-only")
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
