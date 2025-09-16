#!/bin/bash

set -euo pipefail

echo "🚀 TPU-MLIR Two-Stage Conversion (udocker)"
echo "========================================="

# -------- Config (edit if needed) --------
DOCKER_IMAGE="sophgo/tpuc_dev:latest"
CONTAINER_NAME="tpu_mlir_two_stage"
CHIP="cv180x"   # MaixCam CV181x family

# Defaults use your latest run dirs; override via env or args if needed
DET_ONNX_DEFAULT="runs/pokemon_det1_yolo11n_2565/weights/best.onnx"
CLS_ONNX_DEFAULT="runs/pokemon_cls1025_yolo11n_2242/weights/best.onnx"

DET_LIST_DEFAULT="data/calib_det_list.txt"
DET_DIR_DEFAULT="data/calib_det"
CLS_LIST_DEFAULT="data/calib_cls_list.txt"
CLS_DIR_DEFAULT="data/calib_cls"

OUT_DIR_DEFAULT="models/maixcam"

# Allow overrides from environment
DET_ONNX="${DET_ONNX:-$DET_ONNX_DEFAULT}"
CLS_ONNX="${CLS_ONNX:-$CLS_ONNX_DEFAULT}"
DET_LIST="${DET_LIST:-$DET_LIST_DEFAULT}"
DET_DIR="${DET_DIR:-$DET_DIR_DEFAULT}"
CLS_LIST="${CLS_LIST:-$CLS_LIST_DEFAULT}"
CLS_DIR="${CLS_DIR:-$CLS_DIR_DEFAULT}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

# -------- Validation --------
fail=false
for f in "$DET_ONNX" "$CLS_ONNX" "$DET_LIST" "$CLS_LIST"; do
  if [ ! -f "$f" ]; then
    echo "❌ Missing file: $f"; fail=true
  fi
done
for d in "$DET_DIR" "$CLS_DIR"; do
  if [ ! -d "$d" ]; then
    echo "❌ Missing directory: $d"; fail=true
  fi
done
${fail} && { echo "Aborting due to missing inputs."; exit 1; }

mkdir -p "$OUT_DIR"

echo "✅ Detector ONNX: $DET_ONNX"
echo "✅ Classifier ONNX: $CLS_ONNX"
echo "✅ Det calib list: $DET_LIST (dir: $DET_DIR)"
echo "✅ Cls calib list: $CLS_LIST (dir: $CLS_DIR)"
echo "📁 Output dir: $OUT_DIR"

# -------- choose runtime: docker > udocker --------
RUN_MODE="udocker"
if command -v docker >/dev/null 2>&1; then
  RUN_MODE="docker"
fi

if [ "$RUN_MODE" = "udocker" ]; then
  if ! command -v udocker >/dev/null 2>&1; then
    echo "📦 Installing udocker (pip user) ..."
    if command -v pip >/dev/null 2>&1; then
      pip install --user udocker || true
      export PATH="$HOME/.local/bin:$PATH"
    fi
  fi
  if ! command -v udocker >/dev/null 2>&1; then
    echo "📦 Installing udocker (curl script) ..."
    curl -fsSL https://raw.githubusercontent.com/indigo-dc/udocker/main/udocker.py -o udocker
    chmod +x udocker
    mkdir -p ~/.local/bin
    mv udocker ~/.local/bin/
    export PATH="$HOME/.local/bin:$PATH"
  fi
  command -v udocker >/dev/null 2>&1 || { echo "❌ udocker installation failed"; exit 1; }

  echo "🧹 Cleaning old container (if any)"
  udocker --allow-root rm "$CONTAINER_NAME" >/dev/null 2>&1 || true

  echo "📥 Pulling $DOCKER_IMAGE"
  udocker --allow-root pull "$DOCKER_IMAGE"

  echo "🔧 Creating container $CONTAINER_NAME"
  udocker --allow-root create --name="$CONTAINER_NAME" "$DOCKER_IMAGE"

  echo "🐍 Running conversion in container (udocker)..."
  RUNCMD=(udocker --allow-root run -e PYTHONUNBUFFERED=1 -v "$(pwd):/workspace" "$CONTAINER_NAME" bash -lc)
else
  echo "🐳 Using Docker runtime"
  echo "📥 Pulling $DOCKER_IMAGE"
  docker pull "$DOCKER_IMAGE"
  echo "🐍 Running conversion in container (docker)..."
  RUNCMD=(docker run --rm -e PYTHONUNBUFFERED=1 -v "$(pwd)":/workspace "$DOCKER_IMAGE" bash -lc)
fi

"${RUNCMD[@]}" "
set -euo pipefail
cd /workspace
echo '📦 Using system tpu-mlir from container'

# Ensure TPU-MLIR is available in the container
if ! python3 - <<'PY'
try:
    import tpu_mlir
    print('✅ TPU-MLIR already present')
except Exception as e:
    raise SystemExit(1)
PY
then
  echo '📦 Installing TPU-MLIR==1.21.1 in container...'
  python3 -m pip install -q --no-cache-dir tpu-mlir==1.21.1 || pip install -q --no-cache-dir tpu-mlir==1.21.1
  python3 - <<'PY'
import tpu_mlir, sys
print('✅ TPU-MLIR installed at', tpu_mlir.__file__)
print('Python', sys.version)
PY
fi

# Detector
python -m tpu_mlir.tools.model_transform \
  --model_name pokemon_det1 \
  --model_def '$DET_ONNX' \
  --input_shapes [[1,3,256,256]] \
  --keep_input_fp32 \
  --test_input '$DET_LIST' \
  --mlir pokemon_det1.mlir

python -m tpu_mlir.tools.run_calibration \
  --mlir pokemon_det1.mlir \
  --dataset '$DET_DIR' \
  --input_num 256 \
  --calibration_table pokemon_det1_cali_table

python -m tpu_mlir.tools.model_deploy \
  --mlir pokemon_det1.mlir \
  --quant_input \
  --calibration_table pokemon_det1_cali_table \
  --chip $CHIP \
  --model pokemon_det1_int8.cvimodel

# Classifier
python -m tpu_mlir.tools.model_transform \
  --model_name pokemon_cls1025 \
  --model_def '$CLS_ONNX' \
  --input_shapes [[1,3,224,224]] \
  --keep_input_fp32 \
  --test_input '$CLS_LIST' \
  --mlir pokemon_cls1025.mlir

python -m tpu_mlir.tools.run_calibration \
  --mlir pokemon_cls1025.mlir \
  --dataset '$CLS_DIR' \
  --input_num 256 \
  --calibration_table pokemon_cls1025_cali_table

python -m tpu_mlir.tools.model_deploy \
  --mlir pokemon_cls1025.mlir \
  --quant_input \
  --calibration_table pokemon_cls1025_cali_table \
  --chip $CHIP \
  --model pokemon_cls1025_int8.cvimodel

# Move artifacts to output dir
mkdir -p '$OUT_DIR'
mv -f pokemon_det1_int8.cvimodel '$OUT_DIR/'
mv -f pokemon_cls1025_int8.cvimodel '$OUT_DIR/'
mv -f pokemon_det1.mlir pokemon_det1_cali_table '$OUT_DIR/'
mv -f pokemon_cls1025.mlir pokemon_cls1025_cali_table '$OUT_DIR/'

# Create MUDs
cat > '$OUT_DIR/pokemon_det1.mud' << 'MUD'
type: yolo11
cvimodel: /root/models/pokemon_det1_int8.cvimodel
input:
  format: rgb
  width: 256
  height: 256
preprocess:
  mean: [0.0, 0.0, 0.0]
  scale: [0.003922, 0.003922, 0.003922]
postprocess:
  conf_threshold: 0.35
  iou_threshold: 0.45
  anchors: []
labels:
  num: 1
  names: ["pokemon"]
MUD

cat > '$OUT_DIR/pokemon_cls1025.mud' << 'MUD'
type: classifier
cvimodel: /root/models/pokemon_cls1025_int8.cvimodel
input:
  format: rgb
  width: 224
  height: 224
preprocess:
  mean: [0.0, 0.0, 0.0]
  scale: [0.003922, 0.003922, 0.003922]
labels:
  num: 1025
  file: /root/models/classes.txt
MUD

echo '🎉 Conversion finished in container.'
"

echo ""
echo "📁 Outputs in $OUT_DIR:"
ls -lh "$OUT_DIR" | sed 's/^/  /'

echo ""
echo "✅ Done. Copy the two .cvimodel and .mud files and classes.txt to /root/models on the device."

