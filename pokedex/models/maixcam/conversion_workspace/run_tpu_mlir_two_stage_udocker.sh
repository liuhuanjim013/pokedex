#!/bin/bash

set -euo pipefail

echo "🚀 TPU-MLIR Two-Stage Conversion (udocker)"
echo "========================================="

# -------- Config (edit if needed) --------
DOCKER_IMAGE="sophgo/tpuc_dev:latest"
CONTAINER_NAME="tpu_mlir_two_stage"
CHIP="k230"   # Default to K230 MaixCam family; set CHIP=cv180x to target CV18xx

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
  RUNTIME_CMD=(udocker --allow-root run -e PYTHONUNBUFFERED=1 \
    -e CHIP="$CHIP" -e DET_ONNX="$DET_ONNX" -e CLS_ONNX="$CLS_ONNX" \
    -e DET_LIST="$DET_LIST" -e DET_DIR="$DET_DIR" \
    -e CLS_LIST="$CLS_LIST" -e CLS_DIR="$CLS_DIR" \
    -e OUT_DIR="$OUT_DIR" \
    -v "$(pwd):/workspace" "$CONTAINER_NAME" bash -lc)
else
  echo "🐳 Using Docker runtime"
  echo "📥 Pulling $DOCKER_IMAGE"
  docker pull "$DOCKER_IMAGE"
  echo "🐍 Running conversion in container (docker)..."
  RUNTIME_CMD=(docker run --rm -e PYTHONUNBUFFERED=1 \
    -e CHIP="$CHIP" -e DET_ONNX="$DET_ONNX" -e CLS_ONNX="$CLS_ONNX" \
    -e DET_LIST="$DET_LIST" -e DET_DIR="$DET_DIR" \
    -e CLS_LIST="$CLS_LIST" -e CLS_DIR="$CLS_DIR" \
    -e OUT_DIR="$OUT_DIR" \
    -v "$(pwd)":/workspace "$DOCKER_IMAGE" bash -lc)
fi
# Write container-side script to avoid host variable expansion issues
TMP_SCRIPT="$(pwd)/.tmp_two_stage_convert.sh"
cat > "$TMP_SCRIPT" << 'INSIDE'
set -euo pipefail
cd /workspace
echo "📦 Inside container: $(python3 -V)"

WHEEL="/workspace/pokedex/models/maixcam/conversion_workspace/tpu_mlir_packages/tpu_mlir-1.21.1-py3-none-any.whl"
if python3 -c "import tpu_mlir" 2>/dev/null; then
  echo '✅ TPU-MLIR already present'
else
  echo '📦 Installing TPU-MLIR==1.21.1 in container...'
  if [ -f "$WHEEL" ]; then
    python3 -m pip install -q --no-cache-dir "$WHEEL"
  else
    python3 -m pip install -q --no-cache-dir tpu-mlir==1.21.1
  fi
fi

if command -v model_transform.py >/dev/null 2>&1; then
  MT=model_transform.py
  RC=run_calibration.py
  MD=model_deploy.py
else
  # Fallback to module form
  MT="python3 -m tpu_mlir.tools.model_transform"
  RC="python3 -m tpu_mlir.tools.run_calibration"
  MD="python3 -m tpu_mlir.tools.model_deploy"
fi

echo "🔧 Transform DET: $DET_ONNX"
$MT --model_name pokemon_det1 --model_def "$DET_ONNX" \
    --input_shapes [[1,3,256,256]] \
    --mean 0,0,0 --scale 0.003922,0.003922,0.003922 \
    --pixel_format rgb \
    --mlir pokemon_det1.mlir

echo "🧮 Calibrate DET"
$RC pokemon_det1.mlir --dataset "$DET_DIR" --input_num 256 \
    -o pokemon_det1_cali_table

echo "🏗️  Deploy DET"
$MD --mlir pokemon_det1.mlir --quantize INT8 \
    --calibration_table pokemon_det1_cali_table --chip "$CHIP" \
    --model pokemon_det1_int8.cvimodel

echo "🔧 Transform CLS: $CLS_ONNX"
$MT --model_name pokemon_cls1025 --model_def "$CLS_ONNX" \
    --input_shapes [[1,3,224,224]] \
    --mean 0,0,0 --scale 0.003922,0.003922,0.003922 \
    --pixel_format rgb \
    --mlir pokemon_cls1025.mlir

echo "🧮 Calibrate CLS"
$RC pokemon_cls1025.mlir --dataset "$CLS_DIR" --input_num 256 \
    -o pokemon_cls1025_cali_table

echo "🏗️  Deploy CLS"
$MD --mlir pokemon_cls1025.mlir --quantize INT8 \
    --calibration_table pokemon_cls1025_cali_table --chip "$CHIP" \
    --model pokemon_cls1025_int8.cvimodel

mkdir -p "$OUT_DIR"
mv -f pokemon_det1_int8.cvimodel "$OUT_DIR/"
mv -f pokemon_cls1025_int8.cvimodel "$OUT_DIR/"
mv -f pokemon_det1.mlir pokemon_det1_cali_table "$OUT_DIR/"
mv -f pokemon_cls1025.mlir pokemon_cls1025_cali_table "$OUT_DIR/"

cat > "$OUT_DIR/pokemon_det1.mud" << 'MUD'
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
  file: /root/models/pokemon_det1_labels.txt
MUD

cat > "$OUT_DIR/pokemon_cls1025.mud" << 'MUD'
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
INSIDE

"${RUNTIME_CMD[@]}" "bash /workspace/.tmp_two_stage_convert.sh"
RC=$?
rm -f ".tmp_two_stage_convert.sh" || true
exit $RC

echo ""
echo "📁 Outputs in $OUT_DIR:"
ls -lh "$OUT_DIR" | sed 's/^/  /'

echo ""
echo "✅ Done. Copy the two .cvimodel and .mud files and classes.txt to /root/models on the device."

