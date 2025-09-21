#!/usr/bin/env python3
"""
Detector Triplet Test (PyTorch, ONNX, CVI in container)

Purpose:
- Verify the detector works end-to-end across three backends:
  1) Original PyTorch `.pt` via Ultralytics API
  2) ONNX via onnxruntime
  3) CVI `.cvimodel` via TPU-MLIR model_runner inside docker/udocker

What it does:
- Runs the same images through all three backends
- Extracts a single best bounding box (cx, cy, w, h) in detector input space
- Compares pairwise IoU between boxes and validates minimum IoU threshold
- Saves intermediate inputs/outputs in an output directory per image

Defaults follow the two-stage pipeline conventions (1-class detector @256):
- PT: runs/pokemon_det1_yolo11n_2565/weights/best.pt
- ONNX: runs/pokemon_det1_yolo11n_2565/weights/best.onnx
- CVI: models/maixcam/pokemon_det1_int8.cvimodel
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detector Triplet Test (PT/ONNX/CVI)")
    p.add_argument("images", nargs="*", default=[
        "images/0001_001.jpg",
        "images/0004_407.jpg",
        "images/0007_794.jpg",
        "images/0025_2997.jpg",
        "images/0150_17608.jpg",
    ], help="Image paths to test")
    p.add_argument("--pt", dest="pt_path", type=str,
                   default="runs/pokemon_det1_yolo11n_2565/weights/best.pt",
                   help="Path to PyTorch detector .pt")
    p.add_argument("--onnx", dest="onnx_path", type=str,
                   default="runs/pokemon_det1_yolo11n_2565/weights/best.onnx",
                   help="Path to detector .onnx")
    p.add_argument("--cvi", dest="cvi_path", type=str,
                   default="models/maixcam/pokemon_det1_int8.cvimodel",
                   help="Path to detector .cvimodel")
    p.add_argument("--det-size", type=int, default=256, help="Detector input size (square)")
    p.add_argument("--iou-thr", type=float, default=0.30, help="IoU threshold for agreement")
    p.add_argument("--out-dir", type=str, default="detector_triplet_out", help="Output directory")
    p.add_argument("--use-docker", action="store_true", help="Force docker (if available)")
    p.add_argument("--use-udocker", action="store_true", help="Force udocker (attempt auto-install)")
    return p.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def preprocess_nchw_rgb01(img_bgr: np.ndarray, size: int) -> np.ndarray:
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]  # (1, 3, H, W)
    return x


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def iou_cxcywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    def to_xyxy(c):
        cx, cy, w, h = c
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return x1, y1, x2, y2

    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def find_packed_by_channels(outputs: Dict[str, np.ndarray], channels: int) -> Tuple[Optional[str], Optional[np.ndarray]]:
    for k, arr in outputs.items():
        x = arr
        # squeeze any singleton dims while >2D
        while x.ndim > 2 and 1 in x.shape:
            x = np.squeeze(x, axis=np.where(np.array(x.shape) == 1)[0][0])
        if x.ndim == 3:
            if x.shape[0] == channels:
                return k, x.reshape(channels, -1)
            if x.shape[2] == channels:
                x2 = np.transpose(x, (2, 0, 1))
                return k, x2.reshape(channels, -1)
        elif x.ndim == 2:
            if x.shape[0] == channels:
                return k, x
            if x.shape[1] == channels:
                return k, x.T
    return None, None


def load_npz_outputs(npz_path: str) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def run_pt_detector(pt_path: str, img_bgr: np.ndarray, det_size: int) -> Tuple[Tuple[float, float, float, float], float]:
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(f"Ultralytics not available: {e}")

    model = YOLO(pt_path)

    # Predict at target size; use .xywhn normalized for robust scaling
    results = model.predict(img_bgr[:, :, ::-1], imgsz=det_size, verbose=False)  # BGR->RGB
    if not results or len(results) == 0:
        raise RuntimeError("PT: No results")
    r0 = results[0]
    if not hasattr(r0, "boxes") or r0.boxes is None or len(r0.boxes) == 0:
        raise RuntimeError("PT: No detections")
    conf = r0.boxes.conf.cpu().numpy()
    best = int(np.argmax(conf))
    xywhn = r0.boxes.xywhn.cpu().numpy()[best]  # normalized [0,1]
    cx, cy, w, h = (xywhn * np.array([det_size, det_size, det_size, det_size])).tolist()
    score = float(conf[best])
    return (float(cx), float(cy), float(w), float(h)), score


def run_onnx_detector(onnx_path: str, img_bgr: np.ndarray, det_size: int, out_dir: str) -> Tuple[Tuple[float, float, float, float], float]:
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError(f"ONNXRuntime not available: {e}")

    x = preprocess_nchw_rgb01(img_bgr, det_size)
    in_npz = os.path.join(out_dir, "onnx_input.npz")
    np.savez(in_npz, images=x)

    sess = ort.InferenceSession(onnx_path)
    # Heuristic: common input name is 'images' or first input
    inputs = sess.get_inputs()
    inp_name = inputs[0].name if inputs else "images"
    outputs = sess.get_outputs()
    out_names = [o.name for o in outputs] or None
    outs = sess.run(out_names, {inp_name: x})

    # Collect to dict for unified parsing
    outs_dict: Dict[str, np.ndarray] = {}
    for i, arr in enumerate(outs):
        key = out_names[i] if out_names and i < len(out_names) else f"out{i}"
        outs_dict[key] = arr

    # Expect 5 channels: cx, cy, w, h, score
    key, packed = find_packed_by_channels(outs_dict, 5)
    if packed is None:
        raise RuntimeError("ONNX: Detector packed output (5 channels) not found")
    bbox = packed[0:4, :]  # (4, P)
    score = sigmoid(packed[4, :])
    pos = int(np.argmax(score))
    bb = bbox[:, pos]
    return (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])), float(score[pos])


def which(cmd: str) -> Optional[str]:
    from shutil import which as swhich
    return swhich(cmd)


def ensure_udocker_installed() -> Optional[str]:
    if which("udocker"):
        return "udocker"
    # Try pip user install
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "udocker"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        pass
    if which("udocker"):
        return "udocker"
    # Try curl script
    try:
        home_local_bin = os.path.join(str(Path.home()), ".local", "bin")
        os.makedirs(home_local_bin, exist_ok=True)
        url = "https://raw.githubusercontent.com/indigo-dc/udocker/main/udocker.py"
        dst = os.path.join(home_local_bin, "udocker")
        subprocess.run(["curl", "-fsSL", url, "-o", dst], check=True)
        os.chmod(dst, 0o755)
        os.environ["PATH"] = home_local_bin + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
    return "udocker" if which("udocker") else None


def run_cvi_in_container(cvi_path: str, input_npz: str, out_npz: str, use_docker: bool, use_udocker: bool) -> None:
    image = os.environ.get("TPU_MLIR_IMAGE", "sophgo/tpuc_dev:latest")
    workdir = os.getcwd()
    container_cmd: Optional[List[str]] = None

    # Write a container-side runner script that resolves model_runner tool
    tmp_script = os.path.join(workdir, ".tmp_triplet_runner.sh")
    script_body = r'''set -euo pipefail
cd /workspace
echo "📦 Inside container: $(python3 -V)"

WHEEL="/workspace/pokedex/models/maixcam/conversion_workspace/tpu_mlir_packages/tpu_mlir-1.21.1-py3-none-any.whl"
if python3 -c "import tpu_mlir" 2>/dev/null; then
  echo '✅ TPU-MLIR present'
else
  echo '📦 Installing TPU-MLIR==1.21.1 ...'
  if [ -f "$WHEEL" ]; then
    python3 -m pip install -q --no-cache-dir "$WHEEL"
  else
    python3 -m pip install -q --no-cache-dir tpu-mlir==1.21.1
  fi
fi

RUNNER=""
if command -v model_runner >/dev/null 2>&1; then
  RUNNER="model_runner"
elif [ -f "/usr/local/bin/model_runner.py" ]; then
  RUNNER="python3 /usr/local/bin/model_runner.py"
else
  RUNNER="python3 -m tpu_mlir.tools.model_runner"
fi

echo "▶️  $RUNNER --model $MODEL --input $INPUT --output $OUTPUT --dump_all_tensors"
$RUNNER --model "$MODEL" --input "$INPUT" --output "$OUTPUT" --dump_all_tensors
'''
    with open(tmp_script, "w", encoding="utf-8") as f:
        f.write(script_body)

    if use_docker and which("docker"):
        subprocess.run(["docker", "pull", image], check=False)
        container_cmd = [
            "docker", "run", "--rm",
            "-e", "PYTHONUNBUFFERED=1",
            "-e", f"MODEL=/workspace/{cvi_path}",
            "-e", f"INPUT=/workspace/{input_npz}",
            "-e", f"OUTPUT=/workspace/{out_npz}",
            "-v", f"{workdir}:/workspace",
            image,
            "bash", "-lc",
            "bash /workspace/.tmp_triplet_runner.sh"
        ]
    else:
        # Try udocker
        if not which("udocker") and use_udocker:
            ensure_udocker_installed()
        if which("udocker"):
            # Prepare udocker runtime (install/setup engines on first use)
            try:
                subprocess.run(["udocker", "--allow-root", "install"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception:
                pass
            try:
                subprocess.run(["udocker", "--allow-root", "setup"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception:
                pass

            cname = f"tpuc_dev_{os.getpid()}"
            # Clean stale container if present
            subprocess.run(["udocker", "--allow-root", "rm", cname], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Pull image and create container
            pr = subprocess.run(["udocker", "--allow-root", "pull", image], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if pr.returncode != 0:
                raise RuntimeError(f"udocker pull failed for image '{image}': {pr.stderr.strip() or pr.stdout.strip()}")

            cr = subprocess.run(["udocker", "--allow-root", "create", f"--name={cname}", image], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if cr.returncode != 0:
                raise RuntimeError(f"udocker create failed: {cr.stderr.strip() or cr.stdout.strip()}")

            container_cmd = [
                "udocker", "--allow-root", "run",
                f"-e=PYTHONUNBUFFERED=1",
                f"-e=MODEL=/workspace/{cvi_path}",
                f"-e=INPUT=/workspace/{input_npz}",
                f"-e=OUTPUT=/workspace/{out_npz}",
                f"-v={workdir}:/workspace",
                cname,
                "bash", "-lc",
                "bash /workspace/.tmp_triplet_runner.sh"
            ]
        else:
            # Fallback: try host model_runner
            mr = "/usr/local/bin/model_runner"
            if not os.path.exists(mr):
                raise RuntimeError("No docker available; udocker not available or not requested; and host model_runner missing. Install docker or pass --use-udocker, or install model_runner on host.")
            container_cmd = [mr, "--model", cvi_path, "--input", input_npz, "--output", out_npz, "--dump_all_tensors"]

    assert container_cmd is not None
    r = subprocess.run(container_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"model_runner failed: {r.stderr.strip() or r.stdout.strip()}\nCMD: {' '.join(container_cmd)}\nHint: If using udocker, try: export TPU_MLIR_IMAGE=sophgo/tpuc_dev:latest and ensure network access, or try a different tag.")
    if not os.path.exists(out_npz):
        raise RuntimeError("model_runner produced no output")


def run_cvi_detector(cvi_path: str, img_bgr: np.ndarray, det_size: int, out_dir: str, use_docker: bool, use_udocker: bool) -> Tuple[Tuple[float, float, float, float], float]:
    x = preprocess_nchw_rgb01(img_bgr, det_size)
    in_npz = os.path.join(out_dir, "cvi_input.npz")
    out_npz = os.path.join(out_dir, "cvi_out.npz")
    np.savez(in_npz, images=x)

    run_cvi_in_container(cvi_path, in_npz, out_npz, use_docker, use_udocker)

    outs = load_npz_outputs(out_npz)
    key, packed = find_packed_by_channels(outs, 5)
    if packed is None:
        raise RuntimeError("CVI: Detector packed output (5 channels) not found")
    bbox = packed[0:4, :]
    score = sigmoid(packed[4, :])
    pos = int(np.argmax(score))
    bb = bbox[:, pos]
    return (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])), float(score[pos])


def main() -> int:
    args = parse_args()
    print("🧪 Detector Triplet Test (PT/ONNX/CVI)")
    print(f"PT:   {args.pt_path}")
    print(f"ONNX: {args.onnx_path}")
    print(f"CVI:  {args.cvi_path}")
    print(f"Images: {len(args.images)}  | det_size={args.det_size}  | IoU thr={args.iou_thr}")

    all_ok = True
    for img_path in args.images:
        print("\n== Image ==")
        print(img_path)
        if not os.path.exists(img_path):
            print("❌ Missing image")
            all_ok = False
            continue
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print("❌ Failed to read image")
            all_ok = False
            continue

        img_out_dir = os.path.join(args.out_dir, Path(img_path).stem)
        ensure_dir(img_out_dir)

        try:
            pt_box, pt_score = run_pt_detector(args.pt_path, img_bgr, args.det_size)
            print(f"PT:  score={pt_score:.4f} box(cx,cy,w,h)@{args.det_size}={tuple(round(v,2) for v in pt_box)}")
        except Exception as e:
            print(f"❌ PT failed: {e}")
            all_ok = False
            continue

        try:
            onnx_box, onnx_score = run_onnx_detector(args.onnx_path, img_bgr, args.det_size, img_out_dir)
            print(f"ONNX: score={onnx_score:.4f} box={tuple(round(v,2) for v in onnx_box)}")
        except Exception as e:
            print(f"❌ ONNX failed: {e}")
            all_ok = False
            continue

        try:
            cvi_box, cvi_score = run_cvi_detector(args.cvi_path, img_bgr, args.det_size, img_out_dir, args.use_docker, args.use_udocker)
            print(f"CVI: score={cvi_score:.4f} box={tuple(round(v,2) for v in cvi_box)}")
        except Exception as e:
            print(f"❌ CVI failed: {e}")
            all_ok = False
            continue

        # Compare IoU across pairs
        iou_pt_onnx = iou_cxcywh(pt_box, onnx_box)
        iou_pt_cvi = iou_cxcywh(pt_box, cvi_box)
        iou_onnx_cvi = iou_cxcywh(onnx_box, cvi_box)
        print(f"IoU PT-ONNX={iou_pt_onnx:.3f}  PT-CVI={iou_pt_cvi:.3f}  ONNX-CVI={iou_onnx_cvi:.3f}")

        if min(iou_pt_onnx, iou_pt_cvi, iou_onnx_cvi) < args.iou_thr:
            print("⚠️  IoU below threshold for at least one pair")
            all_ok = False
        else:
            print("✅ Detector consistent across backends")

    print("\nDone.")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())

