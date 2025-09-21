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
    p.add_argument("--presence-thr", type=float, default=0.35, help="Presence threshold on score/conf")
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


def make_augmented_variants(img_bgr: np.ndarray, size: int) -> Dict[str, np.ndarray]:
    """Create variants: original-resized, shrink-and-offcenter, and absent background.
    Returns dict: name -> BGR image size (size,size).
    """
    # Original resized
    orig = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)

    # Absent: uniform mid-gray background
    absent = np.full((size, size, 3), 127, dtype=np.uint8)

    # Shrink and place off-center
    # Choose scale so object occupies 30%-60% of area
    scale = float(np.random.uniform(0.45, 0.7))
    target_w = int(max(8, round(size * scale)))
    target_h = target_w
    small = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 127, dtype=np.uint8)
    # pick an off-center location
    max_x = max(0, size - target_w)
    max_y = max(0, size - target_h)
    # ensure distance from center is at least 10% of size
    attempts = 0
    while True:
        x = int(np.random.uniform(0, max_x + 1))
        y = int(np.random.uniform(0, max_y + 1))
        cx = x + target_w / 2.0
        cy = y + target_h / 2.0
        if abs(cx - size / 2.0) > 0.1 * size or abs(cy - size / 2.0) > 0.1 * size or attempts > 10:
            break
        attempts += 1
    canvas[y:y+target_h, x:x+target_w] = cv2.resize(small, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return {"orig": orig, "offcenter": canvas, "absent": absent}


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


class CVIRunner:
    def __init__(self, use_docker: bool, use_udocker: bool):
        self.image = os.environ.get("TPU_MLIR_IMAGE", "sophgo/tpuc_dev:latest")
        self.workdir = os.getcwd()
        self.use_docker = use_docker and which("docker") is not None
        self.use_udocker = (not self.use_docker) and use_udocker
        self.cname: Optional[str] = None
        self._ensure_tmp_script()
        if self.use_docker:
            subprocess.run(["docker", "pull", self.image], check=False)
        else:
            if self.use_udocker:
                if not which("udocker"):
                    ensure_udocker_installed()
                # Setup udocker engines
                try:
                    subprocess.run(["udocker", "--allow-root", "install"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except Exception:
                    pass
                try:
                    subprocess.run(["udocker", "--allow-root", "setup"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except Exception:
                    pass
                # Pull once
                pr = subprocess.run(["udocker", "--allow-root", "pull", self.image], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if pr.returncode != 0:
                    raise RuntimeError(f"udocker pull failed for image '{self.image}': {pr.stderr.strip() or pr.stdout.strip()}")
                # Create named container once
                self.cname = f"tpuc_dev_{os.getpid()}"
                # Clean stale
                subprocess.run(["udocker", "--allow-root", "rm", self.cname], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                cr = subprocess.run(["udocker", "--allow-root", "create", f"--name={self.cname}", self.image], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if cr.returncode != 0:
                    raise RuntimeError(f"udocker create failed: {cr.stderr.strip() or cr.stdout.strip()}")
            else:
                # Host fallback handled in run()
                pass

    def _ensure_tmp_script(self) -> None:
        tmp_script = os.path.join(self.workdir, ".tmp_triplet_runner.sh")
        if os.path.exists(tmp_script):
            return
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

    def run(self, cvi_path: str, input_npz: str, out_npz: str) -> None:
        if self.use_docker:
            cmd = [
                "docker", "run", "--rm",
                "-e", "PYTHONUNBUFFERED=1",
                "-e", f"MODEL=/workspace/{cvi_path}",
                "-e", f"INPUT=/workspace/{input_npz}",
                "-e", f"OUTPUT=/workspace/{out_npz}",
                "-v", f"{self.workdir}:/workspace",
                self.image,
                "bash", "-lc",
                "bash /workspace/.tmp_triplet_runner.sh"
            ]
        elif self.use_udocker and self.cname:
            cmd = [
                "udocker", "--allow-root", "run",
                f"-e=PYTHONUNBUFFERED=1",
                f"-e=MODEL=/workspace/{cvi_path}",
                f"-e=INPUT=/workspace/{input_npz}",
                f"-e=OUTPUT=/workspace/{out_npz}",
                f"-v={self.workdir}:/workspace",
                self.cname,
                "bash", "-lc",
                "bash /workspace/.tmp_triplet_runner.sh"
            ]
        else:
            # Host fallback
            mr = "/usr/local/bin/model_runner"
            if not os.path.exists(mr):
                raise RuntimeError("No docker; no udocker; and host model_runner missing.")
            cmd = [mr, "--model", cvi_path, "--input", input_npz, "--output", out_npz, "--dump_all_tensors"]

        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"model_runner failed: {r.stderr.strip() or r.stdout.strip()}\nCMD: {' '.join(cmd)}")
        if not os.path.exists(out_npz):
            raise RuntimeError("model_runner produced no output")


def run_cvi_in_container(cvi_path: str, input_npz: str, out_npz: str, use_docker: bool, use_udocker: bool) -> None:
    runner = CVIRunner(use_docker=use_docker, use_udocker=use_udocker)
    runner.run(cvi_path, input_npz, out_npz)
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

    # Prepare augmented tasks
    tasks: List[Tuple[str, str, np.ndarray]] = []  # (img_path, variant, bgr)
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"❌ Missing image: {img_path}")
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"❌ Failed to read image: {img_path}")
            continue
        variants = make_augmented_variants(img, args.det_size)
        for vname, vbgr in variants.items():
            tasks.append((img_path, vname, vbgr))

    if not tasks:
        print("No images to process.")
        return 1

    results: Dict[Tuple[str, str], Dict[str, Tuple[Tuple[float, float, float, float], float, bool]]] = {}

    # Run PT over all tasks
    print("\n▶️  Running PT over all augmented variants...")
    for img_path, vname, vbgr in tasks:
        key = (img_path, vname)
        out_dir = os.path.join(args.out_dir, Path(img_path).stem, vname)
        ensure_dir(out_dir)
        try:
            box, score = run_pt_detector(args.pt_path, vbgr, args.det_size)
            present = score >= args.presence_thr
            results.setdefault(key, {})["pt"] = (box, score, present)
        except Exception as e:
            results.setdefault(key, {})["pt_err"] = ((0.0, 0.0, 0.0, 0.0), 0.0, False)
            print(f"PT fail [{img_path} {vname}]: {e}")

    # Run ONNX over all tasks
    print("\n▶️  Running ONNX over all augmented variants...")
    for img_path, vname, vbgr in tasks:
        key = (img_path, vname)
        out_dir = os.path.join(args.out_dir, Path(img_path).stem, vname)
        ensure_dir(out_dir)
        try:
            box, score = run_onnx_detector(args.onnx_path, vbgr, args.det_size, out_dir)
            present = score >= args.presence_thr
            results.setdefault(key, {})["onnx"] = (box, score, present)
    	except Exception as e:
            results.setdefault(key, {})["onnx_err"] = ((0.0, 0.0, 0.0, 0.0), 0.0, False)
            print(f"ONNX fail [{img_path} {vname}]: {e}")

    # Run CVI over all tasks
    print("\n▶️  Running CVI over all augmented variants...")
    cvi_runner = CVIRunner(use_docker=args.use_docker, use_udocker=args.use_udocker)
    for img_path, vname, vbgr in tasks:
        key = (img_path, vname)
        out_dir = os.path.join(args.out_dir, Path(img_path).stem, vname)
        ensure_dir(out_dir)
        try:
            # Save input npz and run in container via runner instance
            x = preprocess_nchw_rgb01(vbgr, args.det_size)
            in_npz = os.path.join(out_dir, "cvi_input.npz")
            out_npz = os.path.join(out_dir, "cvi_out.npz")
            np.savez(in_npz, images=x)
            cvi_runner.run(args.cvi_path, in_npz, out_npz)
            outs = load_npz_outputs(out_npz)
            keyp, packed = find_packed_by_channels(outs, 5)
            if packed is None:
                raise RuntimeError("CVI packed output (5) not found")
            bbox = packed[0:4, :]
            score = sigmoid(packed[4, :])
            pos = int(np.argmax(score))
            bb = bbox[:, pos]
            s = float(score[pos])
            present = s >= args.presence_thr
            results.setdefault(key, {})["cvi"] = ((float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])), s, present)
        except Exception as e:
            results.setdefault(key, {})["cvi_err"] = ((0.0, 0.0, 0.0, 0.0), 0.0, False)
            print(f"CVI fail [{img_path} {vname}]: {e}")

    # Report per image
    print("\n📊 Summary per image and variant:")
    overall_ok = True
    for img_path in args.images:
        group = [k for k in results.keys() if k[0] == img_path]
        if not group:
            continue
        print(f"\n== {img_path} ==")
        for vname in ("orig", "offcenter", "absent"):
            key = (img_path, vname)
            if key not in results:
                continue
            r = results[key]
            def fmt(name):
                if name in r:
                    box, score, present = r[name]
                    return f"{name.upper()}: present={int(present)} score={score:.3f} box=({box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f})"
                elif f"{name}_err" in r:
                    return f"{name.upper()}: ERROR"
                else:
                    return f"{name.upper()}: N/A"
            print(f"- Variant: {vname}")
            print("  "+fmt("pt"))
            print("  "+fmt("onnx"))
            print("  "+fmt("cvi"))
            # IoU if at least two present
            if all(k in r for k in ("pt", "onnx", "cvi")):
                ptb, _, ptp = r["pt"]
                onb, _, onp = r["onnx"]
                cvb, _, cvp = r["cvi"]
                if sum([ptp, onp, cvp]) >= 2:
                    i12 = iou_cxcywh(ptb, onb) if (ptp and onp) else None
                    i13 = iou_cxcywh(ptb, cvb) if (ptp and cvp) else None
                    i23 = iou_cxcywh(onb, cvb) if (onp and cvp) else None
                    def f(x):
                        return f"{x:.3f}" if x is not None else "-"
                    print(f"  IoU PT-ONNX={f(i12)} PT-CVI={f(i13)} ONNX-CVI={f(i23)}")
                    if any(v is not None and v < args.iou_thr for v in (i12, i13, i23)):
                        overall_ok = False

    print("\nDone.")
    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(main())

