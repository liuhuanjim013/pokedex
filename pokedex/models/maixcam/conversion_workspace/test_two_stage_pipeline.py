#!/usr/bin/env python3
"""
Two-Stage Detector + Classifier CVIModel Test (Docker/TPU-MLIR)

- Runs detector (YOLO11n 1-class @256) to get best box
- Crops input with 15% padding and runs classifier (YOLO11n-cls 1025 @224)
- Uses model_runner in the TPU-MLIR container

Usage:
  python test_two_stage_pipeline.py \
    --det models/maixcam/pokemon_det1_int8.cvimodel \
    --cls models/maixcam/pokemon_cls1025_int8.cvimodel \
    --classes classes.txt \
    --images images/0001_001.jpg images/0025_2997.jpg

If no images are provided, a small default set is used (Bulbasaur/Pikachu etc.).
"""

import argparse
import os
import sys
import subprocess
from typing import List, Tuple
import numpy as np
import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage CVIModel test (det+cls)")
    p.add_argument("--det", type=str, default="models/maixcam/pokemon_det1_int8.cvimodel",
                   help="Detector CVIModel path")
    p.add_argument("--cls", type=str, default="models/maixcam/pokemon_cls1025_int8.cvimodel",
                   help="Classifier CVIModel path")
    p.add_argument("--classes", type=str, default="classes.txt",
                   help="classes.txt with 1025 names (one per line)")
    p.add_argument("--pad", type=float, default=0.15, help="Crop padding fraction around det box")
    p.add_argument("--det-size", type=int, default=256, help="Detector input size")
    p.add_argument("--cls-size", type=int, default=224, help="Classifier input size")
    p.add_argument("--out-dir", type=str, default="runner_out_two_stage", help="model_runner outputs")
    p.add_argument("images", nargs="*", default=[
        "images/0001_001.jpg",
        "images/0004_407.jpg",
        "images/0007_794.jpg",
        "images/0025_2997.jpg",
        "images/0150_17608.jpg",
    ])
    return p.parse_args()


def load_class_names(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x)
    exps = np.exp(x - x_max)
    return exps / np.sum(exps)


def preprocess_nchw_rgb01(img_bgr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x


def run_model_runner(model_path: str, input_npz: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "out.npz")
    cmd = [
        "/usr/local/bin/model_runner",
        "--model", model_path,
        "--input", input_npz,
        "--output", out_path,
        "--dump_all_tensors",
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print("❌ model_runner failed:")
        print(r.stderr.strip())
        raise RuntimeError("model_runner error")
    if not os.path.exists(out_path):
        raise RuntimeError("model_runner produced no output")
    return out_path


def load_npz_outputs(npz_path: str) -> dict:
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def find_packed_by_channels(outputs: dict, channels: int) -> Tuple[str, np.ndarray]:
    for k, arr in outputs.items():
        x = arr
        # squeeze singletons
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


def det_infer_best_box(det_model: str, img_bgr: np.ndarray, det_size: int, out_dir: str) -> Tuple[Tuple[float, float, float, float], float]:
    # Prepare detector input
    x_det = preprocess_nchw_rgb01(img_bgr, (det_size, det_size))
    in_npz = os.path.join(out_dir, "det_input.npz")
    np.savez(in_npz, images=x_det)
    det_npz = run_model_runner(det_model, in_npz, out_dir)
    outs = load_npz_outputs(det_npz)
    # Expect 5 channels: [cx, cy, w, h, score]
    key, packed = find_packed_by_channels(outs, 5)
    if packed is None:
        raise RuntimeError("Detector output with 5 channels not found")
    bbox = packed[0:4, :]  # (4, P)
    score = sigmoid(packed[4, :])  # (P,)
    best_pos = int(np.argmax(score))
    best_bbox = bbox[:, best_pos]
    best_score = float(score[best_pos])
    return (float(best_bbox[0]), float(best_bbox[1]), float(best_bbox[2]), float(best_bbox[3])), best_score


def crop_with_pad(img_bgr: np.ndarray, box_cxcywh: Tuple[float, float, float, float], pad: float, det_size: int) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    sx = W / float(det_size)
    sy = H / float(det_size)
    cx, cy, w, h = box_cxcywh
    # Map to original scale
    cx_o = cx * sx
    cy_o = cy * sy
    w_o = w * sx
    h_o = h * sy
    # Pad
    w_o *= (1.0 + pad)
    h_o *= (1.0 + pad)
    x1 = max(int(cx_o - w_o / 2), 0)
    y1 = max(int(cy_o - h_o / 2), 0)
    x2 = min(int(cx_o + w_o / 2), W - 1)
    y2 = min(int(cy_o + h_o / 2), H - 1)
    if x2 <= x1 or y2 <= y1:
        # Fallback center square
        s = min(W, H)
        x1 = (W - s) // 2
        y1 = (H - s) // 2
        x2 = x1 + s - 1
        y2 = y1 + s - 1
    return img_bgr[y1:y2+1, x1:x2+1].copy()


def cls_infer_topk(cls_model: str, crop_bgr: np.ndarray, cls_size: int, out_dir: str, class_names: List[str], k: int = 5) -> List[Tuple[int, str, float]]:
    x_cls = preprocess_nchw_rgb01(crop_bgr, (cls_size, cls_size))
    in_npz = os.path.join(out_dir, "cls_input.npz")
    np.savez(in_npz, images=x_cls)
    cls_npz = run_model_runner(cls_model, in_npz, out_dir)
    outs = load_npz_outputs(cls_npz)
    # Find logits vector (1025)
    key, packed = find_packed_by_channels(outs, len(class_names) if class_names else 1025)
    if packed is None:
        # try to search any 1xN output
        vec = None
        for k2, arr in outs.items():
            v = np.squeeze(arr)
            if v.ndim == 1 and (len(class_names) == 0 or v.shape[0] == len(class_names)):
                vec = v
                break
        if vec is None:
            raise RuntimeError("Classifier logits not found in outputs")
        logits = vec.astype(np.float32)
    else:
        logits = np.squeeze(packed)
        if logits.ndim > 1:
            logits = np.mean(logits, axis=1)
    probs = softmax(logits)
    idxs = np.argsort(probs)[-k:][::-1]
    results = []
    for i in idxs:
        name = class_names[i] if 0 <= i < len(class_names) else f"id_{i}"
        results.append((int(i), name, float(probs[i])))
    return results


def main() -> int:
    args = parse_args()
    class_names = load_class_names(args.classes)
    if not class_names:
        print("⚠️  classes.txt not found or empty; results will show IDs only")
        class_names = [f"id_{i}" for i in range(1025)]

    print("🧪 Two-Stage CVIModel Test")
    print(f"Detector:   {args.det}")
    print(f"Classifier: {args.cls}")
    print(f"Classes:    {args.classes} ({len(class_names)})")

    ok = True
    for img_path in args.images:
        print("\n== Test ==")
        print(f"📸 Image: {img_path}")
        if not os.path.exists(img_path):
            print("❌ Missing image")
            ok = False
            continue
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print("❌ Failed to read image")
            ok = False
            continue
        try:
            box, score = det_infer_best_box(args.det, img_bgr, args.det_size, args.out_dir)
            print(f"🔎 Det best score: {score:.4f}, box (cx,cy,w,h @ {args.det_size}): {box}")
            crop = crop_with_pad(img_bgr, box, args.pad, args.det_size)
            print(f"✂️  Crop: {crop.shape[1]}x{crop.shape[0]}")
            top5 = cls_infer_topk(args.cls, crop, args.cls_size, args.out_dir, class_names, k=5)
            print("🥇 Top-5:")
            for rank, (cid, name, prob) in enumerate(top5, 1):
                print(f"  {rank}. {name} (ID {cid}) : {prob:.4f}")
        except Exception as e:
            print(f"❌ Error: {e}")
            ok = False

    print("\nDone.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

