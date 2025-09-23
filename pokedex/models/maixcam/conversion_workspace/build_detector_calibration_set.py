#!/usr/bin/env python3
"""
Build a detector calibration set with off-center positives and negative backgrounds.

Outputs:
- A folder of 256x256 JPEGs (default: data/calib_det)
- A list file with absolute image paths (default: data/calib_det_list.txt)

Usage (defaults are sensible for this repo):
  python build_detector_calibration_set.py \
    --src-images data/yolo_dataset/train/images \
    --out-dir data/calib_det \
    --list-path data/calib_det_list.txt \
    --imgsz 256 --num-pos 600 --num-neg 400 --seed 0
"""

import argparse
import glob
import os
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build detector calibration set")
    p.add_argument("--src-images", nargs="+", default=["data/yolo_dataset/train/images"],
                   help="One or more source image directories for positives")
    p.add_argument("--out-dir", type=str, default="data/calib_det",
                   help="Output directory for calibration images")
    p.add_argument("--list-path", type=str, default="data/calib_det_list.txt",
                   help="Path to write absolute list of calibration images")
    p.add_argument("--imgsz", type=int, default=256, help="Calibration image size (square)")
    p.add_argument("--num-pos", type=int, default=600, help="Number of off-center positive samples")
    p.add_argument("--num-neg", type=int, default=400, help="Number of negative background samples")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    return p.parse_args()


def find_images(dirs: List[str]) -> List[str]:
    paths: List[str] = []
    for d in dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(glob.glob(os.path.join(d, ext)))
    return sorted(paths)


def make_offcenter(img: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        canvas = np.full((size, size, 3), 127, dtype=np.uint8)
        return canvas
    s = int(min(h, w) * rng.uniform(0.35, 0.6))
    s = max(8, min(s, min(h, w)))
    x = rng.randint(0, max(0, w - s))
    y = rng.randint(0, max(0, h - s))
    crop = img[y:y + s, x:x + s]
    canvas = np.full((size, size, 3), 127, dtype=np.uint8)
    tw = int(s * rng.uniform(0.5, 0.8))
    th = tw
    crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
    ox = rng.randint(0, size - tw)
    oy = rng.randint(0, size - th)
    canvas[oy:oy + th, ox:ox + tw] = crop
    return canvas


def make_negative(kind: int, size: int, rng_np: np.random.Generator) -> np.ndarray:
    if kind == 0:
        return np.full((size, size, 3), 127, dtype=np.uint8)
    if kind == 1:
        color = rng_np.integers(0, 256, size=3, dtype=np.uint8)
        return np.tile(color[None, None, :], (size, size, 1))
    if kind == 2:
        return rng_np.normal(128, 40, (size, size, 3)).clip(0, 255).astype(np.uint8)
    x = np.linspace(0, 255, size, dtype=np.uint8)
    return np.dstack([
        np.tile(x, (size, 1)),
        np.full((size, size), 128, np.uint8),
        np.flipud(np.tile(x, (size, 1)))
    ])


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    # Collect positives
    src_imgs = find_images(args.src_images)
    rng.shuffle(src_imgs)
    src_imgs = src_imgs[: args.num_pos * 2]  # oversample a bit in case of read failures

    written: List[str] = []

    # Write off-center positives
    pos_written = 0
    for i, p in enumerate(src_imgs):
        if pos_written >= args.num_pos:
            break
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        out = make_offcenter(im, args.imgsz, rng)
        op = out_dir / f"pos_off_{pos_written:05d}.jpg"
        cv2.imwrite(str(op), out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        written.append(str(op.resolve()))
        pos_written += 1

    # Write negatives
    for i in range(args.num_neg):
        out = make_negative(i % 4, args.imgsz, rng_np)
        op = out_dir / f"neg_{i:05d}.jpg"
        cv2.imwrite(str(op), out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        written.append(str(op.resolve()))

    # Save list
    list_path = Path(args.list_path)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(written))

    print(f"Wrote {len(written)} calibration images to {out_dir}")
    print(f"List file: {list_path} ({len(written)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

