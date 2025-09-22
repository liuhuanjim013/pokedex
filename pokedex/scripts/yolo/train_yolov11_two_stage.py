#!/usr/bin/env python3
"""
Two-Stage YOLOv11n Training Script (Detector + Classifier)

Stage A: YOLO11n detector (1 class "pokemon") at 256
Stage B: YOLO11n-cls classifier (1025 classes) at 224

- Uses Ultralytics CLI under the hood (detect/classify train/export)
- Compatible with Colab, W&B, and existing dataset layout
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root for imports like src.data.pokemon_names
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.data.pokemon_names import POKEMON_NAMES
except Exception:
    POKEMON_NAMES = {}


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("two_stage_train")


def run_cmd(cmd: str, logger: logging.Logger) -> None:
    logger.info(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/export two-stage YOLO11n pipeline")
    parser.add_argument("--det-data", type=str, default="configs/yolov11/pokemon_det1.yaml",
                        help="Detector data YAML (1-class)")
    parser.add_argument("--det-model", type=str, default="yolo11n.pt",
                        help="Ultralytics detector model to start from")
    parser.add_argument("--det-imgsz", type=int, default=256, help="Detector image size")
    parser.add_argument("--det-epochs", type=int, default=80, help="Detector epochs")
    parser.add_argument("--det-batch", type=int, default=32, help="Detector batch size")
    parser.add_argument("--det-run-name", type=str, default="pokemon_det1_yolo11n_256",
                        help="Detector run name")

    parser.add_argument("--cls-model", type=str, default="yolo11n-cls.pt",
                        help="Ultralytics classifier model to start from")
    parser.add_argument("--cls-data", type=str, default="data/classify_dataset",
                        help="Classifier dataset root (train/ and val/ subdirs)")
    parser.add_argument("--cls-imgsz", type=int, default=224, help="Classifier image size")
    parser.add_argument("--cls-epochs", type=int, default=80, help="Classifier epochs")
    parser.add_argument("--cls-batch", type=int, default=64, help="Classifier batch size")
    parser.add_argument("--cls-run-name", type=str, default="pokemon_cls1025_yolo11n_224",
                        help="Classifier run name")

    parser.add_argument("--export", action="store_true", help="Export ONNX after training")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset")
    parser.add_argument("--outdir", type=str, default="models/maixcam/exports",
                        help="Directory to copy exports into")
    parser.add_argument("--no-resume", action="store_true", help="Disable auto-resume if checkpoints exist")
    return parser.parse_args()


def _yolo_cli() -> str:
    """Return the YOLO CLI launcher. Prefer 'yolo' if on PATH, else 'python -m ultralytics'."""
    if shutil.which("yolo"):
        return "yolo"
    # Fallback to python -m ultralytics
    return f"{sys.executable} -m ultralytics"
def _class_dir_name(pokemon_id: str) -> str:
    name = POKEMON_NAMES.get(pokemon_id, None)
    return f"{pokemon_id}_{name}" if name else pokemon_id


def _find_latest_run_dir(task: str, base_name: str) -> Path:
    """Find an existing Ultralytics run dir that matches base_name or base_name<idx> under runs/<task>/.

    Returns Path('') if none found.
    """
    task_root = Path("runs") / task
    if not task_root.exists():
        return Path("")
    # Exact match first
    exact = task_root / base_name
    if exact.exists():
        return exact
    # Fuzzy match: base_name, base_name2, base_name3 ... pick most recent mtime
    candidates = []
    for d in task_root.iterdir():
        if not d.is_dir():
            continue
        if d.name == base_name or d.name.startswith(base_name):
            candidates.append((d.stat().st_mtime, d))
    if not candidates:
        return Path("")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def ensure_classify_dataset(yolo_root: Path, cls_root: Path, logger: logging.Logger) -> None:
    """Build a classification dataset from existing YOLO dataset via symlinks.

    yolo_root structure:
      yolo_root/
        train/images/*.jpg
        validation/images/*.jpg

    cls_root structure created:
      cls_root/
        train/<0001_bulbasaur>/*.jpg
        val/<0001_bulbasaur>/*.jpg
    """
    # If dataset already present with expected structure, skip
    train_dir = cls_root / "train"
    val_dir = cls_root / "val"
    if train_dir.exists() and val_dir.exists():
        logger.info(f"Classifier dataset exists at {cls_root}, skipping build.")
        return

    logger.info(f"Building classifier dataset at {cls_root} from {yolo_root} ...")
    for split_yolo, split_cls in (("train", "train"), ("validation", "val")):
        src_images_dir = yolo_root / split_yolo / "images"
        if not src_images_dir.exists():
            logger.warning(f"Missing YOLO images dir: {src_images_dir}")
            continue
        dst_split_dir = cls_root / split_cls
        dst_split_dir.mkdir(parents=True, exist_ok=True)

        for img_path in src_images_dir.rglob("*"):
            if not img_path.is_file():
                continue
            suffix = img_path.suffix.lower()
            if suffix not in (".jpg", ".jpeg", ".png"):
                continue
            stem = img_path.stem  # e.g., 0001_001
            pokemon_id = stem.split("_")[0]
            class_dir = dst_split_dir / _class_dir_name(pokemon_id)
            class_dir.mkdir(parents=True, exist_ok=True)

            dst = class_dir / img_path.name
            if dst.exists():
                continue
            try:
                os.symlink(img_path.resolve(), dst)
            except OSError:
                shutil.copy2(img_path, dst)


def ensure_det1_dataset(yolo_root: Path, det_root: Path, logger: logging.Logger) -> None:
    """Create a 1-class detector dataset by rewriting all labels' class_ids to 0.

    Keeps original xywh normalized coordinates; mirrors directory structure.
    Images are symlinked (fallback to copy), labels are rewritten.
    """
    splits = ["train", "validation", "test"]
    for split in splits:
        src_img_dir = yolo_root / split / "images"
        src_lbl_dir = yolo_root / split / "labels"
        dst_img_dir = det_root / split / "images"
        dst_lbl_dir = det_root / split / "labels"
        if not src_img_dir.exists() or not src_lbl_dir.exists():
            logger.warning(f"Skipping split '{split}' (missing in source): {src_img_dir} or {src_lbl_dir}")
            continue
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Link/copy images
        for img_path in src_img_dir.rglob("*"):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            dst_img = dst_img_dir / img_path.name
            if not dst_img.exists():
                try:
                    os.symlink(img_path.resolve(), dst_img)
                except OSError:
                    shutil.copy2(img_path, dst_img)

        # Rewrite labels
        for lbl_path in src_lbl_dir.rglob("*.txt"):
            rel = lbl_path.name
            dst_lbl = dst_lbl_dir / rel
            if dst_lbl.exists():
                continue
            try:
                with open(lbl_path, "r", encoding="utf-8") as f_in, open(dst_lbl, "w", encoding="utf-8") as f_out:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        # Force class id to 0, keep bbox
                        f_out.write("0 " + " ".join(parts[1:5]) + "\n")
            except Exception as e:
                logger.warning(f"Failed to rewrite label {lbl_path}: {e}")

        # Add synthetic negatives (empty labels) to improve presence detection
        try:
            from PIL import Image
            import numpy as np
        except Exception as e:
            logger.warning(f"PIL/numpy not available for negative generation: {e}")
            continue

        def _save_jpg(img_arr: np.ndarray, out_path: Path) -> None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.fromarray(img_arr.astype(np.uint8), mode="RGB")
            img.save(str(out_path), format="JPEG", quality=90)

        # target size close to detector input; Ultralytics will resize as needed
        size = 256
        rng = np.random.default_rng(2025)
        if split == "train":
            num_negs = 500
        elif split == "validation":
            num_negs = 200
        else:
            num_negs = 200

        logger.info(f"Generating {num_negs} negative images for split '{split}' …")
        for i in range(num_negs):
            mode = i % 4
            if mode == 0:
                # solid gray
                arr = np.full((size, size, 3), 127, dtype=np.uint8)
            elif mode == 1:
                # solid random color
                color = rng.integers(0, 256, size=3, dtype=np.uint8)
                arr = np.tile(color[None, None, :], (size, size, 1))
            elif mode == 2:
                # gaussian noise
                arr = rng.normal(loc=128, scale=40, size=(size, size, 3)).clip(0, 255).astype(np.uint8)
            else:
                # horizontal gradient
                x = np.linspace(0, 255, size, dtype=np.uint8)
                arr = np.dstack([np.tile(x, (size, 1)), np.full((size, size), 128, np.uint8), np.flipud(np.tile(x, (size, 1)))])

            neg_stem = f"neg_{i:05d}"
            out_img = dst_img_dir / f"{neg_stem}.jpg"
            out_lbl = dst_lbl_dir / f"{neg_stem}.txt"
            if not out_img.exists():
                _save_jpg(arr, out_img)
            if not out_lbl.exists():
                with open(out_lbl, "w", encoding="utf-8") as f:
                    f.write("")  # empty label => no objects


def write_det1_yaml(template_yaml: Path, out_yaml: Path, det_root: Path, logger: logging.Logger) -> None:
    """Write a detector YAML pointing to det_root using the known subpaths."""
    content = (
        "path: " + str(det_root).replace("\\", "/") + "\n"
        "train: train/images\n"
        "val: validation/images\n"
        "test: test/images\n"
        "names: [\"pokemon\"]\n"
        "nc: 1\n"
    )
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Wrote detector YAML -> {out_yaml} (path={det_root})")



def main() -> None:
    logger = setup_logging()
    args = parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # 1) Train detector (1 class) - ensure relabeled dataset and YAML pointing to it
    det_src_root = Path("data/yolo_dataset")
    det_out_root = Path("data/yolo_dataset_det1")
    ensure_det1_dataset(det_src_root, det_out_root, logger)

    det_yaml_autogen = Path("configs/yolov11/pokemon_det1_autogen.yaml")
    write_det1_yaml(Path(args.det_data), det_yaml_autogen, det_out_root, logger)

    # Build detector train/export flow with resume/skip logic
    resolved_det_dir = _find_latest_run_dir("detect", args.det_run_name)
    det_run_dir = resolved_det_dir if resolved_det_dir else (Path("runs") / "detect" / args.det_run_name)
    det_last = det_run_dir / "weights" / "last.pt"
    det_best_path = det_run_dir / "weights" / "best.pt"
    yolo_cli = _yolo_cli()

    if args.export and det_best_path.exists():
        logger.info(f"Detector best.pt found at {det_best_path}, skipping training and exporting only.")
    else:
        if det_last.exists() and not args.no_resume:
            det_train_cmd = (
                f"{yolo_cli} detect train resume=True project=runs name={det_run_dir.name} exist_ok=True save_period=1"
            )
        else:
            det_train_cmd = (
                f"{yolo_cli} detect train model={args.det_model} data={det_yaml_autogen} "
                f"imgsz={args.det_imgsz} epochs={args.det_epochs} batch={args.det_batch} "
                # Augmentations to improve off-center robustness
                f"degrees=5 translate=0.20 scale=0.50 shear=0.0 flipud=0.0 fliplr=0.5 "
                f"hsv_h=0.015 hsv_s=0.70 hsv_v=0.40 mosaic=0.10 mixup=0.0 "
                f"cos_lr=True project=runs name={args.det_run_name} exist_ok=True save_period=1"
            )
        run_cmd(det_train_cmd, logger)
        # refresh paths in case Ultralytics created a new indexed run dir
        det_run_dir = _find_latest_run_dir("detect", args.det_run_name) or det_run_dir
        det_best_path = (det_run_dir / "weights" / "best.pt")

    if args.export:
        # fallback if best not found
        det_best_str = str(det_best_path if det_best_path.exists() else Path("runs/detect/train/weights/best.pt"))
        det_export_cmd = (
            f"{yolo_cli} export model={det_best_str} format=onnx opset={args.opset} "
            f"imgsz={args.det_imgsz} dynamic=False simplify=True"
        )
        run_cmd(det_export_cmd, logger)

        det_onnx = Path(det_best_str).with_suffix(".onnx")
        if det_onnx.exists():
            run_cmd(f"cp {det_onnx} {os.path.join(args.outdir, 'best_det.onnx')}", logger)

    # 2) Train classifier (1025 classes)
    # Auto-build classification dataset if needed
    try:
        ensure_classify_dataset(Path("data/yolo_dataset"), Path(args.cls_data), logger)
    except Exception as e:
        logger.warning(f"Failed to auto-build classification dataset: {e}")

    # Build classifier train/export flow with resume/skip logic
    resolved_cls_dir = _find_latest_run_dir("classify", args.cls_run_name)
    cls_run_dir = resolved_cls_dir if resolved_cls_dir else (Path("runs") / "classify" / args.cls_run_name)
    cls_last = cls_run_dir / "weights" / "last.pt"
    cls_best_path = cls_run_dir / "weights" / "best.pt"
    if args.export and cls_best_path.exists():
        logger.info(f"Classifier best.pt found at {cls_best_path}, skipping training and exporting only.")
    else:
        if cls_last.exists() and not args.no_resume:
            cls_train_cmd = (
                f"{yolo_cli} classify train resume=True project=runs name={cls_run_dir.name} exist_ok=True save_period=1"
            )
        else:
            cls_train_cmd = (
                f"{yolo_cli} classify train model={args.cls_model} data={args.cls_data} "
                f"imgsz={args.cls_imgsz} epochs={args.cls_epochs} batch={args.cls_batch} "
                f"cos_lr=True project=runs name={args.cls_run_name} exist_ok=True save_period=1"
            )
        run_cmd(cls_train_cmd, logger)
        cls_run_dir = _find_latest_run_dir("classify", args.cls_run_name) or cls_run_dir
        cls_best_path = (cls_run_dir / "weights" / "best.pt")

    if args.export:
        cls_best_str = str(cls_best_path if cls_best_path.exists() else Path("runs/classify/train/weights/best.pt"))
        cls_export_cmd = (
            f"{yolo_cli} export model={cls_best_str} format=onnx opset={args.opset} "
            f"imgsz={args.cls_imgsz} dynamic=False simplify=True"
        )
        run_cmd(cls_export_cmd, logger)

        cls_onnx = Path(cls_best_str).with_suffix(".onnx")
        if cls_onnx.exists():
            run_cmd(f"cp {cls_onnx} {os.path.join(args.outdir, 'best_cls.onnx')}", logger)

    logger.info("Two-stage training complete.")
    if args.export:
        logger.info(f"Exports staged under: {args.outdir}")


if __name__ == "__main__":
    main()

