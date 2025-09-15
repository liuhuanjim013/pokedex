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
import subprocess
import sys
from pathlib import Path


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
    parser.add_argument("--cls-data", type=str, default="dataset",
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
    return parser.parse_args()


def main() -> None:
    logger = setup_logging()
    args = parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # 1) Train detector (1 class)
    det_train_cmd = (
        f"yolo detect train model={args.det_model} data={args.det_data} "
        f"imgsz={args.det_imgsz} epochs={args.det_epochs} batch={args.det_batch} "
        f"cos_lr=True project=runs name={args.det_run_name}"
    )
    run_cmd(det_train_cmd, logger)

    det_best = f"runs/detect/{args.det_run_name}/weights/best.pt"
    if not Path(det_best).exists():
        # fallback to default Ultralytics path
        det_best = "runs/detect/train/weights/best.pt"

    if args.export:
        det_export_cmd = (
            f"yolo export model={det_best} format=onnx opset={args.opset} "
            f"imgsz={args.det_imgsz} dynamic=False simplify=True"
        )
        run_cmd(det_export_cmd, logger)

        det_onnx = Path(det_best).with_suffix(".onnx")
        if det_onnx.exists():
            run_cmd(f"cp {det_onnx} {os.path.join(args.outdir, 'best_det.onnx')}", logger)

    # 2) Train classifier (1025 classes)
    cls_train_cmd = (
        f"yolo classify train model={args.cls_model} data={args.cls_data} "
        f"imgsz={args.cls_imgsz} epochs={args.cls_epochs} batch={args.cls_batch} "
        f"cos_lr=True project=runs name={args.cls_run_name}"
    )
    run_cmd(cls_train_cmd, logger)

    cls_best = f"runs/classify/{args.cls_run_name}/weights/best.pt"
    if not Path(cls_best).exists():
        cls_best = "runs/classify/train/weights/best.pt"

    if args.export:
        cls_export_cmd = (
            f"yolo export model={cls_best} format=onnx opset={args.opset} "
            f"imgsz={args.cls_imgsz} dynamic=False simplify=True"
        )
        run_cmd(cls_export_cmd, logger)

        cls_onnx = Path(cls_best).with_suffix(".onnx")
        if cls_onnx.exists():
            run_cmd(f"cp {cls_onnx} {os.path.join(args.outdir, 'best_cls.onnx')}", logger)

    logger.info("Two-stage training complete.")
    if args.export:
        logger.info(f"Exports staged under: {args.outdir}")


if __name__ == "__main__":
    main()

