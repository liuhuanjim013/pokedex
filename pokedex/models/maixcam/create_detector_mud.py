#!/usr/bin/env python3
"""
Create detector MUD and labels files for MaixCam runtime.

This writes two files into the specified output directory (defaults to this folder):
 - pokemon_det1.mud
 - pokemon_det1_labels.txt

The MUD references device-side absolute paths so you can copy both files
along with the cvimodel to /root/models on the device and run immediately.

Usage examples:
  python pokedex/models/maixcam/create_detector_mud.py
  python pokedex/models/maixcam/create_detector_mud.py \
      --cvimodel-device-path /root/models/pokemon_det1_int8.cvimodel \
      --labels-device-path /root/models/pokemon_det1_labels.txt \
      --out-dir pokedex/models/maixcam
"""

import argparse
import os
from pathlib import Path


DEFAULT_CVIMODEL = "/root/models/pokemon_det1_int8.cvimodel"
DEFAULT_LABELS = "/root/models/pokemon_det1_labels.txt"


MUD_TEMPLATE = """type: yolo11
cvimodel: {cvimodel}
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
  file: {labels}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MaixCam detector MUD and labels")
    parser.add_argument("--cvimodel-device-path", type=str, default=DEFAULT_CVIMODEL,
                        help="Absolute device path to detector cvimodel inside MUD")
    parser.add_argument("--labels-device-path", type=str, default=DEFAULT_LABELS,
                        help="Absolute device path to labels txt inside MUD")
    parser.add_argument("--label", type=str, default="pokemon", help="Single class label text")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent),
                        help="Directory to write pokemon_det1.mud and pokemon_det1_labels.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mud_path = out_dir / "pokemon_det1.mud"
    labels_path = out_dir / "pokemon_det1_labels.txt"

    mud_text = MUD_TEMPLATE.format(cvimodel=args.cvimodel_device_path, labels=args.labels_device_path)
    mud_path.write_text(mud_text, encoding="utf-8")
    labels_path.write_text(args.label + "\n", encoding="utf-8")

    print(f"Wrote MUD     -> {mud_path}")
    print(f"Wrote labels  -> {labels_path}")
    print("Done. Copy these along with the cvimodel to /root/models on the device.")


if __name__ == "__main__":
    main()

