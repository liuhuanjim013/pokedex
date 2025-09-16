#!/usr/bin/env python3
"""
MaixCam Two-Stage Test (Detector + Classifier)

Requirements on device:
- Copy to /root (or run from project dir mounted)
- Models in /root/models/:
  - pokemon_det1_int8.cvimodel, pokemon_det1.mud
  - pokemon_cls1025_int8.cvimodel, pokemon_cls1025.mud
  - classes.txt (1025 lines)

Controls:
- Press Ctrl+C to exit
"""

import math
import time

try:
    from maix import camera, display, image, nn
except Exception as e:
    raise SystemExit(f"This script must run on MaixCam. Import error: {e}")


# Paths on device
DET_MUD = "/root/models/pokemon_det1.mud"
CLS_MUD = "/root/models/pokemon_cls1025.mud"
CLASSES_TXT = "/root/models/classes.txt"


# Runtime params
DET_SIZE = 256
CLS_SIZE = 224
DET_CONF = 0.35
DET_IOU = 0.45
CROP_PAD = 0.15
AGREE_N = 3  # temporal smoothing: agree in last N frames


def load_classes(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = [l.strip() for l in f if l.strip()]
        return names
    except Exception:
        return [f"id_{i}" for i in range(1025)]


def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]


def pad_and_clip(x, y, w, h, pad, W, H):
    w2 = w * (1.0 + pad)
    h2 = h * (1.0 + pad)
    x1 = int(max(x - w2 / 2, 0))
    y1 = int(max(y - h2 / 2, 0))
    x2 = int(min(x + w2 / 2, W - 1))
    y2 = int(min(y + h2 / 2, H - 1))
    if x2 <= x1 or y2 <= y1:
        # fallback center square
        s = min(W, H)
        x1 = (W - s) // 2
        y1 = (H - s) // 2
        x2 = x1 + s - 1
        y2 = y1 + s - 1
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def get_best_box_yolo(detector, frame_det):
    # Try high-level YOLO API
    try:
        boxes = detector.detect(frame_det, conf=DET_CONF, iou=DET_IOU)
        # boxes may be list of dicts/tuples; pick highest confidence
        best = None
        best_score = -1.0
        for b in boxes or []:
            # Common fields: x,y,w,h,score/conf
            x = getattr(b, "x", None) or b.get("x", None)
            y = getattr(b, "y", None) or b.get("y", None)
            w = getattr(b, "w", None) or b.get("w", None)
            h = getattr(b, "h", None) or b.get("h", None)
            s = (getattr(b, "score", None) or getattr(b, "conf", None)
                 or b.get("score", None) or b.get("conf", None) or 0.0)
            if x is None or y is None or w is None or h is None:
                continue
            if s > best_score:
                best = (float(x + w / 2), float(y + h / 2), float(w), float(h))
                best_score = float(s)
        return best, best_score
    except Exception:
        return None, 0.0


def main():
    names = load_classes(CLASSES_TXT)

    cam = camera.Camera(640, 480)
    disp = display.Display()

    # Load detector
    det = None
    try:
        det = nn.YOLO(DET_MUD)
    except Exception as e:
        raise SystemExit(f"Failed to load detector: {e}")

    # Load classifier (try high-level, fallback to generic)
    cls = None
    cls_is_generic = False
    try:
        cls = nn.Classifier(CLS_MUD)
    except Exception:
        try:
            cls = nn.NN(CLS_MUD)
            cls_is_generic = True
        except Exception as e:
            raise SystemExit(f"Failed to load classifier: {e}")

    recent = []  # temporal smoothing buffer of class ids

    while True:
        frame = cam.read()
        W, H = frame.width(), frame.height()

        # Detector inference (resize to DET_SIZE if needed)
        frame_det = frame.resize(DET_SIZE, DET_SIZE)
        best_box, best_score = get_best_box_yolo(det, frame_det)

        if best_box is None:
            # center crop fallback
            s = int(min(W, H) * 0.6)
            x = (W - s) // 2
            y = (H - s) // 2
            crop = frame.crop(x, y, s, s).resize(CLS_SIZE, CLS_SIZE)
        else:
            # map box back to original frame scale
            cx, cy, bw, bh = best_box
            sx = W / float(DET_SIZE)
            sy = H / float(DET_SIZE)
            cx *= sx
            cy *= sy
            bw *= sx
            bh *= sy
            x, y, w, h = pad_and_clip(cx, cy, bw, bh, CROP_PAD, W, H)
            crop = frame.crop(x, y, w, h).resize(CLS_SIZE, CLS_SIZE)

        # Classifier inference
        top1_id = 0
        top1_p = 0.0
        top1_name = "unknown"
        try:
            if not cls_is_generic:
                results = cls.classify(crop)  # expect list of (id, prob) or objects
                # normalize to (id, prob)
                parsed = []
                for r in results or []:
                    cid = getattr(r, "id", None)
                    if cid is None:
                        cid = r[0] if isinstance(r, (list, tuple)) and r else 0
                    prob = (getattr(r, "prob", None) or getattr(r, "confidence", None)
                            or (r[1] if isinstance(r, (list, tuple)) and len(r) > 1 else 0.0))
                    parsed.append((int(cid), float(prob)))
                if parsed:
                    top1_id, top1_p = max(parsed, key=lambda x: x[1])
            else:
                # Generic NN: forward_image -> logits
                logits = cls.forward_image(crop)
                # try to_float_list()
                try:
                    logits = logits.to_float_list()
                except Exception:
                    # fallback: to_list()
                    logits = logits.to_list()
                probs = softmax([float(v) for v in logits])
                top1_id = max(range(len(probs)), key=lambda i: probs[i])
                top1_p = probs[top1_id]
        except Exception:
            # robust fallback
            top1_id, top1_p = (0, 0.0)

        # Temporal smoothing
        recent.append(top1_id)
        if len(recent) > AGREE_N:
            recent.pop(0)
        stable = recent.count(recent[-1]) >= AGREE_N

        if 0 <= top1_id < len(names):
            top1_name = names[top1_id]

        # Annotate and display
        label = f"{top1_name} {top1_p*100:.1f}%{' 🔒' if stable else ''}"
        y_text = 4
        frame.draw_string(4, y_text, label, scale=1.0, color=(255, 255, 0))
        if best_box is not None:
            # draw bbox (projected to original)
            cx, cy, bw, bh = best_box
            sx = W / float(DET_SIZE)
            sy = H / float(DET_SIZE)
            cx *= sx; cy *= sy; bw *= sx; bh *= sy
            x1 = int(cx - bw / 2); y1 = int(cy - bh / 2)
            frame.draw_rect(x1, y1, int(bw), int(bh), color=(0, 255, 0))
        disp.show(frame)

        # slight delay to keep UI responsive
        time.sleep(0.01)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

