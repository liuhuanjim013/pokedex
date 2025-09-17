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
DET_MUD = "/root/models/pokemon_det1_device.mud"
CLS_MUD = "/root/models/pokemon_cls1025.mud"
CLASSES_TXT = "/root/models/classes.txt"


# Runtime params
DET_SIZE = 256
CLS_SIZE = 224
DET_CONF = 0.35
DET_IOU = 0.45
CROP_PAD = 0.15
AGREE_N = 3  # temporal smoothing: agree in last N frames
MEAN = (0.0, 0.0, 0.0)
SCALE = (1.0/255.0, 1.0/255.0, 1.0/255.0)
COL_YELLOW = image.Color(255, 255, 0)
COL_GREEN = image.Color(0, 255, 0)


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

    # Load detector (generic NN; some firmware lacks nn.YOLO)
    try:
        if hasattr(nn, 'YOLO11'):
            det = nn.YOLO11(DET_MUD)
        elif hasattr(nn, 'YOLO'):
            det = nn.YOLO(DET_MUD)
        else:
            det = nn.NN(DET_MUD)
    except Exception as e:
        # Try direct cvimodel load as probe
        try:
            if hasattr(nn, 'YOLO11'):
                det = nn.YOLO11("/root/models/pokemon_det1_int8.cvimodel")
            elif hasattr(nn, 'YOLO'):
                det = nn.YOLO("/root/models/pokemon_det1_int8.cvimodel")
            else:
                det = nn.NN("/root/models/pokemon_det1_int8.cvimodel")
            print("ℹ️ Loaded detector directly from cvimodel (MUD parse failed)")
        except Exception as e2:
            print(f"⚠️ Detector load failed ({e}); direct load failed ({e2}). Using center-crop fallback.")
            det = None

    # Load classifier (try high-level, fallback to generic, then direct cvimodel)
    cls = None
    cls_is_generic = False
    try:
        cls = nn.Classifier(CLS_MUD)
    except Exception:
        try:
            cls = nn.NN(CLS_MUD)
            cls_is_generic = True
        except Exception:
            # direct cvimodel
            try:
                cls = nn.NN("/root/models/pokemon_cls1025_int8.cvimodel")
                cls_is_generic = True
                print("ℹ️ Loaded classifier directly from cvimodel (MUD parse failed)")
            except Exception as e:
                raise SystemExit(f"Failed to load classifier: {e}")

    recent = []  # temporal smoothing buffer of class ids

    while True:
        frame = cam.read()
        W, H = frame.width(), frame.height()

        # Detector inference (if loaded)
        best_box, best_score = None, 0.0
        if det is not None:
            frame_det = frame.resize(DET_SIZE, DET_SIZE)
            # Try YOLO wrapper first
            try:
                if hasattr(det, 'detect'):
                    best_box, best_score = get_best_box_yolo(det, frame_det)
            except Exception:
                best_box, best_score = None, 0.0
            # Generic forward decode
            if best_box is None:
                try:
                    out = det.forward_image(frame_det, mean=MEAN, scale=SCALE)
                    # Find a tensor whose length is multiple of 5 (cx,cy,w,h,score)*P
                    vals = None
                    chosen = None
                    for k in out.keys():
                        t = out[k]
                        try:
                            arr = t.to_float_list()
                        except Exception:
                            continue
                        if isinstance(arr, list) and len(arr) >= 5 and (len(arr) % 5) == 0:
                            vals = arr
                            chosen = k
                            break
                    if vals is None:
                        raise RuntimeError('detector output invalid')
                    ch = 5
                    P = len(vals) // ch
                    best_i = 0
                    best_s = -1.0
                    for i in range(P):
                        s = vals[i*ch + 4]
                        s = 1.0 / (1.0 + math.exp(-max(min(s, 500.0), -500.0)))
                        if s > best_s:
                            best_s = s
                            best_i = i
                    cx = vals[best_i*ch + 0]
                    cy = vals[best_i*ch + 1]
                    bw = vals[best_i*ch + 2]
                    bh = vals[best_i*ch + 3]
                    best_box, best_score = (float(cx), float(cy), float(bw), float(bh)), float(best_s)
                except Exception:
                    best_box, best_score = None, 0.0

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
                # Generic NN: forward_image -> Tensors (use key)
                outs = cls.forward_image(crop, mean=MEAN, scale=SCALE)
                name = None
                keys = outs.keys()
                name = keys[0] if keys else None
                if name is None:
                    raise RuntimeError('classifier output empty')
                vec = outs[name]
                # to_float_list for Tensor
                arr = vec.to_float_list() if hasattr(vec, 'to_float_list') else []
                if not arr:
                    raise RuntimeError('classifier tensor conversion failed')
                if 'softmax' in name.lower():
                    probs = [float(v) for v in arr]
                else:
                    probs = softmax([float(v) for v in arr])
                top1_id = int(max(range(len(probs)), key=lambda i: probs[i]))
                top1_p = float(probs[top1_id])
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
        frame.draw_string(4, y_text, label, COL_YELLOW, 1.0)
        if best_box is not None:
            # draw bbox (projected to original)
            cx, cy, bw, bh = best_box
            sx = W / float(DET_SIZE)
            sy = H / float(DET_SIZE)
            cx *= sx; cy *= sy; bw *= sx; bh *= sy
            x1 = int(cx - bw / 2); y1 = int(cy - bh / 2)
            frame.draw_rect(x1, y1, int(bw), int(bh), COL_GREEN)
        disp.show(frame)

        # slight delay to keep UI responsive
        time.sleep(0.01)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

