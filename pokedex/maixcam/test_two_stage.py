#!/usr/bin/env python3
"""
MaixCam Two-Stage Test (Detector + Classifier)

Requirements on device:
- Copy to /root (or run from project dir mounted)
- Models in /root/models/:
  - pokemon_det1_int8.cvimodel
  - pokemon_cls1025_int8.cvimodel
  - classes.txt (1025 lines)
  - (MUD files optional; not used)

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
DET_MODEL = "/root/models/pokemon_det1_int8.cvimodel"
DET_MUD = "/root/models/pokemon_det1.mud"
CLS_MODEL = "/root/models/pokemon_cls1025_int8.cvimodel"
CLS_MUD = "/root/models/pokemon_cls1025.mud"
CLASSES_TXT = "/root/models/classes.txt"


# Runtime params
DET_SIZE = 256
CLS_SIZE = 224
DET_CONF = 0.15
DET_IOU = 0.45
CROP_PAD = 0.15
AGREE_N = 3  # temporal smoothing: agree in last N frames
MEAN = (0.0, 0.0, 0.0)
SCALE = (1.0/255.0, 1.0/255.0, 1.0/255.0)
COL_YELLOW = image.Color(255, 255, 0)
COL_GREEN = image.Color(0, 255, 0)
COL_BLACK = image.Color(0, 0, 0)
COL_RED = image.Color(255, 0, 0)
TEXT_SCALE = 1.6


def load_classes(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = [l.strip() for l in f if l.strip()]
        return names
    except Exception:
        return [f"id_{i}" for i in range(1025)]


def sigmoid(x: float) -> float:
    x = max(min(x, 500.0), -500.0)
    return 1.0 / (1.0 + math.exp(-x))


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


def get_best_box_yolo(detector, frame_img):
    # Try high-level YOLO API
    try:
        boxes = detector.detect(frame_img, conf=DET_CONF, iou=DET_IOU)

        def get_val(obj, *keys):
            for k in keys:
                v = getattr(obj, k, None)
                if v is not None:
                    return v
                try:
                    v = obj.get(k, None)  # dict-like
                    if v is not None:
                        return v
                except Exception:
                    pass
            return None

        def parse_box(b):
            # Try explicit center-width-height
            cx = get_val(b, 'cx', 'center_x', 'xc')
            cy = get_val(b, 'cy', 'center_y', 'yc')
            bw = get_val(b, 'w', 'width')
            bh = get_val(b, 'h', 'height')
            if cx is not None and cy is not None and bw is not None and bh is not None:
                return float(cx), float(cy), float(bw), float(bh)

            # Try combined bbox field
            bbox = get_val(b, 'bbox', 'box', 'rect', 'xywh', 'tlbr')
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                bx0, by0, bw0, bh0 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                # Heuristic: if third and fourth look like bottom-right, convert to w,h
                if bw0 > bx0 and bh0 > by0:
                    w2 = max(0.0, bw0 - bx0)
                    h2 = max(0.0, bh0 - by0)
                    return float(bx0 + w2 / 2.0), float(by0 + h2 / 2.0), w2, h2
                else:
                    return float(bx0 + bw0 / 2.0), float(by0 + bh0 / 2.0), float(bw0), float(bh0)

            # Try top-left + size
            x = get_val(b, 'x', 'x1', 'left')
            y = get_val(b, 'y', 'y1', 'top')
            w = get_val(b, 'w', 'width')
            h = get_val(b, 'h', 'height')
            if x is not None and y is not None and w is not None and h is not None:
                return float(x + w / 2.0), float(y + h / 2.0), float(w), float(h)

            # Try corners
            x1 = get_val(b, 'x1', 'left')
            y1 = get_val(b, 'y1', 'top')
            x2 = get_val(b, 'x2', 'right')
            y2 = get_val(b, 'y2', 'bottom')
            if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                x1 = float(x1); y1 = float(y1); x2 = float(x2); y2 = float(y2)
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                return float(x1 + bw / 2.0), float(y1 + bh / 2.0), bw, bh

            # Tuple/list fallback
            if isinstance(b, (list, tuple)):
                if len(b) >= 4:
                    x, y, w, h = b[0], b[1], b[2], b[3]
                    return float(x + w / 2.0), float(y + h / 2.0), float(w), float(h)
            return None

        # boxes may be list of dicts/tuples/objects; pick highest confidence
        best = None
        best_score = -1.0
        for b in boxes or []:
            parsed = parse_box(b)
            if parsed is None:
                continue
            score = get_val(b, 'score', 'conf', 'prob', 'confidence')
            try:
                s = float(score) if score is not None else 0.0
            except Exception:
                s = 0.0
            if s > best_score:
                best = parsed
                best_score = s
        if best is not None:
            cx, cy, bw, bh = best
            # If values appear normalized, scale to pixel coordinates of the image used for detect
            if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 1.5:
                dw = float(frame_img.width())
                dh = float(frame_img.height())
                cx *= dw; cy *= dh; bw *= dw; bh *= dh
                best = (cx, cy, bw, bh)
        return best, best_score
    except Exception:
        return None, 0.0


def main():
    names = load_classes(CLASSES_TXT)

    cam = camera.Camera(640, 480)
    disp = display.Display()

    # Load detector with robust fallback chain (prefer MUD)
    det = None
    det_errors = []
    def _try(name, fn):
        try:
            m = fn()
            print(f"ℹ️ Loaded detector via {name}")
            return m
        except Exception as _e:
            det_errors.append((name, str(_e)))
            return None

    # Prefer MUD if wrappers exist
    if hasattr(nn, 'YOLO11'):
        det = det or _try("YOLO11(mud)", lambda: nn.YOLO11(DET_MUD))
        det = det or _try("YOLO11(cvimodel)", lambda: nn.YOLO11(DET_MODEL))
    if hasattr(nn, 'YOLO'):
        det = det or _try("YOLO(mud)", lambda: nn.YOLO(DET_MUD))
        det = det or _try("YOLO(cvimodel)", lambda: nn.YOLO(DET_MODEL))
    # Generic backends
    det = det or _try("NN(cvimodel)", lambda: nn.NN(DET_MODEL))
    if hasattr(nn, 'Infer'):
        det = det or _try("Infer(cvimodel)", lambda: nn.Infer(model=DET_MODEL))
    if det is None:
        print("⚠️ Detector load failed across all backends. Using center-crop fallback.")
        for n, err in det_errors[-3:]:
            print(f"   - {n} → {err}")

    # Load classifier (prefer MUD, then cvimodel)
    cls = None
    cls_is_generic = True
    try:
        if hasattr(nn, 'Classifier'):
            try:
                # Try MUD first if available
                try:
                    cls = nn.Classifier(CLS_MUD)
                    cls_is_generic = False
                except Exception:
                    cls = nn.Classifier(CLS_MODEL)
                    cls_is_generic = False
            except Exception:
                cls = nn.NN(CLS_MODEL)
                cls_is_generic = True
        else:
            cls = nn.NN(CLS_MODEL)
            cls_is_generic = True
        print("ℹ️ Loaded classifier")
    except Exception as e:
        raise SystemExit(f"Failed to load classifier: {e}")

    recent = []  # temporal smoothing buffer of class ids

    while True:
        frame = cam.read()
        W, H = frame.width(), frame.height()
        rect = None

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
            rect = (x, y, s, s)
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
            rect = (x, y, w, h)

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
                # Use sigmoid per-class (YOLO-style multi-label head)
                probs = [sigmoid(float(v)) for v in arr]
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
        label = f"{top1_name} {top1_p*100:.1f}%{' [stable]' if stable else ''}"
        x_text = 6
        y_text = 6
        # Draw text outline (shadow) for better visibility
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)):
            frame.draw_string(x_text + dx, y_text + dy, label, COL_BLACK, TEXT_SCALE)
        frame.draw_string(x_text, y_text, label, COL_YELLOW, TEXT_SCALE)
        if rect is not None:
            color = COL_GREEN if best_box is not None and best_score >= DET_CONF else COL_RED
            frame.draw_rect(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]), color)
        disp.show(frame)

        # slight delay to keep UI responsive
        time.sleep(0.01)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

