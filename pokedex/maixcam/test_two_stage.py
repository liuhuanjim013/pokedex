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
import os
import time

try:
    from maix import camera, display, image, nn
except Exception as e:
    raise SystemExit(f"This script must run on MaixCam. Import error: {e}")


# Paths on device
DET_MODEL = "/root/models/pokemon_det1_int8.cvimodel"
CLS_MODEL = "/root/models/pokemon_cls1025_int8.cvimodel"
CLASSES_TXT = "/root/models/classes.txt"


# Runtime params
DET_SIZE = 256
CLS_SIZE = 224
DET_CONF = 0.01
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


def clamp_rect(x, y, w, h, W, H):
    x = int(max(0, min(x, W - 1)))
    y = int(max(0, min(y, H - 1)))
    w = int(max(1, min(w, W - x)))
    h = int(max(1, min(h, H - y)))
    return x, y, w, h


def get_best_box_yolo(detector, frame_img):
    # Try high-level YOLO API
    try:
        boxes = detector.detect(frame_img, conf=DET_CONF, iou=DET_IOU, conf_thres=DET_CONF, iou_thres=DET_IOU)
        try:
            n = len(boxes) if boxes is not None else 0
            print(f"[det] wrapper boxes: {n}")
            if n:
                # Log up to first 3 boxes
                log_k = min(3, n)
                for i in range(log_k):
                    b = boxes[i]
                    s = (getattr(b, 'score', None) or getattr(b, 'conf', None)
                         or (b.get('score', None) if hasattr(b, 'get') else None)
                         or (b.get('conf', None) if hasattr(b, 'get') else None))
                    x = getattr(b, 'x', None) or (b.get('x', None) if hasattr(b, 'get') else None)
                    y = getattr(b, 'y', None) or (b.get('y', None) if hasattr(b, 'get') else None)
                    w = getattr(b, 'w', None) or (b.get('w', None) if hasattr(b, 'get') else None)
                    h = getattr(b, 'h', None) or (b.get('h', None) if hasattr(b, 'get') else None)
                    print(f"[det] box[{i}]: x={x} y={y} w={w} h={h} score={s}")
        except Exception:
            pass

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

    # Load detector via NN only (YOLO wrapper unreliable on this firmware)
    det = None
    try:
        det = nn.NN(DET_MODEL)
        print("ℹ️ Loaded detector via NN(cvimodel)")
    except Exception as e2:
        print(f"⚠️ Detector cvimodel load failed: {e2}. Using center-crop fallback.")
        det = None

    # Load classifier using working backend only (nn.NN with cvimodel)
    cls = None
    cls_is_generic = True
    try:
        cls = nn.NN(CLS_MODEL)
        cls_is_generic = True
        print("ℹ️ Loaded classifier via NN(cvimodel)")
    except Exception as e:
        raise SystemExit(f"Failed to load classifier: {e}")

    recent = []  # temporal smoothing buffer of class ids

    while True:
        frame = cam.read()
        W, H = frame.width(), frame.height()
        rect = None

        # Detector inference (if loaded) — manual decode only
        best_box, best_score = None, 0.0
        if det is not None:
            frame_det = frame.resize(DET_SIZE, DET_SIZE)
            try:
                out = det.forward_image(frame_det, mean=MEAN, scale=SCALE)
                # Debug: list tensor keys and first tensor length/head
                try:
                    keys = list(out.keys())
                    print(f"[det] tensor keys: {keys}")
                    if keys:
                        t0 = out[keys[0]]
                        if hasattr(t0, 'to_float_list'):
                            arr0 = t0.to_float_list()
                            print(f"[det] first tensor len={len(arr0)} head={arr0[:10] if len(arr0)>0 else []}")
                except Exception:
                    pass
                # Channel-first layout: [cx_all, cy_all, w_all, h_all, score_all] concatenated
                vals = None
                for k in out.keys():
                    t = out[k]
                    try:
                        arr = t.to_float_list()
                    except Exception:
                        continue
                    if isinstance(arr, list) and len(arr) >= 5 and (len(arr) % 5) == 0:
                        vals = arr
                        break
                if vals is None:
                    raise RuntimeError('detector output invalid')
                P = len(vals) // 5
                cx_all = vals[0:P]
                cy_all = vals[P:2*P]
                w_all  = vals[2*P:3*P]
                h_all  = vals[3*P:4*P]
                s_raw  = vals[4*P:5*P]
                # Sigmoid scores
                s_all = []
                best_i = 0
                best_s = -1.0
                for i in range(P):
                    s = 1.0 / (1.0 + math.exp(-max(min(s_raw[i], 500.0), -500.0)))
                    s_all.append(s)
                    if s > best_s:
                        best_s = s
                        best_i = i
                # Logging: score stats and small top-3
                try:
                    min_s = min(s_all) if s_all else 0.0
                    max_s = max(s_all) if s_all else 0.0
                    top3 = sorted(((s_all[i], i) for i in range(P)), reverse=True)[:3]
                    print(f"[det] best_i={best_i}/{P} score={best_s:.3f} min={min_s:.3f} max={max_s:.3f} top3={[ (i, round(s,3)) for s,i in top3 ]}")
                except Exception:
                    pass
                cx = cx_all[best_i]
                cy = cy_all[best_i]
                bw = w_all[best_i]
                bh = h_all[best_i]
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
            # map detected box back to original frame scale (no padding for drawn rect)
            cx, cy, bw, bh = best_box
            sx = W / float(det_scale_w)
            sy = H / float(det_scale_h)
            cx *= sx
            cy *= sy
            bw *= sx
            bh *= sy
            # rect for drawing (no pad) with clamping
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            rect = clamp_rect(x1, y1, int(bw), int(bh), W, H)
            # crop for classifier uses padded & clipped box
            x, y, w, h = pad_and_clip(cx, cy, bw, bh, CROP_PAD, W, H)
            crop = frame.crop(x, y, w, h).resize(CLS_SIZE, CLS_SIZE)
        # Debug: print final rect on original frame
        if rect is not None:
            print(f"[rect] x={rect[0]} y={rect[1]} w={rect[2]} h={rect[3]} (W={W}, H={H})")

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

