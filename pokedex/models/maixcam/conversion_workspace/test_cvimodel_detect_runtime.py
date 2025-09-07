#!/usr/bin/env python3
import os, sys, json, subprocess, shutil, glob, re
import numpy as np
import cv2
from pathlib import Path

MODEL_PATH = "maixcam_deployment/pokemon_classifier_int8.cvimodel"
MUD_PATH   = "maixcam_deployment/pokemon_classifier.mud"
IMAGE_PATH = "images/0001_001.jpg"
OUT_DIR    = "runner_out"

NUM_CLASSES = 1025
C = 4 + NUM_CLASSES
P = 1344

# ---------- helpers ----------
def read_mud_mean_scale(mud_path):
    mean, scale = [0,0,0], [1,1,1]
    if not os.path.exists(mud_path):
        return mean, scale
    try:
        s = open(mud_path, "rb").read().decode("utf-8", errors="ignore")
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            meta = json.loads(s[start:end+1])
            m  = meta.get("mean") or meta.get("MEAN") or meta.get("preprocess", {}).get("mean")
            sc = meta.get("scale") or meta.get("SCALE") or meta.get("preprocess", {}).get("scale")
            if isinstance(m, (list, tuple)) and len(m) >= 3: mean  = [float(m[0]), float(m[1]), float(m[2])]
            if isinstance(sc, (list, tuple)) and len(sc) >= 3: scale = [float(sc[0]), float(sc[1]), float(sc[2])]
    except Exception:
        pass
    return mean, scale

def preprocess_variants(img_path, mean, scale, size=(256,256)):
    """
    Produce several plausible input variants:
      - NCHW float32 (RGB)
      - NHWC float32 (RGB)
      - NCHW uint8  (RGB)
      - NHWC uint8  (RGB)
      - NCHW/NHWC uint8 (BGR) just in case
    """
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR)

    # RGB float32 (with mean/scale)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb_fs = rgb.copy()
    for c in range(3):
        rgb_fs[..., c] = (rgb_fs[..., c] - mean[c]) * scale[c]

    # Variants
    variants = []
    # 1) NCHW float32 RGB
    variants.append(("input", "nchw_f32_rgb", np.transpose(rgb_fs, (2,0,1))[None, ...]))
    # 2) NHWC float32 RGB
    variants.append(("input", "nhwc_f32_rgb", rgb_fs[None, ...]))
    # 3) NCHW uint8 RGB (0..255)
    variants.append(("input", "nchw_u8_rgb", np.transpose(rgb.astype(np.uint8), (2,0,1))[None, ...]))
    # 4) NHWC uint8 RGB
    variants.append(("input", "nhwc_u8_rgb", rgb.astype(np.uint8)[None, ...]))
    # 5) NCHW uint8 BGR
    variants.append(("input", "nchw_u8_bgr", np.transpose(bgr.astype(np.uint8), (2,0,1))[None, ...]))
    # 6) NHWC uint8 BGR
    variants.append(("input", "nhwc_u8_bgr", bgr.astype(np.uint8)[None, ...]))

    return variants

def find_runners():
    cmds = []
    for name in ("model_runner", "model_runner.py", "tpuc-run", "tpuc_model_run", "run_cmodel.py"):
        p = shutil.which(name)
        if p: cmds.append([p])
    # explicit python for model_runner.py path
    extra = [
        "/usr/local/bin/model_runner.py",
        "/usr/local/lib/python3.10/dist-packages/tpu_mlir/python/model_runner.py",
        "/usr/local/lib/python3.8/dist-packages/tpu_mlir/python/model_runner.py",
        "/opt/tpu-mlir/python/model_runner.py",
        "/workspace/tpu-mlir/python/model_runner.py",
    ]
    for p in extra:
        if os.path.exists(p):
            cmds.append(["python3", p])
    return cmds

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def try_runner(model_cmd, model_path, in_npz_path, out_dir):
    """
    Try multiple flag styles commonly seen across TPU-MLIR toolchains.
    Returns list of produced files if successful, else raises.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "out.npz")  # file, not directory

    flag_matrix = [
        ["--model", model_path, "--input", in_npz_path,
         "--output", out_file, "--dump_all_tensors"],  # add this to see all outputs
    ]

    errors = []
    for flags in flag_matrix:
        cmd = model_cmd + flags
        print(f"➡️  Trying: {' '.join(cmd)}")
        p = run(cmd)
        if p.returncode == 0:
            # look exactly for the file we asked for
            if os.path.isfile(out_file):
                print(f"✅ Runner wrote: {out_file}")
                return [out_file]
            else:
                # some builds dump to default name; sweep for *.npz/ *.npy nearby
                outs = sorted(glob.glob(os.path.join(out_dir, "*.npz")) +
                              glob.glob(os.path.join(out_dir, "*.npy")))
                if outs:
                    print(f"ℹ️  Found outputs despite missing {out_file}: {outs}")
                    return outs
                print("ℹ️  Return code 0 but no files; trying next flags …")
        else:
            print("❌ Runner failed. STDERR:")
            print(p.stderr.strip() or "(empty)")
            errors.append((" ".join(cmd), p.stderr.strip()))

    print("\n--- Attempts exhausted. Last errors ---")
    for c, e in errors[-4:]:
        print("CMD:", c)
        print("ERR:", e or "(empty)")
        print("-" * 60)
    raise RuntimeError("No runner flag combo produced outputs.")

def load_outputs(npz_path):
    outs = {}
    with np.load(npz_path) as d:
        for k in d.files:
            outs[k] = d[k]
    print("🧾 NPZ keys & shapes:")
    for k, v in outs.items():
        print(f"  - {k}: {v.shape} {v.dtype}")
    return outs

def select_heads(outs):
    # Normalize shapes to (C,P) candidates
    def to_CP(a):
        a = np.asarray(a)
        while a.ndim > 2 and 1 in a.shape:
            ax = [i for i, d in enumerate(a.shape) if d == 1][0]
            a = np.squeeze(a, axis=ax)
        if a.ndim == 2:
            if a.shape[0] < a.shape[1]:
                return a  # (C,P)
            else:
                return a.T  # (P,C) -> (C,P)
        if a.ndim == 3:
            if a.shape[0] < a.shape[1] and a.shape[0] < a.shape[2]:
                return a.reshape(a.shape[0], -1)       # (C,H,W)->(C,P)
            if a.shape[2] < a.shape[0] and a.shape[2] < a.shape[1]:
                return np.transpose(a, (2,0,1)).reshape(a.shape[2], -1)  # (H,W,C)->(C,P)
        return None

    bbox_candidates, cls_candidates = [], []
    for k, v in outs.items():
        cp = to_CP(v)
        if cp is None: 
            continue
        C, P = cp.shape
        if C == 4:
            bbox_candidates.append((k, cp))
        elif C == 1025:  # your classes.txt size
            cls_candidates.append((k, cp))

    return bbox_candidates, cls_candidates

def pick_packed_head(outs, num_classes=1024):
    # look for any (C,P,1) or (1,C,P,1) with C == 4+1+num_classes
    wantC = 4 + 1 + num_classes
    for k, v in outs.items():
        a = np.asarray(v)
        # squeeze singletons
        while a.ndim > 2 and 1 in a.shape:
            a = np.squeeze(a, axis=[i for i,d in enumerate(a.shape) if d==1][0])
        if a.ndim == 3 and a.shape[0] == wantC:
            # (C,H,W) -> (C,P)
            cp = a.reshape(wantC, -1)
            return k, cp
        if a.ndim == 3 and a.shape[2] == wantC:
            # (H,W,C) -> (C,P)
            cp = np.transpose(a, (2,0,1)).reshape(wantC, -1)
            return k, cp
        if a.ndim == 2 and a.shape[0] == wantC:
            return k, a
        if a.ndim == 2 and a.shape[1] == wantC:
            return k, a.T
    return None, None

def analyze_detection(bbox_cp, cls_cp=None, class_names_path="maixcam_deployment/classes.txt"):
    bbox = bbox_cp  # (4,P)
    P = bbox.shape[1]

    if cls_cp is not None:
        # (1025,P) -> pick best by max logit
        max_logits = cls_cp.max(axis=0)
        pos = int(np.argmax(max_logits))
        score = float(max_logits[pos])
        logits = cls_cp[:, pos]
        ex = np.exp(logits - np.max(logits))
        probs = ex / np.sum(ex)
        top5 = np.argsort(probs)[-5:][::-1]
        bbox_best = bbox[:, pos]
        print(f"🏆 Best pos={pos} score={score:.6f}")
        print(f"🧱 BBox [cx,cy,w,h] = [{bbox_best[0]:.3f},{bbox_best[1]:.3f},{bbox_best[2]:.3f},{bbox_best[3]:.3f}]")
        print("🥇 Top-5 classes (id:prob):", [(int(i), float(probs[i])) for i in top5])

        if os.path.exists(class_names_path):
            names = [l.strip() for l in open(class_names_path) if l.strip()]
            print("🏷️  Names:", [names[i] if i < len(names) else f"id_{i}" for i in top5])
    else:
        # No class head available – report bbox stats and a few highest-area boxes
        areas = (bbox[2, :] * np.maximum(bbox[3, :], 1e-6))
        idx = np.argsort(areas)[-5:][::-1]
        print("ℹ️  No class logits found. Showing top boxes by area:")
        for rank, i in enumerate(idx, 1):
            print(f"  {rank}. pos={int(i)} bbox=[{bbox[0,i]:.3f},{bbox[1,i]:.3f},{bbox[2,i]:.3f},{bbox[3,i]:.3f}] area={areas[i]:.3f}")

def analyze_packed_head(packed_cp, num_classes=1024, class_names_path="maixcam_deployment/classes.txt"):
    """Analyze packed YOLO head: bbox(4) + objectness(1) + classes(num_classes)"""
    bbox = packed_cp[0:4, :]           # (4, P)
    obj  = packed_cp[4:5, :]           # (1, P)
    cls  = packed_cp[5:, :]            # (num_classes, P)

    # softmax over classes (optional; depends on training head)
    logits = cls
    ex = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    probs = ex / np.sum(ex, axis=0, keepdims=True)   # (num_classes, P)

    # YOLO-style score = obj * class prob
    best_cls = np.argmax(probs, axis=0)              # (P,)
    best_cls_prob = probs[best_cls, np.arange(probs.shape[1])]
    score = (obj.ravel()) * best_cls_prob            # (P,)

    pos = int(np.argmax(score))
    print(f"🏆 Best pos={pos} score={float(score[pos]):.6f}")
    print(f"🧱 BBox [cx,cy,w,h] = {bbox[:, pos].tolist()}")

    # top-5 classes at best position
    top5 = np.argsort(probs[:, pos])[-5:][::-1]
    print("🥇 Top-5 classes (id:prob):", [(int(i), float(probs[i, pos])) for i in top5])

    # optional names
    if os.path.exists(class_names_path):
        names = [l.strip() for l in open(class_names_path) if l.strip()]
        print("🏷️ Names:", [names[i] if i < len(names) else f"id_{i}" for i in top5])

def main():
    print("🧪 CVIModel Detect Runtime Test")
    for p, lab in [(MODEL_PATH,"model"), (IMAGE_PATH,"image")]:
        if not os.path.exists(p):
            print(f"❌ {lab} not found: {p}")
            sys.exit(1)

    mean, scale = read_mud_mean_scale(MUD_PATH)
    print(f"📋 Preprocess mean={mean} scale={scale}")

    # Show model_runner help to understand CLI options
    print("🔍 Model runner help:")
    try:
        result = subprocess.run(["/usr/local/bin/model_runner", "-h"], 
                              capture_output=True, text=True, timeout=10)
        print(result.stdout)
    except Exception as e:
        print(f"Could not get help: {e}")

    variants = preprocess_variants(IMAGE_PATH, mean, scale, (256,256))
    runners = find_runners()
    if not runners:
        print("❌ No model runner binaries/scripts found in PATH.")
        sys.exit(2)

    if os.path.isdir(OUT_DIR):
        for f in glob.glob(os.path.join(OUT_DIR, "*")):
            os.remove(f)
    else:
        os.makedirs(OUT_DIR, exist_ok=True)

    # Try a matrix: input key names × input variants × runner binaries × flag styles
    input_keys = ["input", "input0", "in0", "0", "images", "data"]
    last_err = None
    for (default_key, tag, arr) in variants:
        for key in input_keys:
            npz_path = f"runner_in_{tag}_{key}.npz"
            np.savez(npz_path, **{key: arr})
            for cmd in runners:
                print(f"\n🔎 Variant={tag}, key={key}, runner={' '.join(cmd)}")
                try:
                    outs = try_runner(cmd, MODEL_PATH, npz_path, OUT_DIR)
                    print(f"📦 Outputs: {outs[:3]}{' ...' if len(outs)>3 else ''}")
                    all_outs = load_outputs(outs[0])  # returns dict
                    
                    # Prefer packed head if available
                    packed_key, packed_cp = pick_packed_head(all_outs, num_classes=1024)
                    if packed_cp is not None:
                        print(f"✅ Using packed tensor: {packed_key} shape={packed_cp.shape}")
                        analyze_packed_head(packed_cp, num_classes=1024)
                        print("\n🎉 Detection analysis complete.")
                        return
                    
                    # Fallback to separate heads
                    bbox_cands, cls_cands = select_heads(all_outs)
                    if not bbox_cands:
                        raise RuntimeError("No 4-channel bbox tensor found in outputs")

                    # pick first candidates (or add heuristics if multiple)
                    bbox_key, bbox_cp = bbox_cands[0]
                    cls_cp = cls_cands[0][1] if cls_cands else None

                    print(f"✅ Using bbox tensor: {bbox_key} shape={bbox_cp.shape}")
                    if cls_cp is not None:
                        print(f"✅ Using class tensor: {cls_cands[0][0]} shape={cls_cp.shape}")

                    analyze_detection(bbox_cp, cls_cp)
                    print("\n🎉 Detection analysis complete.")
                    return
                except Exception as e:
                    print(f"⚠️  This attempt failed: {e}")
                    last_err = e
    print(f"\n❌ All attempts failed. Last error: {last_err}")
    sys.exit(3)

if __name__ == "__main__":
    main()
