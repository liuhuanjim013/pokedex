#!/usr/bin/env python3
import os, sys, json, subprocess, shutil, glob
import numpy as np
import cv2

MODEL_PATH = "maixcam_deployment/pokemon_classifier_int8.cvimodel"
MUD_PATH   = "maixcam_deployment/pokemon_classifier.mud"
IMAGE_PATH = "images/0001_001.jpg"
OUT_DIR    = "runner_out"

# Expectation from your note: image 0001 ⇒ Pokémon id 1
EXPECTED_ID = 1     # 1-based index
NUM_CLASSES = 1025  # len(classes.txt)

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
            for k in ("preprocess","PREPROCESS","Preprocess"):
                if k in meta:
                    meta = {**meta, **meta[k]}
            m  = meta.get("mean") or meta.get("MEAN")
            sc = meta.get("scale") or meta.get("SCALE")
            if isinstance(m, (list, tuple)) and len(m) >= 3: mean  = [float(m[0]), float(m[1]), float(m[2])]
            if isinstance(sc, (list, tuple)) and len(sc) >= 3: scale = [float(sc[0]), float(sc[1]), float(sc[2])]
    except Exception:
        pass
    return mean, scale

def load_class_names(path="maixcam_deployment/classes.txt"):
    names = []
    if os.path.exists(path):
        names = [l.strip() for l in open(path, "r", encoding="utf-8") if l.strip()]
    return names

def guess_numeric_id(name):
    # Attempt to parse a leading numeric like "001 bulbasaur" -> 1
    i = 0
    while i < len(name) and name[i].isdigit():
        i += 1
    if i > 0:
        try:
            return int(name[:i])
        except:
            pass
    return None

def npz_save(path, **arrays):
    np.savez(path, **arrays)

def runner_cmds():
    cmds = []
    for name in ("/usr/local/bin/model_runner", "/usr/local/bin/model_runner.py"):
        if os.path.exists(name):
            cmds.append([name])
    # also python model_runner.py explicitly
    if os.path.exists("/usr/local/bin/model_runner.py"):
        cmds.append(["python3", "/usr/local/bin/model_runner.py"])
    return cmds

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def try_runner(model_cmd, model_path, in_npz_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "out.npz")
    cmd = model_cmd + ["--model", model_path, "--input", in_npz_path,
                       "--output", out_file, "--dump_all_tensors"]
    print("➡️  Running:", " ".join(cmd))
    p = run(cmd)
    if p.returncode != 0:
        print("❌ Runner failed.\nSTDERR:\n", p.stderr.strip() or "(empty)")
        raise RuntimeError("runner error")
    if os.path.isfile(out_file):
        print(f"✅ Runner wrote: {out_file}")
        return out_file
    # Fallback sweep
    outs = sorted(glob.glob(os.path.join(out_dir, "*.npz")) + glob.glob(os.path.join(out_dir, "*.npy")))
    if outs:
        print(f"ℹ️  Found outputs: {outs}")
        return outs[0]
    raise RuntimeError("no outputs produced")

def load_npz_as_dict(path):
    outs = {}
    with np.load(path) as d:
        for k in d.files:
            outs[k] = d[k]
    print("🧾 NPZ keys & shapes:")
    for k, v in outs.items():
        print(f"  - {k}: {v.shape} {v.dtype}")
    return outs

def squeeze_all(a):
    a = np.asarray(a)
    # squeeze all singletons
    while a.ndim > 0 and 1 in a.shape and a.ndim > 2:
        ax = [i for i, d in enumerate(a.shape) if d == 1][0]
        a = np.squeeze(a, axis=ax)
    return a

def pick_packed_head(outs, classes=NUM_CLASSES):
    wantC = 4 + classes   # packed = 4 bbox + classes (no objectness)
    for k, v in outs.items():
        a = squeeze_all(v)
        if a.ndim == 3:
            if a.shape[0] == wantC:   # (C,H,W)
                return k, a.reshape(wantC, -1)
            if a.shape[2] == wantC:   # (H,W,C)
                return k, np.transpose(a, (2,0,1)).reshape(wantC, -1)
        if a.ndim == 2:
            if a.shape[0] == wantC: return k, a
            if a.shape[1] == wantC: return k, a.T
    return None, None

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def decode_packed(packed_cp, class_names):
    """packed_cp: (4+classes, P). Return best (class_id, prob, bbox, pos)."""
    bbox = packed_cp[0:4, :]             # (4, P)
    cls_logits = packed_cp[4:, :]        # (classes, P)
    cls_probs  = sigmoid(cls_logits)     # per-class sigmoid
    best_cls   = np.argmax(cls_probs, axis=0)               # (P,)
    best_prob  = cls_probs[best_cls, np.arange(cls_probs.shape[1])]
    pos        = int(np.argmax(best_prob))
    cid        = int(best_cls[pos])      # 0-based
    prob       = float(best_prob[pos])
    bb         = bbox[:, pos]
    # Pretty print
    top5 = np.argsort(cls_probs[:, pos])[-5:][::-1]
    print(f"🏆 Best pos={pos} class_id(0-based)={cid} prob={prob:.6f}")
    print(f"🧱 BBox [cx,cy,w,h] = [{bb[0]:.3f},{bb[1]:.3f},{bb[2]:.3f},{bb[3]:.3f}]")
    if class_names:
        def nm(i): return class_names[i] if 0 <= i < len(class_names) else f"id_{i}"
        print("🥇 Top-5:", [(int(i), nm(int(i)), float(cls_probs[i, pos])) for i in top5])
    return cid, prob, bb, pos

def build_variants(img_path, mean, scale, size=(256,256)):
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rgb_f = rgb.astype(np.float32)
    rgb_f_01 = rgb_f / 255.0

    # Apply mean/scale if non-trivial
    def apply_ms(img):
        out = img.copy().astype(np.float32)
        for c in range(3):
            out[..., c] = (out[..., c] - mean[c]) * scale[c]
        return out

    rgb_fs = apply_ms(rgb_f) if (mean != [0,0,0] or scale != [1,1,1]) else rgb_f

    variants = []
    # Prioritize common export expectations:

    # 1) "images" NHWC uint8 BGR (very common for int8 exports)
    variants.append(("images", "nhwc_u8_bgr", bgr[None, ...]))

    # 2) "images" NCHW float32 RGB normalized 0-1
    variants.append(("images", "nchw_f32_rgb_01", np.transpose(rgb_f_01, (2,0,1))[None, ...]))

    # 3) "input" NCHW float32 RGB normalized 0-1
    variants.append(("input", "nchw_f32_rgb_01", np.transpose(rgb_f_01, (2,0,1))[None, ...]))

    # 4) "images" NCHW float32 RGB with mean/scale (if provided)
    variants.append(("images", "nchw_f32_rgb_ms", np.transpose(rgb_fs, (2,0,1))[None, ...]))

    # 5) "input" NCHW uint8 RGB
    variants.append(("input", "nchw_u8_rgb", np.transpose(rgb.astype(np.uint8), (2,0,1))[None, ...]))

    return variants

def main():
    print("🧪 CVIModel Detect Runtime Test (fixed decoding)")
    for p, lab in [(MODEL_PATH,"model"), (IMAGE_PATH,"image")]:
        if not os.path.exists(p):
            print(f"❌ {lab} not found: {p}")
            sys.exit(1)

    mean, scale = read_mud_mean_scale(MUD_PATH)
    print(f"📋 Preprocess mean={mean} scale={scale}")

    class_names = load_class_names()
    if class_names:
        print(f"🏷️ Loaded {len(class_names)} class names")

    variants = build_variants(IMAGE_PATH, mean, scale, (256,256))
    cmds = runner_cmds()
    if not cmds:
        print("❌ No model_runner found.")
        sys.exit(2)

    if os.path.isdir(OUT_DIR):
        for f in glob.glob(os.path.join(OUT_DIR, "*")):
            os.remove(f)
    else:
        os.makedirs(OUT_DIR, exist_ok=True)

    # Try variants; stop when expected id is detected
    last_err = None
    for key, tag, arr in variants:
        npz_path = f"runner_in_{tag}_{key}.npz"
        npz_save(npz_path, **{key: arr})

        for cmd in cmds:
            print(f"\n🔎 Variant={tag}, key={key}, runner={' '.join(cmd)}")
            try:
                out_path = try_runner(cmd, MODEL_PATH, npz_path, OUT_DIR)
                outs = load_npz_as_dict(out_path)

                # Prefer packed head 4+classes, with no objectness
                packed_key, packed_cp = pick_packed_head(outs, classes=NUM_CLASSES)
                if packed_cp is None:
                    raise RuntimeError("No packed head (4+classes) found in outputs")

                print(f"✅ Using packed tensor: {packed_key} shape={packed_cp.shape}")
                cid0, prob, bb, pos = decode_packed(packed_cp, class_names)

                # Convert 0-based to 1-based for comparison with your label convention
                predicted_id_1based = cid0 + 1

                if class_names:
                    name = class_names[cid0] if 0 <= cid0 < len(class_names) else f"id_{predicted_id_1based}"
                    print(f"🔢 Predicted id(1-based)={predicted_id_1based} name={name}")
                else:
                    print(f"🔢 Predicted id(1-based)={predicted_id_1based}")

                if predicted_id_1based == EXPECTED_ID:
                    print(f"\n🎯 Matched expected Pokémon #{EXPECTED_ID} using variant={tag}, key={key}, runner={' '.join(cmd)}")
                    print("🎉 Detection analysis complete.")
                    return
                else:
                    # If names carry leading numbers, give a hint
                    if class_names:
                        gid = guess_numeric_id(class_names[cid0])
                        if gid:
                            print(f"ℹ️ Name suggests numeric id {gid} for '{class_names[cid0]}'")

            except Exception as e:
                print(f"⚠️ Attempt failed: {e}")
                last_err = e

    print(f"\n❌ All attempts tried. Last error: {last_err}")
    sys.exit(3)

if __name__ == "__main__":
    main()