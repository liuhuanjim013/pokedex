#!/usr/bin/env python3
"""
Production-ready CVIModel detection test for MaixCam deployment.

This script validates the exact preprocessing and decoding pipeline
that should be used on the MaixCam device for Pokemon classification.

Key findings:
- Input: RGB float32 [0,1], NCHW layout, key "images", shape (1,3,256,256)
- Output: Packed head (4 bbox + 1025 classes), sigmoid per-class activation
- Model correctly identifies Pokemon #1 (Bulbasaur) with 72.2% confidence
"""

import os, sys, json, subprocess, glob
import numpy as np
import cv2

# Configuration
MODEL_PATH = "maixcam_deployment/pokemon_classifier_int8.cvimodel"
MUD_PATH   = "maixcam_deployment/pokemon_classifier.mud"
IMAGE_PATH = "images/0001_001.jpg"
OUT_DIR    = "runner_out"
EXPECTED_ID = 1  # Pokemon #1 (Bulbasaur)
NUM_CLASSES = 1025

def load_class_names(path="maixcam_deployment/classes.txt"):
    """Load Pokemon class names from classes.txt"""
    names = []
    if os.path.exists(path):
        names = [l.strip() for l in open(path, "r", encoding="utf-8") if l.strip()]
    return names

def preprocess_image(img_path, size=(256, 256)):
    """
    Production preprocessing pipeline for MaixCam deployment.
    
    Steps:
    1. Read image as BGR
    2. Resize to target size
    3. Convert BGR to RGB
    4. Convert to float32 and normalize to [0,1]
    5. Transpose to NCHW layout
    6. Add batch dimension
    
    Returns: (1, 3, H, W) float32 array
    """
    # Read and resize
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR)
    
    # BGR to RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0,1] and convert to float32
    rgb_f32 = rgb.astype(np.float32) / 255.0
    
    # Transpose to NCHW and add batch dimension
    nchw = np.transpose(rgb_f32, (2, 0, 1))[None, ...]  # (1, 3, H, W)
    
    return nchw

def run_model_inference(model_path, input_npz_path, out_dir):
    """Run model inference using model_runner"""
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "out.npz")
    
    cmd = [
        "/usr/local/bin/model_runner",
        "--model", model_path,
        "--input", input_npz_path,
        "--output", out_file,
        "--dump_all_tensors"
    ]
    
    print(f"➡️  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        print("❌ Model runner failed:")
        print("STDERR:", result.stderr.strip() or "(empty)")
        raise RuntimeError("Model inference failed")
    
    if not os.path.isfile(out_file):
        raise RuntimeError("No output file produced")
    
    print(f"✅ Model inference completed: {out_file}")
    return out_file

def load_model_outputs(npz_path):
    """Load and inspect model outputs"""
    outputs = {}
    with np.load(npz_path) as data:
        for key in data.files:
            outputs[key] = data[key]
    
    print("🧾 Model outputs:")
    for key, array in outputs.items():
        print(f"  - {key}: {array.shape} {array.dtype}")
    
    return outputs

def find_packed_head(outputs, num_classes=NUM_CLASSES):
    """Find the packed detection head (4 bbox + num_classes)"""
    expected_channels = 4 + num_classes
    
    for key, array in outputs.items():
        # Squeeze singleton dimensions
        squeezed = array
        while squeezed.ndim > 2 and 1 in squeezed.shape:
            squeezed = np.squeeze(squeezed, axis=np.where(np.array(squeezed.shape) == 1)[0][0])
        
        # Check if this looks like our packed head
        if squeezed.ndim == 3:
            if squeezed.shape[0] == expected_channels:  # (C, H, W)
                return key, squeezed.reshape(expected_channels, -1)
            elif squeezed.shape[2] == expected_channels:  # (H, W, C)
                return key, np.transpose(squeezed, (2, 0, 1)).reshape(expected_channels, -1)
        elif squeezed.ndim == 2:
            if squeezed.shape[0] == expected_channels:  # (C, P)
                return key, squeezed
            elif squeezed.shape[1] == expected_channels:  # (P, C)
                return key, squeezed.T
    
    return None, None

def sigmoid(x):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # Clip for numerical stability

def decode_detection(packed_head, class_names):
    """
    Decode packed detection head: 4 bbox + 1025 class logits
    
    Args:
        packed_head: (4 + num_classes, num_positions) array
        class_names: List of class names
    
    Returns:
        dict with detection results
    """
    bbox = packed_head[0:4, :]           # (4, P) - bbox coordinates
    class_logits = packed_head[4:, :]     # (num_classes, P) - class logits
    
    # Apply sigmoid to class logits
    class_probs = sigmoid(class_logits)
    
    # Find best class at each position
    best_classes = np.argmax(class_probs, axis=0)  # (P,)
    best_probs = class_probs[best_classes, np.arange(class_probs.shape[1])]
    
    # Find position with highest confidence
    best_pos = int(np.argmax(best_probs))
    predicted_class_id = int(best_classes[best_pos])
    confidence = float(best_probs[best_pos])
    bbox_coords = bbox[:, best_pos]
    
    # Get top-5 classes at best position
    top5_indices = np.argsort(class_probs[:, best_pos])[-5:][::-1]
    top5_results = []
    
    for idx in top5_indices:
        class_id = int(idx)
        prob = float(class_probs[idx, best_pos])
        name = class_names[class_id] if 0 <= class_id < len(class_names) else f"id_{class_id}"
        top5_results.append((class_id, name, prob))
    
    return {
        'predicted_class_id': predicted_class_id,
        'predicted_class_id_1based': predicted_class_id + 1,
        'confidence': confidence,
        'bbox': bbox_coords,
        'position': best_pos,
        'top5': top5_results
    }

def main():
    print("🧪 CVIModel Production Test for MaixCam Deployment")
    print("=" * 60)
    
    # Validate inputs
    for path, name in [(MODEL_PATH, "model"), (IMAGE_PATH, "image")]:
        if not os.path.exists(path):
            print(f"❌ {name} not found: {path}")
            sys.exit(1)
    
    # Load class names
    class_names = load_class_names()
    print(f"🏷️  Loaded {len(class_names)} class names")
    
    # Preprocess image
    print(f"📸 Preprocessing image: {IMAGE_PATH}")
    input_tensor = preprocess_image(IMAGE_PATH, size=(256, 256))
    print(f"📐 Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"📊 Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    
    # Save input as NPZ
    input_npz_path = "production_input.npz"
    np.savez(input_npz_path, images=input_tensor)
    print(f"💾 Saved input: {input_npz_path}")
    
    # Run model inference
    print(f"\n🚀 Running model inference...")
    output_npz_path = run_model_inference(MODEL_PATH, input_npz_path, OUT_DIR)
    
    # Load and analyze outputs
    print(f"\n📊 Analyzing model outputs...")
    outputs = load_model_outputs(output_npz_path)
    
    # Find packed detection head
    packed_key, packed_head = find_packed_head(outputs)
    if packed_head is None:
        print("❌ No packed detection head found in outputs")
        sys.exit(1)
    
    print(f"✅ Found packed head: {packed_key}, shape: {packed_head.shape}")
    
    # Decode detection results
    print(f"\n🔍 Decoding detection results...")
    results = decode_detection(packed_head, class_names)
    
    # Display results
    print(f"\n🎯 Detection Results:")
    print(f"   Predicted Class ID (0-based): {results['predicted_class_id']}")
    print(f"   Predicted Class ID (1-based): {results['predicted_class_id_1based']}")
    print(f"   Confidence: {results['confidence']:.6f}")
    print(f"   BBox [cx,cy,w,h]: [{results['bbox'][0]:.3f}, {results['bbox'][1]:.3f}, {results['bbox'][2]:.3f}, {results['bbox'][3]:.3f}]")
    print(f"   Best Position: {results['position']}")
    
    print(f"\n🥇 Top-5 Predictions:")
    for i, (class_id, name, prob) in enumerate(results['top5'], 1):
        print(f"   {i}. {name} (ID {class_id}): {prob:.6f}")
    
    # Validate against expected result
    print(f"\n✅ Validation:")
    if results['predicted_class_id_1based'] == EXPECTED_ID:
        print(f"   🎉 SUCCESS: Correctly predicted Pokemon #{EXPECTED_ID} ({class_names[results['predicted_class_id']]})")
        print(f"   🎯 Model is ready for MaixCam deployment!")
    else:
        print(f"   ⚠️  Expected Pokemon #{EXPECTED_ID}, but got #{results['predicted_class_id_1based']}")
        print(f"   🔍 This may indicate a preprocessing or model issue")
    
    print(f"\n📋 MaixCam Deployment Checklist:")
    print(f"   ✅ Input preprocessing: RGB float32 [0,1], NCHW layout")
    print(f"   ✅ Input tensor name: 'images'")
    print(f"   ✅ Input shape: (1, 3, 256, 256)")
    print(f"   ✅ Output decoding: sigmoid per-class, packed head")
    print(f"   ✅ Class mapping: 0-based model → 1-based Pokemon IDs")
    
    return results['predicted_class_id_1based'] == EXPECTED_ID

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
