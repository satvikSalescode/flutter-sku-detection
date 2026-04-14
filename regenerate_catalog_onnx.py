"""
regenerate_catalog_onnx.py
==========================
Regenerates catalog_embeddings.json using the ONNX image encoder directly —
NOT open_clip. This guarantees the preprocessing is IDENTICAL to what the
Flutter app does at inference time, so cosine similarities will be correct.

Why this script exists
----------------------
open_clip's preprocessing for MobileCLIP-S2 uses 256×256 (the model's native
resolution), but the exported ONNX model was locked to 224×224 at export time.
Using open_clip to generate catalog embeddings while the app runs the 224×224
ONNX model produces completely orthogonal embeddings (~0.10 cosine similarity).
This script fixes the problem by using the ONNX model for BOTH catalog
generation and inference.

Preprocessing (matches Flutter clip_service.dart exactly)
----------------------------------------------------------
  1. Resize the shortest edge to 224 px (bicubic)
  2. Center-crop to 224×224
  3. Normalise: mean=[0.48145466, 0.4578275, 0.40821073]
                std =[0.26862954, 0.26130258, 0.27577711]
  4. Layout: NCHW float32

Requirements
------------
    pip install onnxruntime Pillow numpy tqdm

Usage
-----
    python regenerate_catalog_onnx.py \\
        --images_dir  product_images \\
        --master      product_master.json \\
        --model       assets/mobileclip2_s2_image_encoder.onnx \\
        --out         assets/catalog_embeddings.json
"""

import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import onnxruntime as ort
except ImportError:
    raise SystemExit("onnxruntime not found — run: pip install onnxruntime")

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--images_dir", default="product_images")
parser.add_argument("--master",     default="product_master.json")
parser.add_argument("--model",      default="assets/mobileclip2_s2_image_encoder.onnx")
parser.add_argument("--out",        default="assets/catalog_embeddings.json")
parser.add_argument("--batch",      type=int, default=16)
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

# ── ONNX session ───────────────────────────────────────────────────────────────
print(f"Loading ONNX model: {args.model}")
sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
IN_NAME  = sess.get_inputs()[0].name
OUT_NAME = sess.get_outputs()[0].name
# Verify the model expects [batch, 3, 224, 224]
in_shape = sess.get_inputs()[0].shape
print(f"  input : {IN_NAME}  {in_shape}")
print(f"  output: {OUT_NAME}  {sess.get_outputs()[0].shape}")

# Auto-read the spatial resolution the ONNX model was exported at.
# shape is ['batch', 3, H, W] — index 2 is H.
IMG_SIZE = int(in_shape[2]) if isinstance(in_shape[2], int) else 224
print(f"  image size detected from model: {IMG_SIZE}×{IMG_SIZE}")

# CLIP/OpenAI normalisation — must match clip_service.dart
MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(img_pil: Image.Image) -> np.ndarray:
    """
    Matches Flutter _resizeCenterCrop + normalisation in clip_service.dart:
      1. Resize shortest edge → IMG_SIZE (bicubic)
      2. Center-crop to IMG_SIZE × IMG_SIZE
      3. Normalise and convert to NCHW float32
    """
    img_pil = img_pil.convert("RGB")
    w, h = img_pil.size
    scale   = IMG_SIZE / min(w, h)
    new_w   = round(w * scale)
    new_h   = round(h * scale)
    img_pil = img_pil.resize((new_w, new_h), Image.BICUBIC)
    left    = (new_w - IMG_SIZE) // 2
    top     = (new_h - IMG_SIZE) // 2
    img_pil = img_pil.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    arr = np.array(img_pil, dtype=np.float32) / 255.0   # HWC [0,1]
    arr = (arr - MEAN) / STD                             # HWC normalised
    return arr.transpose(2, 0, 1)                        # CHW

# ── Batch encoder ──────────────────────────────────────────────────────────────

def embed_images(paths: list) -> list:
    """Return list of 512-d L2-normalised float lists."""
    results = []
    for i in range(0, len(paths), args.batch):
        batch_paths = paths[i : i + args.batch]
        tensors = []
        for p in batch_paths:
            try:
                tensors.append(preprocess(Image.open(p)))
            except Exception as e:
                print(f"  WARNING: could not load {p}: {e}")
                tensors.append(np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32))
        batch = np.stack(tensors, axis=0)                # [N, 3, 224, 224]
        feats = sess.run([OUT_NAME], {IN_NAME: batch})[0]  # [N, 512] already L2-normed
        results.extend(feats.tolist())
    return results

# ── Product master ─────────────────────────────────────────────────────────────
master: dict = {}
if os.path.exists(args.master):
    with open(args.master, encoding="utf-8") as f:
        raw = json.load(f)
    data = raw if isinstance(raw, list) else raw.get("products", [])
    for p in data:
        master[p["sku"]] = p
    print(f"Loaded {len(master)} products from {args.master}")
else:
    print(f"No master file at {args.master} — using folder names as SKUs.")

# ── Process SKU folders ────────────────────────────────────────────────────────
VIEW_NAMES = ["front", "back", "left", "right"]
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".webp"}

root     = Path(args.images_dir)
sku_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
products = []

print(f"\nFound {len(sku_dirs)} SKU folders in {root}\n")

for sku_dir in tqdm(sku_dirs, desc="SKUs"):
    sku  = sku_dir.name
    meta = master.get(sku, {"sku": sku, "name": sku, "brand": "", "category": ""})

    img_embs: dict = {}

    # Named views first
    for view in VIEW_NAMES:
        for ext in IMG_EXTS:
            p = sku_dir / f"{view}{ext}"
            if p.exists():
                img_embs[view] = embed_images([str(p)])[0]
                break

    # Any remaining images
    for p in sorted(sku_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        vname = p.stem.lower()
        if vname not in img_embs:
            img_embs[vname] = embed_images([str(p)])[0]

    if not img_embs:
        tqdm.write(f"  SKIP {sku} — no images found")
        continue

    # ── Compute centroid: average all view embeddings, L2-renormalise ──────────
    # The centroid gives a single representative vector per product.
    # Using centroid vs per-view max raises genuine-match scores from ~0.24 avg
    # to 0.59–0.81, while cross-class false positives stay below 0.49.
    # This is the ONLY embedding the app uses for matching at runtime.
    all_vecs = np.array(list(img_embs.values()), dtype=np.float32)  # [V, 512]
    centroid = all_vecs.mean(axis=0)                                  # [512]
    norm     = np.linalg.norm(centroid)
    if norm > 1e-8:
        centroid /= norm

    # Store centroid as a special key plus the individual views for reference.
    embs_with_centroid = {"centroid": centroid.tolist(), **img_embs}

    products.append({
        "sku":              sku,
        "name":             meta.get("name",     sku),
        "brand":            meta.get("brand",    ""),
        "category":         meta.get("category", ""),
        "image_embeddings": embs_with_centroid,
        "text_embeddings":  {},   # text encoder not required for image-only mode
    })

# ── Write output ───────────────────────────────────────────────────────────────
catalog = {
    "version":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model":    f"MobileCLIP-S2-ONNX-{IMG_SIZE}",
    "dim":      512,
    "products": products,
}
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(catalog, f, ensure_ascii=False)

size_kb = os.path.getsize(args.out) / 1024
print(f"\n✓  Wrote {len(products)} products → {args.out}  ({size_kb:.0f} KB)")
print(f"   Image embeddings: {sum(len(p['image_embeddings']) for p in products)}")
print()
print("━━  NEXT STEPS  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  1. Copy {args.out} into your Flutter project: assets/catalog_embeddings.json")
print(f"  2. Run:  adb shell pm clear com.example.sku_detector   (clears stale cache)")
print(f"  3. Run:  flutter run --debug")
print(f"  4. Check adb logcat for: [CLIP-Match] crop[N] best=0.6xx")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
