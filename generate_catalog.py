"""
generate_catalog.py
───────────────────
Generates a catalog_embeddings.json from a folder of reference product images
using DINOv2-small embeddings. The output JSON is loaded by the Flutter app
for on-device SKU matching.

Usage:
    # PyTorch (requires transformers + torch)
    python generate_catalog.py --catalog reference_images/

    # ONNX (faster, no PyTorch needed at runtime)
    python generate_catalog.py --catalog reference_images/ --model onnx --onnx-path dinov2_small.onnx

    # Custom output path
    python generate_catalog.py --catalog reference_images/ --output assets/catalog_embeddings.json

Input folder layout:
    reference_images/
    ├── Coke_Can_Red/
    │   ├── front.jpg
    │   ├── back.jpg
    │   └── ...
    └── Sprite_Can_Green/
        └── front.jpg
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Dependency check ───────────────────────────────────────────────────────────

def _require(package, import_name=None):
    import importlib
    try:
        return importlib.import_module(import_name or package)
    except ImportError:
        print(f"[ERROR] Missing: {package}  →  pip install {package}")
        sys.exit(1)

np  = _require("numpy")
PIL = _require("Pillow", "PIL")
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────

RESIZE_TO   = 256          # shortest edge before center crop
CROP_SIZE   = 224          # final square crop fed to DINOv2
EMB_DIM     = 384          # DINOv2-small CLS token dimension
MEAN        = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD         = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img_path: str) -> np.ndarray:
    """
    Loads an image and returns a float32 NCHW tensor [1, 3, 224, 224]
    matching DINOv2's expected preprocessing exactly.

    Pipeline:
        1. Open as RGB
        2. Resize shortest edge → 256 px  (BICUBIC, preserve aspect ratio)
        3. Center crop 224 × 224
        4. Scale pixels to [0, 1]  (÷ 255)
        5. Normalize:  (pixel − mean) / std   (ImageNet stats, per channel)
        6. Transpose to NCHW
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # Step 2: resize shortest edge to RESIZE_TO
    scale  = RESIZE_TO / min(w, h)
    new_w  = round(w * scale)
    new_h  = round(h * scale)
    img    = img.resize((new_w, new_h), Image.BICUBIC)

    # Step 3: center crop to CROP_SIZE × CROP_SIZE
    x0  = (new_w - CROP_SIZE) // 2
    y0  = (new_h - CROP_SIZE) // 2
    img = img.crop((x0, y0, x0 + CROP_SIZE, y0 + CROP_SIZE))

    # Steps 4–6: numpy pipeline
    arr = np.array(img, dtype=np.float32) / 255.0   # [224, 224, 3]
    arr = (arr - MEAN) / STD                          # normalize
    arr = arr.transpose(2, 0, 1)[np.newaxis]          # [1, 3, 224, 224]
    return arr

# ── Embedder — PyTorch ────────────────────────────────────────────────────────

class PyTorchEmbedder:
    def __init__(self):
        print("  Loading DINOv2-small from HuggingFace (cached after first run)…")
        torch        = _require("torch")
        transformers = _require("transformers")
        from transformers import AutoModel

        self.torch = torch
        self.model = AutoModel.from_pretrained("facebook/dinov2-small")
        self.model.eval()
        print(f"  ✅  PyTorch model ready  ({EMB_DIM}-dim embeddings)")

    def embed(self, tensor: np.ndarray) -> np.ndarray:
        """tensor: [1, 3, 224, 224] float32 ndarray → [384] L2-normed ndarray"""
        t = self.torch.from_numpy(tensor)
        with self.torch.no_grad():
            out = self.model(pixel_values=t).last_hidden_state  # [1, 257, 384]
        cls = out[0, 0, :].numpy()                               # [384]
        return cls / np.linalg.norm(cls)

# ── Embedder — ONNX ───────────────────────────────────────────────────────────

class ONNXEmbedder:
    def __init__(self, onnx_path: str):
        ort = _require("onnxruntime")

        if not os.path.exists(onnx_path):
            print(f"[ERROR] ONNX model not found: {onnx_path}")
            print("        Run  python export_dinov2.py  first.")
            sys.exit(1)

        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        if size_mb < 10:
            print(f"[ERROR] {onnx_path} is only {size_mb:.1f} MB — weights appear missing.")
            print("        Re-export with  python export_dinov2.py  (requires dynamo=False).")
            sys.exit(1)

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sess     = ort.InferenceSession(onnx_path, sess_options=opts)
        self.inp_name = self.sess.get_inputs()[0].name
        print(f"  ✅  ONNX model ready  ({onnx_path}, {size_mb:.0f} MB, {EMB_DIM}-dim)")

    def embed(self, tensor: np.ndarray) -> np.ndarray:
        """tensor: [1, 3, 224, 224] float32 ndarray → [384] L2-normed ndarray"""
        out = self.sess.run(None, {self.inp_name: tensor})[0]  # [1, 257, 384]
        cls = out[0, 0, :]                                      # [384]
        return cls / np.linalg.norm(cls)

# ── Image discovery ───────────────────────────────────────────────────────────

def discover_products(catalog_dir: str) -> dict[str, list[str]]:
    """
    Walks catalog_dir and returns {product_name: [sorted image paths]}.
    Only immediate subdirectories are treated as products — loose images at
    the top level are ignored with a warning.
    """
    root = Path(catalog_dir)
    if not root.is_dir():
        print(f"[ERROR] Catalog directory not found: {catalog_dir}")
        sys.exit(1)

    products = {}
    loose    = [p for p in root.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if loose:
        print(f"  ⚠️   Ignoring {len(loose)} loose image(s) at root level "
              f"(place them inside a product subfolder)")

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        imgs = sorted(
            p for p in subdir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        if not imgs:
            print(f"  ⚠️   Skipping empty folder: {subdir.name}")
            continue
        products[subdir.name] = [str(p) for p in imgs]

    return products

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate DINOv2 catalog embeddings from reference product images.")
    parser.add_argument("--catalog",   required=True,
                        help="Path to reference_images/ folder")
    parser.add_argument("--output",    default="catalog_embeddings.json",
                        help="Output JSON path (default: catalog_embeddings.json)")
    parser.add_argument("--model",     choices=["pytorch", "onnx"], default="pytorch",
                        help="Embedding backend (default: pytorch)")
    parser.add_argument("--onnx-path", default="dinov2_small.onnx",
                        help="Path to dinov2_small.onnx (required when --model onnx)")
    args = parser.parse_args()

    print("=" * 60)
    print("  DINOv2 Catalog Generator")
    print("=" * 60)

    # ── Load embedder ─────────────────────────────────────────────────────────
    print(f"\nBackend: {args.model.upper()}")
    if args.model == "onnx":
        embedder = ONNXEmbedder(args.onnx_path)
    else:
        embedder = PyTorchEmbedder()

    # ── Discover products ─────────────────────────────────────────────────────
    print(f"\nScanning: {args.catalog}")
    products_map = discover_products(args.catalog)

    if not products_map:
        print("[ERROR] No product subfolders with images found.")
        sys.exit(1)

    total_imgs = sum(len(v) for v in products_map.values())
    print(f"  Found {len(products_map)} products, {total_imgs} images total\n")

    # ── Embed ─────────────────────────────────────────────────────────────────
    catalog_products: dict = {}
    all_embeddings: list[tuple[str, str, np.ndarray]] = []  # (product, angle, emb)

    for product_name, img_paths in products_map.items():
        print(f"  [{product_name}]")
        angles     = []
        embeddings = []

        for img_path in img_paths:
            angle = Path(img_path).stem      # filename without extension
            try:
                tensor = preprocess(img_path)
                emb    = embedder.embed(tensor)  # [384] L2-normed
                angles.append(angle)
                embeddings.append(emb)
                all_embeddings.append((product_name, angle, emb))
                print(f"    ✅  {angle:<20s} norm={np.linalg.norm(emb):.4f}")
            except Exception as e:
                print(f"    ⚠️   {angle} — FAILED: {e}")

        if embeddings:
            catalog_products[product_name] = {
                "angles":     angles,
                "embeddings": [e.tolist() for e in embeddings],
            }

    if not catalog_products:
        print("[ERROR] No embeddings generated. Check your images.")
        sys.exit(1)

    # ── Build JSON ────────────────────────────────────────────────────────────
    num_embeddings = sum(len(v["angles"]) for v in catalog_products.values())
    output_data = {
        "model":          "dinov2-small",
        "embedding_dim":  EMB_DIM,
        "version":        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "num_products":   len(catalog_products),
        "num_embeddings": num_embeddings,
        "products":       catalog_products,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # ── Self-test ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("SELF-TEST: Each product's first image should match its own")
    print("           other angles best (intra-product > inter-product)")
    print("─" * 60)

    all_pass    = True
    warned_once = False

    for product_name, data in catalog_products.items():
        embs   = [np.array(e) for e in data["embeddings"]]
        angles = data["angles"]

        if len(embs) < 2:
            print(f"  {product_name}: only 1 image — skipping self-test "
                  f"(add more angles for a meaningful test)")
            continue

        query     = embs[0]
        query_ang = angles[0]

        # Cosine similarity of query vs every OTHER embedding in catalog
        best_score   = -1.0
        best_product = ""
        best_angle   = ""

        for other_prod, other_data in catalog_products.items():
            other_embs   = [np.array(e) for e in other_data["embeddings"]]
            other_angles = other_data["angles"]
            for i, (oemb, oang) in enumerate(zip(other_embs, other_angles)):
                # Skip comparing the query with itself
                if other_prod == product_name and oang == query_ang:
                    continue
                score = float(np.dot(query, oemb))   # both L2-normed → dot = cosine
                if score > best_score:
                    best_score   = score
                    best_product = other_prod
                    best_angle   = oang

        ok = best_product == product_name
        if not ok:
            all_pass = False

        mark = "✅" if ok else "❌ WARNING"
        print(f"  {product_name} [{query_ang}]")
        print(f"    → best match: {best_product} [{best_angle}]  "
              f"(score={best_score:.4f})  {mark}")

        if not ok and not warned_once:
            warned_once = True
            print()
            print("  ⚠️   Cross-product false match detected!")
            print("       Possible causes:")
            print("         • Reference images are too generic (no visible label)")
            print("         • Products look visually very similar")
            print("         • Image quality is too low")
            print()

    print()
    if all_pass:
        print("  ✅  All products pass the self-test")
    else:
        print("  ⚠️   Some products failed — review reference images before deploying")

    # ── Summary ───────────────────────────────────────────────────────────────
    file_kb = out_path.stat().st_size / 1024
    print()
    print("=" * 60)
    print("  Catalog generated")
    print("=" * 60)
    print(f"  Products        : {len(catalog_products)}")
    print(f"  Total embeddings: {num_embeddings}")
    print(f"  Embedding dim   : {EMB_DIM}")
    print(f"  File size       : {file_kb:.0f} KB")
    print(f"  Saved to        : {out_path}")
    print()
    print("  Next step:")
    print(f"    cp {out_path} assets/catalog_embeddings.json")
    print()


if __name__ == "__main__":
    main()
