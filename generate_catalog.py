"""
generate_catalog.py
───────────────────
Generates a catalog_embeddings.json from a folder of reference product images
using DINOv3-small (facebook/dinov3-vits16-pretrain-lvd1689m) embeddings.
The output JSON is loaded by the Flutter app for on-device SKU matching.

⚠️  IMPORTANT: This catalog is ONLY compatible with the DINOv3 ONNX model
    (dinov3_small.onnx). Do NOT use a DINOv2-generated catalog with DINOv3
    or vice versa — the embedding spaces are different and matching will fail.

Usage:
    # PyTorch backend (requires transformers + torch + HF token for gated repo)
    python generate_catalog.py --catalog reference_images/ --token hf_xxx

    # ONNX backend (faster, no transformers needed; use the exported ONNX)
    python generate_catalog.py --catalog reference_images/ --model onnx --onnx-path dinov3_small.onnx

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

DINOv3 vs DINOv2 differences handled here:
    • Model ID : facebook/dinov3-vits16-pretrain-lvd1689m  (gated repo — needs token)
    • Output   : pooler_output [batch, 384]  instead of last_hidden_state[:, 0, :]
    • Context  : torch.inference_mode()      instead of torch.no_grad()
    • ONNX out : out[0, :]                   instead of out[0, 0, :]  (2D not 3D)
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

MODEL_ID    = "facebook/dinov3-vits16-pretrain-lvd1689m"
RESIZE_TO   = 256          # shortest edge before center crop
CROP_SIZE   = 224          # final square crop fed to DINOv3
EMB_DIM     = 384          # DINOv3-small pooler_output dimension
MEAN        = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD         = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img_path: str) -> np.ndarray:
    """
    Loads an image and returns a float32 NCHW tensor [1, 3, 224, 224]
    matching DINOv3's expected preprocessing exactly (identical to DINOv2).

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
    def __init__(self, token: str | None = None):
        print(f"  Loading DINOv3-small from HuggingFace (cached after first run)…")
        print(f"  Model: {MODEL_ID}")
        torch        = _require("torch")
        _            = _require("transformers")
        from transformers import AutoModel

        self.torch = torch
        hf_kwargs  = {"token": token} if token else {}
        self.model = AutoModel.from_pretrained(MODEL_ID, **hf_kwargs)
        self.model.eval()
        print(f"  ✅  PyTorch DINOv3 model ready  ({EMB_DIM}-dim embeddings)")

    def embed(self, tensor: np.ndarray) -> np.ndarray:
        """tensor: [1, 3, 224, 224] float32 ndarray → [384] L2-normed ndarray

        DINOv3 uses pooler_output [1, 384] — no CLS-token slicing needed.
        Uses torch.inference_mode() (preferred over no_grad for inference).
        """
        t = self.torch.from_numpy(tensor)
        with self.torch.inference_mode():
            out = self.model(pixel_values=t).pooler_output  # [1, 384]
        emb = out[0].numpy()                                 # [384]
        return emb / np.linalg.norm(emb)

# ── Embedder — ONNX ───────────────────────────────────────────────────────────

class ONNXEmbedder:
    def __init__(self, onnx_path: str):
        ort = _require("onnxruntime")

        if not os.path.exists(onnx_path):
            print(f"[ERROR] ONNX model not found: {onnx_path}")
            print("        Run  python export_dinov3.py  first.")
            sys.exit(1)

        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        if size_mb < 10:
            print(f"[ERROR] {onnx_path} is only {size_mb:.1f} MB — weights appear missing.")
            print("        Re-export with  python export_dinov3.py")
            sys.exit(1)

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sess     = ort.InferenceSession(onnx_path, sess_options=opts)
        self.inp_name = self.sess.get_inputs()[0].name

        # Detect output shape to confirm this is the DINOv3 wrapper
        # (DINOv3 outputs [batch, 384] — 2D; DINOv2 outputs [batch, seq, 384] — 3D)
        out_shape = self.sess.get_outputs()[0].shape
        self.is_2d = len(out_shape) == 2
        shape_desc = f"[batch, {EMB_DIM}] pooler_output" if self.is_2d \
                     else f"[batch, seq, {EMB_DIM}] hidden_states"
        print(f"  ✅  ONNX model ready  ({onnx_path}, {size_mb:.0f} MB)")
        print(f"       Output: {shape_desc}")
        if not self.is_2d:
            print("  ⚠️  Warning: output looks like DINOv2 (3D) — expected DINOv3 (2D).")
            print("       Make sure you are using dinov3_small.onnx, not dinov2_small.onnx.")

    def embed(self, tensor: np.ndarray) -> np.ndarray:
        """tensor: [1, 3, 224, 224] float32 ndarray → [384] L2-normed ndarray

        DINOv3 ONNX wrapper outputs pooler_output as [1, 384] (2D).
        DINOv2 outputs last_hidden_state as [1, seq_len, 384] (3D) with CLS at [0].
        This embedder handles both but logs a warning for the DINOv2 case.
        """
        out = self.sess.run(None, {self.inp_name: tensor})[0]
        if self.is_2d:
            emb = out[0, :]      # [384]  — DINOv3 pooler_output
        else:
            emb = out[0, 0, :]   # [384]  — DINOv2 CLS token (fallback)
        return emb / np.linalg.norm(emb)

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
        description="Generate DINOv3 catalog embeddings from reference product images.")
    parser.add_argument("--catalog",   required=True,
                        help="Path to reference_images/ folder")
    parser.add_argument("--output",    default="catalog_embeddings.json",
                        help="Output JSON path (default: catalog_embeddings.json)")
    parser.add_argument("--model",     choices=["pytorch", "onnx"], default="pytorch",
                        help="Embedding backend (default: pytorch)")
    parser.add_argument("--onnx-path", default="dinov3_small.onnx",
                        help="Path to dinov3_small.onnx (required when --model onnx)")
    parser.add_argument("--token",     default=None,
                        help="HuggingFace access token (hf_...) for the gated DINOv3 repo")
    args = parser.parse_args()

    # Accept token from env var too
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print("=" * 60)
    print("  DINOv3 Catalog Generator")
    print("=" * 60)

    # ── Load embedder ─────────────────────────────────────────────────────────
    print(f"\nBackend: {args.model.upper()}")
    if args.model == "onnx":
        embedder = ONNXEmbedder(args.onnx_path)
    else:
        embedder = PyTorchEmbedder(token=token)

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
        "model":          "dinov3-small",
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
    print("SELF-TEST RESULTS:")
    print("Each product's first angle must match its OWN other angles better")
    print("than any other product.  (intra-product cosine > inter-product)")
    print("─" * 60)

    all_pass      = True
    warned_once   = False
    best_intra_overall = -1.0   # track global max intra-product score
    best_inter_overall = -1.0   # track global max inter-product score

    for product_name, data in catalog_products.items():
        embs   = [np.array(e) for e in data["embeddings"]]
        angles = data["angles"]

        if len(embs) < 2:
            print(f"  {product_name}: only 1 image — skipping "
                  f"(add more angles for a meaningful test)")
            continue

        query     = embs[0]
        query_ang = angles[0]

        # Cosine similarity of query vs every OTHER embedding in the full catalog
        best_score   = -1.0
        best_product = ""
        best_angle   = ""

        for other_prod, other_data in catalog_products.items():
            other_embs   = [np.array(e) for e in other_data["embeddings"]]
            other_angles = other_data["angles"]
            for oemb, oang in zip(other_embs, other_angles):
                if other_prod == product_name and oang == query_ang:
                    continue                               # skip self
                score = float(np.dot(query, oemb))        # cosine (both L2-normed)
                if score > best_score:
                    best_score   = score
                    best_product = other_prod
                    best_angle   = oang

        ok = best_product == product_name
        if not ok:
            all_pass = False
            best_inter_overall = max(best_inter_overall, best_score)
        else:
            best_intra_overall = max(best_intra_overall, best_score)

        mark = "✅" if ok else "❌"
        print(f"  {product_name} {query_ang} → "
              f"best match: {best_product} {best_angle} "
              f"({best_score:.2f}) {mark}")

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
    print("  DINOv3 Catalog generated ✅")
    print("=" * 60)
    print(f"  Model           : {MODEL_ID}")
    print(f"  Products        : {len(catalog_products)}")
    print(f"  Total embeddings: {num_embeddings}")
    print(f"  Embedding dim   : {EMB_DIM}")
    print(f"  File size       : {file_kb:.0f} KB")
    print(f"  Saved to        : {out_path}")
    print()
    print("  ⚠️  This catalog is ONLY compatible with dinov3_small.onnx")
    print("     Do NOT mix with a DINOv2-generated catalog.")
    print()
    print("  Next step:")
    print(f"    cp {out_path} assets/catalog_embeddings.json")
    print()


if __name__ == "__main__":
    main()
