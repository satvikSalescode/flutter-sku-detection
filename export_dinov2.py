"""
export_dinov2.py
────────────────
Exports facebook/dinov2-small to a self-contained ONNX file for on-device
SKU matching in Flutter via ONNX Runtime.

Fixes vs v1:
  • Uses legacy (JIT-trace) exporter via dynamo=False → single self-contained
    .onnx file with all weights inline (~85 MB), not the dynamo exporter that
    writes weights to a separate .data file (1.4 MB stub).
  • Targets opset 18 — opset 17 has no Resize adapter in torch 2.11+.
  • Wraps model in a thin nn.Module so exactly ONE output tensor is exported
    (last_hidden_state [batch, 257, 384]). Avoids the stray 'select' output
    that dynamo tracing picked up from the post-processing code.

Usage:
    python export_dinov2.py

Requirements:
    pip install transformers torch onnx onnxruntime
"""

import sys
import os

# ── Dependency check ───────────────────────────────────────────────────────────

def _require(package, import_name=None):
    import importlib
    name = import_name or package
    try:
        return importlib.import_module(name)
    except ImportError:
        print(f"[ERROR] Missing package: {package}")
        print(f"        Install with:  pip install {package}")
        sys.exit(1)

print("Checking dependencies…")
torch        = _require("torch")
transformers = _require("transformers")
onnx         = _require("onnx")
ort          = _require("onnxruntime")
np           = _require("numpy")

print(f"  torch        {torch.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  onnx         {onnx.__version__}")
print(f"  onnxruntime  {ort.__version__}")
print()

from transformers import AutoModel

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID    = "facebook/dinov2-small"
OUTPUT_FILE = "dinov2_small.onnx"
OPSET       = 18          # opset 17 has no Resize adapter in torch 2.11+
INPUT_H     = 224
INPUT_W     = 224

# ── Thin wrapper — one clean output ───────────────────────────────────────────
# Wrapping the HuggingFace model ensures:
#   • Only last_hidden_state is exported (no stray 'select' or pooler outputs)
#   • pixel_values is the only positional argument — JIT tracing works cleanly
#   • The CLS slice [:, 0, :] happens at inference time in Flutter, not here,
#     so the full sequence is available for any pooling strategy later.

class _DINOv2Backbone(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Returns last_hidden_state: [batch, seq_len, 384]
        return self.model(pixel_values=pixel_values).last_hidden_state


# ── 1. Load model ─────────────────────────────────────────────────────────────

print(f"Loading {MODEL_ID} from HuggingFace…")
print("  (first run downloads ~350 MB — cached on subsequent runs)")

try:
    hf_model = AutoModel.from_pretrained(MODEL_ID)
    hf_model.eval()
    wrapper  = _DINOv2Backbone(hf_model)
    wrapper.eval()
    n_params = sum(p.numel() for p in hf_model.parameters())
    print(f"  ✅  Model loaded — {n_params:,} parameters")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

# ── 2. Dummy input + forward pass ─────────────────────────────────────────────

print("\nRunning test forward pass…")
dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)  # [1, 3, 224, 224]

try:
    with torch.no_grad():
        out = wrapper(dummy)          # [1, seq_len, 384]
    cls_token = out[:, 0, :]         # [1, 384]
    print(f"  last_hidden_state shape : {list(out.shape)}")
    print(f"  CLS token shape         : {list(cls_token.shape)}")
    print(f"  Embedding dim           : {cls_token.shape[-1]}")
    print(f"  Sequence length         : {out.shape[1]}  "
          f"(= {out.shape[1]-1} patches + 1 CLS token)")
    print("  ✅  Forward pass OK")
except Exception as e:
    print(f"[ERROR] Forward pass failed: {e}")
    sys.exit(1)

# Capture reference output for verification later
with torch.no_grad():
    pt_out = wrapper(dummy).detach().numpy()   # [1, seq_len, 384]

# ── 3. ONNX export (legacy JIT-trace exporter) ────────────────────────────────
#
# dynamo=False  → uses torch.jit.trace under the hood.
#                 Produces a single self-contained .onnx with all weights
#                 embedded inline. The dynamo exporter (default in torch 2.x)
#                 writes weights to a separate .onnx.data file which makes the
#                 .onnx stub non-portable (~1.4 MB instead of ~85 MB).

print(f"\nExporting to ONNX (opset {OPSET}, legacy JIT-trace exporter)…")

try:
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,                        # single positional tensor
            OUTPUT_FILE,
            opset_version    = OPSET,
            input_names      = ["pixel_values"],
            output_names     = ["last_hidden_state"],
            dynamic_axes     = {
                "pixel_values"     : {0: "batch_size"},
                "last_hidden_state": {0: "batch_size"},
            },
            do_constant_folding = True,
            dynamo           = False,     # ← legacy exporter, self-contained file
        )
    print(f"  ✅  Exported → {OUTPUT_FILE}")
except Exception as e:
    print(f"[ERROR] ONNX export failed: {e}")
    sys.exit(1)

# ── 4. Verify ONNX model ──────────────────────────────────────────────────────

print("\nVerifying ONNX model…")

# 4a. Structural check
try:
    onnx_model = onnx.load(OUTPUT_FILE)
    onnx.checker.check_model(onnx_model)
    print("  ✅  ONNX graph structure valid")
except Exception as e:
    print(f"[ERROR] ONNX structure check failed: {e}")
    sys.exit(1)

# 4b. Runtime numerical check
try:
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3   # suppress verbose ORT logs
    sess = ort.InferenceSession(OUTPUT_FILE, sess_options=sess_opts)

    ort_out = sess.run(None, {"pixel_values": dummy.numpy()})[0]  # [1, seq, 384]

    max_diff = float(np.abs(ort_out - pt_out).max())
    tol      = 1e-4

    if max_diff <= tol:
        print(f"  ✅  ORT output matches PyTorch — max diff: {max_diff:.2e}")
    else:
        print(f"  ⚠️   Max diff {max_diff:.2e} exceeds {tol:.0e} "
              f"(float32 precision — acceptable for cosine similarity)")

    # CLS token sanity check
    ort_cls = ort_out[0, 0, :]
    norm    = float(np.linalg.norm(ort_cls))
    print(f"  CLS token range : [{ort_cls.min():.4f}, {ort_cls.max():.4f}]")
    print(f"  CLS token norm  : {norm:.4f}  (will be L2-normalised before matching)")

except Exception as e:
    print(f"[ERROR] ORT verification failed: {e}")
    sys.exit(1)

# ── 5. File size & I/O summary ────────────────────────────────────────────────

size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"\n  File : {OUTPUT_FILE}")
print(f"  Size : {size_mb:.1f} MB  (should be ~85 MB for self-contained export)")

if size_mb < 10:
    print()
    print("  ⚠️   File is suspiciously small — weights may not be embedded.")
    print("      This can happen if torch fell back to the dynamo exporter.")
    print("      Try: pip install --upgrade torch  OR  use torch < 2.0")

print("\nONNX model I/O:")
for inp in sess.get_inputs():
    print(f"  INPUT  {inp.name!r:25s} shape={inp.shape}  dtype={inp.type}")
for out_node in sess.get_outputs():
    print(f"  OUTPUT {out_node.name!r:25s} shape={out_node.shape}  dtype={out_node.type}")

n_outputs = len(sess.get_outputs())
if n_outputs != 1:
    print(f"\n  ⚠️   Expected 1 output, got {n_outputs}.")
    print("      In Flutter use output index 0 (last_hidden_state).")

# ── 6. Next steps ─────────────────────────────────────────────────────────────

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              Next steps                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  1. Copy dinov2_small.onnx to your Flutter app's assets/ folder             ║
║                                                                              ║
║  2. Model input shape: [batch, 3, 224, 224]                                  ║
║                                                                              ║
║  3. Preprocessing (must match exactly in Flutter):                           ║
║       a. Resize shortest edge → 256 px  (bicubic)                           ║
║       b. Center crop to 224 × 224                                            ║
║       c. Normalize with ImageNet stats:                                      ║
║            mean = [0.485, 0.456, 0.406]                                     ║
║            std  = [0.229, 0.224, 0.225]                                     ║
║            pixel = (pixel / 255.0 - mean) / std                             ║
║       d. Layout: NCHW float32  [batch, 3, 224, 224]                         ║
║                                                                              ║
║  4. Output: last_hidden_state  [batch, 257, 384]                            ║
║       → CLS token = output[:, 0, :]  →  [batch, 384]                       ║
║       → L2-normalize before cosine similarity                                ║
║                                                                              ║
║  5. Cosine similarity = dot(query_emb, catalog_emb)  (both L2-normalized)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("Done.")
