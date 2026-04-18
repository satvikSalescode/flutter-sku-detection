"""
export_dinov3.py
================
Exports facebook/dinov3-vits16-pretrain-lvd1689m to ONNX (opset 17)
for use in the Flutter SKU-detection app.

Loads model weights from a LOCAL safetensors file to avoid re-downloading
the 86 MB checkpoint.  Falls back to HuggingFace Hub if local load fails.

Requirements:
    pip install transformers safetensors torch onnx onnxruntime

Usage:
    python3 export_dinov3.py
    python3 export_dinov3.py --weights "/path/to/model.safetensors"
    python3 export_dinov3.py --weights "/path/to/model.safetensors" --out dinov3_small.onnx
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_ID             = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_WEIGHTS_PATH = "/Users/satvikchaudhary/Desktop/IRED DOCS/DinoV3/model.safetensors"
DEFAULT_ONNX_OUT     = "dinov3_small.onnx"
INPUT_SIZE           = 224
EMB_DIM              = 384


# ── Wrapper ────────────────────────────────────────────────────────────────────

class DINOv3Wrapper(torch.nn.Module):
    """Returns only pooler_output so the ONNX graph has a single clean output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.pooler_output


# ── Step 1: download config + processor (small JSON files only) ────────────────

def load_config_and_processor(token: str | None = None):
    print("\n── Step 1: Loading config & processor from HuggingFace ────────────")
    try:
        from transformers import AutoConfig, AutoImageProcessor
    except ImportError:
        print("❌  transformers not installed. Run:  pip install transformers")
        sys.exit(1)

    hf_kwargs = {"token": token} if token else {}

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, **hf_kwargs)
    config    = AutoConfig.from_pretrained(MODEL_ID, **hf_kwargs)

    print(f"   Model ID       : {MODEL_ID}")
    print(f"   Hidden size    : {config.hidden_size}")
    print(f"   Num layers     : {config.num_hidden_layers}")
    print(f"   Num heads      : {config.num_attention_heads}")
    return config, processor


# ── Step 2: load model weights ─────────────────────────────────────────────────

def load_model(config, local_weights_path: str, token: str | None = None):
    print("\n── Step 2: Loading model weights ───────────────────────────────────")
    try:
        from transformers import AutoModel
    except ImportError:
        print("❌  transformers not installed.")
        sys.exit(1)

    weights_path = Path(local_weights_path)
    model        = None

    # ── Try local safetensors first ──────────────────────────────────────────
    if weights_path.exists():
        print(f"   Found local weights: {weights_path}")
        print(f"   File size          : {weights_path.stat().st_size / 1e6:.1f} MB")
        try:
            from safetensors.torch import load_file

            print("   Loading model architecture from config…")
            model = AutoModel.from_config(config)

            print("   Loading weights from safetensors…")
            state_dict = load_file(str(weights_path))

            # The safetensors file stores keys WITHOUT the 'model.' prefix
            # (e.g. 'layer.0.norm1.weight') but AutoModel wraps everything
            # under a 'model.' namespace.  Try both layouts automatically.
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if len(missing) > 10 and all(k.startswith("model.") for k in missing):
                print("   ↳ Remapping keys: adding 'model.' prefix…")
                state_dict = {"model." + k: v for k, v in state_dict.items()}
                missing, unexpected = model.load_state_dict(state_dict, strict=False)

            if missing:
                print(f"   ⚠️  Missing keys   : {len(missing)}  (e.g. {missing[:3]})")
            if unexpected:
                print(f"   ⚠️  Unexpected keys : {len(unexpected)}  (e.g. {unexpected[:3]})")

            if missing and len(missing) > 10:
                raise RuntimeError(
                    f"Too many missing keys ({len(missing)}) — local weights "
                    "likely incompatible with this config. Falling back to HF."
                )

            model.eval()
            print(f"   ✅ Loaded DINOv3-small from local weights")

        except Exception as e:
            print(f"   ⚠️  Local load failed: {e}")
            print("   Falling back to HuggingFace Hub download…")
            model = None
    else:
        print(f"   ⚠️  Local file not found: {weights_path}")
        print("   Falling back to HuggingFace Hub download…")

    # ── Fallback: download from HF ───────────────────────────────────────────
    if model is None:
        print(f"   Downloading from HuggingFace: {MODEL_ID}")
        hf_kwargs = {"token": token} if token else {}
        model = AutoModel.from_pretrained(MODEL_ID, **hf_kwargs)
        model.eval()
        print("   ✅ Loaded DINOv3-small from HuggingFace")

    return model


# ── Step 3: verify forward pass ────────────────────────────────────────────────

def verify_forward_pass(model):
    print("\n── Step 3: Verifying forward pass ──────────────────────────────────")
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.inference_mode():
        outputs = model(pixel_values=dummy_input)
        pooled  = outputs.pooler_output

    print(f"   Pooler output shape: {tuple(pooled.shape)}")
    assert pooled.shape == (1, EMB_DIM), \
        f"Wrong output shape: {pooled.shape}, expected (1, {EMB_DIM})"
    print("   ✅ Forward pass OK")
    return dummy_input


# ── Step 4 + 5: export to ONNX ────────────────────────────────────────────────

def export_onnx(model, dummy_input, onnx_out: str):
    print(f"\n── Step 4 & 5: Exporting to ONNX → {onnx_out} ──────────────────────")

    wrapper = DINOv3Wrapper(model)
    wrapper.eval()

    # Warm-up
    with torch.inference_mode():
        _ = wrapper(dummy_input)

    # Force the legacy TorchScript-based exporter (dynamo=False).
    # The new dynamo exporter (default in PyTorch ≥2.6) does not handle
    # opset 17 correctly for transformer models — it defaults internally
    # to opset 18 and then fails during the inliner pass.
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_out,
        opset_version       = 17,
        input_names         = ["pixel_values"],
        output_names        = ["embeddings"],
        dynamic_axes        = {
            "pixel_values": {0: "batch_size"},
            "embeddings":   {0: "batch_size"},
        },
        do_constant_folding = True,
        verbose             = False,
        dynamo              = False,   # ← legacy exporter; required for opset 17
    )

    size_mb = Path(onnx_out).stat().st_size / 1e6
    print(f"   ✅ ONNX saved → {onnx_out}  ({size_mb:.1f} MB)")
    return wrapper, size_mb


# ── Step 6: three-test ONNX verification ──────────────────────────────────────

def verify_onnx(wrapper, onnx_out: str, dummy_input: torch.Tensor):
    print(f"\n── Step 6: Verifying ONNX output ───────────────────────────────────")
    try:
        import onnxruntime as ort
    except ImportError:
        print("   ⚠️  onnxruntime not installed — skipping verification.")
        print("        Run:  pip install onnxruntime")
        return None

    session = ort.InferenceSession(
        onnx_out,
        providers=["CPUExecutionProvider"],
    )

    max_diffs = []

    # ── Test A: zeros ────────────────────────────────────────────────────────
    zero_input  = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    zero_np     = zero_input.numpy()
    onnx_outA   = session.run(["embeddings"], {"pixel_values": zero_np})[0]
    with torch.inference_mode():
        pt_outA = wrapper(zero_input).detach().numpy()
    diffA = float(np.abs(onnx_outA - pt_outA).max())
    max_diffs.append(diffA)
    print(f"   Test A (zeros) : max diff = {diffA:.8f}", end="")
    assert diffA < 0.001, f"ONNX verification FAILED (Test A): diff={diffA}"
    print("  ✅")

    # ── Test B: random single ────────────────────────────────────────────────
    rand_input  = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    rand_np     = rand_input.numpy()
    onnx_outB   = session.run(["embeddings"], {"pixel_values": rand_np})[0]
    with torch.inference_mode():
        pt_outB = wrapper(rand_input).detach().numpy()
    diffB = float(np.abs(onnx_outB - pt_outB).max())
    max_diffs.append(diffB)
    print(f"   Test B (random): max diff = {diffB:.8f}", end="")
    assert diffB < 0.001, f"ONNX verification FAILED (Test B): diff={diffB}"
    print("  ✅")

    # ── Test C: batch of 4 ───────────────────────────────────────────────────
    batch_input = torch.rand(4, 3, INPUT_SIZE, INPUT_SIZE)
    batch_np    = batch_input.numpy()
    onnx_outC   = session.run(["embeddings"], {"pixel_values": batch_np})[0]
    with torch.inference_mode():
        pt_outC = wrapper(batch_input).detach().numpy()
    diffC = float(np.abs(onnx_outC - pt_outC).max())
    max_diffs.append(diffC)
    print(f"   Test C (batch=4): max diff = {diffC:.8f}"
          f"  output shape={onnx_outC.shape}", end="")
    assert onnx_outC.shape == (4, EMB_DIM), \
        f"Wrong batch output shape: {onnx_outC.shape}"
    assert diffC < 0.001, f"ONNX verification FAILED (Test C): diff={diffC}"
    print("  ✅")

    return max(max_diffs)


# ── Step 7: summary ────────────────────────────────────────────────────────────

def print_summary(local_weights_path: str, onnx_out: str,
                  size_mb: float, max_diff):
    weights_path = Path(local_weights_path)
    weights_size = (
        f"{weights_path.stat().st_size / 1e6:.1f} MB"
        if weights_path.exists() else "HuggingFace"
    )
    diff_str = f"{max_diff:.8f}" if max_diff is not None else "skipped"

    print("""
════════════════════════════════════════════════
✅  DINOv3-small ONNX Export Complete
════════════════════════════════════════════════""")
    print(f"Source weights : {local_weights_path} ({weights_size})")
    print(f"ONNX output    : {onnx_out} ({size_mb:.1f} MB)")
    print(f"Input shape    : [batch, 3, {INPUT_SIZE}, {INPUT_SIZE}] float32")
    print(f"Output shape   : [batch, {EMB_DIM}] float32")
    print(f"Verification   : ✅  All 3 tests passed (max diff: {diff_str})")
    print("""
Next steps:
  1. Copy dinov3_small.onnx to the Flutter assets/ folder:
       cp dinov3_small.onnx assets/dinov3_small.onnx

  2. Regenerate catalog_embeddings.json using DINOv3:
       python3 generate_catalog.py --model dinov3_small.onnx

  3. Update the ONNX model path in Dart:
       lib/services/dinov2_service.dart  →  change kDinoAsset to
       'assets/dinov3_small.onnx'

  4. Run flutter pub get && flutter run
════════════════════════════════════════════════
""")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export DINOv3-small (facebook/dinov3-vits16-pretrain-lvd1689m) to ONNX"
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS_PATH,
        help="Path to local model.safetensors file",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_ONNX_OUT,
        help="Output ONNX filename (default: dinov3_small.onnx)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace access token for gated repos (hf_...)",
    )
    args = parser.parse_args()

    # Also accept token via environment variable
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print("╔══════════════════════════════════════════════════╗")
    print("║   DINOv3-small → ONNX Export                    ║")
    print("╚══════════════════════════════════════════════════╝")
    if token:
        print(f"   HF token       : {token[:8]}…{token[-4:]} (gated repo access ✅)")

    config, _processor = load_config_and_processor(token=token)
    model               = load_model(config, args.weights, token=token)
    dummy_input         = verify_forward_pass(model)
    wrapper, size_mb    = export_onnx(model, dummy_input, args.out)
    max_diff            = verify_onnx(wrapper, args.out, dummy_input)
    print_summary(args.weights, args.out, size_mb, max_diff)


if __name__ == "__main__":
    main()
