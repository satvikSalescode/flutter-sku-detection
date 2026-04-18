"""
convert_model.py
================
Exports your YOLOv8 best.pt model to ONNX (opset 19) for use in the Flutter app.
No TFLite conversion needed — the app uses ONNX Runtime directly.

Requirements:
    pip install ultralytics

Usage:
    python3 convert_model.py
"""

import argparse
import json
import sys
from pathlib import Path


def convert(model_path: str, imgsz: int):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌  ultralytics not installed. Run:  pip install ultralytics")
        sys.exit(1)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌  Model file not found: {model_path}")
        sys.exit(1)

    print(f"✅  Loading model: {model_path}")
    model = YOLO(str(model_path))

    # opset=17 — best compatibility with onnxruntime 1.4.1 on Android
    # (opset 17 introduced LayerNormalization needed by attention layers in YOLO26m)
    print(f"⚙️   Exporting to ONNX opset 17 (imgsz={imgsz}) …")
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        opset=17,
    )
    print(f"✅  ONNX saved → {onnx_path}")

    # Print metadata
    print("\n── Model metadata ─────────────────────────────────────────────")
    class_names = list(model.names.values()) if hasattr(model, "names") else []
    num_classes  = len(class_names)
    print(f"   Number of classes : {num_classes}")
    print(f"   Class names       : {class_names}")
    print(f"   Input size        : {imgsz}×{imgsz}")

    # Write labels file
    labels_path = Path("assets/labels.txt")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("\n".join(class_names))
    print(f"   Labels written to : {labels_path}")

    meta = {"num_classes": num_classes, "class_names": class_names, "input_size": imgsz}
    Path("model_meta.json").write_text(json.dumps(meta, indent=2))

    print("───────────────────────────────────────────────────────────────")
    print(f"\n📲  Next step — copy the ONNX file to assets/:")
    print(f'   cp "{onnx_path}" assets/\n')


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 .pt → ONNX opset 19")
    parser.add_argument(
        "--model",
        default="/Users/satvikchaudhary/Desktop/IRED DOCS/All_Epochs/epoch80_stripped.pt",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    convert(args.model, args.imgsz)


if __name__ == "__main__":
    main()
