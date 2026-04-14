import 'package:flutter/material.dart';

/// A single object detection from the YOLO model.
class DetectionResult {
  /// Bounding box in *image-pixel* coordinates (before any scaling).
  final Rect boundingBox;

  /// Class index (0-based).
  final int classIndex;

  /// Class label string (e.g. "sku_label_1").
  final String label;

  /// Confidence score in [0, 1].
  final double confidence;

  const DetectionResult({
    required this.boundingBox,
    required this.classIndex,
    required this.label,
    required this.confidence,
  });

  @override
  String toString() =>
      'DetectionResult(label=$label, conf=${confidence.toStringAsFixed(2)}, box=$boundingBox)';
}
