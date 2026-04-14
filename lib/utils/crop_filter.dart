import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../models/detection_result.dart';

/// Filters raw YOLO detections before crop extraction.
/// Steps:
///   1. Drop detections below confidence threshold
///   2. Drop crops smaller than minSize × minSize pixels
///   3. NMS: remove overlapping boxes (IoU > iouThreshold), keep highest confidence
class CropFilter {
  final double confThreshold;
  final int    minSize;
  final double iouThreshold;

  const CropFilter({
    this.confThreshold = 0.30,
    this.minSize       = 32,
    this.iouThreshold  = 0.80,
  });

  List<DetectionResult> filter(List<DetectionResult> detections) {
    // 1. Confidence + size filter
    var kept = detections.where((d) {
      if (d.confidence < confThreshold) return false;
      final w = d.boundingBox.width;
      final h = d.boundingBox.height;
      if (w < minSize || h < minSize) return false;
      return true;
    }).toList();

    // 2. Sort descending by confidence
    kept.sort((a, b) => b.confidence.compareTo(a.confidence));

    // 3. Greedy NMS
    final suppressed = List<bool>.filled(kept.length, false);
    for (int i = 0; i < kept.length; i++) {
      if (suppressed[i]) continue;
      for (int j = i + 1; j < kept.length; j++) {
        if (suppressed[j]) continue;
        if (_iou(kept[i].boundingBox, kept[j].boundingBox) > iouThreshold) {
          suppressed[j] = true;
        }
      }
    }

    return [
      for (int i = 0; i < kept.length; i++)
        if (!suppressed[i]) kept[i],
    ];
  }

  static double _iou(Rect a, Rect b) {
    final interL = math.max(a.left,   b.left);
    final interT = math.max(a.top,    b.top);
    final interR = math.min(a.right,  b.right);
    final interB = math.min(a.bottom, b.bottom);
    if (interR <= interL || interB <= interT) return 0;
    final inter = (interR - interL) * (interB - interT);
    final union = a.width * a.height + b.width * b.height - inter;
    return union > 0 ? inter / union : 0;
  }
}
