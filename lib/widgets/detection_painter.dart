import 'package:flutter/material.dart';
import '../models/detection_result.dart';

/// Paints YOLO bounding boxes on top of the displayed image.
/// Each class index gets a consistent palette colour.
/// Label shows the YOLO class name and confidence percentage.
class DetectionPainter extends CustomPainter {
  final List<DetectionResult> detections;
  final Size imageSize;
  final Size displaySize;

  const DetectionPainter({
    required this.detections,
    required this.imageSize,
    required this.displaySize,
  });

  static const List<Color> _palette = [
    Color(0xFFFF6B6B), Color(0xFF4ECDC4), Color(0xFF45B7D1),
    Color(0xFF96CEB4), Color(0xFFFFEEA9), Color(0xFF88D8B0),
    Color(0xFFFF8B94), Color(0xFFA8E6CF),
  ];

  @override
  void paint(Canvas canvas, Size size) {
    // Scale factors from image pixels → display pixels
    final sx = displaySize.width  / imageSize.width;
    final sy = displaySize.height / imageSize.height;

    for (final det in detections) {
      final color = _palette[det.classIndex % _palette.length];
      final label = '${det.label}  ${(det.confidence * 100).toStringAsFixed(0)}%';
      _drawBox(canvas, det.boundingBox, label, color, sx, sy);
    }
  }

  void _drawBox(Canvas canvas, Rect box, String label, Color color,
      double sx, double sy) {
    final rect = Rect.fromLTRB(
        box.left * sx, box.top * sy, box.right * sx, box.bottom * sy);

    // Stroke
    canvas.drawRect(rect,
        Paint()
          ..color       = color
          ..style       = PaintingStyle.stroke
          ..strokeWidth = 2.5);

    // Translucent fill
    canvas.drawRect(rect,
        Paint()
          ..color = color.withOpacity(0.08)
          ..style = PaintingStyle.fill);

    // Label background + text
    final tp = TextPainter(
      text: TextSpan(
        text: label,
        style: const TextStyle(
          color:      Colors.white,
          fontSize:   11,
          fontWeight: FontWeight.w600,
          shadows:    [Shadow(color: Colors.black, blurRadius: 2)],
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: displaySize.width);

    final lr = Rect.fromLTWH(
        rect.left, rect.top - tp.height - 4, tp.width + 8, tp.height + 4);
    canvas.drawRect(lr, Paint()..color = color.withOpacity(0.85));
    tp.paint(canvas, Offset(lr.left + 4, lr.top + 2));
  }

  @override
  bool shouldRepaint(DetectionPainter old) =>
      old.detections   != detections  ||
      old.displaySize  != displaySize;
}
