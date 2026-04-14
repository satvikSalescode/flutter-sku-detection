import 'package:flutter/material.dart';

// ── Box annotation ────────────────────────────────────────────────────────────

/// A single annotated detection to paint on the shelf image.
class BoxAnnotation {
  /// Bounding box in original image-pixel coordinates.
  final Rect box;

  /// True → green (matched), false → red (unknown/competitor).
  final bool isMatched;

  /// Short label painted above the box (e.g. "Coke_Can_Red 87%" or "Unknown").
  final String label;

  const BoxAnnotation({
    required this.box,
    required this.isMatched,
    required this.label,
  });
}

// ── Painter ───────────────────────────────────────────────────────────────────

/// Paints coloured bounding boxes and labels over a shelf image.
///
/// Green  (#00C9A7) = matched catalog products
/// Red    (#FF5252) = unknown / competitor products
///
/// Image coordinates are scaled to fit [displaySize] while preserving aspect
/// ratio (BoxFit.contain) — the same transform Flutter applies to Image.memory.
class ShelfPainter extends CustomPainter {
  final List<BoxAnnotation> annotations;
  final Size imageSize;
  final Size displaySize;

  static const _green = Color(0xFF00C9A7);
  static const _red   = Color(0xFFFF5252);

  const ShelfPainter({
    required this.annotations,
    required this.imageSize,
    required this.displaySize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (annotations.isEmpty) return;

    // Compute the BoxFit.contain scale + offset so boxes align with
    // the image rendered by Image.memory / Image.file.
    final scaleX = displaySize.width  / imageSize.width;
    final scaleY = displaySize.height / imageSize.height;
    final scale  = scaleX < scaleY ? scaleX : scaleY; // min → letterbox

    final offsetX = (displaySize.width  - imageSize.width  * scale) / 2;
    final offsetY = (displaySize.height - imageSize.height * scale) / 2;

    for (final ann in annotations) {
      final color = ann.isMatched ? _green : _red;
      _drawBox(canvas, ann.box, ann.label, color, scale, offsetX, offsetY,
          showLabel: ann.isMatched);
    }
  }

  void _drawBox(
    Canvas canvas,
    Rect imgBox,
    String label,
    Color color,
    double scale,
    double dx,
    double dy, {
    bool showLabel = true,
  }) {
    // Map from image pixels → display pixels
    final rect = Rect.fromLTRB(
      imgBox.left   * scale + dx,
      imgBox.top    * scale + dy,
      imgBox.right  * scale + dx,
      imgBox.bottom * scale + dy,
    );

    // Stroke border
    canvas.drawRect(
      rect,
      Paint()
        ..color       = color
        ..style       = PaintingStyle.stroke
        ..strokeWidth = 2.0,
    );

    // Translucent fill
    canvas.drawRect(
      rect,
      Paint()
        ..color = color.withOpacity(0.08)
        ..style = PaintingStyle.fill,
    );

    // Label pill — only for matched products
    if (!showLabel) return;

    final tp = TextPainter(
      text: TextSpan(
        text: label,
        style: const TextStyle(
          color:      Colors.white,
          fontSize:   10,
          fontWeight: FontWeight.w700,
          height:     1.2,
          shadows: [Shadow(color: Colors.black87, blurRadius: 3)],
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: displaySize.width * 0.6);

    // Pill — clamped so it never overflows left edge
    final pillH = tp.height + 6;
    final pillW = tp.width  + 10;
    final pillL = rect.left.clamp(0.0, displaySize.width - pillW);
    final pillT = (rect.top - pillH).clamp(0.0, displaySize.height - pillH);

    final pillRect = RRect.fromRectAndRadius(
      Rect.fromLTWH(pillL, pillT, pillW, pillH),
      const Radius.circular(4),
    );
    canvas.drawRRect(pillRect, Paint()..color = color.withOpacity(0.88));
    tp.paint(canvas, Offset(pillL + 5, pillT + 3));
  }

  @override
  bool shouldRepaint(ShelfPainter old) =>
      old.annotations != annotations || old.displaySize != displaySize;
}
