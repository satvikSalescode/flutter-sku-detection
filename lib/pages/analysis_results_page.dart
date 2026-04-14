import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

import '../services/vision_pipeline.dart';
import '../widgets/shelf_painter.dart';

// ── Page ──────────────────────────────────────────────────────────────────────

/// Displays the complete [ShelfAnalysis] produced by [VisionPipeline].
///
/// Shows:
///   • Annotated shelf image — green boxes for matched products,
///     red boxes for unknown / competitor products.
///   • Summary card — totals, shelf-share progress bar, timing.
///   • "Your products" section — one row per recognised SKU with
///     facing count, average similarity, and a thumbnail.
///   • "Unknown products" section — thumbnail grid for unmatched crops
///     with the nearest catalog match shown below each.
class AnalysisResultsPage extends StatefulWidget {
  final img.Image    shelfImage;
  final ShelfAnalysis analysis;

  const AnalysisResultsPage({
    super.key,
    required this.shelfImage,
    required this.analysis,
  });

  @override
  State<AnalysisResultsPage> createState() => _AnalysisResultsPageState();
}

class _AnalysisResultsPageState extends State<AnalysisResultsPage> {
  // ── Palette ──────────────────────────────────────────────────────────────────
  static const _teal  = Color(0xFF00C9A7);
  static const _red   = Color(0xFFFF5252);
  static const _card  = Color(0xFF1A1A2E);
  static const _bg    = Color(0xFF0F0F1A);
  static const _grey  = Color(0xFF252540);

  // ── Pre-encoded thumbnails ───────────────────────────────────────────────────

  // One JPEG thumbnail per matched product (first crop image).
  late final List<Uint8List> _matchedThumbs;

  // One JPEG thumbnail per unknown detection.
  late final List<Uint8List> _unknownThumbs;

  // Flattened annotation list for [ShelfPainter].
  late final List<BoxAnnotation> _annotations;

  @override
  void initState() {
    super.initState();
    _buildAnnotations();
    _encodeThumbnails();
  }

  void _buildAnnotations() {
    final list = <BoxAnnotation>[];

    for (final product in widget.analysis.products) {
      final pct = (product.avgSimilarity * 100).toStringAsFixed(0);
      final name = product.productName.replaceAll('_', ' ');
      for (final box in product.boundingBoxes) {
        list.add(BoxAnnotation(
          box:       box,
          isMatched: true,
          label:     '$name ($pct%)',
        ));
      }
    }

    for (final unknown in widget.analysis.unknowns) {
      list.add(BoxAnnotation(
        box:       unknown.boundingBox,
        isMatched: false,
        label:     'Unknown',
      ));
    }

    _annotations = list;
  }

  void _encodeThumbnails() {
    _matchedThumbs = widget.analysis.products.map((p) {
      // Use the first crop image as the representative thumbnail.
      return Uint8List.fromList(img.encodeJpg(p.cropImages.first, quality: 75));
    }).toList();

    _unknownThumbs = widget.analysis.unknowns.map((u) {
      return Uint8List.fromList(img.encodeJpg(u.cropImage, quality: 75));
    }).toList();
  }

  // ── Build ────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final a = widget.analysis;

    return Scaffold(
      backgroundColor: _bg,
      appBar: _buildAppBar(a),
      body: ListView(
        padding: const EdgeInsets.only(bottom: 40),
        children: [
          // ── Annotated image ──────────────────────────────────────────────────
          _AnnotatedShelf(
            shelfImage:  widget.shelfImage,
            annotations: _annotations,
          ),

          const SizedBox(height: 16),

          // ── Summary card ─────────────────────────────────────────────────────
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: _SummaryCard(analysis: a),
          ),

          const SizedBox(height: 20),

          // ── Your products ────────────────────────────────────────────────────
          if (a.products.isNotEmpty) ...[
            _SectionHeader(
              icon:  Icons.inventory_2_outlined,
              label: 'Your Products',
              color: _teal,
              count: a.products.length,
            ),
            ...List.generate(a.products.length, (i) => Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
              child: _ProductRow(
                product:   a.products[i],
                thumbnail: _matchedThumbs[i],
              ),
            )),
          ],

          // ── Unknown products ─────────────────────────────────────────────────
          if (a.unknowns.isNotEmpty) ...[
            const SizedBox(height: 8),
            _SectionHeader(
              icon:  Icons.help_outline,
              label: 'Unknown / Competitor',
              color: _red,
              count: a.unknowns.length,
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: GridView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount:   3,
                  crossAxisSpacing: 8,
                  mainAxisSpacing:  8,
                  childAspectRatio: 0.78,
                ),
                itemCount: a.unknowns.length,
                itemBuilder: (_, i) => _UnknownCard(
                  unknown:   a.unknowns[i],
                  thumbnail: _unknownThumbs[i],
                ),
              ),
            ),
          ],

          // ── Empty state ──────────────────────────────────────────────────────
          if (a.products.isEmpty && a.unknowns.isEmpty)
            const Padding(
              padding: EdgeInsets.all(40),
              child: Center(
                child: Text(
                  'No products detected.\nTry a clearer shelf photo.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.white38, fontSize: 14),
                ),
              ),
            ),
        ],
      ),
    );
  }

  AppBar _buildAppBar(ShelfAnalysis a) {
    return AppBar(
      backgroundColor: _card,
      leading: IconButton(
        icon: const Icon(Icons.arrow_back, color: Colors.white),
        onPressed: () => Navigator.of(context).pop(),
      ),
      title: Text(
        '${a.totalDetections} product${a.totalDetections == 1 ? '' : 's'} detected',
        style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
      ),
      actions: [
        Padding(
          padding: const EdgeInsets.only(right: 12),
          child: Center(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color:        _teal.withOpacity(0.15),
                borderRadius: BorderRadius.circular(20),
                border:       Border.all(color: _teal.withOpacity(0.35)),
              ),
              child: Text(
                '${a.timing.totalMs}ms',
                style: const TextStyle(
                    color: _teal, fontSize: 11, fontWeight: FontWeight.bold),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

// ── Annotated shelf image panel ────────────────────────────────────────────────

class _AnnotatedShelf extends StatelessWidget {
  final img.Image          shelfImage;
  final List<BoxAnnotation> annotations;

  const _AnnotatedShelf({
    required this.shelfImage,
    required this.annotations,
  });

  @override
  Widget build(BuildContext context) {
    final jpegBytes = Uint8List.fromList(
      img.encodeJpg(shelfImage, quality: 88),
    );
    final imageSize = Size(
      shelfImage.width.toDouble(),
      shelfImage.height.toDouble(),
    );

    return Container(
      height: 300,
      color:  Colors.black,
      child: LayoutBuilder(
        builder: (_, constraints) {
          final displaySize = Size(constraints.maxWidth, constraints.maxHeight);
          return Stack(
            fit: StackFit.expand,
            children: [
              Image.memory(jpegBytes, fit: BoxFit.contain),
              if (annotations.isNotEmpty)
                CustomPaint(
                  painter: ShelfPainter(
                    annotations: annotations,
                    imageSize:   imageSize,
                    displaySize: displaySize,
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}

// ── Summary card ──────────────────────────────────────────────────────────────

class _SummaryCard extends StatelessWidget {
  final ShelfAnalysis analysis;

  static const _teal = Color(0xFF00C9A7);
  static const _red  = Color(0xFFFF5252);
  static const _card = Color(0xFF1A1A2E);

  const _SummaryCard({required this.analysis});

  @override
  Widget build(BuildContext context) {
    final a       = analysis;
    final share   = a.shelfSharePercent;
    final t       = a.timing;

    return Container(
      padding:    const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color:        _card,
        borderRadius: BorderRadius.circular(14),
        border:       Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Total ────────────────────────────────────────────────────────────
          _Row(
            icon:  Icons.grid_view_rounded,
            color: Colors.white70,
            label: 'Total products detected',
            value: '${a.totalDetections}',
          ),
          const Divider(color: Colors.white10, height: 20),

          // ── Your products + progress bar ─────────────────────────────────────
          _Row(
            icon:  Icons.check_circle_outline,
            color: _teal,
            label: 'Your products',
            value: '${a.matchedCount}  (${share.toStringAsFixed(1)}%)',
            valueColor: _teal,
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value:            (share / 100).clamp(0.0, 1.0),
              backgroundColor:  Colors.white10,
              valueColor:       const AlwaysStoppedAnimation(_teal),
              minHeight:        6,
            ),
          ),
          const SizedBox(height: 14),

          // ── Unknown ──────────────────────────────────────────────────────────
          _Row(
            icon:  Icons.help_outline,
            color: _red,
            label: 'Unknown / competitor',
            value: '${a.unknownCount}',
            valueColor: _red,
          ),
          const Divider(color: Colors.white10, height: 20),

          // ── Timing ──────────────────────────────────────────────────────────
          Row(
            children: [
              const Icon(Icons.timer_outlined, size: 14, color: Colors.white38),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  'Processed in ${t.totalMs}ms  '
                  '(YOLO: ${t.yoloMs}ms  '
                  'Embedding: ${t.embeddingMs}ms  '
                  'Match: ${t.matchingMs}ms)',
                  style: const TextStyle(
                      color: Colors.white38, fontSize: 11, height: 1.4),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _Row extends StatelessWidget {
  final IconData icon;
  final Color    color;
  final String   label;
  final String   value;
  final Color    valueColor;

  const _Row({
    required this.icon,
    required this.color,
    required this.label,
    required this.value,
    this.valueColor = Colors.white,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(icon, size: 15, color: color),
        const SizedBox(width: 8),
        Expanded(
          child: Text(label,
              style: const TextStyle(color: Colors.white60, fontSize: 13)),
        ),
        Text(value,
            style: TextStyle(
                color:      valueColor,
                fontSize:   13,
                fontWeight: FontWeight.bold)),
      ],
    );
  }
}

// ── Section header ────────────────────────────────────────────────────────────

class _SectionHeader extends StatelessWidget {
  final IconData icon;
  final String   label;
  final Color    color;
  final int      count;

  const _SectionHeader({
    required this.icon,
    required this.label,
    required this.color,
    required this.count,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 10),
      child: Row(
        children: [
          Icon(icon, size: 15, color: color),
          const SizedBox(width: 6),
          Text(
            label.toUpperCase(),
            style: TextStyle(
                color:          color,
                fontSize:       11,
                fontWeight:     FontWeight.bold,
                letterSpacing:  1.1),
          ),
          const SizedBox(width: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
            decoration: BoxDecoration(
              color:        color.withOpacity(0.15),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Text('$count',
                style: TextStyle(
                    color: color, fontSize: 11, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
  }
}

// ── Matched product row ───────────────────────────────────────────────────────

class _ProductRow extends StatelessWidget {
  final DetectedProduct product;
  final Uint8List       thumbnail;

  static const _teal = Color(0xFF00C9A7);
  static const _card = Color(0xFF1A1A2E);
  static const _grey = Color(0xFF252540);

  const _ProductRow({required this.product, required this.thumbnail});

  @override
  Widget build(BuildContext context) {
    final name  = product.productName.replaceAll('_', ' ');
    final score = (product.avgSimilarity * 100).toStringAsFixed(1);

    return Container(
      decoration: BoxDecoration(
        color:        _card,
        borderRadius: BorderRadius.circular(12),
        border:       Border.all(color: _teal.withOpacity(0.2)),
      ),
      child: Row(
        children: [
          // Thumbnail
          ClipRRect(
            borderRadius: const BorderRadius.only(
              topLeft:    Radius.circular(11),
              bottomLeft: Radius.circular(11),
            ),
            child: Image.memory(
              thumbnail,
              width:  64,
              height: 64,
              fit:    BoxFit.cover,
            ),
          ),

          const SizedBox(width: 12),

          // Name + facings
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(name,
                    style: const TextStyle(
                        color:      Colors.white,
                        fontSize:   13,
                        fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Row(
                  children: [
                    _Chip(
                      label: '${product.facingCount} facing${product.facingCount == 1 ? '' : 's'}',
                      color: _teal.withOpacity(0.15),
                      textColor: _teal,
                    ),
                    const SizedBox(width: 6),
                    _Chip(
                      label: '$score% avg',
                      color: Colors.white.withOpacity(0.07),
                      textColor: Colors.white54,
                    ),
                  ],
                ),
              ],
            ),
          ),

          // Score bar (vertical)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Column(
              children: [
                Text(
                  '$score%',
                  style: const TextStyle(
                      color:      _teal,
                      fontSize:   12,
                      fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 4),
                SizedBox(
                  width:  4,
                  height: 32,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(2),
                    child: RotatedBox(
                      quarterTurns: 3,
                      child: LinearProgressIndicator(
                        value:           (product.avgSimilarity).clamp(0.0, 1.0),
                        backgroundColor: Colors.white10,
                        valueColor:      const AlwaysStoppedAnimation(_teal),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  final String label;
  final Color  color;
  final Color  textColor;

  const _Chip({
    required this.label,
    required this.color,
    required this.textColor,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
      decoration: BoxDecoration(
        color:        color,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(label,
          style: TextStyle(
              color:      textColor,
              fontSize:   10,
              fontWeight: FontWeight.w600)),
    );
  }
}

// ── Unknown detection card ────────────────────────────────────────────────────

class _UnknownCard extends StatelessWidget {
  final UnknownDetection unknown;
  final Uint8List        thumbnail;

  static const _red  = Color(0xFFFF5252);
  static const _card = Color(0xFF1A1A2E);

  const _UnknownCard({required this.unknown, required this.thumbnail});

  @override
  Widget build(BuildContext context) {
    final nearest = unknown.nearestProduct.replaceAll('_', ' ');
    final score   = (unknown.bestScore * 100).toStringAsFixed(0);

    return Container(
      decoration: BoxDecoration(
        color:        _card,
        borderRadius: BorderRadius.circular(10),
        border:       Border.all(color: _red.withOpacity(0.3)),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Thumbnail (fills top portion)
          Expanded(
            child: Image.memory(thumbnail, fit: BoxFit.cover),
          ),

          // Footer
          Padding(
            padding: const EdgeInsets.fromLTRB(5, 5, 5, 6),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  nearest,
                  maxLines:  1,
                  overflow:  TextOverflow.ellipsis,
                  style: const TextStyle(
                      color:      Colors.white60,
                      fontSize:   9,
                      fontWeight: FontWeight.w500),
                ),
                const SizedBox(height: 2),
                Text(
                  '$score%',
                  style: const TextStyle(
                      color:      _red,
                      fontSize:   10,
                      fontWeight: FontWeight.bold),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
