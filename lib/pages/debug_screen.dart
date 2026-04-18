import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'dart:io';

import '../services/vision_pipeline.dart';
import '../services/dinov3_service.dart';
import '../services/catalog_service.dart';

// ── Palette (matches app theme) ───────────────────────────────────────────────
const _bg    = Color(0xFF0F0F1A);
const _card  = Color(0xFF1A1A2E);
const _teal  = Color(0xFF00C9A7);
const _red   = Color(0xFFFF5252);
const _amber = Color(0xFFFFB300);
const _dim   = Color(0xFF8888AA);

// ── ONNX model identity ───────────────────────────────────────────────────────
// Derived from the asset constant — no runtime reflection needed.
const _onnxFilename    = 'dinov3_small.onnx';
const _onnxModelName   = 'dinov3-small';

/// Development/debug screen for verifying the DINOv3 migration.
/// Accessible from Settings → "Debug / Verification".
class DebugScreen extends StatefulWidget {
  final VisionPipeline pipeline;

  const DebugScreen({super.key, required this.pipeline});

  @override
  State<DebugScreen> createState() => _DebugScreenState();
}

class _DebugScreenState extends State<DebugScreen> {
  // ── Self-test state ──────────────────────────────────────────────────────────
  bool   _selfTestRunning = false;
  bool?  _selfTestPassed;

  // ── Quick match state ─────────────────────────────────────────────────────────
  bool           _matchRunning = false;
  _MatchResult?  _matchResult;

  // ── Catalog health state ──────────────────────────────────────────────────────
  bool                        _healthRunning  = false;
  List<_ProductHealth>?       _healthResults;

  // ── Helpers ───────────────────────────────────────────────────────────────────

  CatalogService  get _catalog => widget.pipeline.catalogService;
  DinoV3Service   get _dino    => widget.pipeline.dinov3Service;

  bool get _catalogMismatch {
    final cat = _catalog.catalogModel.toLowerCase();
    return cat.isNotEmpty && !cat.contains('dinov3');
  }

  /// Cosine similarity of two L2-normalised vectors.
  static double _cosine(List<double> a, List<double> b) {
    double dot = 0;
    for (int i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot;
  }

  // ── Actions ───────────────────────────────────────────────────────────────────

  Future<void> _runSelfTest() async {
    setState(() { _selfTestRunning = true; _selfTestPassed = null; });
    final passed = await _dino.selfTest();
    if (mounted) setState(() { _selfTestRunning = false; _selfTestPassed = passed; });
  }

  Future<void> _runQuickMatch() async {
    setState(() { _matchRunning = true; _matchResult = null; });

    try {
      final xf = await ImagePicker().pickImage(
        source: ImageSource.camera,
        imageQuality: 90,
      );
      if (xf == null) { setState(() => _matchRunning = false); return; }

      final bytes   = await File(xf.path).readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) throw Exception('Could not decode image');

      final analysis = await widget.pipeline.analyzeShelf(decoded);

      // Top-3 matches by avg similarity
      final top3 = [...analysis.products]
        ..sort((a, b) => b.avgSimilarity.compareTo(a.avgSimilarity));

      if (mounted) {
        setState(() {
          _matchRunning = false;
          _matchResult  = _MatchResult(
            yoloDetections:  analysis.totalDetections,
            embeddingsCount: analysis.totalDetections,
            matchesFound:    analysis.matchedCount,
            top3:            top3.take(3).toList(),
            timing:          analysis.timing,
          );
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _matchRunning = false;
          _matchResult  = _MatchResult.error(e.toString());
        });
      }
    }
  }

  Future<void> _runCatalogHealth() async {
    setState(() { _healthRunning = true; _healthResults = null; });

    final results = <_ProductHealth>[];

    for (final product in _catalog.products) {
      final embs   = product.embeddings;
      final angles = product.angles;

      if (embs.length < 2) {
        results.add(_ProductHealth(
          name:          product.name,
          angleCount:    embs.length,
          similarity:    null,
          comparedAngles: angles.isNotEmpty ? [angles[0], '—'] : ['—', '—'],
        ));
        continue;
      }

      // Pick front vs back if available; otherwise first two angles
      int idxA = product.angleIndex('front');
      int idxB = product.angleIndex('back');
      if (idxA < 0) idxA = 0;
      if (idxB < 0) idxB = 1;
      if (idxA == idxB) idxB = idxA == 0 ? 1 : 0;

      final sim = _cosine(embs[idxA], embs[idxB]);
      results.add(_ProductHealth(
        name:           product.name,
        angleCount:     embs.length,
        similarity:     sim,
        comparedAngles: [angles[idxA], angles[idxB]],
      ));
    }

    if (mounted) setState(() { _healthRunning = false; _healthResults = results; });
  }

  // ── Build ─────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bg,
      appBar: AppBar(
        backgroundColor: _card,
        title: const Text('Debug / Verification',
            style: TextStyle(color: Colors.white, fontSize: 16)),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _buildModelInfo(),
          const SizedBox(height: 16),
          _buildSelfTest(),
          const SizedBox(height: 16),
          _buildQuickMatch(),
          const SizedBox(height: 16),
          _buildCatalogHealth(),
          const SizedBox(height: 32),
        ],
      ),
    );
  }

  // ── Section: Model Info ───────────────────────────────────────────────────────

  Widget _buildModelInfo() {
    final cat          = _catalog;
    final catModel     = cat.catalogModel.isEmpty ? 'unknown' : cat.catalogModel;
    final mismatch     = _catalogMismatch;

    return _Section(
      title: 'Model Info',
      icon:  Icons.memory_rounded,
      children: [
        _InfoRow('ONNX file',      _onnxFilename),
        _InfoRow('ONNX model',     _onnxModelName),
        const Divider(color: Colors.white12, height: 24),
        _InfoRow('Catalog model',  catModel,
            valueColor: mismatch ? _red : _teal),
        _InfoRow('Products',       '${cat.products.length}'),
        _InfoRow('Embeddings',     '${cat.numAngles}'),
        _InfoRow('Embedding dim',  '${cat.embeddingDim}'),
        _InfoRow('Version',        cat.catalogVersion.isEmpty
            ? '—' : cat.catalogVersion),
        if (mismatch) ...[
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color:        _red.withOpacity(0.12),
              border:       Border.all(color: _red.withOpacity(0.6)),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              children: [
                const Icon(Icons.warning_amber_rounded, color: _red, size: 18),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'MISMATCH: Catalog was generated with "$catModel" '
                    'but ONNX model is "$_onnxModelName".\n'
                    'Regenerate catalog using generate_catalog.py.',
                    style: const TextStyle(color: _red, fontSize: 12, height: 1.4),
                  ),
                ),
              ],
            ),
          ),
        ] else ...[
          const SizedBox(height: 8),
          Row(
            children: const [
              Icon(Icons.check_circle_rounded, color: _teal, size: 16),
              SizedBox(width: 6),
              Text('Catalog and ONNX model are compatible',
                  style: TextStyle(color: _teal, fontSize: 12)),
            ],
          ),
        ],
      ],
    );
  }

  // ── Section: Self-test ────────────────────────────────────────────────────────

  Widget _buildSelfTest() {
    return _Section(
      title: 'Self-Test',
      icon:  Icons.science_rounded,
      children: [
        const Text(
          'Runs a checkerboard image through DINOv3 twice and verifies '
          'the output is a valid, deterministic 384-dim L2-normalised vector.',
          style: TextStyle(color: _dim, fontSize: 12, height: 1.5),
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            ElevatedButton.icon(
              onPressed: _selfTestRunning ? null : _runSelfTest,
              icon: _selfTestRunning
                  ? const SizedBox(
                      width: 14, height: 14,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.white))
                  : const Icon(Icons.play_arrow_rounded, size: 18),
              label: const Text('Run self-test'),
              style: ElevatedButton.styleFrom(
                backgroundColor: _teal,
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              ),
            ),
            const SizedBox(width: 16),
            if (_selfTestPassed != null)
              Row(
                children: [
                  Icon(
                    _selfTestPassed! ? Icons.check_circle_rounded
                                     : Icons.cancel_rounded,
                    color: _selfTestPassed! ? _teal : _red,
                    size: 20,
                  ),
                  const SizedBox(width: 6),
                  Text(
                    _selfTestPassed! ? 'PASSED ✅' : 'FAILED ❌',
                    style: TextStyle(
                      color:      _selfTestPassed! ? _teal : _red,
                      fontWeight: FontWeight.bold,
                      fontSize:   14,
                    ),
                  ),
                ],
              ),
          ],
        ),
      ],
    );
  }

  // ── Section: Quick Match Test ─────────────────────────────────────────────────

  Widget _buildQuickMatch() {
    final r = _matchResult;
    return _Section(
      title: 'Quick Match Test',
      icon:  Icons.camera_alt_rounded,
      children: [
        const Text(
          'Captures one photo, runs the full YOLO → DINOv3 → catalog pipeline '
          'and shows detailed results.',
          style: TextStyle(color: _dim, fontSize: 12, height: 1.5),
        ),
        const SizedBox(height: 12),
        ElevatedButton.icon(
          onPressed: _matchRunning ? null : _runQuickMatch,
          icon: _matchRunning
              ? const SizedBox(
                  width: 14, height: 14,
                  child: CircularProgressIndicator(
                      strokeWidth: 2, color: Colors.white))
              : const Icon(Icons.camera_alt_rounded, size: 18),
          label: const Text('Test with camera'),
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF3D4DFF),
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          ),
        ),
        if (r != null) ...[
          const SizedBox(height: 16),
          if (r.error != null)
            Text('Error: ${r.error}',
                style: const TextStyle(color: _red, fontSize: 12))
          else ...[
            // Summary row
            _MetricGrid(metrics: [
              _Metric('YOLO detections', '${r.yoloDetections}'),
              _Metric('Embeddings',      '${r.embeddingsCount}'),
              _Metric('Matches',         '${r.matchesFound}'),
            ]),
            const SizedBox(height: 12),
            // Timing
            const Text('Pipeline timing',
                style: TextStyle(color: _dim, fontSize: 11,
                    fontWeight: FontWeight.w600)),
            const SizedBox(height: 4),
            _MetricGrid(metrics: [
              _Metric('YOLO',       '${r.timing.yoloMs} ms'),
              _Metric('Embed',      '${r.timing.embeddingMs} ms'),
              _Metric('Match',      '${r.timing.matchingMs} ms'),
              _Metric('Total',      '${r.timing.totalMs} ms'),
            ]),
            // Top-3 matches
            if (r.top3.isNotEmpty) ...[
              const SizedBox(height: 12),
              const Text('Top matches',
                  style: TextStyle(color: _dim, fontSize: 11,
                      fontWeight: FontWeight.w600)),
              const SizedBox(height: 6),
              ...r.top3.asMap().entries.map((e) {
                final rank = e.key + 1;
                final p    = e.value;
                final pct  = (p.avgSimilarity * 100).toStringAsFixed(1);
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 3),
                  child: Row(
                    children: [
                      Text('#$rank ',
                          style: const TextStyle(color: _teal,
                              fontWeight: FontWeight.bold, fontSize: 13)),
                      Expanded(
                        child: Text(p.productName,
                            style: const TextStyle(
                                color: Colors.white, fontSize: 13)),
                      ),
                      Text('$pct%  ×${p.facingCount}',
                          style: const TextStyle(color: _dim, fontSize: 12)),
                    ],
                  ),
                );
              }),
            ],
          ],
        ],
      ],
    );
  }

  // ── Section: Catalog Health ───────────────────────────────────────────────────

  Widget _buildCatalogHealth() {
    return _Section(
      title: 'Catalog Health',
      icon:  Icons.health_and_safety_rounded,
      children: [
        const Text(
          'Computes cosine similarity between the front and back reference '
          'embeddings for each product. Score > 0.5 = healthy.',
          style: TextStyle(color: _dim, fontSize: 12, height: 1.5),
        ),
        const SizedBox(height: 12),
        ElevatedButton.icon(
          onPressed: _healthRunning ? null : _runCatalogHealth,
          icon: _healthRunning
              ? const SizedBox(
                  width: 14, height: 14,
                  child: CircularProgressIndicator(
                      strokeWidth: 2, color: Colors.white))
              : const Icon(Icons.bar_chart_rounded, size: 18),
          label: const Text('Check catalog health'),
          style: ElevatedButton.styleFrom(
            backgroundColor: _amber,
            foregroundColor: Colors.black,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          ),
        ),
        if (_healthResults != null) ...[
          const SizedBox(height: 16),
          ..._healthResults!.map((h) => _buildHealthRow(h)),
        ],
      ],
    );
  }

  Widget _buildHealthRow(_ProductHealth h) {
    final sim     = h.similarity;
    final noData  = sim == null;
    final weak    = !noData && sim < 0.3;
    final ok      = !noData && sim >= 0.5;

    final Color color = noData ? _dim : (ok ? _teal : (weak ? _red : _amber));
    final String simStr = noData
        ? 'only ${h.angleCount} angle${h.angleCount == 1 ? "" : "s"} — skip'
        : sim.toStringAsFixed(3);

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(h.name,
                    style: const TextStyle(
                        color: Colors.white, fontSize: 13,
                        fontWeight: FontWeight.w600)),
              ),
              Text('${h.angleCount} angles',
                  style: const TextStyle(color: _dim, fontSize: 11)),
            ],
          ),
          const SizedBox(height: 4),
          Row(
            children: [
              // Similarity bar
              Expanded(
                child: Stack(
                  children: [
                    Container(
                      height: 6,
                      decoration: BoxDecoration(
                        color:        Colors.white12,
                        borderRadius: BorderRadius.circular(3),
                      ),
                    ),
                    if (!noData)
                      FractionallySizedBox(
                        widthFactor: sim.clamp(0.0, 1.0),
                        child: Container(
                          height: 6,
                          decoration: BoxDecoration(
                            color:        color,
                            borderRadius: BorderRadius.circular(3),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              const SizedBox(width: 10),
              Text(simStr,
                  style: TextStyle(
                      color: color, fontSize: 12,
                      fontWeight: FontWeight.w600)),
            ],
          ),
          if (!noData) ...[
            const SizedBox(height: 2),
            Text(
              '${h.comparedAngles[0]} ↔ ${h.comparedAngles[1]}  '
              '${weak ? "⚠️  Weak reference — consider re-uploading images" : ""}',
              style: TextStyle(
                  color: weak ? _red : _dim, fontSize: 10),
            ),
          ],
          const SizedBox(height: 4),
          const Divider(color: Colors.white10, height: 1),
        ],
      ),
    );
  }
}

// ── Reusable widgets ──────────────────────────────────────────────────────────

class _Section extends StatelessWidget {
  final String        title;
  final IconData      icon;
  final List<Widget>  children;

  const _Section({
    required this.title,
    required this.icon,
    required this.children,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color:        _card,
        borderRadius: BorderRadius.circular(12),
        border:       Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 10),
            child: Row(
              children: [
                Icon(icon, color: _teal, size: 18),
                const SizedBox(width: 8),
                Text(title.toUpperCase(),
                    style: const TextStyle(
                        color:          _teal,
                        fontSize:       11,
                        fontWeight:     FontWeight.w800,
                        letterSpacing:  1.2)),
              ],
            ),
          ),
          const Divider(color: Colors.white10, height: 1),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: children,
            ),
          ),
        ],
      ),
    );
  }
}

class _InfoRow extends StatelessWidget {
  final String  label;
  final String  value;
  final Color?  valueColor;

  const _InfoRow(this.label, this.value, {this.valueColor});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 120,
            child: Text(label,
                style: const TextStyle(color: _dim, fontSize: 12)),
          ),
          Expanded(
            child: Text(value,
                style: TextStyle(
                    color:      valueColor ?? Colors.white,
                    fontSize:   12,
                    fontWeight: FontWeight.w600)),
          ),
        ],
      ),
    );
  }
}

class _Metric {
  final String label;
  final String value;
  const _Metric(this.label, this.value);
}

class _MetricGrid extends StatelessWidget {
  final List<_Metric> metrics;
  const _MetricGrid({required this.metrics});

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 12,
      runSpacing: 8,
      children: metrics.map((m) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color:        Colors.white.withOpacity(0.05),
          borderRadius: BorderRadius.circular(8),
          border:       Border.all(color: Colors.white12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(m.label,
                style: const TextStyle(color: _dim, fontSize: 10)),
            const SizedBox(height: 2),
            Text(m.value,
                style: const TextStyle(
                    color:      Colors.white,
                    fontSize:   16,
                    fontWeight: FontWeight.bold)),
          ],
        ),
      )).toList(),
    );
  }
}

// ── Data models ───────────────────────────────────────────────────────────────

class _MatchResult {
  final int             yoloDetections;
  final int             embeddingsCount;
  final int             matchesFound;
  final List<DetectedProduct> top3;
  final PipelineTiming  timing;
  final String?         error;

  const _MatchResult({
    required this.yoloDetections,
    required this.embeddingsCount,
    required this.matchesFound,
    required this.top3,
    required this.timing,
    this.error,
  });

  factory _MatchResult.error(String msg) => _MatchResult(
    yoloDetections:  0,
    embeddingsCount: 0,
    matchesFound:    0,
    top3:            [],
    timing:          const PipelineTiming(
        yoloMs: 0, filterMs: 0, embeddingMs: 0, matchingMs: 0, totalMs: 0),
    error: msg,
  );
}

class _ProductHealth {
  final String        name;
  final int           angleCount;
  final double?       similarity;   // null = not enough angles to compare
  final List<String>  comparedAngles;

  const _ProductHealth({
    required this.name,
    required this.angleCount,
    required this.similarity,
    required this.comparedAngles,
  });
}
