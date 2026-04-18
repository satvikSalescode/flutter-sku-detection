import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

import '../services/vision_pipeline.dart';
import 'analysis_results_page.dart';

// ── Page ──────────────────────────────────────────────────────────────────────

class DetectionPage extends StatefulWidget {
  /// Fully initialised pipeline (YOLO + DINOv3 + catalog loaded).
  final VisionPipeline pipeline;

  const DetectionPage({super.key, required this.pipeline});

  @override
  State<DetectionPage> createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage> {
  static const _teal = Color(0xFF00C9A7);
  static const _card = Color(0xFF1A1A2E);
  static const _bg   = Color(0xFF0F0F1A);

  File?      _imageFile;
  img.Image? _decoded;
  Size?      _imageSize;

  bool   _isProcessing = false;
  String _stepLabel    = '';
  String? _error;

  // ── Image pick & pipeline ──────────────────────────────────────────────────

  Future<void> _pick(ImageSource source) async {
    final xf = await ImagePicker().pickImage(source: source, imageQuality: 95);
    if (xf == null) return;

    final file    = File(xf.path);
    final bytes   = await file.readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      setState(() => _error = 'Cannot decode image');
      return;
    }

    setState(() {
      _imageFile  = file;
      _decoded    = decoded;
      _imageSize  = Size(decoded.width.toDouble(), decoded.height.toDouble());
      _error      = null;
    });

    await _runPipeline(decoded);
  }

  Future<void> _runPipeline(img.Image decoded) async {
    setState(() {
      _isProcessing = true;
      _stepLabel    = 'Running YOLO detection…';
    });

    try {
      // Show sequential step labels so the user knows what's happening.
      // The pipeline itself decides timing; we update the label on a short delay.
      _updateStepLabel('Running YOLO detection…', 0);
      _updateStepLabel('Generating DINOv3 embeddings…', 600);
      _updateStepLabel('Matching against catalog…', 1400);

      final analysis = await widget.pipeline.analyzeShelf(decoded);

      if (!mounted) return;
      Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => AnalysisResultsPage(
          shelfImage: decoded,
          analysis:   analysis,
        ),
      ));
    } catch (e) {
      if (mounted) setState(() => _error = 'Analysis failed: $e');
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  /// Updates [_stepLabel] after [delayMs] milliseconds.
  void _updateStepLabel(String label, int delayMs) {
    Future.delayed(Duration(milliseconds: delayMs), () {
      if (mounted && _isProcessing) {
        setState(() => _stepLabel = label);
      }
    });
  }

  void _snack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content:         Text(msg),
      backgroundColor: _card,
      behavior:        SnackBarBehavior.floating,
    ));
  }

  // ── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bg,
      appBar: AppBar(
        backgroundColor: _card,
        title: const Text('SKU Detector',
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        actions: [
          // Re-analyse badge (only when an image is loaded + pipeline idle)
          if (_imageFile != null && !_isProcessing)
            Padding(
              padding: const EdgeInsets.only(right: 12),
              child: IconButton(
                icon:    const Icon(Icons.refresh, color: _teal),
                tooltip: 'Re-analyse',
                onPressed: () {
                  if (_decoded != null) _runPipeline(_decoded!);
                },
              ),
            ),
        ],
      ),
      body: Column(
        children: [
          // ── Image preview ────────────────────────────────────────────────────
          Expanded(
            child: _imageFile == null ? _emptyState() : _imagePreview(),
          ),

          // ── Bottom action bar ────────────────────────────────────────────────
          _bottomBar(),
        ],
      ),
    );
  }

  // ── Empty state ────────────────────────────────────────────────────────────

  Widget _emptyState() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 80, height: 80,
            decoration: BoxDecoration(
              color:        _teal.withOpacity(0.08),
              borderRadius: BorderRadius.circular(20),
              border:       Border.all(color: _teal.withOpacity(0.2)),
            ),
            child: const Icon(Icons.add_a_photo_outlined,
                size: 36, color: _teal),
          ),
          const SizedBox(height: 20),
          const Text('Point at a shelf and analyse',
              style: TextStyle(
                  color:      Colors.white,
                  fontSize:   16,
                  fontWeight: FontWeight.w600)),
          const SizedBox(height: 6),
          const Text('YOLO · DINOv3 · On-device',
              style: TextStyle(color: Colors.white38, fontSize: 12)),
        ],
      ),
    );
  }

  // ── Image preview + overlay ────────────────────────────────────────────────

  Widget _imagePreview() {
    return Stack(
      fit: StackFit.expand,
      children: [
        // Image
        Image.file(_imageFile!, fit: BoxFit.contain),

        // Processing overlay
        if (_isProcessing)
          Container(
            color: Colors.black54,
            child: Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const SizedBox(
                    width:  44,
                    height: 44,
                    child: CircularProgressIndicator(
                        color: _teal, strokeWidth: 3),
                  ),
                  const SizedBox(height: 18),
                  Text(_stepLabel,
                      style: const TextStyle(
                          color:      Colors.white,
                          fontSize:   14,
                          fontWeight: FontWeight.w500)),
                  const SizedBox(height: 8),
                  const Text('Please wait…',
                      style: TextStyle(color: Colors.white38, fontSize: 12)),
                ],
              ),
            ),
          ),

        // Error banner
        if (_error != null)
          Positioned(
            left: 16, right: 16, bottom: 16,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color:        Colors.red.shade900.withOpacity(0.92),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                children: [
                  const Icon(Icons.error_outline,
                      color: Colors.white70, size: 18),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(_error!,
                        style: const TextStyle(
                            color: Colors.white, fontSize: 12)),
                  ),
                ],
              ),
            ),
          ),
      ],
    );
  }

  // ── Bottom bar ────────────────────────────────────────────────────────────

  Widget _bottomBar() {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
      decoration: BoxDecoration(
        color:  _card,
        border: Border(top: BorderSide(color: Colors.white.withOpacity(0.07))),
      ),
      child: Row(
        children: [
          Expanded(
            child: _ActionButton(
              icon:    Icons.photo_library_outlined,
              label:   'Gallery',
              onTap:   _isProcessing ? null : () => _pick(ImageSource.gallery),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: _ActionButton(
              icon:    Icons.camera_alt_outlined,
              label:   'Camera',
              primary: true,
              onTap:   _isProcessing ? null : () => _pick(ImageSource.camera),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Reusable button ───────────────────────────────────────────────────────────

class _ActionButton extends StatelessWidget {
  final IconData    icon;
  final String      label;
  final bool        primary;
  final VoidCallback? onTap;

  static const _teal = Color(0xFF00C9A7);
  static const _grey = Color(0xFF252540);

  const _ActionButton({
    required this.icon,
    required this.label,
    this.primary = false,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedOpacity(
        opacity:  onTap == null ? 0.4 : 1.0,
        duration: const Duration(milliseconds: 150),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 14),
          decoration: BoxDecoration(
            color:        primary ? _teal : _grey,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon,
                  color: primary ? Colors.black : _teal, size: 20),
              const SizedBox(width: 8),
              Text(label,
                  style: TextStyle(
                      color:      primary ? Colors.black : Colors.white70,
                      fontWeight: FontWeight.w600,
                      fontSize:   14)),
            ],
          ),
        ),
      ),
    );
  }
}
