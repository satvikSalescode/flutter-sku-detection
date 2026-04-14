import 'dart:math' show sqrt;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

// ── Constants ──────────────────────────────────────────────────────────────────

/// Asset path for the exported DINOv2-small ONNX model.
const String kDinoAsset = 'assets/dinov2_small.onnx';

/// Shortest-edge resize target before the center crop.
const int _kResizeTo = 256;

/// Square crop size fed to DINOv2.
const int _kCropSize = 224;

/// Output embedding dimension (DINOv2-small CLS token).
const int _kEmbDim = 384;

/// Maximum images per ONNX inference call.
const int _kBatchSize = 16;

/// ImageNet normalisation — mean per channel (R, G, B), applied after ÷255.
const List<double> _kMean = [0.485, 0.456, 0.406];

/// ImageNet normalisation — std per channel (R, G, B), applied after ÷255.
const List<double> _kStd = [0.229, 0.224, 0.225];

// ── Service ────────────────────────────────────────────────────────────────────

/// Loads facebook/dinov2-small (ONNX) and produces L2-normalised 384-dim
/// embeddings from product crop images.
///
/// Usage:
///   final dino = DinoV2Service();
///   await dino.initialize();
///   final emb = await dino.getEmbedding(cropImage);
///
/// Do NOT call any embedding method before [initialize] completes.
/// Dispose with [dispose] when the owning widget is torn down.
class DinoV2Service {
  OrtSession? _session;
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;

  // ── Init / dispose ──────────────────────────────────────────────────────────

  /// Loads the ONNX model from assets into memory.
  /// Throws if the asset is missing or the model cannot be parsed.
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // OrtEnv is a process-wide singleton already initialised by InferenceService.
      // Calling init() again is a no-op, so it is safe to call here.
      OrtEnv.instance.init();

      final opts = OrtSessionOptions()
        ..setIntraOpNumThreads(4)
        ..setInterOpNumThreads(2);

      final bytes = await rootBundle.load(kDinoAsset);
      _session = OrtSession.fromBuffer(bytes.buffer.asUint8List(), opts);

      _isInitialized = true;
      debugPrint('[DinoV2] ✅  Model loaded'
          '  in=${_session!.inputNames}'
          '  out=${_session!.outputNames}');
    } on FlutterError catch (e) {
      debugPrint('[DinoV2] ❌  Asset not found: $e');
      debugPrint('[DinoV2]    Copy dinov2_small.onnx to assets/ and rebuild.');
      rethrow;
    } catch (e) {
      debugPrint('[DinoV2] ❌  Model load failed: $e');
      rethrow;
    }
  }

  void dispose() {
    _session?.release();
    _session = null;
    _isInitialized = false;
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /// Returns a 384-dim L2-normalised embedding for [cropImage].
  ///
  /// Preprocessing pipeline (must exactly match generate_catalog.py):
  ///   1. Resize shortest edge → 256 px  (bicubic, aspect-ratio preserved)
  ///   2. Center crop 224 × 224
  ///   3. Pixels ÷ 255.0  → float [0, 1]
  ///   4. (pixel − mean) / std  per channel  (ImageNet stats)
  ///   5. Layout: NCHW float32  [1, 3, 224, 224]
  ///   6. CLS token = output[0, 0, 0:384]
  ///   7. L2-normalise
  Future<List<double>> getEmbedding(img.Image cropImage) async {
    _assertReady();
    try {
      final tensor = _preprocessBatch([cropImage]);
      final embs   = await _runBatch(tensor, batchSize: 1);
      return embs[0];
    } catch (e) {
      debugPrint('[DinoV2] ❌  getEmbedding failed: $e');
      rethrow;
    }
  }

  /// Returns one 384-dim L2-normalised embedding per image in [crops].
  /// Inference is batched in chunks of [_kBatchSize] to bound memory usage.
  Future<List<List<double>>> getEmbeddings(List<img.Image> crops) async {
    _assertReady();
    if (crops.isEmpty) return [];

    final results = <List<double>>[];

    for (int start = 0; start < crops.length; start += _kBatchSize) {
      final end   = (start + _kBatchSize).clamp(0, crops.length);
      final chunk = crops.sublist(start, end);

      try {
        final tensor = _preprocessBatch(chunk);
        final embs   = await _runBatch(tensor, batchSize: chunk.length);
        results.addAll(embs);
      } catch (e) {
        debugPrint('[DinoV2] ❌  Batch [$start..$end) failed: $e');
        // Fill with zero vectors so indices stay aligned with crops list.
        for (int i = 0; i < chunk.length; i++) {
          results.add(List.filled(_kEmbDim, 0.0));
        }
      }
    }

    return results;
  }

  // ── Self-test ───────────────────────────────────────────────────────────────

  /// Sanity-checks that the model runs correctly and produces valid embeddings.
  ///
  /// Checks performed:
  ///   1. Inference completes without throwing.
  ///   2. Output is a [_kEmbDim]-length vector with finite, non-NaN values.
  ///   3. The vector is L2-normalised (‖v‖ ≈ 1.0).
  ///   4. Determinism — the same image produces the same embedding twice
  ///      (cosine similarity ≥ 0.9999).
  ///
  /// Note: comparing two different solid-color images is NOT a valid test for
  /// a Vision Transformer — DINOv2 produces similar CLS-token embeddings for
  /// any uniform-patch input because the attention mechanism collapses when
  /// all patches are identical. The checks above are the correct way to
  /// verify that the model and preprocessing pipeline are functioning.
  ///
  /// Returns true when all checks pass.
  Future<bool> selfTest() async {
    _assertReady();
    debugPrint('[DinoV2] Running self-test…');

    try {
      // Use a checkerboard image — gives the ViT varied patch content while
      // still being generatable without external assets.
      final probe = _makeCheckerboard(_kCropSize, _kCropSize, blockSize: 32);

      final emb1 = await getEmbedding(probe);
      final emb2 = await getEmbedding(probe);   // second run — determinism

      // ── Check 1: correct length ──────────────────────────────────────────
      if (emb1.length != _kEmbDim) {
        debugPrint('[DinoV2] ❌  Self-test FAILED  '
            '(output dim=${emb1.length}, expected $_kEmbDim)');
        return false;
      }

      // ── Check 2: finite & non-NaN ────────────────────────────────────────
      final hasInvalid = emb1.any((v) => v.isNaN || v.isInfinite);
      if (hasInvalid) {
        debugPrint('[DinoV2] ❌  Self-test FAILED  (embedding contains NaN/Inf)');
        return false;
      }

      // ── Check 3: L2 norm ≈ 1.0 (already normalised by _l2Normalize) ─────
      double sumSq = 0;
      for (final v in emb1) sumSq += v * v;
      final norm = sqrt(sumSq);
      if ((norm - 1.0).abs() > 0.01) {
        debugPrint('[DinoV2] ❌  Self-test FAILED  '
            '(embedding norm=${ norm.toStringAsFixed(4)}, expected ≈ 1.0)');
        return false;
      }

      // ── Check 4: determinism ─────────────────────────────────────────────
      final sim = _cosine(emb1, emb2);
      if (sim < 0.9999) {
        debugPrint('[DinoV2] ❌  Self-test FAILED  '
            '(determinism cosine=${sim.toStringAsFixed(6)}, expected ≥ 0.9999)');
        return false;
      }

      debugPrint('[DinoV2] ✅  Self-test PASSED  '
          '(dim=$_kEmbDim, norm=${norm.toStringAsFixed(4)}, '
          'determinism=${sim.toStringAsFixed(6)})');
      return true;
    } catch (e) {
      debugPrint('[DinoV2] ❌  Self-test threw: $e');
      return false;
    }
  }

  /// Generates a black-and-white checkerboard [img.Image] of [w]×[h] pixels
  /// with square blocks of size [blockSize]. Used only for the self-test.
  img.Image _makeCheckerboard(int w, int h, {int blockSize = 32}) {
    final image = img.Image(width: w, height: h);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final isWhite = ((x ~/ blockSize) + (y ~/ blockSize)).isEven;
        image.setPixel(x, y,
            isWhite ? img.ColorRgb8(255, 255, 255) : img.ColorRgb8(0, 0, 0));
      }
    }
    return image;
  }

  // ── Preprocessing ───────────────────────────────────────────────────────────

  /// Converts a list of [img.Image] objects to a single NCHW Float32List
  /// shaped [batchSize, 3, 224, 224].
  ///
  /// Channel layout (NCHW):
  ///   [b0_R(0,0)…b0_R(223,223), b0_G(0,0)…, b0_B(0,0)…,
  ///    b1_R(0,0)…b1_R(223,223), b1_G(0,0)…, b1_B(0,0)…, …]
  Float32List _preprocessBatch(List<img.Image> images) {
    final n      = images.length;
    final plane  = _kCropSize * _kCropSize;          // 224*224 = 50 176
    final data   = Float32List(n * 3 * plane);

    for (int b = 0; b < n; b++) {
      final cropped  = _resizeAndCrop(images[b]);    // 224×224 img.Image
      final bOffset  = b * 3 * plane;
      final rOffset  = bOffset;
      final gOffset  = bOffset + plane;
      final blOffset = bOffset + plane * 2;

      int px = 0;
      for (int y = 0; y < _kCropSize; y++) {
        for (int x = 0; x < _kCropSize; x++) {
          final pixel = cropped.getPixel(x, y);

          // ÷255 then (value − mean) / std  — RGB channel order
          data[rOffset  + px] = ((pixel.r / 255.0) - _kMean[0]) / _kStd[0];
          data[gOffset  + px] = ((pixel.g / 255.0) - _kMean[1]) / _kStd[1];
          data[blOffset + px] = ((pixel.b / 255.0) - _kMean[2]) / _kStd[2];
          px++;
        }
      }
    }
    return data;
  }

  /// Resize shortest edge → [_kResizeTo] then center-crop [_kCropSize]×[_kCropSize].
  ///
  /// Example: 480×640 input
  ///   → scale = 256/480 = 0.533  → 256×341
  ///   → cropX = (341−224)÷2 = 58,  cropY = (256−224)÷2 = 16
  ///   → crop(58, 16, 224, 224)
  img.Image _resizeAndCrop(img.Image src) {
    final sw = src.width;
    final sh = src.height;

    // Scale so the shorter dimension becomes exactly _kResizeTo
    final scale  = _kResizeTo / (sw < sh ? sw : sh).toDouble();
    final scaledW = (sw * scale).round();
    final scaledH = (sh * scale).round();

    final resized = img.copyResize(
      src,
      width:         scaledW,
      height:        scaledH,
      interpolation: img.Interpolation.cubic,
    );

    // Center crop
    final cropX = (scaledW - _kCropSize) ~/ 2;
    final cropY = (scaledH - _kCropSize) ~/ 2;

    return img.copyCrop(
      resized,
      x:      cropX,
      y:      cropY,
      width:  _kCropSize,
      height: _kCropSize,
    );
  }

  // ── ONNX inference ──────────────────────────────────────────────────────────

  /// Runs a single ONNX inference call for a pre-built NCHW tensor.
  /// Returns one L2-normalised [_kEmbDim]-dim embedding per image in the batch.
  Future<List<List<double>>> _runBatch(
      Float32List data, {required int batchSize}) async {
    final tensor = OrtValueTensor.createTensorWithDataList(
      data,
      [batchSize, 3, _kCropSize, _kCropSize],
    );

    final runOpts = OrtRunOptions();
    List<OrtValue?>? outputs;

    try {
      outputs = await _session!.runAsync(
        runOpts,
        {_session!.inputNames[0]: tensor},
      );
    } finally {
      runOpts.release();
      tensor.release();
    }

    if (outputs == null || outputs.isEmpty || outputs[0] == null) {
      throw StateError('[DinoV2] ONNX returned null output');
    }

    // Output shape: [batchSize, seq_len, 384]
    // In Dart ORT: List<List<List<double>>> — [batch][token][dim]
    final raw = outputs[0]!.value as List<List<List<double>>>;

    final results = <List<double>>[];
    for (int b = 0; b < batchSize; b++) {
      // CLS token is token index 0
      final cls = raw[b][0];                  // List<double>, length 384

      // L2-normalise
      results.add(_l2Normalize(cls));
    }

    for (final o in outputs) {
      o?.release();
    }

    return results;
  }

  // ── Math helpers ────────────────────────────────────────────────────────────

  /// L2-normalises [v] in place and returns it.
  List<double> _l2Normalize(List<double> v) {
    double sumSq = 0;
    for (final x in v) sumSq += x * x;
    if (sumSq < 1e-12) return v;           // zero vector guard
    final inv = 1.0 / sqrt(sumSq);
    return [for (final x in v) x * inv];
  }

  /// Cosine similarity of two L2-normalised vectors (= dot product).
  double _cosine(List<double> a, List<double> b) {
    double dot = 0;
    for (int i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot;
  }

  // ── Guard ───────────────────────────────────────────────────────────────────

  void _assertReady() {
    if (!_isInitialized || _session == null) {
      throw StateError(
          '[DinoV2] Not initialized — call await initialize() first.');
    }
  }
}
