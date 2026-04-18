import 'dart:math' show sqrt;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

// ── Constants ──────────────────────────────────────────────────────────────────

/// Asset path for the exported DINOv3-small ONNX model.
const String kDinoV3Asset = 'assets/dinov3_small.onnx';

/// Shortest-edge resize target before the center crop (standard ViT-S/16).
const int _kV3ResizeTo = 256;

/// Square crop size fed to DINOv3.
const int _kV3CropSize = 224;

/// Output embedding dimension (DINOv3-small pooler_output).
const int _kV3EmbDim = 384;

/// Maximum images per ONNX inference call.
const int _kV3BatchSize = 16;

/// ImageNet normalisation — mean per channel (R, G, B), applied after ÷255.
const List<double> _kV3Mean = [0.485, 0.456, 0.406];

/// ImageNet normalisation — std per channel (R, G, B), applied after ÷255.
const List<double> _kV3Std = [0.229, 0.224, 0.225];

// ── Service ────────────────────────────────────────────────────────────────────

/// Loads facebook/dinov3-vits16-pretrain-lvd1689m (ONNX) and produces
/// L2-normalised 384-dim embeddings from product crop images.
///
/// Key difference from DinoV2Service: the ONNX wrapper returns
/// `pooler_output` directly as shape [batch, 384] — a flat 2-D tensor —
/// rather than the full hidden-state sequence [batch, seq_len, 384].
/// No CLS-token slicing is needed here.
///
/// Usage:
///   final dino = DinoV3Service();
///   await dino.initialize();
///   final emb = await dino.getEmbedding(cropImage);
///
/// Do NOT call any embedding method before [initialize] completes.
/// Dispose with [dispose] when the owning widget is torn down.
class DinoV3Service {
  OrtSession? _session;
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;

  // ── Init / dispose ──────────────────────────────────────────────────────────

  /// Loads the ONNX model from assets into memory.
  /// Throws if the asset is missing or the model cannot be parsed.
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      OrtEnv.instance.init();

      final opts = OrtSessionOptions()
        ..setIntraOpNumThreads(4)
        ..setInterOpNumThreads(2);

      final bytes = await rootBundle.load(kDinoV3Asset);
      _session = OrtSession.fromBuffer(bytes.buffer.asUint8List(), opts);

      _isInitialized = true;
      debugPrint('[DinoV3] ✅  Model loaded'
          '  in=${_session!.inputNames}'
          '  out=${_session!.outputNames}');
    } on FlutterError catch (e) {
      debugPrint('[DinoV3] ❌  Asset not found: $e');
      debugPrint('[DinoV3]    Copy dinov3_small.onnx to assets/ and rebuild.');
      rethrow;
    } catch (e) {
      debugPrint('[DinoV3] ❌  Model load failed: $e');
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
  /// Preprocessing pipeline (must match export_dinov3.py + generate_catalog.py):
  ///   1. Resize shortest edge → 256 px  (bicubic, aspect-ratio preserved)
  ///   2. Center crop 224 × 224
  ///   3. Pixels ÷ 255.0  → float [0, 1]
  ///   4. (pixel − mean) / std  per channel  (ImageNet stats)
  ///   5. Layout: NCHW float32  [1, 3, 224, 224]
  ///   6. Output: pooler_output [1, 384]  (no CLS-token extraction needed)
  ///   7. L2-normalise
  Future<List<double>> getEmbedding(img.Image cropImage) async {
    _assertReady();
    try {
      final tensor = _preprocessBatch([cropImage]);
      final embs   = await _runBatch(tensor, batchSize: 1);
      return embs[0];
    } catch (e) {
      debugPrint('[DinoV3] ❌  getEmbedding failed: $e');
      rethrow;
    }
  }

  /// Returns one 384-dim L2-normalised embedding per image in [crops].
  /// Inference is batched in chunks of [_kV3BatchSize] to bound memory usage.
  Future<List<List<double>>> getEmbeddings(List<img.Image> crops) async {
    _assertReady();
    if (crops.isEmpty) return [];

    final results = <List<double>>[];

    for (int start = 0; start < crops.length; start += _kV3BatchSize) {
      final end   = (start + _kV3BatchSize).clamp(0, crops.length);
      final chunk = crops.sublist(start, end);

      try {
        final tensor = _preprocessBatch(chunk);
        final embs   = await _runBatch(tensor, batchSize: chunk.length);
        results.addAll(embs);
      } catch (e) {
        debugPrint('[DinoV3] ❌  Batch [$start..$end) failed: $e');
        for (int i = 0; i < chunk.length; i++) {
          results.add(List.filled(_kV3EmbDim, 0.0));
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
  ///   2. Output is a [_kV3EmbDim]-length vector with finite, non-NaN values.
  ///   3. The vector is L2-normalised (‖v‖ ≈ 1.0).
  ///   4. Determinism — same input → same embedding (cosine ≥ 0.9999).
  ///
  /// Returns true when all checks pass.
  Future<bool> selfTest() async {
    _assertReady();
    debugPrint('[DinoV3] Running self-test…');

    try {
      final probe = _makeCheckerboard(_kV3CropSize, _kV3CropSize, blockSize: 32);

      final emb1 = await getEmbedding(probe);
      final emb2 = await getEmbedding(probe);

      // Check 1: correct length
      if (emb1.length != _kV3EmbDim) {
        debugPrint('[DinoV3] ❌  Self-test FAILED  '
            '(output dim=${emb1.length}, expected $_kV3EmbDim)');
        return false;
      }

      // Check 2: finite & non-NaN
      if (emb1.any((v) => v.isNaN || v.isInfinite)) {
        debugPrint('[DinoV3] ❌  Self-test FAILED  (embedding contains NaN/Inf)');
        return false;
      }

      // Check 3: L2 norm ≈ 1.0
      double sumSq = 0;
      for (final v in emb1) sumSq += v * v;
      final norm = sqrt(sumSq);
      if ((norm - 1.0).abs() > 0.01) {
        debugPrint('[DinoV3] ❌  Self-test FAILED  '
            '(embedding norm=${norm.toStringAsFixed(4)}, expected ≈ 1.0)');
        return false;
      }

      // Check 4: determinism
      final sim = _cosine(emb1, emb2);
      if (sim < 0.9999) {
        debugPrint('[DinoV3] ❌  Self-test FAILED  '
            '(determinism cosine=${sim.toStringAsFixed(6)}, expected ≥ 0.9999)');
        return false;
      }

      debugPrint('[DinoV3] ✅  Self-test PASSED  '
          '(dim=$_kV3EmbDim, norm=${norm.toStringAsFixed(4)}, '
          'determinism=${sim.toStringAsFixed(6)})');
      return true;
    } catch (e) {
      debugPrint('[DinoV3] ❌  Self-test threw: $e');
      return false;
    }
  }

  // ── Preprocessing ───────────────────────────────────────────────────────────

  /// Converts a list of [img.Image] objects to a single NCHW Float32List
  /// shaped [batchSize, 3, 224, 224].
  Float32List _preprocessBatch(List<img.Image> images) {
    final n     = images.length;
    final plane = _kV3CropSize * _kV3CropSize;
    final data  = Float32List(n * 3 * plane);

    for (int b = 0; b < n; b++) {
      final cropped  = _resizeAndCrop(images[b]);
      final bOffset  = b * 3 * plane;
      final rOffset  = bOffset;
      final gOffset  = bOffset + plane;
      final blOffset = bOffset + plane * 2;

      int px = 0;
      for (int y = 0; y < _kV3CropSize; y++) {
        for (int x = 0; x < _kV3CropSize; x++) {
          final pixel = cropped.getPixel(x, y);
          data[rOffset  + px] = ((pixel.r / 255.0) - _kV3Mean[0]) / _kV3Std[0];
          data[gOffset  + px] = ((pixel.g / 255.0) - _kV3Mean[1]) / _kV3Std[1];
          data[blOffset + px] = ((pixel.b / 255.0) - _kV3Mean[2]) / _kV3Std[2];
          px++;
        }
      }
    }
    return data;
  }

  /// Resize shortest edge → [_kV3ResizeTo] then center-crop [_kV3CropSize]×[_kV3CropSize].
  img.Image _resizeAndCrop(img.Image src) {
    final sw     = src.width;
    final sh     = src.height;
    final scale  = _kV3ResizeTo / (sw < sh ? sw : sh).toDouble();
    final scaledW = (sw * scale).round();
    final scaledH = (sh * scale).round();

    final resized = img.copyResize(
      src,
      width:         scaledW,
      height:        scaledH,
      interpolation: img.Interpolation.cubic,
    );

    final cropX = (scaledW - _kV3CropSize) ~/ 2;
    final cropY = (scaledH - _kV3CropSize) ~/ 2;

    return img.copyCrop(
      resized,
      x:      cropX,
      y:      cropY,
      width:  _kV3CropSize,
      height: _kV3CropSize,
    );
  }

  // ── ONNX inference ──────────────────────────────────────────────────────────

  /// Runs a single ONNX inference call for a pre-built NCHW tensor.
  ///
  /// DINOv3 wrapper output shape: [batchSize, 384]  (flat pooler_output).
  /// In Dart ORT this is: List<List<double>> — [batch][dim].
  /// This differs from DinoV2Service which outputs [batch][seq_len][dim].
  Future<List<List<double>>> _runBatch(
      Float32List data, {required int batchSize}) async {
    final tensor = OrtValueTensor.createTensorWithDataList(
      data,
      [batchSize, 3, _kV3CropSize, _kV3CropSize],
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
      throw StateError('[DinoV3] ONNX returned null output');
    }

    // Output shape: [batchSize, 384]  → List<List<double>> in Dart ORT
    final raw = outputs[0]!.value as List<List<double>>;

    final results = <List<double>>[];
    for (int b = 0; b < batchSize; b++) {
      results.add(_l2Normalize(raw[b]));
    }

    for (final o in outputs) {
      o?.release();
    }

    return results;
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────

  List<double> _l2Normalize(List<double> v) {
    double sumSq = 0;
    for (final x in v) sumSq += x * x;
    if (sumSq < 1e-12) return v;
    final inv = 1.0 / sqrt(sumSq);
    return [for (final x in v) x * inv];
  }

  double _cosine(List<double> a, List<double> b) {
    double dot = 0;
    for (int i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot;
  }

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

  void _assertReady() {
    if (!_isInitialized || _session == null) {
      throw StateError(
          '[DinoV3] Not initialized — call await initialize() first.');
    }
  }
}
