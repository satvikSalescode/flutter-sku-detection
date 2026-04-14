import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

import '../models/detection_result.dart';

// ──────────────────────────────────────────────────────────────────────────────
// ★  UPDATE THESE IF YOUR MODEL DIFFERS
// ──────────────────────────────────────────────────────────────────────────────
/// ONNX model file inside assets/
const String kModelAsset = 'assets/best.onnx';

/// Labels file inside assets/
const String kLabelsAsset = 'assets/labels.txt';

/// Input resolution fed to the ONNX model.
/// Set to 640 — re-export best.onnx with imgsz=640 if you haven't already.
const int kInputSize = 640;

/// Detections below this confidence are ignored
const double kConfThreshold = 0.25;
// ──────────────────────────────────────────────────────────────────────────────

class InferenceService {
  OrtSession? _session;
  List<String> _labels = [];
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;
  List<String> get labels => _labels;

  // ── Initialisation ─────────────────────────────────────────────────────────

  Future<void> initialize() async {
    if (_isInitialized) return;
    OrtEnv.instance.init();
    await _loadLabels();
    await _loadModel();
    _isInitialized = true;
  }

  Future<void> _loadLabels() async {
    final raw = await rootBundle.loadString(kLabelsAsset);
    _labels = raw
        .split('\n')
        .map((l) => l.trim())
        .where((l) => l.isNotEmpty && !l.startsWith('#'))
        .toList();
  }

  Future<void> _loadModel() async {
    // Use 4 CPU threads for inference
    final sessionOptions = OrtSessionOptions()
      ..setIntraOpNumThreads(4)
      ..setInterOpNumThreads(4);

    final modelBytes = await rootBundle.load(kModelAsset);
    _session = OrtSession.fromBuffer(
      modelBytes.buffer.asUint8List(),
      sessionOptions,
    );

    print('[InferenceService] ✅  Model loaded');
    print('[InferenceService] Input  names : ${_session!.inputNames}');
    print('[InferenceService] Output names : ${_session!.outputNames}');
  }

  // ── Inference ──────────────────────────────────────────────────────────────

  /// Runs inference on [imageFile] and returns detected objects.
  Future<List<DetectionResult>> detect(File imageFile) async {
    if (!_isInitialized) throw StateError('Call initialize() first');

    // 1. Decode & resize
    final bytes    = await imageFile.readAsBytes();
    final original = img.decodeImage(bytes);
    if (original == null) throw Exception('Could not decode image');

    final resized = img.copyResize(
      original,
      width:         kInputSize,
      height:        kInputSize,
      interpolation: img.Interpolation.linear,
    );

    // 2. Build NCHW float32 input tensor  [1, 3, kInputSize, kInputSize]
    //    ONNX models expect NCHW (channels first), unlike TFLite (NHWC).
    final inputData = _imageToNCHW(resized);
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputData,
      [1, 3, kInputSize, kInputSize],
    );

    // 3. Run
    final runOptions = OrtRunOptions();
    final inputs     = {_session!.inputNames[0]: inputTensor};
    final sw         = Stopwatch()..start();
    final outputs    = await _session!.runAsync(runOptions, inputs);
    sw.stop();
    runOptions.release();
    print('[InferenceService] ⚡  Inference took ${sw.elapsedMilliseconds} ms');

    inputTensor.release();

    // 4. Decode output
    final results = _decodeOutput(outputs!, original.width, original.height);
    for (final out in outputs) {
      out?.release();
    }

    return results;
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  /// Converts image to a flat Float32List in NCHW order (R plane, G plane, B plane).
  Float32List _imageToNCHW(img.Image image) {
    final size = kInputSize * kInputSize;
    final data = Float32List(3 * size);

    final rOffset = 0;
    final gOffset = size;
    final bOffset = size * 2;

    int i = 0;
    for (int y = 0; y < kInputSize; y++) {
      for (int x = 0; x < kInputSize; x++) {
        final pixel = image.getPixel(x, y);
        data[rOffset + i] = pixel.r / 255.0;
        data[gOffset + i] = pixel.g / 255.0;
        data[bOffset + i] = pixel.b / 255.0;
        i++;
      }
    }
    return data;
  }

  /// Decodes model output (shape [1, 300, 6]) into DetectionResult list.
  /// Each of the 300 rows: [x1, y1, x2, y2, confidence, class_id]
  /// Coordinates are normalised to the input size (640×640).
  List<DetectionResult> _decodeOutput(
    List<OrtValue?> outputs,
    int origW,
    int origH,
  ) {
    // Shape: [1, 300, 6]  →  outer list = batch, inner = [300][6]
    final raw = outputs[0]?.value as List<List<List<double>>>;
    final rows = raw[0]; // [300, 6]

    final results = <DetectionResult>[];

    for (final row in rows) {
      final conf = row[4];
      if (conf < kConfThreshold) continue;

      final classIndex = row[5].round();

      // Coords are normalised to kInputSize — scale to original image pixels
      final x1 = (row[0] / kInputSize * origW).clamp(0.0, origW.toDouble());
      final y1 = (row[1] / kInputSize * origH).clamp(0.0, origH.toDouble());
      final x2 = (row[2] / kInputSize * origW).clamp(0.0, origW.toDouble());
      final y2 = (row[3] / kInputSize * origH).clamp(0.0, origH.toDouble());

      if (x2 <= x1 || y2 <= y1) continue;

      final label = (classIndex < _labels.length)
          ? _labels[classIndex]
          : 'class_$classIndex';

      results.add(DetectionResult(
        boundingBox: Rect.fromLTRB(x1, y1, x2, y2),
        classIndex:  classIndex,
        label:       label,
        confidence:  conf,
      ));
    }

    return results;
  }

  void dispose() {
    _session?.release();
    OrtEnv.instance.release();
    _isInitialized = false;
  }
}
