import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart' show Rect;
import 'package:image/image.dart' as img;

import '../config/vision_config.dart';
import '../models/detection_result.dart';
import '../utils/crop_filter.dart';
import 'backend_api_client.dart';
import 'catalog_service.dart';
import 'dinov3_service.dart';
import 'inference_service.dart';

// ── Result models ─────────────────────────────────────────────────────────────

/// Wall-clock milliseconds for each pipeline step plus the total.
class PipelineTiming {
  final int yoloMs;
  final int filterMs;
  final int embeddingMs;
  final int matchingMs;
  final int totalMs;

  const PipelineTiming({
    required this.yoloMs,
    required this.filterMs,
    required this.embeddingMs,
    required this.matchingMs,
    required this.totalMs,
  });

  @override
  String toString() =>
      'PipelineTiming(yolo=${yoloMs}ms  filter=${filterMs}ms  '
      'emb=${embeddingMs}ms  match=${matchingMs}ms  total=${totalMs}ms)';
}

/// One recognised SKU with all its detections on the shelf.
class DetectedProduct {
  /// Canonical product name from the catalog (e.g. "Coke_Can_Red").
  final String productName;

  /// How many distinct crops matched this product (facings count).
  final int facingCount;

  /// Mean cosine similarity across all matched facings.
  final double avgSimilarity;

  /// Bounding boxes (image-pixel coords) for every facing on the shelf.
  final List<Rect> boundingBoxes;

  /// Crop thumbnails for every facing, in the same order as [boundingBoxes].
  /// Empty when the result came from the backend (no local crops available).
  final List<img.Image> cropImages;

  /// OCR text extracted by the backend for this product, or null if unavailable.
  final String? ocrText;

  /// How the match was determined, e.g. "visual + ocr". Null for on-device results.
  final String? matchMethod;

  const DetectedProduct({
    required this.productName,
    required this.facingCount,
    required this.avgSimilarity,
    required this.boundingBoxes,
    required this.cropImages,
    this.ocrText,
    this.matchMethod,
  });

  @override
  String toString() =>
      'DetectedProduct($productName ×$facingCount  '
      'avg=${avgSimilarity.toStringAsFixed(3)})';
}

/// A crop that did not match any catalog product (below score threshold).
class UnknownDetection {
  /// Bounding box (image-pixel coords) of the unmatched crop.
  final Rect boundingBox;

  /// The raw crop image.
  final img.Image cropImage;

  /// The highest cosine similarity the crop achieved (still below threshold).
  final double bestScore;

  /// Name of the catalog product that was the closest match.
  /// Will be "Unknown" when the catalog is empty or the embedding is invalid.
  final String nearestProduct;

  /// OCR text extracted by the backend for this unknown crop, or null.
  final String? ocrText;

  const UnknownDetection({
    required this.boundingBox,
    required this.cropImage,
    required this.bestScore,
    required this.nearestProduct,
    this.ocrText,
  });
}

/// Full result returned by [VisionPipeline.analyzeShelf].
class ShelfAnalysis {
  /// Total crops that survived YOLO filtering (= matchedCount + unknownCount).
  final int totalDetections;

  /// Number of crops confidently matched to a catalog product.
  final int matchedCount;

  /// Number of crops that could not be matched.
  final int unknownCount;

  /// Percentage of total detections that were successfully matched.
  /// Formula: matchedCount / totalDetections × 100.
  final double shelfSharePercent;

  /// Matched products, sorted by facing count (descending).
  final List<DetectedProduct> products;

  /// Unmatched crops with their best-guess nearest product.
  final List<UnknownDetection> unknowns;

  /// Per-step and total wall-clock durations.
  final PipelineTiming timing;

  const ShelfAnalysis({
    required this.totalDetections,
    required this.matchedCount,
    required this.unknownCount,
    required this.shelfSharePercent,
    required this.products,
    required this.unknowns,
    required this.timing,
  });

  @override
  String toString() =>
      'ShelfAnalysis(total=$totalDetections  matched=$matchedCount  '
      'unknown=$unknownCount  share=${shelfSharePercent.toStringAsFixed(1)}%  '
      '${timing.totalMs}ms)';
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

/// Orchestrates the full YOLO → DINOv3 → catalog-matching pipeline.
///
/// Typical lifecycle:
///   final pipeline = VisionPipeline(
///     yoloService:    inferenceService,
///     dinov3Service:  dinoV3Service,
///     catalogService: catalogService,
///   );
///   await pipeline.initialize();
///   final analysis = await pipeline.analyzeShelf(shelfImage);
///
/// [initialize] must complete before [analyzeShelf] is called.
/// The pipeline does not own the injected services — do not call
/// [DinoV3Service.dispose] from [initialize].
class VisionPipeline {
  /// YOLO object-detection service (must already be initialised).
  final InferenceService yoloService;

  /// DINOv3 embedding service (initialised by [initialize]).
  final DinoV3Service dinov3Service;

  /// Catalog matching service (loaded by [initialize]).
  final CatalogService catalogService;

  VisionPipeline({
    required this.yoloService,
    required this.dinov3Service,
    required this.catalogService,
  });

  bool _isInitialized = false;

  /// True after [initialize] has completed successfully.
  bool get isInitialized => _isInitialized;

  // ── Init ─────────────────────────────────────────────────────────────────────

  /// Initialises the pipeline based on [VisionConfig.onDevice].
  ///
  /// On-device mode: loads DINOv3 model + catalog, runs self-test.
  /// Backend mode:   skips DINOv3 and catalog entirely (~40 MB saved);
  ///                 pings the backend health endpoint as an early warning.
  ///
  /// Neither a self-test failure nor a failed health check is fatal here —
  /// both log a warning so the app can still launch. Hard failures (ONNX
  /// crash, missing asset) throw before reaching these checks.
  ///
  /// Calling [initialize] more than once is a no-op.
  Future<void> initialize() async {
    if (_isInitialized) return;

    final mode = VisionConfig.onDevice ? 'on-device' : 'backend';
    debugPrint('[VisionPipeline] Initializing ($mode mode)…');

    if (VisionConfig.onDevice) {
      // ── On-device: load DINOv3 + catalog, run self-test ──────────────────
      await dinov3Service.initialize();
      await catalogService.load();

      final selfTestPassed = await dinov3Service.selfTest();
      if (!selfTestPassed) {
        debugPrint(
            '[VisionPipeline] ⚠️  Self-test warning — model loaded but '
            'self-test did not pass. Continuing anyway; real-image accuracy '
            'may be affected. Check dinov3_service.dart preprocessing if '
            'matching results look wrong.');
      }

      _isInitialized = true;
      debugPrint(
          '[VisionPipeline] ✅  Ready (on-device)  '
          '(${catalogService.products.length} products, '
          '${catalogService.numAngles} reference vectors)');
    } else {
      // ── Backend: skip DINOv3 + catalog; verify API connectivity ──────────
      final client    = BackendApiClient(baseUrl: VisionConfig.backendBaseUrl);
      final reachable = await client.checkHealth();
      if (!reachable) {
        debugPrint(
            '[VisionPipeline] ⚠️  Backend health check failed — '
            '${VisionConfig.backendBaseUrl} may be unreachable. '
            'analyzeShelf calls will fail until the API is available.');
      }

      _isInitialized = true;
      debugPrint(
          '[VisionPipeline] ✅  Ready (backend mode)  '
          'API: ${VisionConfig.backendBaseUrl}');
    }
  }

  // ── Main entry point ─────────────────────────────────────────────────────────

  /// Runs the full analysis pipeline on [shelfImage] and returns a
  /// [ShelfAnalysis] containing matched products, unmatched crops, facing
  /// counts, shelf-share percentage, and per-step timing.
  ///
  /// Steps 1–2 (YOLO + filtering) always run on-device.
  /// Steps 3–5 run on-device or via the backend API depending on
  /// [VisionConfig.onDevice].  The returned [ShelfAnalysis] is identical
  /// in structure regardless of which path was taken.
  Future<ShelfAnalysis> analyzeShelf(img.Image shelfImage) async {
    _assertReady();

    final totalSw = Stopwatch()..start();

    // ── Step 1: YOLO detection ────────────────────────────────────────────────
    //
    // Always runs on-device.  InferenceService.detect() accepts a File, so we
    // encode [shelfImage] to a temporary JPEG, run detection, then delete it.
    final sw1 = Stopwatch()..start();
    final rawDetections = await _runYolo(shelfImage);
    final yoloMs = sw1.elapsedMilliseconds;
    debugPrint('[VisionPipeline] Step 1: YOLO detected '
        '${rawDetections.length} products in ${yoloMs}ms');

    // ── Step 2: Filter ────────────────────────────────────────────────────────
    //
    // Drop low-confidence, tiny, and overlapping boxes.
    // Crops are only extracted in on-device mode (Step 3 below).
    final sw2 = Stopwatch()..start();

    final filter = const CropFilter(
      confThreshold: 0.30,
      minSize:       32,
      iouThreshold:  0.80,
    );
    final filtered = filter.filter(rawDetections);
    final removed  = rawDetections.length - filtered.length;

    final filterMs = sw2.elapsedMilliseconds;
    debugPrint('[VisionPipeline] Step 2: ${filtered.length} crops after filtering '
        '(removed $removed) in ${filterMs}ms');

    // Early-exit when nothing survives the filter (same for both modes).
    if (filtered.isEmpty) {
      totalSw.stop();
      debugPrint('[VisionPipeline] Early exit — 0 detections after filter');
      return ShelfAnalysis(
        totalDetections:   0,
        matchedCount:      0,
        unknownCount:      0,
        shelfSharePercent: 0.0,
        products:          [],
        unknowns:          [],
        timing: PipelineTiming(
          yoloMs:      yoloMs,
          filterMs:    filterMs,
          embeddingMs: 0,
          matchingMs:  0,
          totalMs:     totalSw.elapsedMilliseconds,
        ),
      );
    }

    // ── Steps 3–5: On-device or backend ───────────────────────────────────────

    if (VisionConfig.onDevice) {
      // ══ EXISTING ON-DEVICE FLOW — unchanged ══════════════════════════════

      // Step 3: Extract crops from the in-memory image, then embed with DINOv3.
      final crops = _extractCrops(shelfImage, filtered);

      final sw3 = Stopwatch()..start();
      final embeddings = await dinov3Service.getEmbeddings(crops);
      final embeddingMs = sw3.elapsedMilliseconds;
      debugPrint('[VisionPipeline] Step 3: Generated ${embeddings.length} '
          'embeddings in ${embeddingMs}ms');

      // Step 4: Catalog matching.
      final sw4 = Stopwatch()..start();
      final matchResults = catalogService.matchCrops(embeddings);
      final matchedCount = matchResults.where((r) => r.isMatched).length;
      final matchingMs   = sw4.elapsedMilliseconds;
      debugPrint('[VisionPipeline] Step 4: Matched '
          '$matchedCount/${matchResults.length} products in ${matchingMs}ms');

      // Step 5: Aggregate.
      totalSw.stop();
      final analysis = _aggregate(
        detections:   filtered,
        crops:        crops,
        matchResults: matchResults,
        timing: PipelineTiming(
          yoloMs:      yoloMs,
          filterMs:    filterMs,
          embeddingMs: embeddingMs,
          matchingMs:  matchingMs,
          totalMs:     totalSw.elapsedMilliseconds,
        ),
      );

      debugPrint(
          '[VisionPipeline] Complete (on-device) — '
          '${analysis.matchedCount} matched, '
          '${analysis.unknownCount} unknown, '
          '${analysis.shelfSharePercent.toStringAsFixed(1)}% shelf share  '
          '${analysis.timing.totalMs}ms');

      return analysis;

    } else {
      // ══ BACKEND FLOW ═════════════════════════════════════════════════════
      //
      // Send the image + YOLO detections to the backend API.
      // DINOv3 and CatalogService are NOT used here — they were not loaded.

      File? tempFile;
      try {
        tempFile = await _writeTempJpeg(shelfImage);

        final client = BackendApiClient(baseUrl: VisionConfig.backendBaseUrl);
        final yoloDetections = filtered
            .map(YoloDetection.fromDetectionResult)
            .toList();

        final backendAnalysis = await client.analyzeShelf(
          imageFile:       tempFile,
          detections:      yoloDetections,
          imageWidth:      shelfImage.width,
          imageHeight:     shelfImage.height,
          enableOcr:       VisionConfig.enableOcr,
          scoreThreshold:  VisionConfig.scoreThreshold,
          marginThreshold: VisionConfig.marginThreshold,
        );

        totalSw.stop();

        // Patch timing: replace the placeholder yoloMs/filterMs (0) that the
        // client set with the actual on-device measurements from Steps 1–2.
        final analysis = ShelfAnalysis(
          totalDetections:   backendAnalysis.totalDetections,
          matchedCount:      backendAnalysis.matchedCount,
          unknownCount:      backendAnalysis.unknownCount,
          shelfSharePercent: backendAnalysis.shelfSharePercent,
          products:          backendAnalysis.products,
          unknowns:          backendAnalysis.unknowns,
          timing: PipelineTiming(
            yoloMs:      yoloMs,
            filterMs:    filterMs,
            embeddingMs: backendAnalysis.timing.embeddingMs, // server-side time
            matchingMs:  backendAnalysis.timing.matchingMs,
            totalMs:     totalSw.elapsedMilliseconds,
          ),
        );

        debugPrint(
            '[VisionPipeline] Complete (backend) — '
            '${analysis.matchedCount} matched, '
            '${analysis.unknownCount} unknown, '
            '${analysis.shelfSharePercent.toStringAsFixed(1)}% shelf share  '
            '${analysis.timing.totalMs}ms');

        return analysis;

      } finally {
        try { await tempFile?.delete(); } catch (_) {}
      }
    }
  }

  // ── Private helpers ───────────────────────────────────────────────────────────

  /// Encodes [shelfImage] to a temp JPEG file, runs YOLO, then deletes the file.
  ///
  /// The round-trip (encode → detect → delete) is necessary because
  /// [InferenceService.detect] accepts a [File].  Quality 92 JPEG preserves
  /// enough detail for bounding-box detection; DINOv3 crops are extracted
  /// from the original in-memory [img.Image] (Step 2), not from this file.
  Future<List<DetectionResult>> _runYolo(img.Image shelfImage) async {
    File? tempFile;
    try {
      final jpegBytes = img.encodeJpg(shelfImage, quality: 92);
      final tempDir   = Directory.systemTemp;
      final stamp     = DateTime.now().microsecondsSinceEpoch;
      tempFile = File('${tempDir.path}/vp_yolo_$stamp.jpg');
      await tempFile.writeAsBytes(jpegBytes, flush: true);
      return await yoloService.detect(tempFile);
    } finally {
      // Best-effort cleanup — do not let a delete failure abort the pipeline.
      try {
        await tempFile?.delete();
      } catch (_) {}
    }
  }

  /// Writes [image] to a temporary JPEG file (quality 92) and returns the
  /// [File].  The caller is responsible for deleting it after use.
  /// Used in backend mode to produce a file for [BackendApiClient.analyzeShelf].
  Future<File> _writeTempJpeg(img.Image image) async {
    final jpegBytes = img.encodeJpg(image, quality: 92);
    final stamp     = DateTime.now().microsecondsSinceEpoch;
    final file      = File('${Directory.systemTemp.path}/vp_backend_$stamp.jpg');
    await file.writeAsBytes(jpegBytes, flush: true);
    return file;
  }

  /// Cuts one [img.Image] crop per entry in [detections] from [shelfImage].
  /// Coordinates are clamped so they never exceed image bounds.
  List<img.Image> _extractCrops(
    img.Image shelfImage,
    List<DetectionResult> detections,
  ) {
    return detections.map((det) {
      final b = det.boundingBox;
      final x = b.left.round().clamp(0, shelfImage.width  - 1);
      final y = b.top.round().clamp(0, shelfImage.height - 1);
      final w = b.width.round().clamp(1,  shelfImage.width  - x);
      final h = b.height.round().clamp(1, shelfImage.height - y);
      return img.copyCrop(shelfImage, x: x, y: y, width: w, height: h);
    }).toList();
  }

  /// Combines per-crop [MatchResult]s into [DetectedProduct] / [UnknownDetection]
  /// lists and computes shelf-share.
  ShelfAnalysis _aggregate({
    required List<DetectionResult> detections,
    required List<img.Image> crops,
    required List<MatchResult> matchResults,
    required PipelineTiming timing,
  }) {
    // Accumulate matched crops by product name.
    final productAccs = <String, _ProductAcc>{};
    final unknowns    = <UnknownDetection>[];

    for (int i = 0; i < matchResults.length; i++) {
      final result = matchResults[i];
      final det    = detections[i];
      final crop   = crops[i];

      if (result.isMatched) {
        // putIfAbsent returns the value — apply cascade to the _ProductAcc,
        // not to the Map (hence the extra parentheses).
        (productAccs.putIfAbsent(
          result.productName,
          () => _ProductAcc(result.productName),
        ))
            ..scores.add(result.score)
            ..boxes.add(det.boundingBox)
            ..cropImages.add(crop);
      } else {
        // nearestProduct: use result.productName which is the top scoring
        // catalog entry (could be 'Unknown' / 'Error' for edge cases).
        final nearest = (result.productName == 'Error' ||
                result.productName == 'Unknown')
            ? _topScoreProduct(result)
            : result.productName;

        unknowns.add(UnknownDetection(
          boundingBox:    det.boundingBox,
          cropImage:      crop,
          bestScore:      result.score,
          nearestProduct: nearest,
        ));
      }
    }

    // Convert accumulators → DetectedProduct, sorted by facing count desc.
    final products = productAccs.values
        .map((acc) => DetectedProduct(
              productName:   acc.name,
              facingCount:   acc.scores.length,
              avgSimilarity: acc.scores.reduce((a, b) => a + b) / acc.scores.length,
              boundingBoxes: acc.boxes,
              cropImages:    acc.cropImages,
            ))
        .toList()
      ..sort((a, b) => b.facingCount.compareTo(a.facingCount));

    final matchedCount      = matchResults.where((r) => r.isMatched).length;
    final totalDetections   = detections.length;
    final shelfSharePercent = totalDetections > 0
        ? matchedCount / totalDetections * 100.0
        : 0.0;

    return ShelfAnalysis(
      totalDetections:   totalDetections,
      matchedCount:      matchedCount,
      unknownCount:      unknowns.length,
      shelfSharePercent: shelfSharePercent,
      products:          products,
      unknowns:          unknowns,
      timing:            timing,
    );
  }

  /// Returns the catalog product name with the highest score from [result.allScores],
  /// or "Unknown" when allScores is empty.
  static String _topScoreProduct(MatchResult result) {
    if (result.allScores.isEmpty) return 'Unknown';
    return result.allScores.entries
        .reduce((a, b) => a.value > b.value ? a : b)
        .key;
  }

  void _assertReady() {
    if (!_isInitialized) {
      throw StateError(
          '[VisionPipeline] Not initialized — call await initialize() first.');
    }
  }
}

// ── Internal accumulator ──────────────────────────────────────────────────────

/// Mutable accumulator used during [VisionPipeline._aggregate].
class _ProductAcc {
  final String name;
  final List<double>    scores     = [];
  final List<Rect>      boxes      = [];
  final List<img.Image> cropImages = [];

  _ProductAcc(this.name);
}
