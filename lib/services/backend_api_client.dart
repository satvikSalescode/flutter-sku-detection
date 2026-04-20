import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart' show Rect;
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;

import '../models/detection_result.dart';
import 'vision_pipeline.dart';

// ── Request DTO ───────────────────────────────────────────────────────────────

/// Serialisable representation of a single YOLO detection for the API request.
/// Created from a [DetectionResult] via [YoloDetection.fromDetectionResult].
class YoloDetection {
  final Rect   boundingBox;
  final double confidence;
  final int    classId;

  const YoloDetection({
    required this.boundingBox,
    required this.confidence,
    required this.classId,
  });

  factory YoloDetection.fromDetectionResult(DetectionResult r) => YoloDetection(
        boundingBox: r.boundingBox,
        confidence:  r.confidence,
        classId:     r.classIndex,
      );

  Map<String, dynamic> toJson() => {
        'bbox': [
          boundingBox.left,
          boundingBox.top,
          boundingBox.right,
          boundingBox.bottom,
        ],
        'confidence': confidence,
        'class_id':   classId,
      };
}

// ── Client ────────────────────────────────────────────────────────────────────

/// Sends a shelf image + YOLO detections to the backend API and returns a
/// [ShelfAnalysis] — the same model returned by the on-device pipeline so the
/// UI remains completely unaware of which path was taken.
///
/// Endpoint: POST {baseUrl}/api/v1/analyze
///
/// The client:
///   • Re-encodes the image to JPEG quality 85 before base64 (smaller payload)
///   • Sets a 30-second timeout for the entire round-trip
///   • Logs request payload size (MB) and server response time (ms)
///   • Maps any network/server error into a descriptive [BackendApiException]
class BackendApiClient {
  final String baseUrl;

  /// HTTP timeout for the full round-trip (encode → send → receive → parse).
  static const _timeout = Duration(seconds: 30);

  const BackendApiClient({required this.baseUrl});

  // ── Health check ───────────────────────────────────────────────────────────

  /// Lightweight bool health check used at pipeline initialisation.
  /// Does not throw — false is treated as a warning only.
  Future<bool> checkHealth() async {
    final result = await checkHealthDetailed();
    return result.connected;
  }

  /// Rich health check that parses the backend response for model/device info.
  ///
  /// Expected response from GET {baseUrl}/health:
  /// ```json
  /// { "status": "ok", "model": "dinov3-vitl16", "device": "cuda" }
  /// ```
  /// Returns a [HealthCheckResult] regardless of outcome — never throws.
  Future<HealthCheckResult> checkHealthDetailed() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 10));

      debugPrint('[BackendApiClient] Health → ${response.statusCode}');

      if (response.statusCode < 200 || response.statusCode >= 300) {
        return HealthCheckResult.failed(
            'Server returned ${response.statusCode}');
      }

      // Try to parse model / device from response body.
      String model  = '';
      String device = '';
      try {
        final json = jsonDecode(response.body) as Map<String, dynamic>;
        model  = json['model']  as String? ?? '';
        device = json['device'] as String? ?? '';
      } catch (_) {
        // Health endpoint might return plain "ok" — that's fine.
      }

      return HealthCheckResult.ok(model: model, device: device);
    } catch (e) {
      debugPrint('[BackendApiClient] Health check failed: $e');
      return HealthCheckResult.failed(e.toString()
          .replaceAll('SocketException: ', '')
          .replaceAll('TimeoutException: ', 'Timeout — '));
    }
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  Future<ShelfAnalysis> analyzeShelf({
    required File             imageFile,
    required List<YoloDetection> detections,
    required int              imageWidth,
    required int              imageHeight,
    bool    enableOcr        = false,
    double? scoreThreshold,
    double? marginThreshold,
  }) async {
    // ── 1. Compress image to JPEG-85 and base64-encode ─────────────────────
    final compressedBytes = await _compressImage(imageFile);
    final base64Image     = base64Encode(compressedBytes);
    final payloadMb       = compressedBytes.lengthInBytes / (1024 * 1024);

    debugPrint('[BackendApiClient] Payload: ${payloadMb.toStringAsFixed(2)} MB  '
        '(${detections.length} detections)');

    // ── 2. Build request JSON ───────────────────────────────────────────────
    final body = <String, dynamic>{
      'image':        base64Image,
      'detections':   detections.map((d) => d.toJson()).toList(),
      'image_width':  imageWidth,
      'image_height': imageHeight,
      'enable_ocr':   enableOcr,
    };
    if (scoreThreshold  != null) body['score_threshold']  = scoreThreshold;
    if (marginThreshold != null) body['margin_threshold'] = marginThreshold;

    // ── 3. POST to backend ──────────────────────────────────────────────────
    final uri = Uri.parse('$baseUrl/api/v1/analyze');
    final sw  = Stopwatch()..start();

    http.Response response;
    try {
      response = await http
          .post(
            uri,
            headers: {'Content-Type': 'application/json'},
            body:    jsonEncode(body),
          )
          .timeout(_timeout);
    } on SocketException catch (e) {
      throw BackendApiException('Network error: ${e.message}');
    } on HttpException catch (e) {
      throw BackendApiException('HTTP error: ${e.message}');
    } on Exception catch (e) {
      // Covers TimeoutException and anything else
      throw BackendApiException('Request failed: $e');
    }

    sw.stop();
    debugPrint('[BackendApiClient] Response: ${response.statusCode}  '
        '${sw.elapsedMilliseconds}ms');

    // ── 4. Parse response ───────────────────────────────────────────────────
    if (response.statusCode != 200) {
      throw BackendApiException(
          'Server returned ${response.statusCode}: ${response.body}');
    }

    final Map<String, dynamic> json;
    try {
      json = jsonDecode(response.body) as Map<String, dynamic>;
    } catch (_) {
      throw BackendApiException('Invalid JSON response from server');
    }

    // ── 5. Map to ShelfAnalysis ─────────────────────────────────────────────
    return _parseResponse(json, sw.elapsedMilliseconds);
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  /// Decodes [imageFile], re-encodes as JPEG at quality 85, returns bytes.
  /// Run on an isolate via [compute] to keep the UI thread free.
  Future<Uint8List> _compressImage(File imageFile) async {
    final bytes = await imageFile.readAsBytes();
    return await compute(_compressIsolate, bytes);
  }

  static Uint8List _compressIsolate(Uint8List inputBytes) {
    final decoded = img.decodeImage(inputBytes);
    if (decoded == null) {
      // Fall back to original bytes if we can't decode
      return inputBytes;
    }
    return Uint8List.fromList(img.encodeJpg(decoded, quality: 85));
  }

  /// Converts the backend JSON response into a [ShelfAnalysis].
  ///
  /// Actual backend response shape:
  /// ```json
  /// {
  ///   "status": "success",
  ///   "processing_time_ms": 412.7,
  ///   "total_detections": 5,
  ///   "matched_count": 3,
  ///   "unknown_count": 2,
  ///   "shelf_share_percent": 60.0,
  ///   "results": [
  ///     {
  ///       "detection_index": 0,
  ///       "bbox": [x1, y1, x2, y2],
  ///       "status": "matched" | "unknown",
  ///       "product_name": "COKECNRED" | null,
  ///       "product_sku": "COKECNRED"  | null,
  ///       "similarity_score": 0.87,
  ///       "margin": 0.18,
  ///       "match_type": "visual",
  ///       "ocr_text": null
  ///     }
  ///   ],
  ///   "product_summary": [
  ///     {
  ///       "product_name": "COKECNRED",
  ///       "product_sku":  "COKECNRED",
  ///       "facing_count": 3,
  ///       "avg_similarity": 0.87
  ///     }
  ///   ]
  /// }
  /// ```
  ShelfAnalysis _parseResponse(Map<String, dynamic> json, int roundTripMs) {
    final rawResults  = (json['results']         as List<dynamic>?) ?? [];
    final rawSummary  = (json['product_summary'] as List<dynamic>?) ?? [];
    final serverMs    = (json['processing_time_ms'] as num?)?.toInt() ?? 0;

    // ── Split results into matched / unknown ──────────────────────────────
    // Group matched per-detection entries by product_name so we can attach
    // the correct bounding boxes to each DetectedProduct.
    final matchedByProduct = <String, List<Map<String, dynamic>>>{};
    final unknownResults   = <Map<String, dynamic>>[];

    for (final r in rawResults) {
      final map    = r as Map<String, dynamic>;
      final status = map['status'] as String? ?? 'unknown';
      final name   = map['product_name'] as String?;
      if (status == 'matched' && name != null) {
        matchedByProduct.putIfAbsent(name, () => []).add(map);
      } else {
        unknownResults.add(map);
      }
    }

    // ── DetectedProduct list — built from product_summary + matched results
    final products = rawSummary.map((s) {
      final map     = s as Map<String, dynamic>;
      final name    = map['product_name'] as String? ?? 'Unknown';
      final matched = matchedByProduct[name] ?? [];

      // Collect bounding boxes for every facing of this product.
      final boxes = matched
          .map((r) => _parseBox(r['bbox'] as List<dynamic>))
          .toList();

      // OCR text: first non-null entry across all facings.
      final ocrText = matched
          .map((r) => r['ocr_text'] as String?)
          .firstWhere((t) => t != null && t.isNotEmpty, orElse: () => null);

      // Match method: take from first facing (same for all).
      final matchType = matched.isNotEmpty
          ? matched.first['match_type'] as String?
          : null;

      return DetectedProduct(
        productName:   name,
        facingCount:   (map['facing_count'] as int?) ?? matched.length,
        avgSimilarity: (map['avg_similarity'] as num?)?.toDouble() ?? 0.0,
        boundingBoxes: boxes,
        cropImages:    [],       // no local crops in backend mode
        ocrText:       ocrText,
        matchMethod:   matchType,
      );
    }).toList()
      ..sort((a, b) => b.facingCount.compareTo(a.facingCount));

    // ── UnknownDetection list — built from results where status == "unknown"
    final placeholder = img.Image(width: 1, height: 1);
    final unknowns = unknownResults.map((r) {
      return UnknownDetection(
        boundingBox:    _parseBox(r['bbox'] as List<dynamic>),
        cropImage:      placeholder,
        bestScore:      (r['similarity_score'] as num?)?.toDouble() ?? 0.0,
        nearestProduct: r['product_name'] as String? ?? 'Unknown',
        ocrText:        r['ocr_text']     as String?,
      );
    }).toList();

    // ── Counts / share ────────────────────────────────────────────────────
    final totalDetections   = (json['total_detections']   as int?)
        ?? (products.fold<int>(0, (s, p) => s + p.facingCount) + unknowns.length);
    final matchedCount      = (json['matched_count']      as int?)
        ?? products.fold<int>(0, (s, p) => s + p.facingCount);
    final unknownCount      = (json['unknown_count']      as int?) ?? unknowns.length;
    final shelfSharePercent = (json['shelf_share_percent'] as num?)?.toDouble()
        ?? (totalDetections > 0 ? matchedCount / totalDetections * 100.0 : 0.0);

    // ── Timing ────────────────────────────────────────────────────────────
    final timing = PipelineTiming(
      yoloMs:      0,          // patched upstream in VisionPipeline
      filterMs:    0,
      embeddingMs: serverMs,   // server processing time from response
      matchingMs:  0,
      totalMs:     roundTripMs,
    );

    return ShelfAnalysis(
      totalDetections:   totalDetections,
      matchedCount:      matchedCount,
      unknownCount:      unknownCount,
      shelfSharePercent: shelfSharePercent,
      products:          products,
      unknowns:          unknowns,
      timing:            timing,
    );
  }

  // ── Geometry helpers ───────────────────────────────────────────────────────

  static Rect _parseBox(List<dynamic> coords) {
    return Rect.fromLTRB(
      (coords[0] as num).toDouble(),
      (coords[1] as num).toDouble(),
      (coords[2] as num).toDouble(),
      (coords[3] as num).toDouble(),
    );
  }

}

// ── Health result ─────────────────────────────────────────────────────────────

/// Result of a backend health check.
class HealthCheckResult {
  final bool    connected;
  /// e.g. "dinov3-vitl16" — populated when [connected] is true.
  final String  model;
  /// e.g. "cuda" or "cpu" — populated when [connected] is true.
  final String  device;
  /// Human-readable error message — populated when [connected] is false.
  final String  error;

  const HealthCheckResult._({
    required this.connected,
    required this.model,
    required this.device,
    required this.error,
  });

  factory HealthCheckResult.ok({String model = '', String device = ''}) =>
      HealthCheckResult._(
          connected: true, model: model, device: device, error: '');

  factory HealthCheckResult.failed(String error) =>
      HealthCheckResult._(
          connected: false, model: '', device: '', error: error);
}

// ── Exception ─────────────────────────────────────────────────────────────────

class BackendApiException implements Exception {
  final String message;
  const BackendApiException(this.message);

  @override
  String toString() => 'BackendApiException: $message';
}
