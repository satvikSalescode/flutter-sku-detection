import 'dart:convert';
import 'dart:io';
import 'dart:math' show sqrt;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

// ── Asset path (bundled fallback) ──────────────────────────────────────────────

const String kCatalogAsset = 'assets/catalog_embeddings.json';

/// Filename written to the app's documents directory.
const String kLocalCatalogFilename = 'catalog_embeddings.json';

/// The 4 canonical angle slots every product is expected to fill.
const List<String> kStandardAngles       = ['front', 'back', 'left', 'right'];
const List<String> kStandardAngleLabels  = ['Front', 'Back', 'Left', 'Right'];

// ── Data models ────────────────────────────────────────────────────────────────

/// One product in the reference catalog.
///
/// [angles] and [embeddings] are parallel lists — angles[i] names the view
/// angle for embeddings[i].  Both lists always have the same length.
class ProductReference {
  final String name;

  /// Angle keys for each embedding (e.g. "front", "back", "left", "right").
  final List<String> angles;

  /// L2-normalised 384-dim embeddings, one per angle.
  final List<List<double>> embeddings;

  const ProductReference({
    required this.name,
    required this.angles,
    required this.embeddings,
  });

  int get angleCount => embeddings.length;

  /// Returns true when an embedding exists for [angleKey] (case-insensitive).
  bool hasAngle(String angleKey) =>
      angles.any((a) => a.toLowerCase() == angleKey.toLowerCase());

  /// Index of [angleKey] in [angles], or -1 if absent.
  int angleIndex(String angleKey) => angles.indexWhere(
      (a) => a.toLowerCase() == angleKey.toLowerCase());

  /// How many of the 4 standard angles (front/back/left/right) are filled.
  int get standardAnglesFilled =>
      kStandardAngles.where(hasAngle).length;
}

/// Result returned by [CatalogService.matchCrop] for one query embedding.
class MatchResult {
  final String productName;
  final double score;
  final double margin;
  final bool isMatched;
  final Map<String, double> allScores;

  const MatchResult({
    required this.productName,
    required this.score,
    required this.margin,
    required this.isMatched,
    required this.allScores,
  });

  @override
  String toString() =>
      'MatchResult(product=$productName, score=${score.toStringAsFixed(3)}, '
      'margin=${margin.toStringAsFixed(3)}, matched=$isMatched)';
}

// ── Service ────────────────────────────────────────────────────────────────────

/// Loads, persists, and serves the DINOv3 catalog.
///
/// Load priority:
///   1. Local documents directory  (updated via [updateProductEmbedding] or
///      [loadFromUrl] — persists across app restarts)
///   2. Bundled asset              (read-only, shipped with the APK/IPA)
///
/// Typical lifecycle:
///   final catalog = CatalogService();
///   await catalog.load();
///   final result = catalog.matchCrop(embedding);
class CatalogService {
  // ── Catalog state ───────────────────────────────────────────────────────────

  List<ProductReference> products      = [];
  String catalogModel                  = '';
  int    embeddingDim                  = 384;

  /// ISO-8601 timestamp of the last catalog update.
  String catalogVersion                = '';

  /// Whether the catalog was loaded from a local file (vs. bundled asset).
  bool   loadedFromLocal               = false;

  // ── Flat index ──────────────────────────────────────────────────────────────

  List<List<double>> flatEmbeddings    = [];
  List<String>       flatProductNames  = [];

  // ── Thresholds ──────────────────────────────────────────────────────────────

  double scoreThreshold  = 0.65;

  // ── State ───────────────────────────────────────────────────────────────────

  bool _isLoaded = false;
  bool get isLoaded  => _isLoaded;
  bool get isEmpty   => products.isEmpty;
  int  get numAngles => flatEmbeddings.length;

  // ── Local file path ─────────────────────────────────────────────────────────

  /// Returns the [File] in the app's documents directory.
  /// The file may or may not exist yet.
  static Future<File> localFile() async {
    final dir = await getApplicationDocumentsDirectory();
    return File('${dir.path}/$kLocalCatalogFilename');
  }

  /// Size of the local catalog file in bytes, or 0 if it doesn't exist yet.
  static Future<int> localFileSizeBytes() async {
    final f = await localFile();
    return f.existsSync() ? await f.length() : 0;
  }

  // ── Load ────────────────────────────────────────────────────────────────────

  /// Loads the catalog — prefers a locally saved file, falls back to the
  /// bundled asset.  Calling [load] a second time is a no-op.
  Future<void> load() async {
    if (_isLoaded) return;

    try {
      final local = await localFile();

      if (local.existsSync()) {
        // ── Local file found ───────────────────────────────────────────────
        final raw  = await local.readAsString();
        final json = jsonDecode(raw) as Map<String, dynamic>;
        _parseJson(json);
        _buildFlatIndex();
        loadedFromLocal = true;
        _isLoaded       = true;
        debugPrint('[CatalogService] ✅  Loaded from local storage: '
            '${products.length} products, ${flatEmbeddings.length} embeddings '
            '(dim=$embeddingDim, v=$catalogVersion)');
      } else {
        // ── Fall back to bundled asset ─────────────────────────────────────
        await _loadFromAsset();
      }
    } on FlutterError catch (e) {
      debugPrint('[CatalogService] ❌  Asset not found: $e');
      debugPrint('[CatalogService]    Run generate_catalog.py, '
          'copy output to assets/catalog_embeddings.json, then rebuild.');
      rethrow;
    } catch (e) {
      debugPrint('[CatalogService] ❌  Parse error: $e');
      rethrow;
    }
  }

  /// Loads the bundled asset directly (used internally and for "reset").
  Future<void> _loadFromAsset() async {
    final raw  = await rootBundle.loadString(kCatalogAsset);
    final json = jsonDecode(raw) as Map<String, dynamic>;
    _parseJson(json);
    _buildFlatIndex();
    loadedFromLocal = false;
    _isLoaded       = true;
    debugPrint('[CatalogService] ✅  Loaded from bundled asset: '
        '${products.length} products, ${flatEmbeddings.length} embeddings '
        '(dim=$embeddingDim, v=$catalogVersion)');
  }

  // ── Save ────────────────────────────────────────────────────────────────────

  /// Serialises the current catalog and writes it to local storage.
  Future<void> save() async {
    final json = _toJson();
    final text = const JsonEncoder.withIndent('  ').convert(json);
    final f    = await localFile();
    await f.writeAsString(text, flush: true);
    loadedFromLocal = true;
    debugPrint('[CatalogService] 💾  Saved to ${f.path}  '
        '(${(text.length / 1024).toStringAsFixed(1)} KB)');
  }

  // ── Update one embedding ────────────────────────────────────────────────────

  /// Adds or replaces the embedding for [angleKey] in product [productName].
  ///
  /// If the product doesn't exist yet it is created with a single angle.
  /// After updating, the flat index is rebuilt and the catalog is saved to
  /// local storage.
  Future<void> updateProductEmbedding({
    required String productName,
    required String angleKey,
    required List<double> embedding,
  }) async {
    final idx = products.indexWhere((p) => p.name == productName);

    if (idx >= 0) {
      final p        = products[idx];
      final angleIdx = p.angleIndex(angleKey);

      if (angleIdx >= 0) {
        // Replace existing embedding for this angle
        final newEmbs = List<List<double>>.from(p.embeddings);
        newEmbs[angleIdx] = embedding;
        products[idx] = ProductReference(
            name: p.name, angles: p.angles, embeddings: newEmbs);
      } else {
        // Append new angle
        products[idx] = ProductReference(
          name:       p.name,
          angles:     [...p.angles, angleKey],
          embeddings: [...p.embeddings, embedding],
        );
      }
    } else {
      // Create new product
      products.add(ProductReference(
        name:       productName,
        angles:     [angleKey],
        embeddings: [embedding],
      ));
    }

    catalogVersion = DateTime.now().toIso8601String();
    _buildFlatIndex();
    await save();

    debugPrint('[CatalogService] ✅  Updated: $productName / $angleKey  '
        '(total ${flatEmbeddings.length} vectors)');
  }

  // ── URL sync ────────────────────────────────────────────────────────────────

  /// Downloads a catalog JSON from [url], replaces the current catalog, and
  /// saves it to local storage.
  ///
  /// Throws on HTTP errors, invalid JSON, or network failure.
  Future<void> loadFromUrl(String url) async {
    debugPrint('[CatalogService] 🌐  Fetching catalog from $url');

    final client = HttpClient();
    try {
      final request  = await client.getUrl(Uri.parse(url));
      final response = await request.close();

      if (response.statusCode != 200) {
        throw Exception(
            'HTTP ${response.statusCode}: $url');
      }

      final buffer = StringBuffer();
      await for (final chunk in response.transform(utf8.decoder)) {
        buffer.write(chunk);
      }
      final jsonStr = buffer.toString();
      final json    = jsonDecode(jsonStr) as Map<String, dynamic>;

      _parseJson(json);
      _buildFlatIndex();
      _isLoaded       = true;
      loadedFromLocal = true;
      catalogVersion  = DateTime.now().toIso8601String();

      await save();

      debugPrint('[CatalogService] ✅  Synced from URL: '
          '${products.length} products, ${flatEmbeddings.length} embeddings');
    } finally {
      client.close();
    }
  }

  // ── Matching (unchanged logic) ──────────────────────────────────────────────

  MatchResult matchCrop(List<double> cropEmbedding) {
    if (isEmpty) {
      return const MatchResult(
          productName: 'Unknown', score: 0, margin: 0,
          isMatched: false, allScores: {});
    }

    if (_isZeroVector(cropEmbedding)) {
      debugPrint('[CatalogService] ⚠️  Zero embedding received.');
      return const MatchResult(
          productName: 'Error', score: 0, margin: 0,
          isMatched: false, allScores: {});
    }

    final flatScores = List<double>.generate(
      flatEmbeddings.length,
      (i) => _dot(cropEmbedding, flatEmbeddings[i]),
    );

    final productBest = <String, double>{};
    for (int i = 0; i < flatEmbeddings.length; i++) {
      final name = flatProductNames[i];
      final s    = flatScores[i];
      if (!productBest.containsKey(name) || s > productBest[name]!) {
        productBest[name] = s;
      }
    }

    final ranked      = productBest.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    final best        = ranked[0];
    final secondScore = ranked.length > 1 ? ranked[1].value : best.value;
    final margin      = best.value - secondScore;
    final matched     = best.value >= scoreThreshold;

    return MatchResult(
      productName: best.key,
      score:       best.value,
      margin:      margin,
      isMatched:   matched,
      allScores:   productBest,
    );
  }

  List<MatchResult> matchCrops(List<List<double>> cropEmbeddings) {
    if (cropEmbeddings.isEmpty) return [];

    final numCrops = cropEmbeddings.length;
    final numFlat  = flatEmbeddings.length;

    if (isEmpty) {
      return List.generate(numCrops, (_) => const MatchResult(
          productName: 'Unknown', score: 0, margin: 0,
          isMatched: false, allScores: {}));
    }

    final sim = List<double>.filled(numCrops * numFlat, 0.0);
    for (int c = 0; c < numCrops; c++) {
      final base = c * numFlat;
      for (int i = 0; i < numFlat; i++) {
        sim[base + i] = _dot(cropEmbeddings[c], flatEmbeddings[i]);
      }
    }

    final results = <MatchResult>[];
    for (int c = 0; c < numCrops; c++) {
      if (_isZeroVector(cropEmbeddings[c])) {
        results.add(const MatchResult(
            productName: 'Error', score: 0, margin: 0,
            isMatched: false, allScores: {}));
        continue;
      }

      final base        = c * numFlat;
      final productBest = <String, double>{};
      for (int i = 0; i < numFlat; i++) {
        final name = flatProductNames[i];
        final s    = sim[base + i];
        if (!productBest.containsKey(name) || s > productBest[name]!) {
          productBest[name] = s;
        }
      }

      final ranked      = productBest.entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value));
      final best        = ranked[0];
      final secondScore = ranked.length > 1 ? ranked[1].value : best.value;
      final margin      = best.value - secondScore;
      final matched     = best.value >= scoreThreshold;

      results.add(MatchResult(
        productName: best.key,
        score:       best.value,
        margin:      margin,
        isMatched:   matched,
        allScores:   productBest,
      ));
    }
    return results;
  }

  // ── Diagnostics ─────────────────────────────────────────────────────────────

  void debugDump() {
    debugPrint('[CatalogService] ── Catalog dump ──');
    debugPrint('  model=$catalogModel  dim=$embeddingDim  v=$catalogVersion');
    for (final p in products) {
      debugPrint('  ${p.name}  angles=${p.angles}  (${p.angleCount} embeddings)');
    }
    debugPrint('  flat index: ${flatEmbeddings.length} vectors');
  }

  // ── JSON parsing / serialisation ─────────────────────────────────────────────

  void _parseJson(Map<String, dynamic> json) {
    catalogModel   = json['model']         as String? ?? '';
    embeddingDim   = json['embedding_dim'] as int?    ?? 384;
    catalogVersion = json['version']       as String? ?? '';

    final productsMap = json['products'] as Map<String, dynamic>? ?? {};
    products = [];

    for (final entry in productsMap.entries) {
      final name    = entry.key;
      final data    = entry.value as Map<String, dynamic>;
      final rawEmbs = data['embeddings'] as List<dynamic>;

      final embeddings = rawEmbs.map((row) {
        return (row as List<dynamic>)
            .map((v) => (v as num).toDouble())
            .toList();
      }).toList();

      // Parse stored angles; fall back to generic names if absent.
      final rawAngles = data['angles'] as List<dynamic>?;
      final angles = rawAngles != null
          ? rawAngles.map((a) => a as String).toList()
          : List.generate(embeddings.length, (i) => 'angle_$i');

      if (embeddings.isEmpty) {
        debugPrint('[CatalogService] ⚠️  $name has no embeddings — skipping.');
        continue;
      }

      products.add(ProductReference(
          name: name, angles: angles, embeddings: embeddings));
    }
  }

  Map<String, dynamic> _toJson() {
    final prods = <String, dynamic>{};
    for (final p in products) {
      prods[p.name] = {
        'angles':     p.angles,
        'embeddings': p.embeddings,
      };
    }
    return {
      'model':          catalogModel.isEmpty ? 'dinov3-small' : catalogModel,
      'embedding_dim':  embeddingDim,
      'version':        catalogVersion.isEmpty
          ? DateTime.now().toIso8601String()
          : catalogVersion,
      'num_products':   products.length,
      'num_embeddings': flatEmbeddings.length,
      'products':       prods,
    };
  }

  void _buildFlatIndex() {
    flatEmbeddings   = [];
    flatProductNames = [];
    for (final p in products) {
      for (final emb in p.embeddings) {
        flatEmbeddings.add(emb);
        flatProductNames.add(p.name);
      }
    }
  }

  // ── Math ─────────────────────────────────────────────────────────────────────

  static double _dot(List<double> a, List<double> b) {
    double sum = 0;
    for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
  }

  static bool _isZeroVector(List<double> v) {
    double sumSq = 0;
    for (final x in v) sumSq += x * x;
    return sumSq < 1e-12;
  }
}
