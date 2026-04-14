import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

import '../services/catalog_service.dart';
import '../services/dinov2_service.dart';

// ── Palette ───────────────────────────────────────────────────────────────────

const _kTeal   = Color(0xFF00C9A7);
const _kRed    = Color(0xFFFF5252);
const _kAmber  = Color(0xFFFFB74D);
const _kCard   = Color(0xFF1A1A2E);
const _kBg     = Color(0xFF0F0F1A);
const _kGrey   = Color(0xFF252540);

// ═══════════════════════════════════════════════════════════════════════════════
// CATALOG LIST PAGE
// ═══════════════════════════════════════════════════════════════════════════════

/// Shows catalog metadata, product list, and provides URL sync.
class CatalogPage extends StatefulWidget {
  final CatalogService  catalogService;
  final DinoV2Service   dinov2Service;

  const CatalogPage({
    super.key,
    required this.catalogService,
    required this.dinov2Service,
  });

  @override
  State<CatalogPage> createState() => _CatalogPageState();
}

class _CatalogPageState extends State<CatalogPage> {
  int    _fileSizeKb  = 0;
  bool   _isSyncing   = false;

  CatalogService get _cs => widget.catalogService;

  @override
  void initState() {
    super.initState();
    _loadFileSize();
  }

  Future<void> _loadFileSize() async {
    final bytes = await CatalogService.localFileSizeBytes();
    if (mounted) setState(() => _fileSizeKb = (bytes / 1024).ceil());
  }

  // ── URL sync ────────────────────────────────────────────────────────────────

  Future<void> _showSyncDialog() async {
    final ctrl = TextEditingController();
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: _kCard,
        title: const Text('Sync Catalog from URL',
            style: TextStyle(color: Colors.white, fontSize: 16)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Paste the URL of a catalog_embeddings.json generated '
              'by the desktop portal.',
              style: TextStyle(color: Colors.white54, fontSize: 12, height: 1.4),
            ),
            const SizedBox(height: 16),
            TextField(
              controller:   ctrl,
              autofocus:    true,
              keyboardType: TextInputType.url,
              style:        const TextStyle(color: Colors.white, fontSize: 13),
              decoration: InputDecoration(
                hintText:       'https://example.com/catalog_embeddings.json',
                hintStyle:      const TextStyle(color: Colors.white30, fontSize: 12),
                filled:         true,
                fillColor:      _kGrey,
                border:         OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                    borderSide: BorderSide.none),
                contentPadding: const EdgeInsets.symmetric(
                    horizontal: 12, vertical: 10),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel',
                style: TextStyle(color: Colors.white38)),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: _kTeal),
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Sync',
                style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );

    if (confirmed != true || !mounted) return;
    final url = ctrl.text.trim();
    if (url.isEmpty) return;

    setState(() => _isSyncing = true);
    try {
      await _cs.loadFromUrl(url);
      await _loadFileSize();
      if (mounted) {
        setState(() {});
        _snack('✅  Catalog synced: ${_cs.products.length} products', _kTeal);
      }
    } catch (e) {
      if (mounted) _snack('❌  Sync failed: $e', _kRed);
    } finally {
      if (mounted) setState(() => _isSyncing = false);
    }
  }

  // ── Build ────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final products = _cs.products;

    return Scaffold(
      backgroundColor: _kBg,
      appBar: AppBar(
        backgroundColor: _kCard,
        leading: IconButton(
          icon:      const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: const Text('Product Catalog',
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        actions: [
          // Sync button
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: _isSyncing
                ? const Padding(
                    padding: EdgeInsets.all(14),
                    child: SizedBox(
                        width: 20, height: 20,
                        child: CircularProgressIndicator(
                            color: _kTeal, strokeWidth: 2)),
                  )
                : IconButton(
                    icon:     const Icon(Icons.sync, color: _kTeal),
                    tooltip:  'Sync catalog from URL',
                    onPressed: _showSyncDialog,
                  ),
          ),
        ],
      ),
      body: Column(
        children: [
          // ── Stats card ─────────────────────────────────────────────────────
          _StatsCard(
            productCount:   products.length,
            totalImages:    _cs.numAngles,
            lastUpdated:    _cs.catalogVersion,
            fileSizeKb:     _fileSizeKb,
            loadedFromLocal: _cs.loadedFromLocal,
          ),

          // ── Product list ────────────────────────────────────────────────────
          Expanded(
            child: products.isEmpty
                ? _emptyState()
                : ListView.builder(
                    padding: const EdgeInsets.fromLTRB(16, 8, 16, 32),
                    itemCount: products.length,
                    itemBuilder: (_, i) => Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: _ProductListTile(
                        product:  products[i],
                        onTap:    () => _openDetail(products[i]),
                      ),
                    ),
                  ),
          ),
        ],
      ),
    );
  }

  Widget _emptyState() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.inventory_2_outlined,
              size: 56, color: Colors.white.withOpacity(0.12)),
          const SizedBox(height: 16),
          const Text('No products in catalog',
              style: TextStyle(color: Colors.white38, fontSize: 14)),
          const SizedBox(height: 6),
          const Text('Run generate_catalog.py or sync from a URL',
              style: TextStyle(color: Colors.white24, fontSize: 12)),
        ],
      ),
    );
  }

  Future<void> _openDetail(ProductReference product) async {
    await Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => ProductDetailPage(
        product:        product,
        catalogService: widget.catalogService,
        dinov2Service:  widget.dinov2Service,
      ),
    ));
    // Refresh list after returning (user may have added embeddings)
    if (mounted) setState(() {});
  }

  void _snack(String msg, Color color) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content:         Text(msg),
      backgroundColor: color,
      behavior:        SnackBarBehavior.floating,
    ));
  }
}

// ── Stats card ────────────────────────────────────────────────────────────────

class _StatsCard extends StatelessWidget {
  final int    productCount;
  final int    totalImages;
  final String lastUpdated;
  final int    fileSizeKb;
  final bool   loadedFromLocal;

  const _StatsCard({
    required this.productCount,
    required this.totalImages,
    required this.lastUpdated,
    required this.fileSizeKb,
    required this.loadedFromLocal,
  });

  String _formatVersion(String v) {
    if (v.isEmpty) return 'Unknown';
    try {
      final dt = DateTime.parse(v).toLocal();
      return '${dt.year}-${dt.month.toString().padLeft(2, '0')}-'
          '${dt.day.toString().padLeft(2, '0')}  '
          '${dt.hour.toString().padLeft(2, '0')}:'
          '${dt.minute.toString().padLeft(2, '0')}';
    } catch (_) {
      return v;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin:  const EdgeInsets.all(16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color:        _kCard,
        borderRadius: BorderRadius.circular(14),
        border:       Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      child: Column(
        children: [
          Row(
            children: [
              _StatBadge(
                label: '$productCount',
                sub:   'products',
                color: _kTeal,
              ),
              const SizedBox(width: 12),
              _StatBadge(
                label: '$totalImages',
                sub:   'ref images',
                color: Colors.white70,
              ),
              const SizedBox(width: 12),
              _StatBadge(
                label: fileSizeKb > 0 ? '${fileSizeKb}KB' : 'asset',
                sub:   loadedFromLocal ? 'local' : 'bundled',
                color: loadedFromLocal ? _kTeal : _kAmber,
              ),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              const Icon(Icons.schedule, size: 13, color: Colors.white38),
              const SizedBox(width: 5),
              Expanded(
                child: Text(
                  'Last updated: ${_formatVersion(lastUpdated)}',
                  style: const TextStyle(color: Colors.white38, fontSize: 11),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _StatBadge extends StatelessWidget {
  final String label;
  final String sub;
  final Color  color;

  const _StatBadge({
    required this.label,
    required this.sub,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color:        color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(10),
        border:       Border.all(color: color.withOpacity(0.25)),
      ),
      child: Column(
        children: [
          Text(label,
              style: TextStyle(
                  color:      color,
                  fontSize:   18,
                  fontWeight: FontWeight.bold)),
          Text(sub,
              style: const TextStyle(
                  color: Colors.white38, fontSize: 10)),
        ],
      ),
    );
  }
}

// ── Product list tile ─────────────────────────────────────────────────────────

class _ProductListTile extends StatelessWidget {
  final ProductReference product;
  final VoidCallback     onTap;

  const _ProductListTile({required this.product, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final filled = product.standardAnglesFilled;
    final total  = 4; // front/back/left/right
    final color  = filled == total ? _kTeal
        : filled > 0             ? _kAmber
        :                          _kRed;

    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color:        _kCard,
          borderRadius: BorderRadius.circular(12),
          border:       Border.all(color: color.withOpacity(0.25)),
        ),
        child: Row(
          children: [
            // Status dot
            Container(
              width: 10, height: 10,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle),
            ),
            const SizedBox(width: 12),

            // Product name
            Expanded(
              child: Text(
                product.name.replaceAll('_', ' '),
                style: const TextStyle(
                    color:      Colors.white,
                    fontSize:   14,
                    fontWeight: FontWeight.w500),
              ),
            ),

            // Image count chip
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color:        color.withOpacity(0.15),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                '$filled / $total images',
                style: TextStyle(
                    color:      color,
                    fontSize:   11,
                    fontWeight: FontWeight.bold),
              ),
            ),
            const SizedBox(width: 8),
            const Icon(Icons.chevron_right, color: Colors.white24, size: 20),
          ],
        ),
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRODUCT DETAIL PAGE
// ═══════════════════════════════════════════════════════════════════════════════

/// Shows the 4 angle slots for one product and lets the user add/replace
/// embeddings by capturing a photo.
class ProductDetailPage extends StatefulWidget {
  final ProductReference product;
  final CatalogService   catalogService;
  final DinoV2Service    dinov2Service;

  const ProductDetailPage({
    super.key,
    required this.product,
    required this.catalogService,
    required this.dinov2Service,
  });

  @override
  State<ProductDetailPage> createState() => _ProductDetailPageState();
}

class _ProductDetailPageState extends State<ProductDetailPage> {
  // Track which angle is currently being processed (show spinner in that slot).
  String? _processingAngle;

  /// Get the live product reference from the catalog (reflects updates).
  ProductReference get _product {
    return widget.catalogService.products.firstWhere(
      (p) => p.name == widget.product.name,
      orElse: () => widget.product,
    );
  }

  // ── Image pick → embed → save ─────────────────────────────────────────────

  Future<void> _pickAndEmbed(String angleKey, String angleLabel) async {
    // Show picker choice
    final source = await _showSourceDialog(angleLabel);
    if (source == null || !mounted) return;

    setState(() => _processingAngle = angleKey);

    try {
      // 1. Pick image
      final xf = await ImagePicker().pickImage(
        source:       source,
        imageQuality: 95,
      );
      if (xf == null || !mounted) { setState(() => _processingAngle = null); return; }

      // 2. Decode
      final bytes   = await File(xf.path).readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) throw Exception('Cannot decode image');

      // 3. Generate DINOv2 embedding
      _showStatus('Generating embedding…');
      final embedding = await widget.dinov2Service.getEmbedding(decoded);

      // 4. Save to catalog
      _showStatus('Saving…');
      await widget.catalogService.updateProductEmbedding(
        productName: widget.product.name,
        angleKey:    angleKey,
        embedding:   embedding,
      );

      if (mounted) {
        setState(() => _processingAngle = null);
        _snack(
          '✅  Embedding generated for '
          '${widget.product.name.replaceAll("_", " ")} / $angleLabel',
          _kTeal,
        );
      }
    } catch (e) {
      if (mounted) {
        setState(() => _processingAngle = null);
        _snack('❌  Failed: $e', _kRed);
      }
    }
  }

  Future<ImageSource?> _showSourceDialog(String angle) {
    return showModalBottomSheet<ImageSource>(
      context:       context,
      backgroundColor: _kCard,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 8),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Padding(
                padding: const EdgeInsets.all(12),
                child: Text(
                  'Add reference image — $angle',
                  style: const TextStyle(
                      color: Colors.white70, fontWeight: FontWeight.w600),
                ),
              ),
              ListTile(
                leading: const Icon(Icons.camera_alt, color: _kTeal),
                title:   const Text('Camera',
                    style: TextStyle(color: Colors.white)),
                onTap: () => Navigator.of(ctx).pop(ImageSource.camera),
              ),
              ListTile(
                leading: const Icon(Icons.photo_library_outlined,
                    color: _kTeal),
                title:   const Text('Gallery',
                    style: TextStyle(color: Colors.white)),
                onTap: () => Navigator.of(ctx).pop(ImageSource.gallery),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final p       = _product;
    final filled  = p.standardAnglesFilled;

    return Scaffold(
      backgroundColor: _kBg,
      appBar: AppBar(
        backgroundColor: _kCard,
        leading: IconButton(
          icon:      const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              p.name.replaceAll('_', ' '),
              style: const TextStyle(
                  color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16),
            ),
            Text(
              '$filled / 4 reference images',
              style: TextStyle(
                  color: _slotCountColor(filled), fontSize: 11),
            ),
          ],
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // ── Instruction banner ─────────────────────────────────────────────
          Container(
            padding: const EdgeInsets.all(12),
            margin:  const EdgeInsets.only(bottom: 20),
            decoration: BoxDecoration(
              color:        _kTeal.withOpacity(0.08),
              borderRadius: BorderRadius.circular(10),
              border:       Border.all(color: _kTeal.withOpacity(0.2)),
            ),
            child: const Row(
              children: [
                Icon(Icons.info_outline, size: 16, color: _kTeal),
                SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Tap an empty slot to capture a reference photo. '
                    'A DINOv2 embedding is generated on-device and saved locally.',
                    style: TextStyle(
                        color: _kTeal, fontSize: 11, height: 1.4),
                  ),
                ),
              ],
            ),
          ),

          // ── 4 Angle slots ──────────────────────────────────────────────────
          GridView.count(
            crossAxisCount:   2,
            crossAxisSpacing: 12,
            mainAxisSpacing:  12,
            childAspectRatio: 1.05,
            shrinkWrap:       true,
            physics:          const NeverScrollableScrollPhysics(),
            children: List.generate(4, (i) {
              final key   = kStandardAngles[i];
              final label = kStandardAngleLabels[i];
              final isFilled     = p.hasAngle(key);
              final isProcessing = _processingAngle == key;
              return _AngleSlot(
                label:        label,
                angleKey:     key,
                isFilled:     isFilled,
                isProcessing: isProcessing,
                onTap: isProcessing
                    ? null
                    : () => _pickAndEmbed(key, label),
              );
            }),
          ),

          // ── Extra angles (non-standard) ────────────────────────────────────
          if (p.angles.any((a) => !kStandardAngles.contains(a.toLowerCase()))) ...[
            const SizedBox(height: 24),
            const Text('Additional angles',
                style: TextStyle(
                    color:      _kTeal,
                    fontSize:   11,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1.1)),
            const SizedBox(height: 10),
            ...p.angles
                .asMap()
                .entries
                .where((e) =>
                    !kStandardAngles.contains(e.value.toLowerCase()))
                .map((e) => _ExtraAngleTile(
                      angleKey: e.value,
                      index:    e.key,
                    )),
          ],
        ],
      ),
    );
  }

  Color _slotCountColor(int filled) {
    if (filled == 4) return _kTeal;
    if (filled > 0)  return _kAmber;
    return _kRed;
  }

  void _showStatus(String msg) {
    // Non-intrusive status update — just logs; slot spinner already visible.
    debugPrint('[ProductDetailPage] $msg');
  }

  void _snack(String msg, Color color) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content:         Text(msg, style: const TextStyle(fontSize: 13)),
      backgroundColor: color,
      behavior:        SnackBarBehavior.floating,
      duration:        const Duration(seconds: 3),
    ));
  }
}

// ── Angle slot card ───────────────────────────────────────────────────────────

class _AngleSlot extends StatelessWidget {
  final String       label;
  final String       angleKey;
  final bool         isFilled;
  final bool         isProcessing;
  final VoidCallback? onTap;

  const _AngleSlot({
    required this.label,
    required this.angleKey,
    required this.isFilled,
    required this.isProcessing,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final color = isFilled ? _kTeal : Colors.white24;

    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        decoration: BoxDecoration(
          color:        isFilled
              ? _kTeal.withOpacity(0.08)
              : _kCard,
          borderRadius: BorderRadius.circular(14),
          border:       Border.all(
            color: isFilled
                ? _kTeal.withOpacity(0.45)
                : Colors.white.withOpacity(0.08),
            width: isFilled ? 1.5 : 1.0,
          ),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Icon / spinner
            if (isProcessing)
              const SizedBox(
                width: 32, height: 32,
                child: CircularProgressIndicator(
                    color: _kTeal, strokeWidth: 2.5),
              )
            else if (isFilled)
              Container(
                width: 48, height: 48,
                decoration: BoxDecoration(
                  color:  _kTeal.withOpacity(0.15),
                  shape:  BoxShape.circle,
                ),
                child: const Icon(Icons.check, color: _kTeal, size: 26),
              )
            else
              Container(
                width: 48, height: 48,
                decoration: BoxDecoration(
                  color:  Colors.white.withOpacity(0.05),
                  shape:  BoxShape.circle,
                ),
                child: const Icon(Icons.add_a_photo_outlined,
                    color: Colors.white38, size: 22),
              ),

            const SizedBox(height: 10),

            // Angle label
            Text(
              label,
              style: TextStyle(
                  color:      isFilled ? _kTeal : Colors.white54,
                  fontSize:   13,
                  fontWeight: FontWeight.w600),
            ),

            const SizedBox(height: 2),

            // Status text
            Text(
              isProcessing
                  ? 'Processing…'
                  : isFilled
                      ? 'Tap to replace'
                      : 'Tap to add',
              style: TextStyle(
                  color:    isFilled
                      ? _kTeal.withOpacity(0.7)
                      : Colors.white24,
                  fontSize: 10),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Extra (non-standard) angle tile ──────────────────────────────────────────

class _ExtraAngleTile extends StatelessWidget {
  final String angleKey;
  final int    index;

  const _ExtraAngleTile({required this.angleKey, required this.index});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin:  const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color:        _kCard,
        borderRadius: BorderRadius.circular(10),
        border:       Border.all(color: _kTeal.withOpacity(0.2)),
      ),
      child: Row(
        children: [
          const Icon(Icons.check_circle, color: _kTeal, size: 16),
          const SizedBox(width: 10),
          Text(
            angleKey,
            style: const TextStyle(color: Colors.white70, fontSize: 13),
          ),
          const Spacer(),
          Text(
            'embedding #${index + 1}',
            style: const TextStyle(color: Colors.white24, fontSize: 11),
          ),
        ],
      ),
    );
  }
}
