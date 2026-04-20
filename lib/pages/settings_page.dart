import 'package:flutter/material.dart';

import '../config/vision_config.dart';
import '../services/backend_api_client.dart';
import '../services/settings_service.dart';
import '../services/vision_pipeline.dart';
import 'catalog_page.dart';
import 'debug_screen.dart';

class SettingsPage extends StatefulWidget {
  final VisionPipeline pipeline;

  const SettingsPage({super.key, required this.pipeline});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  static const _bg    = Color(0xFF0F0F1A);
  static const _card  = Color(0xFF1A1A2E);
  static const _teal  = Color(0xFF00C9A7);
  static const _red   = Color(0xFFFF4C6A);
  static const _amber = Color(0xFFFFC947);

  late SettingsService _s;
  late TextEditingController _urlController;

  // ── Connection test state ──────────────────────────────────────────────────
  bool               _isTesting    = false;
  HealthCheckResult? _healthResult;

  // ── Catalog sync state ─────────────────────────────────────────────────────
  bool _isSyncing = false;

  @override
  void initState() {
    super.initState();
    _s = SettingsService.instance;
    _urlController = TextEditingController(text: _s.backendBaseUrl);
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  // ── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bg,
      appBar: AppBar(
        backgroundColor: _card,
        title: const Text('Settings',
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        actions: [
          IconButton(
            icon:     Icon(Icons.restore, color: _teal),
            tooltip:  'Reset defaults',
            onPressed: _resetDefaults,
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.only(bottom: 32),
        children: [
          // ── Catalog ────────────────────────────────────────────────────────
          _section('Catalog'),
          _navTile(
            icon:     Icons.inventory_2_outlined,
            label:    'Manage Catalog',
            subtitle: '${widget.pipeline.catalogService.products.length} '
                'product${widget.pipeline.catalogService.products.length == 1 ? "" : "s"}'
                ' · ${widget.pipeline.catalogService.numAngles} reference images',
            onTap:    _openCatalog,
          ),

          // ── Vision Processing ──────────────────────────────────────────────
          _section('Vision Processing'),
          _modeToggleTile(),
          if (_s.onDevice) ...[
            _catalogInfoTile(),
            _syncCatalogButton(),
          ] else ...[
            _urlInputTile(),
            _switchTile(
              label:    'Enable OCR',
              subtitle: 'Use text recognition for better accuracy on similar products',
              value:    _s.enableOcr,
              onChanged: (v) => setState(() => VisionConfig.enableOcr = v),
            ),
            _testConnectionTile(),
          ],
          _comparisonCard(),

          // ── Detection ──────────────────────────────────────────────────────
          _section('Detection'),
          _sliderTile(
            label:     'Confidence threshold',
            value:     _s.confThreshold,
            min:       0.05,
            max:       0.95,
            divisions: 18,
            format:    (v) => v.toStringAsFixed(2),
            onChanged: (v) => setState(() => _s.confThreshold = v),
          ),
          _sliderTile(
            label:     'Min crop size (px)',
            value:     _s.minCropSize.toDouble(),
            min:       16,
            max:       128,
            divisions: 14,
            format:    (v) => '${v.round()} px',
            onChanged: (v) => setState(() => _s.minCropSize = v.round()),
          ),
          _sliderTile(
            label:     'Max detections',
            value:     _s.maxDetections.toDouble(),
            min:       50,
            max:       500,
            divisions: 9,
            format:    (v) => v.round().toString(),
            onChanged: (v) => setState(() => _s.maxDetections = v.round()),
          ),

          // ── Developer ──────────────────────────────────────────────────────
          _section('Developer'),
          _switchTile(
            label:    'Show debug info',
            subtitle: 'Displays YOLO timing and detection details',
            value:    _s.showDebug,
            onChanged: (v) => setState(() => _s.showDebug = v),
          ),
          _navTile(
            icon:     Icons.bug_report_outlined,
            label:    'Debug / Verification',
            subtitle: 'Model info · self-test · pipeline test · catalog health',
            onTap:    _openDebugScreen,
          ),
        ],
      ),
    );
  }

  // ── Vision Processing tiles ────────────────────────────────────────────────

  Widget _modeToggleTile() {
    final onDevice = _s.onDevice;
    return Container(
      margin:  const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      decoration: BoxDecoration(
          color: _card, borderRadius: BorderRadius.circular(12)),
      child: SwitchListTile(
        contentPadding: EdgeInsets.zero,
        title: const Text('On-device processing',
            style: TextStyle(color: Colors.white70, fontSize: 13)),
        subtitle: Text(
          onDevice
              ? 'DINOv3-small, 384-dim, ~44 MB'
              : 'Cloud processing  (DINOv3-Large, 1024-dim)',
          style: TextStyle(
              color: onDevice ? _teal : _amber,
              fontSize: 11,
              fontWeight: FontWeight.w500),
        ),
        value:       onDevice,
        activeColor: _teal,
        onChanged:   _onModeToggle,
      ),
    );
  }

  Widget _catalogInfoTile() {
    final products   = widget.pipeline.catalogService.products.length;
    final embeddings = widget.pipeline.catalogService.numAngles;
    final lastSync   = _formatLastSync(_s.catalogLastSyncMs);

    return Container(
      margin:  const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
          color: _card, borderRadius: BorderRadius.circular(12)),
      child: Row(
        children: [
          Icon(Icons.storage_outlined, color: _teal, size: 18),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              'Products: $products   '
              'Embeddings: $embeddings   '
              'Last sync: $lastSync',
              style: const TextStyle(color: Colors.white60, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }

  Widget _syncCatalogButton() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      child: SizedBox(
        width: double.infinity,
        child: OutlinedButton.icon(
          style: OutlinedButton.styleFrom(
            foregroundColor: _teal,
            side: BorderSide(color: _teal.withOpacity(0.5)),
            padding: const EdgeInsets.symmetric(vertical: 12),
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10)),
          ),
          icon: _isSyncing
              ? SizedBox(
                  width: 14, height: 14,
                  child: CircularProgressIndicator(
                      strokeWidth: 2, color: _teal))
              : const Icon(Icons.sync, size: 16),
          label: Text(
              _isSyncing ? 'Syncing…' : 'Sync catalog',
              style: const TextStyle(fontSize: 13)),
          onPressed: _isSyncing ? null : _syncCatalog,
        ),
      ),
    );
  }

  Widget _urlInputTile() {
    return Container(
      margin:  const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      padding: const EdgeInsets.fromLTRB(16, 10, 16, 12),
      decoration: BoxDecoration(
          color: _card, borderRadius: BorderRadius.circular(12)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Backend URL',
              style: TextStyle(color: Colors.white70, fontSize: 13)),
          const SizedBox(height: 8),
          TextField(
            controller:   _urlController,
            style:        const TextStyle(color: Colors.white, fontSize: 13),
            keyboardType: TextInputType.url,
            autocorrect:  false,
            decoration: InputDecoration(
              isDense:  true,
              hintText: 'Android: http://10.0.2.2:8000  ·  iOS: http://localhost:8000',
              hintStyle: const TextStyle(color: Colors.white24, fontSize: 13),
              contentPadding: const EdgeInsets.symmetric(
                  horizontal: 12, vertical: 10),
              filled:    true,
              fillColor: _bg,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(8),
                borderSide: BorderSide(color: _teal.withOpacity(0.3)),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(8),
                borderSide: BorderSide(color: _teal.withOpacity(0.3)),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(8),
                borderSide: const BorderSide(color: _teal),
              ),
            ),
            onSubmitted: (v) {
              final url = v.trim();
              if (url.isNotEmpty) {
                VisionConfig.backendBaseUrl = url;
                _urlController.text = url;
                setState(() => _healthResult = null); // reset connection badge
              }
            },
            onTapOutside: (_) {
              final url = _urlController.text.trim();
              if (url.isNotEmpty) VisionConfig.backendBaseUrl = url;
              FocusScope.of(context).unfocus();
            },
          ),
        ],
      ),
    );
  }

  Widget _testConnectionTile() {
    final r = _healthResult;
    return Container(
      margin:  const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
      decoration: BoxDecoration(
          color: _card, borderRadius: BorderRadius.circular(12)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Status line (only shown after a test)
          if (r != null) ...[
            Row(
              children: [
                Icon(
                  r.connected ? Icons.check_circle : Icons.cancel,
                  color: r.connected ? _teal : _red,
                  size: 16,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    r.connected
                        ? 'Connected ✅'
                            '${r.model.isNotEmpty ? "  —  Model: ${r.model}" : ""}'
                            '${r.device.isNotEmpty ? ",  Device: ${r.device}" : ""}'
                        : 'Connection failed ❌  —  ${r.error}',
                    style: TextStyle(
                        color: r.connected ? _teal : _red,
                        fontSize: 12,
                        fontWeight: FontWeight.w500),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
          ],
          // Test button
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              style: OutlinedButton.styleFrom(
                foregroundColor: _teal,
                side: BorderSide(color: _teal.withOpacity(0.5)),
                padding: const EdgeInsets.symmetric(vertical: 11),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10)),
              ),
              icon: _isTesting
                  ? SizedBox(
                      width: 14, height: 14,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: _teal))
                  : const Icon(Icons.wifi_tethering, size: 16),
              label: Text(
                  _isTesting ? 'Testing…' : 'Test connection',
                  style: const TextStyle(fontSize: 13)),
              onPressed: _isTesting ? null : _testConnection,
            ),
          ),
        ],
      ),
    );
  }

  Widget _comparisonCard() {
    return Container(
      margin:  const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color:        _card,
        borderRadius: BorderRadius.circular(12),
        border:       Border.all(color: _teal.withOpacity(0.15)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header row
          Container(
            decoration: BoxDecoration(
              color: _teal.withOpacity(0.08),
              borderRadius: const BorderRadius.vertical(top: Radius.circular(12)),
            ),
            child: Row(
              children: [
                _compHeader('On-device', isActive: _s.onDevice),
                Container(width: 1, height: 36,
                    color: _teal.withOpacity(0.2)),
                _compHeader('Cloud', isActive: !_s.onDevice),
              ],
            ),
          ),
          // Comparison rows
          _compRow('Model',
              'DINOv3-small (21M)', 'DINOv3-Large (300M)'),
          _compDivider(),
          _compRow('Embeddings',
              '384-dim', '1024-dim'),
          _compDivider(),
          _compRow('Processing',
              '~1–2 s', '~2–3 s + network'),
          _compDivider(),
          _compRow('Connectivity',
              'Works offline', 'Needs internet'),
          _compDivider(),
          _compRow('Accuracy',
              'Good', 'Best'),
          _compDivider(),
          _compRow('OCR',
              'No', 'Available'),
        ],
      ),
    );
  }

  Widget _compHeader(String label, {required bool isActive}) {
    return Expanded(
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 10),
        child: Text(
          label,
          textAlign: TextAlign.center,
          style: TextStyle(
            color:      isActive ? _teal : Colors.white38,
            fontSize:   12,
            fontWeight: FontWeight.bold,
            letterSpacing: 0.8,
          ),
        ),
      ),
    );
  }

  Widget _compRow(String label, String left, String right) {
    const labelStyle = TextStyle(color: Colors.white38, fontSize: 11);
    const valStyle   = TextStyle(color: Colors.white70, fontSize: 12);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
      child: Row(
        children: [
          SizedBox(
            width: 90,
            child: Text(label, style: labelStyle),
          ),
          Expanded(child: Text(left,  style: valStyle)),
          Container(width: 1, height: 16, color: _teal.withOpacity(0.15)),
          const SizedBox(width: 14),
          Expanded(child: Text(right, style: valStyle)),
        ],
      ),
    );
  }

  Widget _compDivider() => Divider(
      height: 1, thickness: 1, color: _teal.withOpacity(0.08),
      indent: 14, endIndent: 14);

  // ── Actions ────────────────────────────────────────────────────────────────

  Future<void> _onModeToggle(bool newOnDevice) async {
    final confirmed = await _showModeConfirmDialog(newOnDevice);
    if (!confirmed) return;

    setState(() {
      VisionConfig.onDevice = newOnDevice;
      _healthResult         = null; // reset connection badge on mode change
    });
  }

  Future<bool> _showModeConfirmDialog(bool switchingToOnDevice) async {
    final toCloud    = !switchingToOnDevice;
    final title      = toCloud ? 'Switch to Cloud mode?' : 'Switch to On-device mode?';
    final body       = toCloud
        ? 'The DINOv3 model will be unloaded from memory.\n\n'
          'Product matching will be performed by the server. '
          'Requires an internet connection.'
        : 'DINOv3-small will be loaded into memory (~44 MB).\n\n'
          'Product matching will run entirely on this device. '
          'No internet required.';

    final result = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: _card,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Text(title,
            style: const TextStyle(color: Colors.white, fontSize: 16,
                fontWeight: FontWeight.bold)),
        content: Text(body,
            style: const TextStyle(color: Colors.white60, fontSize: 13,
                height: 1.5)),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel',
                style: TextStyle(color: Colors.white38)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text('Switch',
                style: TextStyle(
                    color: _teal, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
    return result ?? false;
  }

  Future<void> _testConnection() async {
    // Save whatever is in the URL field first
    final url = _urlController.text.trim();
    if (url.isNotEmpty) VisionConfig.backendBaseUrl = url;

    setState(() { _isTesting = true; _healthResult = null; });

    final client = BackendApiClient(baseUrl: VisionConfig.backendBaseUrl);
    final result = await client.checkHealthDetailed();

    if (mounted) setState(() { _isTesting = false; _healthResult = result; });
  }

  Future<void> _syncCatalog() async {
    setState(() => _isSyncing = true);
    try {
      await widget.pipeline.catalogService.load();
      _s.catalogLastSyncMs = DateTime.now().millisecondsSinceEpoch;
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(
              'Catalog synced — '
              '${widget.pipeline.catalogService.products.length} products'),
          backgroundColor: _teal,
          behavior: SnackBarBehavior.floating,
        ));
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Sync failed: $e'),
          backgroundColor: _red,
          behavior: SnackBarBehavior.floating,
        ));
      }
    } finally {
      if (mounted) setState(() => _isSyncing = false);
    }
  }

  // ── Navigation ────────────────────────────────────────────────────────────

  void _openDebugScreen() => Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => DebugScreen(pipeline: widget.pipeline)));

  void _openCatalog() => Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => CatalogPage(
            catalogService: widget.pipeline.catalogService,
            dinov3Service:  widget.pipeline.dinov3Service,
          )));

  // ── Shared widget helpers ─────────────────────────────────────────────────

  Widget _section(String title) => Padding(
        padding: const EdgeInsets.fromLTRB(16, 20, 16, 4),
        child: Text(
          title,
          style: const TextStyle(
              color:         _teal,
              fontWeight:    FontWeight.bold,
              fontSize:      12,
              letterSpacing: 1.2),
        ),
      );

  Widget _navTile({
    required IconData     icon,
    required String       label,
    required String       subtitle,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        margin:  const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        decoration: BoxDecoration(
          color:        _card,
          borderRadius: BorderRadius.circular(12),
          border:       Border.all(color: _teal.withOpacity(0.2)),
        ),
        child: Row(
          children: [
            Icon(icon, color: _teal, size: 20),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(label,
                      style: const TextStyle(
                          color:      Colors.white,
                          fontSize:   13,
                          fontWeight: FontWeight.w500)),
                  const SizedBox(height: 2),
                  Text(subtitle,
                      style: const TextStyle(
                          color:    Colors.white38,
                          fontSize: 11)),
                ],
              ),
            ),
            const Icon(Icons.chevron_right,
                color: Colors.white24, size: 20),
          ],
        ),
      ),
    );
  }

  Widget _sliderTile({
    required String  label,
    required double  value,
    required double  min,
    required double  max,
    required int     divisions,
    required String Function(double) format,
    required ValueChanged<double>    onChanged,
  }) {
    return Container(
      margin:  const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
          color: _card, borderRadius: BorderRadius.circular(12)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(label,
                  style: const TextStyle(
                      color: Colors.white70, fontSize: 13)),
              Text(format(value),
                  style: const TextStyle(
                      color:      _teal,
                      fontWeight: FontWeight.bold,
                      fontSize:   13)),
            ],
          ),
          Slider(
            value:         value,
            min:           min,
            max:           max,
            divisions:     divisions,
            activeColor:   _teal,
            inactiveColor: _teal.withOpacity(0.2),
            onChanged:     onChanged,
          ),
        ],
      ),
    );
  }

  Widget _switchTile({
    required String            label,
    required String            subtitle,
    required bool              value,
    required ValueChanged<bool> onChanged,
  }) {
    return Container(
      margin:  const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      decoration: BoxDecoration(
          color: _card, borderRadius: BorderRadius.circular(12)),
      child: SwitchListTile(
        contentPadding: EdgeInsets.zero,
        title: Text(label,
            style: const TextStyle(color: Colors.white70, fontSize: 13)),
        subtitle: Text(subtitle,
            style: const TextStyle(color: Colors.white38, fontSize: 11)),
        value:       value,
        activeColor: _teal,
        onChanged:   onChanged,
      ),
    );
  }

  // ── Misc helpers ──────────────────────────────────────────────────────────

  String _formatLastSync(int epochMs) {
    if (epochMs == 0) return 'Never';
    final diff = DateTime.now()
        .difference(DateTime.fromMillisecondsSinceEpoch(epochMs));
    if (diff.inSeconds < 60)  return 'Just now';
    if (diff.inMinutes < 60)  return '${diff.inMinutes}m ago';
    if (diff.inHours   < 24)  return '${diff.inHours}h ago';
    return '${diff.inDays}d ago';
  }

  void _resetDefaults() async {
    await _s.resetToDefaults();
    _urlController.text = _s.backendBaseUrl;
    setState(() { _healthResult = null; });
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content:         const Text('Settings reset to defaults'),
        backgroundColor: _teal,
        behavior:        SnackBarBehavior.floating,
      ));
    }
  }
}
