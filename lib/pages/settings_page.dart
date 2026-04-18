import 'package:flutter/material.dart';

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
  static const _bg   = Color(0xFF0F0F1A);
  static const _card = Color(0xFF1A1A2E);
  static const _teal = Color(0xFF00C9A7);

  late SettingsService _s;

  @override
  void initState() {
    super.initState();
    _s = SettingsService.instance;
  }

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
          // ── Catalog section ────────────────────────────────────────────────
          _section('Catalog'),
          _navTile(
            icon:     Icons.inventory_2_outlined,
            label:    'Manage Catalog',
            subtitle: '${widget.pipeline.catalogService.products.length} '
                'product${widget.pipeline.catalogService.products.length == 1 ? "" : "s"}'
                ' · ${widget.pipeline.catalogService.numAngles} reference images',
            onTap:    _openCatalog,
          ),

          // ── Detection section ──────────────────────────────────────────────
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

          // ── Developer section ──────────────────────────────────────────────
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

  // ── Navigation ────────────────────────────────────────────────────────────

  void _openDebugScreen() {
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => DebugScreen(pipeline: widget.pipeline),
    ));
  }

  void _openCatalog() {
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => CatalogPage(
        catalogService: widget.pipeline.catalogService,
        dinov3Service:  widget.pipeline.dinov3Service,
      ),
    ));
  }

  // ── Widgets ───────────────────────────────────────────────────────────────

  Widget _section(String title) => Padding(
        padding: const EdgeInsets.fromLTRB(16, 20, 16, 4),
        child: Text(
          title,
          style: const TextStyle(
              color:          _teal,
              fontWeight:     FontWeight.bold,
              fontSize:       12,
              letterSpacing:  1.2),
        ),
      );

  Widget _navTile({
    required IconData    icon,
    required String      label,
    required String      subtitle,
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
                          color: Colors.white, fontSize: 13,
                          fontWeight: FontWeight.w500)),
                  const SizedBox(height: 2),
                  Text(subtitle,
                      style: const TextStyle(
                          color: Colors.white38, fontSize: 11)),
                ],
              ),
            ),
            const Icon(Icons.chevron_right, color: Colors.white24, size: 20),
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
    required ValueChanged<double> onChanged,
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
                      color: _teal,
                      fontWeight: FontWeight.bold,
                      fontSize: 13)),
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
    required String label,
    required String subtitle,
    required bool   value,
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

  void _resetDefaults() async {
    await _s.resetToDefaults();
    setState(() {});
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content:         const Text('Settings reset to defaults'),
        backgroundColor: _teal,
        behavior:        SnackBarBehavior.floating,
      ));
    }
  }
}
