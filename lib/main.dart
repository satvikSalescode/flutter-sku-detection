import 'package:flutter/material.dart';

import 'pages/detection_page.dart';
import 'pages/settings_page.dart';
import 'services/catalog_service.dart';
import 'services/dinov2_service.dart';
import 'services/inference_service.dart';
import 'services/settings_service.dart';
import 'services/vision_pipeline.dart';

void main() {
  runApp(const SkuDetectorApp());
}

class SkuDetectorApp extends StatelessWidget {
  const SkuDetectorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SKU Detector',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        colorScheme: ColorScheme.dark(
          primary:   const Color(0xFF00C9A7),
          secondary: const Color(0xFFFF6B6B),
          surface:   const Color(0xFF1A1A2E),
        ),
      ),
      home: const _InitScreen(),
    );
  }
}

// ── Splash / init ──────────────────────────────────────────────────────────────

class _InitScreen extends StatefulWidget {
  const _InitScreen();
  @override
  State<_InitScreen> createState() => _InitScreenState();
}

class _InitScreenState extends State<_InitScreen> {
  static const _teal = Color(0xFF00C9A7);

  // Services — created once and held for the lifetime of the app.
  final _yolo    = InferenceService();
  final _dino    = DinoV2Service();
  final _catalog = CatalogService();
  late  VisionPipeline _pipeline;

  String _status   = 'Starting…';
  bool   _hasError = false;

  @override
  void initState() {
    super.initState();
    _pipeline = VisionPipeline(
      yoloService:    _yolo,
      dinov2Service:  _dino,
      catalogService: _catalog,
    );
    WidgetsBinding.instance.addPostFrameCallback((_) => _init());
  }

  Future<void> _init() async {
    try {
      // Step 1 ─ YOLO
      _setStatus('Loading YOLO model…');
      await _yolo.initialize();

      // Step 2 ─ App settings
      _setStatus('Loading settings…');
      await SettingsService.init();

      // Step 3 ─ DINOv2 + catalog + self-test
      // VisionPipeline.initialize() handles all three sub-steps internally:
      //   dino.initialize() → catalog.load() → dino.selfTest()
      _setStatus('Loading DINOv2 + catalog…\n(first launch may take ~5 s)');
      await _pipeline.initialize();

      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (_) => HomePage(pipeline: _pipeline, yolo: _yolo),
        ),
      );
    } catch (e) {
      if (mounted) {
        setState(() {
          _status   = 'Startup failed:\n$e';
          _hasError = true;
        });
      }
    }
  }

  void _setStatus(String msg) {
    if (mounted) setState(() => _status = msg);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F0F1A),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Logo
              Container(
                width: 80, height: 80,
                decoration: BoxDecoration(
                  color:        _teal.withOpacity(0.13),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: _teal.withOpacity(0.35)),
                ),
                child: const Icon(Icons.document_scanner_rounded,
                    size: 40, color: _teal),
              ),
              const SizedBox(height: 24),

              // Title
              const Text('SKU Detector',
                  style: TextStyle(
                      color:      Colors.white,
                      fontSize:   24,
                      fontWeight: FontWeight.bold)),
              const SizedBox(height: 6),
              const Text('YOLO · DINOv2 · On-device',
                  style: TextStyle(color: Colors.white38, fontSize: 12)),

              const SizedBox(height: 40),

              // Spinner / error icon
              if (!_hasError)
                const SizedBox(
                  width: 36, height: 36,
                  child: CircularProgressIndicator(
                      color: _teal, strokeWidth: 3),
                )
              else
                Icon(Icons.error_outline,
                    color: Colors.red.shade400, size: 40),

              const SizedBox(height: 20),

              // Status text
              Text(
                _status,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color:    _hasError ? Colors.red.shade300 : Colors.white54,
                  fontSize: 13,
                  height:   1.5,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Home shell ─────────────────────────────────────────────────────────────────

class HomePage extends StatefulWidget {
  final VisionPipeline pipeline;
  final InferenceService yolo;   // kept for dispose()

  const HomePage({
    super.key,
    required this.pipeline,
    required this.yolo,
  });

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _tab = 0;

  @override
  void dispose() {
    widget.yolo.dispose();
    widget.pipeline.dinov2Service.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F0F1A),
      body: IndexedStack(
        index: _tab,
        children: [
          DetectionPage(pipeline: widget.pipeline),
          SettingsPage(pipeline: widget.pipeline),
        ],
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color:  const Color(0xFF1A1A2E),
          border: Border(
              top: BorderSide(color: Colors.white.withOpacity(0.07))),
        ),
        child: BottomNavigationBar(
          currentIndex:        _tab,
          onTap:               (i) => setState(() => _tab = i),
          backgroundColor:     Colors.transparent,
          elevation:           0,
          selectedItemColor:   const Color(0xFF00C9A7),
          unselectedItemColor: Colors.white38,
          selectedFontSize:    11,
          unselectedFontSize:  11,
          type: BottomNavigationBarType.fixed,
          items: const [
            BottomNavigationBarItem(
                icon: Icon(Icons.document_scanner_rounded), label: 'Detect'),
            BottomNavigationBarItem(
                icon: Icon(Icons.tune), label: 'Settings'),
          ],
        ),
      ),
    );
  }
}
