import 'package:shared_preferences/shared_preferences.dart';

/// App-wide configuration, persisted in SharedPreferences.
/// Only YOLO detection settings remain — matching/OCR/catalog settings
/// were removed along with the MobileCLIP pipeline.
class SettingsService {
  static SettingsService? _instance;
  static SettingsService get instance => _instance!;

  late final SharedPreferences _prefs;

  SettingsService._();

  static Future<SettingsService> init() async {
    _instance ??= SettingsService._();
    _instance!._prefs = await SharedPreferences.getInstance();
    return _instance!;
  }

  Future<void> load() async => await SettingsService.init();

  Future<void> resetToDefaults() async => await _prefs.clear();

  // ── Detection ──────────────────────────────────────────────────────────────

  double get confThreshold => _prefs.getDouble('conf_threshold') ?? 0.30;
  set confThreshold(double v) => _prefs.setDouble('conf_threshold', v);

  int get minCropSize => _prefs.getInt('min_crop_size') ?? 32;
  set minCropSize(int v) => _prefs.setInt('min_crop_size', v);

  int get maxDetections => _prefs.getInt('max_detections') ?? 500;
  set maxDetections(int v) => _prefs.setInt('max_detections', v);

  // ── Developer ──────────────────────────────────────────────────────────────

  bool get showDebug => _prefs.getBool('show_debug') ?? false;
  set showDebug(bool v) => _prefs.setBool('show_debug', v);
}
