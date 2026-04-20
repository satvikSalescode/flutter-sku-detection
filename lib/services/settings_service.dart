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

  // ── Pipeline mode ──────────────────────────────────────────────────────────

  /// True = everything runs on-device (default).
  /// False = YOLO on-device, matching delegated to backend API.
  bool get onDevice => _prefs.getBool('on_device') ?? false;
  set onDevice(bool v) => _prefs.setBool('on_device', v);

  /// Default is http://localhost:8000 which works for:
  ///   - Physical Android device after: adb reverse tcp:8000 tcp:8000
  ///   - iOS Simulator (shares host network natively)
  /// For Android emulator without adb reverse, use: http://10.0.2.2:8000
  /// For production: https://api.salescode.ai
  String get backendBaseUrl =>
      _prefs.getString('backend_base_url') ?? 'http://localhost:8000';
  set backendBaseUrl(String v) => _prefs.setString('backend_base_url', v);

  bool get enableOcr => _prefs.getBool('enable_ocr') ?? false;
  set enableOcr(bool v) => _prefs.setBool('enable_ocr', v);

  // ── Catalog sync ───────────────────────────────────────────────────────────

  /// Epoch milliseconds of the last successful catalog load/sync.
  /// 0 = never synced.
  int get catalogLastSyncMs => _prefs.getInt('catalog_last_sync_ms') ?? 0;
  set catalogLastSyncMs(int v) => _prefs.setInt('catalog_last_sync_ms', v);
}
