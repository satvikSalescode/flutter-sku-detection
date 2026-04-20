import '../services/settings_service.dart';

/// Central configuration for the vision pipeline mode.
///
/// All persistent settings are stored in [SettingsService] (SharedPreferences).
/// In-memory overrides ([scoreThreshold], [marginThreshold]) are not persisted
/// and default to null — the backend uses its own defaults when null.
///
/// Usage:
///   if (VisionConfig.onDevice) {
///     // existing on-device pipeline
///   } else {
///     // BackendApiClient path
///   }
class VisionConfig {
  VisionConfig._(); // not instantiable

  // ── Pipeline mode ──────────────────────────────────────────────────────────

  /// If true, run everything on-device (existing YOLO + DINOv3 + catalog flow).
  /// If false, YOLO runs on-device; matching is delegated to the backend API.
  static bool get onDevice => SettingsService.instance.onDevice;
  static set onDevice(bool v) => SettingsService.instance.onDevice = v;

  // ── Backend ────────────────────────────────────────────────────────────────

  /// Root URL of the backend API.
  /// Default: http://10.0.2.2:8000 (Android emulator → host localhost).
  /// iOS Simulator: http://localhost:8000
  /// Production:    https://api.salescode.ai
  /// Only used when [onDevice] is false.
  static String get backendBaseUrl => SettingsService.instance.backendBaseUrl;
  static set backendBaseUrl(String v) =>
      SettingsService.instance.backendBaseUrl = v;

  /// Whether to request OCR annotations from the backend.
  /// Only used when [onDevice] is false.
  static bool get enableOcr => SettingsService.instance.enableOcr;
  static set enableOcr(bool v) => SettingsService.instance.enableOcr = v;

  // ── Backend threshold overrides ────────────────────────────────────────────
  // These are runtime-only (not persisted). When null, the backend uses its
  // own configured defaults.

  /// Optional score threshold override sent to the backend (0.0–1.0).
  /// null = use backend default.
  static double? scoreThreshold;

  /// Optional margin threshold override sent to the backend (0.0–1.0).
  /// null = use backend default.
  static double? marginThreshold;
}
