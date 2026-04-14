# Android Configuration

Flutter created `android/app/build.gradle.kts` (Kotlin DSL — note the `.kts` extension).
Open that file and make the two edits below.

---

## 1. android/app/build.gradle.kts

**Add `aaptOptions` inside the `android {` block** (prevents the .onnx file being compressed):

Find this line:
```kotlin
android {
```

And add `aaptOptions` right after it, so it looks like:
```kotlin
android {
    aaptOptions {
        noCompress += listOf("onnx")
    }
    ...
```

**Change `minSdk`** — find the `defaultConfig {` block and change minSdk to 24:
```kotlin
    defaultConfig {
        ...
        minSdk = 24       // ← change from whatever it was (usually 21) to 24
        ...
    }
```

---

## 2. android/app/src/main/AndroidManifest.xml

Add camera and storage permissions inside `<manifest>` **before** the `<application` tag:

```xml
<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"
    android:maxSdkVersion="32"/>
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES"/>
<uses-feature android:name="android.hardware.camera" android:required="false"/>
```

---

## What the final build.gradle.kts looks like (abbreviated)

```kotlin
android {
    aaptOptions {
        noCompress += listOf("onnx")
    }
    namespace = "com.example.sku_detector"
    compileSdk = ...

    defaultConfig {
        applicationId = "com.example.sku_detector"
        minSdk = 24
        targetSdk = ...
        versionCode = ...
        versionName = ...
    }
    ...
}
```

---

## Verify hardware acceleration is working

After `flutter run`, look in the Terminal for:
```
[InferenceService] 🚀  NNAPI hardware acceleration enabled
[InferenceService] ⚡  Inference took <N> ms
```
Inference time under ~200 ms = hardware accelerated. Over 1000 ms = CPU fallback.
