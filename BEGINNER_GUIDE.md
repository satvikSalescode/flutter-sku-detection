# Complete Beginner Guide – From Zero to Shareable APK

This guide walks you through every single step, assuming you have never used Flutter before.
By the end you will have a `.apk` file you can send to anyone and install on any Android phone.

---

## PART 1 – Install the Tools (one-time setup)

Everything is installed through **Homebrew** — the standard Mac package manager.
Open **Terminal** (press ⌘ Space, type "Terminal", hit Enter) and run the commands below.

---

### 1A. Install Homebrew (if you don't have it)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After it finishes, follow any instructions it prints about adding brew to your PATH
(usually one extra command starting with `echo 'eval ...'`). Then verify:

```bash
brew --version
```

---

### 1B. Install Java (JDK 17)

Android build tools require Java. Install it with one command:

```bash
brew install --cask temurin@17
```

Verify:
```bash
java -version
# Should print: openjdk version "17.x.x" ...
```

---

### 1C. Install Android Studio

Flutter needs Android Studio's SDK and build tools — even if you never open it again after setup.

```bash
brew install --cask android-studio
```

After it installs, **open Android Studio once** (find it in Launchpad or Applications):
- Click through the setup wizard
- Choose **"Standard"** installation when asked
- Let it download the Android SDK, Build Tools, and Emulator
- Click **Finish**

You only need to do this once. After the wizard completes you can close Android Studio.

---

### 1D. Install Flutter SDK

```bash
brew install --cask flutter
```

Verify:
```bash
flutter --version
# Should print: Flutter 3.x.x ...
```

> **Note:** If Terminal says `flutter: command not found` after installing,
> close the Terminal window and open a new one, then try again.

---

### 1E. Run Flutter Doctor

This is the magic command that checks your entire setup:

```bash
flutter doctor
```

You will see a checklist. The most common fix needed is accepting Android licenses:

```bash
flutter doctor --android-licenses
# Press 'y' and Enter for every prompt that appears
```

Run `flutter doctor` again after that. When it shows ✓ next to Flutter and Android toolchain
(Xcode warnings are fine — you're only targeting Android), you are ready to proceed.

---

## PART 2 – Set Up the Project

### 2A. Create the Flutter project

Open Terminal and run:

```bash
# Go to the folder that has all the files I already created for you
cd "/Users/satvikchaudhary/Documents/Claude/flutter-sku-detection"

# This creates the Flutter project skeleton IN this folder
# It will NOT overwrite the lib/ or assets/ files already here
flutter create . --project-name sku_detector
```

You will now have all the standard Flutter files (android/, ios/, pubspec.yaml, etc.)
mixed with the source code files that are already there.

---

### 2B. Convert your model first

Before building the app, you need the TFLite model file in the assets/ folder.

```bash
# Make sure ultralytics is installed
pip install ultralytics

# Run the converter
python convert_model.py
```

After it runs, you will see a line like:
```
✅  Export complete → /Users/satvikchaudhary/Desktop/IRED DOCS/All_Epochs/best_saved_model/
```

Copy the .tflite file from that folder into assets/:
```bash
# The exact filename may differ — check what was exported
cp "/Users/satvikchaudhary/Desktop/IRED DOCS/All_Epochs/best_saved_model/best_float32.tflite" assets/
```

Also check assets/labels.txt was generated with your class names.

---

### 2C. Update the class count

Open `lib/services/inference_service.dart` in any text editor.
Find this line near the top:

```dart
const int kNumClasses = 1;  // ← change this!
```

Change `1` to whatever number `convert_model.py` printed as "Number of classes".
Save the file.

---

### 2D. Apply Android configuration

Open `android/app/build.gradle` in a text editor and make these edits:

**Find `defaultConfig {` and change minSdk:**
```gradle
defaultConfig {
    minSdkVersion 24      // change from whatever it was to 24
    ...
}
```

**Add aaptOptions just below the `android {` opening line:**
```gradle
android {
    aaptOptions {
        noCompress "tflite"
    }
    ...
}
```

**At the bottom, inside `dependencies {`, add:**
```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu-api:2.14.0'
    ...
}
```

Open `android/app/src/main/AndroidManifest.xml` and add these lines
just before the `<application` tag:

```xml
<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"
    android:maxSdkVersion="32"/>
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES"/>
<uses-feature android:name="android.hardware.camera" android:required="false"/>
```

---

### 2E. Install Flutter packages

```bash
flutter pub get
```

This downloads all the dependencies listed in pubspec.yaml (image_picker, tflite_flutter, etc.)

---

## PART 3 – Build the APK

This is the single command that produces the file you can share:

```bash
flutter build apk --release
```

This takes 2-5 minutes the first time.
When it finishes you will see:

```
✓ Built build/app/outputs/flutter-apk/app-release.apk (XX.X MB)
```

That file is your shareable app!

---

## PART 4 – Share the APK

### Option A – WhatsApp / Telegram
Just send the APK file directly in a chat.
The file is at:
```
/Users/satvikchaudhary/Documents/Claude/flutter-sku-detection/build/app/outputs/flutter-apk/app-release.apk
```

### Option B – Google Drive / iCloud
Upload it and share the link.

### Option C – USB cable
Connect the phone, copy the file to Downloads folder on the phone.

---

## PART 5 – Install on Android Phone

The person receiving the APK (including yourself) needs to do this once:

1. **On the Android phone**, open **Settings**
2. Search for **"Install unknown apps"** or go to:
   `Settings → Apps → Special app access → Install unknown apps`
3. Allow the app you are using to send/open the file (e.g. WhatsApp, Files, Chrome)
4. Open the APK file → tap **Install**
5. Open **SKU Detector** from the home screen

---

## PART 6 – Test It

1. Open the app → you will see a loading screen while the model loads
2. Tap **Camera** to take a photo, or **Gallery** to pick an existing image
3. The app runs YOLO inference on the image using the phone's GPU
4. Bounding boxes and class labels appear on the image
5. The app bar shows: number of detected objects + inference time in milliseconds

---

## Common Problems & Fixes

| Problem | What to do |
|---------|-----------|
| `flutter: command not found` | Close Terminal and open a new window; or run `brew install --cask flutter` again |
| `flutter doctor` shows Android SDK missing | Open Android Studio (installed via brew) and complete its setup wizard |
| `brew: command not found` | Run the Homebrew install curl command from Part 1A |
| `minSdkVersion` error | Make sure you set it to 24 in android/app/build.gradle |
| APK installs but crashes immediately | Check that assets/best_float32.tflite exists and kModelAsset matches the filename |
| "App not installed" error on phone | The phone is blocking unknown sources — follow Part 5 above |
| Very slow inference | The GPU delegate may not be active — check the app bar time; >500ms means CPU fallback |

---

## Quick Reference – All Commands in Order

```bash
# ── INSTALL (one time only) ────────────────────────────────────────────────
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install --cask temurin@17
brew install --cask android-studio     # then open it once and run the setup wizard
brew install --cask flutter
flutter doctor --android-licenses      # press 'y' to all prompts

# ── PROJECT SETUP (one time only) ──────────────────────────────────────────
cd "/Users/satvikchaudhary/Documents/Claude/flutter-sku-detection"
flutter create . --project-name sku_detector

# ── CONVERT MODEL (once, or whenever you update best.pt) ───────────────────
pip install ultralytics
python convert_model.py
cp "/Users/satvikchaudhary/Desktop/IRED DOCS/All_Epochs/best_saved_model/best_float32.tflite" assets/

# ── BUILD & SHARE ───────────────────────────────────────────────────────────
flutter pub get
flutter build apk --release

# Open the folder containing the APK
open build/app/outputs/flutter-apk/
```
