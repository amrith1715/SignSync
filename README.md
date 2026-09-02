# 🤟 SignSync: Real-Time Sign Language to Speech Translator

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands_3D-teal.svg)](https://developers.google.com/mediapipe)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-GraphModel-orange.svg)](https://www.tensorflow.org/js)
[![Web Speech API](https://img.shields.io/badge/Web_Speech-TTS-blue.svg)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)

> **A real-time, browser-based computer vision system that translates American Sign Language (ASL) fingerspelling gestures into text and natural spoken voice — requiring only a standard laptop webcam and zero specialized hardware.**

---

## 🎯 Overview & Problem Statement

Millions of deaf and hard-of-hearing individuals communicate primarily through sign language, yet the vast majority of hearing people never learn it. Existing digital interpreter solutions are often expensive, require specialized sensory gloves, or rely on paid cloud APIs with high latency.

**SignSync** solves this communication gap by running **100% client-side inside the browser**:
- **Zero Cloud Latency**: All image processing and ML inference runs locally on your CPU/GPU.
- **Universal Accessibility**: Works on any device with a modern browser and a webcam.
- **Privacy-Preserving**: Video streams never leave the user's device.

---

## ⚙️ System Architecture

```
  [ Webcam Feed ] 
         │
         ▼
  [ MediaPipe Hands ] ───▶ 21 Invariant 3D Keypoints (x, y, z)
         │
         ├───▶ [ Engine 1: 3D Landmark Classifier ] ──▶ Real-Time Gesture Mapping (< 0.2ms)
         │                                                        │
         └───▶ [ Engine 2: 28×28 Pixel CNN ] ──────────▶ Deep Learning Benchmark
                                                                  │
                                                                  ▼
                                                      [ Temporal Vote Filter ]
                                                                  │
                                                                  ├──▶ [ Real-Time Text Transcription ]
                                                                  │
                                                                  └──▶ [ Text-to-Speech Voice Output 🔊 ]
```

### 🧠 Dual-Engine Architecture
SignSync features an interactive **Engine Switcher** (`🎯 Landmark 3D` vs. `🧠 CNN Pixel`):
1. **MediaPipe 3D Landmark Engine (Recommended)**:
   - Evaluates 21 normalized 3D keypoint vectors, inter-finger angles, and pinch distances.
   - **100% invariant** to room background clutter, lighting conditions, shadows, and skin tones.
   - Executes in **$< 0.2\text{ ms}$** directly on CPU (5,000+ FPS potential).
2. **28×28 Deep CNN Engine**:
   - Deep Convolutional Neural Network (Conv2D $\to$ BatchNorm $\to$ MaxPool $\to$ GlobalAvgPool $\to$ Dense) trained on the 24-class Sign MNIST benchmark.
   - Converted to TensorFlow.js GraphModel format for browser inference.

---

## ✨ Features

- ⚡ **Instant Auto-Start**: Automatically boots up the camera and classifier upon page load.
- 🔊 **Voice Feedback**: Real-time Text-to-Speech (TTS) automatically speaks formed letters.
- 👁️ **Live Diagnostic HUD**: Toggle **Debug View** to inspect real-time finger extension states (`T:🟢 I:🟢 M:⚪ R:⚪ P:⚪`), hand tilt, and top-3 candidate probabilities.
- 🔄 **Left & Right Hand Support**: Click **Flip Hand** to mirror keypoints for left-handed signing or mirrored webcams.
- 🖼️ **Image Upload Mode**: Upload static sign language images to test individual classifications.
- 🎨 **Modern Glassmorphism UI**: Dark-mode, fluid responsive design built with vanilla CSS.

---

## 📁 Repository Structure

```
signsync_github/
├── index.html              # Main web application (GitHub Pages entry point)
├── signsync_app.html       # Application source
├── label_mapping.json      # 24-class ASL alphabet mapping (A-Y, skipping J & Z)
├── model.json              # TensorFlow.js GraphModel architecture definition
├── group1-shard1of1.bin    # TensorFlow.js GraphModel weights (2.36 MB)
├── tfjs_model/             # Standalone model directory
│   ├── model.json
│   └── group1-shard1of1.bin
├── serve_with_cors.py      # Lightweight local HTTP server script with CORS
├── requirements.txt        # Optional Python dependencies
├── .gitignore              # Ignores virtual environments & temp files
└── README.md               # Project documentation
```

---

## 🚀 Quick Start Guide

### Option 1: Run Locally (Fastest)

1. Clone or download this repository:
   ```bash
   git clone https://github.com/your-username/SignSync.git
   cd SignSync
   ```

2. Start a local HTTP server:
   ```bash
   # Using Python built-in server:
   python -m http.server 8000

   # Or using the included CORS server:
   python serve_with_cors.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

4. Allow webcam access, hold your hand in front of the camera, and start signing!

---

### Option 2: Deploy Free on GitHub Pages (1-Click Hosting)

Because SignSync is 100% client-side with an `index.html` file, you can host it live on the web for free:

1. Push this repository to GitHub.
2. In your GitHub repository, navigate to **Settings** $\to$ **Pages**.
3. Under **Branch**, select `main` (or `master`) and folder `/ (root)`.
4. Click **Save**.
5. Your live app will be accessible worldwide at:
   `https://<your-username>.github.io/<repo-name>/`

---

## 🖐️ ASL Fingerspelling Reference

| Sign | Gesture Description | Key Landmark Features |
| :---: | :--- | :--- |
| **A** | Fist with thumb resting beside index | 4 fingers curled, thumb pointing straight up |
| **B** | Flat hand facing forward | 4 fingers extended straight up, thumb tucked across palm |
| **C** | Open curved hand forming a 'C' | All fingers and thumb curved with opening |
| **D** | Index finger pointing up | Index extended, middle/ring/pinky touching thumb |
| **F** | "OK" sign | Thumb and index touching in circle, other 3 fingers straight up |
| **I** | Pinky straight up | Only pinky extended, other 4 fingers curled |
| **L** | "L" shape at 90 degrees | Thumb and index extended perpendicular, others curled |
| **V** | Peace sign | Index and middle extended and spread apart |
| **W** | 3 fingers straight up | Index, middle, and ring extended and spread |
| **Y** | "Hang Loose" / "Shaka" | Thumb and pinky extended, middle 3 curled |

---

## 🛠️ Technology Stack

- **MediaPipe Hands**: Real-time 3D hand tracking & 21-landmark extraction
- **TensorFlow.js**: Client-side neural network execution
- **Web Speech API**: In-browser Text-to-Speech audio synthesis
- **HTML5 Canvas & Vanilla JS**: Zero frontend framework bloat, fast 60 FPS rendering

---

## 📜 License

This project is licensed under the MIT License - feel free to use and extend it for academic and research purposes.