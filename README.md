# SignSync

SignSync is a local ASL fingerspelling app. Its primary recognition mode uses the trained models and preprocessing from the adjacent `sign-language-cv` repository. Webcam frames are sent to a Python service on `127.0.0.1`; no camera data is sent to a remote inference service.

## Recognition flow

```text
Webcam → local SignSync service → original MediaPipe HandLandmarker
       → wrist/palm normalization → movement router
          ├─ static MLP: 24 letters (A–I, K–Y)
          └─ 30-frame LSTM: J, Z, or STATIC transition
       → recognized letter → SignSync text and browser speech
```

The adapter imports `HandTracker`, normalization, model classes, scaler, and checkpoints directly from `sign-language-cv`. It does not copy, convert, or retrain the models. SignSync also retains its earlier browser-based landmark and TensorFlow.js engines as optional modes.

## Windows setup

Keep the repositories beside each other using these folder names:

```text
Downloads\
  signsync\
  sign-language-cv\
```

Create an isolated environment in SignSync and install the small runtime dependency set:

```powershell
cd $HOME\Downloads\signsync
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the local site and model service:

```powershell
.\.venv\Scripts\python.exe serve_with_cors.py
```

Open <http://127.0.0.1:8001> in a modern browser and allow camera access. The first model request loads the checkpoints and MediaPipe task asset; keep the terminal running while you use the app. Press Ctrl+C to stop it.

If the recognition repository is in another location, set its path before starting the server:

```powershell
$env:SIGN_LANGUAGE_CV_PATH = 'D:\path\to\sign-language-cv'
.\.venv\Scripts\python.exe serve_with_cors.py
```

If the cloned repository already has a working environment, it can run the same service without installing a second one:

```powershell
cd $HOME\Downloads\signsync
..\sign-language-cv\.venv\Scripts\python.exe serve_with_cors.py
```

The included `requirements.txt` lists only packages needed by the adapter and cloned inference source. Python 3.12 is the recommended fresh environment. The service reports a clear error if the repository, either checkpoint, or `models\hand_landmarker.task` is missing.

## Current scope and limits

- Static model outputs are A–I and K–Y; J and Z require motion and use the temporal model. The MLP has no J class, and the LSTM's `STATIC` transition output is intentionally ignored.
- A recognized letter is appended once while its prediction remains active. The user can clear the text, and the browser's Web Speech API can speak it.
- Static letters must remain stable for 1.5 seconds before they are appended. J/Z events are appended when the LSTM recognizes them. This prevents frame-by-frame duplicates and reduces MLP pose detections during a motion sign.
- The app constructs a letter string. It does not infer words or translate continuous ASL grammar. ASL fingerspelling is not general sign-language translation.
- The original repository reports 99.5% static and 100% J/Z test accuracy, but its saved test split is not available here to reproduce those figures. Its dataset notes that the J/Z model was trained on one person. Treat those numbers as the repository's report, not a guarantee for new signers.
- The cloned repository README claims MIT licensing, but the checkout contains no `LICENSE` file. Confirm the upstream license before redistributing its source or model artifacts.
- Webcam permissions and the actual integrated camera path depend on a working camera and browser. A blank-frame API smoke check does not validate recognition on live gestures.
- The full model mode needs the local Python service; static hosting alone cannot serve this inference API. The older browser engines remain optional.

## Extending toward words and speech

The HTTP adapter separates camera capture from recognition and returns the letter, confidence, routing mode, and landmarks for each frame. A later word layer can consume stable letter events, add explicit word boundaries and editing, and keep its transcription separate from any sequence-to-word model. Word-level recognition needs suitable continuous-sign data and a separately trained temporal model; the current alphabet checkpoints cannot provide it. Text-to-speech can then read the confirmed recognized text rather than individual letters.
