# SignSync

SignSync is a local ASL fingerspelling application. The Python service, trained checkpoints, MediaPipe task asset, inference code, and preprocessing are stored in this repository. It does not need the separate training checkout at runtime.

## Recognition modes

- **Live webcam:** streams frames from the browser to the local Python service. The service extracts MediaPipe landmarks and routes static poses to the MLP and motion to the LSTM.
- **Image test:** runs one still image through the same 63-feature normalization and static MLP used by the live recognizer.
- **Video test:** samples 30 frames in order, normalizes 63 landmark values per frame, adds 63 velocity values, and runs the J/Z LSTM on the resulting `(30, 126)` sequence.

The MLP recognizes A–I and K–Y. The LSTM recognizes J, Z, and an internal STATIC transition class. SignSync appends stable letter events to a text string; it does not translate continuous ASL or infer words. Browser speech reads the accumulated text.

## Project layout

```text
signsync/
  models/
    mlp.pt
    lstm_jz.pt
    hand_landmarker.task
  recognition/
    main_app.py
    hand_tracker.py
    math_utils.py
    sequence_utils.py
  recognition_service.py
  serve_with_cors.py
  index.html
  tfjs_model/             # Optional browser model retained from the original app
```

Model classes and the MLP scaler/classes are stored in the checkpoints. `recognition/` contains the architectures, routing, landmark extraction, normalization, and sequence feature code. The `data/` training arrays are not needed for runtime inference and are not copied.

## Windows setup

Use Python 3.11. On this Windows setup it is installed in the per-user LocalAppData folder. From PowerShell in the SignSync folder:

```powershell
cd C:\Users\adwai\Downloads\signsync
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe serve_with_cors.py
```

Open <http://127.0.0.1:8001> and allow camera access. The site and API use the same local origin. Keep the terminal open while using SignSync; press Ctrl+C to stop it. The current local Python 3.11 environment has already loaded both checkpoints and the hand tracker successfully.

The service binds to `127.0.0.1`. Set `SIGNSYNC_PORT` before starting it if port 8001 is occupied:

```powershell
$env:SIGNSYNC_PORT = '8002'
.\.venv\Scripts\python.exe serve_with_cors.py
```

## Local API

- `GET /api/status` loads and reports model readiness, supported classes, and the LSTM sequence length.
- `POST /api/recognize` accepts a JPEG frame from the webcam.
- `POST /api/test-image?expected=A` accepts a JPEG, PNG, or WebP image (up to 10 MB).
- `POST /api/test-video?expected=J&name=clip.mp4` accepts a supported video (up to 50 MB).
- `GET /api/gallery` reads `assets/asl/manifest.json`.

Uploads are processed locally and temporary video files are removed after inference. Webcam and test media are not sent to a remote inference service. The optional TensorFlow.js browser demo remains available from its model controls; its browser libraries are loaded from their configured CDNs.

## Reference media

The testing tabs work with your own files. The reference catalog is optional. Add licensed image samples under `assets/asl/static/<LETTER>/` for A–I and K–Y, and video clips under `assets/asl/dynamic/J/` or `assets/asl/dynamic/Z/`; register each file in `assets/asl/manifest.json`. Keep each file's source and license in its manifest entry. The built-in J/Z reference points to a Wikimedia Commons video and requires an internet connection for playback and inference. Its attribution and CC BY-SA 3.0 terms are recorded in the manifest.

The repository currently does not include static reference photographs or local J/Z test videos. Model readiness and sample-array inference can be tested without them, but a real image/video classification needs a suitable hand media file. Video formats depend on the installed OpenCV codecs, and every one of the 30 sampled frames must contain a detectable hand.

## Attribution and licensing

The migrated ASL model source checkout described itself as MIT-licensed in its README, but no `LICENSE` file was present. `THIRD_PARTY_NOTICES.md` records this provenance and the unresolved redistribution status. The MLP and LSTM checkpoints are the existing trained SignSync assets; no retraining or weight conversion was performed.

The MediaPipe hand landmarker task was copied from Google's published MediaPipe model URL. MediaPipe source code is Apache-2.0 licensed; the model bundle's separate redistribution terms should be confirmed before distributing SignSync outside your own local use. See `THIRD_PARTY_NOTICES.md` for source URLs and this limitation. The reference video manifest records its CC BY-SA 3.0 attribution.
