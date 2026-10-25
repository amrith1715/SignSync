# SignSync

SignSync is a local ASL fingerspelling application. The Python service, trained checkpoints, MediaPipe task asset, inference code, and preprocessing are stored in this repository. It does not need the separate training checkout at runtime.

For public deployment on Vercel with the Python API on an Oracle Cloud VM, see [DEPLOYMENT.md](DEPLOYMENT.md).

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
  dynamic_sign_videos/    # J/Z animated references and posters for the ASL chart
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

The local service binds to `127.0.0.1`. Set `SIGNSYNC_PORT` before starting it if port 8001 is occupied:

```powershell
$env:SIGNSYNC_PORT = '8002'
.\.venv\Scripts\python.exe serve_with_cors.py
```

## Application navigation and testing

The sidebar opens Recognition, Testing Lab, ASL Alphabet, and About. Mobile
screens use a collapsible menu. Recognition includes the real hand landmark
overlay, confirmation delay, spelled text editing, and browser speech.

In Testing Lab, choose multiple images (JPG/PNG/WebP) and/or videos
(MP4/MOV/AVI/MKV/WebM/OGV), up to 4 MB per file. Review the queue, then click
**Predict all**. Files are sent to the recognition API one at a time, and each
file receives its own result. Remove any queued file before prediction.
The image and video panels sit side by side on desktop. Live recognition pauses
when leaving its workspace and resumes on return after file testing finishes.

The current static MLP was adapted using the 24 chart images and replay of the
existing landmark training dataset. All 24 known images passed repeated API
checks on October 23, 2026. They are adaptation examples, not an independent
accuracy benchmark. Original-data regression accuracy changed from 99.89% to
99.78%. See [the full report](reports/static_signs_report.txt) and
[the latest retest](reports/static_signs_latest_retest.txt).

Run the automated static-image check with the local server running:

```powershell
.\.venv\Scripts\python.exe audit_static_signs.py --url http://127.0.0.1:8001 --output reports/static_signs_retest.json
```

This writes JSON and TXT results for `static_signs_images`. The active checkpoint
is `models/mlp.pt`; the original is preserved as
`models/mlp_before_static_adaptation.pt`. `adapt_static_mlp.py` documents the
adaptation procedure and requires the original external landmark dataset at its
configured local path. It is not required to run the application.

## Local API

- `GET /api/status` loads and reports model readiness, supported classes, and the LSTM sequence length.
- `POST /api/recognize` accepts a JPEG frame from the webcam.
- `POST /api/test-image?expected=A` accepts a JPEG, PNG, or WebP image (up to 4 MB).
- `POST /api/test-video?expected=J&name=clip.mp4` accepts a supported video (up to 4 MB).
- `GET /api/gallery` reads `assets/asl/manifest.json`.

Uploads are processed locally and temporary video files are removed after inference. Webcam and test media are not sent to a remote inference service. Legacy TensorFlow.js files remain in the repository, but the application uses the local MLP/LSTM pipeline.

## Reference media

The alphabet view uses 24 user-supplied static photographs under
`assets/asl/static/<LETTER>.png`, retaining their descriptions. These derive from
the supplied StartASL.com chart (copyright 2008). J and Z have six-second animated
hand references under `dynamic_sign_videos/`, with white backgrounds, slow motion,
native playback controls, and a fade before looping. They show the movement
from the learner's viewpoint. The separate `static_signs_images`
folder contains the 224 x 224 RGB reference images used in the audit.

Video formats depend on installed OpenCV codecs. Every one of the 30 sampled
frames must contain a detectable hand.

The generated illustrations use the static photographs as appearance references.
Their source artwork and complete image-generation prompts are preserved in
`dynamic_sign_videos/source/generation.json`. They have not been independently
reviewed by a fluent ASL signer and are not recognition accuracy test data.
`scripts/render-dynamic-signs.py` reproduces the clips using Pillow, NumPy,
OpenCV, and FFmpeg; these rendering tools are not needed to play the videos.

## Attribution and licensing

The migrated ASL model source checkout described itself as MIT-licensed in its README, but no `LICENSE` file was present. `THIRD_PARTY_NOTICES.md` records this provenance and the unresolved redistribution status. The static MLP was subsequently adapted as documented above; the LSTM checkpoint was retained.

The MediaPipe hand landmarker task was copied from Google's published MediaPipe model URL. MediaPipe source code is Apache-2.0 licensed; the model bundle's separate redistribution terms should be confirmed before distributing SignSync outside your own local use. See `THIRD_PARTY_NOTICES.md` for source URLs, generated-media provenance, and the legacy Commons reference.
