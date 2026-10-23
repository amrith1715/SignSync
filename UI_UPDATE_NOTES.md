# UI and landmark update — 23 October 2026

The missing skeleton was caused by a data contract mismatch: the Python API sends
21 `[x, y, z]` arrays, but the JavaScript renderer read `.x` and `.y` properties.
Canvas therefore received NaN coordinates. Increasing marker size did not fix it.

`assets/landmark-overlay.js` validates the actual API arrays, maps the backend's
mirrored coordinates into the unmirrored preview, and draws 21 small outlined
points with the 21 standard hand connections. No extra tracker or model is used.
Video and canvas share the same contained rectangle. Recognition inputs,
normalization, checkpoints, 63/126 features and 30-frame sequence are unchanged.

`index.html` also improves badge layering/contrast, camera error messages, request
timeouts, camera stop/disconnect cleanup, frame callback cancellation, mobile
spacing, and the speech button's missing ID. API image/video testing remains
available; previously removed upload panels and the CNN selector stay removed.

## Checks performed

- Local server started with the existing Python environment; readiness returned
  24 static classes, J/Z, and sequence length 30.
- Real MediaPipe detection on a padded user-provided C reference crop returned
  21 finite landmark arrays. A blank frame returned no hand.
- An isolated headless Chrome page rendered that real API response: 21 points,
  6,216 nontransparent canvas pixels; a no-hand response cleared every pixel.
- Coordinate checks passed for unmirrored and mirrored preview mappings; invalid
  landmark data is rejected. JavaScript parsed and no uncaught page errors were
  reported in the final browser check.
- At widths 1280, 768 and 390, video/canvas rectangles matched, both used contain,
  and the document had no horizontal overflow. All 24 chart images decoded.
- Camera permission denial displayed a clear camera error in headless Chrome.
- Offline image route returned a low-confidence result for the C reference
  (about 47%, no confident letter). The live route classified it as B. This is
  evidence of detection/transport, not evidence of correct letter recognition.
- Offline video route processed 30 frames successfully. A control video repeating
  the same still frame was classified as Z at high confidence: a model limitation
  requiring real movement tests. No genuine J/Z video was available.
- An invalid image upload returned HTTP 400 with a readable error.

## Still needs manual verification

Real webcam permission/start/stop/restart, hand motion alignment and latency,
camera FPS under normal use, actual static-letter and J/Z accuracy, audio playback,
and physical device resizing. Automated browser rendering used real recorded
MediaPipe landmarks, not a live hand. No accuracy or FPS guarantee is made.

Refresh http://127.0.0.1:8001 with Ctrl+F5 before checking the updated overlay.

## Follow-up: isolate drawing from predictions

After a report of prediction interference, removed the overlay-validation throw
from the prediction handler. A valid letter now updates even if landmark data is
missing or drawing fails. Drawing runs once on the next browser paint and has its
own error handling; stopping the camera cancels pending paint work. Busy responses
clear old landmarks without resetting the letter hold timer. Camera uploads still
come directly from the video element, never from the overlay canvas.

Regression checks injected missing landmark data and a renderer exception while
supplying valid prediction responses: letter output and hold state survived both;
no-hand results cleared the overlay. JavaScript parsing and diff checks passed.
These checks do not establish live recognition accuracy or latency.

## Follow-up: unstable visible letters

The visible result previously changed on every response; J/Z were immediately
spoken and appended after one response. The UI now requires three consecutive
matching results in the same mode. Static letters also need 250 ms before display
and retain the original 1.5-second acceptance hold. Unknown, low-confidence, or
changing results show a reading state instead of a stale letter. The checkpoints,
backend routing, feature extraction and model confidence threshold are unchanged.

Sequence regression checks passed for fluctuating letters, stable static display
and acceptance, one-frame J rejection, repeated J acceptance without duplicates,
low confidence, and hand loss. This adds confirmation latency and can suppress
brief dynamic detections; genuine J/Z gestures still need manual testing. It does
not correct consistently wrong model classifications.

## Motion confirmation delay

Following reports of X/Z and I/J confusion, display confirmation now requires
600 ms of matching predictions and at least three responses, including for J/Z.
After the last motion-mode response, static display also waits 600 ms for motion
to settle. Candidate or mode changes restart confirmation. Static text acceptance
retains its 1.5-second hold. The UI shows "Confirming sign..." while waiting;
camera capture and the LSTM's 30-frame input continue without a sleep or delay.
Sequence checks passed for I-to-J transitions, alternating X/Z, stable J/Z after
600 ms and static text acceptance after 1.5 seconds. Real gestures remain to be
checked; the delay can reject brief detections and does not fix model accuracy.
