"""SignSync's local static and dynamic ASL recognition runtime."""

from __future__ import annotations

import os
import threading
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class RuntimeBusyError(RuntimeError):
    """The single-frame inference slot is occupied; callers should drop frames."""


class RecognitionRuntime:
    """Loads SignSync's trained models and keeps one temporal predictor per server."""

    def __init__(self) -> None:
        self.root = Path(__file__).resolve().parent
        self.models = self.root / "models"
        required = [
            self.root / "recognition" / "main_app.py",
            self.models / "mlp.pt",
            self.models / "lstm_jz.pt",
            self.models / "hand_landmarker.task",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "SignSync recognition files are missing: " + ", ".join(missing)
                + ". Restore the files under SignSync's recognition/ and models/ folders."
            )

        from recognition import main_app
        from recognition.hand_tracker import HandTracker

        self._main_app = main_app
        mlp, scaler, mlp_classes = main_app.load_mlp(
            self.models / "mlp.pt"
        )
        lstm, lstm_classes, seq_len = main_app.load_lstm(
            self.models / "lstm_jz.pt"
        )
        self.predictor = main_app.HybridPredictor(
            mlp_model=mlp,
            mlp_scaler=scaler,
            mlp_classes=mlp_classes,
            lstm_model=lstm,
            lstm_classes=lstm_classes,
            seq_len=seq_len,
            smoothing=10,
            min_conf=0.60,
            movement_thr=0.0005,
            min_motion_frames=8,
            cooldown_frames=20,
        )
        self.static_letters = [str(letter) for letter in mlp_classes]
        self.dynamic_letters = [
            str(letter) for letter in lstm_classes if str(letter) in {"J", "Z"}
        ]
        self.tracker = HandTracker(
            model_path=self.models / "hand_landmarker.task",
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
        )
        self.tracker.start()
        self._lock = threading.Lock()
        self._request_gate = threading.Lock()
        self._closed = False

    def process_jpeg(self, payload: bytes) -> dict[str, Any]:
        if not self._request_gate.acquire(blocking=False):
            raise RuntimeBusyError("Recognition is busy; this frame was dropped.")
        try:
            return self._process_jpeg_locked(payload)
        finally:
            self._request_gate.release()

    def test_image(self, payload: bytes, expected: str | None = None) -> dict[str, Any]:
        """Run a still reference through the production tracker and static MLP."""
        if not self._request_gate.acquire(blocking=False):
            raise RuntimeBusyError("Recognition is busy; try again shortly.")
        try:
            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                raise ValueError("The file is not a readable image.")
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0 or height * width > 20_000_000:
                raise ValueError("Image dimensions must be valid and no larger than 20 megapixels.")
            # Match the camera pipeline's orientation before extracting features.
            frame = cv2.flip(frame, 1)
            # Landmark x/z values are normalized by image width, while y uses
            # height. Preserve the camera's 4:3 coordinate geometry for uploads
            # by padding (never stretching) before extracting landmarks.
            camera_width = max(width, round(height * 4 / 3))
            camera_height = max(height, round(width * 3 / 4))
            top = (camera_height - height) // 2
            left = (camera_width - width) // 2
            frame = cv2.copyMakeBorder(
                frame, top, camera_height - height - top,
                left, camera_width - width - left,
                cv2.BORDER_CONSTANT, value=(255, 255, 255),
            )
            height, width = frame.shape[:2]
            with self._lock:
                self.predictor.reset()
                # An unrelated uploaded photo must not inherit webcam tracking.
                # Keep the same MediaPipe asset and thresholds, in IMAGE mode.
                with type(self.tracker)(
                    model_path=self.models / "hand_landmarker.task",
                    min_detection_confidence=0.75,
                    min_tracking_confidence=0.75,
                    static_image_mode=True,
                ) as image_tracker:
                    _, hand = image_tracker.process_frame(frame, draw=False)
                    if hand is None:
                        # Tight reference crops can leave too little context for
                        # palm detection. Add a camera-shaped margin, without
                        # stretching the hand or modifying the uploaded file.
                        padded_height = int(max(height * 1.35, width * 1.35 * .75))
                        padded_width = round(padded_height * 4 / 3)
                        top = (padded_height - height) // 2
                        left = (padded_width - width) // 2
                        padded = cv2.copyMakeBorder(
                            frame, top, padded_height - height - top,
                            left, padded_width - width - left,
                            cv2.BORDER_CONSTANT, value=(255, 255, 255),
                        )
                        _, hand = image_tracker.process_frame(padded, draw=False)
                if hand is None:
                    raise ValueError("No hand was detected in the image.")
                normalized = self._main_app.normalize_flat(hand.flat)
                letter, confidence = self.predictor._predict_mlp(normalized)
                self.predictor.reset()
            return self._test_result("static", letter, confidence, expected, 1)
        finally:
            with self._lock:
                self.predictor.reset()
            self._request_gate.release()

    def test_video(
        self, payload: bytes, suffix: str, expected: str | None = None,
        start_seconds: float | None = None, end_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Sample a clip in temporal order and run the original 30-frame LSTM."""
        if not self._request_gate.acquire(blocking=False):
            raise RuntimeBusyError("Recognition is busy; try again shortly.")
        temp_path = None
        capture = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as stream:
                stream.write(payload)
                temp_path = stream.name
            capture = cv2.VideoCapture(temp_path)
            if not capture.isOpened():
                capture.release()
                raise ValueError("The file is not a readable supported video.")
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0:
                capture.release()
                raise ValueError("Could not determine the video's frame count.")
            if count < self.predictor.seq_len:
                capture.release()
                raise ValueError(f"Video needs at least {self.predictor.seq_len} decodable frames.")
            first_frame, last_frame = 0, count - 1
            if start_seconds is not None or end_seconds is not None:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if not np.isfinite(fps) or fps <= 0:
                    capture.release()
                    raise ValueError("Could not determine this video's frame rate.")
                duration = count / fps
                start = 0.0 if start_seconds is None else float(start_seconds)
                end = duration if end_seconds is None else float(end_seconds)
                if start < 0 or end <= start or end > duration + (1.0 / fps):
                    capture.release()
                    raise ValueError("The requested video segment is outside the media duration.")
                first_frame = min(count - 1, max(0, int(start * fps)))
                last_frame = min(count - 1, max(first_frame, int(end * fps) - 1))
            indices = np.linspace(first_frame, last_frame, self.predictor.seq_len, dtype=int)
            normalized_frames = []
            with self._lock:
                self.predictor.reset()
                for index in indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise ValueError("Could not decode enough frames from this video.")
                    _, hand = self.tracker.process_frame(frame, draw=False)
                    if hand is None:
                        raise ValueError(f"No hand detected in sampled frame {len(normalized_frames) + 1}/{self.predictor.seq_len}.")
                    normalized_frames.append(self._main_app.normalize_flat(hand.flat))
                sequence = np.stack(normalized_frames, axis=0)
                features = self._main_app.add_velocity_features(sequence)
                import torch
                with torch.no_grad():
                    probabilities = torch.softmax(
                        self.predictor.lstm_model(torch.tensor(features, dtype=torch.float32).unsqueeze(0)), dim=1
                    ).squeeze()
                confidence = float(probabilities.max().item())
                letter = str(self.predictor.lstm_classes[probabilities.argmax().item()])
            if letter == "STATIC" or confidence < self.predictor.min_conf:
                letter = None
            return self._test_result("dynamic", letter, confidence, expected, len(normalized_frames))
        finally:
            if capture is not None:
                capture.release()
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            with self._lock:
                self.predictor.reset()
            self._request_gate.release()

    @staticmethod
    def _test_result(kind: str, letter: str | None, confidence: float, expected: str | None, frames: int) -> dict[str, Any]:
        expected = expected.upper() if expected else None
        return {
            "media_type": kind, "expected": expected, "predicted": letter,
            "confidence": float(confidence),
            "correct": (letter == expected) if expected else None,
            "frames_processed": frames,
        }

    def _process_jpeg_locked(self, payload: bytes) -> dict[str, Any]:
        total_started = time.perf_counter()
        decode_started = total_started
        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("The request did not contain a valid JPEG frame.")

        # Match the original desktop app, which mirrors the camera frame
        # before MediaPipe extraction and model preprocessing.
        frame = cv2.flip(frame, 1)
        decode_ms = (time.perf_counter() - decode_started) * 1000
        with self._lock:
            if self._closed:
                raise RuntimeError("The recognition service has been stopped.")
            tracker_started = time.perf_counter()
            _, hand = self.tracker.process_frame(frame, draw=False)
            mediapipe_ms = (time.perf_counter() - tracker_started) * 1000
            if hand is None:
                self.predictor.reset()
                return {
                    "hand_detected": False, "letter": None, "confidence": 0.0,
                    "timings_ms": {
                        "decode": round(decode_ms, 2),
                        "mediapipe": round(mediapipe_ms, 2),
                        "inference": 0.0,
                        "pipeline": round((time.perf_counter() - total_started) * 1000, 2),
                    },
                }

            inference_started = time.perf_counter()
            letter, confidence, mode = self.predictor.predict(hand)
            inference_ms = (time.perf_counter() - inference_started) * 1000
            model_timings = {
                key: round(float(value), 2)
                for key, value in self.predictor.last_timing_ms.items()
                if key.endswith("_ms")
            }
            return {
                "hand_detected": True,
                "landmarks": hand.landmarks_array.tolist(),
                "letter": letter,
                "confidence": float(confidence),
                "mode": mode,
                "timings_ms": {
                    "decode": round(decode_ms, 2),
                    "mediapipe": round(mediapipe_ms, 2),
                    "inference": round(inference_ms, 2),
                    "pipeline": round((time.perf_counter() - total_started) * 1000, 2),
                    **model_timings,
                },
            }

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self.tracker.stop()
                self._closed = True


_runtime: RecognitionRuntime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> RecognitionRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = RecognitionRuntime()
    return _runtime


def status() -> dict[str, Any]:
    runtime = get_runtime()
    return {
        "ready": True,
        "static_letters": runtime.static_letters,
        "dynamic_letters": runtime.dynamic_letters,
        "sequence_length": runtime.predictor.seq_len,
        "source": "SignSync",
    }
