"""Adapter that exposes the cloned sign-language-cv predictor to SignSync."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class RuntimeBusyError(RuntimeError):
    """The single-frame inference slot is occupied; callers should drop frames."""


def _repository_root() -> Path:
    configured = os.environ.get("SIGN_LANGUAGE_CV_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "sign-language-cv"


class RecognitionRuntime:
    """Loads the original models and keeps one temporal predictor per server."""

    def __init__(self) -> None:
        self.repository = _repository_root()
        required = [
            self.repository / "src" / "main_app.py",
            self.repository / "saved_models" / "mlp.pt",
            self.repository / "saved_models" / "lstm_jz.pt",
            self.repository / "models" / "hand_landmarker.task",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Recognition files were not found: " + ", ".join(missing)
                + ". Set SIGN_LANGUAGE_CV_PATH to the cloned repository."
            )

        source = self.repository / "src"
        sys.path.insert(0, str(source))
        import main_app  # type: ignore[import-not-found]
        from hand_tracker import HandTracker  # type: ignore[import-not-found]

        self._main_app = main_app
        mlp, scaler, mlp_classes = main_app.load_mlp(
            self.repository / "saved_models" / "mlp.pt"
        )
        lstm, lstm_classes, seq_len = main_app.load_lstm(
            self.repository / "saved_models" / "lstm_jz.pt"
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
            model_path=self.repository / "models" / "hand_landmarker.task",
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
        "source": "sign-language-cv",
    }
