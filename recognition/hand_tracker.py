"""
hand_tracker.py
---------------
Wrapper around the MediaPipe Tasks API for landmark extraction.

NOTE: The 'solutions' API (mp.solutions.hands) was removed in MediaPipe versions 0.10.13 and later. This module uses the new Tasks API, which is the official and maintained way to use MediaPipe in modern versions.

The public interface (HandLandmarks, HandTracker) is identical to the previous version—the rest of the project requires no changes.

The 'hand_landmarker.task' model file is automatically downloaded to the 'models/' folder if it doesn't already exist.

Landmark index map:

0 - Wrist

1 - Thumb CMC
5 - Index MCP
9 - Heart MCP

2 - Thumb MCP
6 - Index PIP
10 - Heart PIP

3 - Thumb IP
7 - Index DIP 11 - Middle DIP

4 - Thumb TIP 8 - Index TIP 12 - Middle TIP

13 - Ring MCP 17 - Pinky MCP

14 - Ring PIP 18 - Pinky PIP

15 - Ring DIP 19 - Pinky DIP

16 - Ring TIP 20 - Pinky TIP

Dependencies:

pip install mediapipe opencv-python numpy
"""

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import mediapipe as mp                           # type: ignore[import]
import numpy as np
from mediapipe.tasks import python as mp_python  # type: ignore[import]
from mediapipe.tasks.python import vision as mp_vision  # type: ignore[import]


# Model — automatic download
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def _ensure_model(model_path: Path) -> Path:
    """Download the model if it doesn't exist. Return the file path."""
    if model_path.exists():
        return model_path
    print(f"[HandTracker] Model not found. Downloading '{model_path}'...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_MODEL_URL, model_path)
    print(f"[HandTracker] Model downloaded successfully.")
    return model_path

# Hand connections for drawing the skeleton with OpenCV

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (0, 9), (9, 10), (10, 11), (11, 12),      # Heart
    (0, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # Little finger
    (5, 9), (9, 13), (13, 17),                # Palm
]

# Data container

@dataclass
class HandLandmarks:
    """
    Container with landmark data from a hand detection.
    Attributes:
    landmarks_mp: List of native landmarks from MediaPipe Tasks.
    landmarks_array: NumPy array of form (21, 3). Columns: [x, y, z].
    x, y normalized to [0, 1]; z relative to the wrist.
    handedness: "Left" or "Right" according to MediaPipe.
    """
    landmarks_mp: Any
    landmarks_array: np.ndarray
    handedness: str

    @property
    def wrist(self) -> np.ndarray:
        """Wrist coordinates [x, y, z] (landmark 0)."""
        return self.landmarks_array[0]

    @property
    def flat(self) -> np.ndarray:
        """
        1D vector with 63 features ready for machine learning.
        Order: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        """
        return self.landmarks_array.flatten()

# HandTracker

class HandTracker:
    """
    Clean interface over the MediaPipe Tasks API for landmark extraction.
    Using as a context manager (recommended):
    with HandTracker() as tracker:
    annotated, hand = tracker.process_frame(frame)

    Manual usage:
    tracker = HandTracker()
    tracker.start()
    annotated, hand = tracker.process_frame(frame)
    tracker.stop()

    Args:
    model_path: Path to the hand_landmarker.task file.

    Automatically downloaded if it doesn't exist.
    max_num_hands: Maximum number of hands to detect.
    min_detection_confidence: Detection threshold [0.0, 1.0].
    min_tracking_confidence: Tracking threshold between frames [0.0, 1.0].
    """

    # Semantic indexes of key points
    WRIST      = 0
    THUMB_TIP  = 4
    INDEX_TIP  = 8
    MIDDLE_TIP = 12
    RING_TIP   = 16
    PINKY_TIP  = 20

    NUM_LANDMARKS = 21
    FEATURE_SIZE  = NUM_LANDMARKS * 3  # 63

    # Colors for drawing (GBR)
    _COLOR_JOINT = (0, 255, 128)
    _COLOR_BONE  = (255, 200, 0)

    def __init__(
        self,
        model_path: Path = _DEFAULT_MODEL_PATH,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        static_image_mode: bool = False,
        # Legacy parameter ignored — maintained for compatibility with
        # calls that included static_image_mode or model_complexity
        **_kwargs: Any,
    ) -> None:
        self._model_path             = _ensure_model(Path(model_path))
        self._static_image_mode      = static_image_mode
        self._max_num_hands          = max_num_hands
        self._min_detection_conf     = min_detection_confidence
        self._min_tracking_conf      = min_tracking_confidence
        self._landmarker: Any        = None
        self._last_timestamp_ms: int = 0

    # Life cicle

    def start(self) -> None:
        """Initializes the MediaPipe HandLandmarker model."""
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(self._model_path)
            ),
            running_mode=(mp_vision.RunningMode.IMAGE if self._static_image_mode else mp_vision.RunningMode.VIDEO),
            num_hands=self._max_num_hands,
            min_hand_detection_confidence=self._min_detection_conf,
            min_hand_presence_confidence=self._min_detection_conf,
            min_tracking_confidence=self._min_tracking_conf,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def stop(self) -> None:
        """Release the model's resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandTracker":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # Frame processing
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        draw: bool = True,
    ) -> Tuple[np.ndarray, Optional[HandLandmarks]]:
        """
        Processes an OpenCV BGR frame and extracts the hand landmarks.

        Args:
        frame_bgr: Frame captured by cv2.VideoCapture (BGR format).

        Returns:

        (annotated_frame, hand_landmarks):

        - annotated_frame: Frame with the skeleton drawn.

        - hand_landmarks: HandLandmarks or None if there is no hand.

        Raises:

        RuntimeError: If called without initializing the tracker.
        """
        if self._landmarker is None:
            raise RuntimeError(
                "HandTracker not initialized." 
                "Use 'with HandTracker() as tracker:' or call .start() first."
            )

        # The Tasks API requires RGB and a timestamp in milliseconds
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        result = (self._landmarker.detect(mp_image) if self._static_image_mode
                  else self._landmarker.detect_for_video(mp_image, timestamp_ms))

        # Web applications render landmarks in their own overlay.  Let them
        # skip this copy and OpenCV skeleton when only model landmarks are used.
        annotated      = frame_bgr.copy() if draw else frame_bgr
        hand_landmarks = None

        if result.hand_landmarks:
            lm_list    = result.hand_landmarks[0]
            handedness = (
                result.handedness[0][0].category_name
                if result.handedness else "Unknown"
            )

            landmarks_array = self._to_array(lm_list, frame_bgr.shape)
            if draw:
                self._draw_skeleton(annotated, lm_list)

            hand_landmarks = HandLandmarks(
                landmarks_mp=lm_list,
                landmarks_array=landmarks_array,
                handedness=handedness,
            )

        return annotated, hand_landmarks

    # Internal Utilities
    def _to_array(self, lm_list: Any, shape: tuple) -> np.ndarray:
        """
        Convert the Tasks API landmark list to an array (21, 3).

        In the Tasks API, x and y are still normalized to [0,1].
        z is the depth relative to the wrist (same convention as before).
        """
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in lm_list],
            dtype=np.float32,
        )

    def _draw_skeleton(self, frame: np.ndarray, lm_list: Any) -> None:
        """Draws the skeleton of the hand on the frame using OpenCV."""
        h, w = frame.shape[:2]

        # Convert normalized coordinates to pixels
        pts = [
            (int(lm.x * w), int(lm.y * h))
            for lm in lm_list
        ]

        # Bones
        for start, end in _HAND_CONNECTIONS:
            cv2.line(frame, pts[start], pts[end], self._COLOR_BONE, 2, cv2.LINE_AA)

        # Joints
        for pt in pts:
            cv2.circle(frame, pt, 5, self._COLOR_JOINT, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)  
