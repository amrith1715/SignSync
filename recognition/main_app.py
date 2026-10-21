"""
main_app.py
-----------
Real-time ASL alphabet recognition (A-Z) with word autocorrection
and text-to-speech.

Pipeline:
    Camera -> MediaPipe -> MLP (static A-Y) / LSTM (motion J,Z)
           -> SymSpell autocorrect -> pyttsx3 TTS

Controls:
    [ENTER]      -- Confirm suggested word: speak it and add to sentence.
    [BACKSPACE]  -- Delete last detected letter.
    [SPACE]      -- Discard current letters (clear word in progress).
    [Q] / [ESC]  -- Quit.

Usage:
    python -m recognition.main_app
    python -m recognition.main_app --mlp models/mlp.pt --lstm models/lstm_jz.pt
    python -m recognition.main_app --camera 1 --tts_rate 150
"""

import argparse
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from .hand_tracker import HandTracker, HandLandmarks
from .math_utils import normalize_flat
from .sequence_utils import add_velocity_features


# ---------------------------------------------------------------------------
# Model architectures  (identical to training notebooks)
# ---------------------------------------------------------------------------

class SignLanguageMLP(nn.Module):
    def __init__(self, n_features=63, n_classes=24, dropout=(0.4, 0.3, 0.2)):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(n_features, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(dropout[0])
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(dropout[1])
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(dropout[2])
        )
        self.output = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.block3(self.block2(self.block1(x))))


class SignLanguageLSTM(nn.Module):
    def __init__(
        self,
        input_size:  int   = 126,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        n_classes:   int   = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_mlp(model_path: Path):
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    model = SignLanguageMLP(**ck["model_config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"[MLP]  Loaded: {model_path.name}  "
          f"acc={ck['test_acc']*100:.1f}%  classes={len(ck['classes'])}")
    return model, ck["scaler"], ck["classes"]


def load_lstm(model_path: Path):
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    model = SignLanguageLSTM(**ck["model_config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"[LSTM] Loaded: {model_path.name}  "
          f"acc={ck['test_acc']*100:.1f}%  "
          f"classes={list(ck['classes'])}  seq_len={ck['seq_len']}")
    return model, ck["classes"], ck["seq_len"]


# ---------------------------------------------------------------------------
# Word suggester  (SymSpell)
# ---------------------------------------------------------------------------

class WordSuggester:
    """
    Uses SymSpell to suggest the most likely English word from a sequence
    of detected letters.

    SymSpell finds the closest dictionary word tolerating insertions,
    deletions and substitutions up to max_edit_distance. This handles:
      - Missing letters   (HLO    -> HELLO)
      - Extra letters     (HELLLOO -> HELLO)
      - Wrong letters     (HLELO  -> HELLO)

    Args:
        max_edit_distance : maximum allowed edit distance.
        min_letters       : minimum letters before suggesting.
    """

    def __init__(self, max_edit_distance: int = 2, min_letters: int = 2):
        self._available = False
        try:
            from symspellpy import SymSpell, Verbosity
            import pkg_resources

            self._sym_spell = SymSpell(
                max_dictionary_edit_distance=max_edit_distance,
                prefix_length=7,
            )
            dict_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_dictionary_en_82_765.txt"
            )
            self._sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
            self._Verbosity   = Verbosity
            self._available   = True
            self._max_edit    = max_edit_distance
            self._min_letters = min_letters
            print("[WordSuggester] SymSpell loaded — autocorrect enabled.")
        except ImportError:
            print("[WordSuggester] symspellpy not found — autocorrect disabled.")
            print("                Install with: pip install symspellpy")
        except Exception as e:
            print(f"[WordSuggester] Error loading dictionary: {e}")

    def suggest(self, letters: list[str]) -> Optional[str]:
        """
        Return the best word suggestion for the given letter sequence.
        Returns None if no good match is found or too few letters.
        """
        if not self._available or len(letters) < self._min_letters:
            return None
        input_word  = "".join(letters).lower()
        suggestions = self._sym_spell.lookup(
            input_word,
            self._Verbosity.CLOSEST,
            max_edit_distance=self._max_edit,
        )
        if suggestions:
            return suggestions[0].term.upper()
        return None


# ---------------------------------------------------------------------------
# Text-to-speech engine  (pyttsx3, background thread)
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    Text-to-speech using a subprocess per word.
    Avoids the Windows SAPI5 singleton lockup in pyttsx3.
    """

    def __init__(self, rate: int = 150, voice: str = "female"):
        self._rate     = rate
        self._voice    = voice
        self._queue:   deque = deque()
        self._lock     = threading.Lock()
        self._speaking = False
        self._available = True
        self._thread   = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f"[TTS] Ready — rate={rate} wpm.")

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        with self._lock:
            self._queue.append(text.strip())

    def _worker(self) -> None:
        while True:
            text = None
            with self._lock:
                if self._queue:
                    text = self._queue.popleft()
            if text:
                self._speaking = True
                try:
                    # Each word runs in a completely fresh Python process
                    # This bypasses the pyttsx3 Windows singleton problem
                    import subprocess
                    script = (
                        f"import pyttsx3; "
                        f"e = pyttsx3.init(); "
                        f"e.setProperty('rate', {self._rate}); "
                        f"e.say('{text}'); "
                        f"e.runAndWait()"
                    )
                    subprocess.run(
                        ["python", "-c", script],
                        timeout=15,
                        creationflags=0x08000000  # CREATE_NO_WINDOW on Windows
                    )
                except Exception as e:
                    print(f"[TTS] Error: {e}")
                finally:
                    self._speaking = False
            else:
                time.sleep(0.05)

    @property
    def is_speaking(self) -> bool:
        return self._speaking


# ---------------------------------------------------------------------------
# HybridPredictor
# ---------------------------------------------------------------------------

class HybridPredictor:
    """Routes each frame to MLP (static A-Y) or LSTM (motion J,Z)."""

    MOVEMENT_THRESHOLD = 0.0005
    MIN_MOTION_FRAMES  = 8
    COOLDOWN_FRAMES    = 20

    def __init__(
        self,
        mlp_model,
        mlp_scaler,
        mlp_classes,
        lstm_model,
        lstm_classes,
        seq_len:           int   = 30,
        smoothing:         int   = 10,
        min_conf:          float = 0.60,
        movement_thr:      float = MOVEMENT_THRESHOLD,
        min_motion_frames: int   = MIN_MOTION_FRAMES,
        cooldown_frames:   int   = COOLDOWN_FRAMES,
    ):
        self.mlp_model    = mlp_model
        self.mlp_scaler   = mlp_scaler
        self.mlp_classes  = mlp_classes
        self.lstm_model   = lstm_model
        self.lstm_classes = lstm_classes
        self.seq_len      = seq_len
        self.min_conf     = min_conf

        self.movement_thr      = movement_thr
        self.min_motion_frames = min_motion_frames
        self.cooldown_frames   = cooldown_frames

        self._buffer:      deque = deque(maxlen=seq_len)
        self._mlp_history: deque = deque(maxlen=smoothing)
        self._motion_streak  = 0
        self._cooldown_count = 0
        self._lstm_active    = False
        self.last_timing_ms: dict[str, float | str] = {}

    @property
    def mode(self) -> str:
        return "LSTM" if self._lstm_active else "MLP"

    def predict(self, hand: HandLandmarks) -> tuple[Optional[str], float, str]:
        started = time.perf_counter()
        self.last_timing_ms = {}
        norm = normalize_flat(hand.flat)
        normalization_ms = (time.perf_counter() - started) * 1000
        routing_started = time.perf_counter()
        self._buffer.append(norm.copy())

        movement = 0.0
        if len(self._buffer) >= 10:
            buf_array = np.stack(list(self._buffer)[-10:], axis=0)
            movement  = float(buf_array.var(axis=0).mean())

        self._update_motion_state(movement)
        routing_ms = (time.perf_counter() - routing_started) * 1000

        if self._lstm_active and len(self._buffer) >= self.seq_len:
            letter, conf = self._predict_lstm()
        else:
            letter, conf = self._predict_mlp(norm)

        self.last_timing_ms.update({
            "normalization_ms": normalization_ms,
            "routing_ms": routing_ms,
            "mode": self.mode,
        })
        return letter, conf, self.mode

    def reset(self) -> None:
        self._buffer.clear()
        self._mlp_history.clear()
        self._motion_streak  = 0
        self._cooldown_count = 0
        self._lstm_active    = False

    def _update_motion_state(self, movement: float) -> None:
        if movement >= self.movement_thr:
            self._motion_streak  += 1
            self._cooldown_count  = 0
            if self._motion_streak >= self.min_motion_frames:
                self._lstm_active = True
        else:
            self._cooldown_count += 1
            self._motion_streak   = 0
            if self._cooldown_count >= self.cooldown_frames:
                self._lstm_active    = False
                self._cooldown_count = 0

    def _predict_mlp(self, norm_flat: np.ndarray) -> tuple[Optional[str], float]:
        preprocessing_started = time.perf_counter()
        scaled = self.mlp_scaler.transform(norm_flat.reshape(1, -1))
        tensor = torch.tensor(scaled, dtype=torch.float32)
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000
        inference_started = time.perf_counter()
        with torch.no_grad():
            probs = torch.softmax(self.mlp_model(tensor), dim=1).squeeze()
        inference_ms = (time.perf_counter() - inference_started) * 1000
        self.last_timing_ms.update({
            "mlp_preprocessing_ms": preprocessing_ms,
            "mlp_inference_ms": inference_ms,
        })
        conf   = probs.max().item()
        letter = str(self.mlp_classes[probs.argmax().item()])
        self._mlp_history.append(letter)
        if conf >= self.min_conf:
            return Counter(self._mlp_history).most_common(1)[0][0], conf
        return None, conf

    def _predict_lstm(self) -> tuple[Optional[str], float]:
        preprocessing_started = time.perf_counter()
        norm_seq = np.stack(list(self._buffer), axis=0)
        feat_seq = add_velocity_features(norm_seq)
        tensor   = torch.tensor(feat_seq, dtype=torch.float32).unsqueeze(0)
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000
        inference_started = time.perf_counter()
        with torch.no_grad():
            probs = torch.softmax(self.lstm_model(tensor), dim=1).squeeze()
        inference_ms = (time.perf_counter() - inference_started) * 1000
        self.last_timing_ms.update({
            "lstm_preprocessing_ms": preprocessing_ms,
            "lstm_inference_ms": inference_ms,
        })
        conf   = probs.max().item()
        letter = str(self.lstm_classes[probs.argmax().item()])
        if letter == "STATIC":
            return None, conf
        if conf >= self.min_conf:
            return letter, conf
        return None, conf


# ---------------------------------------------------------------------------
# UI Overlay
# ---------------------------------------------------------------------------

class InferenceOverlay:
    """
    Renders the full UI overlay.

    Layout:
        ┌──────────────────────────────────────────┐
        │ [Letter] [Conf] [Mode]   [Hand]  [FPS]   │  top bar
        ├──────────────────────────────────────────┤
        │              camera feed                 │
        ├──────────────────────────────────────────┤
        │  Letters : H · E · L · L · O            │
        │  Word    : HELLO          [ENTER]        │
        │  Sentence: HI WORLD ...                  │
        │  [ENTER] confirm | [BKSP] del | [SPC] clr│
        └──────────────────────────────────────────┘
    """

    GREEN  = (0,   220,  80)
    RED    = (30,   30, 220)
    YELLOW = (0,   210, 255)
    CYAN   = (200, 200,   0)
    WHITE  = (255, 255, 255)
    DARK   = (20,   20,  20)
    GRAY   = (130, 130, 130)
    ORANGE = (0,   160, 255)
    PURPLE = (200,  80, 200)

    FONT      = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

    def draw(
        self,
        frame: np.ndarray,
        *,
        letter:         Optional[str],
        confidence:     float,
        hand_detected:  bool,
        mode:           str,
        fps:            float,
        letter_history: list[str],
        suggestion:     Optional[str],
        sentence:       list[str],
        tts_speaking:   bool,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        out  = frame.copy()

        # ── Top bar ───────────────────────────────────────────────────────────
        panel = out.copy()
        cv2.rectangle(panel, (0, 0), (w, 95), self.DARK, -1)
        cv2.addWeighted(panel, 0.60, out, 0.40, 0, out)

        if letter and hand_detected:
            lc, dl = self.GREEN, letter
        elif hand_detected:
            lc, dl = self.YELLOW, "?"
        else:
            lc, dl = self.RED, "-"
        cv2.putText(out, dl, (18, 80), self.FONT_BOLD, 2.8, lc, 3, cv2.LINE_AA)

        if hand_detected:
            bx, by, bw, bh = 95, 18, 190, 14
            fill  = int(bw * max(0.0, min(confidence, 1.0)))
            color = (self.GREEN if confidence >= 0.85
                     else self.YELLOW if confidence >= 0.60 else self.RED)
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), self.GRAY, -1)
            cv2.rectangle(out, (bx, by), (bx + fill, by + bh), color, -1)
            cv2.putText(out, f"Conf: {confidence*100:.0f}%",
                        (bx, by - 3), self.FONT, 0.42, self.WHITE, 1, cv2.LINE_AA)
            mc = self.ORANGE if mode == "LSTM" else self.CYAN
            cv2.putText(out, f"[{mode}]", (95, 62),
                        self.FONT_BOLD, 0.58, mc, 1, cv2.LINE_AA)
            cv2.putText(out,
                        "motion (J/Z)" if mode == "LSTM" else "static (A-Y)",
                        (95, 78), self.FONT, 0.38, self.GRAY, 1, cv2.LINE_AA)

        cv2.putText(out, "HAND OK" if hand_detected else "NO HAND",
                    (w - 160, 30), self.FONT, 0.68,
                    self.GREEN if hand_detected else self.RED, 2, cv2.LINE_AA)
        cv2.putText(out, f"FPS: {fps:.0f}",
                    (w - 110, 55), self.FONT, 0.50, self.GRAY, 1, cv2.LINE_AA)
        if tts_speaking:
            cv2.putText(out, "SPEAKING...", (w - 155, 78),
                        self.FONT, 0.46, self.PURPLE, 1, cv2.LINE_AA)

        # ── Bottom panel ──────────────────────────────────────────────────────
        bot = out.copy()
        cv2.rectangle(bot, (0, h - 118), (w, h), self.DARK, -1)
        cv2.addWeighted(bot, 0.62, out, 0.38, 0, out)
        cv2.line(out, (0, h - 118), (w, h - 118), self.GRAY, 1)

        letters_y = h - 95
        word_y    = h - 62
        sent_y    = h - 32
        inst_y    = h - 8

        # Letter history
        if letter_history:
            cv2.putText(out, "Letters:", (8, letters_y),
                        self.FONT, 0.46, self.GRAY, 1, cv2.LINE_AA)
            cv2.putText(out, " · ".join(letter_history[-14:]), (80, letters_y),
                        self.FONT_BOLD, 0.56, self.WHITE, 1, cv2.LINE_AA)
        else:
            cv2.putText(out, "Letters: (spell a word...)", (8, letters_y),
                        self.FONT, 0.46, self.GRAY, 1, cv2.LINE_AA)

        # Word suggestion
        if suggestion:
            cv2.putText(out, "Word:", (8, word_y),
                        self.FONT, 0.46, self.GRAY, 1, cv2.LINE_AA)
            cv2.putText(out, suggestion, (70, word_y),
                        self.FONT_BOLD, 0.82, self.GREEN, 2, cv2.LINE_AA)
            cv2.putText(out, "[ENTER to confirm]", (w - 200, word_y),
                        self.FONT, 0.38, self.YELLOW, 1, cv2.LINE_AA)
        elif letter_history:
            cv2.putText(out, "Word:  (no suggestion)", (8, word_y),
                        self.FONT, 0.46, self.GRAY, 1, cv2.LINE_AA)

        # Sentence
        if sentence:
            cv2.putText(out, "Sentence:", (8, sent_y),
                        self.FONT, 0.44, self.GRAY, 1, cv2.LINE_AA)
            cv2.putText(out, " ".join(sentence[-6:]), (105, sent_y),
                        self.FONT_BOLD, 0.56, self.PURPLE, 1, cv2.LINE_AA)

        # Instructions
        cv2.putText(out,
            "[ENTER] confirm  |  [R] read sentence  |  "
            "[BKSP] del  |  [SPC] clear  |  [Q] quit",
            (8, inst_y), self.FONT, 0.35, self.GRAY, 1, cv2.LINE_AA)

        return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    mlp_path:   Path,
    lstm_path:  Path,
    camera_idx: int = 0,
    tts_rate:   int = 150,
) -> None:
    print("\n=== ASL Sign Language Translator — A to Z ===\n")

    mlp_model,  mlp_scaler,  mlp_classes = load_mlp(mlp_path)
    lstm_model, lstm_classes, seq_len    = load_lstm(lstm_path)

    predictor = HybridPredictor(
        mlp_model         = mlp_model,
        mlp_scaler        = mlp_scaler,
        mlp_classes       = mlp_classes,
        lstm_model        = lstm_model,
        lstm_classes      = lstm_classes,
        seq_len           = seq_len,
        smoothing         = 10,
        min_conf          = 0.60,
        movement_thr      = 0.0005,
        min_motion_frames = 8,
        cooldown_frames   = 20,
    )

    suggester = WordSuggester(max_edit_distance=2, min_letters=2)
    tts       = TTSEngine(rate=tts_rate, voice="female")
    overlay   = InferenceOverlay()

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_idx}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,           30)

    print("Controls:")
    print("  [ENTER]     — Confirm word (speak + add to sentence)")
    print("  [BACKSPACE] — Delete last letter")
    print("  [SPACE]     — Discard current word")
    print("  [Q] / [ESC] — Quit\n")

    letter_history:    list[str]     = []
    sentence:          list[str]     = []
    last_confirmed:    Optional[str] = None
    last_confirm_time: float         = 0.0
    CONFIRM_INTERVAL                 = 1.5

    fps_counter, fps_timer, fps_display = 0, time.monotonic(), 0.0

    with HandTracker(
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
    ) as tracker:

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame              = cv2.flip(frame, 1)
            annotated, hand    = tracker.process_frame(frame)
            letter, conf, mode = None, 0.0, "MLP"

            if hand is not None:
                try:
                    letter, conf, mode = predictor.predict(hand)
                except Exception as exc:
                    print(f"[WARN] {exc}")

                now = time.monotonic()
                if (letter is not None
                        and letter != last_confirmed
                        and now - last_confirm_time >= CONFIRM_INTERVAL):
                    letter_history.append(letter)
                    last_confirmed    = letter
                    last_confirm_time = now
                    print(f"  + {letter}  [{mode}]  {conf*100:.0f}%")
            else:
                predictor.reset()
                last_confirmed = None

            suggestion = suggester.suggest(letter_history)

            fps_counter += 1
            elapsed = time.monotonic() - fps_timer
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_timer   = time.monotonic()

            display = overlay.draw(
                annotated,
                letter         = letter,
                confidence     = conf,
                hand_detected  = (hand is not None),
                mode           = mode,
                fps            = fps_display,
                letter_history = letter_history,
                suggestion     = suggestion,
                sentence       = sentence,
                tts_speaking   = tts.is_speaking,
            )

            cv2.imshow("ASL Translator — A to Z", display)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):         # Q / ESC
                break

            elif key == 13:                              # ENTER — confirm word
                word = suggestion or "".join(letter_history)
                if word:
                    sentence.append(word)
                    tts.speak(word)
                    print(f"  Confirmed: '{word}'  |  Sentence: {' '.join(sentence)}")
                    letter_history.clear()
                    last_confirmed = None
            
            elif key == ord("r") or key == ord("R"):   # R — read full sentence
                if sentence:
                    full = " ".join(sentence)
                    tts.speak(full)
                    print(f"  [Reading sentence]: {full}")
                else:
                    print("  [Sentence is empty]")

            elif key == 8:                               # BACKSPACE
                if letter_history:
                    removed = letter_history.pop()
                    last_confirmed = letter_history[-1] if letter_history else None
                    print(f"  [Deleted: {removed}]")

            elif key == ord(" "):                        # SPACE — clear word
                letter_history.clear()
                last_confirmed = None
                print("  [Word cleared]")

    cap.release()
    cv2.destroyAllWindows()
    print("\n=== Session ended ===")
    if sentence:
        print(f"Full sentence: {' '.join(sentence)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time ASL translator A-Z with autocorrect and TTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mlp",      type=Path, default=Path(__file__).resolve().parent.parent / "models" / "mlp.pt",
                        help="Phase 1 MLP checkpoint.")
    parser.add_argument("--lstm",     type=Path, default=Path(__file__).resolve().parent.parent / "models" / "lstm_jz.pt",
                        help="Phase 2 LSTM checkpoint.")
    parser.add_argument("--camera",   type=int,  default=0,
                        help="Camera device index.")
    parser.add_argument("--tts_rate", type=int,  default=150,
                        help="TTS speech rate in words per minute.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        mlp_path   = args.mlp,
        lstm_path  = args.lstm,
        camera_idx = args.camera,
        tts_rate   = args.tts_rate,
    )
