"""
sequence_utils.py
-----------------
Preprocessing and I/O utilities for hand landmark SEQUENCES (Phase 2 — J & Z).

This module is the temporal counterpart of math_utils.py.
Where math_utils.py handles single frames (N, 63), this module handles
sequences of frames (N, seq_len, 63).

NORMALISATION PIPELINE FOR SEQUENCES
──────────────────────────────────────
Each sequence is a (seq_len, 63) array of RAW MediaPipe landmark coordinates.
The same two-step normalisation from math_utils is applied FRAME BY FRAME:

  Per-frame step 1 — Spatial normalisation (translation to origin):
      p'_i[t] = p_i[t] - p_0[t]      ∀i ∈ {0..20}, ∀t ∈ {0..seq_len-1}
      Wrist is moved to (0,0,0) in every frame independently.
      Makes the sequence invariant to hand position in the camera.

  Per-frame step 2 — Scale normalisation (division by palm length):
      p''_i[t] = p'_i[t] / d[t]      d[t] = ‖p'_9[t]‖₂
      Makes the sequence invariant to hand distance from the camera.

After per-frame normalisation the sequence still has shape (seq_len, 63).

VELOCITY FEATURES (optional but recommended)
─────────────────────────────────────────────
Velocity captures HOW the hand moves between frames, which is the key
discriminator for J vs Z:
  - J: smooth continuous arc of the pinky finger downward.
  - Z: two sharp direction changes (zig + zag).

  delta[t] = norm_seq[t+1] - norm_seq[t]     ∀t ∈ {0..seq_len-2}

We pad the first delta with zeros so the output keeps shape (seq_len, 63):

  velocity[0]   = zeros(63)            ← no previous frame
  velocity[t]   = norm_seq[t] - norm_seq[t-1]    ∀t ∈ {1..seq_len-1}

The final feature vector per frame (with velocity) is the concatenation:
  features[t] = [norm_seq[t], velocity[t]]   → shape (seq_len, 126)

Public functions:
    normalise_sequence(raw_seq)             — normalise a (seq_len, 63) array
    add_velocity_features(norm_seq)         — append velocity → (seq_len, 126)
    preprocess_sequence(raw_seq, velocity)  — full pipeline for one sequence
    load_raw_sequences(prefix)              — load .npy files from capture
    combine_sequence_files(prefixes, out)   — merge multi-person .npy files
    preprocess_all(raw_prefix, out, ...)    — full pipeline: load→norm→save
    verify_normalisation(sequences)         — sanity-check the processed data
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

# Import the per-frame normalisation from Phase 1 — no duplication
from .math_utils import normalize_landmarks, FEATURE_SIZE, NUM_LANDMARKS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEQ_FEATURE_SIZE          = FEATURE_SIZE           # 63  — position only
SEQ_FEATURE_SIZE_WITH_VEL = FEATURE_SIZE * 2       # 126 — position + velocity

WRIST_IDX    = 0
PALM_REF_IDX = 9

TARGET_LETTERS = ["J", "Z"]


# ---------------------------------------------------------------------------
# Core mathematics
# ---------------------------------------------------------------------------

def normalise_sequence(raw_seq: np.ndarray) -> np.ndarray:
    """
    Apply per-frame spatial and scale normalisation to a raw sequence.

    The same two-step pipeline from math_utils.normalize_landmarks is
    applied independently to every frame:
      1. Translate wrist to origin.
      2. Divide by palm length (wrist → middle-finger MCP distance).

    Args:
        raw_seq : float32 array of shape (seq_len, 63) — RAW MediaPipe
                  landmark coordinates captured by capture_sequences.py.

    Returns:
        float32 array of shape (seq_len, 63) — normalised sequence.
        Frames where the palm distance is too small (hand out of frame)
        are replaced by linear interpolation from neighbouring frames.

    Raises:
        ValueError : if raw_seq has unexpected shape.
    """
    if raw_seq.ndim != 2 or raw_seq.shape[1] != FEATURE_SIZE:
        raise ValueError(
            f"Expected shape (seq_len, {FEATURE_SIZE}), got {raw_seq.shape}."
        )

    seq_len     = raw_seq.shape[0]
    norm_seq    = np.zeros_like(raw_seq, dtype=np.float32)
    bad_frames  = []

    for t in range(seq_len):
        frame_flat = raw_seq[t]                            # (63,)
        try:
            norm_seq[t] = normalize_landmarks(
                frame_flat.reshape(NUM_LANDMARKS, 3)
            ).flatten()
        except ValueError:
            # Palm too small — mark for interpolation
            bad_frames.append(t)

    # ── Interpolate bad frames ────────────────────────────────────────────────
    # This handles brief moments when the hand partially leaves the frame
    for t in bad_frames:
        # Find nearest good neighbours
        prev_good = next((t - i for i in range(1, t + 1) if (t - i) not in bad_frames), None)
        next_good = next((t + i for i in range(1, seq_len - t) if (t + i) not in bad_frames), None)

        if prev_good is not None and next_good is not None:
            # Linear interpolation between neighbours
            alpha     = (t - prev_good) / (next_good - prev_good)
            norm_seq[t] = (1 - alpha) * norm_seq[prev_good] + alpha * norm_seq[next_good]
        elif prev_good is not None:
            norm_seq[t] = norm_seq[prev_good]
        elif next_good is not None:
            norm_seq[t] = norm_seq[next_good]
        else:
            # All frames bad — return zeros (sequence will be discarded later)
            norm_seq[t] = np.zeros(FEATURE_SIZE, dtype=np.float32)

    return norm_seq


def add_velocity_features(norm_seq: np.ndarray) -> np.ndarray:
    """
    Append per-frame velocity features to a normalised sequence.

    Velocity is the difference between consecutive normalised frames:
        velocity[0]   = zeros(63)                     (no prior frame)
        velocity[t]   = norm_seq[t] - norm_seq[t-1]   for t ≥ 1

    Velocity is the key discriminator for J vs Z:
    - J produces a smooth, monotonically increasing velocity in one direction.
    - Z produces two abrupt direction changes detectable as sign flips.

    Args:
        norm_seq : float32 array of shape (seq_len, 63) — normalised sequence.

    Returns:
        float32 array of shape (seq_len, 126) — [position | velocity] per frame.
    """
    if norm_seq.ndim != 2 or norm_seq.shape[1] != FEATURE_SIZE:
        raise ValueError(
            f"Expected shape (seq_len, {FEATURE_SIZE}), got {norm_seq.shape}."
        )

    seq_len  = norm_seq.shape[0]
    velocity = np.zeros_like(norm_seq, dtype=np.float32)   # (seq_len, 63)

    # velocity[0] stays zero (no prior frame available)
    velocity[1:] = norm_seq[1:] - norm_seq[:-1]            # (seq_len-1, 63)

    return np.concatenate([norm_seq, velocity], axis=1)    # (seq_len, 126)


def preprocess_sequence(
    raw_seq: np.ndarray,
    use_velocity: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single RAW sequence.

    Steps:
        1. Per-frame spatial + scale normalisation.
        2. (Optional) Append velocity features.

    Args:
        raw_seq      : float32 (seq_len, 63) — RAW landmarks from capture.
        use_velocity : if True, output shape is (seq_len, 126);
                       if False, output shape is (seq_len, 63).

    Returns:
        Preprocessed float32 array ready for the LSTM.
    """
    norm = normalise_sequence(raw_seq)
    if use_velocity:
        return add_velocity_features(norm)
    return norm


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_raw_sequences(
    prefix: Path | str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the two .npy files produced by capture_sequences.py.

    Args:
        prefix : Path prefix used when capturing, e.g.
                 "data/raw/sequences/sequences_p1".
                 Two files are expected:
                   <prefix>.npy         → (N, seq_len, 63) float32
                   <prefix>_labels.npy  → (N,) str

    Returns:
        (sequences, labels):
            sequences — float32 (N, seq_len, 63)
            labels    — str array (N,)  e.g. ['J','J','Z','J',...]
    """
    prefix = Path(prefix)
    seq_path    = Path(str(prefix) + ".npy")
    labels_path = Path(str(prefix) + "_labels.npy")

    if not seq_path.exists():
        raise FileNotFoundError(f"Sequences file not found: {seq_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    sequences = np.load(seq_path,    allow_pickle=False).astype(np.float32)
    labels    = np.load(labels_path, allow_pickle=True ).astype(str)

    if sequences.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Mismatch: {sequences.shape[0]} sequences but {labels.shape[0]} labels."
        )

    return sequences, labels


def combine_sequence_files(
    prefixes: list[Path | str],
    output_prefix: Path | str,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge multiple per-person .npy sequence files into one combined file.

    Equivalent to what build_dataset.py does for Phase 1 CSVs.
    All files must have the same seq_len and feature size.

    Args:
        prefixes      : list of path prefixes, one per person.
        output_prefix : prefix for the combined output files.
        verbose       : print progress and statistics.

    Returns:
        (all_sequences, all_labels):
            all_sequences — float32 (N_total, seq_len, 63)
            all_labels    — str (N_total,)
    """
    all_seqs:   list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for prefix in prefixes:
        seqs, lbls = load_raw_sequences(prefix)
        all_seqs.append(seqs)
        all_labels.append(lbls)
        if verbose:
            unique, counts = np.unique(lbls, return_counts=True)
            dist = {u: c for u, c in zip(unique, counts)}
            print(f"  Loaded {prefix}: {len(seqs)} sequences  {dist}")

    combined_seqs   = np.concatenate(all_seqs,   axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    # Save combined files
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(output_prefix) + ".npy",        combined_seqs)
    np.save(str(output_prefix) + "_labels.npy", combined_labels)

    if verbose:
        unique, counts = np.unique(combined_labels, return_counts=True)
        print(f"\nCombined: {combined_seqs.shape}  labels: {dict(zip(unique, counts))}")
        print(f"Saved to: {output_prefix.resolve()}")

    return combined_seqs, combined_labels


def save_processed_sequences(
    sequences: np.ndarray,
    labels: np.ndarray,
    output_prefix: Path | str,
    verbose: bool = True,
) -> None:
    """
    Save preprocessed (normalised) sequences to disk.

    Args:
        sequences     : float32 (N, seq_len, 63 or 126) — normalised.
        labels        : str (N,) — letter labels.
        output_prefix : path prefix for the two output .npy files.
        verbose       : print confirmation.
    """
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(output_prefix) + ".npy",        sequences.astype(np.float32))
    np.save(str(output_prefix) + "_labels.npy", labels.astype(object))

    if verbose:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"[sequence_utils] Saved {len(sequences)} sequences → {output_prefix}")
        print(f"  Shape  : {sequences.shape}")
        print(f"  Classes: {dict(zip(unique, counts))}")


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_all(
    raw_prefix:       Path | str,
    output_prefix:    Path | str,
    use_velocity:     bool = True,
    verbose:          bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline: load RAW → normalise → (velocity) → save.

    This is the sequence equivalent of math_utils.preprocess_csv().
    Run this once after combining all per-person files.

    Args:
        raw_prefix    : prefix of the combined raw .npy files.
        output_prefix : prefix for the processed output .npy files.
        use_velocity  : whether to append velocity features (recommended).
        verbose       : print progress and statistics.

    Returns:
        (processed_sequences, labels):
            processed_sequences — float32 (N, seq_len, 63 or 126)
            labels              — str (N,)
    """
    raw_seqs, labels = load_raw_sequences(raw_prefix)

    n_total   = len(raw_seqs)
    seq_len   = raw_seqs.shape[1]
    feat_size = SEQ_FEATURE_SIZE_WITH_VEL if use_velocity else SEQ_FEATURE_SIZE

    if verbose:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"[sequence_utils] Loaded {n_total} RAW sequences from '{raw_prefix}'")
        print(f"  Shape per sequence : ({seq_len}, {FEATURE_SIZE})")
        print(f"  Class distribution : {dict(zip(unique, counts))}")
        print(f"  Velocity features  : {'YES → output (seq_len, 126)' if use_velocity else 'NO → output (seq_len, 63)'}")
        print()

    processed_list: list[np.ndarray] = []
    labels_clean:   list[str]        = []
    n_dropped = 0

    for i, (raw_seq, label) in enumerate(zip(raw_seqs, labels)):
        try:
            proc = preprocess_sequence(raw_seq, use_velocity=use_velocity)
            processed_list.append(proc)
            labels_clean.append(label)
        except Exception as e:
            if verbose:
                print(f"  [!] Sequence {i} ({label}) dropped: {e}")
            n_dropped += 1

    processed = np.stack(processed_list, axis=0).astype(np.float32)
    labels_out = np.array(labels_clean, dtype=str)

    if verbose:
        if n_dropped > 0:
            print(f"\n[sequence_utils] {n_dropped} sequences dropped (corrupt landmarks).")
        print(f"\n[sequence_utils] Processed shape: {processed.shape}")

    save_processed_sequences(processed, labels_out, output_prefix, verbose=verbose)

    if verbose:
        verify_normalisation(processed, use_velocity=use_velocity)

    return processed, labels_out


# ---------------------------------------------------------------------------
# Verification / sanity check
# ---------------------------------------------------------------------------

def verify_normalisation(
    sequences: np.ndarray,
    use_velocity: bool = True,
    n_samples: int = 500,
) -> None:
    """
    Sanity-check that per-frame normalisation was applied correctly.

    Checks (on a random subsample of frames):
      1. Wrist (landmark 0) is at (0, 0, 0) in every frame.
      2. Palm norm (landmark 9) is exactly 1.0 in every frame.

    Args:
        sequences    : float32 (N, seq_len, 63 or 126) — processed sequences.
        use_velocity : True if the array has 126 features (pos + vel).
        n_samples    : number of sequences to check (for speed).
    """
    # Use only the position channels (first 63 features)
    pos = sequences[:n_samples, :, :FEATURE_SIZE]   # (n, seq_len, 63)

    # Flatten to (n * seq_len, 63) for vectorised checks
    flat = pos.reshape(-1, FEATURE_SIZE)             # (n * seq_len, 63)

    wrist_coords  = flat[:, :3]                      # x0, y0, z0
    palm_coords   = flat[:, PALM_REF_IDX*3 : PALM_REF_IDX*3+3]   # x9,y9,z9
    palm_norms    = np.linalg.norm(palm_coords, axis=1)

    wrist_max_abs = np.abs(wrist_coords).max()
    palm_mean     = palm_norms.mean()
    palm_std      = palm_norms.std()

    print("\n[sequence_utils] Normalisation verification:")
    print("=" * 52)
    print(f"  Wrist at origin   max|val| : {wrist_max_abs:.2e}  "
          f"{'✅' if wrist_max_abs < 1e-4 else '⚠️  check pipeline'}")
    print(f"  Palm norm         mean     : {palm_mean:.6f}  "
          f"{'✅' if abs(palm_mean - 1.0) < 1e-3 else '⚠️  check pipeline'}")
    print(f"  Palm norm         std      : {palm_std:.6f}  "
          f"{'✅' if palm_std < 1e-4 else '⚠️  check pipeline'}")

    if use_velocity:
        vel = sequences[:n_samples, :, FEATURE_SIZE:]    # (n, seq_len, 63)
        vel_frame0 = vel[:, 0, :]                        # should be all zeros
        vel0_max   = np.abs(vel_frame0).max()
        print(f"  Velocity frame 0  max|val| : {vel0_max:.2e}  "
              f"{'✅' if vel0_max < 1e-6 else '⚠️  first frame should be zero'}")

    print("=" * 52)


# ---------------------------------------------------------------------------
# CLI — run directly to preprocess sequences
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess raw J/Z sequences: normalise per-frame and "
            "optionally append velocity features."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw/sequences/sequences_combined"),
        help="Prefix of the combined raw .npy files (without extension).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/sequences_norm"),
        help="Prefix for the processed output .npy files.",
    )
    parser.add_argument(
        "--no_velocity",
        action="store_true",
        default=False,
        help="Disable velocity features (output shape: seq_len×63 instead of seq_len×126).",
    )
    parser.add_argument(
        "--combine",
        nargs="+",
        type=Path,
        default=None,
        metavar="PREFIX",
        help=(
            "If provided, combine these per-person prefixes into --raw before "
            "preprocessing. Example: --combine data/raw/sequences/sequences_p1 "
            "data/raw/sequences/sequences_p2"
        ),
    )
    args = parser.parse_args()

    # Optional: combine per-person files first
    if args.combine:
        print("Combining per-person sequence files...")
        combine_sequence_files(
            prefixes      = args.combine,
            output_prefix = args.raw,
            verbose       = True,
        )
        print()

    # Run the full preprocessing pipeline
    preprocess_all(
        raw_prefix    = args.raw,
        output_prefix = args.out,
        use_velocity  = not args.no_velocity,
        verbose       = True,
    )
