"""Export the existing trained models, and check ONNX against PyTorch.

Run with the project virtual environment after installing onnx and onnxruntime.
The original checkpoints are never modified. No retraining is performed.
"""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recognition.main_app import load_mlp, load_lstm

OUT = ROOT / "assets" / "browser-models"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sequence-data', type=Path, help='Optional existing (N,30,126) recorded feature sequences for parity validation.')
args = parser.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
mlp, scaler, static_classes = load_mlp(ROOT / "models/mlp.pt")
lstm, dynamic_classes, seq_len = load_lstm(ROOT / "models/lstm_jz.pt")
rng = np.random.default_rng(1729)
checks = {}
fixtures = []
for name, model, shape in [("static", mlp, (1, 63)), ("dynamic", lstm, (1, seq_len, 126))]:
    target = OUT / f"{name}.onnx"
    torch.onnx.export(model, torch.zeros(shape), str(target), input_names=["input"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(target)))
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    max_error = 0.0
    for index in range(24):
        sample = rng.normal(0, 0.25 if index % 2 else 1, shape).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(sample)).numpy()
        actual = session.run(None, {"input": sample})[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
        assert np.argmax(actual) == np.argmax(expected)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        if index < 3:
            fixtures.append({"model": name, "shape": shape, "input": sample.flatten().tolist(),
                             "logits": expected.flatten().tolist()})
    checks[name] = {"cases": 24, "max_absolute_logit_error": max_error,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

if args.sequence_data:
    recorded = np.load(args.sequence_data, allow_pickle=False)
    labels = np.load(args.sequence_data.with_name(args.sequence_data.stem + '_labels.npy'), allow_pickle=True)
    assert recorded.shape[1:] == (seq_len, 126)
    session = ort.InferenceSession(str(OUT / 'dynamic.onnx'), providers=['CPUExecutionProvider'])
    max_error = 0.0
    counts = {}
    for sample, label in zip(recorded, labels):
        sample = sample[None].astype(np.float32)
        with torch.no_grad():
            expected = lstm(torch.from_numpy(sample)).numpy()
        actual = session.run(None, {'input': sample})[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
        assert np.argmax(actual) == np.argmax(expected)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        label = str(label)
        if counts.get(label, 0) < 2:
            fixtures.append({'model':'dynamic', 'shape':list(sample.shape), 'input':sample.flatten().tolist(),
                             'logits':expected.flatten().tolist(), 'recorded_label':label})
        counts[label] = counts.get(label, 0) + 1
    checks['recorded_dynamic'] = {'cases':len(recorded), 'class_counts':counts,
                                  'max_absolute_logit_error':max_error,
                                  'purpose':'Conversion parity only; not held-out accuracy evaluation'}

metadata = {
    "static_letters": [str(x) for x in static_classes],
    "lstm_classes": [str(x) for x in dynamic_classes],
    "sequence_length": int(seq_len), "confidence_threshold": 0.60,
    "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
    "motion_threshold": 0.0005, "min_motion_frames": 8, "cooldown_frames": 20,
    "smoothing": 10,
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
shutil.copyfile(ROOT / "models/hand_landmarker.task", OUT / "hand_landmarker.task")
report = ROOT / "reports/browser-model-export.json"
report.write_text(json.dumps(checks, indent=2), encoding="utf-8")
(ROOT / "reports/browser-model-fixtures.json").write_text(json.dumps(fixtures), encoding="utf-8")
print(json.dumps(checks, indent=2))
