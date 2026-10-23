"""Supervised chart adaptation with replay; chart scores are training-set scores."""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from recognition_service import RecognitionRuntime

ROOT = Path(__file__).resolve().parent
DATA = Path(r'C:\Users\adwai\Downloads\sign-language-cv\data\processed\landmarks_norm_v2.csv')
torch.set_num_threads(2)
torch.manual_seed(42)
rng = np.random.default_rng(42)
runtime = RecognitionRuntime()
features, labels = [], []
original_predict = runtime.predictor._predict_mlp
def capture(feature):
    features.append(feature.copy())
    return original_predict(feature)
runtime.predictor._predict_mlp = capture
try:
    for path in sorted((ROOT / 'static_signs_images').glob('*.png')):
        runtime.test_image(path.read_bytes())
        labels.append(path.stem)
finally:
    runtime.close()
assert len(features) == len(labels) == 24
base_checkpoint = ROOT / 'models/mlp_before_static_adaptation.pt'
if not base_checkpoint.exists():
    base_checkpoint = ROOT / 'models/mlp.pt'
checkpoint = torch.load(base_checkpoint, map_location='cpu', weights_only=False)
model = runtime.predictor.mlp_model
model.load_state_dict(checkpoint['model_state_dict'])
scaler = checkpoint['scaler']
classes = list(checkpoint['classes'])
data = pd.read_csv(DATA)
X = data.drop(columns='label').to_numpy(dtype=np.float32)
y = np.array([classes.index(v) for v in data.label])
train, val = train_test_split(np.arange(len(y)), test_size=.25, random_state=42, stratify=y)
X = torch.tensor(scaler.transform(X), dtype=torch.float32)
y = torch.tensor(y)
reference = np.stack(features)
chart = torch.tensor(scaler.transform(reference), dtype=torch.float32)
chart_y = torch.tensor([classes.index(v) for v in labels])
def accuracy(inputs, targets):
    model.eval()
    with torch.no_grad():
        probs = model(inputs).softmax(1)
        return float((probs.argmax(1) == targets).float().mean()), probs
baseline, _ = accuracy(X[val], y[val])
optimizer = torch.optim.Adam(model.parameters(), lr=0.00015)
loss_fn = torch.nn.CrossEntropyLoss()
best = None
history = []
for epoch in range(1, 81):
    # Keep pretrained BatchNorm statistics; gradients still update MLP weights.
    model.eval()
    for _ in range(8):
        indices = rng.choice(train, size=256, replace=False)
        augmented = reference + rng.normal(0, .008, reference.shape).astype(np.float32)
        aug = torch.tensor(scaler.transform(augmented), dtype=torch.float32)
        optimizer.zero_grad()
        loss = loss_fn(model(X[indices]), y[indices]) + loss_fn(model(chart), chart_y) + .5 * loss_fn(model(aug), chart_y)
        loss.backward()
        optimizer.step()
    retained, _ = accuracy(X[val], y[val])
    chart_acc, probs = accuracy(chart, chart_y)
    minimum = float(probs[range(24), chart_y].min())
    history.append(dict(epoch=epoch, regression_accuracy=retained, chart_training_accuracy=chart_acc, minimum_chart_confidence=minimum))
    print(history[-1], flush=True)
    if retained >= baseline - .005 and chart_acc == 1 and minimum >= .85:
        best = copy.deepcopy(model.state_dict())
        break
report = dict(dataset=str(DATA), dataset_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
    source_rows=len(data), regression_rows=len(val), baseline_regression_accuracy=baseline,
    caveat='Chart images are adaptation/training examples. Regression split may overlap original checkpoint training; it is not an independent test.',
    history=history, accepted=best is not None)
if best is not None:
    checkpoint['model_state_dict'] = best
    checkpoint['static_reference_adaptation'] = report
    # Preserve historical metric, distinguish it from the current regression metric.
    checkpoint['original_test_acc'] = checkpoint.get('test_acc')
    checkpoint['test_acc'] = history[-1]['regression_accuracy']
    torch.save(checkpoint, ROOT / 'models/mlp_static_candidate.pt')
(ROOT / 'reports/static_adaptation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
if best is None:
    raise SystemExit('No candidate passed the acceptance gates; production weights unchanged.')
