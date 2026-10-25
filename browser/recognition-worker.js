import * as ort from 'onnxruntime-web/wasm';
import {FilesetResolver, HandLandmarker} from '@mediapipe/tasks-vision';

const asset = path => new URL(path, self.location.href).href;
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = asset('./vendor/onnx/');
let metadata, staticModel, dynamicModel, tracker, initialized;
let trackerMode = '', timestamp = 0;
let buffer = [], history = [], motionStreak = 0, cooldown = 0, dynamic = false;
let visionFiles;

async function initialize() {
  if (!initialized) initialized = (async () => {
    const response = await fetch(asset('./browser-models/metadata.json'));
    if (!response.ok) throw new Error('Could not download the recognition model settings.');
    metadata = await response.json();
    [staticModel, dynamicModel, visionFiles] = await Promise.all([
      ort.InferenceSession.create(asset('./browser-models/static.onnx'), {executionProviders: ['wasm']}),
      ort.InferenceSession.create(asset('./browser-models/dynamic.onnx'), {executionProviders: ['wasm']}),
      FilesetResolver.forVisionTasks(asset('./vendor/vision'))
    ]);
    await setTracker('VIDEO', true);
  })().catch(error => { initialized = null; throw error; });
  await initialized;
}

async function setTracker(mode, fresh = false) {
  if (tracker && (fresh || trackerMode !== mode)) {
    tracker.close(); tracker = null;
  }
  if (!tracker) tracker = await HandLandmarker.createFromOptions(visionFiles, {
    baseOptions: {modelAssetPath: asset('./browser-models/hand_landmarker.task'), delegate: 'CPU'},
    runningMode: mode, numHands: 1, minHandDetectionConfidence: 0.75,
    minHandPresenceConfidence: 0.75, minTrackingConfidence: 0.75
  });
  trackerMode = mode;
}

function reset() {
  buffer = []; history = []; motionStreak = 0; cooldown = 0; dynamic = false;
}

function normalize(landmarks) {
  const flat = Float32Array.from(landmarks.flatMap(p => [p.x, p.y, p.z]));
  const translated = Float32Array.from(flat, (v, i) => v - flat[i % 3]);
  const scale = Math.hypot(...translated.slice(27, 30));
  if (scale < 1e-6) throw new Error('Keep your whole hand visible in the frame.');
  return Float32Array.from(translated, v => v / scale);
}

function velocities(sequence) {
  const features = new Float32Array(sequence.length * 126);
  sequence.forEach((frame, t) => {
    features.set(frame, t * 126);
    if (t) for (let i = 0; i < 63; i++) features[t * 126 + 63 + i] = frame[i] - sequence[t - 1][i];
  });
  return features;
}

async function logits(model, input, shape) {
  const tensor = new ort.Tensor('float32', input, shape);
  const result = await model.run({input: tensor});
  const values = Array.from(result.logits.data);
  tensor.dispose(); result.logits.dispose();
  return values;
}

function choice(values, labels) {
  const max = Math.max(...values);
  const exps = values.map(v => Math.exp(v - max));
  const confidence = 1 / exps.reduce((sum, v) => sum + v, 0);
  const label = labels[values.indexOf(max)];
  return {letter: label !== 'STATIC' && confidence >= metadata.confidence_threshold ? label : null,
    confidence, rawLabel: label};
}

async function predictStatic(norm, smooth = false) {
  // StandardScaler performs the subtraction and division in float32 in Python.
  const scaled = Float32Array.from(norm, (v, i) => Math.fround(v - metadata.scaler_mean[i]) / metadata.scaler_scale[i]);
  const result = choice(await logits(staticModel, scaled, [1, 63]), metadata.static_letters);
  if (smooth) {
    history.push(result.rawLabel);
    if (history.length > metadata.smoothing) history.shift();
    const counts = new Map();
    history.forEach(label => counts.set(label, (counts.get(label) || 0) + 1));
    if (result.letter) result.letter = [...counts].sort((a, b) => b[1] - a[1])[0][0];
  }
  return result;
}

async function predictDynamic(sequence) {
  return choice(await logits(dynamicModel, velocities(sequence), [1, metadata.sequence_length, 126]), metadata.lstm_classes);
}

function draw(bitmap, {mirror = false, pad = false, margin = 1} = {}) {
  const w = bitmap.width, h = bitmap.height;
  if (w * h > 20_000_000) throw new Error('Use an image smaller than 20 megapixels.');
  const width = pad ? Math.max(w, Math.round(h * 4 / 3)) : w;
  const height = pad ? Math.max(h, Math.round(w * 3 / 4)) : h;
  const canvas = new OffscreenCanvas(Math.round(width * margin), Math.round(height * margin));
  const context = canvas.getContext('2d');
  context.fillStyle = '#fff'; context.fillRect(0, 0, canvas.width, canvas.height);
  if (mirror) { context.translate(canvas.width, 0); context.scale(-1, 1); }
  context.drawImage(bitmap, Math.floor((canvas.width - w) / 2), Math.floor((canvas.height - h) / 2));
  return canvas;
}

function detect(canvas, image = false) {
  timestamp = Math.max(timestamp + 1, performance.now());
  const result = image ? tracker.detect(canvas) : tracker.detectForVideo(canvas, timestamp);
  return result.landmarks[0] || null;
}

async function live(blob) {
  await setTracker('VIDEO');
  const started = performance.now();
  const bitmap = await createImageBitmap(blob);
  let landmarks;
  try { landmarks = detect(draw(bitmap, {mirror: true})); } finally { bitmap.close(); }
  const detectionMs = performance.now() - started;
  if (!landmarks) {
    reset();
    return {hand_detected: false, letter: null, confidence: 0, timings_ms: {pipeline: detectionMs, mediapipe: detectionMs}};
  }
  const norm = normalize(landmarks);
  buffer.push(norm);
  if (buffer.length > metadata.sequence_length) buffer.shift();
  let movement = 0;
  if (buffer.length >= 10) {
    const recent = buffer.slice(-10);
    for (let i = 0; i < 63; i++) {
      const mean = recent.reduce((sum, row) => sum + row[i], 0) / 10;
      movement += recent.reduce((sum, row) => sum + (row[i] - mean) ** 2, 0) / 10 / 63;
    }
  }
  if (movement >= metadata.motion_threshold) {
    motionStreak++; cooldown = 0;
    if (motionStreak >= metadata.min_motion_frames) dynamic = true;
  } else {
    cooldown++; motionStreak = 0;
    if (cooldown >= metadata.cooldown_frames) { dynamic = false; cooldown = 0; }
  }
  const useDynamic = dynamic && buffer.length >= metadata.sequence_length;
  const result = useDynamic ? await predictDynamic(buffer) : await predictStatic(norm, true);
  const inferenceMs = performance.now() - started - detectionMs;
  return {hand_detected: true, letter: result.letter, confidence: result.confidence,
    landmarks: landmarks.map(p => [p.x, p.y, p.z]), mode: useDynamic ? 'LSTM' : 'MLP',
    timings_ms: {pipeline: performance.now() - started, mediapipe: detectionMs,
      [useDynamic ? 'lstm_inference_ms' : 'mlp_inference_ms']: inferenceMs}};
}

async function handle(type, payload) {
  if (type === 'reset') { reset(); return {}; }
  await initialize();
  if (type === 'status') return {ready: true, static_letters: metadata.static_letters,
    dynamic_letters: metadata.lstm_classes.filter(x => x !== 'STATIC'),
    sequence_length: metadata.sequence_length, source: 'Browser', backend: 'ONNX WebAssembly'};
  if (type === 'recognize') return live(payload);
  if (type === 'test-image') {
    reset(); await setTracker('IMAGE', true);
    const bitmap = await createImageBitmap(payload);
    let landmarks;
    try {
      landmarks = detect(draw(bitmap, {mirror: true, pad: true}), true);
      if (!landmarks) landmarks = detect(draw(bitmap, {mirror: true, pad: true, margin: 1.35}), true);
    } finally { bitmap.close(); }
    if (!landmarks) throw new Error('No hand was detected. Use a clear photo with the entire hand visible.');
    const result = await predictStatic(normalize(landmarks));
    return {predicted: result.letter, confidence: result.confidence, media_type: 'static', frames_processed: 1};
  }
  if (type === 'video-start') { reset(); await setTracker('VIDEO', true); return {}; }
  if (type === 'video-frame') {
    try {
      const landmarks = detect(draw(payload));
      return landmarks ? Array.from(normalize(landmarks)) : null;
    } finally { payload.close(); }
  }
  if (type === 'video-finish') {
    if (payload.length !== metadata.sequence_length) throw new Error('The video did not provide a complete gesture sequence.');
    const result = await predictDynamic(payload);
    reset();
    return {predicted: result.letter, confidence: result.confidence, media_type: 'dynamic', frames_processed: payload.length};
  }
  // Used by the reproducible conversion check, with known Python model outputs.
  if (type === 'model-parity') return logits(payload.model === 'static' ? staticModel : dynamicModel,
    Float32Array.from(payload.input), payload.shape);
  throw new Error('Unknown recognition operation.');
}

let queue = Promise.resolve();
self.onmessage = ({data: {id, type, payload}}) => {
  queue = queue.then(async () => {
    try { self.postMessage({id, result: await handle(type, payload)}); }
    catch (error) { self.postMessage({id, error: error.message || String(error)}); }
  });
};
