import {cp, mkdir} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export async function buildBrowser(destination) {
  await mkdir(path.join(destination, 'vendor/onnx'), {recursive: true});
  await cp(path.join(root, 'node_modules/@mediapipe/tasks-vision/wasm'), path.join(destination, 'vendor/vision'), {recursive: true});
  for (const file of ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs']) {
    await cp(path.join(root, 'node_modules/onnxruntime-web/dist', file), path.join(destination, 'vendor/onnx', file));
  }
  await build({entryPoints: [path.join(root, 'browser/recognition-worker.js')],
    outfile: path.join(destination, 'browser-worker.js'), bundle: true, format: 'iife',
    platform: 'browser', target: 'es2022', minify: true, legalComments: 'eof'});
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await buildBrowser(path.join(root, 'assets'));
}
