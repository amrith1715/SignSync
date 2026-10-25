import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const publicRoot = path.join(root, 'vercel-dist');
const mime = {'.html':'text/html', '.js':'text/javascript', '.mjs':'text/javascript', '.wasm':'application/wasm',
  '.json':'application/json', '.css':'text/css', '.png':'image/png', '.jpg':'image/jpeg', '.mp4':'video/mp4'};
const server = http.createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    const target = path.resolve(publicRoot, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!target.startsWith(publicRoot + path.sep) || req.method !== 'GET') { res.writeHead(404).end(); return; }
    const data = await fs.readFile(target);
    res.setHeader('Content-Type', mime[path.extname(target)] || 'application/octet-stream');
    if (req.headers.range) {
      const [, startText, endText] = /bytes=(\d+)-(\d*)/.exec(req.headers.range) || [];
      const start = Number(startText), end = endText ? Number(endText) : data.length - 1;
      res.writeHead(206, {'Content-Range':`bytes ${start}-${end}/${data.length}`, 'Accept-Ranges':'bytes', 'Content-Length':end-start+1});
      res.end(data.subarray(start, end + 1));
    } else res.writeHead(200, {'Content-Length':data.length, 'Accept-Ranges':'bytes'}).end(data);
  } catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const base = process.env.CHECK_URL || `http://127.0.0.1:${server.address().port}`;
const report = {url: base, apiRequests: [], errors: []};
const browser = await chromium.launch({channel: 'msedge', headless: true,
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream']});
try {
  const context = await browser.newContext({permissions:['camera'], viewport:{width:1280,height:900}});
  const page = await context.newPage();
  page.on('pageerror', error => report.errors.push(error.message));
  page.on('console', msg => { if (msg.type() === 'error') console.log('BROWSER:', msg.text().slice(0, 350)); });
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) report.apiRequests.push(request.url()); });
  await page.goto(base + '/#testing');
  await page.waitForFunction(() => state.recognitionReady || document.querySelector('#modelPanel').style.display === 'block', null, {timeout:120000});
  const ready = await page.evaluate(() => ({ready:state.recognitionReady, error:document.querySelector('#projectModelError').textContent}));
  console.log('READY', ready);
  assert.equal(ready.ready, true, ready.error);

  const fixtures = JSON.parse(await fs.readFile(path.join(root, 'reports/browser-model-fixtures.json'), 'utf8'));
  report.parity = await page.evaluate(async fixtures => {
    const worker = new Worker('/assets/browser-worker.js');
    let id = 0;
    const results = [];
    try {
      for (const fixture of fixtures) {
        const values = await new Promise((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error('Parity worker timed out')), 60000);
          worker.onmessage = ({data}) => { clearTimeout(timer); data.error ? reject(new Error(data.error)) : resolve(data.result); };
          worker.onerror = event => { clearTimeout(timer); reject(new Error(event.message)); };
          worker.postMessage({id: ++id, type:'model-parity', payload:fixture});
        });
        const maxError = Math.max(...values.map((v, i) => Math.abs(v - fixture.logits[i])));
        results.push({model:fixture.model,maxError, sameClass:values.indexOf(Math.max(...values)) === fixture.logits.indexOf(Math.max(...fixture.logits))});
      }
    } finally { worker.terminate(); }
    return results;
  }, fixtures);
  assert(report.parity.every(item => item.maxError < 0.0001 && item.sameClass));
  console.log('PARITY', report.parity);

  const letters = [...'ABCDEFGHIKLMNOPQRSTUVWXY'];
  await page.locator('#staticTestUpload').setInputFiles(letters.map(letter => path.join(root, `assets/asl/static/${letter}.png`)));
  await page.getByRole('button', {name:'Predict all', exact:true}).click();
  await page.waitForFunction(() => document.querySelector('#mediaTestStatus').textContent.startsWith('Finished'), null, {timeout:120000});
  report.images = await page.locator('.media-test-queue-item').allTextContents();
  console.log('IMAGES', report.images);
  report.staticCorrect = report.images.filter((text, i) => text.includes(`Prediction: ${letters[i]} ·`)).length;
  assert(report.images.every(text => text.includes('Prediction:')));
  for (const letter of letters) await page.getByRole('button', {name:'Remove', exact:true}).first().click();

  await page.locator('#dynamicTestUpload').setInputFiles(['J','Z'].map(letter => path.join(root, `dynamic_sign_videos/${letter}.mp4`)));
  await page.getByRole('button', {name:'Predict all', exact:true}).click();
  await page.waitForFunction(() => document.querySelector('#mediaTestStatus').textContent.startsWith('Finished'), null, {timeout:150000});
  report.videos = await page.locator('.media-test-queue-item').allTextContents();
  report.videoSummary = await page.locator('#mediaTestStatus').textContent();
  console.log('VIDEOS', report.videos);
  // Illustrations are not accuracy benchmarks, but must reach the inference path.
  assert.equal(report.videoSummary, 'Finished 2 of 2: 2 succeeded.');

  const invalid = await page.evaluate(async () => {
    const file = new Blob(['invalid image'], {type:'image/png'});
    const response = await SignSyncBrowser.request('test-image', {body:file});
    return {status:response.status, result:await response.json()};
  });
  assert.equal(invalid.status, 400);
  report.invalidFile = invalid;
  await page.getByRole('link', {name:'Recognition', exact:true}).click();
  await page.locator('#liveToggleBtn').click();
  await page.waitForFunction(() => state.cameraActive, null, {timeout:15000});
  await page.waitForFunction(() => state.projectInferenceCount > 0, null, {timeout:30000});
  report.camera = await page.evaluate(() => ({active:state.cameraActive, error:state.backendErrorShown}));
  assert.equal(report.camera.error, false);
  await page.locator('#liveToggleBtn').click();
  assert.equal(await page.evaluate(() => state.cameraActive), false);
  assert.deepEqual(report.apiRequests, []);
  assert.deepEqual(report.errors, []);
  await page.screenshot({path:path.join(root,'reports/browser-check.png')});
  await fs.writeFile(path.join(root,'reports/browser-check.json'), JSON.stringify(report,null,2));
  console.log('PASS: browser-only model parity, multi-image/video inference, invalid input, and camera start/stop.');
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
