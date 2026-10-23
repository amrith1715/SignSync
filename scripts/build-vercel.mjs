import { cp, mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'vercel-dist');
const apiBase = (process.env.SIGNSYNC_API_BASE || '').trim().replace(/\/+$/, '');

if (apiBase) {
  const url = new URL(apiBase);
  if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash || url.pathname !== '/') {
    throw new Error('SIGNSYNC_API_BASE must be an HTTPS origin, with no credentials, path, query, or fragment (example: https://api.example.com).');
  }
} else if (process.env.VERCEL) {
  console.warn('SIGNSYNC_API_BASE is unset. The frontend will deploy, but recognition needs a separately hosted HTTPS API. Set the variable and redeploy to enable recognition.');
}

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await cp(path.join(root, 'index.html'), path.join(output, 'index.html'));
await cp(path.join(root, 'assets'), path.join(output, 'assets'), { recursive: true });
await cp(path.join(root, 'label_mapping.json'), path.join(output, 'label_mapping.json'));
await cp(path.join(root, 'model.json'), path.join(output, 'model.json'));
await cp(path.join(root, 'group1-shard1of1.bin'), path.join(output, 'group1-shard1of1.bin'));
await cp(path.join(root, 'tfjs_model'), path.join(output, 'tfjs_model'), { recursive: true });

const configPath = path.join(output, 'assets', 'deployment-config.js');
const config = `// Generated at build time from SIGNSYNC_API_BASE.\nwindow.SIGNSYNC_API_BASE = ${JSON.stringify(apiBase)};\n`;
await writeFile(configPath, config, 'utf8');

console.log(`Prepared SignSync static frontend at ${path.relative(root, output)} (API: ${apiBase || 'local relative path'}).`);
