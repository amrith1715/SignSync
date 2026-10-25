import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pathlib
import urllib.request
from urllib.parse import urlparse, parse_qs

ROOT = str(pathlib.Path(__file__).parent.resolve())
PORT = int(os.environ.get('PORT', os.environ.get('SIGNSYNC_PORT', '8001')))
HOST = os.environ.get('HOST', '127.0.0.1')
ALLOWED_ORIGINS = {
    origin.strip().rstrip('/')
    for origin in os.environ.get('SIGNSYNC_ALLOWED_ORIGINS', '').split(',')
    if origin.strip()
}
ALLOWED_ORIGINS.update({f'http://127.0.0.1:{PORT}', f'http://localhost:{PORT}'})
COMMONS_ALPHABET_VIDEO = (
    'https://upload.wikimedia.org/wikipedia/commons/6/64/'
    'American_Sign_Language_demo_-_Travis_Dougherty.ogv'
)
LETTER_CLIPS = {'J': (26.19, 27.19), 'Z': (39.88, 40.97)}


def _fetch_commons_alphabet_video(url):
    if url != COMMONS_ALPHABET_VIDEO:
        raise ValueError('Only the reviewed Wikimedia Commons alphabet video is accepted by this reference route.')
    request = urllib.request.Request(url, headers={'User-Agent': 'SignSync/1.0 (local ASL reference testing)'})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            final = urlparse(response.geturl())
            if final.scheme != 'https' or final.hostname != 'upload.wikimedia.org':
                raise ValueError('The media source redirected outside Wikimedia Commons.')
            content_length = response.headers.get('Content-Length')
            if content_length and int(content_length) > 12 * 1024 * 1024:
                raise ValueError('The reference video exceeds the 12 MB download limit.')
            chunks = []
            size = 0
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > 12 * 1024 * 1024:
                    raise ValueError('The reference video exceeds the 12 MB download limit.')
                chunks.append(chunk)
            if not chunks:
                raise ValueError('The reference video download was empty.')
            return b''.join(chunks)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f'Could not retrieve the Commons reference video: {exc}') from exc

class CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        origin = self.headers.get('Origin', '').rstrip('/')
        if origin in ALLOWED_ORIGINS:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def _json(self, status, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == '/api/gallery':
            manifest = pathlib.Path(ROOT) / 'assets' / 'asl' / 'manifest.json'
            try:
                payload = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {'items': []}
                self._json(200, payload)
            except Exception as exc:
                self._json(500, {'error': f'Could not read ASL media manifest: {exc}'})
            return
        if route == '/api/status':
            try:
                from recognition_service import status
                self._json(200, status())
            except Exception as exc:
                self._json(503, {'ready': False, 'error': str(exc)})
            return
        # This service is also reachable from the public internet. Serve only
        # the browser app's static allowlist; never expose Python, checkpoints,
        # datasets, reports, or repository metadata from the project root.
        static_path = pathlib.Path(super().translate_path(self.path)).resolve()
        public_files = {
            pathlib.Path(ROOT) / 'index.html',
            pathlib.Path(ROOT) / 'label_mapping.json',
            pathlib.Path(ROOT) / 'model.json',
            pathlib.Path(ROOT) / 'group1-shard1of1.bin',
            *(pathlib.Path(ROOT) / 'dynamic_sign_videos' / filename
              for filename in ('J.mp4', 'Z.mp4', 'J.jpg', 'Z.jpg')),
        }
        public_directories = [
            pathlib.Path(ROOT) / 'assets',
            pathlib.Path(ROOT) / 'tfjs_model',
        ]
        if route == '/' or static_path in public_files or any(
            static_path == directory.resolve() or directory.resolve() in static_path.parents
            for directory in public_directories
        ):
            super().do_GET()
        else:
            self._json(404, {'error': 'Not found.'})

    def do_POST(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if route not in {'/api/recognize', '/api/test-image', '/api/test-video', '/api/test-commons-video'}:
            self._json(404, {'error': 'Unknown API endpoint.'})
            return
        try:
            if route == '/api/test-commons-video':
                size = int(self.headers.get('Content-Length', '0'))
                if size <= 0 or size > 8192:
                    self._json(413, {'error': 'Reference request must be between 1 byte and 8 KB.'})
                    return
                request_data = json.loads(self.rfile.read(size).decode('utf-8'))
                label = str(request_data.get('label', '')).upper()
                if label not in LETTER_CLIPS:
                    self._json(400, {'error': 'Choose the reviewed J or Z clip.'})
                    return
                from recognition_service import RuntimeBusyError, get_runtime
                video = _fetch_commons_alphabet_video(request_data.get('url', ''))
                start, end = LETTER_CLIPS[label]
                result = get_runtime().test_video(video, '.ogv', label, start, end)
                self._json(200, result)
                return
            size = int(self.headers.get('Content-Length', '0'))
            # Keep individual media tests below Vercel Functions' 4.5 MB
            # request body cap if these routes are ever proxied through one.
            limit = 4 * 1024 * 1024 if route in {'/api/test-image', '/api/test-video'} else 10 * 1024 * 1024
            if size <= 0 or size > limit:
                self._json(413, {'error': f'Upload must be between 1 byte and {limit // (1024 * 1024)} MB.'})
                return
            payload = self.rfile.read(size)
            if len(payload) != size:
                self._json(400, {'error': 'Incomplete frame upload.'})
                return
            from recognition_service import RuntimeBusyError, get_runtime
            runtime = get_runtime()
            if route == '/api/test-image':
                expected = parse_qs(parsed.query).get('expected', [None])[0]
                if expected and expected.upper() not in set('ABCDEFGHI KLMNOPQRSTUVWXY'.replace(' ', '')):
                    self._json(400, {'error': 'Static expected labels must be A–I or K–Y.'})
                    return
                result = runtime.test_image(payload, expected)
            elif route == '/api/test-video':
                expected = parse_qs(parsed.query).get('expected', [None])[0]
                if expected and expected.upper() not in {'J', 'Z'}:
                    self._json(400, {'error': 'Dynamic expected labels must be J or Z.'})
                    return
                suffix = pathlib.Path(parse_qs(parsed.query).get('name', ['clip.mp4'])[0]).suffix.lower()
                if suffix not in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.ogv'}:
                    self._json(415, {'error': 'Use an MP4, MOV, AVI, MKV, WebM, or OGV video.'})
                    return
                result = runtime.test_video(payload, suffix, expected)
            else:
                result = runtime.process_jpeg(payload)
            self._json(200, result)
        except RuntimeBusyError as exc:
            self._json(429, {'error': str(exc), 'frame_dropped': True})
        except ValueError as exc:
            self._json(400, {'error': str(exc)})
        except Exception as exc:
            self._json(500, {'error': str(exc)})

    def log_message(self, format, *args):
        return

os.chdir(ROOT)
server = ThreadingHTTPServer((HOST, PORT), CORSHandler)
print(f'Serving {ROOT} on {HOST}:{PORT}')
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    server.server_close()
    try:
        from recognition_service import _runtime
        if _runtime is not None:
            _runtime.close()
    except Exception:
        pass
