import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pathlib

ROOT = str(pathlib.Path(__file__).parent.resolve())
PORT = int(os.environ.get('SIGNSYNC_PORT', '8001'))

class CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        origin = self.headers.get('Origin')
        if origin in {f'http://127.0.0.1:{PORT}', f'http://localhost:{PORT}'}:
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
        if self.path.split('?', 1)[0] == '/api/status':
            try:
                from recognition_service import status
                self._json(200, status())
            except Exception as exc:
                self._json(503, {'ready': False, 'error': str(exc)})
            return
        super().do_GET()

    def do_POST(self):
        if self.path.split('?', 1)[0] != '/api/recognize':
            self._json(404, {'error': 'Unknown API endpoint.'})
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size <= 0 or size > 5 * 1024 * 1024:
                self._json(413, {'error': 'Frame must be between 1 byte and 5 MB.'})
                return
            payload = self.rfile.read(size)
            if len(payload) != size:
                self._json(400, {'error': 'Incomplete frame upload.'})
                return
            from recognition_service import RuntimeBusyError, get_runtime
            result = get_runtime().process_jpeg(payload)
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
server = ThreadingHTTPServer(('127.0.0.1', PORT), CORSHandler)
print(f'Serving {ROOT} on http://127.0.0.1:{PORT}')
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
