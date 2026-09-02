import os
import socketserver
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = r"C:\Users\adwai\Downloads\sign_language"
PORT = 8001

class CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def log_message(self, format, *args):
        return

os.chdir(ROOT)
server = ThreadingHTTPServer(('127.0.0.1', PORT), CORSHandler)
print(f'Serving {ROOT} on http://127.0.0.1:{PORT}')
server.serve_forever()
