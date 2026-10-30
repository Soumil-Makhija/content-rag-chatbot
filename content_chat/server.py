"""Single-user loopback HTTP API; no remote bind, accounts, or telemetry."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sqlite3
import sys
from urllib.parse import unquote, urlsplit

from .models import ModelProtocolError, ModelUnavailable
from .service import Service
from .store import Store

MAX_BODY = 3 * 1024 * 1024
WEB = Path(__file__).with_name('web')


def make_server(database, model, port=8765) -> HTTPServer:
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError('Port must be an integer from 0 to 65535.')

    class Handler(BaseHTTPRequestHandler):
        server_version = 'ContentChat/0.1'
        sys_version = ''

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, *_args):
            # Avoid recording questions/document text or untrusted URL content.
            pass

        def reply(self, status, data, content_type='application/json; charset=utf-8'):
            body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def guard(self):
            port = self.server.server_port
            hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
            if self.headers.get('Host') not in hosts:
                self.reply(403, {'error': 'Invalid local Host header.'})
                return False
            origin = self.headers.get('Origin')
            if origin is not None and origin not in {f'http://{host}' for host in hosts}:
                self.reply(403, {'error': 'Cross-origin access is not allowed.'})
                return False
            return True

        def body(self):
            if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
                raise ValueError('Requests must use application/json.')
            if self.headers.get('Transfer-Encoding'):
                raise ValueError('Chunked request bodies are not supported.')
            try:
                size = int(self.headers.get('Content-Length', '-1'))
            except ValueError as exc:
                raise ValueError('Invalid Content-Length.') from exc
            if not 0 <= size <= MAX_BODY:
                raise ValueError('JSON request must be at most 3 MiB.')
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError('Incomplete request body.')
            def invalid_constant(_value):
                raise ValueError('JSON must contain finite numbers.')
            try:
                result = json.loads(raw, parse_constant=invalid_constant)
            except (ValueError, UnicodeError) as exc:
                raise ValueError('Invalid JSON request.') from exc
            if not isinstance(result, dict):
                raise ValueError('JSON request must be an object.')
            return result

        def handle_request(self, method):
            if not self.guard():
                return
            static = {'/': ('index.html', 'text/html; charset=utf-8'),
                      '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                      '/style.css': ('style.css', 'text/css; charset=utf-8')}
            try:
                path = urlsplit(self.path).path
                if method == 'GET' and path in static:
                    filename, mime = static[path]
                    self.reply(200, (WEB / filename).read_bytes(), mime)
                    return
                if method == 'GET' and path == '/api/health':
                    try:
                        health = model.health()
                    except (ModelUnavailable, ModelProtocolError) as exc:
                        health = {'available': False, 'ready': False, 'error': str(exc)}
                    self.reply(200, {'application': 'ready', 'models': health})
                    return
                parts = [unquote(part) for part in path.strip('/').split('/')]
                if len(parts) < 4 or parts[:2] != ['api', 'collections']:
                    self.reply(404, {'error': 'Route not found.'}); return
                collection, resource = parts[2:4]
                payload = self.body() if method == 'POST' else {}
                with Store(database) as store:
                    service = Service(store, model)
                    if resource == 'documents' and len(parts) == 4:
                        if method == 'GET':
                            result = store.list_documents(collection)
                        elif method == 'POST':
                            result = service.import_document(collection, payload.get('title'), payload.get('text'), payload.get('document_id'))
                        else:
                            self.reply(405, {'error': 'Method not allowed.'}); return
                    elif resource == 'documents' and len(parts) == 5:
                        if method == 'DELETE':
                            if not store.delete(collection, parts[4]):
                                self.reply(404, {'error': 'Document not found.'}); return
                            result = {'deleted': True}
                        elif method == 'GET':
                            result = store.get_document(collection, parts[4])
                            if result is None:
                                self.reply(404, {'error': 'Document not found.'}); return
                        else:
                            self.reply(405, {'error': 'Method not allowed.'}); return
                    elif resource == 'sources' and len(parts) == 5 and method == 'GET':
                        result = store.source(collection, parts[4])
                        if result is None:
                            self.reply(404, {'error': 'Source not found; it may have been replaced or deleted.'}); return
                    elif resource == 'ask' and len(parts) == 4 and method == 'POST':
                        result = service.ask(collection, payload.get('question'), payload.get('limit', 5))
                        for source in result['citations']:
                            source['url'] = f'/api/collections/{collection}/sources/{source["id"]}'
                    elif resource == 'search' and len(parts) == 4 and method == 'POST':
                        result = service.search(collection, payload.get('question'), payload.get('limit', 5))
                    else:
                        self.reply(404, {'error': 'Route not found.'}); return
                self.reply(200, result)
            except ValueError as exc:
                self.reply(400, {'error': str(exc)})
            except ModelUnavailable as exc:
                self.reply(503, {'error': str(exc)})
            except ModelProtocolError as exc:
                self.reply(502, {'error': str(exc)})
            except (OSError, sqlite3.Error):
                self.reply(500, {'error': 'Local storage failed. Check the database path, permissions and disk space.'})

        def do_GET(self): self.handle_request('GET')
        def do_POST(self): self.handle_request('POST')
        def do_DELETE(self): self.handle_request('DELETE')

    return HTTPServer(('127.0.0.1', port), Handler)


def serve(database, model, port=8765):
    # Validate/create the database before announcing readiness.
    with Store(database):
        pass
    with make_server(database, model, port) as server:
        print(f'Content Chat: http://127.0.0.1:{server.server_port} (single-user local access)', file=sys.stderr, flush=True)
        server.serve_forever()
