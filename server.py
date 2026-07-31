#!/usr/bin/env python3
"""Live dashboard for Ollama. Stdlib only."""
import http.server, json, os, socketserver, threading

import config
import samplers
import sources
import control

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
MAX_JSON_BODY = 64 * 1024
REQUEST_TIMEOUT_SEC = 30


def _read_template(name):
    with open(os.path.join(TEMPLATE_DIR, name), "rb") as f:
        return f.read()


INDEX_HTML = _read_template("index.html")
CONTROL_HTML = _read_template("control.html")
APP_CSS = _read_template("app.css")
APP_JS = _read_template("app.js")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(REQUEST_TIMEOUT_SEC)

    def log_message(self, *a): pass

    def _read_json(self):
        try:
            n = int(self.headers.get("Content-Length", "0") or 0)
        except (TypeError, ValueError):
            raise ValueError("invalid Content-Length")
        if n < 0 or n > MAX_JSON_BODY:
            raise ValueError(f"request body must be between 0 and {MAX_JSON_BODY} bytes")
        if not n: return {}
        try:
            body = json.loads(self.rfile.read(n))
        except Exception as e:
            raise ValueError(f"invalid JSON: {e}")
        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object")
        return body

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, "application/json", json.dumps(obj).encode())

    def _load_opts(self, body):
        # Ollama supports num_ctx, num_gpu and keep_alive per load. There is no
        # per-load parallelism setting and no instance identifier.
        return {k: body.get(k) for k in ("context", "gpu", "ttl")}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, "text/html; charset=utf-8", INDEX_HTML)
        if self.path == "/control":
            return self._send(200, "text/html; charset=utf-8", CONTROL_HTML)
        if self.path == "/assets/app.css":
            return self._send(200, "text/css; charset=utf-8", APP_CSS)
        if self.path == "/assets/app.js":
            return self._send(200, "text/javascript; charset=utf-8", APP_JS)
        if self.path == "/api/state":
            return self._json(200, sources.state())
        # Small fast-moving slice, safe to poll many times a second.
        if self.path == "/api/live":
            return self._json(200, sources.live())
        if self.path == "/api/control/jobs":
            return self._json(200, control.get_jobs())
        if self.path.startswith("/api/control/catalog"):
            return self._json(200, control.catalog(force="refresh=1" in self.path))
        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        try:
            body = self._read_json()
            if self.path == "/api/control/load":
                model = (body.get("model") or "").strip()
                if not model: return self._json(400, {"error": "model required"})
                return self._json(200, {"started": control.start_load(
                    model, **self._load_opts(body))})
            if self.path == "/api/control/load/fit":
                model = (body.get("model") or "").strip()
                if not model: return self._json(400, {"error": "model required"})
                return self._json(200, control.estimate_fit(model))
            if self.path == "/api/control/unload":
                if body.get("all"):
                    control.unload_all()
                    return self._json(200, {"ok": True})
                # Ollama identifies a loaded model by its name; `identifier` is
                # accepted so an older client still works.
                name = (body.get("model") or body.get("identifier") or "").strip()
                if not name:
                    return self._json(400, {"error": "model or all required"})
                control.unload_model(name)
                return self._json(200, {"ok": True})
            if self.path == "/api/control/download":
                name = (body.get("name") or "").strip()
                if not name: return self._json(400, {"error": "name required"})
                return self._json(200, {"started": control.start_download(name)})
            if self.path == "/api/control/benchmark":
                return self._json(200, {"started": control.start_benchmark(
                    models=body.get("models"), prompt=body.get("prompt"),
                    warmups=body.get("warmups", 1), runs=body.get("runs", 3),
                    num_predict=body.get("num_predict", 128),
                    context=body.get("context"),
                )})
            if self.path == "/api/control/jobs/clear":
                control.clear_finished_jobs()
                return self._json(200, {"ok": True})
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:
            return self._json(500, {"error": str(e)})
        self._send(404, "text/plain", b"not found")

    def do_DELETE(self):
        try:
            body = self._read_json()
            if self.path == "/api/control/model":
                name = (body.get("name") or "").strip()
                if not name: return self._json(400, {"error": "name required"})
                result = control.delete_model(name, body.get("confirm"))
                return self._json(200 if result["ok"] else 400, result)
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:
            return self._json(500, {"error": str(e)})
        self._send(404, "text/plain", b"not found")


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, **kwargs):
        self._request_slots = threading.BoundedSemaphore(32)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        self._request_slots.acquire()
        try:
            super().process_request(request, client_address)
        except Exception:
            self._request_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


if __name__ == "__main__":
    sources.start_pcie_monitor()
    samplers.start_all()          # also starts the journald follower
    with ThreadedServer((config.HOST, config.PORT), Handler) as s:
        print(f"ollama dashboard on http://{config.HOST}:{config.PORT}")
        s.serve_forever()
