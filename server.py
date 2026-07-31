#!/usr/bin/env python3
"""Live dashboard for Ollama. Stdlib only."""
import http.server, json, os, socketserver

import config
import samplers
import sources
import control

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def _read_template(name):
    with open(os.path.join(TEMPLATE_DIR, name), "rb") as f:
        return f.read()


INDEX_HTML = _read_template("index.html")
CONTROL_HTML = _read_template("control.html")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a): pass

    def _read_json(self):
        n = int(self.headers.get("Content-Length", "0") or 0)
        if not n: return {}
        try: return json.loads(self.rfile.read(n))
        except Exception: return {}

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
        body = self._read_json()
        try:
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
            if self.path == "/api/control/jobs/clear":
                control.clear_finished_jobs()
                return self._json(200, {"ok": True})
        except Exception as e:
            return self._json(500, {"error": str(e)})
        self._send(404, "text/plain", b"not found")

    def do_DELETE(self):
        body = self._read_json()
        try:
            if self.path == "/api/control/model":
                name = (body.get("name") or "").strip()
                if not name: return self._json(400, {"error": "name required"})
                result = control.delete_model(name, body.get("confirm"))
                return self._json(200 if result["ok"] else 400, result)
        except Exception as e:
            return self._json(500, {"error": str(e)})
        self._send(404, "text/plain", b"not found")


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    sources.start_pcie_monitor()
    samplers.start_all()          # also starts the journald follower
    with ThreadedServer((config.HOST, config.PORT), Handler) as s:
        print(f"ollama dashboard on http://{config.HOST}:{config.PORT}")
        s.serve_forever()
