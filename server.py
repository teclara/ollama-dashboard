#!/usr/bin/env python3
"""Live dashboard for Ollama. Stdlib only."""
import http.server, json, os, socketserver

import config
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

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, "text/html; charset=utf-8", INDEX_HTML)
        if self.path == "/control":
            return self._send(200, "text/html; charset=utf-8", CONTROL_HTML)
        if self.path == "/api/state":
            s = sources.state()
            s["server_config"] = sources.server_config()
            return self._json(200, s)
        if self.path == "/api/control/pulls":
            return self._json(200, control.get_pulls())
        if self.path.startswith("/api/control/library_remote"):
            return self._json(200, control.library_remote(force="refresh=1" in self.path))
        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        body = self._read_json()
        try:
            if self.path == "/api/control/pull":
                name = body.get("name", "").strip()
                if not name: return self._json(400, {"error": "name required"})
                return self._json(200, {"started": control.start_pull(name)})
            if self.path == "/api/control/unload":
                control.unload_model(body["name"])
                return self._json(200, {"ok": True})
            if self.path == "/api/control/test":
                return self._json(200, control.run_scenario(
                    body.get("model"), body.get("scenario"), body.get("custom_prompt")))
            if self.path == "/api/control/pulls/clear":
                control.clear_finished_pulls()
                return self._json(200, {"ok": True})
        except Exception as e:
            return self._json(500, {"error": str(e)})
        self._send(404, "text/plain", b"not found")

    def do_DELETE(self):
        body = self._read_json()
        try:
            if self.path == "/api/control/model":
                control.delete_model(body["name"])
                return self._json(200, {"ok": True})
        except Exception as e:
            return self._json(500, {"error": str(e)})
        self._send(404, "text/plain", b"not found")


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    sources.start_pcie_monitor()
    with ThreadedServer((config.HOST, config.PORT), Handler) as s:
        print(f"ollama dashboard on http://{config.HOST}:{config.PORT}")
        s.serve_forever()
