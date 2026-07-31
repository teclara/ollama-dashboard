import json, threading, unittest, urllib.error, urllib.request

import server


class TestRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = server.ThreadedServer(("127.0.0.1", 0), server.Handler)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def _req(self, path, method="GET", body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_dashboard_served(self):
        code, body = self._req("/")
        self.assertEqual(code, 200)
        self.assertIn(b"<html", body.lower())

    def test_control_panel_served(self):
        self.assertEqual(self._req("/control")[0], 200)

    def test_shared_app_assets_are_served(self):
        self.assertEqual(self._req("/assets/app.css")[0], 200)
        self.assertEqual(self._req("/assets/app.js")[0], 200)

    def test_unknown_path_404s(self):
        self.assertEqual(self._req("/nope")[0], 404)

    def test_state_returns_expected_keys(self):
        code, body = self._req("/api/state")
        self.assertEqual(code, 200)
        s = json.loads(body)
        for k in ("gpu", "loaded", "library", "requests", "stats_5m",
                  "top_endpoints", "by_client", "problems", "log_age_s",
                  "disk", "service", "host", "settings", "ollama_ok"):
            self.assertIn(k, s)

    def test_state_has_no_removed_lmstudio_keys(self):
        s = json.loads(self._req("/api/state")[1])
        # model_activity was parsed out of LM Studio's logs. Ollama's GIN lines
        # carry no model name, so attribution moved to samplers.attribute().
        self.assertNotIn("model_activity", s)
        self.assertNotIn("lms_ok", s)

    def test_state_never_contains_credentials(self):
        body = self._req("/api/state")[1]
        for probe in (b"hfDownloadToken", b"API_KEY=", b"sk-"):
            self.assertNotIn(probe, body)

    def test_jobs_endpoint(self):
        code, body = self._req("/api/control/jobs")
        self.assertEqual(code, 200)
        self.assertIsInstance(json.loads(body), dict)

    def test_load_requires_a_model(self):
        code, body = self._req("/api/control/load", "POST", {})
        self.assertEqual(code, 400)
        self.assertIn("model", json.loads(body)["error"])

    def test_download_requires_a_name(self):
        code, body = self._req("/api/control/download", "POST", {"name": "  "})
        self.assertEqual(code, 400)

    def test_benchmark_requires_models(self):
        code, body = self._req("/api/control/benchmark", "POST", {})
        self.assertEqual(code, 400)
        self.assertIn("models", json.loads(body)["error"])

    def test_delete_requires_matching_confirmation(self):
        code, body = self._req("/api/control/model", "DELETE",
                               {"name": "x", "confirm": "y"})
        self.assertEqual(code, 400)
        self.assertIn("confirmation", json.loads(body)["error"].lower())

    def test_delete_requires_a_name(self):
        self.assertEqual(self._req("/api/control/model", "DELETE", {})[0], 400)

    def test_unload_requires_identifier_or_all(self):
        self.assertEqual(self._req("/api/control/unload", "POST", {})[0], 400)

    def test_json_body_must_be_an_object(self):
        self.assertEqual(self._req("/api/control/load", "POST", ["model"])[0], 400)

    def test_json_body_size_is_bounded(self):
        code, body = self._req("/api/control/load", "POST", {"model": "x" * 70000})
        self.assertEqual(code, 400)
        self.assertIn("request body", json.loads(body)["error"])


if __name__ == "__main__":
    unittest.main()
