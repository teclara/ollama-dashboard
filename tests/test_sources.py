import json, os, tempfile, unittest

import sources


class TestSettings(unittest.TestCase):
    def _write(self, d, obj):
        p = os.path.join(d, "settings.json")
        with open(p, "w") as f: json.dump(obj, f)
        return p

    def test_whitelisted_keys_pass_through(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, {
                "downloadsFolder": "/models",
                "defaultContextLength": {"type": "custom", "value": 65536},
                "modelLoadingGuardrails": {"mode": "high"},
                "enableLocalService": True,
                "useHFProxy": True,
            })
            s = sources.settings(p)
            self.assertEqual(s["downloadsFolder"], "/models")
            self.assertEqual(s["defaultContextLength"]["value"], 65536)
            self.assertTrue(s["enableLocalService"])

    def test_credentials_never_leak(self):
        """hfDownloadToken and friends must not appear anywhere in the output."""
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, {
                "downloadsFolder": "/models",
                "hfSearchToken": "hf_SEARCHSECRET",
                "hfDownloadToken": "hf_DOWNLOADSECRET",
                "credentials": {"nested": "hf_NESTEDSECRET"},
            })
            s = sources.settings(p)
            blob = json.dumps(s)
            self.assertNotIn("SEARCHSECRET", blob)
            self.assertNotIn("DOWNLOADSECRET", blob)
            self.assertNotIn("NESTEDSECRET", blob)
            self.assertNotIn("hfSearchToken", s)
            self.assertNotIn("hfDownloadToken", s)

    def test_no_raw_field(self):
        """The Ollama version dumped the whole file as `raw`. It must not come back."""
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, {"downloadsFolder": "/models", "secret": "x"})
            self.assertNotIn("raw", sources.settings(p))

    def test_unreadable_file_is_not_fatal(self):
        s = sources.settings("/nonexistent/settings.json")
        self.assertIn("error", s)
        self.assertEqual(s.get("downloadsFolder"), None)


class TestModelsRoot(unittest.TestCase):
    def test_prefers_downloads_folder(self):
        self.assertEqual(sources.models_root({"downloadsFolder": "/custom"}), "/custom")

    def test_falls_back_when_absent(self):
        import config
        self.assertEqual(sources.models_root({}), config.MODELS_DIR_FALLBACK)

    def test_falls_back_when_empty_string(self):
        import config
        self.assertEqual(sources.models_root({"downloadsFolder": ""}),
                         config.MODELS_DIR_FALLBACK)


class TestNoOllamaRemnants(unittest.TestCase):
    def test_gin_parsing_is_gone(self):
        self.assertFalse(hasattr(sources, "GIN_RE"))
        self.assertFalse(hasattr(sources, "parse_latency_ms"))
        self.assertFalse(hasattr(sources, "top_clients"))

    def test_hardware_panels_survive(self):
        for name in ("gpu", "gpu_processes", "nvidia_versions", "host",
                     "pcie", "tailscale", "start_pcie_monitor"):
            self.assertTrue(hasattr(sources, name), name)


if __name__ == "__main__":
    unittest.main()
