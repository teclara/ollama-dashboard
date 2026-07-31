import os, unittest
from unittest import mock

import sources


class TestNoOllamaRemnants(unittest.TestCase):
    def test_gin_parsing_is_gone(self):
        self.assertFalse(hasattr(sources, "GIN_RE"))
        self.assertFalse(hasattr(sources, "parse_latency_ms"))
        self.assertFalse(hasattr(sources, "top_clients"))

    def test_hardware_panels_survive(self):
        for name in ("gpu", "gpu_processes", "nvidia_versions", "host",
                     "pcie", "tailscale", "start_pcie_monitor"):
            self.assertTrue(hasattr(sources, name), name)


class TestParseSystemdEnvironment(unittest.TestCase):
    def test_parses_the_real_output(self):
        # `systemctl show ollama --property=Environment` emits one
        # space-separated line, quoting only values that need it.
        text = ('Environment=OLLAMA_HOST=http://0.0.0.0:11434 "OLLAMA_ORIGINS=*" '
                'OLLAMA_FLASH_ATTENTION=true OLLAMA_MAX_LOADED_MODELS=1\n')
        env = sources.parse_systemd_environment(text)
        self.assertEqual(env["OLLAMA_HOST"], "http://0.0.0.0:11434")
        self.assertEqual(env["OLLAMA_ORIGINS"], "*")
        self.assertEqual(env["OLLAMA_FLASH_ATTENTION"], "true")
        self.assertEqual(env["OLLAMA_MAX_LOADED_MODELS"], "1")

    def test_ignores_non_ollama_vars(self):
        text = 'Environment=PATH=/usr/bin OLLAMA_KEEP_ALIVE=30m\n'
        env = sources.parse_systemd_environment(text)
        self.assertNotIn("PATH", env)
        self.assertEqual(env["OLLAMA_KEEP_ALIVE"], "30m")

    def test_redacts_secret_looking_names(self):
        # The old whitelist cannot work here: we cannot enumerate what the user
        # may add later, so anything secret-shaped is denied instead.
        text = ('Environment=OLLAMA_API_KEY=sk-abc123 OLLAMA_AUTH_TOKEN=t '
                'OLLAMA_SECRET=s OLLAMA_HOST=http://x\n')
        env = sources.parse_systemd_environment(text)
        self.assertNotIn("sk-abc123", str(env))
        self.assertEqual(env["OLLAMA_API_KEY"], "<redacted>")
        self.assertEqual(env["OLLAMA_AUTH_TOKEN"], "<redacted>")
        self.assertEqual(env["OLLAMA_SECRET"], "<redacted>")
        self.assertEqual(env["OLLAMA_HOST"], "http://x")

    def test_empty_input(self):
        self.assertEqual(sources.parse_systemd_environment(""), {})
        self.assertEqual(sources.parse_systemd_environment(None), {})


class TestModelsRoot(unittest.TestCase):
    def test_prefers_the_servers_own_setting(self):
        self.assertEqual(
            sources.models_root({"OLLAMA_MODELS": "/mnt/models"}), "/mnt/models")

    def test_falls_back_to_config(self):
        from config import MODELS_DIR_FALLBACK
        self.assertEqual(sources.models_root({}), MODELS_DIR_FALLBACK)
        self.assertEqual(sources.models_root(None), MODELS_DIR_FALLBACK)


class TestDisk(unittest.TestCase):
    def test_statvfs_walks_up_past_an_unreadable_dir(self):
        # /usr/share/ollama is 0750 ollama:ollama. A process without that group
        # gets PermissionError on the models dir but the filesystem totals are
        # identical for any ancestor on the same mount.
        real = os.statvfs

        def fake(path):
            if path.endswith("/models"):
                raise PermissionError(13, "Permission denied")
            return real("/")

        with mock.patch("sources.os.statvfs", side_effect=fake), \
             mock.patch("sources.os.path.isdir", return_value=True), \
             mock.patch("sources._du", return_value=None):
            info = sources.disk("/usr/share/ollama/.ollama/models")
        self.assertGreater(info["fs_total"], 0)

    def test_falls_back_to_summing_tags_when_du_is_denied(self):
        with mock.patch("sources.os.path.isdir", return_value=True), \
             mock.patch("sources._du", return_value=None), \
             mock.patch("sources.ollama.library",
                        return_value=[{"size": 100}, {"size": 200}]):
            info = sources.disk("/anywhere")
        self.assertEqual(info["models_size"], 300)
        self.assertTrue(info["approximate"])

    def test_du_result_is_exact_and_reports_orphans(self):
        # du counts every blob; summing /api/tags counts only referenced ones.
        # The difference is unreferenced blob space and is worth surfacing.
        with mock.patch("sources.os.path.isdir", return_value=True), \
             mock.patch("sources._du", return_value=67460511656), \
             mock.patch("sources.ollama.library",
                        return_value=[{"size": 48866552236}]):
            info = sources.disk("/anywhere")
        self.assertEqual(info["models_size"], 67460511656)
        self.assertFalse(info["approximate"])
        self.assertEqual(info["orphan_bytes"], 67460511656 - 48866552236)

    def test_missing_directory_returns_zeroes_not_an_exception(self):
        with mock.patch("sources.os.path.isdir", return_value=False):
            info = sources.disk("/nope")
        self.assertEqual(info["models_size"], 0)
        self.assertIsNone(info["models_dir"])


class TestServiceInfo(unittest.TestCase):
    def test_reports_engine_version_from_the_api(self):
        with mock.patch("sources._systemctl", side_effect=Exception("no systemd")), \
             mock.patch("sources.ollama.version", return_value={"version": "0.32.5"}):
            info = sources.service_info()
        self.assertEqual(info["engine"]["version"], "0.32.5")
        self.assertEqual(info["engine"]["name"], "ollama")

    def test_survives_systemctl_failure(self):
        with mock.patch("sources._systemctl", side_effect=Exception("boom")), \
             mock.patch("sources.ollama.version", return_value={}):
            info = sources.service_info()
        self.assertEqual(info["active"], "unknown")
        self.assertIsNone(info["pid"])


if __name__ == "__main__":
    unittest.main()


class TestDiskPermissionDenied(unittest.TestCase):
    """/usr/share/ollama is 0750 ollama:ollama. os.path.isdir() is False for a
    process outside that group, exactly as it is for a directory that does not
    exist — but the two must not be reported the same way."""

    def test_unreadable_store_is_flagged_approximate_not_reported_as_zero(self):
        with mock.patch("sources._readable_dir", return_value=(True, False)), \
             mock.patch("sources.ollama.library",
                        return_value=[{"size": 100}, {"size": 200}]), \
             mock.patch("sources._statvfs_walk_up", return_value=None):
            info = sources.disk("/usr/share/ollama/.ollama/models")
        self.assertEqual(info["models_size"], 300)
        self.assertTrue(info["approximate"],
                        "an unreadable store reported as exact is a confident lie")
        self.assertIsNotNone(info["models_dir"])

    def test_unreadable_store_still_reports_filesystem_totals(self):
        fake = mock.Mock(f_blocks=1000, f_frsize=4096, f_bavail=250)
        with mock.patch("sources._readable_dir", return_value=(True, False)), \
             mock.patch("sources.ollama.library", return_value=[]), \
             mock.patch("sources._statvfs_walk_up", return_value=fake):
            info = sources.disk("/anywhere")
        self.assertEqual(info["fs_total"], 4096000)
        self.assertEqual(info["fs_free"], 1024000)

    def test_genuinely_missing_directory_is_not_flagged_approximate(self):
        with mock.patch("sources._readable_dir", return_value=(False, False)):
            info = sources.disk("/definitely/not/here")
        self.assertIsNone(info["models_dir"])
        self.assertEqual(info["models_size"], 0)
        self.assertFalse(info["approximate"])
