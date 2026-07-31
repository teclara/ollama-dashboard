import importlib, os, unittest


class TestConfig(unittest.TestCase):
    def _fresh(self, **env):
        """Reload config with a patched environment."""
        old = {k: os.environ.get(k) for k in env}
        os.environ.update({k: v for k, v in env.items() if v is not None})
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
        try:
            import config
            return importlib.reload(config)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_defaults(self):
        c = self._fresh(OLLAMA_URL=None, OLLAMA_DASHBOARD_PORT=None,
                        OLLAMA_DASHBOARD_SYSTEMD_USER=None)
        self.assertEqual(c.OLLAMA_URL, "http://localhost:11434")
        self.assertEqual(c.PORT, 11435)
        self.assertEqual(c.HOST, "127.0.0.1")
        self.assertEqual(c.SYSTEMD_UNIT, "ollama")
        self.assertEqual(c.JOURNAL_UNIT, "ollama")

    def test_service_is_a_system_unit_not_a_user_unit(self):
        # Ollama installs to /etc/systemd/system. LM Studio was a user unit;
        # inheriting True here would make every systemctl call silently fail.
        self.assertFalse(self._fresh(OLLAMA_DASHBOARD_SYSTEMD_USER=None).SYSTEMD_USER)

    def test_env_override_strips_trailing_slash(self):
        c = self._fresh(OLLAMA_URL="http://box:11434/", OLLAMA_DASHBOARD_PORT="9000")
        self.assertEqual(c.OLLAMA_URL, "http://box:11434")
        self.assertEqual(c.PORT, 9000)

    def test_bad_int_falls_back_to_default(self):
        self.assertEqual(self._fresh(OLLAMA_DASHBOARD_PORT="not-a-number").PORT, 11435)

    def test_out_of_range_port_falls_back_to_default(self):
        self.assertEqual(self._fresh(OLLAMA_DASHBOARD_PORT="0").PORT, 11435)
        self.assertEqual(self._fresh(OLLAMA_DASHBOARD_PORT="70000").PORT, 11435)

    def test_nonpositive_sampling_interval_falls_back_to_default(self):
        c = self._fresh(OLLAMA_DASHBOARD_LOADED_SAMPLE_SEC="0")
        self.assertEqual(c.LOADED_SAMPLE_SEC, 2)

    def test_negative_buffer_length_falls_back_to_default(self):
        c = self._fresh(OLLAMA_DASHBOARD_LOG_WINDOW_LINES="-1")
        self.assertEqual(c.LOG_WINDOW_LINES, 2000)

    def test_models_dir_reads_ollamas_own_var(self):
        c = self._fresh(OLLAMA_MODELS="/mnt/big/models")
        self.assertEqual(c.MODELS_DIR_FALLBACK, "/mnt/big/models")

    def test_models_dir_default_is_the_system_store(self):
        c = self._fresh(OLLAMA_MODELS=None)
        self.assertEqual(c.MODELS_DIR_FALLBACK, "/usr/share/ollama/.ollama/models")

    def test_dashboard_vars_do_not_squat_ollamas_namespace(self):
        # A dashboard var named OLLAMA_MODELS or OLLAMA_HOST would collide with
        # the real server's configuration. Only OLLAMA_URL is permitted bare.
        import config
        src = open(config.__file__, encoding="utf-8").read()
        import re
        bare = set(re.findall(r'_env\("(OLLAMA_(?!DASHBOARD_)[A-Z_]+)"', src))
        bare |= set(re.findall(r'_path\("(OLLAMA_(?!DASHBOARD_)[A-Z_]+)"', src))
        self.assertEqual(bare - {"OLLAMA_URL", "OLLAMA_MODELS"}, set())

    def test_no_lmstudio_names_remain(self):
        c = self._fresh()
        leftovers = [n for n in dir(c) if "LMSTUDIO" in n.upper() or n == "LMS_BIN"]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
