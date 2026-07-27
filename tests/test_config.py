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
        c = self._fresh(LMSTUDIO_URL=None, LMSTUDIO_DASHBOARD_PORT=None)
        self.assertEqual(c.LMSTUDIO_URL, "http://localhost:1234")
        self.assertEqual(c.PORT, 11435)
        self.assertEqual(c.HOST, "127.0.0.1")
        self.assertTrue(c.SYSTEMD_USER)
        self.assertEqual(c.SYSTEMD_UNIT, "lmstudio-server")
        self.assertTrue(c.LMS_BIN.endswith("/.lmstudio/bin/lms"))
        self.assertNotIn("~", c.LOG_DIR)  # expanded

    def test_env_override(self):
        c = self._fresh(LMSTUDIO_URL="http://box:4321/", LMSTUDIO_DASHBOARD_PORT="9000")
        self.assertEqual(c.LMSTUDIO_URL, "http://box:4321")  # trailing slash stripped
        self.assertEqual(c.PORT, 9000)

    def test_bad_int_falls_back_to_default(self):
        c = self._fresh(LMSTUDIO_DASHBOARD_PORT="not-a-number")
        self.assertEqual(c.PORT, 11435)

    def test_systemd_user_is_boolean_from_string(self):
        self.assertFalse(self._fresh(LMSTUDIO_SYSTEMD_USER="0").SYSTEMD_USER)
        self.assertTrue(self._fresh(LMSTUDIO_SYSTEMD_USER="1").SYSTEMD_USER)

    def test_no_ollama_names_remain(self):
        c = self._fresh()
        leftovers = [n for n in dir(c) if "OLLAMA" in n.upper()]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
