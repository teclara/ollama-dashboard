import unittest
from unittest import mock

import ollama
from tests.helpers import fixture_json


class TestNormalizeLoaded(unittest.TestCase):
    def test_maps_real_payload(self):
        out = ollama.normalize_loaded(fixture_json("api_ps_loaded.json"))
        m = out[0]
        self.assertEqual(m["model_key"], "gemma4:12b")
        # Ollama has one name. All three identifier slots collapse onto it,
        # unlike LM Studio where load key, instance name, and index key differed.
        self.assertEqual(m["identifier"], m["model_key"])
        self.assertEqual(m["display_name"], m["model_key"])
        self.assertEqual(m["arch"], "gemma4")
        self.assertEqual(m["quant"], "Q4_K_M")
        self.assertEqual(m["params"], "11.9B")
        self.assertEqual(m["context"], 8192)

    def test_fully_resident_model_reports_no_cpu_spill(self):
        m = ollama.normalize_loaded(fixture_json("api_ps_loaded.json"))[0]
        self.assertEqual(m["size"], m["size_vram"])
        self.assertEqual(m["cpu_bytes"], 0)
        self.assertTrue(m["fully_gpu"])

    def test_partial_offload_is_detected(self):
        raw = {"models": [{"name": "big:70b", "size": 1000, "size_vram": 600,
                           "details": {}}]}
        m = ollama.normalize_loaded(raw)[0]
        self.assertEqual(m["cpu_bytes"], 400)
        self.assertFalse(m["fully_gpu"])

    def test_ttl_derived_from_expires_at(self):
        raw = {"models": [{"name": "m", "details": {},
                           "expires_at": "2026-07-30T23:39:24.821098399-04:00"}]}
        with mock.patch("ollama.time.time", return_value=1785468864.0):
            # 2026-07-31T03:34:24Z == 1785468864; expiry is 5 minutes later.
            m = ollama.normalize_loaded(raw)[0]
        self.assertEqual(m["ttl_s"], 300)

    def test_expired_ttl_clamps_to_zero_not_negative(self):
        raw = {"models": [{"name": "m", "details": {},
                           "expires_at": "2020-01-01T00:00:00.000000000-04:00"}]}
        m = ollama.normalize_loaded(raw)[0]
        self.assertEqual(m["ttl_s"], 0)

    def test_missing_expires_at_is_none_not_zero(self):
        # None renders as "no TTL"; 0 would render as "expires now".
        m = ollama.normalize_loaded({"models": [{"name": "m", "details": {}}]})[0]
        self.assertIsNone(m["ttl_s"])

    def test_empty_payload(self):
        self.assertEqual(ollama.normalize_loaded(fixture_json("api_ps_empty.json")), [])

    def test_tolerates_garbage(self):
        self.assertEqual(ollama.normalize_loaded(None), [])
        self.assertEqual(ollama.normalize_loaded({}), [])
        self.assertEqual(ollama.normalize_loaded({"models": "nonsense"}), [])


class TestNormalizeLibrary(unittest.TestCase):
    def test_maps_real_payload(self):
        out = ollama.normalize_library(fixture_json("api_tags.json"))
        by_key = {m["model_key"]: m for m in out}
        m = by_key["nomic-embed-text:latest"]
        self.assertEqual(m["arch"], "nomic-bert")
        self.assertEqual(m["quant"], "F16")
        self.assertEqual(m["params"], "137M")
        self.assertTrue(m["embedding"])
        self.assertFalse(m["tools"])

    def test_capabilities_become_flags(self):
        raw = {"models": [{"name": "a:1", "details": {},
                           "capabilities": ["completion", "tools", "thinking", "vision"]}]}
        m = ollama.normalize_library(raw)[0]
        self.assertTrue(m["tools"])
        self.assertTrue(m["thinking"])
        self.assertTrue(m["vision"])
        self.assertFalse(m["embedding"])

    def test_missing_context_length_is_none_not_zero(self):
        # /api/tags omits details.context_length for some models (gemma4:31b
        # lacks it, ornith:35b has it). Zero would render as "0 token context".
        raw = {"models": [{"name": "a:1", "details": {"family": "x"}}]}
        self.assertIsNone(ollama.normalize_library(raw)[0]["max_context"])

    def test_sorted_by_model_key(self):
        raw = {"models": [{"name": "z:1", "details": {}}, {"name": "a:1", "details": {}}]}
        self.assertEqual([m["model_key"] for m in ollama.normalize_library(raw)],
                         ["a:1", "z:1"])


class TestJoinLibrary(unittest.TestCase):
    def test_marks_loaded_models(self):
        disk = [{"model_key": "a:1", "loaded": False},
                {"model_key": "b:1", "loaded": False}]
        out = ollama.join_library(disk, {"a:1"})
        self.assertTrue(out[0]["loaded"])
        self.assertFalse(out[1]["loaded"])


class TestLiveWrappers(unittest.TestCase):
    def test_loaded_models_returns_empty_on_transport_error(self):
        # The server being down must degrade to an empty list, never a 500.
        with mock.patch("ollama.api_get", side_effect=ollama.OllamaError("down")):
            self.assertEqual(ollama.loaded_models(), [])

    def test_library_returns_empty_on_transport_error(self):
        with mock.patch("ollama.api_get", side_effect=ollama.OllamaError("down")):
            self.assertEqual(ollama.library(), [])

    def test_ping_is_false_when_unreachable(self):
        with mock.patch("ollama.api_get", side_effect=ollama.OllamaError("down")):
            self.assertFalse(ollama.ping())

    def test_ping_is_true_when_version_responds(self):
        with mock.patch("ollama.api_get", return_value={"version": "0.32.5"}):
            self.assertTrue(ollama.ping())

    def test_library_passes_loaded_names_through(self):
        tags = {"models": [{"name": "a:1", "details": {}}]}
        with mock.patch("ollama.api_get", return_value=tags):
            out = ollama.library(loaded=[{"model_key": "a:1"}])
        self.assertTrue(out[0]["loaded"])


if __name__ == "__main__":
    unittest.main()
