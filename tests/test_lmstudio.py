import unittest

import lmstudio
from tests.helpers import fixture, fixture_json


class TestNormalizeLoaded(unittest.TestCase):
    def test_maps_real_payload(self):
        out = lmstudio.normalize_loaded(fixture_json("lms_ps_loaded.json"))
        self.assertEqual(len(out), 1)
        m = out[0]
        self.assertEqual(m["identifier"], "google/gemma-4-31b")
        self.assertEqual(m["model_key"], "google/gemma-4-31b")
        self.assertEqual(m["display_name"], "Gemma 4 31B")
        self.assertEqual(m["arch"], "gemma4")
        self.assertEqual(m["quant"], "Q4_K_M")
        self.assertEqual(m["quant_bits"], 4)
        self.assertEqual(m["params"], "31B")
        self.assertEqual(m["size"], 19887882864)
        self.assertEqual(m["context"], 32768)
        self.assertEqual(m["max_context"], 262144)
        self.assertEqual(m["status"], "idle")
        self.assertEqual(m["queued"], 0)
        self.assertEqual(m["parallel"], 4)
        self.assertTrue(m["vision"])
        self.assertTrue(m["tools"])

    def test_null_ttl_becomes_none_not_zero(self):
        # ttlMs is null when no TTL is set; 0 would render as "expires now"
        m = lmstudio.normalize_loaded(fixture_json("lms_ps_loaded.json"))[0]
        self.assertIsNone(m["ttl_s"])

    def test_empty_payload(self):
        self.assertEqual(lmstudio.normalize_loaded(fixture_json("lms_ps_empty.json")), [])

    def test_tolerates_missing_optional_fields(self):
        out = lmstudio.normalize_loaded([{"modelKey": "bare"}])
        self.assertEqual(out[0]["model_key"], "bare")
        self.assertIsNone(out[0]["quant"])
        self.assertIsNone(out[0]["quant_bits"])
        self.assertEqual(out[0]["size"], 0)
        self.assertFalse(out[0]["vision"])

    def test_ttl_ms_converts_to_seconds(self):
        out = lmstudio.normalize_loaded([{"modelKey": "x", "ttlMs": 300000}])
        self.assertEqual(out[0]["ttl_s"], 300)


class TestNormalizeDisk(unittest.TestCase):
    def test_maps_real_payload(self):
        out = lmstudio.normalize_disk(fixture_json("lms_ls.json"))
        keys = {m["model_key"] for m in out}
        self.assertIn("google/gemma-4-31b", keys)
        self.assertIn("cyberpal2.0-20b-i1", keys)
        gemma = next(m for m in out if m["model_key"] == "google/gemma-4-31b")
        self.assertEqual(gemma["arch"], "gemma4")
        self.assertTrue(gemma["vision"])

    def test_indexed_id_is_carried_and_differs_from_model_key(self):
        """indexed_id is the only key that joins to the model index. Delete needs it."""
        out = lmstudio.normalize_disk(fixture_json("lms_ls.json"))
        by_key = {m["model_key"]: m for m in out}
        self.assertEqual(
            by_key["cyberpal2.0-20b-i1"]["indexed_id"],
            "mradermacher/CyberPal2.0-20B-i1-GGUF/CyberPal2.0-20B.i1-MXFP4_MOE.gguf")
        # For catalog models the two happen to coincide
        self.assertEqual(by_key["google/gemma-4-31b"]["indexed_id"], "google/gemma-4-31b")

    def test_every_model_carries_an_indexed_id(self):
        out = lmstudio.normalize_disk(fixture_json("lms_ls.json"))
        self.assertTrue(all(m["indexed_id"] for m in out))

    def test_sorted_by_model_key(self):
        out = lmstudio.normalize_disk(fixture_json("lms_ls.json"))
        self.assertEqual([m["model_key"] for m in out],
                         sorted(m["model_key"] for m in out))


class TestApiStates(unittest.TestCase):
    def test_extracts_load_state(self):
        states = lmstudio.api_states(fixture_json("api_v0_models.json"))
        self.assertEqual(states["google/gemma-4-31b"], "loaded")
        self.assertEqual(states["qwen/qwen3.6-27b"], "not-loaded")

    def test_malformed_payload_yields_empty(self):
        self.assertEqual(lmstudio.api_states({}), {})
        self.assertEqual(lmstudio.api_states({"data": "nonsense"}), {})


class TestJoinLibrary(unittest.TestCase):
    def test_marks_loaded_rows(self):
        disk = lmstudio.normalize_disk(fixture_json("lms_ls.json"))
        states = lmstudio.api_states(fixture_json("api_v0_models.json"))
        out = lmstudio.join_library(disk, states)
        gemma = next(m for m in out if m["model_key"] == "google/gemma-4-31b")
        self.assertTrue(gemma["loaded"])
        qwen = next(m for m in out if m["model_key"] == "qwen/qwen3.6-27b")
        self.assertFalse(qwen["loaded"])

    def test_model_missing_from_states_defaults_to_not_loaded(self):
        out = lmstudio.join_library([{"model_key": "ghost"}], {})
        self.assertFalse(out[0]["loaded"])

    def test_state_for_unknown_model_does_not_crash(self):
        disk = [{"model_key": "a"}]
        out = lmstudio.join_library(disk, {"b": "loaded"})
        self.assertEqual(len(out), 1)


class TestParseEngine(unittest.TestCase):
    def test_picks_the_selected_runtime(self):
        eng = lmstudio.parse_engine(fixture("lms_runtime_ls.txt"))
        self.assertEqual(eng["name"], "llama.cpp-linux-x86_64-nvidia-cuda12-avx2")
        self.assertEqual(eng["version"], "2.27.1")

    def test_no_selection_yields_empty(self):
        self.assertEqual(lmstudio.parse_engine("LLM ENGINE   SELECTED\nfoo@1.0\n"),
                         {"name": None, "version": None})


if __name__ == "__main__":
    unittest.main()
