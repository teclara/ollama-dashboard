import unittest
from unittest import mock

import control
import ollama


class TestDeleteModel(unittest.TestCase):
    """Ollama has a real delete endpoint, so the LM Studio version's model-index
    resolution, path containment checks, and rmtree are all gone. What must
    survive is the confirmation guard and the refusal to delete a loaded model."""

    def setUp(self):
        control.clear_all_jobs()

    def test_confirmation_must_match_exactly(self):
        for confirm in ("gemma4", None, "", "gemma4:12B"):
            with self.subTest(confirm=confirm):
                out = control.delete_model("gemma4:12b", confirm)
                self.assertFalse(out["ok"])
                self.assertIn("confirmation", out["error"])

    def test_a_matching_confirmation_is_not_rejected_by_the_guard(self):
        with mock.patch("control.ollama.loaded_models", return_value=[]), \
             mock.patch("control.ollama.api_delete", return_value={}):
            self.assertTrue(control.delete_model("gemma4:12b", "gemma4:12b")["ok"])

    def test_refuses_to_delete_a_loaded_model(self):
        with mock.patch("control.ollama.loaded_models",
                        return_value=[{"model_key": "gemma4:12b"}]):
            out = control.delete_model("gemma4:12b", "gemma4:12b")
        self.assertFalse(out["ok"])
        self.assertIn("unload", out["error"])

    def test_a_different_model_being_loaded_does_not_block_the_delete(self):
        with mock.patch("control.ollama.loaded_models",
                        return_value=[{"model_key": "other:1"}]), \
             mock.patch("control.ollama.api_delete", return_value={}):
            self.assertTrue(control.delete_model("gemma4:12b", "gemma4:12b")["ok"])

    def test_calls_the_delete_api(self):
        with mock.patch("control.ollama.loaded_models", return_value=[]), \
             mock.patch("control.ollama.api_delete", return_value={}) as api:
            out = control.delete_model("gemma4:12b", "gemma4:12b")
        self.assertTrue(out["ok"])
        self.assertEqual(out["removed"], ["gemma4:12b"])
        api.assert_called_once_with("/api/delete", {"model": "gemma4:12b"})

    def test_does_not_call_the_api_when_the_guard_rejects(self):
        with mock.patch("control.ollama.api_delete") as api:
            control.delete_model("gemma4:12b", "wrong")
        api.assert_not_called()

    def test_reports_api_failure(self):
        with mock.patch("control.ollama.loaded_models", return_value=[]), \
             mock.patch("control.ollama.api_delete",
                        side_effect=ollama.OllamaError("HTTP 404")):
            out = control.delete_model("nope:1", "nope:1")
        self.assertFalse(out["ok"])
        self.assertIn("HTTP 404", out["error"])

    def test_refuses_to_delete_when_loaded_state_cannot_be_verified(self):
        with mock.patch("control.ollama.loaded_models",
                        side_effect=ollama.OllamaError("connection refused")), \
             mock.patch("control.ollama.api_delete") as api:
            out = control.delete_model("gemma4:12b", "gemma4:12b")
        self.assertFalse(out["ok"])
        self.assertIn("verify", out["error"])
        api.assert_not_called()

    def test_strict_loaded_check_rejects_malformed_payload(self):
        with mock.patch("ollama.api_get", return_value={}), \
             mock.patch("control.ollama.api_delete") as api:
            out = control.delete_model("gemma4:12b", "gemma4:12b")
        self.assertFalse(out["ok"])
        self.assertIn("verify", out["error"])
        api.assert_not_called()

    def test_strict_loaded_check_rejects_model_without_name(self):
        with mock.patch("ollama.api_get", return_value={"models": [{}]}), \
             mock.patch("control.ollama.api_delete") as api:
            out = control.delete_model("gemma4:12b", "gemma4:12b")
        self.assertFalse(out["ok"])
        self.assertIn("verify", out["error"])
        api.assert_not_called()

    def test_benchmark_blocks_delete_before_live_checks(self):
        control._claim_job(control._BENCHMARK_KEY, "benchmark")
        with mock.patch("control.ollama.loaded_models") as loaded, \
             mock.patch("control.ollama.api_delete") as api:
            out = control.delete_model("gemma4:12b", "gemma4:12b")
        self.assertFalse(out["ok"])
        self.assertIn("benchmark", out["error"])
        loaded.assert_not_called()
        api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
