import unittest
from unittest import mock

import control
import ollama


class TestBuildLoadPayload(unittest.TestCase):
    def test_minimal(self):
        p = control.build_load_payload("gemma4:12b")
        self.assertEqual(p["model"], "gemma4:12b")
        # An empty prompt makes /api/generate a pure load with no generation.
        self.assertEqual(p["prompt"], "")
        self.assertNotIn("options", p)

    def test_context_becomes_num_ctx(self):
        self.assertEqual(
            control.build_load_payload("m", context=8192)["options"]["num_ctx"], 8192)

    def test_gpu_becomes_num_gpu(self):
        self.assertEqual(
            control.build_load_payload("m", gpu=99)["options"]["num_gpu"], 99)

    def test_ttl_becomes_keep_alive(self):
        self.assertEqual(control.build_load_payload("m", ttl="30m")["keep_alive"], "30m")

    def test_blank_options_are_omitted_not_sent_as_null(self):
        # Sending num_ctx: null would override the server default with garbage.
        p = control.build_load_payload("m", context="", gpu=None, ttl="")
        self.assertNotIn("options", p)
        self.assertNotIn("keep_alive", p)

    def test_numeric_strings_are_coerced(self):
        p = control.build_load_payload("m", context="8192", gpu="99")
        self.assertEqual(p["options"]["num_ctx"], 8192)
        self.assertEqual(p["options"]["num_gpu"], 99)

    def test_garbage_numbers_are_dropped_not_forwarded(self):
        self.assertNotIn("options", control.build_load_payload("m", context="banana"))


class TestPullProgress(unittest.TestCase):
    def test_sums_across_layers(self):
        # Ollama reports completed/total per blob digest. Taking the latest
        # pair would make the bar jump backwards each time a layer starts.
        p = control.PullProgress()
        p.update({"digest": "sha256:a", "completed": 100, "total": 100})
        p.update({"digest": "sha256:b", "completed": 50, "total": 200})
        s = p.snapshot()
        self.assertEqual(s["completed"], 150)
        self.assertEqual(s["total"], 300)
        self.assertAlmostEqual(s["pct"], 50.0)

    def test_progress_never_goes_backwards_across_a_real_stream(self):
        p = control.PullProgress()
        events = [
            {"status": "pulling manifest"},
            {"status": "pulling 970aa74c0a90", "digest": "sha256:970a",
             "total": 274290656, "completed": 137145328},
            {"status": "pulling 970aa74c0a90", "digest": "sha256:970a",
             "total": 274290656, "completed": 274290656},
            {"status": "pulling c71d239df917", "digest": "sha256:c71d",
             "total": 11357, "completed": 0},
            {"status": "verifying sha256 digest"},
            {"status": "success"},
        ]
        seen = []
        for e in events:
            p.update(e)
            pct = p.snapshot()["pct"]
            if pct is not None:
                seen.append(pct)
        self.assertEqual(seen, sorted(seen), f"progress regressed: {seen}")

    def test_indeterminate_status_preserves_the_last_percentage(self):
        p = control.PullProgress()
        p.update({"digest": "sha256:a", "completed": 50, "total": 100})
        before = p.snapshot()["pct"]
        p.update({"status": "verifying sha256 digest"})
        self.assertEqual(p.snapshot()["pct"], before)

    def test_status_is_carried_through(self):
        p = control.PullProgress()
        p.update({"status": "pulling manifest"})
        self.assertEqual(p.snapshot()["last_line"], "pulling manifest")

    def test_empty_progress_has_no_percentage(self):
        self.assertIsNone(control.PullProgress().snapshot()["pct"])

    def test_rate_is_derived_from_deltas(self):
        p = control.PullProgress()
        p.update({"digest": "a", "completed": 0, "total": 1000}, now=100.0)
        p.update({"digest": "a", "completed": 500, "total": 1000}, now=102.0)
        self.assertAlmostEqual(p.snapshot()["rate_bps"], 250.0)

    def test_eta_is_derived_from_rate(self):
        p = control.PullProgress()
        p.update({"digest": "a", "completed": 0, "total": 1000}, now=100.0)
        p.update({"digest": "a", "completed": 500, "total": 1000}, now=102.0)
        self.assertAlmostEqual(p.snapshot()["eta_s"], 2.0)


class TestLoadJob(unittest.TestCase):
    def setUp(self):
        control.clear_all_jobs()

    def test_start_load_claims_a_slot(self):
        with mock.patch("control.threading.Thread"):
            self.assertTrue(control.start_load("m"))
            self.assertFalse(control.start_load("m"))

    def test_load_marks_done_on_load_reason(self):
        control._claim_job("m", "load")
        with mock.patch("control.ollama.api_post",
                        return_value={"done": True, "done_reason": "load"}):
            control._run_load("m", {"model": "m", "prompt": ""})
        job = control.get_jobs()["m"]
        self.assertTrue(job["done"])
        self.assertIsNone(job["error"])

    def test_unexpected_done_reason_is_recorded_as_an_error(self):
        control._claim_job("m", "load")
        with mock.patch("control.ollama.api_post",
                        return_value={"done": True, "done_reason": "stop"}):
            control._run_load("m", {"model": "m", "prompt": ""})
        self.assertIn("stop", control.get_jobs()["m"]["error"])

    def test_load_records_an_error_when_the_api_fails(self):
        control._claim_job("m", "load")
        with mock.patch("control.ollama.api_post",
                        side_effect=ollama.OllamaError("HTTP 500")):
            control._run_load("m", {"model": "m", "prompt": ""})
        job = control.get_jobs()["m"]
        self.assertTrue(job["done"])
        self.assertIn("HTTP 500", job["error"])


class TestUnload(unittest.TestCase):
    def test_unload_sends_keep_alive_zero(self):
        with mock.patch("control.ollama.api_post") as post:
            control.unload_model("gemma4:12b")
        payload = post.call_args[0][1]
        self.assertEqual(payload["model"], "gemma4:12b")
        self.assertEqual(payload["keep_alive"], 0)

    def test_unload_all_iterates_loaded_models(self):
        with mock.patch("control.ollama.loaded_models",
                        return_value=[{"model_key": "a:1"}, {"model_key": "b:1"}]), \
             mock.patch("control.ollama.api_post") as post:
            control.unload_all()
        self.assertEqual([c[0][1]["model"] for c in post.call_args_list],
                         ["a:1", "b:1"])

    def test_unload_all_with_nothing_loaded_is_a_noop(self):
        with mock.patch("control.ollama.loaded_models", return_value=[]), \
             mock.patch("control.ollama.api_post") as post:
            control.unload_all()
        post.assert_not_called()


class TestEstimateFit(unittest.TestCase):
    def test_compares_model_size_to_free_vram(self):
        # Replaces `lms load --estimate-only`, which has no Ollama equivalent.
        with mock.patch("control.ollama.library",
                        return_value=[{"model_key": "m", "size": 20_000_000_000}]), \
             mock.patch("control.sources.gpu",
                        return_value={"mem_used": 1000, "mem_total": 32_600}):
            out = control.estimate_fit("m")
        self.assertTrue(out["ok"])
        self.assertEqual(out["model_bytes"], 20_000_000_000)
        self.assertGreater(out["free_bytes"], 0)

    def test_reports_when_a_model_does_not_fit(self):
        with mock.patch("control.ollama.library",
                        return_value=[{"model_key": "m", "size": 90_000_000_000}]), \
             mock.patch("control.sources.gpu",
                        return_value={"mem_used": 1000, "mem_total": 32_600}):
            self.assertFalse(control.estimate_fit("m")["fits"])

    def test_unknown_model(self):
        with mock.patch("control.ollama.library", return_value=[]):
            self.assertFalse(control.estimate_fit("nope")["ok"])

    def test_gpu_error_propagates_as_not_ok(self):
        with mock.patch("control.ollama.library",
                        return_value=[{"model_key": "m", "size": 1}]), \
             mock.patch("control.sources.gpu", return_value={"error": "no nvidia-smi"}):
            out = control.estimate_fit("m")
        self.assertFalse(out["ok"])
        self.assertIn("nvidia-smi", out["error"])


if __name__ == "__main__":
    unittest.main()
