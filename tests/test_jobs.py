import json, unittest
from unittest import mock

import control
import ollama
import samplers


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

    def test_garbage_numbers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_ctx"):
            control.build_load_payload("m", context="banana")

    def test_out_of_range_numbers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_ctx"):
            control.build_load_payload("m", context=0)
        with self.assertRaisesRegex(ValueError, "num_gpu"):
            control.build_load_payload("m", gpu=-1)

    def test_invalid_options_do_not_claim_a_job_slot(self):
        control.clear_all_jobs()
        with self.assertRaises(ValueError):
            control.start_load("m", context=0)
        self.assertEqual(control.get_jobs(), {})


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

    def test_progress_does_not_regress_when_equal_layer_appears(self):
        p = control.PullProgress()
        p.update({"digest": "a", "completed": 100, "total": 100}, now=1.0)
        before = p.snapshot()["pct"]
        p.update({"digest": "b", "completed": 0, "total": 100}, now=2.0)
        self.assertIsNone(p.snapshot()["pct"])
        self.assertLess(before, 100)

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


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        control.clear_all_jobs()

    def test_build_config_applies_defaults_and_deduplicates(self):
        config = control.build_benchmark_config(["a:1", "a:1", "b:1"])
        self.assertEqual(config["models"], ["a:1", "b:1"])
        self.assertEqual(config["warmups"], 1)
        self.assertEqual(config["runs"], 3)
        self.assertEqual(config["num_predict"], 128)
        self.assertTrue(config["prompt"])

    def test_build_config_rejects_invalid_bounds(self):
        with self.assertRaisesRegex(ValueError, "runs"):
            control.build_benchmark_config(["a:1"], runs=0)
        with self.assertRaisesRegex(ValueError, "num_predict"):
            control.build_benchmark_config(["a:1"], num_predict=5000)
        with self.assertRaisesRegex(ValueError, "at least one"):
            control.build_benchmark_config([])

    def test_summarizes_runs_with_medians(self):
        runs = [
            {"generation_tps": 10.0, "prompt_tps": 20.0, "ttft_s": 1.0,
             "load_s": 0.1, "total_s": 3.0, "wall_s": 3.1,
             "prompt_tokens": 10, "output_tokens": 20},
            {"generation_tps": 14.0, "prompt_tps": 30.0, "ttft_s": 0.5,
             "load_s": 0.0, "total_s": 2.0, "wall_s": 2.1,
             "prompt_tokens": 10, "output_tokens": 20},
        ]
        out = control.summarize_benchmark_runs("a:1", runs)
        self.assertEqual(out["generation_tps"], 12.0)
        self.assertEqual(out["prompt_tps"], 25.0)
        self.assertEqual(out["ttft_s"], 0.75)

    def test_benchmark_is_exclusive_with_other_jobs(self):
        self.assertTrue(control._claim_job("load:1", "load"))
        self.assertFalse(control._claim_job(control._BENCHMARK_KEY, "benchmark"))
        control.clear_all_jobs()
        self.assertTrue(control._claim_job(control._BENCHMARK_KEY, "benchmark"))
        self.assertFalse(control._claim_job("load:1", "load"))
        control.clear_all_jobs()
        self.assertTrue(control._begin_immediate_mutation())
        try:
            self.assertFalse(control._claim_job(control._BENCHMARK_KEY, "benchmark"))
        finally:
            control._end_immediate_mutation()

    def test_run_records_measured_results_not_warmup(self):
        config = control.build_benchmark_config(["a:1"], warmups=1, runs=2,
                                                num_predict=16)
        control._claim_job(control._BENCHMARK_KEY, "benchmark")
        sample = {"generation_tps": 12.0, "prompt_tps": 30.0,
                  "ttft_s": 0.2, "load_s": 0.0, "total_s": 1.5,
                  "wall_s": 1.6, "prompt_tokens": 12, "output_tokens": 16,
                  "done_reason": "length"}
        with mock.patch("control.ollama.benchmark_generate", return_value=sample):
            control._run_benchmark(config)
        job = control.get_jobs()[control._BENCHMARK_KEY]
        self.assertTrue(job["done"])
        self.assertEqual(len(job["results"][0]["runs"]), 2)
        self.assertEqual(job["results"][0]["generation_tps"], 12.0)


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

    def test_thread_start_failure_marks_the_job_failed(self):
        with mock.patch("control.threading.Thread") as thread:
            thread.return_value.start.side_effect = RuntimeError("no threads")
            with self.assertRaises(RuntimeError):
                control.start_load("m")
        job = control.get_jobs()["m"]
        self.assertTrue(job["done"])
        self.assertIn("no threads", job["error"])

    def test_active_jobs_are_bounded(self):
        for i in range(control._MAX_ACTIVE_JOBS):
            self.assertTrue(control._claim_job(f"m{i}", "load"))
        self.assertFalse(control._claim_job("one-too-many", "load"))

    def test_finished_job_history_is_bounded(self):
        for i in range(control._MAX_JOB_HISTORY + 10):
            key = f"m{i}"
            self.assertTrue(control._claim_job(key, "load"))
            control._update_job(key, done=True, finished=float(i))
        self.assertLessEqual(len(control.get_jobs()), control._MAX_JOB_HISTORY)


class TestPullJob(unittest.TestCase):
    class Response:
        def __init__(self, events, exit_error=None):
            self.lines = [json.dumps(event).encode() + b"\n" for event in events]
            self.exit_error = exit_error

        def __enter__(self): return self
        def __exit__(self, *args):
            if self.exit_error:
                raise self.exit_error
            return False
        def __iter__(self): return iter(self.lines)

    def setUp(self):
        control.clear_all_jobs()
        control._claim_job("m", "download")

    def test_requires_an_explicit_success_event(self):
        response = self.Response([{"status": "pulling manifest"}])
        with mock.patch("control.urllib.request.urlopen", return_value=response):
            control._run_pull("m", "m")
        job = control.get_jobs()["m"]
        self.assertTrue(job["done"])
        self.assertEqual(job["status"], "failed")
        self.assertIn("before Ollama reported success", job["error"])

    def test_success_event_finishes_and_refreshes_inventory(self):
        response = self.Response([{"status": "success"}])
        with mock.patch("control.urllib.request.urlopen", return_value=response), \
             mock.patch.object(samplers.LIBRARY, "refresh") as refresh:
            control._run_pull("m", "m")
        job = control.get_jobs()["m"]
        self.assertTrue(job["done"])
        self.assertEqual(job["status"], "finished")
        self.assertIsNone(job["error"])
        refresh.assert_called_once_with()

    def test_socket_error_after_success_does_not_overwrite_success(self):
        response = self.Response([{"status": "success"}], OSError("late reset"))
        with mock.patch("control.urllib.request.urlopen", return_value=response), \
             mock.patch.object(samplers.LIBRARY, "refresh"):
            control._run_pull("m", "m")
        job = control.get_jobs()["m"]
        self.assertEqual(job["status"], "finished")
        self.assertIsNone(job["error"])

    def test_success_is_terminal_even_if_more_events_follow(self):
        response = self.Response([
            {"status": "success"},
            {"error": "should not be consumed"},
        ])
        with mock.patch("control.urllib.request.urlopen", return_value=response), \
             mock.patch.object(samplers.LIBRARY, "refresh"):
            control._run_pull("m", "m")
        job = control.get_jobs()["m"]
        self.assertEqual(job["status"], "finished")
        self.assertIsNone(job["error"])


class TestUnload(unittest.TestCase):
    def setUp(self):
        control.clear_all_jobs()

    def test_unload_sends_keep_alive_zero(self):
        with mock.patch("control.ollama.api_post") as post:
            control.unload_model("gemma4:12b")
        payload = post.call_args[0][1]
        self.assertEqual(payload["model"], "gemma4:12b")
        self.assertEqual(payload["keep_alive"], 0)

    def test_unload_all_iterates_loaded_models(self):
        with mock.patch.object(samplers.LOADED, "get",
                               return_value=[{"model_key": "a:1"}, {"model_key": "b:1"}]), \
             mock.patch("control.ollama.api_post") as post:
            control.unload_all()
        self.assertEqual([c[0][1]["model"] for c in post.call_args_list],
                         ["a:1", "b:1"])

    def test_unload_all_with_nothing_loaded_is_a_noop(self):
        with mock.patch.object(samplers.LOADED, "get", return_value=[]), \
             mock.patch("control.ollama.api_post") as post:
            control.unload_all()
        post.assert_not_called()

    def test_benchmark_blocks_unload(self):
        control._claim_job(control._BENCHMARK_KEY, "benchmark")
        with mock.patch("control.ollama.api_post") as post:
            with self.assertRaisesRegex(ValueError, "benchmark"):
                control.unload_model("gemma4:12b")
        post.assert_not_called()


class TestEstimateFit(unittest.TestCase):
    def test_compares_model_size_to_free_vram(self):
        # Replaces `lms load --estimate-only`, which has no Ollama equivalent.
        with mock.patch.object(samplers.LIBRARY, "get",
                               return_value=[{"model_key": "m", "size": 20_000_000_000}]), \
             mock.patch.object(samplers.GPU, "get",
                               return_value={"mem_used": 1000, "mem_total": 32_600}):
            out = control.estimate_fit("m")
        self.assertTrue(out["ok"])
        self.assertEqual(out["model_bytes"], 20_000_000_000)
        self.assertGreater(out["free_bytes"], 0)

    def test_reports_when_a_model_does_not_fit(self):
        with mock.patch.object(samplers.LIBRARY, "get",
                               return_value=[{"model_key": "m", "size": 90_000_000_000}]), \
             mock.patch.object(samplers.GPU, "get",
                               return_value={"mem_used": 1000, "mem_total": 32_600}):
            self.assertFalse(control.estimate_fit("m")["fits"])

    def test_unknown_model(self):
        with mock.patch.object(samplers.LIBRARY, "get", return_value=[]):
            self.assertFalse(control.estimate_fit("nope")["ok"])

    def test_gpu_error_propagates_as_not_ok(self):
        with mock.patch.object(samplers.LIBRARY, "get",
                               return_value=[{"model_key": "m", "size": 1}]), \
             mock.patch.object(samplers.GPU, "get", return_value={"error": "no nvidia-smi"}):
            out = control.estimate_fit("m")
        self.assertFalse(out["ok"])
        self.assertIn("nvidia-smi", out["error"])

    def test_unsampled_gpu_is_reported_without_live_probe(self):
        with mock.patch.object(samplers.LIBRARY, "get",
                               return_value=[{"model_key": "m", "size": 1}]), \
             mock.patch.object(samplers.GPU, "get", return_value={}), \
             mock.patch("sources.gpu") as live_gpu:
            out = control.estimate_fit("m")
        self.assertFalse(out["ok"])
        self.assertIn("not been sampled", out["error"])
        live_gpu.assert_not_called()


if __name__ == "__main__":
    unittest.main()
