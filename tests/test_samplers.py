import threading, time, unittest

import samplers
import sources


class TestSampled(unittest.TestCase):
    def test_computes_synchronously_on_first_access(self):
        """A request arriving before the first background tick must get real
        data, not a hole."""
        calls = []
        s = samplers.Sampled(lambda: calls.append(1) or "value")
        self.assertEqual(s.peek(), (None, False))
        self.assertEqual(s.get(), "value")
        self.assertEqual(len(calls), 1)

    def test_second_access_does_not_recompute(self):
        calls = []
        s = samplers.Sampled(lambda: (calls.append(1), "v")[1])
        s.get(); s.get(); s.get()
        self.assertEqual(len(calls), 1)

    def test_refresh_updates_the_value(self):
        seq = iter(["first", "second"])
        s = samplers.Sampled(lambda: next(seq))
        self.assertEqual(s.get(), "first")
        s.refresh()
        self.assertEqual(s.get(), "second")

    def test_failing_source_keeps_the_last_good_value(self):
        state = {"fail": False}
        def fn():
            if state["fail"]: raise RuntimeError("boom")
            return "good"
        s = samplers.Sampled(fn)
        self.assertEqual(s.get(), "good")
        state["fail"] = True
        s.refresh()  # must not raise
        self.assertEqual(s.get(), "good")

    def test_failing_source_on_cold_start_yields_the_default(self):
        s = samplers.Sampled(lambda: (_ for _ in ()).throw(RuntimeError("boom")), default=[])
        self.assertEqual(s.get(), [])

    def test_peek_never_computes(self):
        calls = []
        s = samplers.Sampled(lambda: calls.append(1))
        self.assertEqual(s.peek(), (None, False))
        self.assertEqual(calls, [])

    def test_concurrent_cold_start_computes_once(self):
        """A burst of first requests must not all shell out simultaneously."""
        calls = []
        def slow():
            calls.append(1)
            time.sleep(0.05)
            return "v"
        s = samplers.Sampled(slow)
        threads = [threading.Thread(target=s.get) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(len(calls), 1)

    def test_age_is_none_before_first_sample(self):
        self.assertIsNone(samplers.Sampled(lambda: 1).age())

    def test_age_tracks_the_last_set(self):
        s = samplers.Sampled(lambda: 1)
        s.refresh()
        self.assertLess(s.age(), 1.0)


class TestGpuCsvParsing(unittest.TestCase):
    # Captured from `nvidia-smi --query-gpu=<GPU_QUERY_FIELDS>` on this machine.
    REAL = ("NVIDIA GeForce RTX 5090, 18074, 32607, 0, 37, N/A, 0, "
            "14.62, 600.00, 1, 4, 4, 16, 0x0000000000000000")

    def test_parses_a_real_row(self):
        g = sources.parse_gpu_csv(self.REAL)
        self.assertEqual(g["name"], "NVIDIA GeForce RTX 5090")
        self.assertEqual(g["mem_used"], 18074)
        self.assertEqual(g["mem_total"], 32607)
        self.assertEqual(g["util"], 0)
        self.assertEqual(g["temp"], 37)
        self.assertEqual(g["power_limit"], 600.0)
        self.assertEqual(g["pcie_gen"], 1)
        self.assertEqual(g["pcie_width_max"], 16)

    def test_na_fields_become_none_not_zero(self):
        """temp_mem reports [N/A] on this card; 0 would render as a real reading."""
        self.assertIsNone(sources.parse_gpu_csv(self.REAL)["temp_mem"])

    def test_streaming_and_one_shot_agree_on_shape(self):
        streamed = sources.parse_gpu_csv(self.REAL)
        one_shot = sources.gpu()
        if "error" not in one_shot:
            self.assertEqual(set(streamed), set(one_shot))


class TestHistoryThrottle(unittest.TestCase):
    def setUp(self):
        sources._HIST.clear()
        sources._LAST_HIST_PUSH[0] = 0.0

    def tearDown(self):
        sources._HIST.clear()
        sources._LAST_HIST_PUSH[0] = 0.0

    G = {"mem_used": 100, "mem_total": 200, "util": 50, "temp": 40}

    def test_rapid_samples_are_throttled(self):
        """At 10 samples/sec an unthrottled 60-slot buffer would cover only 6s."""
        now = 1000.0
        accepted = sum(sources.push_history(self.G, now=now + i * 0.1, min_interval=1)
                       for i in range(50))
        self.assertEqual(accepted, 5)   # 5 seconds of samples -> 5 entries
        self.assertEqual(len(sources._HIST), 5)

    def test_first_sample_always_lands(self):
        self.assertTrue(sources.push_history(self.G, now=1000.0, min_interval=1))

    def test_error_samples_are_skipped(self):
        self.assertFalse(sources.push_history({"error": "no gpu"}, now=1000.0))
        self.assertEqual(len(sources._HIST), 0)

    def test_history_content(self):
        sources.push_history(self.G, now=1000.0, min_interval=1)
        e = sources._HIST[-1]
        self.assertEqual(e["vram_pct"], 50.0)
        self.assertEqual(e["util"], 50)


class TestLivePayload(unittest.TestCase):
    def test_live_is_a_strict_subset_of_state(self):
        live, state = sources.live(), sources.state()
        self.assertTrue(set(live) <= set(state))

    def test_live_carries_the_moving_values(self):
        live = sources.live()
        for k in ("now", "dash_uptime_s", "gpu", "gpu_history", "pcie", "host"):
            self.assertIn(k, live)

    def test_live_omits_the_slow_lists(self):
        """Shipping these 10x/second is the waste the split exists to avoid."""
        live = sources.live()
        for k in ("library", "loaded", "requests", "settings", "model_activity"):
            self.assertNotIn(k, live)


class TestModelTimeline(unittest.TestCase):
    def setUp(self):
        samplers.MODEL_TIMELINE.clear()

    def tearDown(self):
        samplers.MODEL_TIMELINE.clear()

    def test_records_the_resident_model(self):
        samplers.record_timeline([{"model_key": "gemma4:31b"}], now=100.0)
        self.assertEqual(samplers.model_at(100.0), "gemma4:31b")

    def test_records_none_when_nothing_is_loaded(self):
        samplers.record_timeline([], now=100.0)
        self.assertIsNone(samplers.model_at(100.0))

    def test_lookup_uses_the_most_recent_observation_at_or_before(self):
        samplers.record_timeline([{"model_key": "a:1"}], now=100.0)
        samplers.record_timeline([{"model_key": "b:1"}], now=200.0)
        self.assertEqual(samplers.model_at(150.0), "a:1")
        self.assertEqual(samplers.model_at(250.0), "b:1")

    def test_request_before_any_observation_is_unattributed(self):
        samplers.record_timeline([{"model_key": "a:1"}], now=200.0)
        self.assertIsNone(samplers.model_at(100.0))

    def test_multiple_loaded_models_is_ambiguous_not_a_guess(self):
        # With MAX_LOADED_MODELS>1 we cannot know which one served a request.
        # Returning the first would be a confident lie.
        samplers.record_timeline([{"model_key": "a:1"}, {"model_key": "b:1"}],
                                 now=100.0)
        self.assertIsNone(samplers.model_at(100.0))

    def test_timeline_is_bounded(self):
        for i in range(samplers.PS_TIMELINE_LEN + 100):
            samplers.record_timeline([{"model_key": "a:1"}], now=float(i))
        self.assertEqual(len(samplers.MODEL_TIMELINE), samplers.PS_TIMELINE_LEN)

    def test_attribute_tags_rows_in_place(self):
        samplers.record_timeline([{"model_key": "a:1"}], now=100.0)
        rows = [{"kind": "request", "epoch": 150.0, "model": None}]
        out = samplers.attribute(rows)
        self.assertEqual(out[0]["model"], "a:1")

    def test_attribute_leaves_problems_alone(self):
        samplers.record_timeline([{"model_key": "a:1"}], now=100.0)
        rows = [{"kind": "problem", "epoch": 150.0, "model": None}]
        self.assertIsNone(samplers.attribute(rows)[0]["model"])


if __name__ == "__main__":
    unittest.main()
