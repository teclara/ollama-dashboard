import unittest
from unittest import mock

import control
import lmstudio


class TestBuildLoadArgs(unittest.TestCase):
    def test_minimal(self):
        self.assertEqual(control.build_load_args("google/gemma-4-31b"),
                         ["load", "-y", "google/gemma-4-31b"])

    def test_all_options(self):
        args = control.build_load_args(
            "m", context=8192, gpu="max", ttl=300, parallel=2, identifier="fast")
        self.assertEqual(args, ["load", "-y", "m", "-c", "8192", "--gpu", "max",
                                "--ttl", "300", "--parallel", "2",
                                "--identifier", "fast"])

    def test_estimate_only_flag(self):
        self.assertIn("--estimate-only", control.build_load_args("m", estimate=True))

    def test_none_options_are_omitted(self):
        args = control.build_load_args("m", context=None, gpu=None, ttl=None)
        self.assertEqual(args, ["load", "-y", "m"])

    def test_numbers_are_stringified(self):
        args = control.build_load_args("m", context=4096)
        self.assertTrue(all(isinstance(a, str) for a in args))


class TestParseProgress(unittest.TestCase):
    def test_percentage(self):
        self.assertEqual(control.parse_progress("Downloading... 42.5%"), {"pct": 42.5})

    def test_integer_percentage(self):
        self.assertEqual(control.parse_progress("  7% done"), {"pct": 7.0})

    def test_byte_pair(self):
        r = control.parse_progress("1.50 GB / 3.00 GB")
        self.assertAlmostEqual(r["completed"] / r["total"], 0.5, places=2)

    def test_mixed_units(self):
        r = control.parse_progress("512.00 MB / 2.00 GB")
        self.assertAlmostEqual(r["completed"] / r["total"], 0.25, places=2)

    def test_unrecognized_line(self):
        self.assertIsNone(control.parse_progress("Resolving model..."))
        self.assertIsNone(control.parse_progress(""))

    def test_percentages_above_100_rejected(self):
        """Guards against matching an unrelated number followed by %."""
        self.assertIsNone(control.parse_progress("saved 250% of the time"))


# Captured verbatim from `lms get -y qwen3.5` on 2026-07-26, including the
# spinner glyph, bar, and the trailing ANSI cursor-restore escapes.
REAL_PROGRESS = (
    "⠏ [███                  ] 1.48% | 96.97 MB / 6.55 GB "
    "|  9.46 MB/s | ETA 11:22          \x1b[u\x1b[?25l\x1b[s"
)


class TestRealLmsGetOutput(unittest.TestCase):
    """`lms get` draws a live bar: CR-delimited, ANSI-laden, not newline separated."""

    def test_parses_a_real_progress_segment(self):
        r = control.parse_progress(REAL_PROGRESS)
        self.assertEqual(r["pct"], 1.48)
        self.assertEqual(r["completed"], int(96.97 * 1024**2))
        self.assertEqual(r["total"], int(6.55 * 1024**3))
        self.assertEqual(r["rate_bps"], int(9.46 * 1024**2))
        self.assertEqual(r["eta"], "11:22")

    def test_ansi_escapes_are_stripped(self):
        self.assertNotIn("\x1b", control.clean_line(REAL_PROGRESS))

    def test_eta_is_not_mistaken_for_progress(self):
        """ETA 11:22 must not be read as a byte or percent figure."""
        r = control.parse_progress(REAL_PROGRESS)
        self.assertLess(r["pct"], 2)

    def test_segments_split_on_carriage_returns(self):
        """The whole download arrives as one CR-updated line; splitting on \\n alone
        would surface a single unterminated segment and never update the UI."""
        import io
        stream = io.StringIO("\r".join(["a 1.0% | 1.00 MB / 10.00 MB",
                                        "b 2.0% | 2.00 MB / 10.00 MB",
                                        "c 3.0% | 3.00 MB / 10.00 MB"]))
        segs = list(control._stream_segments(stream, chunk_size=7))
        self.assertEqual(len(segs), 3)
        self.assertEqual(control.parse_progress(segs[-1])["pct"], 3.0)

    def test_stream_segments_handles_mixed_crlf(self):
        import io
        segs = list(control._stream_segments(io.StringIO("one\r\ntwo\nthree\r"), 4))
        self.assertEqual([s.strip() for s in segs], ["one", "two", "three"])


class TestEstimateLoad(unittest.TestCase):
    """`lms load --estimate-only` writes to stderr, not stdout."""

    def test_merges_stderr_or_the_estimate_is_lost(self):
        with mock.patch.object(lmstudio, "run_lms", return_value="Estimated GPU Memory: 1 GiB") as m:
            r = control.estimate_load("m")
        self.assertTrue(r["ok"])
        self.assertIn("Estimated GPU Memory", r["output"])
        self.assertTrue(m.call_args.kwargs.get("merge_stderr"),
                        "estimate_load must pass merge_stderr=True")

    def test_empty_output_is_an_error_not_a_silent_success(self):
        with mock.patch.object(lmstudio, "run_lms", return_value="   "):
            r = control.estimate_load("m")
        self.assertFalse(r["ok"])
        self.assertIn("no estimate", r["error"])

    def test_lms_failure_is_reported(self):
        with mock.patch.object(lmstudio, "run_lms",
                               side_effect=lmstudio.LmsError("boom")):
            r = control.estimate_load("m")
        self.assertFalse(r["ok"])
        self.assertIn("boom", r["error"])


class TestJobMap(unittest.TestCase):
    def setUp(self):
        control.clear_all_jobs()

    def test_starts_empty(self):
        self.assertEqual(control.get_jobs(), {})

    def test_private_keys_are_not_exposed(self):
        control._set_job("x", {"kind": "load", "done": True, "_secret": 1})
        self.assertNotIn("_secret", control.get_jobs()["x"])

    def test_clear_finished_keeps_running_jobs(self):
        control._set_job("done", {"kind": "load", "done": True})
        control._set_job("busy", {"kind": "load", "done": False})
        control.clear_finished_jobs()
        self.assertEqual(list(control.get_jobs()), ["busy"])

    def test_duplicate_start_is_refused_while_running(self):
        control._set_job("m", {"kind": "load", "done": False})
        self.assertFalse(control._claim_job("m", "load"))

    def test_restart_allowed_once_finished(self):
        control._set_job("m", {"kind": "load", "done": True})
        self.assertTrue(control._claim_job("m", "load"))


if __name__ == "__main__":
    unittest.main()
