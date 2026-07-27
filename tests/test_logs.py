import os, tempfile, time, unittest

import logs
from tests.helpers import fixture


class TestParseLine(unittest.TestCase):
    def test_request_line(self):
        r = logs.parse_line(
            "[2026-07-25 22:01:27][DEBUG] Received request: POST to /v1/chat/completions with body {")
        self.assertEqual(r["kind"], "request")
        self.assertEqual(r["method"], "POST")
        self.assertEqual(r["path"], "/v1/chat/completions")
        self.assertEqual(r["ts"], "2026-07-25 22:01:27")
        self.assertGreater(r["epoch"], 0)

    def test_request_line_without_body(self):
        r = logs.parse_line("[2026-07-26 22:24:10][DEBUG] Received request: GET to /lmstudio-greeting")
        self.assertEqual(r["method"], "GET")
        self.assertEqual(r["path"], "/lmstudio-greeting")

    def test_completion_line_captures_model_and_message_count(self):
        r = logs.parse_line(
            "[2026-07-25 22:01:28][INFO][qwen/qwen3.6-35b-a3b] "
            "Running chat completion on conversation with 2 messages.")
        self.assertEqual(r["kind"], "completion")
        self.assertEqual(r["model"], "qwen/qwen3.6-35b-a3b")
        self.assertEqual(r["messages"], 2)

    def test_custom_identifier_is_treated_as_the_model(self):
        # `lms load --identifier cpal` makes the bracket tag a custom name
        r = logs.parse_line("[2026-07-26 07:39:32][INFO][cpal] Model generated tool calls: []")
        self.assertEqual(r["kind"], "tool_calls")
        self.assertEqual(r["model"], "cpal")

    def test_stream_start_and_end(self):
        self.assertEqual(
            logs.parse_line("[2026-07-25 22:01:28][INFO][m] Streaming response...")["kind"],
            "stream_start")
        self.assertEqual(
            logs.parse_line("[2026-07-25 22:01:35][INFO][m] Finished streaming response")["kind"],
            "stream_end")

    def test_prediction_line(self):
        r = logs.parse_line("[2026-07-26 00:11:53][INFO][fsec] Generated prediction: {")
        self.assertEqual(r["kind"], "prediction")
        self.assertEqual(r["model"], "fsec")

    def test_authenticator_lines_are_not_models(self):
        r = logs.parse_line(
            "[2026-07-25 22:53:29][INFO][LMSAuthenticator][Client=lms-cli][Endpoint=listLoaded] "
            "Listing loaded models")
        self.assertIsNone(r)

    def test_progress_lines_ignored(self):
        self.assertIsNone(logs.parse_line(
            "[2026-07-25 22:01:33][INFO][m] Prompt processing progress: 6.2%"))

    def test_json_body_continuation_ignored(self):
        self.assertIsNone(logs.parse_line('      "temperature": 0.7,'))
        self.assertIsNone(logs.parse_line("    {"))
        self.assertIsNone(logs.parse_line(""))


class TestParseLines(unittest.TestCase):
    def test_parses_the_real_excerpt(self):
        rows = logs.parse_lines(fixture("server_log_excerpt.log").splitlines())
        kinds = {r["kind"] for r in rows}
        self.assertEqual(kinds, {"request", "completion", "stream_start",
                                 "stream_end", "prediction", "tool_calls"})

    def test_noise_paths_dropped(self):
        rows = logs.parse_lines([
            "[2026-07-25 22:01:27][DEBUG] Received request: GET to /api/v0/models",
            "[2026-07-25 22:01:27][DEBUG] Received request: GET to /lmstudio-greeting",
            "[2026-07-25 22:01:27][DEBUG] Received request: POST to /v1/chat/completions",
        ])
        self.assertEqual([r["path"] for r in rows], ["/v1/chat/completions"])


class TestLogFiles(unittest.TestCase):
    def test_newest_first_across_month_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            for sub, name, mtime in (("2026-06", "2026-06-30.1.log", 1000),
                                     ("2026-07", "2026-07-01.1.log", 2000),
                                     ("2026-07", "2026-07-02.1.log", 3000)):
                os.makedirs(os.path.join(d, sub), exist_ok=True)
                p = os.path.join(d, sub, name)
                open(p, "w").close()
                os.utime(p, (mtime, mtime))
            found = [os.path.basename(p) for p in logs.log_files(d)]
            self.assertEqual(found, ["2026-07-02.1.log", "2026-07-01.1.log", "2026-06-30.1.log"])

    def test_missing_dir_yields_empty(self):
        self.assertEqual(logs.log_files("/nonexistent/path/xyz"), [])


class TestTailLines(unittest.TestCase):
    def test_returns_only_the_tail(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.log")
            with open(p, "w") as f:
                f.write("\n".join(f"line{i}" for i in range(1000)) + "\n")
            got = logs.tail_lines(p, 40)
            self.assertLess(len(got), 10)
            self.assertEqual(got[-1], "line999")

    def test_drops_the_partial_leading_line(self):
        """A byte-offset read lands mid-line; that fragment must be discarded."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.log")
            with open(p, "w") as f:
                f.write("AAAAAAAAAA\nBBBBBBBBBB\nCCCCCCCCCC\n")
            got = logs.tail_lines(p, 16)
            self.assertNotIn("AAAAAAAAAA", got)
            self.assertEqual(got[-1], "CCCCCCCCCC")

    def test_whole_file_when_smaller_than_budget(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.log")
            with open(p, "w") as f: f.write("one\ntwo\n")
            self.assertEqual(logs.tail_lines(p, 1 << 20), ["one", "two"])

    def test_missing_file_yields_empty(self):
        self.assertEqual(logs.tail_lines("/nonexistent/x.log", 1024), [])


class TestReadWindow(unittest.TestCase):
    def test_spans_two_files_when_newest_does_not_cover_the_window(self):
        """A window reaching past the newest file must walk into the previous
        day's file, so the daily rollover does not truncate it."""
        now = time.time()
        recent = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - 30))
        older = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - 60))
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "2026-07"))
            old = os.path.join(d, "2026-07", "2026-07-25.1.log")
            new = os.path.join(d, "2026-07", "2026-07-26.1.log")
            # Both entries sit inside the window, so the newest file alone
            # cannot cover it and the reader must continue into the older one.
            with open(old, "w") as f:
                f.write(f"[{older}][DEBUG] Received request: POST to /v1/embeddings\n")
            with open(new, "w") as f:
                f.write(f"[{recent}][DEBUG] Received request: POST to /v1/chat/completions\n")
            os.utime(old, (1000, 1000))
            os.utime(new, (2000, 2000))
            rows = logs.read_window(d, window_sec=300)
            self.assertEqual([r["path"] for r in rows],
                             ["/v1/embeddings", "/v1/chat/completions"])

    def test_stops_early_once_the_window_is_covered(self):
        """If the newest file already reaches past the cutoff, older files are
        not read at all — that is what keeps the per-poll I/O bounded."""
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "2026-07"))
            old = os.path.join(d, "2026-07", "2026-07-25.1.log")
            new = os.path.join(d, "2026-07", "2026-07-26.1.log")
            with open(old, "w") as f:
                f.write("[2026-07-25 10:00:00][DEBUG] Received request: POST to /v1/embeddings\n")
            with open(new, "w") as f:
                f.write("[2026-07-26 10:00:00][DEBUG] Received request: POST to /v1/chat/completions\n")
            os.utime(old, (1000, 1000)); os.utime(new, (2000, 2000))
            rows = logs.read_window(d, window_sec=300)
            self.assertEqual([r["path"] for r in rows], ["/v1/chat/completions"])

    def test_recent_events_survive_a_flood_of_body_continuation_lines(self):
        """The regression that motivated byte-based windowing: LM Studio logs
        full request bodies, so a line-count window fills with JSON and pushes
        every real event out."""
        now = time.time()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - 10))
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "2026-07"))
            p = os.path.join(d, "2026-07", "2026-07-26.1.log")
            with open(p, "w") as f:
                f.write(f"[{stamp}][DEBUG] Received request: POST to /v1/chat/completions\n")
                # the kind of body spam that swamped the old line budget
                f.write("".join('      "key": "value",\n' for _ in range(5000)))
            rows = logs.read_window(d)
            self.assertEqual([r["path"] for r in rows], ["/v1/chat/completions"])
            self.assertEqual(logs.stats(rows)["count"], 1)

    def test_caps_returned_rows(self):
        now = time.time()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - 5))
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "2026-07"))
            p = os.path.join(d, "2026-07", "a.log")
            with open(p, "w") as f:
                for _ in range(50):
                    f.write(f"[{stamp}][DEBUG] Received request: POST to /v1/chat/completions\n")
            self.assertEqual(len(logs.read_window(d, max_rows=10)), 10)

    def test_missing_dir_yields_empty(self):
        self.assertEqual(logs.read_window("/nonexistent/xyz"), [])


def _rows(now, *specs):
    """specs: (age_seconds, kind, path_or_model)"""
    out = []
    for age, kind, val in specs:
        r = {"ts": "", "epoch": now - age, "kind": kind, "method": "POST",
             "path": None, "model": None, "messages": None}
        if kind == "request": r["path"] = val
        else: r["model"] = val
        out.append(r)
    return out


class TestAggregation(unittest.TestCase):
    def test_stats_counts_only_the_window(self):
        now = time.time()
        rows = _rows(now, (10, "request", "/a"), (20, "request", "/a"), (9999, "request", "/a"))
        s = logs.stats(rows, window_sec=300)
        self.assertEqual(s["count"], 2)
        self.assertEqual(s["window_sec"], 300)
        self.assertAlmostEqual(s["rps"], round(2 / 300, 2))

    def test_stats_empty(self):
        self.assertEqual(logs.stats([], window_sec=300),
                         {"window_sec": 300, "count": 0, "rps": 0})

    def test_top_endpoints_ranked(self):
        now = time.time()
        rows = _rows(now, (1, "request", "/a"), (2, "request", "/a"), (3, "request", "/b"))
        self.assertEqual(logs.top_endpoints(rows, window_sec=300),
                         [{"path": "/a", "count": 2}, {"path": "/b", "count": 1}])

    def test_model_activity_counts_each_kind(self):
        now = time.time()
        rows = _rows(now, (1, "completion", "m1"), (2, "completion", "m1"),
                     (3, "prediction", "m1"), (4, "tool_calls", "m1"),
                     (5, "stream_end", "m1"), (6, "completion", "m2"))
        acts = {a["model"]: a for a in logs.model_activity(rows, window_sec=300)}
        self.assertEqual(acts["m1"]["completions"], 2)
        self.assertEqual(acts["m1"]["predictions"], 1)
        self.assertEqual(acts["m1"]["tool_calls"], 1)
        self.assertEqual(acts["m1"]["streams"], 1)
        self.assertEqual(acts["m2"]["completions"], 1)

    def test_model_activity_sorted_by_completions_desc(self):
        now = time.time()
        rows = _rows(now, (1, "completion", "low"),
                     (2, "completion", "high"), (3, "completion", "high"))
        self.assertEqual([a["model"] for a in logs.model_activity(rows, window_sec=300)],
                         ["high", "low"])


if __name__ == "__main__":
    unittest.main()
