import json
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

import logs
from tests.helpers import fixture


class TestParseDuration(unittest.TestCase):
    def test_microseconds(self):
        self.assertAlmostEqual(logs.parse_duration("51.336µs"), 5.1336e-05)

    def test_microseconds_ascii_us_variant(self):
        self.assertAlmostEqual(logs.parse_duration("51.336us"), 5.1336e-05)

    def test_trailing_zero_stripped_form(self):
        # Go prints 15.8µs, not 15.800µs.
        self.assertAlmostEqual(logs.parse_duration("15.8µs"), 1.58e-05)

    def test_milliseconds(self):
        self.assertAlmostEqual(logs.parse_duration("130.345951ms"), 0.130345951)

    def test_seconds(self):
        self.assertAlmostEqual(logs.parse_duration("1.159649885s"), 1.159649885)

    def test_nanoseconds(self):
        self.assertAlmostEqual(logs.parse_duration("900ns"), 9e-07)

    def test_compound_minutes_and_seconds(self):
        # THE trap. A naive [0-9.]+(µs|ms|s|m) regex matches "2m" and silently
        # discards the 49s, under-reporting a 169-second request as 120.
        self.assertAlmostEqual(logs.parse_duration("2m49s"), 169.0)

    def test_compound_single_digit_seconds(self):
        self.assertAlmostEqual(logs.parse_duration("1m8s"), 68.0)

    def test_compound_with_hours(self):
        self.assertAlmostEqual(logs.parse_duration("1h2m3s"), 3723.0)

    def test_compound_with_fractional_seconds(self):
        self.assertAlmostEqual(logs.parse_duration("7m19.5s"), 439.5)

    def test_bare_minutes(self):
        self.assertAlmostEqual(logs.parse_duration("3m"), 180.0)

    def test_garbage_is_none(self):
        self.assertIsNone(logs.parse_duration(""))
        self.assertIsNone(logs.parse_duration(None))
        self.assertIsNone(logs.parse_duration("banana"))

    def test_ms_is_not_mis_parsed_as_minutes(self):
        # "130ms" must not read as 130 minutes + "s".
        self.assertAlmostEqual(logs.parse_duration("130ms"), 0.13)


class TestParseLine(unittest.TestCase):
    LINE = ('[GIN] 2026/07/30 - 23:25:18 | 200 |      51.336µs '
            '|             ::1 | GET      "/api/ps"')

    def test_parses_every_field(self):
        r = logs.parse_line(self.LINE)
        self.assertEqual(r["kind"], "request")
        self.assertEqual(r["ts"], "2026/07/30 - 23:25:18")
        self.assertEqual(r["status"], 200)
        self.assertAlmostEqual(r["latency_s"], 5.1336e-05)
        self.assertEqual(r["client"], "::1")
        self.assertEqual(r["method"], "GET")
        self.assertEqual(r["path"], "/api/ps")

    def test_epoch_is_populated(self):
        self.assertGreater(logs.parse_line(self.LINE)["epoch"], 0)

    def test_ipv4_client(self):
        line = ('[GIN] 2026/07/30 - 23:25:23 | 200 |     280.772µs '
                '|      172.17.0.3 | GET      "/api/tags"')
        self.assertEqual(logs.parse_line(line)["client"], "172.17.0.3")

    def test_error_status(self):
        line = ('[GIN] 2026/07/30 - 23:00:21 | 401 |  130.345951ms '
                '|       127.0.0.1 | POST     "/api/me"')
        r = logs.parse_line(line)
        self.assertEqual(r["status"], 401)
        self.assertEqual(r["path"], "/api/me")

    def test_compound_latency_in_a_real_line(self):
        line = ('[GIN] 2026/07/30 - 23:14:51 | 200 |         7m19s '
                '|       127.0.0.1 | POST     "/api/pull"')
        self.assertAlmostEqual(logs.parse_line(line)["latency_s"], 439.0)

    def test_head_method(self):
        line = ('[GIN] 2026/07/30 - 23:00:00 | 200 |      12.012µs '
                '|       127.0.0.1 | HEAD     "/"')
        r = logs.parse_line(line)
        self.assertEqual(r["method"], "HEAD")
        self.assertEqual(r["path"], "/")

    def test_query_string_is_kept_on_path(self):
        line = ('[GIN] 2026/07/30 - 23:00:00 | 200 |      12.012µs '
                '|       127.0.0.1 | POST     "/v1/messages?beta=true"')
        self.assertEqual(logs.parse_line(line)["path"], "/v1/messages?beta=true")

    def test_error_level_line(self):
        line = ('time=2026-07-30T23:25:33.825-04:00 level=ERROR '
                'source=sched.go:1 msg="something broke"')
        r = logs.parse_line(line)
        self.assertEqual(r["kind"], "problem")
        self.assertEqual(r["level"], "ERROR")
        self.assertIn("something broke", r["message"])

    def test_uninteresting_line_is_none(self):
        self.assertIsNone(logs.parse_line("load_tensors: offloaded 41/41 layers"))
        self.assertIsNone(logs.parse_line(""))
        self.assertIsNone(logs.parse_line(None))


class TestIsNoise(unittest.TestCase):
    def test_dashboard_polling_from_loopback_is_noise(self):
        r = logs.parse_line('[GIN] 2026/07/30 - 23:25:18 | 200 |      51.336µs '
                            '|             ::1 | GET      "/api/ps"')
        self.assertTrue(logs.is_noise(r))

    def test_same_path_from_a_remote_client_is_real_traffic(self):
        # Open WebUI at 172.17.0.3 polls /api/tags too. Filtering by path alone
        # would erase a real consumer from the client breakdown.
        r = logs.parse_line('[GIN] 2026/07/30 - 23:25:23 | 200 |     280.772µs '
                            '|      172.17.0.3 | GET      "/api/tags"')
        self.assertFalse(logs.is_noise(r))

    def test_inference_from_loopback_is_not_noise(self):
        r = logs.parse_line('[GIN] 2026/07/30 - 23:06:17 | 200 |          1m8s '
                            '|       127.0.0.1 | POST     "/api/chat"')
        self.assertFalse(logs.is_noise(r))


class TestParseLines(unittest.TestCase):
    def test_parses_the_real_journal_excerpt(self):
        rows = logs.parse_lines(fixture("journal_excerpt.log").splitlines())
        self.assertTrue(rows)
        self.assertTrue(all(r["kind"] in ("request", "problem") for r in rows))

    def test_every_request_row_has_a_latency(self):
        rows = logs.parse_lines(fixture("journal_excerpt.log").splitlines())
        reqs = [r for r in rows if r["kind"] == "request"]
        self.assertTrue(reqs)
        self.assertTrue(all(r["latency_s"] is not None for r in reqs),
                        "a latency literal failed to parse")

    def test_noise_is_dropped(self):
        rows = logs.parse_lines(fixture("journal_excerpt.log").splitlines())
        loopback_polls = [r for r in rows if r["kind"] == "request"
                          and r["path"] in ("/api/ps", "/api/version")
                          and r["client"] in ("::1", "127.0.0.1")]
        self.assertEqual(loopback_polls, [])


class TestPercentile(unittest.TestCase):
    def test_median_of_odd_length(self):
        self.assertAlmostEqual(logs.percentile([1, 2, 3], 50), 2.0)

    def test_p95_picks_the_tail(self):
        self.assertAlmostEqual(logs.percentile(list(range(1, 101)), 95), 95.0)

    def test_single_value(self):
        self.assertAlmostEqual(logs.percentile([7.5], 95), 7.5)

    def test_empty_is_none(self):
        self.assertIsNone(logs.percentile([], 50))

    def test_ignores_none_entries(self):
        # A request whose latency failed to parse must not be counted as 0.
        self.assertAlmostEqual(logs.percentile([5, None, 5], 50), 5.0)
        self.assertAlmostEqual(logs.percentile([None, None, 7], 95), 7.0)


def _req(epoch, status=200, latency=0.01, client="1.2.3.4", path="/api/chat"):
    return {"kind": "request", "ts": "", "epoch": epoch, "status": status,
            "latency_s": latency, "client": client, "method": "POST",
            "path": path, "level": None, "message": None, "model": None}


class TestStats(unittest.TestCase):
    def setUp(self):
        self.now = time.time()

    def test_counts_and_rate(self):
        rows = [_req(self.now - i) for i in range(10)]
        s = logs.stats(rows, window_sec=100)
        self.assertEqual(s["count"], 10)
        self.assertAlmostEqual(s["rps"], 0.1)

    def test_error_rate_counts_non_2xx(self):
        rows = [_req(self.now, status=200), _req(self.now, status=200),
                _req(self.now, status=404), _req(self.now, status=401)]
        s = logs.stats(rows, window_sec=100)
        self.assertEqual(s["error_count"], 2)
        self.assertAlmostEqual(s["error_rate"], 50.0)

    def test_3xx_is_not_an_error(self):
        s = logs.stats([_req(self.now, status=304)], window_sec=100)
        self.assertEqual(s["error_count"], 0)

    def test_percentiles_reported_in_seconds(self):
        rows = [_req(self.now, latency=v / 1000.0) for v in range(1, 101)]
        s = logs.stats(rows, window_sec=100)
        self.assertAlmostEqual(s["p50_s"], 0.050, places=3)
        self.assertAlmostEqual(s["p95_s"], 0.095, places=3)

    def test_rows_outside_the_window_are_excluded(self):
        rows = [_req(self.now), _req(self.now - 9999)]
        self.assertEqual(logs.stats(rows, window_sec=60)["count"], 1)

    def test_empty_window_is_all_zeroes_not_none(self):
        s = logs.stats([], window_sec=60)
        self.assertEqual(s["count"], 0)
        self.assertEqual(s["rps"], 0)
        self.assertEqual(s["error_rate"], 0)
        self.assertIsNone(s["p95_s"])


class TestTopEndpoints(unittest.TestCase):
    def test_ranks_by_count_with_errors_and_p95(self):
        now = time.time()
        rows = ([_req(now, path="/api/chat", latency=1.0)] * 3
                + [_req(now, path="/api/show", status=404, latency=0.002)])
        out = logs.top_endpoints(rows, window_sec=100)
        self.assertEqual(out[0]["path"], "/api/chat")
        self.assertEqual(out[0]["count"], 3)
        self.assertEqual(out[0]["errors"], 0)
        self.assertAlmostEqual(out[0]["p95_s"], 1.0)
        self.assertEqual(out[1]["errors"], 1)


class TestByClient(unittest.TestCase):
    def test_groups_by_address(self):
        now = time.time()
        rows = ([_req(now, client="172.17.0.3")] * 4
                + [_req(now, client="127.0.0.1", status=500)])
        out = logs.by_client(rows, window_sec=100)
        self.assertEqual(out[0]["client"], "172.17.0.3")
        self.assertEqual(out[0]["count"], 4)
        self.assertEqual(out[1]["errors"], 1)

    def test_last_seen_is_the_newest_epoch(self):
        now = time.time()
        rows = [_req(now - 50, client="a"), _req(now - 5, client="a")]
        self.assertAlmostEqual(logs.by_client(rows, window_sec=100)[0]["last_seen"],
                               now - 5)


class TestProblems(unittest.TestCase):
    def test_returns_newest_first_and_limits(self):
        rows = [{"kind": "problem", "epoch": i, "level": "ERROR",
                 "message": f"m{i}", "ts": "", "status": None, "latency_s": None,
                 "client": None, "method": None, "path": None, "model": None}
                for i in range(5)]
        out = logs.problems(rows, limit=2)
        self.assertEqual([p["message"] for p in out], ["m4", "m3"])


class TestFollower(unittest.TestCase):
    def setUp(self):
        logs._CURSOR[0] = None

    def tearDown(self):
        logs._CURSOR[0] = None

    def test_read_window_is_empty_before_the_follower_starts(self):
        self.assertEqual(logs.read_window(), [])

    def test_ingest_appends_parsed_rows(self):
        logs._BUF.clear()
        logs._ingest('[GIN] 2026/07/30 - 23:06:17 | 200 |          1m8s '
                     '|      172.17.0.3 | POST     "/api/chat"')
        rows = logs.read_window()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["latency_s"], 68.0)
        logs._BUF.clear()

    def test_ingest_drops_noise(self):
        logs._BUF.clear()
        logs._ingest('[GIN] 2026/07/30 - 23:25:18 | 200 |      51.336µs '
                     '|             ::1 | GET      "/api/ps"')
        self.assertEqual(logs.read_window(), [])

    def test_buffer_is_bounded(self):
        logs._BUF.clear()
        line = ('[GIN] 2026/07/30 - 23:06:17 | 200 |      51.336µs '
                '|      172.17.0.3 | POST     "/api/chat"')
        for _ in range(logs.LOG_WINDOW_LINES + 500):
            logs._ingest(line)
        self.assertEqual(len(logs.read_window()), logs.LOG_WINDOW_LINES)
        logs._BUF.clear()

    def test_journalctl_command_follows_the_configured_unit(self):
        cmd = logs._journal_cmd()
        self.assertIn("journalctl", cmd[0])
        self.assertIn("-u", cmd)
        self.assertIn(logs.JOURNAL_UNIT, cmd)
        self.assertIn("-f", cmd)


class TestFollowerThreading(unittest.TestCase):
    """Covers the two riskiest surfaces _follow_loop introduces: a thread
    parked in a blocking pipe read, and respawn after the subprocess exits.
    Both fake the subprocess via _journal_cmd, so neither depends on the
    real journalctl or on multi-second sleeps beyond a bounded join."""

    def tearDown(self):
        logs.stop_follower()
        logs._STOP.clear()
        logs._STARTED.clear()
        logs._BUF.clear()
        logs._LAST_LINE_TS[0] = 0.0
        logs._PROC[0] = None
        logs._CURSOR[0] = None

    def test_stop_follower_unblocks_a_thread_parked_in_the_blocking_read(self):
        # A subprocess that stays alive and emits nothing, exactly like an
        # idle `journalctl -f`. Before the fix, stop_follower() only set
        # _STOP and never touched the subprocess, so the blocking
        # `for line in proc.stdout` read never noticed and the thread
        # never joined.
        quiet_cmd = [sys.executable, "-c", "import time; time.sleep(5)"]
        with mock.patch.object(logs, "_journal_cmd", lambda: quiet_cmd):
            t = threading.Thread(target=logs._follow_loop, daemon=True)
            t.start()
            time.sleep(0.2)  # let Popen spawn and block on the pipe read
            logs.stop_follower()
            t.join(timeout=2)
            self.assertFalse(t.is_alive())

    def test_cursor_resume_does_not_request_the_backfill_again(self):
        logs._CURSOR[0] = "s=cursor"
        cmd = logs._journal_cmd()
        self.assertIn("--after-cursor=s=cursor", cmd)
        self.assertNotIn("-n", cmd)

    def test_json_output_records_the_cursor_and_message_once(self):
        message = ('[GIN] 2026/07/30 - 23:06:17 | 200 |      51.336µs '
                   '|      172.17.0.3 | POST     "/api/chat"')

        class FakeProc:
            def __init__(self):
                self.stdout = self
                self.sent = False

            def __iter__(self): return self

            def __next__(self):
                if self.sent:
                    logs._STOP.set()
                    raise StopIteration
                self.sent = True
                return json.dumps({"MESSAGE": message, "__CURSOR": "s=next"}) + "\n"

            def terminate(self): pass
            def wait(self, timeout=None): return 0
            def poll(self): return None

        with mock.patch.object(logs.subprocess, "Popen", return_value=FakeProc()):
            logs._follow_loop()
        self.assertEqual(logs._CURSOR[0], "s=next")
        self.assertEqual(len(logs.read_window()), 1)


if __name__ == "__main__":
    unittest.main()


class TestFollowerLiveness(unittest.TestCase):
    """Liveness and freshness are different questions. Conflating them made an
    idle server look like a dead follower and fired a false staleness alarm."""

    def setUp(self):
        logs._BUF.clear()
        logs._LAST_LINE_TS[0] = 0.0
        logs._LAST_READ_TS[0] = 0.0
        logs._PROC[0] = None

    tearDown = setUp

    def test_alive_is_false_with_no_subprocess(self):
        self.assertFalse(logs.follower_alive())

    def test_alive_is_true_while_the_subprocess_runs(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        logs._PROC[0] = proc
        try:
            self.assertTrue(logs.follower_alive())
        finally:
            proc.kill(); proc.wait()

    def test_alive_is_false_once_the_subprocess_exits(self):
        proc = subprocess.Popen([sys.executable, "-c", ""])
        proc.wait()
        logs._PROC[0] = proc
        self.assertFalse(logs.follower_alive())

    def test_noise_only_traffic_does_not_look_like_a_dead_follower(self):
        # THE regression. The dashboard's own loopback polling is filtered out,
        # so it never reaches _LAST_LINE_TS. Reading it must still count as
        # liveness, or a healthy follower on an idle server reads as broken.
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        logs._PROC[0] = proc
        try:
            noise = ('[GIN] 2026/07/30 - 23:25:18 | 200 |      51.336µs '
                     '|             ::1 | GET      "/api/ps"')
            self.assertTrue(logs.is_noise(logs.parse_line(noise)))
            with logs._BUF_LOCK:
                logs._LAST_READ_TS[0] = time.time()
            logs._ingest(noise)
            self.assertEqual(logs.read_window(), [])      # correctly filtered
            self.assertIsNone(logs.follower_age())        # nothing accepted yet
            self.assertLess(logs.follower_read_age(), 5)  # but we DID read
            self.assertTrue(logs.follower_alive())        # and we are healthy
        finally:
            proc.kill(); proc.wait()

    def test_accepted_row_updates_freshness(self):
        line = ('[GIN] 2026/07/30 - 23:06:17 | 200 |          1m8s '
                '|      172.17.0.3 | POST     "/api/chat"')
        event_time = logs.parse_line(line)["epoch"]
        with mock.patch.object(logs.time, "time", return_value=event_time + 5):
            logs._ingest(line)
            self.assertEqual(logs.follower_age(), 5)
