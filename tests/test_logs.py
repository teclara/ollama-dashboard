import unittest

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


if __name__ == "__main__":
    unittest.main()
