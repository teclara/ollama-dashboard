"""Ollama journald reading, GIN access-log parsing, and windowed aggregation.

Ollama logs to journald, not to files, so the byte-tailing and file-rollover
machinery this module used for LM Studio is gone. What replaces it is richer:
GIN access lines carry an HTTP status, a latency, and a client address, none of
which LM Studio's logs contained. Percentiles, error rates, and per-client
breakdowns are therefore possible here for the first time.

What is NOT available is the model name. GIN lines do not carry one, and the
only place the journal names a model is by weights-blob SHA, which does not
match the manifest digest in /api/tags. Model attribution is done instead by
samplers.MODEL_TIMELINE and is inferred, not observed.
"""
import math, re, subprocess, threading, time
from collections import defaultdict, deque
from datetime import datetime

from config import (
    JOURNAL_BACKFILL, JOURNAL_UNIT, LOG_WINDOW_LINES, NOISE_PATHS,
    STATS_WINDOW_SEC,
)

LOOPBACK = {"::1", "127.0.0.1", "localhost"}

# GIN's access line, e.g.
# [GIN] 2026/07/30 - 23:25:18 | 200 |  51.336µs |  ::1 | GET "/api/ps"
_GIN_RE = re.compile(
    r"^\[GIN\]\s+(?P<ts>\d{4}/\d{2}/\d{2} - \d{2}:\d{2}:\d{2})\s*\|"
    r"\s*(?P<status>\d{3})\s*\|"
    r"\s*(?P<latency>\S+)\s*\|"
    r"\s*(?P<client>\S+)\s*\|"
    r"\s*(?P<method>[A-Z]+)\s+\"(?P<path>[^\"]*)\"")

_LEVEL_RE = re.compile(r"level=(?P<level>WARN|ERROR)\b")
_MSG_RE = re.compile(r'msg="(?P<msg>(?:\\.|[^"\\])*)"')

# Go's time.Duration.String(). Sub-second units never compound; h/m/s do, as in
# "2m49s" and "1h2m3.5s". Order matters: µs/ms/ns must be tried before the bare
# "s"/"m" so "130ms" is not read as 130 minutes.
_DUR_UNITS = (("ns", 1e-9), ("µs", 1e-6), ("us", 1e-6), ("ms", 1e-3),
              ("h", 3600.0), ("m", 60.0), ("s", 1.0))
_DUR_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)(ns|µs|us|ms|h|m|s)")


def parse_duration(s):
    """A Go duration literal -> seconds, or None.

    Must sum every component. "2m49s" is 169 seconds; matching only the first
    token would report 120 and quietly under-state every slow request.
    """
    if not s:
        return None
    tokens = _DUR_TOKEN_RE.findall(s.strip())
    if not tokens:
        return None
    # Reject trailing junk so "banana" and "12x" do not parse as partial hits.
    if "".join(a + b for a, b in tokens) != s.strip():
        return None
    total = 0.0
    for value, unit in tokens:
        total += float(value) * dict(_DUR_UNITS)[unit]
    return total


def _epoch(ts):
    try:
        return datetime.strptime(ts, "%Y/%m/%d - %H:%M:%S").timestamp()
    except Exception:
        return 0


def _row(kind, ts, **kw):
    r = {"kind": kind, "ts": ts, "epoch": _epoch(ts) if ts else time.time(),
         "status": None, "latency_s": None, "client": None, "method": None,
         "path": None, "level": None, "message": None, "model": None}
    r.update(kw)
    return r


def parse_line(line):
    """One journal line -> a row, or None if it carries nothing we track."""
    if not line:
        return None

    m = _GIN_RE.match(line)
    if m:
        return _row("request", m.group("ts"),
                    status=int(m.group("status")),
                    latency_s=parse_duration(m.group("latency")),
                    client=m.group("client"),
                    method=m.group("method"),
                    path=m.group("path"))

    m = _LEVEL_RE.search(line)
    if m:
        msg = _MSG_RE.search(line)
        return _row("problem", None, level=m.group("level"),
                    message=(msg.group("msg") if msg else line)[:300])
    return None


def is_noise(row):
    """True for the dashboard's own polling.

    Filtered on path AND loopback, never path alone: other clients hitting the
    same endpoints — the Open WebUI container polls /api/tags every few
    seconds — are real consumers and must stay visible in the client breakdown.
    """
    return (row.get("kind") == "request"
            and row.get("path") in NOISE_PATHS
            and row.get("client") in LOOPBACK)


def parse_lines(lines):
    out = []
    for line in lines:
        r = parse_line(line.rstrip("\n") if isinstance(line, str) else line)
        if r is None or is_noise(r):
            continue
        out.append(r)
    return out


# journald follower -----------------------------------------------------------
#
# One long-lived `journalctl -f`, mirroring the nvidia-smi and dmon streamers in
# samplers.py and sources.py. Spawning journalctl once a second would work but
# would re-read and re-parse the same tail on every tick.

_BUF = deque(maxlen=LOG_WINDOW_LINES)
_BUF_LOCK = threading.Lock()
_LAST_LINE_TS = [0.0]
_STOP = threading.Event()
_STARTED = threading.Event()
_PROC = [None]  # the in-flight journalctl Popen, if any; mutable cell so
                # stop_follower() can reach it from another thread.
_RETRY_DELAY_SEC = 2  # module-level so tests can shrink it for a fast respawn check


def _journal_cmd():
    return ["journalctl", "-u", JOURNAL_UNIT, "-f", "-n", str(JOURNAL_BACKFILL),
            "-o", "cat", "--no-pager"]


def _ingest(line):
    r = parse_line(line)
    if r is None or is_noise(r):
        return False
    with _BUF_LOCK:
        _BUF.append(r)
        _LAST_LINE_TS[0] = time.time()
    return True


def _follow_loop():
    while not _STOP.is_set():
        try:
            proc = subprocess.Popen(_journal_cmd(), stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True, bufsize=1)
            _PROC[0] = proc
            for line in proc.stdout:
                if _STOP.is_set():
                    break
                _ingest(line.rstrip("\n"))
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # terminate() did not land. Kill outright rather than
                # swallowing this and letting the next iteration spawn a
                # second journalctl on top of one still running, which
                # would double-ingest every line.
                proc.kill()
                proc.wait()
        except Exception:
            pass
        finally:
            _PROC[0] = None
        # journalctl exited — vacuum, rotation, or a killed process. Rebuild
        # from the backfill rather than leaving a frozen window on screen.
        _STOP.wait(_RETRY_DELAY_SEC)


def start_follower():
    """Launch the journal follower. Idempotent."""
    if _STARTED.is_set():
        return
    _STARTED.set()
    threading.Thread(target=_follow_loop, daemon=True).start()


def stop_follower():
    """Called by samplers.stop_all() on shutdown.

    Terminating the in-flight journalctl subprocess (if any) is not an
    optional nicety here: `for line in proc.stdout` in _follow_loop blocks
    on the pipe and only re-checks _STOP after a line arrives. Against a
    quiet unit that line may never come, so setting _STOP alone can never
    unblock the thread — the subprocess itself has to be killed to force
    the read to return.
    """
    _STOP.set()
    proc = _PROC[0]
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass
    _STARTED.clear()


def read_window():
    with _BUF_LOCK:
        return list(_BUF)


def follower_age():
    """Seconds since the last accepted line, or None if none ever arrived.

    The UI must surface this. A dead follower otherwise presents a frozen
    window as though it were current.
    """
    with _BUF_LOCK:
        return time.time() - _LAST_LINE_TS[0] if _LAST_LINE_TS[0] else None


# Aggregation ---------------------------------------------------------------

def _recent(rows, window_sec):
    cutoff = time.time() - window_sec
    return [r for r in rows if r.get("epoch", 0) >= cutoff]


def _requests(rows, window_sec):
    return [r for r in _recent(rows, window_sec) if r["kind"] == "request"]


def _is_error(row):
    s = row.get("status")
    return s is not None and s >= 400


def percentile(values, p):
    """Nearest-rank percentile. None for an empty sample.

    math.ceil, not round(x + 0.5): the latter hits Python's banker's rounding
    on exact halves and returns the 96th of 100 samples for p95.
    """
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[max(1, math.ceil(p / 100.0 * len(vals))) - 1]


def stats(rows, window_sec=None):
    window_sec = window_sec or STATS_WINDOW_SEC
    reqs = _requests(rows, window_sec)
    errors = [r for r in reqs if _is_error(r)]
    lat = [r["latency_s"] for r in reqs]
    return {
        "window_sec": window_sec,
        "count": len(reqs),
        "rps": round(len(reqs) / window_sec, 2) if reqs else 0,
        "error_count": len(errors),
        "error_rate": round(len(errors) / len(reqs) * 100, 1) if reqs else 0,
        "p50_s": percentile(lat, 50),
        "p95_s": percentile(lat, 95),
        "p99_s": percentile(lat, 99),
    }


def top_endpoints(rows, window_sec=None, top=8):
    window_sec = window_sec or STATS_WINDOW_SEC
    groups = defaultdict(list)
    for r in _requests(rows, window_sec):
        groups[r["path"]].append(r)
    out = [{"path": p,
            "count": len(rs),
            "errors": sum(1 for r in rs if _is_error(r)),
            "p95_s": percentile([r["latency_s"] for r in rs], 95)}
           for p, rs in groups.items()]
    return sorted(out, key=lambda x: -x["count"])[:top]


def by_client(rows, window_sec=None, top=8):
    """Per-client request counts. Impossible under LM Studio, whose logs
    carried no client address at all."""
    window_sec = window_sec or STATS_WINDOW_SEC
    groups = defaultdict(list)
    for r in _requests(rows, window_sec):
        groups[r["client"]].append(r)
    out = [{"client": c,
            "count": len(rs),
            "errors": sum(1 for r in rs if _is_error(r)),
            "last_seen": max(r["epoch"] for r in rs)}
           for c, rs in groups.items()]
    return sorted(out, key=lambda x: -x["count"])[:top]


def problems(rows, limit=10):
    probs = [r for r in rows if r["kind"] == "problem"]
    return sorted(probs, key=lambda r: -r.get("epoch", 0))[:limit]
