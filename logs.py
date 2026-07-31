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


# Aggregation ---------------------------------------------------------------

def _recent(rows, window_sec):
    cutoff = time.time() - window_sec
    return [r for r in rows if r["epoch"] >= cutoff]


def stats(rows, window_sec=None):
    window_sec = window_sec or STATS_WINDOW_SEC
    recent = [r for r in _recent(rows, window_sec) if r["kind"] == "request"]
    return {"window_sec": window_sec, "count": len(recent),
            "rps": round(len(recent) / window_sec, 2) if recent else 0}


def top_endpoints(rows, window_sec=None, top=8):
    window_sec = window_sec or STATS_WINDOW_SEC
    counts = defaultdict(int)
    for r in _recent(rows, window_sec):
        if r["kind"] == "request": counts[r["path"]] += 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [{"path": p, "count": c} for p, c in ranked[:top]]


_ACTIVITY_KEYS = {"completion": "completions", "prediction": "predictions",
                  "tool_calls": "tool_calls", "stream_end": "streams"}


def model_activity(rows, window_sec=None):
    window_sec = window_sec or STATS_WINDOW_SEC
    by_model = defaultdict(lambda: {"completions": 0, "predictions": 0,
                                    "tool_calls": 0, "streams": 0})
    for r in _recent(rows, window_sec):
        key = _ACTIVITY_KEYS.get(r["kind"])
        if key and r["model"]: by_model[r["model"]][key] += 1
    return sorted([{"model": m, **v} for m, v in by_model.items()],
                  key=lambda x: -x["completions"])
