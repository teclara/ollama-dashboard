"""LM Studio server-log discovery, parsing, and windowed aggregation.

LM Studio's logs carry far less than Ollama's GIN access logs: there is no
HTTP status, no latency, and no client IP anywhere in them. So there are no
percentile, error-rate, or per-client aggregates here — that data does not exist.
What the logs do give is request paths and per-model inference events.
"""
import glob, os, re, time
from collections import defaultdict
from datetime import datetime

from config import (
    LOG_DIR, LOG_TAIL_BYTES, LOG_WINDOW_LINES, NOISE_PATHS, STATS_WINDOW_SEC,
)

_TS = r"\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]"

_REQUEST_RE = re.compile(
    _TS + r"\[\w+\] Received request: (?P<method>[A-Z]+) to (?P<path>\S+)")

# Model events. The bracket tag is the *loaded instance identifier*, which may be
# a custom name from `lms load --identifier`, not the model key. Non-model lines
# (e.g. LMSAuthenticator) occupy the same slot, so the event suffix does the
# discriminating — never the bracket position.
_MODEL_RE = re.compile(
    _TS + r"\[\w+\]\[(?P<model>[^\]]+)\] (?P<event>.+)$")

_EVENTS = (
    (re.compile(r"^Running chat completion on conversation with (\d+) messages"), "completion"),
    (re.compile(r"^Streaming response"), "stream_start"),
    (re.compile(r"^Finished streaming response"), "stream_end"),
    (re.compile(r"^Generated prediction"), "prediction"),
    (re.compile(r"^Model generated tool calls"), "tool_calls"),
)


def _epoch(ts):
    try: return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception: return 0


def _row(ts, kind, **kw):
    r = {"ts": ts, "epoch": _epoch(ts), "kind": kind,
         "method": None, "path": None, "model": None, "messages": None}
    r.update(kw)
    return r


def parse_line(line):
    """One log line -> a row, or None if it carries nothing we track."""
    if not line or not line.startswith("["): return None

    m = _REQUEST_RE.match(line)
    if m:
        return _row(m.group("ts"), "request",
                    method=m.group("method"), path=m.group("path"))

    m = _MODEL_RE.match(line)
    if m:
        model, event = m.group("model"), m.group("event")
        for rx, kind in _EVENTS:
            hit = rx.match(event)
            if hit:
                msgs = int(hit.group(1)) if kind == "completion" else None
                return _row(m.group("ts"), kind, model=model, messages=msgs)
    return None


def parse_lines(lines):
    out = []
    for line in lines:
        r = parse_line(line.rstrip("\n"))
        if r is None: continue
        if r["kind"] == "request" and r["path"] in NOISE_PATHS: continue
        out.append(r)
    return out


def log_files(log_dir=None):
    """All log files, newest-modified first."""
    log_dir = log_dir or LOG_DIR
    try:
        found = glob.glob(os.path.join(log_dir, "*", "*.log"))
        return sorted(found, key=os.path.getmtime, reverse=True)
    except Exception:
        return []


def tail_lines(path, nbytes):
    """Last `nbytes` of a file as lines, dropping any partial leading line."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - nbytes)
            f.seek(start)
            data = f.read()
    except Exception:
        return []
    text = data.decode("utf-8", errors="replace")
    if start and "\n" in text:
        text = text.split("\n", 1)[1]
    return text.splitlines()


def read_window(log_dir=None, window_sec=None, tail_bytes=None, max_rows=None):
    """Parsed rows covering at least the last `window_sec`, oldest first.

    Reading a fixed number of *lines* does not work here. LM Studio logs full
    request bodies at DEBUG, so a single chat completion emits hundreds of
    untimestamped JSON continuation lines — measured on this machine, 528 of
    any 600 consecutive lines were body continuations, and 600 lines spanned
    only 11 seconds. A line budget large enough for a 5-minute window would be
    unbounded. So read by bytes from the tail instead and stop once the parsed
    rows actually reach back past the cutoff, walking into older files so the
    daily rollover does not truncate the window.
    """
    window_sec = window_sec or STATS_WINDOW_SEC
    tail_bytes = tail_bytes or LOG_TAIL_BYTES
    max_rows = max_rows or LOG_WINDOW_LINES
    cutoff = time.time() - window_sec

    rows = []
    for path in log_files(log_dir):
        rows = parse_lines(tail_lines(path, tail_bytes)) + rows
        if rows and rows[0]["epoch"] and rows[0]["epoch"] <= cutoff:
            break  # window covered
        if len(rows) >= max_rows:
            break
    return rows[-max_rows:]


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
