"""Background sampling layer.

The request path never touches Ollama. Under LM Studio the reason was latency:
every `lms` invocation cost ~200ms of Node startup. Ollama's HTTP API is orders
of magnitude faster, so that argument no longer applies — but two others do,
and they are why this layer survived the port:

1. Polling from the request path would flood the GIN access log, which is
   exactly the data the dashboard exists to display. Every dashboard poll would
   appear as a logged, timed, client-attributed request, crowding out the real
   traffic. logs.is_noise filters what does leak through, but not generating it
   is better.
2. `du -sb` over a 60+ GB model store must never block a response.

Each source is sampled by a background thread on its own cadence and the latest
value is held in memory, so state() and live() are effectively free.

GPU samples come from a single long-lived `nvidia-smi --loop-ms` process rather
than one spawn per tick.
"""
import subprocess, threading, time
from collections import deque

import logs
import ollama
import sources
from config import (
    GPU_SAMPLE_MS, HOST_SAMPLE_MS, LOADED_SAMPLE_SEC, LOGS_SAMPLE_SEC,
    PS_TIMELINE_LEN, SLOW_SAMPLE_SEC,
)


class Sampled:
    """Thread-safe holder for one source's most recent value.

    Reads never compute. Before the first background tick callers receive the
    configured default, keeping Ollama and subprocess work off request threads.
    """

    def __init__(self, fn, default=None):
        self._fn = fn
        self._value = default
        self._has = False
        self._ts = 0.0
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()

    def set(self, value):
        with self._lock:
            self._value = value
            self._has = True
            self._ts = time.time()

    def update(self, fn):
        """Atomically transform the latest value after any refresh completes."""
        with self._refresh_lock:
            with self._lock:
                self._value = fn(self._value)
                self._has = True
                self._ts = time.time()
                return self._value

    def peek(self):
        """Latest value without ever computing. (None, False) if never sampled."""
        with self._lock:
            return self._value, self._has

    def get(self):
        return self.peek()[0]

    def refresh(self):
        with self._refresh_lock:
            try:
                self.set(self._fn())
            except Exception:
                # A failing source must never kill its thread or the whole payload;
                # the last good value (or the default) stands.
                if not self.peek()[1]:
                    self.set(self._value)

    def age(self):
        with self._lock:
            return time.time() - self._ts if self._has else None


def _loop(holder, interval_sec, stop):
    while not stop.is_set():
        holder.refresh()
        stop.wait(interval_sec)


def _dependent_loop(holder, dependencies, interval_sec, stop):
    """Sample only after prerequisite holders have completed their first tick."""
    while not stop.is_set():
        if all(dependency.peek()[1] for dependency in dependencies):
            holder.refresh()
            stop.wait(interval_sec)
        else:
            stop.wait(0.1)


# Sources, grouped by how fast they actually change --------------------------

HOST = Sampled(sources.host, {})
LOGS = Sampled(logs.read_window, [])
LOADED = Sampled(ollama.loaded_models, [])
GPU_PROCS = Sampled(sources.gpu_processes, [])
GPU_VERSIONS = Sampled(sources.nvidia_versions, {})
LIBRARY = Sampled(lambda: ollama.library(LOADED.get()), [])
SETTINGS = Sampled(sources.settings, {})
DISK = Sampled(lambda: sources.disk(sources.models_root(SETTINGS.get())), {})
SERVICE = Sampled(sources.service_info, {})
TAILSCALE = Sampled(sources.tailscale, {})
PING = Sampled(ollama.ping, False)

# GPU is streamed rather than polled, so it gets a plain holder.
GPU = Sampled(sources.gpu, {"error": "GPU data has not been sampled yet"})

_STOP = threading.Event()
_STARTED = threading.Event()


# Loaded-model timeline ------------------------------------------------------
#
# GIN access lines carry no model name, and the only place the journal names a
# model is by weights-blob SHA, which does not match the manifest digest in
# /api/tags. So attribution is inferred from residency: record which model was
# loaded at each /api/ps sample, then ask which model was resident while a
# request was in flight.
#
# Three things make this more than a point lookup:
#
#   1. GIN logs a line when a request COMPLETES, not when it starts. A pull
#      that took 7 minutes is stamped at minute 7. Attributing by the log
#      timestamp alone would credit whatever happened to be loaded at the end.
#      Every GIN row carries its own latency, so the real span is known:
#      [epoch - latency_s, epoch].
#
#   2. /api/ps reports expires_at, so residency has a known end, not just a
#      last-seen sample. Without it a stalled sampler would keep attributing
#      new requests to whatever was loaded when it died, forever.
#
#   3. If a request's span crosses a model swap, no single model served it.
#      That returns None. Guessing would be a confident lie the UI renders as
#      fact.
#
# Exact under OLLAMA_MAX_LOADED_MODELS=1. With more than one model resident,
# record_timeline stores None rather than picking one. Anything rendering
# row["model"] must label it inferred, not observed.

MODEL_TIMELINE = deque(maxlen=PS_TIMELINE_LEN)
_TIMELINE_LOCK = threading.Lock()

# How far past the last observation residency may be assumed when expires_at is
# unknown. Covers the sampling gap and its jitter — not an outage.
_TRAILING_GRACE_SEC = 3 * LOADED_SAMPLE_SEC


def record_timeline(loaded, now=None):
    """Record which model was resident at this sample, and when it expires."""
    now = time.time() if now is None else now
    models = [m for m in (loaded or []) if m.get("model_key")]
    if len(models) == 1:
        resident = models[0]["model_key"]
        ttl = models[0].get("ttl_s")
        expires = now + ttl if ttl is not None else None
    else:
        # Nothing loaded, or too many to attribute unambiguously.
        resident, expires = None, None
    with _TIMELINE_LOCK:
        MODEL_TIMELINE.append((now, resident, expires))


def residency_intervals():
    """Observations -> [(start, end, model)], oldest first, half-open [start, end).

    Consecutive samples showing the same model collapse into one interval. An
    interval ends where the next observation contradicts it; the most recent
    one ends at expires_at, or after a short grace period when the TTL is
    unknown. The exact swap instant between two samples is not observable, so
    a boundary is accurate only to within one sampling interval.

    Half-open matters at a swap: with inclusive ends, the instant the old
    model's interval closes and the new one opens would match both and read as
    ambiguous, so every request landing exactly on a sample boundary would go
    unattributed.
    """
    with _TIMELINE_LOCK:
        snapshot = list(MODEL_TIMELINE)

    out = []
    i, n = 0, len(snapshot)
    while i < n:
        start, model, _ = snapshot[i]
        j = i
        while j + 1 < n and snapshot[j + 1][1] == model:
            j += 1
        last_ts, _, last_expires = snapshot[j]
        if j + 1 < n:
            end = snapshot[j + 1][0]   # contradicted by the next sample
        else:
            # Never extrapolate far past the keep_alive expiry: a sampler that
            # stopped an hour ago must not attribute requests made since. The
            # grace floor keeps the observation itself attributable even when
            # the model was already seconds from expiry when we saw it.
            end = max(last_expires or 0.0, last_ts + _TRAILING_GRACE_SEC)
        if model is not None:
            out.append((start, end, model))
        i = j + 1
    return out


def model_during(start, end, intervals=None):
    """Which model was resident for the whole span, or None.

    None when nothing covers the span, and also when the span straddles a
    swap — two models each served part of it and neither is the answer.
    """
    if intervals is None:
        intervals = residency_intervals()
    if end < start:
        return None
    if start == end:
        hits = {m for (s, e, m) in intervals if s <= start < e}
        return hits.pop() if len(hits) == 1 else None

    cursor = start
    resident = None
    for interval_start, interval_end, model in intervals:
        if interval_end <= cursor:
            continue
        if interval_start > cursor:
            return None
        if resident is None:
            resident = model
        elif model != resident:
            return None
        cursor = max(cursor, interval_end)
        if cursor >= end:
            return resident
    return None


def model_at(epoch, intervals=None):
    """Which model was resident at `epoch`, or None if unknown or ambiguous."""
    return model_during(epoch, epoch, intervals)


def attribute(rows):
    """Tag request rows with the model that was resident while they ran.

    Spans, not instants: a request that completed at `epoch` after `latency_s`
    seconds was in flight for [epoch - latency_s, epoch].
    """
    intervals = residency_intervals()   # built once, not per row
    for r in rows:
        if r.get("kind") != "request":
            continue
        end = r.get("epoch") or 0
        latency = r.get("latency_s") or 0
        r["model"] = model_during(end - latency, end, intervals)
    return rows


def _gpu_stream_loop():
    """One long-lived nvidia-smi emitting a CSV row every GPU_SAMPLE_MS.

    Spawning nvidia-smi per tick cost ~27ms each; streaming makes the sample
    rate essentially free.
    """
    cmd = ["nvidia-smi", f"--query-gpu={sources.GPU_QUERY_FIELDS}",
           "--format=csv,noheader,nounits", f"--loop-ms={GPU_SAMPLE_MS}"]
    while not _STOP.is_set():
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True, bufsize=1)
            for line in proc.stdout:
                if _STOP.is_set(): break
                line = line.strip()
                if not line: continue
                try:
                    g = sources.parse_gpu_csv(line)
                except Exception:
                    continue
                GPU.set(g)
                sources.push_history(g)   # throttled internally
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            # nvidia-smi missing or wedged: fall back to one-shot so the panel
            # still shows something (usually its error), and retry the stream.
            GPU.set(sources.gpu())
            _STOP.wait(5)
        _STOP.wait(1)


def _host_loop():
    """Host CPU/RAM. Cheap /proc reads, so sampled at the GPU cadence."""
    interval = HOST_SAMPLE_MS / 1000.0
    while not _STOP.is_set():
        HOST.refresh()
        _STOP.wait(interval)


def _loaded_loop():
    """Sample /api/ps and record what was resident, on one cadence."""
    while not _STOP.is_set():
        LOADED.refresh()
        record_timeline(LOADED.peek()[0] or [])
        _STOP.wait(LOADED_SAMPLE_SEC)


def start_all():
    """Launch every sampler. Idempotent."""
    if _STARTED.is_set(): return
    _STARTED.set()

    logs.start_follower()
    threading.Thread(target=_gpu_stream_loop, daemon=True).start()
    threading.Thread(target=_host_loop, daemon=True).start()
    threading.Thread(target=_loaded_loop, daemon=True).start()

    schedule = [
        (LOGS, LOGS_SAMPLE_SEC),
        (GPU_PROCS, LOADED_SAMPLE_SEC),
        (GPU_VERSIONS, SLOW_SAMPLE_SEC),
        (SETTINGS, SLOW_SAMPLE_SEC),
        (SERVICE, SLOW_SAMPLE_SEC),
        (TAILSCALE, SLOW_SAMPLE_SEC),
        (PING, LOADED_SAMPLE_SEC),
    ]
    for holder, interval in schedule:
        threading.Thread(target=_loop, args=(holder, interval, _STOP),
                         daemon=True).start()
    for holder, dependencies in ((LIBRARY, (LOADED,)), (DISK, (SETTINGS,))):
        threading.Thread(target=_dependent_loop,
                         args=(holder, dependencies, SLOW_SAMPLE_SEC, _STOP),
                         daemon=True).start()


def stop_all():
    """Used by tests; the server itself runs until killed."""
    logs.stop_follower()
    _STOP.set()
    _STARTED.clear()
