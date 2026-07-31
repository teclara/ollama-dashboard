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

    Falls back to computing synchronously on first access, so a request that
    arrives before the first background tick gets real data rather than a hole.
    """

    def __init__(self, fn, default=None):
        self._fn = fn
        self._value = default
        self._has = False
        self._ts = 0.0
        self._lock = threading.Lock()
        self._compute_lock = threading.Lock()

    def set(self, value):
        with self._lock:
            self._value = value
            self._has = True
            self._ts = time.time()

    def peek(self):
        """Latest value without ever computing. (None, False) if never sampled."""
        with self._lock:
            return self._value, self._has

    def get(self):
        value, has = self.peek()
        if has: return value
        # Serialize the cold-start computation so a burst of first requests
        # does not all shell out at once.
        with self._compute_lock:
            value, has = self.peek()
            if has: return value
            self.refresh()
            return self.peek()[0]

    def refresh(self):
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


# Sources, grouped by how fast they actually change --------------------------

HOST = Sampled(sources.host, {})
LOGS = Sampled(logs.read_window, [])
LOADED = Sampled(ollama.loaded_models, [])
GPU_PROCS = Sampled(sources.gpu_processes, [])
LIBRARY = Sampled(lambda: ollama.library(LOADED.get()), [])
SETTINGS = Sampled(sources.settings, {})
DISK = Sampled(lambda: sources.disk(sources.models_root(SETTINGS.get())), {})
SERVICE = Sampled(sources.service_info, {})
TAILSCALE = Sampled(sources.tailscale, {})
PING = Sampled(ollama.ping, False)

# GPU is streamed rather than polled, so it gets a plain holder.
GPU = Sampled(sources.gpu, {})

_STOP = threading.Event()
_STARTED = threading.Event()


# Loaded-model timeline ------------------------------------------------------
#
# GIN access lines carry no model name, and the only place the journal names a
# model is by weights-blob SHA, which does not match the manifest digest in
# /api/tags. So attribution is inferred: record which model was resident at
# each /api/ps sample, then map request timestamps onto that.
#
# Exact under OLLAMA_MAX_LOADED_MODELS=1. Above that we record None rather than
# picking one, so attribution degrades to "unknown" instead of to a wrong
# answer. Anything rendering row["model"] must label it inferred, not observed.

MODEL_TIMELINE = deque(maxlen=PS_TIMELINE_LEN)
_TIMELINE_LOCK = threading.Lock()


def record_timeline(loaded, now=None):
    now = time.time() if now is None else now
    keys = [m.get("model_key") for m in (loaded or []) if m.get("model_key")]
    resident = keys[0] if len(keys) == 1 else None
    with _TIMELINE_LOCK:
        MODEL_TIMELINE.append((now, resident))


def model_at(epoch):
    """Which model was resident at `epoch`, or None if unknown or ambiguous."""
    with _TIMELINE_LOCK:
        snapshot = list(MODEL_TIMELINE)
    found = None
    for ts, model in snapshot:
        if ts <= epoch:
            found = model
        else:
            break
    return found


def attribute(rows):
    """Tag request rows with the model that was resident when they arrived."""
    for r in rows:
        if r.get("kind") == "request":
            r["model"] = model_at(r.get("epoch", 0))
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
        (LIBRARY, SLOW_SAMPLE_SEC),
        (SETTINGS, SLOW_SAMPLE_SEC),
        (DISK, SLOW_SAMPLE_SEC),
        (SERVICE, SLOW_SAMPLE_SEC),
        (TAILSCALE, SLOW_SAMPLE_SEC),
        (PING, LOADED_SAMPLE_SEC),
    ]
    for holder, interval in schedule:
        threading.Thread(target=_loop, args=(holder, interval, _STOP),
                         daemon=True).start()


def stop_all():
    """Used by tests; the server itself runs until killed."""
    logs.stop_follower()
    _STOP.set()
    _STARTED.clear()
