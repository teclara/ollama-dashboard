"""Background sampling layer.

The request path must never shell out. Every `lms` invocation costs roughly
200ms of Node startup, and a single aggregate call used to spend ~840ms of its
~935ms in subprocess spawns — which capped the whole dashboard at about 1.4 Hz
no matter what the browser's poll interval was set to.

So each source is sampled by a background thread on its own cadence and the
latest value is held in memory. `state()` and `live()` then just read those
values, which makes them effectively free and lets the UI refresh as fast as it
likes. It also stops the dashboard from flooding LM Studio's own logs with the
polling traffic it is trying to report on.

GPU samples come from a single long-lived `nvidia-smi --loop-ms` process rather
than one spawn per tick, the same trick the PCIe monitor already used.
"""
import subprocess, threading, time

import lmstudio
import logs
import sources
from config import (
    GPU_SAMPLE_MS, HOST_SAMPLE_MS, LOADED_SAMPLE_SEC, LOGS_SAMPLE_SEC,
    SLOW_SAMPLE_SEC,
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
LOADED = Sampled(lmstudio.loaded_models, [])
GPU_PROCS = Sampled(sources.gpu_processes, [])
LIBRARY = Sampled(lambda: lmstudio.library(LOADED.get()), [])
SETTINGS = Sampled(sources.settings, {})
DISK = Sampled(lambda: sources.disk(sources.models_root(SETTINGS.get())), {})
SERVICE = Sampled(sources.service_info, {})
TAILSCALE = Sampled(sources.tailscale, {})

# GPU is streamed rather than polled, so it gets a plain holder.
GPU = Sampled(sources.gpu, {})

_STOP = threading.Event()
_STARTED = threading.Event()


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


def start_all():
    """Launch every sampler. Idempotent."""
    if _STARTED.is_set(): return
    _STARTED.set()

    threading.Thread(target=_gpu_stream_loop, daemon=True).start()
    threading.Thread(target=_host_loop, daemon=True).start()

    schedule = [
        (LOGS, LOGS_SAMPLE_SEC),
        (LOADED, LOADED_SAMPLE_SEC),
        (GPU_PROCS, LOADED_SAMPLE_SEC),
        (LIBRARY, SLOW_SAMPLE_SEC),
        (SETTINGS, SLOW_SAMPLE_SEC),
        (DISK, SLOW_SAMPLE_SEC),
        (SERVICE, SLOW_SAMPLE_SEC),
        (TAILSCALE, SLOW_SAMPLE_SEC),
    ]
    for holder, interval in schedule:
        threading.Thread(target=_loop, args=(holder, interval, _STOP),
                         daemon=True).start()


def stop_all():
    """Used by tests; the server itself runs until killed."""
    _STOP.set()
    _STARTED.clear()
