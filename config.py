"""Runtime configuration. All values overridable via environment variables.

Dashboard-owned settings are prefixed OLLAMA_DASHBOARD_. Bare OLLAMA_* is
Ollama's own namespace: OLLAMA_MODELS and OLLAMA_HOST are read here as inputs
so the dashboard follows the server's configuration, but nothing in this file
may define a bare OLLAMA_* name of its own. OLLAMA_URL is the one exception —
Ollama does not define it.
"""
import os


def _env(key, default, cast=str):
    v = os.environ.get(key)
    if v is None or v == "": return default
    try: return cast(v)
    except (ValueError, TypeError): return default


def _path(key, default):
    return os.path.expanduser(_env(key, default))


def _bool(key, default):
    return _env(key, "1" if default else "0") not in ("0", "false", "False", "no")


# HTTP server
HOST = _env("OLLAMA_DASHBOARD_HOST", "127.0.0.1")
PORT = _env("OLLAMA_DASHBOARD_PORT", 11435, int)

# Upstream Ollama
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434").rstrip("/")

# Service management. Ollama installs a *system* unit at
# /etc/systemd/system/ollama.service, unlike LM Studio's user unit.
SYSTEMD_UNIT = _env("OLLAMA_DASHBOARD_SYSTEMD_UNIT", "ollama")
SYSTEMD_USER = _bool("OLLAMA_DASHBOARD_SYSTEMD_USER", False)

# On-disk model store. OLLAMA_MODELS is Ollama's own variable; read it so the
# dashboard follows the server rather than contradicting it.
MODELS_DIR_FALLBACK = _path("OLLAMA_MODELS", "/usr/share/ollama/.ollama/models")

# journald
JOURNAL_UNIT = _env("OLLAMA_DASHBOARD_JOURNAL_UNIT", "ollama")
# Lines replayed when the follower (re)starts, so a restart does not blank the
# request window.
JOURNAL_BACKFILL = _env("OLLAMA_DASHBOARD_JOURNAL_BACKFILL", 2000, int)

# Remote catalog scrape
CATALOG_URL = _env("OLLAMA_DASHBOARD_CATALOG_URL", "https://ollama.com/library")
CATALOG_TTL_SEC = _env("OLLAMA_DASHBOARD_CATALOG_TTL", 3600, int)
CATALOG_USER_AGENT = _env("OLLAMA_DASHBOARD_CATALOG_UA", "ollama-dashboard/1.0")

# Rolling buffers and windows
GPU_HISTORY_LEN = _env("OLLAMA_DASHBOARD_GPU_HISTORY_LEN", 60, int)
PCIE_HISTORY_LEN = _env("OLLAMA_DASHBOARD_PCIE_HISTORY_LEN", 60, int)
LOG_WINDOW_LINES = _env("OLLAMA_DASHBOARD_LOG_WINDOW_LINES", 2000, int)
STATS_WINDOW_SEC = _env("OLLAMA_DASHBOARD_STATS_WINDOW_SEC", 300, int)
# Observations of which model was resident, used to attribute requests.
PS_TIMELINE_LEN = _env("OLLAMA_DASHBOARD_PS_TIMELINE_LEN", 900, int)

# Background sampling cadences.
#
# The request path never touches Ollama. Two reasons, neither of them latency:
# polling from the request path would flood the GIN access log that the
# dashboard exists to display, and `du -sb` over a multi-gigabyte model store
# must never block a response.
GPU_SAMPLE_MS = _env("OLLAMA_DASHBOARD_GPU_SAMPLE_MS", 100, int)
HOST_SAMPLE_MS = _env("OLLAMA_DASHBOARD_HOST_SAMPLE_MS", 100, int)
LOGS_SAMPLE_SEC = _env("OLLAMA_DASHBOARD_LOGS_SAMPLE_SEC", 1, int)
LOADED_SAMPLE_SEC = _env("OLLAMA_DASHBOARD_LOADED_SAMPLE_SEC", 2, int)
SLOW_SAMPLE_SEC = _env("OLLAMA_DASHBOARD_SLOW_SAMPLE_SEC", 15, int)
HISTORY_INTERVAL_SEC = _env("OLLAMA_DASHBOARD_HISTORY_INTERVAL_SEC", 1, int)

# Paths the dashboard itself polls. Filtered from request stats only when the
# caller is loopback — other clients hitting the same paths are real traffic
# worth showing. See logs.is_noise.
NOISE_PATHS = {"/api/tags", "/api/ps", "/api/version"}
