"""Runtime configuration. All values overridable via environment variables."""
import os

def _env(key, default, cast=str):
    v = os.environ.get(key)
    if v is None or v == "": return default
    try: return cast(v)
    except (ValueError, TypeError): return default

# HTTP server
HOST = _env("OLLAMA_DASHBOARD_HOST", "127.0.0.1")
PORT = _env("OLLAMA_DASHBOARD_PORT", 11435, int)

# Upstream Ollama
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434").rstrip("/")

# Remote library scrape
LIBRARY_URL = _env("OLLAMA_LIBRARY_URL", "https://ollama.com/library")
LIBRARY_TTL_SEC = _env("OLLAMA_LIBRARY_TTL", 3600, int)
LIBRARY_USER_AGENT = _env("OLLAMA_LIBRARY_UA", "ollama-dashboard/1.0")

# Long-context benchmark corpus (Project Gutenberg Moby-Dick works well).
# If missing, the needle-in-haystack scenario is skipped.
HAYSTACK_PATH = _env("OLLAMA_HAYSTACK_PATH", "/tmp/moby.txt")
HAYSTACK_WORDS = _env("OLLAMA_HAYSTACK_WORDS", 21000, int)

# Read-only systemd override file shown on the control panel.
SYSTEMD_OVERRIDE_PATH = _env(
    "OLLAMA_SYSTEMD_OVERRIDE",
    "/etc/systemd/system/ollama.service.d/override.conf",
)
SYSTEMD_UNIT = _env("OLLAMA_SYSTEMD_UNIT", "ollama")

# Where to look for the on-disk model store (first existing dir wins).
MODEL_DIRS = [
    os.path.expanduser(p) for p in
    _env("OLLAMA_MODEL_DIRS", "/usr/share/ollama/.ollama/models:~/.ollama/models").split(":")
    if p
]

# Rolling buffers and windows
GPU_HISTORY_LEN = _env("OLLAMA_GPU_HISTORY_LEN", 60, int)
PCIE_HISTORY_LEN = _env("OLLAMA_PCIE_HISTORY_LEN", 60, int)
LOG_WINDOW_LINES = _env("OLLAMA_LOG_WINDOW_LINES", 600, int)
STATS_WINDOW_SEC = _env("OLLAMA_STATS_WINDOW_SEC", 300, int)

# Paths excluded from request stats (they're polled by the dashboard itself).
NOISE_PATHS = {"/api/tags", "/api/ps", "/api/version"}
