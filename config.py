"""Runtime configuration. All values overridable via environment variables."""
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
HOST = _env("LMSTUDIO_DASHBOARD_HOST", "127.0.0.1")
PORT = _env("LMSTUDIO_DASHBOARD_PORT", 11435, int)

# Upstream LM Studio
LMSTUDIO_URL = _env("LMSTUDIO_URL", "http://localhost:1234").rstrip("/")
LMS_BIN = _path("LMSTUDIO_LMS_BIN", "~/.lmstudio/bin/lms")

# Service management. LM Studio's headless server is a *user* unit by default.
SYSTEMD_UNIT = _env("LMSTUDIO_SYSTEMD_UNIT", "lmstudio-server")
SYSTEMD_USER = _bool("LMSTUDIO_SYSTEMD_USER", True)

# On-disk locations
LOG_DIR = _path("LMSTUDIO_LOG_DIR", "~/.lmstudio/server-logs")
SETTINGS_PATH = _path("LMSTUDIO_SETTINGS_PATH", "~/.lmstudio/settings.json")
MODEL_INDEX_PATH = _path("LMSTUDIO_MODEL_INDEX", "~/.lmstudio/.internal/model-index-cache.json")
HUB_MODELS_DIR = _path("LMSTUDIO_HUB_MODELS_DIR", "~/.lmstudio/hub/models")
# Used only when settings.json is unreadable; normally downloadsFolder wins.
MODELS_DIR_FALLBACK = _path("LMSTUDIO_MODELS_DIR", "~/.lmstudio/models")

# Remote catalog scrape
CATALOG_URL = _env("LMSTUDIO_CATALOG_URL", "https://lmstudio.ai/models")
CATALOG_TTL_SEC = _env("LMSTUDIO_CATALOG_TTL", 3600, int)
CATALOG_USER_AGENT = _env("LMSTUDIO_CATALOG_UA", "lmstudio-dashboard/1.0")

# Long-context benchmark corpus (Project Gutenberg Moby-Dick works well).
# If missing, the needle-in-haystack scenario is skipped.
HAYSTACK_PATH = _path("LMSTUDIO_HAYSTACK_PATH", "/tmp/moby.txt")
HAYSTACK_WORDS = _env("LMSTUDIO_HAYSTACK_WORDS", 21000, int)

# Rolling buffers and windows
GPU_HISTORY_LEN = _env("LMSTUDIO_GPU_HISTORY_LEN", 60, int)
PCIE_HISTORY_LEN = _env("LMSTUDIO_PCIE_HISTORY_LEN", 60, int)
LOG_WINDOW_LINES = _env("LMSTUDIO_LOG_WINDOW_LINES", 600, int)
STATS_WINDOW_SEC = _env("LMSTUDIO_STATS_WINDOW_SEC", 300, int)

# Paths excluded from request stats (they're polled by the dashboard itself).
NOISE_PATHS = {"/api/v0/models", "/v1/models", "/lmstudio-greeting"}
