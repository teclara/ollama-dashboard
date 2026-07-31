# Ollama Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the dashboard from LM Studio to Ollama, keeping the current architecture and adding the request statistics Ollama's logs make possible.

**Architecture:** `config → ollama → {sources, logs} → samplers → server → templates`. `ollama.py` replaces `lmstudio.py` and makes no subprocess calls — everything is HTTP against `:11434`. `logs.py` is rewritten to stream journald and parse GIN access lines. `control.py` loses its two fragility hotspots (ANSI progress scraping, filesystem delete) to real Ollama APIs.

**Tech Stack:** Python 3, standard library only. `unittest` tests run under `pytest`. No third-party dependencies — this constraint is absolute and existing.

**Spec:** `docs/superpowers/specs/2026-07-30-ollama-dashboard-design.md`

## Global Constraints

- **Standard library only.** No pip installs, ever. The dashboard runs from `/opt` under a bare `python3`.
- **The suite stays green at every commit.** 150 tests pass at the branch point; no commit may reduce that without deleting the corresponding feature.
- **The request path never shells out and never calls Ollama.** All I/O goes through `samplers.py` holders.
- **Dashboard-owned env vars are prefixed `OLLAMA_DASHBOARD_*`.** Never bare `OLLAMA_*` — that namespace is Ollama's own. The sole exception is `OLLAMA_URL`, which Ollama does not define.
- **Ollama's own vars (`OLLAMA_HOST`, `OLLAMA_MODELS`) are read as inputs, never shadowed or overwritten.**
- **No `sudo` anywhere.** Where privilege is missing, degrade and flag it in the payload.
- **Port stays 11435.** Ollama owns 11434.
- Work happens on branch `ollama-port`. Directory rename is Task 11, last, so tests keep running from a stable path throughout.

---

### Task 1: `config.py` — new env namespace

**Files:**
- Modify: `config.py` (whole file)
- Test: `tests/test_config.py` (rewrite)

**Interfaces:**
- Produces: `OLLAMA_URL`, `HOST`, `PORT`, `SYSTEMD_UNIT`, `SYSTEMD_USER`, `MODELS_DIR_FALLBACK`, `JOURNAL_UNIT`, `JOURNAL_BACKFILL`, `PS_TIMELINE_LEN`, `CATALOG_URL`, `CATALOG_TTL_SEC`, `CATALOG_USER_AGENT`, `GPU_HISTORY_LEN`, `PCIE_HISTORY_LEN`, `LOG_WINDOW_LINES`, `STATS_WINDOW_SEC`, `GPU_SAMPLE_MS`, `HOST_SAMPLE_MS`, `LOGS_SAMPLE_SEC`, `LOADED_SAMPLE_SEC`, `SLOW_SAMPLE_SEC`, `HISTORY_INTERVAL_SEC`, `NOISE_PATHS`
- Removed (later tasks must not import): `LMS_BIN`, `LMSTUDIO_URL`, `SETTINGS_PATH`, `MODEL_INDEX_PATH`, `HUB_MODELS_DIR`, `LOG_DIR`, `LOG_TAIL_BYTES`, `HAYSTACK_PATH`, `HAYSTACK_WORDS`

- [ ] **Step 1: Write the failing test**

Replace `tests/test_config.py` entirely. Keep the existing `_fresh` helper — it is correct and reloads the module under a patched environment.

```python
import importlib, os, unittest


class TestConfig(unittest.TestCase):
    def _fresh(self, **env):
        """Reload config with a patched environment."""
        old = {k: os.environ.get(k) for k in env}
        os.environ.update({k: v for k, v in env.items() if v is not None})
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
        try:
            import config
            return importlib.reload(config)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_defaults(self):
        c = self._fresh(OLLAMA_URL=None, OLLAMA_DASHBOARD_PORT=None,
                        OLLAMA_DASHBOARD_SYSTEMD_USER=None)
        self.assertEqual(c.OLLAMA_URL, "http://localhost:11434")
        self.assertEqual(c.PORT, 11435)
        self.assertEqual(c.HOST, "127.0.0.1")
        self.assertEqual(c.SYSTEMD_UNIT, "ollama")
        self.assertEqual(c.JOURNAL_UNIT, "ollama")

    def test_service_is_a_system_unit_not_a_user_unit(self):
        # Ollama installs to /etc/systemd/system. LM Studio was a user unit;
        # inheriting True here would make every systemctl call silently fail.
        self.assertFalse(self._fresh(OLLAMA_DASHBOARD_SYSTEMD_USER=None).SYSTEMD_USER)

    def test_env_override_strips_trailing_slash(self):
        c = self._fresh(OLLAMA_URL="http://box:11434/", OLLAMA_DASHBOARD_PORT="9000")
        self.assertEqual(c.OLLAMA_URL, "http://box:11434")
        self.assertEqual(c.PORT, 9000)

    def test_bad_int_falls_back_to_default(self):
        self.assertEqual(self._fresh(OLLAMA_DASHBOARD_PORT="not-a-number").PORT, 11435)

    def test_models_dir_reads_ollamas_own_var(self):
        c = self._fresh(OLLAMA_MODELS="/mnt/big/models")
        self.assertEqual(c.MODELS_DIR_FALLBACK, "/mnt/big/models")

    def test_models_dir_default_is_the_system_store(self):
        c = self._fresh(OLLAMA_MODELS=None)
        self.assertEqual(c.MODELS_DIR_FALLBACK, "/usr/share/ollama/.ollama/models")

    def test_dashboard_vars_do_not_squat_ollamas_namespace(self):
        # A dashboard var named OLLAMA_MODELS or OLLAMA_HOST would collide with
        # the real server's configuration. Only OLLAMA_URL is permitted bare.
        import config
        src = open(config.__file__, encoding="utf-8").read()
        import re
        bare = set(re.findall(r'_env\("(OLLAMA_(?!DASHBOARD_)[A-Z_]+)"', src))
        bare |= set(re.findall(r'_path\("(OLLAMA_(?!DASHBOARD_)[A-Z_]+)"', src))
        self.assertEqual(bare - {"OLLAMA_URL", "OLLAMA_MODELS"}, set())

    def test_no_lmstudio_names_remain(self):
        c = self._fresh()
        leftovers = [n for n in dir(c) if "LMSTUDIO" in n.upper() or n == "LMS_BIN"]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_config.py -v`
Expected: FAIL — `AttributeError: module 'config' has no attribute 'OLLAMA_URL'`

- [ ] **Step 3: Write the implementation**

Replace `config.py` entirely. `_env`, `_path`, and `_bool` are unchanged — copy them verbatim.

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_config.py -v`
Expected: PASS (8 tests). The rest of the suite is now broken — that is expected and resolved by Tasks 2-9.

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "Rewrite config for Ollama env vars"
```

---

### Task 2: `ollama.py` — API client and normalization

**Files:**
- Create: `ollama.py`
- Delete: `lmstudio.py`
- Create: `tests/test_ollama.py`
- Delete: `tests/test_lmstudio.py`
- Create: `tests/fixtures/api_tags.json`, `tests/fixtures/api_ps_loaded.json`, `tests/fixtures/api_ps_empty.json`
- Delete: `tests/fixtures/lms_ls.json`, `lms_ps_empty.json`, `lms_ps_loaded.json`, `lms_runtime_ls.txt`, `model_index_cache.json`, `api_v0_models.json`

**Interfaces:**
- Consumes: `config.OLLAMA_URL`
- Produces:
  - `class OllamaError(Exception)`
  - `api_get(path, timeout=5) -> dict` / `api_post(path, payload, timeout=5) -> dict`
  - `normalize_loaded(raw: dict) -> list[dict]` — takes the whole `/api/ps` body
  - `normalize_library(raw: dict) -> list[dict]` — takes the whole `/api/tags` body
  - `join_library(disk, loaded_names) -> list[dict]`
  - `loaded_models() -> list[dict]`, `library(loaded=None) -> list[dict]`
  - `show(model) -> dict`, `version() -> dict`, `ping() -> bool`

- [ ] **Step 1: Capture the fixtures**

Run these against the live instance. Load a model first so `api_ps_loaded.json` is not empty.

```bash
cd ~/GitHub/lmstudio-dashboard
curl -s http://localhost:11434/api/tags > tests/fixtures/api_tags.json
curl -s http://localhost:11434/api/generate \
  -d '{"model":"gemma4:12b","prompt":"","keep_alive":"10m","options":{"num_ctx":8192}}' >/dev/null
curl -s http://localhost:11434/api/ps > tests/fixtures/api_ps_loaded.json
echo '{"models":[]}' > tests/fixtures/api_ps_empty.json
git rm tests/fixtures/lms_ls.json tests/fixtures/lms_ps_empty.json \
       tests/fixtures/lms_ps_loaded.json tests/fixtures/lms_runtime_ls.txt \
       tests/fixtures/model_index_cache.json tests/fixtures/api_v0_models.json
```

Then hand-edit `tests/fixtures/api_ps_loaded.json` to add a **second** entry with `size_vram` lower than `size`, so the partial-offload path has coverage. Real shape to base it on:

```json
{"models":[{"name":"gemma4:12b","model":"gemma4:12b","size":8060739255,
  "digest":"4eb23ef187e2...","details":{"parent_model":"","format":"gguf",
  "family":"gemma4","families":["gemma4"],"parameter_size":"11.9B",
  "quantization_level":"Q4_K_M"},
  "expires_at":"2026-07-30T23:39:24.821098399-04:00",
  "size_vram":8060739255,"context_length":8192}]}
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_ollama.py`:

```python
import unittest
from unittest import mock

import ollama
from tests.helpers import fixture_json


class TestNormalizeLoaded(unittest.TestCase):
    def test_maps_real_payload(self):
        out = ollama.normalize_loaded(fixture_json("api_ps_loaded.json"))
        m = out[0]
        self.assertEqual(m["model_key"], "gemma4:12b")
        # Ollama has one name. All three identifier slots collapse onto it,
        # unlike LM Studio where load key, instance name, and index key differed.
        self.assertEqual(m["identifier"], m["model_key"])
        self.assertEqual(m["display_name"], m["model_key"])
        self.assertEqual(m["arch"], "gemma4")
        self.assertEqual(m["quant"], "Q4_K_M")
        self.assertEqual(m["params"], "11.9B")
        self.assertEqual(m["context"], 8192)

    def test_fully_resident_model_reports_no_cpu_spill(self):
        m = ollama.normalize_loaded(fixture_json("api_ps_loaded.json"))[0]
        self.assertEqual(m["size"], m["size_vram"])
        self.assertEqual(m["cpu_bytes"], 0)
        self.assertTrue(m["fully_gpu"])

    def test_partial_offload_is_detected(self):
        raw = {"models": [{"name": "big:70b", "size": 1000, "size_vram": 600,
                           "details": {}}]}
        m = ollama.normalize_loaded(raw)[0]
        self.assertEqual(m["cpu_bytes"], 400)
        self.assertFalse(m["fully_gpu"])

    def test_ttl_derived_from_expires_at(self):
        raw = {"models": [{"name": "m", "details": {},
                           "expires_at": "2026-07-30T23:39:24.821098399-04:00"}]}
        with mock.patch("ollama.time.time", return_value=1785642864.0):
            # 2026-07-31T03:34:24Z == 1785642864; expiry is 5 minutes later.
            m = ollama.normalize_loaded(raw)[0]
        self.assertEqual(m["ttl_s"], 300)

    def test_expired_ttl_clamps_to_zero_not_negative(self):
        raw = {"models": [{"name": "m", "details": {},
                           "expires_at": "2020-01-01T00:00:00.000000000-04:00"}]}
        m = ollama.normalize_loaded(raw)[0]
        self.assertEqual(m["ttl_s"], 0)

    def test_missing_expires_at_is_none_not_zero(self):
        # None renders as "no TTL"; 0 would render as "expires now".
        m = ollama.normalize_loaded({"models": [{"name": "m", "details": {}}]})[0]
        self.assertIsNone(m["ttl_s"])

    def test_empty_payload(self):
        self.assertEqual(ollama.normalize_loaded(fixture_json("api_ps_empty.json")), [])

    def test_tolerates_garbage(self):
        self.assertEqual(ollama.normalize_loaded(None), [])
        self.assertEqual(ollama.normalize_loaded({}), [])
        self.assertEqual(ollama.normalize_loaded({"models": "nonsense"}), [])


class TestNormalizeLibrary(unittest.TestCase):
    def test_maps_real_payload(self):
        out = ollama.normalize_library(fixture_json("api_tags.json"))
        by_key = {m["model_key"]: m for m in out}
        m = by_key["nomic-embed-text:latest"]
        self.assertEqual(m["arch"], "nomic-bert")
        self.assertEqual(m["quant"], "F16")
        self.assertEqual(m["params"], "137M")
        self.assertTrue(m["embedding"])
        self.assertFalse(m["tools"])

    def test_capabilities_become_flags(self):
        raw = {"models": [{"name": "a:1", "details": {},
                           "capabilities": ["completion", "tools", "thinking", "vision"]}]}
        m = ollama.normalize_library(raw)[0]
        self.assertTrue(m["tools"])
        self.assertTrue(m["thinking"])
        self.assertTrue(m["vision"])
        self.assertFalse(m["embedding"])

    def test_missing_context_length_is_none_not_zero(self):
        # /api/tags omits details.context_length for some models (gemma4:31b
        # lacks it, ornith:35b has it). Zero would render as "0 token context".
        raw = {"models": [{"name": "a:1", "details": {"family": "x"}}]}
        self.assertIsNone(ollama.normalize_library(raw)[0]["max_context"])

    def test_sorted_by_model_key(self):
        raw = {"models": [{"name": "z:1", "details": {}}, {"name": "a:1", "details": {}}]}
        self.assertEqual([m["model_key"] for m in ollama.normalize_library(raw)],
                         ["a:1", "z:1"])


class TestJoinLibrary(unittest.TestCase):
    def test_marks_loaded_models(self):
        disk = [{"model_key": "a:1", "loaded": False},
                {"model_key": "b:1", "loaded": False}]
        out = ollama.join_library(disk, {"a:1"})
        self.assertTrue(out[0]["loaded"])
        self.assertFalse(out[1]["loaded"])


class TestLiveWrappers(unittest.TestCase):
    def test_loaded_models_returns_empty_on_transport_error(self):
        # The server being down must degrade to an empty list, never a 500.
        with mock.patch("ollama.api_get", side_effect=ollama.OllamaError("down")):
            self.assertEqual(ollama.loaded_models(), [])

    def test_library_returns_empty_on_transport_error(self):
        with mock.patch("ollama.api_get", side_effect=ollama.OllamaError("down")):
            self.assertEqual(ollama.library(), [])

    def test_ping_is_false_when_unreachable(self):
        with mock.patch("ollama.api_get", side_effect=ollama.OllamaError("down")):
            self.assertFalse(ollama.ping())

    def test_ping_is_true_when_version_responds(self):
        with mock.patch("ollama.api_get", return_value={"version": "0.32.5"}):
            self.assertTrue(ollama.ping())

    def test_library_passes_loaded_names_through(self):
        tags = {"models": [{"name": "a:1", "details": {}}]}
        with mock.patch("ollama.api_get", return_value=tags):
            out = ollama.library(loaded=[{"model_key": "a:1"}])
        self.assertTrue(out[0]["loaded"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ollama.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ollama'`

- [ ] **Step 4: Write the implementation**

Create `ollama.py`:

```python
"""All Ollama coupling: the HTTP API and payload normalization.

Nothing else in the codebase should know Ollama's field names.

Unlike the LM Studio module this replaces, there is no CLI here and no
identifier problem. LM Studio distinguished a load key, a loaded-instance
name, and a model-index key, none of them interchangeable. Ollama has one
name — `gemma4:31b` — that serves all three roles.
"""
import json, time, urllib.error, urllib.request
from datetime import datetime

from config import OLLAMA_URL


class OllamaError(Exception):
    """Ollama is unreachable, errored, or returned something unparseable."""


def _request(req, timeout):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        raise OllamaError(f"{req.get_method()} {req.full_url} -> HTTP {e.code}")
    except Exception as e:
        raise OllamaError(f"{req.get_method()} {req.full_url} failed: {e}")
    if not body:
        return {}
    try:
        return json.loads(body)
    except ValueError as e:
        raise OllamaError(f"{req.full_url} returned invalid JSON: {e}")


def api_get(path, timeout=5):
    return _request(urllib.request.Request(f"{OLLAMA_URL}{path}"), timeout)


def api_post(path, payload, timeout=5):
    req = urllib.request.Request(
        f"{OLLAMA_URL}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return _request(req, timeout)


# Normalization (pure) ------------------------------------------------------

def _rows(raw):
    models = (raw or {}).get("models") if isinstance(raw, dict) else None
    return models if isinstance(models, list) else []


def _caps(m):
    caps = m.get("capabilities") or []
    return {
        "vision": "vision" in caps,
        "tools": "tools" in caps,
        "thinking": "thinking" in caps,
        "embedding": "embedding" in caps,
    }


# Ollama emits nine fractional digits ("...24.821098399-04:00") but
# datetime.fromisoformat accepts at most six, so the fraction is truncated
# before parsing. Verified against the live server's expires_at values.
_ISO_FRAC_RE = re.compile(r"(\.\d{1,9})(?=[+-]\d{2}:\d{2}$|Z$)")


def _ttl_seconds(expires_at, now=None):
    """`expires_at` -> whole seconds remaining.

    None when absent (no TTL) rather than 0, which the UI would render as
    "expires now". Clamped at 0 so an already-expired entry never shows a
    negative countdown.
    """
    if not expires_at:
        return None
    cleaned = _ISO_FRAC_RE.sub(lambda m: m.group(1)[:7], expires_at)
    try:
        dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError:
        return None
    now = time.time() if now is None else now
    return max(0, int(dt.timestamp() - now))


def normalize_loaded(raw):
    """`GET /api/ps` body -> stable internal shape."""
    out = []
    for m in _rows(raw):
        d = m.get("details") or {}
        size = m.get("size") or 0
        vram = m.get("size_vram") or 0
        out.append({
            "model_key": m.get("name"),
            "identifier": m.get("name"),
            "display_name": m.get("name"),
            "arch": d.get("family"),
            "quant": d.get("quantization_level"),
            "params": d.get("parameter_size"),
            "size": size,
            "size_vram": vram,
            # When these differ, layers are on the CPU. On a single 32 GB card
            # running ~30B Q4 models this is the load-health signal that matters.
            "cpu_bytes": max(0, size - vram),
            "fully_gpu": size > 0 and size == vram,
            "context": m.get("context_length"),
            "ttl_s": _ttl_seconds(m.get("expires_at")),
            "digest": m.get("digest"),
            **_caps(m),
        })
    return out


def normalize_library(raw):
    """`GET /api/tags` body -> stable internal shape, sorted by model key."""
    out = []
    for m in _rows(raw):
        d = m.get("details") or {}
        out.append({
            "model_key": m.get("name"),
            "display_name": m.get("name"),
            "arch": d.get("family"),
            "quant": d.get("quantization_level"),
            "params": d.get("parameter_size"),
            "size": m.get("size") or 0,
            # Absent for some models; None, never 0.
            "max_context": d.get("context_length"),
            "digest": m.get("digest"),
            "modified_at": m.get("modified_at"),
            "loaded": False,
            **_caps(m),
        })
    return sorted(out, key=lambda x: x["model_key"] or "")


def join_library(disk, loaded_names):
    names = set(loaded_names or ())
    for m in disk:
        m["loaded"] = m.get("model_key") in names
    return disk


# Live wrappers -------------------------------------------------------------

def loaded_models():
    try:
        return normalize_loaded(api_get("/api/ps", timeout=5))
    except OllamaError:
        return []


def library(loaded=None):
    try:
        disk = normalize_library(api_get("/api/tags", timeout=10))
    except OllamaError:
        return []
    if loaded is None:
        loaded = loaded_models()
    return join_library(disk, {m["model_key"] for m in loaded})


def show(model):
    try:
        return api_post("/api/show", {"model": model}, timeout=10)
    except OllamaError:
        return {}


def version():
    try:
        return api_get("/api/version", timeout=2)
    except OllamaError:
        return {}


def ping():
    try:
        return bool(api_get("/api/version", timeout=2))
    except OllamaError:
        return False
```

Add `import re` to the module imports for `_ISO_FRAC_RE`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ollama.py -v`
Expected: PASS (18 tests)

- [ ] **Step 6: Delete the LM Studio module**

```bash
git rm lmstudio.py tests/test_lmstudio.py
```

- [ ] **Step 7: Commit**

```bash
git add ollama.py tests/test_ollama.py tests/fixtures/
git commit -m "Add ollama module wrapping the HTTP API"
```

---

### Task 3: `logs.py` — GIN parsing and Go durations

**Files:**
- Modify: `logs.py` (rewrite the parsing half; aggregation follows in Task 4)
- Create: `tests/fixtures/journal_excerpt.log`
- Delete: `tests/fixtures/server_log_excerpt.log`
- Modify: `tests/test_logs.py` (rewrite)

**Interfaces:**
- Consumes: `config.NOISE_PATHS`, `config.STATS_WINDOW_SEC`, `config.LOG_WINDOW_LINES`, `config.JOURNAL_UNIT`, `config.JOURNAL_BACKFILL`
- Produces:
  - `parse_duration(s) -> float | None` (seconds)
  - `parse_line(line) -> dict | None`
  - `is_noise(row) -> bool`
  - `parse_lines(lines) -> list[dict]`
  - Row shape: `{ts, epoch, kind, status, latency_s, client, method, path, level, message}`

- [ ] **Step 1: Capture the fixture**

```bash
journalctl -u ollama --since "-3h" --no-pager -o cat \
  | grep -E '^\[GIN\]|level=(WARN|ERROR)' \
  > tests/fixtures/journal_excerpt.log
git rm tests/fixtures/server_log_excerpt.log
```

Verify it contains at least one 401 or 404 and at least one compound duration:

```bash
grep -cE '\| (401|404) \|' tests/fixtures/journal_excerpt.log
grep -cE '\| +[0-9]+m[0-9.]+s +\|' tests/fixtures/journal_excerpt.log
```

If either is 0, append these verbatim real lines by hand:

```
[GIN] 2026/07/30 - 22:54:44 | 404 |      37.269µs |      172.17.0.3 | GET      "/models"
[GIN] 2026/07/30 - 23:00:21 | 401 |  130.345951ms |       127.0.0.1 | POST     "/api/me"
[GIN] 2026/07/30 - 23:14:51 | 200 |         7m19s |       127.0.0.1 | POST     "/api/pull"
```

- [ ] **Step 2: Write the failing test**

Replace `tests/test_logs.py`. This task covers parsing only; Task 4 appends the aggregation tests.

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_logs.py -v`
Expected: FAIL — `AttributeError: module 'logs' has no attribute 'parse_duration'`

- [ ] **Step 4: Write the implementation**

Replace the top half of `logs.py` (everything above the `# Aggregation` divider). Keep the divider; Task 4 rewrites what follows.

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_logs.py -v`
Expected: PASS (26 tests)

- [ ] **Step 6: Commit**

```bash
git add logs.py tests/test_logs.py tests/fixtures/journal_excerpt.log
git commit -m "Parse Ollama GIN access lines from journald"
```

---

### Task 4: `logs.py` — journal follower and richer aggregation

**Files:**
- Modify: `logs.py` (everything below the `# Aggregation` divider, plus a new follower section)
- Modify: `tests/test_logs.py` (append)

**Interfaces:**
- Consumes: `parse_lines`, `is_noise` from Task 3
- Produces:
  - `percentile(values, p) -> float | None`
  - `stats(rows, window_sec=None) -> dict` with keys `window_sec, count, rps, error_count, error_rate, p50_s, p95_s, p99_s`
  - `top_endpoints(rows, window_sec=None, top=8) -> list[dict]` with keys `path, count, errors, p95_s`
  - `by_client(rows, window_sec=None, top=8) -> list[dict]` with keys `client, count, errors, last_seen`
  - `problems(rows, limit=10) -> list[dict]`
  - `start_follower()`, `read_window() -> list[dict]`, `follower_age() -> float | None`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_logs.py`:

```python
class TestPercentile(unittest.TestCase):
    def test_median_of_odd_length(self):
        self.assertAlmostEqual(logs.percentile([1, 2, 3], 50), 2.0)

    def test_p95_picks_the_tail(self):
        self.assertAlmostEqual(logs.percentile(list(range(1, 101)), 95), 95.0)

    def test_single_value(self):
        self.assertAlmostEqual(logs.percentile([7.5], 95), 7.5)

    def test_empty_is_none(self):
        self.assertIsNone(logs.percentile([], 50))

    def test_ignores_none_entries(self):
        # A request whose latency failed to parse must not be counted as 0.
        self.assertAlmostEqual(logs.percentile([5, None, 5], 50), 5.0)
        self.assertAlmostEqual(logs.percentile([None, None, 7], 95), 7.0)


def _req(epoch, status=200, latency=0.01, client="1.2.3.4", path="/api/chat"):
    return {"kind": "request", "ts": "", "epoch": epoch, "status": status,
            "latency_s": latency, "client": client, "method": "POST",
            "path": path, "level": None, "message": None, "model": None}


class TestStats(unittest.TestCase):
    def setUp(self):
        self.now = time.time()

    def test_counts_and_rate(self):
        rows = [_req(self.now - i) for i in range(10)]
        s = logs.stats(rows, window_sec=100)
        self.assertEqual(s["count"], 10)
        self.assertAlmostEqual(s["rps"], 0.1)

    def test_error_rate_counts_non_2xx(self):
        rows = [_req(self.now, status=200), _req(self.now, status=200),
                _req(self.now, status=404), _req(self.now, status=401)]
        s = logs.stats(rows, window_sec=100)
        self.assertEqual(s["error_count"], 2)
        self.assertAlmostEqual(s["error_rate"], 50.0)

    def test_3xx_is_not_an_error(self):
        s = logs.stats([_req(self.now, status=304)], window_sec=100)
        self.assertEqual(s["error_count"], 0)

    def test_percentiles_reported_in_seconds(self):
        rows = [_req(self.now, latency=v / 1000.0) for v in range(1, 101)]
        s = logs.stats(rows, window_sec=100)
        self.assertAlmostEqual(s["p50_s"], 0.050, places=3)
        self.assertAlmostEqual(s["p95_s"], 0.095, places=3)

    def test_rows_outside_the_window_are_excluded(self):
        rows = [_req(self.now), _req(self.now - 9999)]
        self.assertEqual(logs.stats(rows, window_sec=60)["count"], 1)

    def test_empty_window_is_all_zeroes_not_none(self):
        s = logs.stats([], window_sec=60)
        self.assertEqual(s["count"], 0)
        self.assertEqual(s["rps"], 0)
        self.assertEqual(s["error_rate"], 0)
        self.assertIsNone(s["p95_s"])


class TestTopEndpoints(unittest.TestCase):
    def test_ranks_by_count_with_errors_and_p95(self):
        now = time.time()
        rows = ([_req(now, path="/api/chat", latency=1.0)] * 3
                + [_req(now, path="/api/show", status=404, latency=0.002)])
        out = logs.top_endpoints(rows, window_sec=100)
        self.assertEqual(out[0]["path"], "/api/chat")
        self.assertEqual(out[0]["count"], 3)
        self.assertEqual(out[0]["errors"], 0)
        self.assertAlmostEqual(out[0]["p95_s"], 1.0)
        self.assertEqual(out[1]["errors"], 1)


class TestByClient(unittest.TestCase):
    def test_groups_by_address(self):
        now = time.time()
        rows = ([_req(now, client="172.17.0.3")] * 4
                + [_req(now, client="127.0.0.1", status=500)])
        out = logs.by_client(rows, window_sec=100)
        self.assertEqual(out[0]["client"], "172.17.0.3")
        self.assertEqual(out[0]["count"], 4)
        self.assertEqual(out[1]["errors"], 1)

    def test_last_seen_is_the_newest_epoch(self):
        now = time.time()
        rows = [_req(now - 50, client="a"), _req(now - 5, client="a")]
        self.assertAlmostEqual(logs.by_client(rows, window_sec=100)[0]["last_seen"],
                               now - 5)


class TestProblems(unittest.TestCase):
    def test_returns_newest_first_and_limits(self):
        rows = [{"kind": "problem", "epoch": i, "level": "ERROR",
                 "message": f"m{i}", "ts": "", "status": None, "latency_s": None,
                 "client": None, "method": None, "path": None, "model": None}
                for i in range(5)]
        out = logs.problems(rows, limit=2)
        self.assertEqual([p["message"] for p in out], ["m4", "m3"])


class TestFollower(unittest.TestCase):
    def test_read_window_is_empty_before_the_follower_starts(self):
        self.assertEqual(logs.read_window(), [])

    def test_ingest_appends_parsed_rows(self):
        logs._BUF.clear()
        logs._ingest('[GIN] 2026/07/30 - 23:06:17 | 200 |          1m8s '
                     '|      172.17.0.3 | POST     "/api/chat"')
        rows = logs.read_window()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["latency_s"], 68.0)
        logs._BUF.clear()

    def test_ingest_drops_noise(self):
        logs._BUF.clear()
        logs._ingest('[GIN] 2026/07/30 - 23:25:18 | 200 |      51.336µs '
                     '|             ::1 | GET      "/api/ps"')
        self.assertEqual(logs.read_window(), [])

    def test_buffer_is_bounded(self):
        logs._BUF.clear()
        line = ('[GIN] 2026/07/30 - 23:06:17 | 200 |      51.336µs '
                '|      172.17.0.3 | POST     "/api/chat"')
        for _ in range(logs.LOG_WINDOW_LINES + 500):
            logs._ingest(line)
        self.assertEqual(len(logs.read_window()), logs.LOG_WINDOW_LINES)
        logs._BUF.clear()

    def test_journalctl_command_follows_the_configured_unit(self):
        cmd = logs._journal_cmd()
        self.assertIn("journalctl", cmd[0])
        self.assertIn("-u", cmd)
        self.assertIn(logs.JOURNAL_UNIT, cmd)
        self.assertIn("-f", cmd)
```

Add `import time` to the top of `tests/test_logs.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_logs.py -v`
Expected: FAIL — `AttributeError: module 'logs' has no attribute 'percentile'`

- [ ] **Step 3: Write the implementation**

Replace everything from the `# Aggregation` divider to the end of `logs.py`:

```python
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
            for line in proc.stdout:
                if _STOP.is_set():
                    break
                _ingest(line.rstrip("\n"))
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass
        # journalctl exited — vacuum, rotation, or a killed process. Rebuild
        # from the backfill rather than leaving a frozen window on screen.
        _STOP.wait(2)


def start_follower():
    """Launch the journal follower. Idempotent."""
    if _STARTED.is_set():
        return
    _STARTED.set()
    threading.Thread(target=_follow_loop, daemon=True).start()


def stop_follower():
    """Used by tests; the server itself runs until killed."""
    _STOP.set()
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_logs.py -v`
Expected: PASS (46 tests)

- [ ] **Step 5: Commit**

```bash
git add logs.py tests/test_logs.py
git commit -m "Add journal follower and request statistics"
```

---

### Task 5: `samplers.py` — rewire and add the model timeline

**Files:**
- Modify: `samplers.py`
- Modify: `tests/test_samplers.py`

**Interfaces:**
- Consumes: `ollama.loaded_models`, `ollama.library`, `logs.read_window`, `sources.*`
- Produces:
  - Holders `HOST, LOGS, LOADED, GPU_PROCS, LIBRARY, SETTINGS, DISK, SERVICE, TAILSCALE, GPU`
  - `MODEL_TIMELINE` — a bounded `deque` of `(epoch, model_key_or_None)`
  - `record_timeline(loaded, now=None)`
  - `model_at(epoch) -> str | None`
  - `attribute(rows) -> list[dict]` — sets `row["model"]` in place, returns rows
  - `start_all()`, `stop_all()`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_samplers.py`:

```python
class TestModelTimeline(unittest.TestCase):
    def setUp(self):
        samplers.MODEL_TIMELINE.clear()

    def tearDown(self):
        samplers.MODEL_TIMELINE.clear()

    def test_records_the_resident_model(self):
        samplers.record_timeline([{"model_key": "gemma4:31b"}], now=100.0)
        self.assertEqual(samplers.model_at(100.0), "gemma4:31b")

    def test_records_none_when_nothing_is_loaded(self):
        samplers.record_timeline([], now=100.0)
        self.assertIsNone(samplers.model_at(100.0))

    def test_lookup_uses_the_most_recent_observation_at_or_before(self):
        samplers.record_timeline([{"model_key": "a:1"}], now=100.0)
        samplers.record_timeline([{"model_key": "b:1"}], now=200.0)
        self.assertEqual(samplers.model_at(150.0), "a:1")
        self.assertEqual(samplers.model_at(250.0), "b:1")

    def test_request_before_any_observation_is_unattributed(self):
        samplers.record_timeline([{"model_key": "a:1"}], now=200.0)
        self.assertIsNone(samplers.model_at(100.0))

    def test_multiple_loaded_models_is_ambiguous_not_a_guess(self):
        # With MAX_LOADED_MODELS>1 we cannot know which one served a request.
        # Returning the first would be a confident lie.
        samplers.record_timeline([{"model_key": "a:1"}, {"model_key": "b:1"}],
                                 now=100.0)
        self.assertIsNone(samplers.model_at(100.0))

    def test_timeline_is_bounded(self):
        for i in range(samplers.PS_TIMELINE_LEN + 100):
            samplers.record_timeline([{"model_key": "a:1"}], now=float(i))
        self.assertEqual(len(samplers.MODEL_TIMELINE), samplers.PS_TIMELINE_LEN)

    def test_attribute_tags_rows_in_place(self):
        samplers.record_timeline([{"model_key": "a:1"}], now=100.0)
        rows = [{"kind": "request", "epoch": 150.0, "model": None}]
        out = samplers.attribute(rows)
        self.assertEqual(out[0]["model"], "a:1")

    def test_attribute_leaves_problems_alone(self):
        samplers.record_timeline([{"model_key": "a:1"}], now=100.0)
        rows = [{"kind": "problem", "epoch": 150.0, "model": None}]
        self.assertIsNone(samplers.attribute(rows)[0]["model"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_samplers.py -v`
Expected: FAIL — `AttributeError: module 'samplers' has no attribute 'MODEL_TIMELINE'`

- [ ] **Step 3: Write the implementation**

Rewrite the `samplers.py` docstring, swap the `lmstudio` import for `ollama`, and add the timeline. The `Sampled` class and `_loop`, `_gpu_stream_loop`, `_host_loop` bodies are unchanged — do not touch them.

New docstring:

```python
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
```

Imports and holders:

```python
import subprocess, threading, time
from collections import deque

import logs
import ollama
import sources
from config import (
    GPU_SAMPLE_MS, HOST_SAMPLE_MS, LOADED_SAMPLE_SEC, LOGS_SAMPLE_SEC,
    PS_TIMELINE_LEN, SLOW_SAMPLE_SEC,
)
```

Replace the `LOGS`, `LOADED`, and `LIBRARY` holder definitions:

```python
HOST = Sampled(sources.host, {})
LOGS = Sampled(logs.read_window, [])
LOADED = Sampled(ollama.loaded_models, [])
GPU_PROCS = Sampled(sources.gpu_processes, [])
LIBRARY = Sampled(lambda: ollama.library(LOADED.get()), [])
SETTINGS = Sampled(sources.settings, {})
DISK = Sampled(lambda: sources.disk(sources.models_root(SETTINGS.get())), {})
SERVICE = Sampled(sources.service_info, {})
TAILSCALE = Sampled(sources.tailscale, {})
GPU = Sampled(sources.gpu, {})
```

Add the timeline section immediately after the holders:

```python
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
```

Add a loop that feeds the timeline. Replace the `LOADED` entry in the `schedule` list with a dedicated thread, since it now does two things:

```python
def _loaded_loop():
    """Sample /api/ps and record what was resident, on one cadence."""
    while not _STOP.is_set():
        LOADED.refresh()
        record_timeline(LOADED.peek()[0] or [])
        _STOP.wait(LOADED_SAMPLE_SEC)
```

In `start_all()`, start the journal follower, drop `LOADED` from `schedule`, and add `_loaded_loop`:

```python
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
    ]
    for holder, interval in schedule:
        threading.Thread(target=_loop, args=(holder, interval, _STOP),
                         daemon=True).start()
```

In `stop_all()`, add `logs.stop_follower()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_samplers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add samplers.py tests/test_samplers.py
git commit -m "Sample Ollama and track which model was resident"
```

---

### Task 6: `sources.py` — systemd settings, disk fallback, service

**Files:**
- Modify: `sources.py`
- Modify: `tests/test_sources.py`

**Interfaces:**
- Consumes: `ollama.version`, `ollama.ping`, `config.SYSTEMD_UNIT`, `config.SYSTEMD_USER`, `config.MODELS_DIR_FALLBACK`
- Produces:
  - `parse_systemd_environment(text) -> dict`
  - `settings() -> dict`
  - `models_root(settings_dict) -> str`
  - `disk(root=None) -> dict` with `models_dir, models_size, approximate, fs_used, fs_total, fs_free, orphan_bytes`
  - `service_info() -> dict`, `state()`, `live()` (unchanged signatures)
- Unchanged, do not touch: `parse_gpu_csv`, `gpu`, `gpu_processes`, `nvidia_versions`, `tailscale`, `push_history`, `get_history`, `host`, the PCIe monitor

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sources.py`:

```python
class TestParseSystemdEnvironment(unittest.TestCase):
    def test_parses_the_real_output(self):
        # `systemctl show ollama --property=Environment` emits one
        # space-separated line, quoting only values that need it.
        text = ('Environment=OLLAMA_HOST=http://0.0.0.0:11434 "OLLAMA_ORIGINS=*" '
                'OLLAMA_FLASH_ATTENTION=true OLLAMA_MAX_LOADED_MODELS=1\n')
        env = sources.parse_systemd_environment(text)
        self.assertEqual(env["OLLAMA_HOST"], "http://0.0.0.0:11434")
        self.assertEqual(env["OLLAMA_ORIGINS"], "*")
        self.assertEqual(env["OLLAMA_FLASH_ATTENTION"], "true")
        self.assertEqual(env["OLLAMA_MAX_LOADED_MODELS"], "1")

    def test_ignores_non_ollama_vars(self):
        text = 'Environment=PATH=/usr/bin OLLAMA_KEEP_ALIVE=30m\n'
        env = sources.parse_systemd_environment(text)
        self.assertNotIn("PATH", env)
        self.assertEqual(env["OLLAMA_KEEP_ALIVE"], "30m")

    def test_redacts_secret_looking_names(self):
        # The old whitelist cannot work here: we cannot enumerate what the user
        # may add later, so anything secret-shaped is denied instead.
        text = ('Environment=OLLAMA_API_KEY=sk-abc123 OLLAMA_AUTH_TOKEN=t '
                'OLLAMA_SECRET=s OLLAMA_HOST=http://x\n')
        env = sources.parse_systemd_environment(text)
        self.assertNotIn("sk-abc123", str(env))
        self.assertEqual(env["OLLAMA_API_KEY"], "<redacted>")
        self.assertEqual(env["OLLAMA_AUTH_TOKEN"], "<redacted>")
        self.assertEqual(env["OLLAMA_SECRET"], "<redacted>")
        self.assertEqual(env["OLLAMA_HOST"], "http://x")

    def test_empty_input(self):
        self.assertEqual(sources.parse_systemd_environment(""), {})
        self.assertEqual(sources.parse_systemd_environment(None), {})


class TestModelsRoot(unittest.TestCase):
    def test_prefers_the_servers_own_setting(self):
        self.assertEqual(
            sources.models_root({"OLLAMA_MODELS": "/mnt/models"}), "/mnt/models")

    def test_falls_back_to_config(self):
        from config import MODELS_DIR_FALLBACK
        self.assertEqual(sources.models_root({}), MODELS_DIR_FALLBACK)
        self.assertEqual(sources.models_root(None), MODELS_DIR_FALLBACK)


class TestDisk(unittest.TestCase):
    def test_statvfs_walks_up_past_an_unreadable_dir(self):
        # /usr/share/ollama is 0750 ollama:ollama. A process without that group
        # gets PermissionError on the models dir but the filesystem totals are
        # identical for any ancestor on the same mount.
        real = os.statvfs

        def fake(path):
            if path.endswith("/models"):
                raise PermissionError(13, "Permission denied")
            return real("/")

        with mock.patch("sources.os.statvfs", side_effect=fake), \
             mock.patch("sources.os.path.isdir", return_value=True), \
             mock.patch("sources._du", return_value=None):
            info = sources.disk("/usr/share/ollama/.ollama/models")
        self.assertGreater(info["fs_total"], 0)

    def test_falls_back_to_summing_tags_when_du_is_denied(self):
        with mock.patch("sources.os.path.isdir", return_value=True), \
             mock.patch("sources._du", return_value=None), \
             mock.patch("sources.ollama.library",
                        return_value=[{"size": 100}, {"size": 200}]):
            info = sources.disk("/anywhere")
        self.assertEqual(info["models_size"], 300)
        self.assertTrue(info["approximate"])

    def test_du_result_is_exact_and_reports_orphans(self):
        # du counts every blob; summing /api/tags counts only referenced ones.
        # The difference is unreferenced blob space and is worth surfacing.
        with mock.patch("sources.os.path.isdir", return_value=True), \
             mock.patch("sources._du", return_value=67460511656), \
             mock.patch("sources.ollama.library",
                        return_value=[{"size": 48866552236}]):
            info = sources.disk("/anywhere")
        self.assertEqual(info["models_size"], 67460511656)
        self.assertFalse(info["approximate"])
        self.assertEqual(info["orphan_bytes"], 67460511656 - 48866552236)

    def test_missing_directory_returns_zeroes_not_an_exception(self):
        with mock.patch("sources.os.path.isdir", return_value=False):
            info = sources.disk("/nope")
        self.assertEqual(info["models_size"], 0)
        self.assertIsNone(info["models_dir"])


class TestServiceInfo(unittest.TestCase):
    def test_reports_engine_version_from_the_api(self):
        with mock.patch("sources._systemctl", side_effect=Exception("no systemd")), \
             mock.patch("sources.ollama.version", return_value={"version": "0.32.5"}):
            info = sources.service_info()
        self.assertEqual(info["engine"]["version"], "0.32.5")
        self.assertEqual(info["engine"]["name"], "ollama")

    def test_survives_systemctl_failure(self):
        with mock.patch("sources._systemctl", side_effect=Exception("boom")), \
             mock.patch("sources.ollama.version", return_value={}):
            info = sources.service_info()
        self.assertEqual(info["active"], "unknown")
        self.assertIsNone(info["pid"])
```

Ensure `tests/test_sources.py` imports `os`, `unittest`, `mock`, and `sources`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sources.py -v`
Expected: FAIL — `AttributeError: module 'sources' has no attribute 'parse_systemd_environment'`

- [ ] **Step 3: Write the implementation**

In `sources.py`, change the imports:

```python
import logs
import ollama
from config import (
    GPU_HISTORY_LEN, HISTORY_INTERVAL_SEC, MODELS_DIR_FALLBACK,
    PCIE_HISTORY_LEN, SYSTEMD_UNIT, SYSTEMD_USER,
)
```

Replace the whole `# Settings and on-disk model store` section:

```python
# Settings and on-disk model store ------------------------------------------
#
# Ollama has no settings file. Its configuration is the systemd unit's
# environment, which is also where this machine's real tuning lives
# (KV_CACHE_TYPE, FLASH_ATTENTION, MAX_LOADED_MODELS).

# A denylist, not a whitelist. Under LM Studio we could enumerate the settings
# worth exposing; here the user may add any OLLAMA_* variable at any time, so
# anything secret-shaped is redacted and everything else passes through.
_SECRET_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)

_ENV_TOKEN_RE = re.compile(r'"([^"]*)"|(\S+)')


def parse_systemd_environment(text):
    """`systemctl show <unit> --property=Environment` -> {name: value}.

    systemd emits one space-separated line and quotes only the values that
    need it, e.g. `OLLAMA_HOST=http://0.0.0.0:11434 "OLLAMA_ORIGINS=*"`.
    """
    out = {}
    for line in (text or "").splitlines():
        if not line.startswith("Environment="):
            continue
        for quoted, bare in _ENV_TOKEN_RE.findall(line[len("Environment="):]):
            token = quoted or bare
            name, sep, value = token.partition("=")
            if not sep or not name.startswith("OLLAMA_"):
                continue
            out[name] = "<redacted>" if _SECRET_RE.search(name) else value
    return out


def settings():
    try:
        raw = _systemctl("show", SYSTEMD_UNIT, "--property=Environment")
    except Exception as e:
        return {"unit": SYSTEMD_UNIT, "error": str(e)}
    env = parse_systemd_environment(raw)
    return {"unit": SYSTEMD_UNIT, **env}


def models_root(settings_dict):
    return (settings_dict or {}).get("OLLAMA_MODELS") or MODELS_DIR_FALLBACK


def _statvfs_walk_up(root):
    """statvfs on `root`, walking toward / past unreadable ancestors.

    /usr/share/ollama is 0750 ollama:ollama, so a dashboard process outside
    that group cannot stat the model dir. Any ancestor on the same mount
    reports identical filesystem totals.
    """
    path = os.path.abspath(root)
    while True:
        try:
            return os.statvfs(path)
        except OSError:
            parent = os.path.dirname(path)
            if parent == path:
                return None
            path = parent


def _du(root):
    """Exact bytes via `du -sb`, or None when not permitted."""
    try:
        return int(subprocess.check_output(
            ["du", "-sb", root], text=True, timeout=30,
            stderr=subprocess.DEVNULL).split()[0])
    except Exception:
        return None


def disk(root=None):
    root = root or models_root(settings())
    info = {"models_dir": None, "models_size": 0, "approximate": False,
            "orphan_bytes": None, "fs_used": 0, "fs_total": 0, "fs_free": 0}
    if not os.path.isdir(root):
        return info
    info["models_dir"] = root

    referenced = sum(m.get("size") or 0 for m in ollama.library())
    exact = _du(root)
    if exact is None:
        # No read access to the store. Summing /api/tags counts only
        # referenced blobs, so this understates the total — flag it.
        info["models_size"] = referenced
        info["approximate"] = True
    else:
        info["models_size"] = exact
        # Everything du sees that no manifest references: reclaimable.
        info["orphan_bytes"] = max(0, exact - referenced)

    st = _statvfs_walk_up(root)
    if st is not None:
        info["fs_total"] = st.f_blocks * st.f_frsize
        info["fs_free"] = st.f_bavail * st.f_frsize
        info["fs_used"] = info["fs_total"] - info["fs_free"]
    return info
```

Replace the tail of `service_info` — everything after the `except Exception: pass` that closes the `/proc` block:

```python
    info["engine"] = {"name": "ollama", "version": ollama.version().get("version")}
    return info
```

Replace `live()` and `state()`:

```python
def state():
    """The full payload. Also served from the sampler cache."""
    import samplers
    rows = samplers.attribute(samplers.LOGS.get())
    return {
        **live(),
        "gpu_processes": samplers.GPU_PROCS.get(),
        "gpu_versions": nvidia_versions(),
        "loaded": samplers.LOADED.get(),
        "library": samplers.LIBRARY.get(),
        "requests": rows[-30:][::-1],
        "stats_5m": logs.stats(rows),
        "top_endpoints": logs.top_endpoints(rows),
        "by_client": logs.by_client(rows),
        "problems": logs.problems(rows),
        "log_age_s": logs.follower_age(),
        "disk": samplers.DISK.get(),
        "service": samplers.SERVICE.get(),
        "tailscale": samplers.TAILSCALE.get(),
        "settings": samplers.SETTINGS.get(),
        # Sampled, never called live — state() must not touch Ollama.
        "ollama_ok": samplers.PING.get(),
    }
```

`live()` is unchanged. Delete `model_activity` from the payload — it is replaced by `attribute` plus `top_endpoints`.

`samplers.PING` does not exist yet. Add it to `samplers.py` beside the other
holders, and add `(PING, LOADED_SAMPLE_SEC)` to the `schedule` list in
`start_all()`:

```python
PING = Sampled(ollama.ping, False)
```

A bare `ollama.ping()` inside `state()` would be a live HTTP call on the
request path, which the global constraints forbid — and it would log itself
into the very request window the dashboard displays.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_sources.py tests/test_samplers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sources.py samplers.py tests/test_sources.py
git commit -m "Read Ollama config from systemd and handle an unreadable model store"
```

---

### Task 7: `control.py` — load, unload, and pull jobs

**Files:**
- Modify: `control.py` (delete lines 62-198 of the original — progress parsing, `_run_job`, load/unload, download — and replace)
- Modify: `tests/test_jobs.py` (rewrite)

**Interfaces:**
- Consumes: `ollama.api_post`, `ollama.loaded_models`, `config.OLLAMA_URL`
- Produces:
  - `build_load_payload(model, context=None, gpu=None, ttl=None) -> dict`
  - `PullProgress` with `.update(event) -> None`, `.snapshot() -> dict`
  - `start_load(model, **opts) -> bool`, `estimate_fit(model) -> dict`
  - `unload_model(name)`, `unload_all()`
  - `start_download(name) -> bool`
  - Unchanged: `_JOBS`, `_set_job`, `_update_job`, `_claim_job`, `get_jobs`, `clear_finished_jobs`, `clear_all_jobs`

- [ ] **Step 1: Write the failing test**

Replace `tests/test_jobs.py`:

```python
import unittest
from unittest import mock

import control


class TestBuildLoadPayload(unittest.TestCase):
    def test_minimal(self):
        p = control.build_load_payload("gemma4:12b")
        self.assertEqual(p["model"], "gemma4:12b")
        # An empty prompt makes /api/generate a pure load with no generation.
        self.assertEqual(p["prompt"], "")
        self.assertNotIn("options", p)

    def test_context_becomes_num_ctx(self):
        p = control.build_load_payload("m", context=8192)
        self.assertEqual(p["options"]["num_ctx"], 8192)

    def test_gpu_becomes_num_gpu(self):
        p = control.build_load_payload("m", gpu=99)
        self.assertEqual(p["options"]["num_gpu"], 99)

    def test_ttl_becomes_keep_alive(self):
        self.assertEqual(control.build_load_payload("m", ttl="30m")["keep_alive"], "30m")

    def test_blank_options_are_omitted_not_sent_as_null(self):
        # Sending num_ctx: null would override the server default with garbage.
        p = control.build_load_payload("m", context="", gpu=None, ttl="")
        self.assertNotIn("options", p)
        self.assertNotIn("keep_alive", p)

    def test_numeric_strings_are_coerced(self):
        p = control.build_load_payload("m", context="8192", gpu="99")
        self.assertEqual(p["options"]["num_ctx"], 8192)
        self.assertEqual(p["options"]["num_gpu"], 99)


class TestPullProgress(unittest.TestCase):
    def test_sums_across_layers(self):
        # Ollama reports completed/total per blob digest. Taking the latest
        # pair would make the bar jump backwards each time a layer starts.
        p = control.PullProgress()
        p.update({"digest": "sha256:a", "completed": 100, "total": 100})
        p.update({"digest": "sha256:b", "completed": 50, "total": 200})
        s = p.snapshot()
        self.assertEqual(s["completed"], 150)
        self.assertEqual(s["total"], 300)
        self.assertAlmostEqual(s["pct"], 50.0)

    def test_progress_never_goes_backwards_across_a_real_stream(self):
        p = control.PullProgress()
        events = [
            {"status": "pulling manifest"},
            {"status": "pulling 970aa74c0a90", "digest": "sha256:970a",
             "total": 274290656, "completed": 137145328},
            {"status": "pulling 970aa74c0a90", "digest": "sha256:970a",
             "total": 274290656, "completed": 274290656},
            {"status": "pulling c71d239df917", "digest": "sha256:c71d",
             "total": 11357, "completed": 0},
            {"status": "verifying sha256 digest"},
            {"status": "success"},
        ]
        seen = []
        for e in events:
            p.update(e)
            pct = p.snapshot()["pct"]
            if pct is not None:
                seen.append(pct)
        self.assertEqual(seen, sorted(seen), f"progress regressed: {seen}")

    def test_indeterminate_status_preserves_the_last_percentage(self):
        p = control.PullProgress()
        p.update({"digest": "sha256:a", "completed": 50, "total": 100})
        before = p.snapshot()["pct"]
        p.update({"status": "verifying sha256 digest"})
        self.assertEqual(p.snapshot()["pct"], before)

    def test_status_is_carried_through(self):
        p = control.PullProgress()
        p.update({"status": "pulling manifest"})
        self.assertEqual(p.snapshot()["last_line"], "pulling manifest")

    def test_empty_progress_has_no_percentage(self):
        self.assertIsNone(control.PullProgress().snapshot()["pct"])

    def test_rate_is_derived_from_deltas(self):
        p = control.PullProgress()
        p.update({"digest": "a", "completed": 0, "total": 1000}, now=100.0)
        p.update({"digest": "a", "completed": 500, "total": 1000}, now=102.0)
        self.assertAlmostEqual(p.snapshot()["rate_bps"], 250.0)

    def test_eta_is_derived_from_rate(self):
        p = control.PullProgress()
        p.update({"digest": "a", "completed": 0, "total": 1000}, now=100.0)
        p.update({"digest": "a", "completed": 500, "total": 1000}, now=102.0)
        self.assertAlmostEqual(p.snapshot()["eta_s"], 2.0)


class TestLoadJob(unittest.TestCase):
    def setUp(self):
        control.clear_all_jobs()

    def test_start_load_claims_a_slot(self):
        with mock.patch("control.threading.Thread"):
            self.assertTrue(control.start_load("m"))
            self.assertFalse(control.start_load("m"))

    def test_load_marks_done_on_load_reason(self):
        with mock.patch("control.ollama.api_post",
                        return_value={"done": True, "done_reason": "load"}):
            control._run_load("m", {"model": "m", "prompt": ""})
        job = control.get_jobs()["m"]
        self.assertTrue(job["done"])
        self.assertIsNone(job["error"])

    def test_load_records_an_error_when_the_api_fails(self):
        import ollama
        with mock.patch("control.ollama.api_post",
                        side_effect=ollama.OllamaError("HTTP 500")):
            control._run_load("m", {"model": "m", "prompt": ""})
        job = control.get_jobs()["m"]
        self.assertTrue(job["done"])
        self.assertIn("HTTP 500", job["error"])


class TestUnload(unittest.TestCase):
    def test_unload_sends_keep_alive_zero(self):
        with mock.patch("control.ollama.api_post") as post:
            control.unload_model("gemma4:12b")
        payload = post.call_args[0][1]
        self.assertEqual(payload["model"], "gemma4:12b")
        self.assertEqual(payload["keep_alive"], 0)

    def test_unload_all_iterates_loaded_models(self):
        with mock.patch("control.ollama.loaded_models",
                        return_value=[{"model_key": "a:1"}, {"model_key": "b:1"}]), \
             mock.patch("control.ollama.api_post") as post:
            control.unload_all()
        self.assertEqual([c[0][1]["model"] for c in post.call_args_list],
                         ["a:1", "b:1"])


class TestEstimateFit(unittest.TestCase):
    def test_compares_model_size_to_free_vram(self):
        # Replaces `lms load --estimate-only`, which has no Ollama equivalent.
        with mock.patch("control.ollama.library",
                        return_value=[{"model_key": "m", "size": 20_000_000_000}]), \
             mock.patch("control.sources.gpu",
                        return_value={"mem_used": 1000, "mem_total": 32_600}):
            out = control.estimate_fit("m")
        self.assertTrue(out["ok"])
        self.assertEqual(out["model_bytes"], 20_000_000_000)
        self.assertGreater(out["free_bytes"], 0)

    def test_unknown_model(self):
        with mock.patch("control.ollama.library", return_value=[]):
            self.assertFalse(control.estimate_fit("nope")["ok"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_jobs.py -v`
Expected: FAIL — `AttributeError: module 'control' has no attribute 'build_load_payload'`

- [ ] **Step 3: Write the implementation**

In `control.py`, replace the imports and delete lines 62-198 of the original (`# Progress parsing` through the end of `start_download`). The `_JOBS` block above it is unchanged.

```python
"""Mutating actions: loads, unloads, pulls, deletes, and the catalog scrape."""
import json, re, threading, time, urllib.request

from config import (
    CATALOG_TTL_SEC, CATALOG_URL, CATALOG_USER_AGENT, OLLAMA_URL,
)
import ollama
import sources
```

Then:

```python
# Load / unload -------------------------------------------------------------
#
# Ollama has no dedicated load endpoint. POST /api/generate with an empty
# prompt loads the model and returns {"done_reason": "load"} without
# generating; keep_alive: 0 on the same endpoint unloads it. Verified against
# 0.32.5 — num_ctx and num_gpu are honoured per load, because Ollama keys a
# distinct runner per option set.
#
# There is no per-load parallelism setting and no custom instance identifier:
# OLLAMA_NUM_PARALLEL is server-wide and instance names do not exist.

def _int_or_none(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def build_load_payload(model, context=None, gpu=None, ttl=None):
    payload = {"model": model, "prompt": ""}
    options = {}
    num_ctx = _int_or_none(context)
    num_gpu = _int_or_none(gpu)
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    if options:
        payload["options"] = options
    if ttl not in (None, ""):
        payload["keep_alive"] = ttl
    return payload


def _run_load(key, payload):
    _update_job(key, status="running")
    try:
        resp = ollama.api_post("/api/generate", payload, timeout=900)
    except ollama.OllamaError as e:
        _update_job(key, done=True, finished=time.time(), status="failed",
                    error=str(e))
        return
    reason = resp.get("done_reason")
    _update_job(key, done=True, finished=time.time(), status="finished",
                last_line=f"done_reason={reason}",
                error=None if reason == "load" else f"unexpected reason {reason!r}")


def start_load(model, **opts):
    if not _claim_job(model, "load"):
        return False
    payload = build_load_payload(model, **opts)
    threading.Thread(target=_run_load, args=(model, payload), daemon=True).start()
    return True


def unload_model(name):
    ollama.api_post("/api/generate", {"model": name, "keep_alive": 0}, timeout=60)


def unload_all():
    for m in ollama.loaded_models():
        if m.get("model_key"):
            unload_model(m["model_key"])


def estimate_fit(model):
    """Model size against free VRAM.

    Replaces `lms load --estimate-only`, which Ollama has no equivalent for.
    This is a size comparison, not a real estimate — it ignores KV cache and
    context, so it is presented as a fit indicator rather than a prediction.
    """
    entry = next((m for m in ollama.library() if m["model_key"] == model), None)
    if entry is None:
        return {"ok": False, "error": f"unknown model {model}"}
    g = sources.gpu()
    if "error" in g:
        return {"ok": False, "error": g["error"]}
    free = max(0, (g["mem_total"] - g["mem_used"])) * 1024 * 1024  # MiB -> bytes
    return {"ok": True, "model": model, "model_bytes": entry["size"],
            "free_bytes": free, "fits": entry["size"] < free}


# Download ------------------------------------------------------------------
#
# POST /api/pull streams newline-delimited JSON. Progress is reported PER BLOB
# DIGEST, not per model:
#
#   {"status":"pulling manifest"}
#   {"status":"pulling 970aa74c0a90","digest":"sha256:970a…",
#    "total":274290656,"completed":274290656}
#   {"status":"verifying sha256 digest"}
#
# Reporting the newest completed/total pair would make the bar jump backwards
# at every layer boundary, so PullProgress sums across digests instead.
# Statuses carrying neither field are indeterminate phases and must leave the
# last known percentage alone rather than resetting it.

class PullProgress:
    def __init__(self):
        self._layers = {}          # digest -> (completed, total)
        self._status = ""
        self._first = None         # (time, completed) for rate derivation
        self._last = None

    def update(self, event, now=None):
        now = time.time() if now is None else now
        status = event.get("status")
        if status:
            self._status = status
        digest = event.get("digest")
        if digest is None or "total" not in event:
            return                 # indeterminate phase; keep what we have
        self._layers[digest] = (event.get("completed") or 0, event.get("total") or 0)
        done = sum(c for c, _ in self._layers.values())
        if self._first is None:
            self._first = (now, done)
        self._last = (now, done)

    def snapshot(self):
        completed = sum(c for c, _ in self._layers.values())
        total = sum(t for _, t in self._layers.values())
        pct = round(completed / total * 100, 1) if total else None
        rate = None
        eta = None
        if self._first and self._last:
            dt = self._last[0] - self._first[0]
            db = self._last[1] - self._first[1]
            if dt > 0 and db > 0:
                rate = db / dt
                if total > completed:
                    eta = (total - completed) / rate
        return {"completed": completed, "total": total, "pct": pct,
                "rate_bps": rate, "eta_s": eta, "last_line": self._status}


def _run_pull(key, name):
    _update_job(key, status="running")
    progress = PullProgress()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/pull", data=json.dumps({"model": name}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("error"):
                    _update_job(key, done=True, finished=time.time(),
                                status="failed", error=event["error"])
                    return
                progress.update(event)
                _update_job(key, **progress.snapshot())
    except Exception as e:
        _update_job(key, done=True, finished=time.time(), status="failed",
                    error=str(e))
        return
    _update_job(key, done=True, finished=time.time(), status="finished",
                error=None)


def start_download(name):
    if not _claim_job(name, "download"):
        return False
    threading.Thread(target=_run_pull, args=(name, name), daemon=True).start()
    return True
```

Update `_claim_job`'s initial dict to carry the new keys:

```python
        _JOBS[key] = {"kind": kind, "status": "starting", "pct": None,
                      "completed": 0, "total": 0, "rate_bps": None,
                      "eta_s": None, "error": None, "done": False,
                      "started": time.time(), "finished": None, "last_line": ""}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_jobs.py -v`
Expected: PASS (20 tests)

- [ ] **Step 5: Commit**

```bash
git add control.py tests/test_jobs.py
git commit -m "Replace lms jobs with Ollama load, unload, and pull"
```

---

### Task 8: `control.py` — delete and catalog

**Files:**
- Modify: `control.py` (delete the benchmark and filesystem-delete sections; replace the catalog parser)
- Modify: `tests/test_delete.py` (rewrite, much smaller)
- Modify: `tests/test_catalog.py` (rewrite)
- Delete: `tests/test_scenarios.py`
- Create: `tests/fixtures/ollama_library.html`
- Delete: `tests/fixtures/catalog_page.html`, `tests/fixtures/chat_completion.json`

**Interfaces:**
- Produces: `delete_model(name, confirm) -> dict`, `parse_catalog_html(html) -> list[dict]`, `catalog(force=False) -> dict`
- Catalog row shape: `{slug, name, description, sizes, capabilities}`

- [ ] **Step 1: Capture the fixture**

```bash
curl -s -A "ollama-dashboard/1.0" https://ollama.com/library \
  > tests/fixtures/ollama_library.html
git rm tests/fixtures/catalog_page.html tests/fixtures/chat_completion.json
git rm tests/test_scenarios.py
```

- [ ] **Step 2: Write the failing test**

Replace `tests/test_delete.py`:

```python
import unittest
from unittest import mock

import control
import ollama


class TestDeleteModel(unittest.TestCase):
    def test_confirmation_must_match_exactly(self):
        self.assertFalse(control.delete_model("gemma4:12b", "gemma4")["ok"])
        self.assertFalse(control.delete_model("gemma4:12b", None)["ok"])
        self.assertFalse(control.delete_model("gemma4:12b", "")["ok"])

    def test_refuses_to_delete_a_loaded_model(self):
        with mock.patch("control.ollama.loaded_models",
                        return_value=[{"model_key": "gemma4:12b"}]):
            out = control.delete_model("gemma4:12b", "gemma4:12b")
        self.assertFalse(out["ok"])
        self.assertIn("unload", out["error"])

    def test_calls_the_delete_api(self):
        with mock.patch("control.ollama.loaded_models", return_value=[]), \
             mock.patch("control.ollama.api_delete", return_value={}) as api:
            out = control.delete_model("gemma4:12b", "gemma4:12b")
        self.assertTrue(out["ok"])
        api.assert_called_once_with("/api/delete", {"model": "gemma4:12b"})

    def test_reports_api_failure(self):
        with mock.patch("control.ollama.loaded_models", return_value=[]), \
             mock.patch("control.ollama.api_delete",
                        side_effect=ollama.OllamaError("HTTP 404")):
            out = control.delete_model("nope:1", "nope:1")
        self.assertFalse(out["ok"])
        self.assertIn("HTTP 404", out["error"])


if __name__ == "__main__":
    unittest.main()
```

Replace `tests/test_catalog.py`:

```python
import unittest
from unittest import mock

import control
from tests.helpers import fixture


class TestParseCatalogHtml(unittest.TestCase):
    def setUp(self):
        self.rows = control.parse_catalog_html(fixture("ollama_library.html"))

    def test_finds_many_models(self):
        self.assertGreater(len(self.rows), 100)

    def test_slugs_are_unique(self):
        slugs = [r["slug"] for r in self.rows]
        self.assertEqual(len(slugs), len(set(slugs)))

    def test_extracts_name_and_description(self):
        by_slug = {r["slug"]: r for r in self.rows}
        self.assertIn("llama3.1", by_slug)
        row = by_slug["llama3.1"]
        self.assertEqual(row["name"], "llama3.1")
        self.assertIn("Llama 3.1", row["description"])

    def test_extracts_parameter_sizes(self):
        by_slug = {r["slug"]: r for r in self.rows}
        self.assertEqual(by_slug["llama3.1"]["sizes"], ["8b", "70b", "405b"])

    def test_extracts_capability_badges(self):
        by_slug = {r["slug"]: r for r in self.rows}
        self.assertIn("tools", by_slug["llama3.1"]["capabilities"])

    def test_capabilities_and_sizes_do_not_mix(self):
        for r in self.rows:
            self.assertNotIn("tools", r["sizes"])
            self.assertNotIn("vision", r["sizes"])

    def test_empty_html(self):
        self.assertEqual(control.parse_catalog_html(""), [])
        self.assertEqual(control.parse_catalog_html(None), [])


class TestCatalogCache(unittest.TestCase):
    def setUp(self):
        control._CAT_CACHE.update({"data": [], "fetched": 0, "error": None})

    def test_serves_from_cache_within_ttl(self):
        with mock.patch("control._fetch_catalog_html",
                        return_value='<a href="/library/x"></a>') as fetch:
            control.catalog()
            control.catalog()
        self.assertEqual(fetch.call_count, 1)

    def test_force_bypasses_the_cache(self):
        with mock.patch("control._fetch_catalog_html", return_value="") as fetch:
            control.catalog()
            control.catalog(force=True)
        self.assertEqual(fetch.call_count, 2)

    def test_fetch_failure_keeps_the_stale_copy_and_reports(self):
        with mock.patch("control._fetch_catalog_html",
                        return_value=fixture("ollama_library.html")):
            control.catalog()
        with mock.patch("control._fetch_catalog_html",
                        side_effect=Exception("network down")):
            out = control.catalog(force=True)
        self.assertIn("network down", out["error"])
        self.assertTrue(out["data"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_delete.py tests/test_catalog.py -v`
Expected: FAIL — `AttributeError: module 'ollama' has no attribute 'api_delete'`

- [ ] **Step 4: Write the implementation**

Add to `ollama.py`, next to `api_post`:

```python
def api_delete(path, payload, timeout=30):
    req = urllib.request.Request(
        f"{OLLAMA_URL}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="DELETE")
    return _request(req, timeout)
```

In `control.py`, delete the entire `# Benchmark scenarios` section (`_HAYSTACK_NEEDLE` through `run_scenario`) and the entire `# Delete` section (`is_inside` through the old `delete_model`). Replace the delete section with:

```python
# Delete --------------------------------------------------------------------
#
# Ollama has a real delete endpoint, so none of the filesystem machinery the
# LM Studio version needed survives: no model index, no path resolution, no
# rmtree, no root containment checks. The confirmation guard stays.

def delete_model(name, confirm):
    """Delete a model. `confirm` must equal `name` exactly."""
    if not confirm or confirm != name:
        return {"ok": False, "error": "confirmation must match the model name exactly"}
    if any(m.get("model_key") == name for m in ollama.loaded_models()):
        return {"ok": False, "error": f"{name} is loaded — unload it first"}
    try:
        ollama.api_delete("/api/delete", {"model": name})
    except ollama.OllamaError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "removed": [name]}
```

Replace the catalog parser. The page is server-rendered; each card is an
`<a href="/library/<slug]">` block with an `<h2>` name, a `<p>` description, and
badge `<span>`s. Size badges match `^\d+(\.\d+)?[bm]$`; everything else in a
badge slot is a capability.

```python
# ollama.com/library catalog scrape ----------------------------------------

_CAT_CACHE = {"data": [], "fetched": 0, "error": None}
_CAT_LOCK = threading.Lock()

_CAT_CARD_RE = re.compile(
    r'href="/library/([^"]+)"(.*?)(?=href="/library/|\Z)', re.S)
_CAT_NAME_RE = re.compile(r'<span class="group-hover:underline truncate">\s*([^<]+?)\s*</span>')
_CAT_DESC_RE = re.compile(r'<p class="[^"]*break-words[^"]*">\s*(.*?)\s*</p>', re.S)
_CAT_BADGE_RE = re.compile(r'<span[^>]*text-xs font-medium[^>]*>\s*([^<]+?)\s*</span>')
_CAT_SIZE_RE = re.compile(r'^\d+(?:\.\d+)?[bm]$', re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def parse_catalog_html(html):
    out, seen = [], set()
    for m in _CAT_CARD_RE.finditer(html or ""):
        slug, body = m.group(1), m.group(2)
        if slug in seen:
            continue
        name = _CAT_NAME_RE.search(body)
        if not name:
            continue          # not a model card
        seen.add(slug)
        desc = _CAT_DESC_RE.search(body)
        sizes, caps = [], []
        for badge in _CAT_BADGE_RE.findall(body):
            b = _html_unescape(badge)
            target = sizes if _CAT_SIZE_RE.match(b) else caps
            if b not in target:
                target.append(b)
        out.append({
            "slug": slug,
            "name": _html_unescape(name.group(1)),
            "description": _html_unescape(_TAG_RE.sub("", desc.group(1))).strip()
                           if desc else "",
            "sizes": sizes,
            "capabilities": caps,
        })
    return out
```

`_fetch_catalog_html` and `catalog` are unchanged apart from now hitting
`CATALOG_URL = https://ollama.com/library`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_delete.py tests/test_catalog.py -v`
Expected: PASS

If `test_extracts_name_and_description` or the badge tests fail, the page
markup has drifted since capture. Inspect the fixture and adjust the regexes to
match — the fixture is the source of truth, not the regexes above.

- [ ] **Step 6: Commit**

```bash
git add control.py ollama.py tests/test_delete.py tests/test_catalog.py \
        tests/fixtures/ollama_library.html
git commit -m "Use the Ollama delete API and scrape ollama.com/library"
```

---

### Task 9: `server.py` — routes

**Files:**
- Modify: `server.py`
- Modify: `tests/test_server.py`

**Interfaces:**
- Consumes: everything from Tasks 1-8
- Produces: routes `GET /`, `/control`, `/api/state`, `/api/live`, `/api/control/jobs`, `/api/control/catalog`; `POST /api/control/{load,load/fit,unload,download,jobs/clear}`; `DELETE /api/control/model`
- Removed: `POST /api/control/test`, `POST /api/control/load/estimate`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py`:

```python
class TestRemovedRoutes(unittest.TestCase):
    def test_benchmark_route_is_gone(self):
        src = open("server.py", encoding="utf-8").read()
        self.assertNotIn("/api/control/test", src)

    def test_estimate_route_is_gone(self):
        src = open("server.py", encoding="utf-8").read()
        self.assertNotIn("/api/control/load/estimate", src)


class TestLoadOpts(unittest.TestCase):
    def test_only_supported_options_are_forwarded(self):
        # parallel and identifier have no Ollama equivalent. Passing them
        # through would raise TypeError in build_load_payload.
        import server
        h = server.Handler.__new__(server.Handler)
        opts = h._load_opts({"context": 8192, "gpu": 99, "ttl": "30m",
                             "parallel": 4, "identifier": "custom"})
        self.assertEqual(set(opts), {"context", "gpu", "ttl"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_server.py -v`
Expected: FAIL — `/api/control/test` still present in `server.py`

- [ ] **Step 3: Write the implementation**

In `server.py`:

Change the module docstring to `"""Live dashboard for Ollama. Stdlib only."""`.

Narrow `_load_opts`:

```python
    def _load_opts(self, body):
        # Ollama supports num_ctx, num_gpu, and keep_alive per load. There is no
        # per-load parallelism setting and no instance identifier.
        return {k: body.get(k) for k in ("context", "gpu", "ttl")}
```

In `do_POST`, delete the `/api/control/test` branch and replace the estimate branch:

```python
            if self.path == "/api/control/load/fit":
                model = (body.get("model") or "").strip()
                if not model: return self._json(400, {"error": "model required"})
                return self._json(200, control.estimate_fit(model))
```

In `__main__`, swap the startup banner and note that `samplers.start_all()` now also starts the journal follower:

```python
if __name__ == "__main__":
    sources.start_pcie_monitor()
    samplers.start_all()          # also starts the journald follower
    with ThreadedServer((config.HOST, config.PORT), Handler) as s:
        print(f"ollama dashboard on http://{config.HOST}:{config.PORT}")
        s.serve_forever()
```

- [ ] **Step 4: Run the whole suite**

Run: `python3 -m pytest -q`
Expected: PASS, all files. This is the first point since Task 1 where the full suite is green — confirm the count is at or above 150.

- [ ] **Step 5: Smoke-test against the live server**

```bash
python3 server.py &
sleep 5
curl -s localhost:11435/api/state | python3 -m json.tool | head -40
curl -s localhost:11435/api/live  | python3 -m json.tool | head -20
kill %1
```

Confirm `ollama_ok` is `true`, `library` is populated, `stats_5m` has `p95_s`,
and `disk.approximate` reflects whether the process has the `ollama` group.

- [ ] **Step 6: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "Route the HTTP API to Ollama controls"
```

---

### Task 10: Templates

**Files:**
- Modify: `templates/index.html`
- Modify: `templates/control.html`

**Interfaces:**
- Consumes: the `/api/state` and `/api/live` payloads as of Task 9

No new tests — these are templates, covered indirectly by `test_server.py`'s
route checks and by the smoke test. `PRODUCT.md`'s design principles govern
every change here; read it before starting.

- [ ] **Step 1: `index.html` — remove what no longer exists**

Delete the model-activity panel that consumed `state.model_activity`, which no
longer exists in the payload.

- [ ] **Step 2: `index.html` — loaded-model VRAM split badge**

Each loaded-model row gains a badge driven by `m.fully_gpu` and `m.cpu_bytes`.
Per `PRODUCT.md` principle 4, pair colour with text — never colour alone:

```html
<span class="badge" data-state="${m.fully_gpu ? 'ok' : 'warn'}">
  ${m.fully_gpu ? 'GPU' : `CPU spill ${fmtBytes(m.cpu_bytes)}`}
</span>
```

- [ ] **Step 3: `index.html` — request table gains status and latency**

Add `status` and `latency_s` columns to the requests table. Render status with
both colour and text, and format latency with `fmtDuration` (add it next to the
existing `fmtBytes` helper):

```js
function fmtDuration(s) {
  if (s == null) return '—';
  if (s < 1e-3) return (s * 1e6).toFixed(0) + 'µs';
  if (s < 1)    return (s * 1e3).toFixed(1) + 'ms';
  if (s < 60)   return s.toFixed(2) + 's';
  return Math.floor(s / 60) + 'm' + (s % 60).toFixed(0) + 's';
}
```

- [ ] **Step 4: `index.html` — stats card gains error rate and percentiles**

The existing `stats_5m` card shows count and rps. Add `error_rate`, `p50_s`,
`p95_s`, `p99_s`. Per principle 2, this is immediate state — keep it above the
request table, not inside it.

- [ ] **Step 5: `index.html` — add the client breakdown**

A compact table over `state.by_client`: client, count, errors, last seen. This
is new signal; place it beside `top_endpoints`, which gains `errors` and
`p95_s` columns.

- [ ] **Step 6: `index.html` — surface staleness and approximation**

Two honesty affordances the payload now carries and the UI must not hide:

- When `state.log_age_s` exceeds 30, mark the request panel stale rather than
  presenting a frozen window as current.
- When `state.disk.approximate` is true, label the model-store figure as a
  lower bound. When `state.disk.orphan_bytes` is non-null and above zero, show
  it as reclaimable.

- [ ] **Step 7: `index.html` — rebuild the settings card**

`state.settings` is now `{unit, OLLAMA_*...}` instead of LM Studio's fixed
keys. Render it as a name/value list over whatever `OLLAMA_*` keys are present,
since the set is not fixed. Values of `<redacted>` render as-is.

- [ ] **Step 8: `index.html` — model attribution must read as inferred**

Where a request row shows `m.model`, label the column "model (inferred)" and
render null as "—". This is a design requirement, not a cosmetic one: the value
is derived from a residency timeline, not observed per request.

- [ ] **Step 9: `control.html` — remove the benchmark panel**

Delete the whole benchmark section and its `/api/control/test` calls: the
scenario picker, the custom-prompt box, and the results pane.

- [ ] **Step 10: `control.html` — narrow the load panel**

Remove the parallel and identifier inputs; keep context, GPU layers, and TTL.
Relabel to Ollama's terms — context becomes `num_ctx`, GPU becomes
`num_gpu` (layer count, where 99 means "all"), TTL becomes `keep_alive`.
Add a read-only line showing `OLLAMA_NUM_PARALLEL` and
`OLLAMA_MAX_LOADED_MODELS` from `state.settings`, so their absence from the
form reads as "server-wide setting" rather than "missing feature".

- [ ] **Step 11: `control.html` — replace the estimate button with fit**

Point the button at `POST /api/control/load/fit` and render
`{model_bytes, free_bytes, fits}`. Label it a fit check, not an estimate — it
ignores KV cache and context, and must not imply otherwise.

- [ ] **Step 12: `control.html` — catalog rows gain descriptions**

Catalog rows now carry `description` and `capabilities` alongside `sizes`.
Render the description as secondary text and the capabilities as badges,
matching the loaded-model badge treatment.

- [ ] **Step 13: Verify in a browser**

```bash
python3 server.py &
sleep 5
xdg-open http://127.0.0.1:11435/ 2>/dev/null || true
```

Walk both pages: load a model with an explicit `num_ctx`, watch the job
progress, pull a small model (`ollama.com/library` has `all-minilm` at ~45 MB)
and confirm the progress bar rises monotonically across layers, then unload and
delete. Kill the server when done.

- [ ] **Step 14: Commit**

```bash
git add templates/
git commit -m "Rework the dashboard and control panel for Ollama"
```

---

### Task 11: Rename, docs, and deployment

**Files:**
- Modify: `README.md`, `PRODUCT.md`
- Create: `ollama-dashboard.service`
- Rename: the repository directory

**Interfaces:** none — this is the cutover.

- [ ] **Step 1: Confirm no LM Studio references remain in code**

```bash
grep -rniE 'lmstudio|lm studio|lms ' --include='*.py' --include='*.html' . \
  | grep -v '^\./docs/' | grep -v '^\./\.git/'
```

Expected: no output. Matches inside `docs/superpowers/` are historical records
of the LM Studio era and must be left alone.

- [ ] **Step 2: Rewrite `README.md`**

Rewrite for Ollama. It must cover: what the dashboard shows; that it talks to
Ollama over HTTP on 11434 and serves on 11435; the `OLLAMA_DASHBOARD_*`
variables from Task 1 with their defaults; that logs come from journald and the
running user needs the `adm` group; that exact disk figures need the `ollama`
group and the dashboard degrades to an approximate figure without it; and the
security warning about binding `0.0.0.0`.

Delete the LM Studio sections wholesale: `lms` CLI setup, `settings.json`,
the model index, and the benchmark scenarios.

- [ ] **Step 3: Update `PRODUCT.md`**

Change only the product name and the two sentences naming LM Studio in
**Users** and **Product Purpose**. The Brand Personality, Anti-references,
Design Principles, and Accessibility sections are unchanged — none of them
depend on which model server sits underneath.

- [ ] **Step 4: Write the systemd unit**

Create `ollama-dashboard.service`:

```ini
[Unit]
Description=Ollama Dashboard
Documentation=file:/opt/ollama-dashboard/README.md
# A *user* unit. Ollama itself is a system unit, so this cannot Require it —
# a user unit cannot depend on a system one. It degrades gracefully when
# Ollama is down, which is a state the dashboard is meant to display.
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/ollama-dashboard
# Bound to all interfaces: LAN, tailnet, and localhost. Note this exposes the
# control panel — which can load, download and delete models, with no auth — to
# anything that can reach this host on the LAN.
Environment=OLLAMA_DASHBOARD_HOST=0.0.0.0
Environment=OLLAMA_DASHBOARD_PORT=11435
Environment=OLLAMA_URL=http://localhost:11434
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 /opt/ollama-dashboard/server.py
Restart=always
RestartSec=5
StartLimitIntervalSec=0

[Install]
WantedBy=default.target
```

- [ ] **Step 5: Commit the code changes**

```bash
git add README.md PRODUCT.md ollama-dashboard.service
git commit -m "Rewrite docs and ship an ollama-dashboard unit"
```

- [ ] **Step 6: Merge to main and rename the directory**

```bash
cd ~/GitHub/lmstudio-dashboard
python3 -m pytest -q                      # green before merging
git checkout main && git merge --no-ff ollama-port -m "Port the dashboard to Ollama"
cd ~/GitHub && mv lmstudio-dashboard ollama-dashboard
cd ollama-dashboard && python3 -m pytest -q
```

- [ ] **Step 7: Deploy and cut over**

```bash
systemctl --user disable --now lmstudio-dashboard.service 2>/dev/null || true
rm -f ~/.config/systemd/user/lmstudio-dashboard.service
sudo rm -rf /opt/lmstudio-dashboard
sudo mkdir -p /opt/ollama-dashboard
sudo cp -r ~/GitHub/ollama-dashboard/*.py ~/GitHub/ollama-dashboard/templates \
           ~/GitHub/ollama-dashboard/README.md /opt/ollama-dashboard/
cp ~/GitHub/ollama-dashboard/ollama-dashboard.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ollama-dashboard.service
systemctl --user status ollama-dashboard.service --no-pager
curl -s localhost:11435/api/state | python3 -m json.tool | head -20
```

- [ ] **Step 8: Verify the deployed service has the `ollama` group**

The unit inherits the login session's groups. If `disk.approximate` is `true`
after deploying, the group has not propagated — log out and back in, then
restart the unit:

```bash
curl -s localhost:11435/api/state | python3 -c \
  "import json,sys; d=json.load(sys.stdin)['disk']; print('approximate:', d['approximate'], 'orphans:', d['orphan_bytes'])"
```

- [ ] **Step 9: Commit any deployment fixes**

```bash
git add -A && git commit -m "Fix deployment issues found during cutover" || true
```

---

## Self-Review

**Spec coverage.** Every section maps to a task: `ollama.py` → 2; samplers
docstring and timeline → 5; logs rewrite → 3 and 4; model attribution → 5 and
template step 8; `control.py` deletions → 7 and 8; per-load options → 7;
catalog → 8; `sources.py` settings, disk, service → 6; `config.py` → 1;
`server.py` and templates → 9 and 10; testing → throughout; migration → 11;
risks → template steps 6 and 8, and Task 11 step 8.

**Two spec claims needed correcting during planning.** The spec said `state()`
would call `ollama.ping()` directly, which violates the never-call-Ollama-from-
the-request-path constraint; Task 6 routes it through a `PING` sampler instead.
The spec's `NOISE_PATHS` filtering was path-only, which would have erased the
Open WebUI container's polling from the client breakdown; Task 3 filters on
path **and** loopback.

**Type consistency.** `model_key` is the model-name key everywhere.
`normalize_loaded` and `normalize_library` both emit it; `join_library`,
`samplers.record_timeline`, `control.unload_all`, `control.delete_model`, and
`control.estimate_fit` all consume it. Latencies are `latency_s` in seconds
throughout; `parse_duration` returns seconds; `stats` exposes `p50_s/p95_s/p99_s`.
Job dicts carry `pct/completed/total/rate_bps/eta_s/last_line`, matching
`PullProgress.snapshot()` exactly, which is what makes `_update_job(**snapshot)`
valid.

**One deliberate deviation from TDD.** Task 10 has no tests. Templates are not
unit-testable in a stdlib-only project without adding a browser dependency,
which the global constraints forbid. They are covered by the route assertions in
Task 9 and by the manual walkthrough in step 13.

---

**Plan complete and saved to `docs/superpowers/plans/2026-07-30-ollama-port.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
