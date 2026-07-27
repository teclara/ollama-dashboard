# LM Studio Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Ollama with LM Studio as the dashboard's backend, keeping every panel that has a data source, removing those that don't, and adding LM Studio's model-loading controls.

**Architecture:** Same stdlib-only HTTP server. Two new modules isolate the LM Studio coupling: `lmstudio.py` wraps the `lms` CLI and the `/api/v0` HTTP API; `logs.py` owns server-log discovery and parsing. `sources.py` keeps the hardware/OS panels and composes the aggregate `state()`. `control.py` holds mutating actions.

**Tech Stack:** Python 3.9+ standard library only. `unittest` for tests. `subprocess` to the `lms` CLI. No third-party packages, ever.

## Global Constraints

- **Python 3 standard library only.** No pip installs, no `requirements.txt`. `subprocess`, `urllib.request`, `json`, `re`, `unittest` are all fair game.
- **Every source function catches broadly** and returns an empty/error-tagged result. One failing panel must never take down `/api/state`.
- **Every `lms` invocation gets an explicit timeout.** Reads: 2–5s. `lms get` / `lms load` run in threads with no wall-clock timeout.
- **Never emit `settings.json` wholesale.** Whitelist keys. `hfSearchToken` and `hfDownloadToken` must never reach a response body.
- **All env vars are `LMSTUDIO_*`.** No `OLLAMA_*` names survive, and no backward-compat shim.
- **The word "Ollama" appears nowhere** in code, templates, or README when the migration is done, except in git history.
- **Tests run with `python3 -m unittest discover -s tests -v`** from the repo root and must pass with LM Studio stopped — fixtures only, no live server.
- **Fixtures already exist** in `tests/fixtures/` (committed): `lms_ps_loaded.json`, `lms_ps_empty.json`, `lms_ls.json`, `api_v0_models.json`, `lms_runtime_ls.txt`, `model_index_cache.json`, `server_log_excerpt.log`, `chat_completion.json`, `catalog_page.html`.

---

## File Structure

| File | Responsibility |
|---|---|
| `config.py` | Environment-driven settings. Rewritten: `LMSTUDIO_*` names. |
| `lmstudio.py` | **New.** All LM Studio coupling: `lms` CLI runner, `/api/v0` HTTP calls, normalization of `ps`/`ls`/model-index payloads. |
| `logs.py` | **New.** Server-log file discovery, line parsing, and windowed aggregation. |
| `sources.py` | Read-only state: GPU, host, PCIe, disk, service, tailscale, settings, and the aggregate `state()`. Delegates models to `lmstudio.py`, requests to `logs.py`. |
| `control.py` | Mutating actions: load, unload, download, delete, benchmark scenarios, catalog scrape. |
| `server.py` | HTTP routing + main. |
| `templates/index.html` | Dashboard UI. |
| `templates/control.html` | Control panel UI. |
| `tests/test_*.py` | `unittest` suites against committed fixtures. |
| `README.md` | Rewritten for LM Studio. |

---

## Task 1: Config rewrite and test scaffolding

**Files:**
- Modify: `config.py` (full rewrite)
- Create: `tests/__init__.py`, `tests/helpers.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `config.HOST`, `config.PORT`, `config.LMSTUDIO_URL`, `config.LMS_BIN`, `config.SYSTEMD_UNIT`, `config.SYSTEMD_USER` (bool), `config.LOG_DIR`, `config.SETTINGS_PATH`, `config.MODEL_INDEX_PATH`, `config.HUB_MODELS_DIR`, `config.MODELS_DIR_FALLBACK`, `config.CATALOG_URL`, `config.CATALOG_TTL_SEC`, `config.CATALOG_USER_AGENT`, `config.HAYSTACK_PATH`, `config.HAYSTACK_WORDS`, `config.GPU_HISTORY_LEN`, `config.PCIE_HISTORY_LEN`, `config.LOG_WINDOW_LINES`, `config.STATS_WINDOW_SEC`, `config.NOISE_PATHS`. Also `tests.helpers.fixture(name)` returning the fixture file's text, and `tests.helpers.fixture_json(name)` returning parsed JSON.

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` as an empty file, then `tests/helpers.py`:

```python
"""Shared test helpers: fixture loading."""
import json, os

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def fixture(name):
    with open(os.path.join(FIXTURE_DIR, name), encoding="utf-8") as f:
        return f.read()


def fixture_json(name):
    return json.loads(fixture(name))
```

Then `tests/test_config.py`:

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
        c = self._fresh(LMSTUDIO_URL=None, LMSTUDIO_DASHBOARD_PORT=None)
        self.assertEqual(c.LMSTUDIO_URL, "http://localhost:1234")
        self.assertEqual(c.PORT, 11435)
        self.assertEqual(c.HOST, "127.0.0.1")
        self.assertTrue(c.SYSTEMD_USER)
        self.assertEqual(c.SYSTEMD_UNIT, "lmstudio-server")
        self.assertTrue(c.LMS_BIN.endswith("/.lmstudio/bin/lms"))
        self.assertNotIn("~", c.LOG_DIR)  # expanded

    def test_env_override(self):
        c = self._fresh(LMSTUDIO_URL="http://box:4321/", LMSTUDIO_DASHBOARD_PORT="9000")
        self.assertEqual(c.LMSTUDIO_URL, "http://box:4321")  # trailing slash stripped
        self.assertEqual(c.PORT, 9000)

    def test_bad_int_falls_back_to_default(self):
        c = self._fresh(LMSTUDIO_DASHBOARD_PORT="not-a-number")
        self.assertEqual(c.PORT, 11435)

    def test_systemd_user_is_boolean_from_string(self):
        self.assertFalse(self._fresh(LMSTUDIO_SYSTEMD_USER="0").SYSTEMD_USER)
        self.assertTrue(self._fresh(LMSTUDIO_SYSTEMD_USER="1").SYSTEMD_USER)

    def test_no_ollama_names_remain(self):
        c = self._fresh()
        leftovers = [n for n in dir(c) if "OLLAMA" in n.upper()]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_config -v`
Expected: FAIL — `AttributeError: module 'config' has no attribute 'LMSTUDIO_URL'`

- [ ] **Step 3: Write the implementation**

Replace `config.py` entirely:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_config -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add config.py tests/__init__.py tests/helpers.py tests/test_config.py
git commit -m "Rewrite config for LM Studio env vars"
```

---

## Task 2: `lmstudio.py` — model state

**Files:**
- Create: `lmstudio.py`
- Create: `tests/test_lmstudio.py`

**Interfaces:**
- Consumes: `config.LMS_BIN`, `config.LMSTUDIO_URL`, `config.MODEL_INDEX_PATH`
- Produces:
  - `LmsError(Exception)`
  - `run_lms(*args, timeout=5) -> str` — raises `LmsError` on failure/missing binary
  - `run_lms_json(*args, timeout=5) -> list|dict` — raises `LmsError` on bad JSON
  - `normalize_loaded(raw: list) -> list[dict]` — pure
  - `normalize_disk(raw: list) -> list[dict]` — pure
  - `api_states(payload: dict) -> dict[str, str]` — pure; model id → `"loaded"`/`"not-loaded"`
  - `join_library(disk: list, states: dict) -> list[dict]` — pure
  - `parse_engine(runtime_ls_text: str) -> dict` — pure; `{"name","version"}`
  - `loaded_models() -> list[dict]`, `library() -> list[dict]`, `engine_info() -> dict` — live wrappers
  - `model_index() -> dict` — fresh read of `MODEL_INDEX_PATH`, never cached

Normalized loaded-model shape (every key always present):
`{identifier, model_key, indexed_id, display_name, publisher, arch, quant, quant_bits, params, size, context, max_context, ttl_s, status, queued, parallel, last_used, vision, tools, type}`

Normalized disk-model shape:
`{model_key, indexed_id, display_name, publisher, arch, quant, quant_bits, params, size, max_context, vision, tools, type, loaded}`

**On the three identifiers** — these are not interchangeable and mixing them up is the single easiest way to break this migration:

| Field | Source | Used for |
|---|---|---|
| `model_key` | `modelKey` | Display, and as the argument to `lms load` |
| `identifier` | `identifier` | The loaded instance name — what `lms unload` takes. Differs from `model_key` when loaded via `--identifier` |
| `indexed_id` | `indexedModelIdentifier` | The **only** key that joins to `model-index-cache.json`, and therefore the only one delete may use |

Verified against the fixtures: `modelKey` matches an index entry for just 2 of 5 models on this machine (`cyberpal2.0-20b-i1`'s index key is the full `mradermacher/CyberPal2.0-20B-i1-GGUF/…gguf` path), while `indexedModelIdentifier` matches for all 5.

- [ ] **Step 1: Write the failing test**

Create `tests/test_lmstudio.py`:

```python
import unittest

import lmstudio
from tests.helpers import fixture, fixture_json


class TestNormalizeLoaded(unittest.TestCase):
    def test_maps_real_payload(self):
        out = lmstudio.normalize_loaded(fixture_json("lms_ps_loaded.json"))
        self.assertEqual(len(out), 1)
        m = out[0]
        self.assertEqual(m["identifier"], "google/gemma-4-31b")
        self.assertEqual(m["model_key"], "google/gemma-4-31b")
        self.assertEqual(m["display_name"], "Gemma 4 31B")
        self.assertEqual(m["arch"], "gemma4")
        self.assertEqual(m["quant"], "Q4_K_M")
        self.assertEqual(m["quant_bits"], 4)
        self.assertEqual(m["params"], "31B")
        self.assertEqual(m["size"], 19887882864)
        self.assertEqual(m["context"], 32768)
        self.assertEqual(m["max_context"], 262144)
        self.assertEqual(m["status"], "idle")
        self.assertEqual(m["queued"], 0)
        self.assertEqual(m["parallel"], 4)
        self.assertTrue(m["vision"])
        self.assertTrue(m["tools"])

    def test_null_ttl_becomes_none_not_zero(self):
        # ttlMs is null when no TTL is set; 0 would render as "expires now"
        m = lmstudio.normalize_loaded(fixture_json("lms_ps_loaded.json"))[0]
        self.assertIsNone(m["ttl_s"])

    def test_empty_payload(self):
        self.assertEqual(lmstudio.normalize_loaded(fixture_json("lms_ps_empty.json")), [])

    def test_tolerates_missing_optional_fields(self):
        out = lmstudio.normalize_loaded([{"modelKey": "bare"}])
        self.assertEqual(out[0]["model_key"], "bare")
        self.assertIsNone(out[0]["quant"])
        self.assertIsNone(out[0]["quant_bits"])
        self.assertEqual(out[0]["size"], 0)
        self.assertFalse(out[0]["vision"])

    def test_ttl_ms_converts_to_seconds(self):
        out = lmstudio.normalize_loaded([{"modelKey": "x", "ttlMs": 300000}])
        self.assertEqual(out[0]["ttl_s"], 300)


class TestNormalizeDisk(unittest.TestCase):
    def test_maps_real_payload(self):
        out = lmstudio.normalize_disk(fixture_json("lms_ls.json"))
        keys = {m["model_key"] for m in out}
        self.assertIn("google/gemma-4-31b", keys)
        self.assertIn("cyberpal2.0-20b-i1", keys)
        gemma = next(m for m in out if m["model_key"] == "google/gemma-4-31b")
        self.assertEqual(gemma["arch"], "gemma4")
        self.assertTrue(gemma["vision"])

    def test_indexed_id_is_carried_and_differs_from_model_key(self):
        """indexed_id is the only key that joins to the model index. Delete needs it."""
        out = lmstudio.normalize_disk(fixture_json("lms_ls.json"))
        by_key = {m["model_key"]: m for m in out}
        self.assertEqual(
            by_key["cyberpal2.0-20b-i1"]["indexed_id"],
            "mradermacher/CyberPal2.0-20B-i1-GGUF/CyberPal2.0-20B.i1-MXFP4_MOE.gguf")
        # For catalog models the two happen to coincide
        self.assertEqual(by_key["google/gemma-4-31b"]["indexed_id"], "google/gemma-4-31b")

    def test_every_model_carries_an_indexed_id(self):
        out = lmstudio.normalize_disk(fixture_json("lms_ls.json"))
        self.assertTrue(all(m["indexed_id"] for m in out))

    def test_sorted_by_model_key(self):
        out = lmstudio.normalize_disk(fixture_json("lms_ls.json"))
        self.assertEqual([m["model_key"] for m in out],
                         sorted(m["model_key"] for m in out))


class TestApiStates(unittest.TestCase):
    def test_extracts_load_state(self):
        states = lmstudio.api_states(fixture_json("api_v0_models.json"))
        self.assertEqual(states["google/gemma-4-31b"], "loaded")
        self.assertEqual(states["qwen/qwen3.6-27b"], "not-loaded")

    def test_malformed_payload_yields_empty(self):
        self.assertEqual(lmstudio.api_states({}), {})
        self.assertEqual(lmstudio.api_states({"data": "nonsense"}), {})


class TestJoinLibrary(unittest.TestCase):
    def test_marks_loaded_rows(self):
        disk = lmstudio.normalize_disk(fixture_json("lms_ls.json"))
        states = lmstudio.api_states(fixture_json("api_v0_models.json"))
        out = lmstudio.join_library(disk, states)
        gemma = next(m for m in out if m["model_key"] == "google/gemma-4-31b")
        self.assertTrue(gemma["loaded"])
        qwen = next(m for m in out if m["model_key"] == "qwen/qwen3.6-27b")
        self.assertFalse(qwen["loaded"])

    def test_model_missing_from_states_defaults_to_not_loaded(self):
        out = lmstudio.join_library([{"model_key": "ghost"}], {})
        self.assertFalse(out[0]["loaded"])

    def test_state_for_unknown_model_does_not_crash(self):
        disk = [{"model_key": "a"}]
        out = lmstudio.join_library(disk, {"b": "loaded"})
        self.assertEqual(len(out), 1)


class TestParseEngine(unittest.TestCase):
    def test_picks_the_selected_runtime(self):
        eng = lmstudio.parse_engine(fixture("lms_runtime_ls.txt"))
        self.assertEqual(eng["name"], "llama.cpp-linux-x86_64-nvidia-cuda12-avx2")
        self.assertEqual(eng["version"], "2.27.1")

    def test_no_selection_yields_empty(self):
        self.assertEqual(lmstudio.parse_engine("LLM ENGINE   SELECTED\nfoo@1.0\n"),
                         {"name": None, "version": None})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_lmstudio -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lmstudio'`

- [ ] **Step 3: Write the implementation**

Create `lmstudio.py`:

```python
"""All LM Studio coupling: the `lms` CLI, the /api/v0 HTTP API, and payload normalization.

Nothing else in the codebase should know LM Studio's field names.
"""
import json, os, re, subprocess, urllib.request

from config import LMS_BIN, LMSTUDIO_URL, MODEL_INDEX_PATH


class LmsError(Exception):
    """The `lms` CLI is missing, failed, or returned something unparseable."""


def run_lms(*args, timeout=5):
    try:
        return subprocess.check_output(
            [LMS_BIN, *args], text=True, timeout=timeout, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise LmsError(f"lms CLI not found at {LMS_BIN}")
    except subprocess.TimeoutExpired:
        raise LmsError(f"lms {' '.join(args)} timed out after {timeout}s")
    except subprocess.CalledProcessError as e:
        raise LmsError(f"lms {' '.join(args)} exited {e.returncode}")


def run_lms_json(*args, timeout=5):
    out = run_lms(*args, timeout=timeout)
    try:
        return json.loads(out)
    except ValueError as e:
        raise LmsError(f"lms {' '.join(args)} returned invalid JSON: {e}")


# Normalization (pure) ------------------------------------------------------

def _quant(m):
    q = m.get("quantization")
    if isinstance(q, dict): return q.get("name"), q.get("bits")
    if isinstance(q, str): return q, None
    return None, None


def normalize_loaded(raw):
    """`lms ps --json` -> stable internal shape."""
    out = []
    for m in raw or []:
        name, bits = _quant(m)
        ttl_ms = m.get("ttlMs")
        out.append({
            "identifier": m.get("identifier") or m.get("modelKey"),
            "model_key": m.get("modelKey"),
            "indexed_id": m.get("indexedModelIdentifier"),
            "display_name": m.get("displayName") or m.get("modelKey"),
            "publisher": m.get("publisher"),
            "arch": m.get("architecture"),
            "quant": name,
            "quant_bits": bits,
            "params": m.get("paramsString"),
            "size": m.get("sizeBytes") or 0,
            "context": m.get("contextLength"),
            "max_context": m.get("maxContextLength"),
            "ttl_s": int(ttl_ms / 1000) if ttl_ms else None,
            "status": m.get("status"),
            "queued": m.get("queued") or 0,
            "parallel": m.get("parallel"),
            "last_used": m.get("lastUsedTime"),
            "vision": bool(m.get("vision")),
            "tools": bool(m.get("trainedForToolUse")),
            "type": m.get("type"),
        })
    return out


def normalize_disk(raw):
    """`lms ls --json` -> stable internal shape, sorted by model key."""
    out = []
    for m in raw or []:
        name, bits = _quant(m)
        out.append({
            "model_key": m.get("modelKey"),
            # The join key for model-index-cache.json. Often differs from modelKey.
            "indexed_id": m.get("indexedModelIdentifier"),
            "display_name": m.get("displayName") or m.get("modelKey"),
            "publisher": m.get("publisher"),
            "arch": m.get("architecture"),
            "quant": name,
            "quant_bits": bits,
            "params": m.get("paramsString"),
            "size": m.get("sizeBytes") or 0,
            "max_context": m.get("maxContextLength"),
            "vision": bool(m.get("vision")),
            "tools": bool(m.get("trainedForToolUse")),
            "type": m.get("type"),
            "loaded": False,
        })
    return sorted(out, key=lambda x: x["model_key"] or "")


def api_states(payload):
    """`GET /api/v0/models` -> {model id: state}."""
    data = (payload or {}).get("data")
    if not isinstance(data, list): return {}
    return {m["id"]: m.get("state") for m in data
            if isinstance(m, dict) and m.get("id")}


def join_library(disk, states):
    for m in disk:
        m["loaded"] = states.get(m.get("model_key")) == "loaded"
    return disk


_ENGINE_RE = re.compile(r"^(?P<name>\S+)@(?P<version>\S+)\s+✓")


def parse_engine(text):
    """Pick the ✓-marked runtime out of `lms runtime ls`."""
    for line in (text or "").splitlines():
        m = _ENGINE_RE.match(line.strip())
        if m: return {"name": m.group("name"), "version": m.group("version")}
    return {"name": None, "version": None}


# Live wrappers -------------------------------------------------------------

def loaded_models():
    try:
        return normalize_loaded(run_lms_json("ps", "--json", timeout=5))
    except LmsError:
        return []


def _api_models():
    try:
        with urllib.request.urlopen(f"{LMSTUDIO_URL}/api/v0/models", timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def library():
    """Models on disk, annotated with load state.

    Reads disk via the CLI so the library still renders when the server is down.
    """
    try:
        disk = normalize_disk(run_lms_json("ls", "--json", timeout=5))
    except LmsError:
        return []
    return join_library(disk, api_states(_api_models()))


def engine_info():
    try:
        return parse_engine(run_lms("runtime", "ls", timeout=5))
    except LmsError:
        return {"name": None, "version": None}


def model_index():
    """Fresh read of LM Studio's model index. Never cached — it is a cache itself."""
    try:
        with open(MODEL_INDEX_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"models": []}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_lmstudio -v`
Expected: PASS, 14 tests

Note: `parse_engine` matches on the literal `✓` character present in `lms runtime ls` output. The fixture is stored with ANSI colour codes already stripped; `run_lms` output is not coloured when stdout is a pipe, so no stripping is needed at runtime. If the live check in Task 12 shows escape codes, add `re.sub(r"\x1b\[[0-9;]*m", "", text)` at the top of `parse_engine` and add a test with a coloured line.

- [ ] **Step 5: Commit**

```bash
git add lmstudio.py tests/test_lmstudio.py
git commit -m "Add lmstudio module wrapping the lms CLI and /api/v0"
```

---

## Task 3: `logs.py` — parsing and aggregation

**Files:**
- Create: `logs.py`
- Create: `tests/test_logs.py`

**Interfaces:**
- Consumes: `config.LOG_DIR`, `config.LOG_WINDOW_LINES`, `config.STATS_WINDOW_SEC`, `config.NOISE_PATHS`
- Produces:
  - `parse_line(line) -> dict|None` — pure
  - `parse_lines(iterable) -> list[dict]` — pure, drops `None`s and noise paths
  - `log_files(log_dir) -> list[str]` — newest-modified first
  - `read_window(log_dir=None, n_lines=None) -> list[dict]`
  - `stats(rows, window_sec=None) -> dict` — `{window_sec, count, rps}`
  - `top_endpoints(rows, window_sec=None, top=8) -> list[dict]` — `{path, count}`
  - `model_activity(rows, window_sec=None) -> list[dict]` — `{model, completions, predictions, tool_calls, streams}`

Row shape: `{ts, epoch, kind, method, path, model, messages}` — keys not applicable to a kind are `None`.

Kinds: `request`, `completion`, `stream_start`, `stream_end`, `prediction`, `tool_calls`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_logs.py`:

```python
import os, tempfile, time, unittest

import logs
from tests.helpers import fixture, FIXTURE_DIR


class TestParseLine(unittest.TestCase):
    def test_request_line(self):
        r = logs.parse_line(
            "[2026-07-25 22:01:27][DEBUG] Received request: POST to /v1/chat/completions with body {")
        self.assertEqual(r["kind"], "request")
        self.assertEqual(r["method"], "POST")
        self.assertEqual(r["path"], "/v1/chat/completions")
        self.assertEqual(r["ts"], "2026-07-25 22:01:27")
        self.assertGreater(r["epoch"], 0)

    def test_request_line_without_body(self):
        r = logs.parse_line("[2026-07-26 22:24:10][DEBUG] Received request: GET to /lmstudio-greeting")
        self.assertEqual(r["method"], "GET")
        self.assertEqual(r["path"], "/lmstudio-greeting")

    def test_completion_line_captures_model_and_message_count(self):
        r = logs.parse_line(
            "[2026-07-25 22:01:28][INFO][qwen/qwen3.6-35b-a3b] "
            "Running chat completion on conversation with 2 messages.")
        self.assertEqual(r["kind"], "completion")
        self.assertEqual(r["model"], "qwen/qwen3.6-35b-a3b")
        self.assertEqual(r["messages"], 2)

    def test_custom_identifier_is_treated_as_the_model(self):
        # `lms load --identifier cpal` makes the bracket tag a custom name
        r = logs.parse_line("[2026-07-26 07:39:32][INFO][cpal] Model generated tool calls: []")
        self.assertEqual(r["kind"], "tool_calls")
        self.assertEqual(r["model"], "cpal")

    def test_stream_start_and_end(self):
        self.assertEqual(
            logs.parse_line("[2026-07-25 22:01:28][INFO][m] Streaming response...")["kind"],
            "stream_start")
        self.assertEqual(
            logs.parse_line("[2026-07-25 22:01:35][INFO][m] Finished streaming response")["kind"],
            "stream_end")

    def test_prediction_line(self):
        r = logs.parse_line("[2026-07-26 00:11:53][INFO][fsec] Generated prediction: {")
        self.assertEqual(r["kind"], "prediction")
        self.assertEqual(r["model"], "fsec")

    def test_authenticator_lines_are_not_models(self):
        r = logs.parse_line(
            "[2026-07-25 22:53:29][INFO][LMSAuthenticator][Client=lms-cli][Endpoint=listLoaded] "
            "Listing loaded models")
        self.assertIsNone(r)

    def test_progress_lines_ignored(self):
        self.assertIsNone(logs.parse_line(
            "[2026-07-25 22:01:33][INFO][m] Prompt processing progress: 6.2%"))

    def test_json_body_continuation_ignored(self):
        self.assertIsNone(logs.parse_line('      "temperature": 0.7,'))
        self.assertIsNone(logs.parse_line("    {"))
        self.assertIsNone(logs.parse_line(""))


class TestParseLines(unittest.TestCase):
    def test_parses_the_real_excerpt(self):
        rows = logs.parse_lines(fixture("server_log_excerpt.log").splitlines())
        kinds = {r["kind"] for r in rows}
        self.assertEqual(kinds, {"request", "completion", "stream_start",
                                 "stream_end", "prediction", "tool_calls"})

    def test_noise_paths_dropped(self):
        rows = logs.parse_lines([
            "[2026-07-25 22:01:27][DEBUG] Received request: GET to /api/v0/models",
            "[2026-07-25 22:01:27][DEBUG] Received request: GET to /lmstudio-greeting",
            "[2026-07-25 22:01:27][DEBUG] Received request: POST to /v1/chat/completions",
        ])
        self.assertEqual([r["path"] for r in rows], ["/v1/chat/completions"])


class TestLogFiles(unittest.TestCase):
    def test_newest_first_across_month_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            for sub, name, mtime in (("2026-06", "2026-06-30.1.log", 1000),
                                     ("2026-07", "2026-07-01.1.log", 2000),
                                     ("2026-07", "2026-07-02.1.log", 3000)):
                os.makedirs(os.path.join(d, sub), exist_ok=True)
                p = os.path.join(d, sub, name)
                open(p, "w").close()
                os.utime(p, (mtime, mtime))
            found = [os.path.basename(p) for p in logs.log_files(d)]
            self.assertEqual(found, ["2026-07-02.1.log", "2026-07-01.1.log", "2026-06-30.1.log"])

    def test_missing_dir_yields_empty(self):
        self.assertEqual(logs.log_files("/nonexistent/path/xyz"), [])


class TestReadWindow(unittest.TestCase):
    def test_spans_two_files_when_newest_is_short(self):
        """A window wider than the newest file must reach into the previous day."""
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "2026-07"))
            old = os.path.join(d, "2026-07", "2026-07-25.1.log")
            new = os.path.join(d, "2026-07", "2026-07-26.1.log")
            with open(old, "w") as f:
                f.write("[2026-07-25 10:00:00][DEBUG] Received request: POST to /v1/embeddings\n")
            with open(new, "w") as f:
                f.write("[2026-07-26 10:00:00][DEBUG] Received request: POST to /v1/chat/completions\n")
            os.utime(old, (1000, 1000))
            os.utime(new, (2000, 2000))
            rows = logs.read_window(d, n_lines=50)
            self.assertEqual([r["path"] for r in rows],
                             ["/v1/embeddings", "/v1/chat/completions"])

    def test_missing_dir_yields_empty(self):
        self.assertEqual(logs.read_window("/nonexistent/xyz", n_lines=10), [])


def _rows(now, *specs):
    """specs: (age_seconds, kind, path_or_model)"""
    out = []
    for age, kind, val in specs:
        r = {"ts": "", "epoch": now - age, "kind": kind, "method": "POST",
             "path": None, "model": None, "messages": None}
        if kind == "request": r["path"] = val
        else: r["model"] = val
        out.append(r)
    return out


class TestAggregation(unittest.TestCase):
    def test_stats_counts_only_the_window(self):
        now = time.time()
        rows = _rows(now, (10, "request", "/a"), (20, "request", "/a"), (9999, "request", "/a"))
        s = logs.stats(rows, window_sec=300)
        self.assertEqual(s["count"], 2)
        self.assertEqual(s["window_sec"], 300)
        self.assertAlmostEqual(s["rps"], round(2 / 300, 2))

    def test_stats_empty(self):
        self.assertEqual(logs.stats([], window_sec=300),
                         {"window_sec": 300, "count": 0, "rps": 0})

    def test_top_endpoints_ranked(self):
        now = time.time()
        rows = _rows(now, (1, "request", "/a"), (2, "request", "/a"), (3, "request", "/b"))
        self.assertEqual(logs.top_endpoints(rows, window_sec=300),
                         [{"path": "/a", "count": 2}, {"path": "/b", "count": 1}])

    def test_model_activity_counts_each_kind(self):
        now = time.time()
        rows = _rows(now, (1, "completion", "m1"), (2, "completion", "m1"),
                     (3, "prediction", "m1"), (4, "tool_calls", "m1"),
                     (5, "stream_end", "m1"), (6, "completion", "m2"))
        acts = {a["model"]: a for a in logs.model_activity(rows, window_sec=300)}
        self.assertEqual(acts["m1"]["completions"], 2)
        self.assertEqual(acts["m1"]["predictions"], 1)
        self.assertEqual(acts["m1"]["tool_calls"], 1)
        self.assertEqual(acts["m1"]["streams"], 1)
        self.assertEqual(acts["m2"]["completions"], 1)

    def test_model_activity_sorted_by_completions_desc(self):
        now = time.time()
        rows = _rows(now, (1, "completion", "low"),
                     (2, "completion", "high"), (3, "completion", "high"))
        self.assertEqual([a["model"] for a in logs.model_activity(rows, window_sec=300)],
                         ["high", "low"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_logs -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'logs'`

- [ ] **Step 3: Write the implementation**

Create `logs.py`:

```python
"""LM Studio server-log discovery, parsing, and windowed aggregation.

LM Studio's logs carry far less than Ollama's GIN access logs: there is no
HTTP status, no latency, and no client IP anywhere in them. So there are no
percentile, error-rate, or per-client aggregates here — that data does not exist.
What the logs do give is request paths and per-model inference events.
"""
import glob, os, re, time
from collections import defaultdict
from datetime import datetime

from config import LOG_DIR, LOG_WINDOW_LINES, NOISE_PATHS, STATS_WINDOW_SEC

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


def read_window(log_dir=None, n_lines=None):
    """Last `n_lines` lines across the newest log files, oldest row first.

    Reads backwards through files until the budget is filled, so a window that
    straddles midnight is not truncated by the daily rollover.
    """
    n_lines = n_lines or LOG_WINDOW_LINES
    chunks, budget = [], n_lines
    for path in log_files(log_dir):
        if budget <= 0: break
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-budget:]
        except Exception:
            continue
        chunks.append(tail)
        budget -= len(tail)
    lines = [l for chunk in reversed(chunks) for l in chunk]
    return parse_lines(lines)


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_logs -v`
Expected: PASS, 19 tests

- [ ] **Step 5: Commit**

```bash
git add logs.py tests/test_logs.py
git commit -m "Add server-log parser and aggregation for LM Studio"
```

---

## Task 4: `sources.py` — settings, disk, service, and `state()`

**Files:**
- Modify: `sources.py` (remove all Ollama code; rewrite `loaded_models`, `all_models`, `parse_logs`, `stats`, `top_clients`, `top_endpoints`, `disk`, `service_info`, `server_config`, `state`)
- Create: `tests/test_sources.py`

**Interfaces:**
- Consumes: `lmstudio.loaded_models()`, `lmstudio.library()`, `lmstudio.engine_info()`, `logs.read_window()`, `logs.stats()`, `logs.top_endpoints()`, `logs.model_activity()`, config paths
- Produces:
  - `settings(path=None) -> dict` — whitelisted keys only
  - `models_root(settings_dict) -> str`
  - `disk(root=None) -> dict` — unchanged shape `{models_dir, models_size, fs_used, fs_total, fs_free}`
  - `service_info() -> dict` — `{active, pid, rss_kb, uptime_s, engine}`
  - `state() -> dict` — new keys: `loaded`, `library`, `requests`, `stats_5m`, `top_endpoints`, `model_activity`, `settings`. Removed keys: `top_clients`.
- **Unchanged and must not be touched:** `gpu`, `nvidia_versions`, `gpu_processes`, `push_history`, `get_history`, `host`, `pcie`, `start_pcie_monitor`, `tailscale`, `parse_latency_ms`/`LAT_RE` (delete these two — they were GIN-specific), `_THROTTLE_BITS`, `_decode_throttle`.

Settings whitelist — exactly these keys, nothing else:
`downloadsFolder`, `defaultContextLength`, `modelLoadingGuardrails`, `enableLocalService`, `useHFProxy`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sources.py`:

```python
import json, os, tempfile, unittest

import sources


class TestSettings(unittest.TestCase):
    def _write(self, d, obj):
        p = os.path.join(d, "settings.json")
        with open(p, "w") as f: json.dump(obj, f)
        return p

    def test_whitelisted_keys_pass_through(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, {
                "downloadsFolder": "/models",
                "defaultContextLength": {"type": "custom", "value": 65536},
                "modelLoadingGuardrails": {"mode": "high"},
                "enableLocalService": True,
                "useHFProxy": True,
            })
            s = sources.settings(p)
            self.assertEqual(s["downloadsFolder"], "/models")
            self.assertEqual(s["defaultContextLength"]["value"], 65536)
            self.assertTrue(s["enableLocalService"])

    def test_credentials_never_leak(self):
        """hfDownloadToken and friends must not appear anywhere in the output."""
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, {
                "downloadsFolder": "/models",
                "hfSearchToken": "hf_SEARCHSECRET",
                "hfDownloadToken": "hf_DOWNLOADSECRET",
                "credentials": {"nested": "hf_NESTEDSECRET"},
            })
            s = sources.settings(p)
            blob = json.dumps(s)
            self.assertNotIn("SEARCHSECRET", blob)
            self.assertNotIn("DOWNLOADSECRET", blob)
            self.assertNotIn("NESTEDSECRET", blob)
            self.assertNotIn("hfSearchToken", s)
            self.assertNotIn("hfDownloadToken", s)

    def test_no_raw_field(self):
        """The Ollama version dumped the whole file as `raw`. It must not come back."""
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, {"downloadsFolder": "/models", "secret": "x"})
            self.assertNotIn("raw", sources.settings(p))

    def test_unreadable_file_is_not_fatal(self):
        s = sources.settings("/nonexistent/settings.json")
        self.assertIn("error", s)
        self.assertEqual(s.get("downloadsFolder"), None)


class TestModelsRoot(unittest.TestCase):
    def test_prefers_downloads_folder(self):
        self.assertEqual(sources.models_root({"downloadsFolder": "/custom"}), "/custom")

    def test_falls_back_when_absent(self):
        import config
        self.assertEqual(sources.models_root({}), config.MODELS_DIR_FALLBACK)

    def test_falls_back_when_empty_string(self):
        import config
        self.assertEqual(sources.models_root({"downloadsFolder": ""}),
                         config.MODELS_DIR_FALLBACK)


class TestNoOllamaRemnants(unittest.TestCase):
    def test_gin_parsing_is_gone(self):
        self.assertFalse(hasattr(sources, "GIN_RE"))
        self.assertFalse(hasattr(sources, "parse_latency_ms"))
        self.assertFalse(hasattr(sources, "top_clients"))

    def test_hardware_panels_survive(self):
        for name in ("gpu", "gpu_processes", "nvidia_versions", "host",
                     "pcie", "tailscale", "start_pcie_monitor"):
            self.assertTrue(hasattr(sources, name), name)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_sources -v`
Expected: FAIL — `AttributeError: module 'sources' has no attribute 'settings'`

- [ ] **Step 3: Write the implementation**

In `sources.py`:

1. Replace the import block at the top:

```python
"""Read-only state sources: models, GPU, logs, disk, service, tailscale, and the aggregate state() call."""
import json, os, re, subprocess, threading, time
from collections import deque
from datetime import datetime

import logs
import lmstudio
from config import (
    GPU_HISTORY_LEN, MODELS_DIR_FALLBACK, PCIE_HISTORY_LEN, SETTINGS_PATH,
    STATS_WINDOW_SEC, SYSTEMD_UNIT, SYSTEMD_USER,
)
```

2. **Delete** these, which were Ollama-specific: `GIN_RE`, `LAT_RE`, `parse_latency_ms`, `parse_log_ts`, `loaded_models`, `all_models`, `parse_logs`, `stats`, `top_clients`, `top_endpoints`, `server_config`. Also delete the now-unused `defaultdict` and `urllib.request` imports.

3. **Keep untouched:** `START`, `_THROTTLE_BITS`, `_decode_throttle`, `gpu`, `_NVIDIA_VERSIONS`, `nvidia_versions`, `gpu_processes`, `_HIST`/`push_history`/`get_history`, `_CPU_LAST`/`_read_cpu_totals`/`host`, the whole PCIe block, `tailscale`.

4. Replace `disk()` with:

```python
def disk(root=None):
    root = root or models_root(settings())
    info = {"models_dir": None, "models_size": 0, "fs_used": 0, "fs_total": 0, "fs_free": 0}
    if not os.path.isdir(root): return info
    info["models_dir"] = root
    try:
        info["models_size"] = int(
            subprocess.check_output(["du", "-sb", root], text=True, timeout=10).split()[0])
    except Exception: pass
    try:
        st = os.statvfs(root)
        info["fs_total"] = st.f_blocks * st.f_frsize
        info["fs_free"] = st.f_bavail * st.f_frsize
        info["fs_used"] = info["fs_total"] - info["fs_free"]
    except Exception: pass
    return info
```

5. Replace `service_info()` with:

```python
def _systemctl(*args):
    cmd = ["systemctl"] + (["--user"] if SYSTEMD_USER else []) + list(args)
    return subprocess.check_output(cmd, text=True, timeout=2)


def service_info():
    info = {"uptime_s": None, "pid": None, "rss_kb": None, "active": "unknown",
            "engine": {"name": None, "version": None}}
    try:
        out = _systemctl("show", SYSTEMD_UNIT,
                         "--property=ActiveState,MainPID,ActiveEnterTimestampMonotonic")
        kv = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
        info["active"] = kv.get("ActiveState", "unknown")
        pid = int(kv.get("MainPID", "0") or 0)
        info["pid"] = pid or None
        if pid:
            try:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            info["rss_kb"] = int(line.split()[1])
                            break
                with open(f"/proc/{pid}/stat") as f:
                    starttime = int(f.read().split()[21])
                clk = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
                with open("/proc/uptime") as f:
                    uptime = float(f.read().split()[0])
                info["uptime_s"] = int(uptime - starttime / clk)
            except Exception: pass
    except Exception: pass
    info["engine"] = lmstudio.engine_info()
    return info
```

6. Add the settings reader (replacing `server_config`):

```python
# Only these keys are ever exposed. settings.json also holds hfSearchToken and
# hfDownloadToken; nothing outside this list may reach a response body.
_SETTINGS_WHITELIST = ("downloadsFolder", "defaultContextLength",
                       "modelLoadingGuardrails", "enableLocalService", "useHFProxy")


def settings(path=None):
    path = path or SETTINGS_PATH
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        return {"path": path, "error": str(e)}
    out = {"path": path}
    out.update({k: raw[k] for k in _SETTINGS_WHITELIST if k in raw})
    return out


def models_root(settings_dict):
    return (settings_dict or {}).get("downloadsFolder") or MODELS_DIR_FALLBACK
```

7. Replace `state()` with:

```python
def state():
    g = gpu()
    push_history(g)
    rows = logs.read_window()
    cfg = settings()
    return {
        "now": datetime.now().isoformat(timespec="seconds"),
        "dash_uptime_s": int(time.time() - START),
        "gpu": g,
        "gpu_processes": gpu_processes(),
        "gpu_versions": nvidia_versions(),
        "gpu_history": get_history(),
        "loaded": lmstudio.loaded_models(),
        "library": lmstudio.library(),
        "requests": rows[-30:][::-1],
        "stats_5m": logs.stats(rows),
        "top_endpoints": logs.top_endpoints(rows),
        "model_activity": logs.model_activity(rows),
        "disk": disk(models_root(cfg)),
        "service": service_info(),
        "tailscale": tailscale(),
        "pcie": pcie(),
        "host": host(),
        "settings": cfg,
        "lms_ok": os.access(config_lms_bin(), os.X_OK),
    }


def config_lms_bin():
    from config import LMS_BIN
    return LMS_BIN
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_sources -v`
Expected: PASS, 9 tests

Then confirm nothing else broke: `python3 -m unittest discover -s tests -v`
Expected: PASS, all tests

- [ ] **Step 5: Verify `state()` runs end to end against the live machine**

Run: `python3 -c "import sources, json; s = sources.state(); print(json.dumps({k: type(v).__name__ for k, v in s.items()}, indent=2))"`
Expected: every key present, no traceback. `loaded` and `library` are lists; `lms_ok` is `True`.

Also confirm no credential leak in the real payload:

Run: `python3 -c "import sources, json; assert 'hfDownloadToken' not in json.dumps(sources.state()); print('clean')"`
Expected: `clean`

- [ ] **Step 6: Commit**

```bash
git add sources.py tests/test_sources.py
git commit -m "Point sources at LM Studio; drop GIN log parsing and client stats"
```

---

## Task 5: `control.py` — benchmark scenarios

**Files:**
- Modify: `control.py` (replace `run_scenario`, keep `SCENARIOS`, `WEATHER_TOOL`, `_haystack`)
- Create: `tests/test_scenarios.py`

**Interfaces:**
- Consumes: `config.LMSTUDIO_URL`, `config.HAYSTACK_PATH`, `config.HAYSTACK_WORDS`
- Produces:
  - `map_chat_response(resp, scenario, model, wall_seconds) -> dict` — pure
  - `run_scenario(model, scenario, custom_prompt=None) -> dict` — live
  - `SCENARIOS` unchanged (same six keys)

Result shape: `{ok, scenario, model, thinking, content, tool_calls, stats}` where `stats` is
`{prompt_tokens, completion_tokens, reasoning_tokens, tokens_per_second, ttft_s, generation_s, stop_reason, wall_seconds}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scenarios.py`:

```python
import unittest

import control
from tests.helpers import fixture_json


class TestMapChatResponse(unittest.TestCase):
    def setUp(self):
        self.resp = fixture_json("chat_completion.json")

    def test_maps_the_real_response(self):
        r = control.map_chat_response(self.resp, "baseline", "google/gemma-4-31b", 1.2)
        self.assertTrue(r["ok"])
        self.assertEqual(r["scenario"], "baseline")
        self.assertEqual(r["model"], "google/gemma-4-31b")
        self.assertEqual(r["tool_calls"], [])

    def test_reasoning_content_becomes_thinking(self):
        r = control.map_chat_response(self.resp, "baseline", "m", 1.0)
        self.assertIn("Target output length", r["thinking"])
        self.assertEqual(r["content"], "")

    def test_stats_come_from_the_server_not_recomputed(self):
        s = control.map_chat_response(self.resp, "baseline", "m", 1.2)["stats"]
        self.assertEqual(s["prompt_tokens"], 26)
        self.assertEqual(s["completion_tokens"], 30)
        self.assertEqual(s["reasoning_tokens"], 27)
        self.assertAlmostEqual(s["tokens_per_second"], 66.0, places=0)
        self.assertAlmostEqual(s["ttft_s"], 0.136, places=3)
        self.assertAlmostEqual(s["generation_s"], 0.591, places=3)
        self.assertEqual(s["stop_reason"], "maxPredictedTokensReached")
        self.assertEqual(s["wall_seconds"], 1.2)

    def test_tool_calls_pass_through(self):
        resp = dict(self.resp)
        resp["choices"] = [{"message": {"content": "", "tool_calls": [
            {"function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'}}]}}]
        r = control.map_chat_response(resp, "tool_call", "m", 1.0)
        self.assertEqual(len(r["tool_calls"]), 1)
        self.assertEqual(r["tool_calls"][0]["function"]["name"], "get_weather")

    def test_missing_stats_block_does_not_crash(self):
        r = control.map_chat_response({"choices": [{"message": {"content": "hi"}}]},
                                      "baseline", "m", 0.5)
        self.assertTrue(r["ok"])
        self.assertEqual(r["content"], "hi")
        self.assertIsNone(r["stats"]["tokens_per_second"])

    def test_empty_choices_does_not_crash(self):
        r = control.map_chat_response({"choices": []}, "baseline", "m", 0.5)
        self.assertEqual(r["content"], "")


class TestScenarios(unittest.TestCase):
    def test_all_six_scenarios_present(self):
        self.assertEqual(set(control.SCENARIOS),
                         {"baseline", "reasoning", "coding", "needle29k",
                          "tool_call", "abliter"})

    def test_unknown_scenario_rejected(self):
        r = control.run_scenario("m", "does-not-exist")
        self.assertFalse(r["ok"])
        self.assertIn("unknown scenario", r["error"])

    def test_custom_requires_a_prompt(self):
        r = control.run_scenario("m", "custom", None)
        self.assertFalse(r["ok"])
        self.assertIn("custom prompt required", r["error"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_scenarios -v`
Expected: FAIL — `AttributeError: module 'control' has no attribute 'map_chat_response'`

- [ ] **Step 3: Write the implementation**

In `control.py`, update the import block:

```python
"""Mutating actions: loads, unloads, downloads, deletes, benchmarks, and the catalog scrape."""
import json, os, re, shutil, subprocess, threading, time, urllib.request

from config import (
    CATALOG_TTL_SEC, CATALOG_URL, CATALOG_USER_AGENT, HAYSTACK_PATH,
    HAYSTACK_WORDS, HUB_MODELS_DIR, LMS_BIN, LMSTUDIO_URL,
)
import lmstudio
```

Keep `_HAYSTACK_NEEDLE`, `_HAYSTACK_QUESTION`, `_HAYSTACK_CACHE`, `_haystack`, `WEATHER_TOOL`, and `SCENARIOS` exactly as they are. Replace `run_scenario` with:

```python
def map_chat_response(resp, scenario, model, wall_seconds):
    """LM Studio's /api/v0/chat/completions response -> the results shape.

    Throughput numbers come from the server's own `stats` block rather than
    being recomputed from token counts and durations.
    """
    choices = resp.get("choices") or []
    msg = (choices[0].get("message") or {}) if choices else {}
    usage = resp.get("usage") or {}
    st = resp.get("stats") or {}
    return {
        "ok": True,
        "scenario": scenario,
        "model": model,
        "thinking": (msg.get("reasoning_content") or "").strip(),
        "content": (msg.get("content") or "").strip(),
        "tool_calls": msg.get("tool_calls") or [],
        "stats": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {})
                                .get("reasoning_tokens", 0),
            "tokens_per_second": st.get("tokens_per_second"),
            "ttft_s": st.get("time_to_first_token"),
            "generation_s": st.get("generation_time"),
            "stop_reason": st.get("stop_reason"),
            "wall_seconds": wall_seconds,
        },
        "model_info": resp.get("model_info") or {},
        "runtime": resp.get("runtime") or {},
    }


def run_scenario(model, scenario, custom_prompt=None):
    if scenario == "custom":
        if not custom_prompt: return {"ok": False, "error": "custom prompt required"}
        prompt, npred, tools = custom_prompt, 400, None
    else:
        sc = SCENARIOS.get(scenario)
        if sc is None: return {"ok": False, "error": f"unknown scenario {scenario}"}
        prompt = sc["prompt"]
        if prompt == "@haystack":
            prompt = _haystack()
            if prompt is None:
                return {"ok": False, "error": f"haystack corpus missing at {HAYSTACK_PATH}"}
        npred = sc.get("npred", 200)
        tools = WEATHER_TOOL if sc.get("tools") else None

    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "stream": False, "max_tokens": npred}
    if tools is not None: payload["tools"] = tools

    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"{LMSTUDIO_URL}/api/v0/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=900).read())
    except Exception as e:
        return {"ok": False, "error": str(e), "wall_seconds": round(time.time() - t0, 1)}
    return map_chat_response(resp, scenario, model, round(time.time() - t0, 1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_scenarios -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add control.py tests/test_scenarios.py
git commit -m "Run benchmarks against LM Studio chat completions"
```

---

## Task 6: `control.py` — load, unload, and download jobs

**Files:**
- Modify: `control.py` (replace the pull machinery with a general job map; add load/unload/download)
- Create: `tests/test_jobs.py`

**Interfaces:**
- Consumes: `config.LMS_BIN`, `lmstudio.run_lms`
- Produces:
  - `build_load_args(model_key, context=None, gpu=None, ttl=None, parallel=None, identifier=None, estimate=False) -> list[str]` — pure
  - `parse_progress(line) -> dict|None` — pure; `{"pct": float}` or `{"completed": int, "total": int}` or `None`
  - `start_load(model_key, **opts) -> bool`
  - `start_download(name) -> bool`
  - `estimate_load(model_key, **opts) -> dict`
  - `unload_model(identifier) -> None`, `unload_all() -> None`
  - `get_jobs() -> dict`, `clear_finished_jobs() -> None`

Job shape: `{kind, status, pct, completed, total, error, done, started, finished, last_line}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_jobs.py`:

```python
import unittest

import control


class TestBuildLoadArgs(unittest.TestCase):
    def test_minimal(self):
        self.assertEqual(control.build_load_args("google/gemma-4-31b"),
                         ["load", "-y", "google/gemma-4-31b"])

    def test_all_options(self):
        args = control.build_load_args(
            "m", context=8192, gpu="max", ttl=300, parallel=2, identifier="fast")
        self.assertEqual(args, ["load", "-y", "m", "-c", "8192", "--gpu", "max",
                                "--ttl", "300", "--parallel", "2",
                                "--identifier", "fast"])

    def test_estimate_only_flag(self):
        self.assertIn("--estimate-only", control.build_load_args("m", estimate=True))

    def test_none_options_are_omitted(self):
        args = control.build_load_args("m", context=None, gpu=None, ttl=None)
        self.assertEqual(args, ["load", "-y", "m"])

    def test_numbers_are_stringified(self):
        args = control.build_load_args("m", context=4096)
        self.assertTrue(all(isinstance(a, str) for a in args))


class TestParseProgress(unittest.TestCase):
    def test_percentage(self):
        self.assertEqual(control.parse_progress("Downloading... 42.5%"), {"pct": 42.5})

    def test_integer_percentage(self):
        self.assertEqual(control.parse_progress("  7% done"), {"pct": 7.0})

    def test_byte_pair(self):
        r = control.parse_progress("1.50 GB / 3.00 GB")
        self.assertAlmostEqual(r["completed"] / r["total"], 0.5, places=2)

    def test_mixed_units(self):
        r = control.parse_progress("512.00 MB / 2.00 GB")
        self.assertAlmostEqual(r["completed"] / r["total"], 0.25, places=2)

    def test_unrecognized_line(self):
        self.assertIsNone(control.parse_progress("Resolving model..."))
        self.assertIsNone(control.parse_progress(""))

    def test_percentages_above_100_rejected(self):
        """Guards against matching an unrelated number followed by %."""
        self.assertIsNone(control.parse_progress("saved 250% of the time"))


class TestJobMap(unittest.TestCase):
    def setUp(self):
        control.clear_all_jobs()

    def test_starts_empty(self):
        self.assertEqual(control.get_jobs(), {})

    def test_private_keys_are_not_exposed(self):
        control._set_job("x", {"kind": "load", "done": True, "_secret": 1})
        self.assertNotIn("_secret", control.get_jobs()["x"])

    def test_clear_finished_keeps_running_jobs(self):
        control._set_job("done", {"kind": "load", "done": True})
        control._set_job("busy", {"kind": "load", "done": False})
        control.clear_finished_jobs()
        self.assertEqual(list(control.get_jobs()), ["busy"])

    def test_duplicate_start_is_refused_while_running(self):
        control._set_job("m", {"kind": "load", "done": False})
        self.assertFalse(control._claim_job("m", "load"))

    def test_restart_allowed_once_finished(self):
        control._set_job("m", {"kind": "load", "done": True})
        self.assertTrue(control._claim_job("m", "load"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_jobs -v`
Expected: FAIL — `AttributeError: module 'control' has no attribute 'build_load_args'`

- [ ] **Step 3: Write the implementation**

In `control.py`, **delete** the entire Ollama block: `_PULLS`, `_PULLS_LOCK`, `_pull_thread`, `start_pull`, `get_pulls`, `clear_finished_pulls`, `delete_model`, `unload_model`, and the whole `ollama.com/library` scrape section (`_LIB_CACHE`, `_LIB_LOCK`, `_LIB_CARD_RE`, `_LIB_FIELD_RES`, `_fetch_library_html`, `parse_library_html`, `library_remote`). `_html_unescape` is reused by the catalog in Task 7 — keep it.

Add the job machinery:

```python
# Background jobs (loads and downloads) ------------------------------------

_JOBS = {}
_JOBS_LOCK = threading.Lock()


def _set_job(key, job):
    with _JOBS_LOCK:
        _JOBS[key] = job


def _update_job(key, **kw):
    with _JOBS_LOCK:
        if key in _JOBS: _JOBS[key].update(kw)


def _claim_job(key, kind):
    """Reserve a job slot. False if one is already running under this key."""
    with _JOBS_LOCK:
        existing = _JOBS.get(key)
        if existing and not existing.get("done"): return False
        _JOBS[key] = {"kind": kind, "status": "starting", "pct": None,
                      "completed": 0, "total": 0, "error": None, "done": False,
                      "started": time.time(), "finished": None, "last_line": ""}
        return True


def get_jobs():
    with _JOBS_LOCK:
        return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                for k, v in _JOBS.items()}


def clear_finished_jobs():
    with _JOBS_LOCK:
        for k in [k for k, v in _JOBS.items() if v.get("done")]:
            del _JOBS[k]


def clear_all_jobs():
    with _JOBS_LOCK:
        _JOBS.clear()


# Progress parsing ----------------------------------------------------------

_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_BYTES_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*([KMGT]?B)\s*/\s*(\d+(?:\.\d+)?)\s*([KMGT]?B)", re.I)
_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


def parse_progress(line):
    """Pull progress out of an `lms get`/`lms load` output line.

    The CLI's progress format is not documented, so this accepts either a
    percentage or a `<done> / <total>` byte pair and returns None otherwise —
    an unrecognized line leaves the job in an indeterminate running state
    rather than failing it.
    """
    if not line: return None
    m = _BYTES_RE.search(line)
    if m:
        done = float(m.group(1)) * _UNITS[m.group(2).upper()]
        total = float(m.group(3)) * _UNITS[m.group(4).upper()]
        return {"completed": int(done), "total": int(total)}
    m = _PCT_RE.search(line)
    if m:
        pct = float(m.group(1))
        if 0 <= pct <= 100: return {"pct": pct}
    return None


def _run_job(key, args, timeout=None):
    """Stream an `lms` subprocess into the job map."""
    try:
        proc = subprocess.Popen([LMS_BIN, *args], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
    except FileNotFoundError:
        _update_job(key, error=f"lms CLI not found at {LMS_BIN}", done=True,
                    finished=time.time())
        return
    _update_job(key, status="running")
    for line in proc.stdout:
        line = line.strip()
        if not line: continue
        _update_job(key, last_line=line[:200])
        prog = parse_progress(line)
        if prog: _update_job(key, **prog)
    code = proc.wait()
    _update_job(key, done=True, finished=time.time(), status="finished",
                error=None if code == 0 else f"lms exited {code}")


# Load / unload -------------------------------------------------------------

def build_load_args(model_key, context=None, gpu=None, ttl=None,
                    parallel=None, identifier=None, estimate=False):
    args = ["load", "-y", model_key]
    for flag, val in (("-c", context), ("--gpu", gpu), ("--ttl", ttl),
                      ("--parallel", parallel), ("--identifier", identifier)):
        if val is not None and val != "":
            args += [flag, str(val)]
    if estimate: args.append("--estimate-only")
    return args


def start_load(model_key, **opts):
    if not _claim_job(model_key, "load"): return False
    args = build_load_args(model_key, **opts)
    threading.Thread(target=_run_job, args=(model_key, args), daemon=True).start()
    return True


def estimate_load(model_key, **opts):
    args = build_load_args(model_key, estimate=True, **opts)
    try:
        return {"ok": True, "output": lmstudio.run_lms(*args, timeout=30).strip()}
    except lmstudio.LmsError as e:
        return {"ok": False, "error": str(e)}


def unload_model(identifier):
    lmstudio.run_lms("unload", identifier, timeout=30)


def unload_all():
    lmstudio.run_lms("unload", "--all", timeout=30)


# Download ------------------------------------------------------------------

def start_download(name):
    if not _claim_job(name, "download"): return False
    threading.Thread(target=_run_job, args=(name, ["get", "-y", name]),
                     daemon=True).start()
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_jobs -v`
Expected: PASS, 13 tests

- [ ] **Step 5: Verify load and unload against the live machine**

The smallest model on this box is `text-embedding-nomic-embed-text-v1.5`. Use it so the check is fast.

Run: `~/.lmstudio/bin/lms unload --all; python3 -c "
import control, time
print('estimate:', control.estimate_load('text-embedding-nomic-embed-text-v1.5')['ok'])
print('started:', control.start_load('text-embedding-nomic-embed-text-v1.5'))
for _ in range(30):
    j = control.get_jobs()['text-embedding-nomic-embed-text-v1.5']
    if j['done']: break
    time.sleep(1)
print('job:', j)
import lmstudio; print('loaded:', [m['identifier'] for m in lmstudio.loaded_models()])
control.unload_all()
print('after unload:', lmstudio.loaded_models())
"`

Expected: the job reaches `done: True` with `error: None`, the model shows up in `loaded`, and the list is empty after unload.

- [ ] **Step 6: Determine the real `lms get` progress format**

`lms get`'s progress output is undocumented. Pick any small model **not already on disk** and capture what it prints:

Run: `~/.lmstudio/bin/lms get -y <some-small-model> 2>&1 | tee /tmp/lmsget.log | tail -20`

Inspect `/tmp/lmsget.log`. If progress lines match neither a `NN%` nor a `X MB / Y MB` shape, add a matching branch to `parse_progress` **and a test asserting on a real captured line** before moving on. If they do match, add a test using one verbatim captured line. Either way this step ends with a new test case in `tests/test_jobs.py`.

If no model is small enough to download comfortably, skip the download and note in the commit message that `parse_progress` is unverified against real `lms get` output — the indeterminate fallback keeps it safe either way.

- [ ] **Step 7: Commit**

```bash
git add control.py tests/test_jobs.py
git commit -m "Replace Ollama pulls with LM Studio load, unload, and download jobs"
```

---

## Task 7: `control.py` — catalog scrape

**Files:**
- Modify: `control.py` (add the catalog section)
- Create: `tests/test_catalog.py`

**Interfaces:**
- Consumes: `config.CATALOG_URL`, `config.CATALOG_TTL_SEC`, `config.CATALOG_USER_AGENT`
- Produces:
  - `parse_catalog_html(html) -> list[dict]` — pure; `{slug, name, sizes}`
  - `catalog(force=False) -> dict` — `{data, cached_age_s}` plus `error` on failure

- [ ] **Step 1: Write the failing test**

Create `tests/test_catalog.py`:

```python
import unittest

import control
from tests.helpers import fixture


class TestParseCatalogHtml(unittest.TestCase):
    def setUp(self):
        self.items = control.parse_catalog_html(fixture("catalog_page.html"))

    def test_finds_every_card(self):
        self.assertEqual(len(self.items), 8)

    def test_extracts_slug_and_name(self):
        by_slug = {i["slug"]: i for i in self.items}
        self.assertEqual(by_slug["qwen3.6"]["name"], "Qwen3.6")
        self.assertEqual(by_slug["gemma-4"]["name"], "Gemma 4")
        self.assertEqual(by_slug["lfm2-24b-a2b"]["name"], "LFM2-24B-A2B")

    def test_sizes_are_deduped_preserving_order(self):
        """Size badges render twice for responsive layouts; each must appear once."""
        by_slug = {i["slug"]: i for i in self.items}
        self.assertEqual(by_slug["qwen3.6"]["sizes"], ["27B", "35B"])
        self.assertEqual(by_slug["gemma-4"]["sizes"],
                         ["5.1B", "7.9B", "12B", "26B", "31B"])
        self.assertEqual(by_slug["qwen3.5"]["sizes"],
                         ["2B", "4B", "9B", "27B", "35B"])

    def test_slug_is_usable_as_a_download_name(self):
        for i in self.items:
            self.assertNotIn("/", i["slug"])
            self.assertNotIn('"', i["slug"])

    def test_unrecognized_markup_returns_empty_not_raises(self):
        self.assertEqual(control.parse_catalog_html("<html><body>nope</body></html>"), [])
        self.assertEqual(control.parse_catalog_html(""), [])

    def test_card_without_a_name_is_skipped_not_fatal(self):
        html = 'href="/models/ghost"><div class="other">x</div>'
        self.assertEqual(control.parse_catalog_html(html), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_catalog -v`
Expected: FAIL — `AttributeError: module 'control' has no attribute 'parse_catalog_html'`

- [ ] **Step 3: Write the implementation**

Add to `control.py`:

```python
# lmstudio.ai/models catalog scrape ----------------------------------------

_CAT_CACHE = {"data": [], "fetched": 0, "error": None}
_CAT_LOCK = threading.Lock()

# The page is a Next.js app, but the model cards are present in the server HTML.
_CAT_CARD_RE = re.compile(r'href="/models/([^"]+)"(.*?)(?=href="/models/|$)', re.S)
_CAT_NAME_RE = re.compile(r'class="text-lg font-medium">\s*([^<]+?)\s*<')
# Anchored on the title attribute, which is more stable than the utility classes.
_CAT_SIZE_RE = re.compile(r'title="Model size: ([^"]+?) parameters"')


def parse_catalog_html(html):
    out = []
    for m in _CAT_CARD_RE.finditer(html or ""):
        slug, body = m.group(1), m.group(2)
        name = _CAT_NAME_RE.search(body)
        if not name: continue  # not a model card
        # Size badges are rendered twice (desktop + mobile); dedupe, keep order.
        sizes, seen = [], set()
        for s in _CAT_SIZE_RE.findall(body):
            if s not in seen:
                seen.add(s)
                sizes.append(s)
        out.append({"slug": slug, "name": _html_unescape(name.group(1)), "sizes": sizes})
    return out


def _fetch_catalog_html():
    req = urllib.request.Request(CATALOG_URL, headers={"User-Agent": CATALOG_USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode("utf-8", errors="replace")


def catalog(force=False):
    now = time.time()
    with _CAT_LOCK:
        if not force and _CAT_CACHE["data"] and (now - _CAT_CACHE["fetched"]) < CATALOG_TTL_SEC:
            return {"data": _CAT_CACHE["data"], "cached_age_s": int(now - _CAT_CACHE["fetched"])}
    try:
        data = parse_catalog_html(_fetch_catalog_html())
        with _CAT_LOCK:
            _CAT_CACHE.update({"data": data, "fetched": now, "error": None})
        return {"data": data, "cached_age_s": 0}
    except Exception as e:
        with _CAT_LOCK:
            _CAT_CACHE["error"] = str(e)
            return {"data": _CAT_CACHE["data"], "error": str(e),
                    "cached_age_s": int(now - _CAT_CACHE["fetched"])}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_catalog -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Verify against the live page**

Run: `python3 -c "import control; c = control.catalog(); print(len(c['data']), 'entries'); print(c['data'][:3])"`
Expected: ~40 entries with real names and sizes. If it returns 0, the page markup changed — update the regexes and the fixture together.

- [ ] **Step 6: Commit**

```bash
git add control.py tests/test_catalog.py
git commit -m "Scrape the LM Studio model catalog"
```

---

## Task 8: `control.py` — delete

**Files:**
- Modify: `control.py` (add the delete section)
- Create: `tests/test_delete.py`

**Interfaces:**
- Consumes: `lmstudio.model_index()`, `lmstudio.library()`, `sources.models_root`, `sources.settings`, `config.HUB_MODELS_DIR`
- Produces:
  - `resolve_delete_targets(index, indexed_id, models_root, hub_root) -> dict` — pure; `{"ok": True, "targets": [paths]}` or `{"ok": False, "error": str}`
  - `is_inside(root, path) -> bool` — pure
  - `delete_model(model_key, confirm) -> dict` — live

This is the most destructive code in the project. Implement it last and do not shortcut the guards.

**Critical:** `resolve_delete_targets` takes an `indexed_id`, **not** a `model_key`. The public `delete_model` entry point still takes a `model_key` (that is what the UI shows) and translates it via `lmstudio.library()`. Passing a `model_key` straight through would make delete fail with "not found in the model index" for every directly-downloaded model — 3 of the 5 on this machine.

- [ ] **Step 1: Write the failing test**

Create `tests/test_delete.py`:

```python
import os, tempfile, unittest

import control
from tests.helpers import fixture_json

MODELS = "/home/dubthecoder/.lmstudio/models"
HUB = "/home/dubthecoder/.lmstudio/hub/models"


class TestIsInside(unittest.TestCase):
    def test_direct_child(self):
        self.assertTrue(control.is_inside("/a/b", "/a/b/c"))

    def test_the_root_itself_is_not_inside(self):
        self.assertFalse(control.is_inside("/a/b", "/a/b"))

    def test_sibling_with_shared_prefix_rejected(self):
        """/a/bad must not pass a /a/b root check via string prefix matching."""
        self.assertFalse(control.is_inside("/a/b", "/a/bad"))

    def test_traversal_rejected(self):
        self.assertFalse(control.is_inside("/a/b", "/a/b/../../etc"))

    def test_unrelated_absolute_path_rejected(self):
        self.assertFalse(control.is_inside("/a/b", "/etc/passwd"))

    def test_symlink_escaping_root_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "root"); outside = os.path.join(d, "outside")
            os.makedirs(root); os.makedirs(outside)
            link = os.path.join(root, "escape")
            os.symlink(outside, link)
            self.assertFalse(control.is_inside(root, link))


class TestResolveDeleteTargets(unittest.TestCase):
    def setUp(self):
        self.index = fixture_json("model_index_cache.json")

    def _resolve(self, key):
        return control.resolve_delete_targets(self.index, key, MODELS, HUB)

    def test_user_model_resolves_to_its_own_directory(self):
        # Note: the indexed id, not the model key "cyberpal2.0-20b-i1"
        r = self._resolve("mradermacher/CyberPal2.0-20B-i1-GGUF/CyberPal2.0-20B.i1-MXFP4_MOE.gguf")
        self.assertTrue(r["ok"], r.get("error"))
        self.assertEqual(r["targets"],
                         [f"{MODELS}/mradermacher/CyberPal2.0-20B-i1-GGUF"])

    def test_model_key_is_not_accepted_where_an_indexed_id_is_required(self):
        """Guards the join-key bug: modelKey matches the index for only some models."""
        r = self._resolve("cyberpal2.0-20b-i1")
        self.assertFalse(r["ok"])
        self.assertIn("not found", r["error"].lower())

    def test_hub_model_resolves_to_weights_and_stub(self):
        """google/gemma-4-31b's weights live under lmstudio-community, not google/."""
        r = self._resolve("google/gemma-4-31b")
        self.assertTrue(r["ok"], r.get("error"))
        self.assertEqual(sorted(r["targets"]), sorted([
            f"{MODELS}/lmstudio-community/gemma-4-31B-it-GGUF",
            f"{HUB}/google/gemma-4-31b",
        ]))

    def test_hub_model_never_resolves_to_its_key_as_a_path(self):
        """The naive bug: treating 'google/gemma-4-31b' as a relative path."""
        r = self._resolve("google/gemma-4-31b")
        self.assertNotIn(f"{MODELS}/google/gemma-4-31b", r["targets"])

    def test_second_hub_model(self):
        r = self._resolve("qwen/qwen3.6-27b")
        self.assertTrue(r["ok"], r.get("error"))
        self.assertIn(f"{MODELS}/lmstudio-community/Qwen3.6-27B-GGUF", r["targets"])

    def test_bundled_model_refused(self):
        r = self._resolve("nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf")
        self.assertFalse(r["ok"])
        self.assertIn("bundled", r["error"].lower())

    def test_unknown_model_refused(self):
        r = self._resolve("no/such-model")
        self.assertFalse(r["ok"])
        self.assertIn("not found", r["error"].lower())

    def test_hub_model_without_a_concrete_entry_refused_not_guessed(self):
        index = {"models": [{"indexedModelIdentifier": "orphan/model",
                             "containingDirAbsolutePath": f"{HUB}/orphan/model",
                             "sourceDirectoryType": "hub"}]}
        r = control.resolve_delete_targets(index, "orphan/model", MODELS, HUB)
        self.assertFalse(r["ok"])
        self.assertIn("could not resolve", r["error"].lower())

    def test_target_outside_permitted_roots_refused(self):
        index = {"models": [{"indexedModelIdentifier": "evil",
                             "containingDirAbsolutePath": "/etc",
                             "sourceDirectoryType": "user"}]}
        r = control.resolve_delete_targets(index, "evil", MODELS, HUB)
        self.assertFalse(r["ok"])
        self.assertIn("outside", r["error"].lower())

    def test_target_equal_to_root_refused(self):
        index = {"models": [{"indexedModelIdentifier": "root",
                             "containingDirAbsolutePath": MODELS,
                             "sourceDirectoryType": "user"}]}
        r = control.resolve_delete_targets(index, "root", MODELS, HUB)
        self.assertFalse(r["ok"])

    def test_empty_index_refused(self):
        r = control.resolve_delete_targets({"models": []}, "anything", MODELS, HUB)
        self.assertFalse(r["ok"])

    def test_malformed_index_refused_not_raised(self):
        r = control.resolve_delete_targets({}, "anything", MODELS, HUB)
        self.assertFalse(r["ok"])


class TestDeleteModelGuards(unittest.TestCase):
    def test_confirmation_mismatch_refused(self):
        r = control.delete_model("google/gemma-4-31b", confirm="wrong")
        self.assertFalse(r["ok"])
        self.assertIn("confirmation", r["error"].lower())

    def test_missing_confirmation_refused(self):
        r = control.delete_model("google/gemma-4-31b", confirm=None)
        self.assertFalse(r["ok"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_delete -v`
Expected: FAIL — `AttributeError: module 'control' has no attribute 'is_inside'`

- [ ] **Step 3: Write the implementation**

Add to `control.py`:

```python
# Delete --------------------------------------------------------------------
#
# LM Studio has no delete API and `lms` has no remove command, so this removes
# files directly. Paths come from LM Studio's own model index, never from the
# `path` field of `lms ls` — that field is a real relative path for directly
# downloaded models but a *virtual identifier* for catalog models, where the
# weights live somewhere else entirely.

def is_inside(root, path):
    """True if `path` resolves strictly inside `root`. Symlink-aware."""
    try:
        root_r = os.path.realpath(root)
        path_r = os.path.realpath(path)
    except Exception:
        return False
    return path_r.startswith(root_r + os.sep) and path_r != root_r


def _index_entries(index):
    models = (index or {}).get("models")
    return models if isinstance(models, list) else []


def resolve_delete_targets(index, indexed_id, models_root, hub_root):
    """Indexed model id -> the directories to remove.

    Takes an `indexedModelIdentifier`, NOT a model key. Those coincide for
    catalog models but not for directly downloaded ones, where the index is
    keyed by the full `<publisher>/<repo>/<file>.gguf` path.

    `user` models resolve to their own directory. `hub` models are virtual
    pointers: their weights live in a separate `user` entry, found via the
    `<id>@<concrete-path>` index entry. Both the weights and the stub go.
    `bundled` models ship with LM Studio and are never deletable.
    """
    entries = _index_entries(index)
    by_id = {e.get("indexedModelIdentifier"): e for e in entries if isinstance(e, dict)}

    entry = by_id.get(indexed_id)
    if entry is None:
        return {"ok": False, "error": f"model {indexed_id} not found in the model index"}

    kind = entry.get("sourceDirectoryType")
    if kind == "bundled":
        return {"ok": False, "error": f"{indexed_id} is a bundled model and cannot be deleted"}

    targets = []
    if kind == "hub":
        # Find the "<id>@<concrete>" entry that names the real weights.
        prefix = indexed_id + "@"
        concrete_key = next((k[len(prefix):] for k in by_id if k.startswith(prefix)), None)
        concrete = by_id.get(concrete_key) if concrete_key else None
        if not concrete or not concrete.get("containingDirAbsolutePath"):
            return {"ok": False,
                    "error": f"could not resolve virtual model {indexed_id} to concrete weights"}
        targets.append(concrete["containingDirAbsolutePath"])
        if entry.get("containingDirAbsolutePath"):
            targets.append(entry["containingDirAbsolutePath"])
    elif kind == "user":
        if not entry.get("containingDirAbsolutePath"):
            return {"ok": False, "error": f"no directory recorded for {indexed_id}"}
        targets.append(entry["containingDirAbsolutePath"])
    else:
        return {"ok": False, "error": f"unknown storage type {kind!r} for {indexed_id}"}

    for t in targets:
        if not (is_inside(models_root, t) or is_inside(hub_root, t)):
            return {"ok": False,
                    "error": f"refusing to delete {t}: outside the permitted model roots"}

    return {"ok": True, "targets": sorted(set(targets))}


def delete_model(model_key, confirm):
    """Delete a model's files. `confirm` must equal `model_key` exactly."""
    if not confirm or confirm != model_key:
        return {"ok": False, "error": "confirmation must match the model key exactly"}

    import sources  # local import: sources imports control-free modules only
    loaded_now = lmstudio.loaded_models()
    loaded = {m["identifier"] for m in loaded_now} | {m["model_key"] for m in loaded_now}
    if model_key in loaded:
        return {"ok": False, "error": f"{model_key} is loaded — unload it first"}

    # The index is keyed by indexedModelIdentifier, which is not the model key
    # for directly downloaded models. Translate before resolving.
    indexed_id = next((m["indexed_id"] for m in lmstudio.library()
                       if m["model_key"] == model_key), None)
    if not indexed_id:
        return {"ok": False, "error": f"unknown model {model_key}"}

    cfg = sources.settings()
    resolved = resolve_delete_targets(
        lmstudio.model_index(), indexed_id, sources.models_root(cfg), HUB_MODELS_DIR)
    if not resolved["ok"]: return resolved

    removed = []
    for t in resolved["targets"]:
        try:
            shutil.rmtree(t)
            removed.append(t)
        except Exception as e:
            return {"ok": False, "error": f"failed removing {t}: {e}", "removed": removed}
    return {"ok": True, "removed": removed}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_delete -v`
Expected: PASS, 22 tests

Then the full suite: `python3 -m unittest discover -s tests -v`
Expected: PASS, all tests

- [ ] **Step 5: Verify resolution against the live index without deleting anything**

Run: `python3 -c "
import control, lmstudio, sources, os
cfg = sources.settings(); idx = lmstudio.model_index()
for m in lmstudio.library():
    r = control.resolve_delete_targets(idx, m['indexed_id'],
                                       sources.models_root(cfg), control.HUB_MODELS_DIR)
    if r['ok']:
        marks = [('OK ' if os.path.isdir(t) else 'MISSING ') + t for t in r['targets']]
        print(f\"{m['model_key']:40} {marks}\")
    else:
        print(f\"{m['model_key']:40} REFUSED: {r['error']}\")
"`

Expected: every listed model resolves, every target is marked `OK` (a real existing directory under `~/.lmstudio/models` or `~/.lmstudio/hub/models`), and the bundled nomic model is refused. Any `MISSING` or unexpected `REFUSED` means the resolution is wrong — stop and fix before going further. **Do not run an actual delete on a model you want to keep.**

- [ ] **Step 6: Commit**

```bash
git add control.py tests/test_delete.py
git commit -m "Add guarded model deletion via the LM Studio model index"
```

---

## Task 9: `server.py` — routing

**Files:**
- Modify: `server.py`
- Create: `tests/test_server.py`

**Interfaces:**
- Consumes: everything above
- Produces: the HTTP surface

| Method | Path | Body | Handler |
|---|---|---|---|
| GET | `/` , `/index.html` | — | dashboard |
| GET | `/control` | — | control panel |
| GET | `/api/state` | — | `sources.state()` |
| GET | `/api/control/jobs` | — | `control.get_jobs()` |
| GET | `/api/control/catalog[?refresh=1]` | — | `control.catalog(force)` |
| POST | `/api/control/load` | `{model, context?, gpu?, ttl?, parallel?, identifier?}` | `control.start_load` |
| POST | `/api/control/load/estimate` | same | `control.estimate_load` |
| POST | `/api/control/unload` | `{identifier}` or `{all: true}` | `control.unload_model` / `unload_all` |
| POST | `/api/control/download` | `{name}` | `control.start_download` |
| POST | `/api/control/test` | `{model, scenario, custom_prompt?}` | `control.run_scenario` |
| POST | `/api/control/jobs/clear` | — | `control.clear_finished_jobs` |
| DELETE | `/api/control/model` | `{name, confirm}` | `control.delete_model` |

- [ ] **Step 1: Write the failing test**

Create `tests/test_server.py`:

```python
import json, threading, unittest, urllib.error, urllib.request

import server


class TestRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = server.ThreadedServer(("127.0.0.1", 0), server.Handler)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def _req(self, path, method="GET", body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_dashboard_served(self):
        code, body = self._req("/")
        self.assertEqual(code, 200)
        self.assertIn(b"<html", body.lower())

    def test_control_panel_served(self):
        self.assertEqual(self._req("/control")[0], 200)

    def test_unknown_path_404s(self):
        self.assertEqual(self._req("/nope")[0], 404)

    def test_state_returns_expected_keys(self):
        code, body = self._req("/api/state")
        self.assertEqual(code, 200)
        s = json.loads(body)
        for k in ("gpu", "loaded", "library", "requests", "stats_5m",
                  "top_endpoints", "model_activity", "disk", "service",
                  "host", "settings"):
            self.assertIn(k, s)

    def test_state_has_no_removed_ollama_keys(self):
        s = json.loads(self._req("/api/state")[1])
        self.assertNotIn("top_clients", s)
        self.assertNotIn("server_config", s)

    def test_state_never_contains_credentials(self):
        self.assertNotIn(b"hfDownloadToken", self._req("/api/state")[1])

    def test_jobs_endpoint(self):
        code, body = self._req("/api/control/jobs")
        self.assertEqual(code, 200)
        self.assertIsInstance(json.loads(body), dict)

    def test_load_requires_a_model(self):
        code, body = self._req("/api/control/load", "POST", {})
        self.assertEqual(code, 400)
        self.assertIn("model", json.loads(body)["error"])

    def test_download_requires_a_name(self):
        code, body = self._req("/api/control/download", "POST", {"name": "  "})
        self.assertEqual(code, 400)

    def test_delete_requires_matching_confirmation(self):
        code, body = self._req("/api/control/model", "DELETE",
                               {"name": "x", "confirm": "y"})
        self.assertEqual(code, 400)
        self.assertIn("confirmation", json.loads(body)["error"].lower())

    def test_delete_requires_a_name(self):
        self.assertEqual(self._req("/api/control/model", "DELETE", {})[0], 400)

    def test_unload_requires_identifier_or_all(self):
        self.assertEqual(self._req("/api/control/unload", "POST", {})[0], 400)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_server -v`
Expected: FAIL — the new routes 404 and `/api/state` still carries `top_clients`.

- [ ] **Step 3: Write the implementation**

In `server.py`, replace the docstring, `do_GET`, `do_POST`, `do_DELETE`, and the `__main__` banner:

```python
#!/usr/bin/env python3
"""Live dashboard for LM Studio. Stdlib only."""
```

```python
    def _load_opts(self, body):
        return {k: body.get(k) for k in
                ("context", "gpu", "ttl", "parallel", "identifier")}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, "text/html; charset=utf-8", INDEX_HTML)
        if self.path == "/control":
            return self._send(200, "text/html; charset=utf-8", CONTROL_HTML)
        if self.path == "/api/state":
            return self._json(200, sources.state())
        if self.path == "/api/control/jobs":
            return self._json(200, control.get_jobs())
        if self.path.startswith("/api/control/catalog"):
            return self._json(200, control.catalog(force="refresh=1" in self.path))
        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        body = self._read_json()
        try:
            if self.path == "/api/control/load":
                model = (body.get("model") or "").strip()
                if not model: return self._json(400, {"error": "model required"})
                return self._json(200, {"started": control.start_load(
                    model, **self._load_opts(body))})
            if self.path == "/api/control/load/estimate":
                model = (body.get("model") or "").strip()
                if not model: return self._json(400, {"error": "model required"})
                return self._json(200, control.estimate_load(
                    model, **self._load_opts(body)))
            if self.path == "/api/control/unload":
                if body.get("all"):
                    control.unload_all()
                    return self._json(200, {"ok": True})
                ident = (body.get("identifier") or "").strip()
                if not ident:
                    return self._json(400, {"error": "identifier or all required"})
                control.unload_model(ident)
                return self._json(200, {"ok": True})
            if self.path == "/api/control/download":
                name = (body.get("name") or "").strip()
                if not name: return self._json(400, {"error": "name required"})
                return self._json(200, {"started": control.start_download(name)})
            if self.path == "/api/control/test":
                return self._json(200, control.run_scenario(
                    body.get("model"), body.get("scenario"), body.get("custom_prompt")))
            if self.path == "/api/control/jobs/clear":
                control.clear_finished_jobs()
                return self._json(200, {"ok": True})
        except Exception as e:
            return self._json(500, {"error": str(e)})
        self._send(404, "text/plain", b"not found")

    def do_DELETE(self):
        body = self._read_json()
        try:
            if self.path == "/api/control/model":
                name = (body.get("name") or "").strip()
                if not name: return self._json(400, {"error": "name required"})
                result = control.delete_model(name, body.get("confirm"))
                return self._json(200 if result["ok"] else 400, result)
        except Exception as e:
            return self._json(500, {"error": str(e)})
        self._send(404, "text/plain", b"not found")
```

And the banner:

```python
        print(f"lmstudio dashboard on http://{config.HOST}:{config.PORT}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_server -v`
Expected: PASS, 12 tests

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "Route the HTTP API to LM Studio controls"
```

---

## Task 10: Dashboard template

**Files:**
- Modify: `templates/index.html`

Read the file first — it is ~391 lines of self-contained HTML/CSS/JS polling `/api/state`. Change only what the data change requires.

- [ ] **Step 1: Update the page identity**

Change the `<title>` and header text from Ollama to LM Studio. Search the file for `ollama` case-insensitively and fix every hit:

Run: `grep -in ollama templates/index.html`

- [ ] **Step 2: Rewire the loaded-models card**

The card reads `state.loaded`. Field names changed from Ollama's `/api/ps` shape to the normalized shape from Task 2: `name`→`display_name`, `size_vram`→`size`, `expires_at`→`ttl_s`. Add the new columns now available: `context` / `max_context`, `status`, `queued`, `parallel`, and `vision`/`tools` badges.

- [ ] **Step 3: Rewire the library card**

`state.library` rows changed from `{name, size, modified_at}` to the normalized disk shape. Drop the `modified_at` column (no equivalent exists). Add `arch`, `quant`, `params`, and a loaded indicator driven by `row.loaded`.

- [ ] **Step 4: Remove the deleted panels**

Delete the "top clients" card entirely. In the stats card, remove the average/p50/p95 latency and error-rate figures — `state.stats_5m` now carries only `{window_sec, count, rps}`. In the request list, remove the status and latency columns; rows now carry `{ts, method, path}`.

- [ ] **Step 5: Add the model-activity card**

Add a card rendering `state.model_activity` — a row per model with `completions`, `predictions`, `tool_calls`, `streams`. Place it where the top-clients card used to sit so the grid layout is unchanged.

- [ ] **Step 6: Update the service card**

`state.service.version` is gone; render `state.service.engine.name` and `state.service.engine.version` instead.

- [ ] **Step 7: Replace the server-config card**

It read `state.server_config.env` and `state.server_config.raw`. It now reads `state.settings`, rendering the whitelisted keys: models folder, default context length, load guardrail mode, local service enabled, HF proxy. **Do not add a raw dump.**

- [ ] **Step 8: Add an lms-missing banner**

When `state.lms_ok` is false, show a banner reading `LM Studio CLI not found at <path>` — a missing CLI empties four panels at once and would otherwise look like an unexplained blank dashboard.

- [ ] **Step 9: Verify in a browser**

Run: `python3 server.py` and open `http://127.0.0.1:11435/`.

Expected: every card renders with real data, no JS console errors, no "undefined" text. Load a model in another terminal (`~/.lmstudio/bin/lms load text-embedding-nomic-embed-text-v1.5 -y`) and confirm the loaded card and library indicator both update within one poll interval.

- [ ] **Step 10: Commit**

```bash
git add templates/index.html
git commit -m "Rework dashboard cards for LM Studio state"
```

---

## Task 11: Control panel template

**Files:**
- Modify: `templates/control.html`

- [ ] **Step 1: Update the page identity**

Run: `grep -in ollama templates/control.html` and fix every hit, including the "ollama.com/library" label.

- [ ] **Step 2: Replace the pull panel with a download panel**

It posted `{name}` to `/api/control/pull` and polled `/api/control/pulls`. Repoint to `/api/control/download` and `/api/control/jobs`. The job shape gained `kind` and `last_line` and lost `rate_bps`; show `last_line` when neither `pct` nor `total` is available, since `lms get` progress output may be indeterminate. Repoint the clear button to `/api/control/jobs/clear`.

- [ ] **Step 3: Add the load panel**

New. A model picker populated from `state.library`, plus optional inputs for context length, GPU offload (`off` / `max` / 0–1), TTL seconds, parallel, and identifier. An "Estimate" button posts to `/api/control/load/estimate` and shows the returned `output` text. A "Load" button posts to `/api/control/load`; progress appears in the same job list as downloads.

- [ ] **Step 4: Update the unload panel**

It posted `{name}` to `/api/control/unload`. It now posts `{identifier}`, sourced from `state.loaded[].identifier` — not the model key, since `lms load --identifier` lets those differ. Add an "Unload all" button posting `{all: true}`.

- [ ] **Step 5: Replace the library panel with the catalog panel**

It fetched `/api/control/library_remote`. Repoint to `/api/control/catalog` (and `?refresh=1` for the refresh button). Rows are now `{slug, name, sizes}` — drop the description, pulls, tags, updated, and capability columns, which the LM Studio catalog does not expose. Clicking a row fills the download box with its `slug`.

- [ ] **Step 6: Update the delete flow**

The old flow sent `{name}` to `DELETE /api/control/model`. It must now also send `confirm`, gathered from a text input where the user types the model key verbatim. Disable the delete button until the typed value matches exactly. Show the returned `removed` paths on success and the `error` on refusal.

- [ ] **Step 7: Update the benchmark results panel**

`stats` changed shape: `prompt_rate`/`eval_rate`/`eval_tokens` are gone, replaced by `tokens_per_second`, `ttft_s`, `generation_s`, `stop_reason`, `completion_tokens`, `reasoning_tokens`. Render time-to-first-token alongside tokens/sec — it is new data the Ollama version could not show.

- [ ] **Step 8: Verify in a browser**

Run: `python3 server.py` and open `http://127.0.0.1:11435/control`.

Expected, exercised end to end: estimate a load, run a load, see it in the job list, see it appear in the loaded list, run a benchmark against it showing TTFT, unload it, load the catalog and click a row to fill the download box. Confirm the delete button stays disabled until the confirmation text matches. **Do not complete a delete on a model you want to keep** — verify the guard by typing a mismatched value and confirming the button stays disabled.

- [ ] **Step 9: Commit**

```bash
git add templates/control.html
git commit -m "Rework control panel for LM Studio load, download, and catalog"
```

---

## Task 12: README and final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite the README**

Update every section: title and intro (LM Studio, not Ollama), the run instructions (`http://localhost:1234` upstream), the systemd unit example (rename the unit to `lmstudio-dashboard`, keep it a user unit), the full configuration table from Task 1's `config.py`, the requirements list (Python 3.9+, LM Studio with `lms` CLI, Linux + systemd for service info, `nvidia-smi` for GPU panels), and the layout block — which must now list `lmstudio.py`, `logs.py`, and `tests/`.

Add a short "What the request panel does and doesn't show" note explaining that LM Studio's logs carry no HTTP status, latency, or client IP, so there are no percentile, error-rate, or per-client figures. Someone comparing against the old dashboard will otherwise assume it broke.

Also document that model deletion removes files from disk directly, since LM Studio has no delete API.

- [ ] **Step 2: Confirm no Ollama references survive**

Run: `grep -rin ollama --include='*.py' --include='*.html' --include='*.md' . | grep -v docs/superpowers | grep -v '^./.git'`
Expected: no output. If the spec or plan documents match, that is fine — they are historical records under `docs/superpowers`.

- [ ] **Step 3: Run the full test suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS, all tests, no errors.

- [ ] **Step 4: Confirm the suite passes with LM Studio stopped**

Run: `systemctl --user stop lmstudio-server && python3 -m unittest discover -s tests 2>&1 | tail -5; systemctl --user start lmstudio-server`
Expected: still PASS — the suite is fixture-driven and must not need a live server.

- [ ] **Step 5: Confirm the dashboard degrades gracefully with LM Studio stopped**

Run: `systemctl --user stop lmstudio-server`, load `http://127.0.0.1:11435/`, then `systemctl --user start lmstudio-server`.

Expected: the page still renders. GPU, host, PCIe, and disk cards are populated. Model cards are empty rather than throwing, and the service card shows the unit as inactive. No unhandled traceback in the server's output.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "Rewrite README for LM Studio"
```

- [ ] **Step 7: Report what changed for the operator**

Summarize for the user, explicitly listing:
- The env vars they must rename in their own systemd unit (`OLLAMA_*` → `LMSTUDIO_*`), since nothing shims the old names and the dashboard will silently use defaults otherwise.
- The panels that no longer exist and why.
- That the working directory is still named `ollama-dashboard` and renaming it is their call.

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Loaded models via `lms ps` | 2, 10 |
| Model library via `lms ls` + `/api/v0/models` | 2, 10 |
| Service info, user unit, engine version | 4, 10 |
| Request telemetry (degraded), panels removed | 3, 4, 10 |
| Server config whitelist, no `raw`, no tokens | 4, 10 |
| Disk from `downloadsFolder` | 4 |
| GPU/host/PCIe/tailscale untouched | 4 (explicit "do not touch" list) |
| Load with options + `--estimate-only` | 6, 11 |
| Unload + unload-all | 6, 11 |
| Download via `lms get` | 6, 11 |
| Catalog scrape | 7, 11 |
| Delete: 3 storage classes, hub indirection, guards | 8, 11 |
| Benchmarks on `/api/v0/chat/completions`, TTFT | 5, 11 |
| Config rename, no shim | 1, 12 |
| Error handling, `lms` missing banner | 2, 4, 10 |
| Testing (all 7 spec rows) | 1–9 |

All spec sections map to tasks. The spec's "implement delete last" instruction is honored — delete is Task 8, after every other backend concern.

**Placeholder scan:** No TBDs. The one genuine unknown, `lms get`'s progress format, is handled by a tolerant parser plus an explicit empirical step (Task 6 Step 6) that ends in a real test, with a specified fallback if the download proves impractical.

**Type consistency:** Verified across tasks — `model_key` (not `name`) throughout the normalized shapes; `identifier` used for unload everywhere, distinct from `model_key`; `get_jobs`/`clear_finished_jobs` named consistently in Tasks 6, 9, and 11; `settings()`/`models_root()` signatures match between Tasks 4 and 8; `map_chat_response`'s `stats` keys match what Task 11 renders; `resolve_delete_targets`'s four-argument signature matches its callers in Task 8's live check.

**One correction made during review.** The first draft had `resolve_delete_targets` keyed on `model_key`. Checking it against the real fixtures showed `modelKey` matches `model-index-cache.json` for only 2 of the 5 models on this machine — delete would have failed with "not found in the model index" for every directly downloaded model, which are precisely the ones worth deleting. The join key is `indexedModelIdentifier`. Fixed in Tasks 2 and 8, with `normalize_disk`/`normalize_loaded` now carrying `indexed_id`, a three-identifier table in Task 2, and two regression tests (`test_indexed_id_is_carried_and_differs_from_model_key`, `test_model_key_is_not_accepted_where_an_indexed_id_is_required`).
