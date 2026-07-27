"""All LM Studio coupling: the `lms` CLI, the /api/v0 HTTP API, and payload normalization.

Nothing else in the codebase should know LM Studio's field names.

Three identifiers travel with every model and they are not interchangeable:

    model_key   what `lms load` takes, and what the UI shows
    identifier  the loaded *instance* name, what `lms unload` takes; differs
                from model_key when loaded via `lms load --identifier`
    indexed_id  the only key that joins to model-index-cache.json, and so the
                only one delete may use. Equals model_key for catalog models
                but is the full `<publisher>/<repo>/<file>.gguf` path for
                directly downloaded ones.
"""
import json, re, subprocess, urllib.request

from config import LMS_BIN, LMSTUDIO_URL, MODEL_INDEX_PATH


class LmsError(Exception):
    """The `lms` CLI is missing, failed, or returned something unparseable."""


def run_lms(*args, timeout=5, merge_stderr=False):
    """Run `lms` and return stdout.

    `lms` writes some human-facing output to stderr rather than stdout —
    `load --estimate-only` is entirely on stderr — so callers that need that
    text pass merge_stderr=True. JSON-producing subcommands write to stdout
    and must NOT merge, or stray diagnostics would corrupt the payload.
    """
    try:
        return subprocess.check_output(
            [LMS_BIN, *args], text=True, timeout=timeout,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.DEVNULL)
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
