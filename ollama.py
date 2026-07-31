"""All Ollama coupling: the HTTP API and payload normalization.

Nothing else in the codebase should know Ollama's field names.

Unlike the LM Studio module this replaces, there is no CLI here and no
identifier problem. LM Studio distinguished a load key, a loaded-instance
name, and a model-index key, none of them interchangeable. Ollama has one
name — `gemma4:31b` — that serves all three roles.
"""
import json, re, time, urllib.error, urllib.request
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
