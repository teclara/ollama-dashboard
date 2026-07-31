"""Mutating actions: loads, unloads, pulls, deletes, and the catalog scrape."""
import json, re, threading, time, urllib.request

from config import (
    CATALOG_TTL_SEC, CATALOG_URL, CATALOG_USER_AGENT, OLLAMA_URL,
)
import ollama
import sources


# Shared helpers ----------------------------------------------------------

def _html_unescape(s):
    return (s.replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"')
             .replace("&lt;", "<").replace("&gt;", ">"))


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
                      "completed": 0, "total": 0, "rate_bps": None,
                      "eta_s": None, "error": None, "done": False,
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
    # nvidia-smi reports MiB.
    free = max(0, (g["mem_total"] - g["mem_used"])) * 1024 * 1024
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


# ollama.com/library catalog scrape ----------------------------------------

_CAT_CACHE = {"data": [], "fetched": 0, "error": None}
_CAT_LOCK = threading.Lock()

_CAT_CARD_RE = re.compile(
    r'href="/library/([^"]+)"(.*?)(?=href="/library/|\Z)', re.S)
_CAT_NAME_RE = re.compile(
    r'<span class="group-hover:underline truncate">\s*([^<]+?)\s*</span>')
_CAT_DESC_RE = re.compile(r'<p class="[^"]*break-words[^"]*">\s*(.*?)\s*</p>', re.S)
_CAT_BADGE_RE = re.compile(r'<span[^>]*text-xs font-medium[^>]*>\s*([^<]+?)\s*</span>')
# Size badges look like 8b / 405b / 137m; anything else in a badge slot is a
# capability (tools, vision, thinking, embedding).
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
