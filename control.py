"""Mutating actions: loads, unloads, pulls, benchmarks, deletes, and catalog."""
import json, re, statistics, threading, time, urllib.request

from config import (
    CATALOG_TTL_SEC, CATALOG_URL, CATALOG_USER_AGENT, OLLAMA_URL,
)
import ollama


# Shared helpers ----------------------------------------------------------

def _html_unescape(s):
    return (s.replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"')
             .replace("&lt;", "<").replace("&gt;", ">"))


# Background jobs (loads and downloads) ------------------------------------

_JOBS = {}
_JOBS_LOCK = threading.Lock()
_ACTIVE_MUTATIONS = [0]
_MAX_ACTIVE_JOBS = 8
_MAX_JOB_HISTORY = 100
_BENCHMARK_KEY = "benchmark:suite"
_BENCHMARK_PROMPT = (
    "Explain why both latency and throughput matter when evaluating a local "
    "language model. Be concrete and concise."
)


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
        active = [job for job in _JOBS.values() if not job.get("done")]
        # Benchmarks must not overlap dashboard-initiated model operations;
        # that would produce plausible but misleading performance numbers.
        if kind == "benchmark" and (active or _ACTIVE_MUTATIONS[0]):
            return False
        if kind != "benchmark" and any(job.get("kind") == "benchmark" for job in active):
            return False
        if sum(1 for job in _JOBS.values() if not job.get("done")) >= _MAX_ACTIVE_JOBS:
            return False
        finished = sorted(
            ((k, v) for k, v in _JOBS.items() if v.get("done") and k != key),
            key=lambda item: item[1].get("finished") or item[1].get("started") or 0,
        )
        while len(_JOBS) >= _MAX_JOB_HISTORY and finished:
            old_key, _ = finished.pop(0)
            del _JOBS[old_key]
        _JOBS[key] = {"kind": kind, "status": "starting", "pct": None,
                      "completed": 0, "total": 0, "rate_bps": None,
                      "eta_s": None, "error": None, "done": False,
                      "started": time.time(), "finished": None, "last_line": ""}
        return True


def _begin_immediate_mutation():
    """Reserve an unload/delete window that cannot overlap a benchmark."""
    with _JOBS_LOCK:
        if any(not job.get("done") and job.get("kind") == "benchmark"
               for job in _JOBS.values()):
            return False
        _ACTIVE_MUTATIONS[0] += 1
        return True


def _end_immediate_mutation():
    with _JOBS_LOCK:
        _ACTIVE_MUTATIONS[0] -= 1


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


# Benchmark -----------------------------------------------------------------

def _bounded_int(value, name, minimum, maximum):
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{name} must be an integer")
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def build_benchmark_config(models, prompt=None, warmups=1, runs=3,
                           num_predict=128, context=None):
    if not isinstance(models, list):
        raise ValueError("models must be an array")
    clean_models = []
    for model in models:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("each model must be a non-empty string")
        model = model.strip()
        if model not in clean_models:
            clean_models.append(model)
    if not clean_models:
        raise ValueError("at least one model is required")
    if len(clean_models) > 20:
        raise ValueError("at most 20 models may be benchmarked at once")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    prompt = _BENCHMARK_PROMPT if prompt is None else prompt.strip()
    if not prompt:
        raise ValueError("prompt is required")
    if len(prompt) > 16000:
        raise ValueError("prompt must be at most 16000 characters")
    return {
        "models": clean_models,
        "prompt": prompt,
        "warmups": _bounded_int(warmups, "warmups", 0, 5),
        "runs": _bounded_int(runs, "runs", 1, 10),
        "num_predict": _bounded_int(num_predict, "num_predict", 1, 2048),
        "context": None if context in (None, "") else
                   _bounded_int(context, "context", 1, 1048576),
    }


def summarize_benchmark_runs(model, runs):
    summary = {"model": model, "status": "finished", "runs": runs}
    for name in ("prompt_tps", "generation_tps", "ttft_s", "load_s",
                 "total_s", "wall_s", "prompt_tokens", "output_tokens"):
        values = [run[name] for run in runs if run.get(name) is not None]
        summary[name] = statistics.median(values) if values else None
    return summary


def _run_benchmark(config):
    results = []
    total_steps = len(config["models"]) * (config["warmups"] + config["runs"])
    completed_steps = 0
    _update_job(
        _BENCHMARK_KEY, status="running", results=[], current_model=None,
        model_index=0, model_total=len(config["models"]), run=0,
        run_total=config["runs"], pct=0.0,
        benchmark={k: v for k, v in config.items() if k != "prompt"},
    )
    for model_index, model in enumerate(config["models"], 1):
        model_started_steps = completed_steps
        measured = []
        failed = None
        try:
            for warmup in range(1, config["warmups"] + 1):
                _update_job(_BENCHMARK_KEY, status="warming", current_model=model,
                            model_index=model_index, run=warmup,
                            run_total=config["warmups"],
                            last_line=f"warm-up {warmup}/{config['warmups']}")
                ollama.benchmark_generate(
                    model, config["prompt"], config["num_predict"], config["context"])
                completed_steps += 1
                _update_job(_BENCHMARK_KEY,
                            pct=round(completed_steps / total_steps * 100, 1))
            for run in range(1, config["runs"] + 1):
                _update_job(_BENCHMARK_KEY, status="running", current_model=model,
                            model_index=model_index, run=run,
                            run_total=config["runs"],
                            last_line=f"measured run {run}/{config['runs']}")
                sample = ollama.benchmark_generate(
                    model, config["prompt"], config["num_predict"], config["context"])
                sample["run"] = run
                measured.append(sample)
                completed_steps += 1
                partial = results + [summarize_benchmark_runs(model, measured)]
                _update_job(_BENCHMARK_KEY, results=partial,
                            pct=round(completed_steps / total_steps * 100, 1))
        except Exception as e:
            failed = str(e)
        if failed:
            # Count skipped runs so progress still reaches a terminal state.
            completed_steps = model_started_steps + total_steps // len(config["models"])
            result = {"model": model, "status": "failed", "error": failed,
                      "runs": measured}
        else:
            result = summarize_benchmark_runs(model, measured)
        results.append(result)
        _update_job(_BENCHMARK_KEY, results=list(results),
                    pct=round(completed_steps / total_steps * 100, 1))
    failures = sum(result["status"] == "failed" for result in results)
    all_failed = failures == len(results)
    _update_job(
        _BENCHMARK_KEY, done=True, finished=time.time(), pct=100.0,
        status="failed" if all_failed else "finished",
        error="all benchmark models failed" if all_failed else None,
        last_line=(f"{len(results) - failures} models measured, {failures} failed"
                   if failures else f"{len(results)} models measured"),
        results=results,
    )


def start_benchmark(**options):
    config = build_benchmark_config(**options)
    import samplers
    library = {model["model_key"]: model for model in samplers.LIBRARY.get() or []}
    if not library:
        raise ValueError("model inventory has not been sampled yet")
    missing = [model for model in config["models"] if model not in library]
    if missing:
        raise ValueError("model is not in the local inventory: " + ", ".join(missing))
    embedding = [model for model in config["models"] if library[model].get("embedding")]
    if embedding:
        raise ValueError("generation benchmark does not support embedding models: " +
                         ", ".join(embedding))
    if not _claim_job(_BENCHMARK_KEY, "benchmark"):
        return False
    try:
        threading.Thread(target=_run_benchmark, args=(config,), daemon=True).start()
    except Exception as e:
        _update_job(_BENCHMARK_KEY, done=True, finished=time.time(), status="failed",
                    error=str(e))
        raise
    return True


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

def _int_option(v, name, minimum):
    if v is None or v == "":
        return None
    try:
        value = int(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def build_load_payload(model, context=None, gpu=None, ttl=None):
    payload = {"model": model, "prompt": ""}
    options = {}
    num_ctx = _int_option(context, "num_ctx", 1)
    num_gpu = _int_option(gpu, "num_gpu", 0)
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
    if reason == "load":
        # Refresh from this background worker so the state endpoint remains a
        # cache-only read while reflecting the completed load promptly.
        import samplers
        samplers.LOADED.refresh()
        samplers.record_timeline(samplers.LOADED.get() or [])
        samplers.LIBRARY.refresh()


def start_load(model, **opts):
    payload = build_load_payload(model, **opts)
    if not _claim_job(model, "load"):
        return False
    try:
        threading.Thread(target=_run_load, args=(model, payload), daemon=True).start()
    except Exception as e:
        _update_job(model, done=True, finished=time.time(), status="failed", error=str(e))
        raise
    return True


def _unload_model(name):
    ollama.api_post("/api/generate", {"model": name, "keep_alive": 0}, timeout=60)
    import samplers
    loaded, loaded_sampled = samplers.LOADED.peek()
    if loaded_sampled:
        remaining = samplers.LOADED.update(
            lambda current: [m for m in current if m.get("model_key") != name])
        samplers.record_timeline(remaining)
    library, library_sampled = samplers.LIBRARY.peek()
    if library_sampled:
        samplers.LIBRARY.update(lambda current: [
            {**m, "loaded": False} if m.get("model_key") == name else m
            for m in current
        ])


def unload_model(name):
    if not _begin_immediate_mutation():
        raise ValueError("cannot unload a model while a benchmark is running")
    try:
        _unload_model(name)
    finally:
        _end_immediate_mutation()


def unload_all():
    if not _begin_immediate_mutation():
        raise ValueError("cannot unload models while a benchmark is running")
    try:
        import samplers
        for m in samplers.LOADED.get() or []:
            if m.get("model_key"):
                _unload_model(m["model_key"])
    finally:
        _end_immediate_mutation()


def estimate_fit(model):
    """Model size against free VRAM.

    Replaces `lms load --estimate-only`, which Ollama has no equivalent for.
    This is a size comparison, not a real estimate — it ignores KV cache and
    context, so it is presented as a fit indicator rather than a prediction.
    """
    import samplers
    entry = next((m for m in samplers.LIBRARY.get() if m["model_key"] == model), None)
    if entry is None:
        return {"ok": False, "error": f"unknown model {model}"}
    g = samplers.GPU.get()
    if not g:
        return {"ok": False, "error": "GPU data has not been sampled yet"}
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
        self._last_raw_pct = None
        self._pct_indeterminate = False

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
        raw_pct = round(completed / total * 100, 1) if total else None
        if raw_pct is not None:
            if self._last_raw_pct is not None and raw_pct < self._last_raw_pct:
                # Discovery of more work means a model-wide percentage is no
                # longer knowable. Keep byte totals, but make the bar honest.
                self._pct_indeterminate = True
            self._last_raw_pct = raw_pct
            # A completed known layer is not the completed pull; more layer
            # totals may still appear. Only the terminal success event sets 100.
            pct = None if self._pct_indeterminate else min(raw_pct, 99.9)
        else:
            pct = None
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
    succeeded = False
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
                if event.get("status") == "success":
                    succeeded = True
                    break
    except Exception as e:
        if not succeeded:
            _update_job(key, done=True, finished=time.time(), status="failed",
                        error=str(e))
            return
    if not succeeded:
        _update_job(key, done=True, finished=time.time(), status="failed",
                    error="pull stream ended before Ollama reported success")
        return
    _update_job(key, done=True, finished=time.time(), status="finished",
                error=None, pct=100.0)
    # This is the background pull thread, so refreshing inventory here keeps
    # the request path cache-only while making the new model visible promptly.
    import samplers
    samplers.LIBRARY.refresh()


def start_download(name):
    if not _claim_job(name, "download"):
        return False
    try:
        threading.Thread(target=_run_pull, args=(name, name), daemon=True).start()
    except Exception as e:
        _update_job(name, done=True, finished=time.time(), status="failed", error=str(e))
        raise
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
            age = int(now - _CAT_CACHE["fetched"]) if _CAT_CACHE["fetched"] else None
            return {"data": _CAT_CACHE["data"], "error": str(e),
                    "cached_age_s": age}


# Delete --------------------------------------------------------------------
#
# Ollama has a real delete endpoint, so none of the filesystem machinery the
# LM Studio version needed survives: no model index, no path resolution, no
# rmtree, no root containment checks. The confirmation guard stays.

def delete_model(name, confirm):
    """Delete a model. `confirm` must equal `name` exactly."""
    if not confirm or confirm != name:
        return {"ok": False, "error": "confirmation must match the model name exactly"}
    if not _begin_immediate_mutation():
        return {"ok": False, "error": "cannot delete a model while a benchmark is running"}
    try:
        try:
            loaded = ollama.loaded_models(strict=True)
        except ollama.OllamaError as e:
            return {"ok": False, "error": f"could not verify loaded models: {e}"}
        if any(m.get("model_key") == name for m in loaded):
            return {"ok": False, "error": f"{name} is loaded — unload it first"}
        try:
            ollama.api_delete("/api/delete", {"model": name})
        except ollama.OllamaError as e:
            return {"ok": False, "error": str(e)}
        # Reconcile sampled inventory immediately without another Ollama call.
        import samplers
        library, sampled = samplers.LIBRARY.peek()
        if sampled:
            samplers.LIBRARY.update(
                lambda current: [m for m in current if m.get("model_key") != name])
        return {"ok": True, "removed": [name]}
    finally:
        _end_immediate_mutation()
