# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python3 server.py                                    # run on http://127.0.0.1:11435
python3 -m pytest -q                                 # full suite (178 tests, ~1.3s)
python3 -m pytest tests/test_logs.py -v              # one file
python3 -m pytest tests/test_logs.py::TestParseDuration::test_compound_minutes_and_seconds -q
python3 -m unittest discover -s tests                # also works; tests are unittest-style
```

Tests need no running Ollama — they use captured fixtures in `tests/fixtures/`. There is no build step, linter config, or dependency manifest.

Deployed as `/opt/ollama-dashboard` + a **user** systemd unit (`systemctl --user status ollama-dashboard`). Redeploy with `sudo cp -r *.py templates README.md /opt/ollama-dashboard/ && systemctl --user restart ollama-dashboard`.

## Hard constraints

- **Python 3 standard library only.** No pip installs, ever. This is absolute — the dashboard runs from `/opt` under a bare `python3`.
- **The request path never calls Ollama and never shells out.** `state()` and `live()` read only from `samplers.Sampled` holders. Adding a live call inside a handler is the single easiest way to break this codebase's design.
- **Dashboard-owned env vars are prefixed `OLLAMA_DASHBOARD_*`.** Bare `OLLAMA_*` is Ollama's own namespace. Only `OLLAMA_URL` (Ollama does not define it) and `OLLAMA_MODELS` (read as an *input*, never shadowed) may appear bare. `tests/test_config.py` enforces this with a static check over `config.py`'s source.
- Dashboard port is 11435; Ollama owns 11434.

## Architecture

```
config → ollama → {sources, logs} → samplers → server → templates
```

- **`ollama.py`** — the only module that knows Ollama's raw field names. Pure HTTP (`/api/ps`, `/api/tags`, `/api/show`, `/api/version`, `/api/delete`); no CLI. Live wrappers degrade to `[]`/`{}`/`False` on transport failure and never raise to callers.
- **`logs.py`** — a long-lived `journalctl -u ollama -f` streamed into a bounded deque, plus GIN access-line parsing and windowed aggregation (percentiles, error rate, per-endpoint, per-client).
- **`samplers.py`** — one background thread per source, each on its own cadence, holding the latest value in a `Sampled`. Also owns the loaded-model timeline.
- **`sources.py`** — GPU/host/disk/service/tailscale reads plus the aggregate `state()` and `live()` payloads.
- **`control.py`** — all mutating actions and the background job map.
- **`templates/`** — vanilla JS, no framework, no build. `index.html` polls `/api/live` at 10 Hz and `/api/state` every 2s.

`PRODUCT.md` governs UI decisions (design principles, anti-references, WCAG 2.2 AA). Read it before changing templates.

### Why the sampler layer exists

Not latency — Ollama's HTTP API is fast. Two other reasons, and `samplers.py`'s docstring says so:

1. Polling from the request path would flood the GIN access log that the dashboard exists to display. Every refresh would appear as a logged, timed, client-attributed request.
2. `du -sb` over a 60+ GB model store must never block a response.

An earlier version of this project targeted LM Studio, where the justification *was* latency (~200 ms of Node startup per `lms` call). Several comments still reference LM Studio to explain why something looks the way it does — those are deliberate, not stale.

## Traps

Each of these shipped as a real bug during the Ollama port. They are cheap to reintroduce.

**Go compound durations.** GIN latencies use `time.Duration.String()`. Sub-second units never compound but h/m/s do: `2m49s` is 169 seconds. A `[0-9.]+(µs|ms|s|m)` regex matches `2m` and silently drops the `49s`, under-reporting every slow request. `parse_duration` sums all components and rejects trailing junk. Unit order matters — `130ms` must not read as 130 minutes.

**Percentiles use `math.ceil`, not `round(x + 0.5)`.** The latter hits banker's rounding on exact halves and returns the 96th of 100 samples for p95.

**Pull progress is per blob digest, not per model.** `/api/pull` emits an independent `completed`/`total` for each layer. Tracking the newest pair walks the bar backwards at every layer boundary — `PullProgress` sums across a digest map. Statuses carrying neither field (`pulling manifest`, `verifying sha256 digest`) are indeterminate phases and must leave the last percentage alone rather than zeroing it.

**Noise filtering is path AND loopback, never path alone.** The dashboard polls `/api/tags` and `/api/ps` itself, but so do real consumers (an Open WebUI container on this host). Filtering by path alone erases them from the client breakdown.

**`os.path.isdir()` is False both for a missing directory and one you cannot traverse.** `/usr/share/ollama` is `0750 ollama:ollama`. Conflating the two made `disk()` report `models_size: 0, approximate: false` about ~98 GB it simply could not see. `_readable_dir()` distinguishes them; unreadable falls back to summing `/api/tags` (referenced blobs only) flagged `approximate: true`.

**Ollama emits nine fractional-second digits** in `expires_at`; `datetime.fromisoformat` accepts at most six on Python < 3.11. `_ISO_FRAC_RE` truncates before parsing.

**`settings()` uses a denylist, not a whitelist.** Users may add any `OLLAMA_*` variable, so names matching `KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL` are redacted rather than enumerating what is safe to show.

## Model attribution is inferred

GIN access lines carry no model name, and the only place the journal names a model is by weights-blob SHA, which does not match the manifest digest in `/api/tags`. So `samplers.record_timeline()` records which model was resident at each `/api/ps` sample and `attribute()` maps request timestamps onto that.

This is exact under `OLLAMA_MAX_LOADED_MODELS=1` (the current server config). With more than one model resident, `record_timeline` records `None` rather than picking one — degrading to "unknown" is correct; guessing would be a confident lie. Any UI rendering `row["model"]` must label it inferred.

**Known limitation:** GIN logs a line when a request *completes*, so a long request (a multi-minute `/api/pull`) is stamped at its end. Attributing by completion timestamp is wrong for requests spanning a model swap. Using `/api/ps`'s `expires_at` to derive residency *intervals* rather than sample points would fix this.

## Honesty affordances

Several payload fields exist so the UI never presents broken input as real data. Preserve them when touching templates:

- `log_age_s` — a dead journal follower must be shown as stale, not as a frozen-but-current window.
- `disk.approximate` — the model-store size is a lower bound when unreadable; the UI renders `≥`.
- `ollama_ok` — an unreachable server empties the model panels; say so.
- `fully_gpu` / `cpu_bytes` — `size` vs `size_vram`; when they differ, layers spilled to CPU.

Status colour is always paired with text (`PRODUCT.md` principle 4) — never carry state by hue alone.

## Fixtures

`tests/fixtures/` are captured from a live instance, not hand-written: `api_tags.json`, `api_ps_loaded.json`, `journal_excerpt.log` (real GIN lines including 401/404s and compound durations), `ollama_library.html`. Recapture with `curl`/`journalctl` rather than editing by hand; the catalog scrape in particular is validated against all ~234 real cards.

`journal_excerpt.log` needs `git add -f` — `.gitignore` has a blanket `*.log`.
