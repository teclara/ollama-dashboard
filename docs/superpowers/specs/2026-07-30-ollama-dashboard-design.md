# Design: port the dashboard from LM Studio to Ollama

**Date:** 2026-07-30
**Status:** approved

## Why

LM Studio was removed from this machine on 2026-07-30. Ollama is now the only
model server, running as a system unit on `:11434`. The dashboard must follow.

This is a **forward port of the current architecture**, not a revert. The
dashboard was previously an Ollama dashboard, and `f67787c` migrated it to LM
Studio — but that migration is also what added the background sampler layer,
the log parser, job tracking, delete guards, and 150 tests. Reverting to
`c4f5e14~1` would discard all of it. The commits before the LM Studio era are
prior art to consult, not a target to return to.

## Environment this targets

Verified live on 2026-07-30:

- Ollama 0.32.5, `/usr/local/bin/ollama`, system unit `ollama.service`
- `OLLAMA_HOST=http://0.0.0.0:11434`, `OLLAMA_ORIGINS=*`,
  `OLLAMA_CONTEXT_LENGTH=32768`, `OLLAMA_FLASH_ATTENTION=true`,
  `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_KEEP_ALIVE=30m`,
  `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`
- Model store `/usr/share/ollama/.ollama/models`, mode `0750 ollama:ollama`
- 4 models on disk; `du` 67.5 GB vs `/api/tags` sum 48.9 GB
- Logs to journald; `dubthecoder` is in `adm` (journal) and `ollama` (store)
- Single RTX 5090, 32.6 GB VRAM

## Scope

**In:** full port of every current feature except benchmarks, plus richer log
statistics that Ollama's logs make possible for the first time.

**Out:** benchmark scenarios (dropped by decision), load estimation (no Ollama
equivalent), any redesign of the visual language — `PRODUCT.md` still governs.

## Architecture

The module graph is unchanged:

```
config → ollama → {sources, logs} → samplers → server → templates
```

### `ollama.py` replaces `lmstudio.py`

All Ollama coupling lives here; nothing else knows Ollama's field names. Unlike
its predecessor it makes **no subprocess calls** — everything is HTTP:

| Function | Endpoint |
|---|---|
| `loaded_models()` | `GET /api/ps` |
| `library()` | `GET /api/tags`, joined with loaded state |
| `show(model)` | `POST /api/show` |
| `version()` | `GET /api/version` |

The current module's central complication disappears. Its docstring explains
three non-interchangeable identifiers — `model_key`, `identifier`,
`indexed_id` — because LM Studio's load key, instance name, and index key all
differ. Ollama has one name (`gemma4:31b`). That entire concept, and the
`indexed_id` field threaded through normalization, delete, and the templates,
is removed.

Normalized loaded-model shape. Keys are kept identical to today's where the
concept survives, so template churn stays minimal:

```
model_key / identifier / display_name  = name          # all one value now
arch                                   = details.family
quant                                  = details.quantization_level
params                                 = details.parameter_size
size                                   = size
size_vram                              = size_vram      # NEW
context                                = context_length
ttl_s                                  = expires_at - now
vision / tools / thinking / embedding  = from capabilities[]
```

Two genuinely new signals Ollama exposes and LM Studio did not:

- **`size_vram` vs `size`** — when these differ, layers are on the CPU. This is
  the single most useful load-health indicator on a 32.6 GB card running ~30B
  Q4 models, and gets an explicit badge rather than being left to arithmetic.
- **`capabilities[]`** — `thinking` and `embedding` join the existing
  `vision` and `tools` flags.

Note `/api/tags` is inconsistent about `details.context_length` (present for
`ornith:35b`, absent for `gemma4:31b`). Library rows leave `max_context` null
when absent rather than inventing a value; `/api/show` fills it on demand for
the detail view.

### `samplers.py` keeps its structure, loses its justification

The module docstring currently justifies the entire layer by `lms` costing
~200ms of Node startup per invocation. Ollama is HTTP at microsecond latency,
so that argument no longer holds and the docstring must be rewritten rather
than left as a false explanation of why the code looks this way.

The layer stays, for two reasons that do still hold:

1. Polling from the request path would flood the GIN access log — the very
   window the dashboard exists to display. This mattered under LM Studio and
   matters more under Ollama, where every poll is a logged, timed, attributed
   request.
2. `du -sb` over a 67 GB model store must never be on the request path.

Sampling cadences are unchanged. One addition: the `/api/ps` sampler also
appends to a **loaded-model timeline** (see below).

### `logs.py` — rewritten against journald

Ollama logs to journald, not to files. The reader becomes a long-lived
`journalctl -u ollama -f -o cat` streamed into a deque, restarted on exit.
This mirrors the existing `_gpu_stream_loop` and `_pcie_dmon_loop` patterns
rather than spawning `journalctl` once per second, and it removes the entire
byte-tailing / file-rollover subsystem (`log_files`, `tail_lines`, and the
`read_window` byte-budget logic), which existed only because LM Studio wrote
multi-megabyte DEBUG request bodies to rotating files.

GIN access lines are the primary signal:

```
[GIN] 2026/07/30 - 23:25:18 | 200 |  51.336µs |  ::1 | GET "/api/ps"
        timestamp            status   latency   client  method  path
```

The duration parser must handle `ns`, `µs`, `ms`, `s`, and `m` suffixes.

**New aggregates**, none of which were possible under LM Studio:

- `stats()` gains **error rate** (non-2xx / total) and **p50/p95/p99 latency**
- `top_endpoints()` gains per-path p95 and error count
- `by_client()` returns per-IP request counts and last-seen — restoring the
  client stats dropped in `6bf4f77`. Already meaningful here: `::1` is local,
  `172.17.0.3` is the Open WebUI container.

`level=WARN|ERROR` lines are collected into a small separate `problems` list.
No attempt is made to parse general llama.cpp loader output.

Existing noise-path filtering carries over with Ollama's paths:
`/api/tags`, `/api/ps`, `/api/version`.

### Model attribution — the one capability loss, and its fix

Today `model_activity()` parses per-model inference events straight from LM
Studio's logs. Ollama's GIN lines carry **no model name**, and the only place
the journal names a model is by weights-blob SHA
(`model=…/blobs/sha256-1278394b…`), which does not match the manifest digest
returned by `/api/tags`. Resolving it would mean reading manifest JSON from the
model store — possible with the `ollama` group, but brittle and privilege-
dependent.

Instead the `/api/ps` sampler keeps a bounded **loaded-model timeline** of
`(timestamp, model)` observations, and request rows are attributed to whichever
model was resident at their timestamp. With `OLLAMA_MAX_LOADED_MODELS=1` this
is exact. If that limit is ever raised, attribution degrades to ambiguous
rather than to wrong, and the UI must label it as inferred rather than
observed.

### `control.py` — same surface, far less machinery

| Action | Implementation |
|---|---|
| Load | `POST /api/generate` with `prompt:""`, `keep_alive`, `options:{num_ctx, num_gpu}` |
| Unload | `POST /api/generate` with `keep_alive: 0` |
| Unload all | iterate `/api/ps` |
| Download | `POST /api/pull`, streaming newline-delimited JSON |
| Delete | `DELETE /api/delete` |
| Catalog | scrape `ollama.com/library` |

Four subsystems are deleted outright:

1. **ANSI/CR progress scraping** (`_ANSI_RE`, `_PCT_RE`, `_BYTES_RE`,
   `_RATE_RE`, `_ETA_RE`, `_stream_segments`, `parse_progress`, `clean_line`,
   ~87 lines). `/api/pull` streams `{status, completed, total, digest}` as
   proper JSON. Ollama sends no rate or ETA, so both are computed from
   `completed` deltas between chunks — a small addition that replaces a large
   fragile parser.

   **Progress is reported per layer, not per model** (verified live). A pull
   emits an independent `completed`/`total` pair for each blob digest:

   ```
   {"status":"pulling manifest"}
   {"status":"pulling 970aa74c0a90","digest":"sha256:970a…","total":274290656,"completed":274290656}
   {"status":"pulling c71d239df917","digest":"sha256:c71d…","total":11357,"completed":11357}
   {"status":"verifying sha256 digest"}
   ```

   Taking the latest `completed/total` as the bar would make it jump backwards
   every time a new layer starts. The job must instead keep a `{digest:
   (completed, total)}` map and report the **sums**. Statuses carrying neither
   field (`pulling manifest`, `verifying sha256 digest`, `writing manifest`,
   `success`) are indeterminate phases and must leave the last known
   percentage in place rather than resetting it to zero.
2. **Filesystem delete machinery** (`resolve_delete_targets`, `is_inside`,
   model-index resolution, `shutil.rmtree`, ~104 lines). Ollama has a real
   delete API. The confirmation guard — `confirm` must equal the model name
   exactly — is retained, as is the refusal to delete a loaded model.
3. **Benchmarks** (`SCENARIOS`, `WEATHER_TOOL`, `_haystack`,
   `map_chat_response`, `run_scenario`, `POST /api/control/test`, ~107 lines).
4. **`estimate_load`** — `lms load --estimate-only` has no Ollama counterpart.

`control.py` drops from 464 to roughly 290 lines and loses both of its
fragility hotspots.

**Load options.** Verified live on 0.32.5: loading `gemma4:12b` with
`{"prompt":"","keep_alive":"5m","options":{"num_ctx":8192,"num_gpu":99}}`
returned `{"done_reason":"load"}` and `/api/ps` then reported
`context_length: 8192` rather than the server default of 32768, with
`expires_at` exactly five minutes out. Unloading with `{"keep_alive":0}`
returned `{"done_reason":"unload"}` and emptied `/api/ps`.

So per-load `num_ctx`, `num_gpu`, and `keep_alive` all work, and
`done_reason` is a clean completion signal for both jobs — no output parsing
needed at all. Ollama does **not** support a per-load parallelism setting or a
custom instance identifier; those are server-wide (`OLLAMA_NUM_PARALLEL`) or
nonexistent. So the panel keeps
context, GPU layers, and TTL as live controls, and shows parallelism as
read-only server config. Where the estimate button was, the panel shows model
size against free VRAM; both numbers are already on the page, so this is a
display change rather than a new feature.

**Catalog.** `ollama.com/library` is server-rendered and parses cleanly: 234
models, each with slug, name, description, capability badges, and size tags.
This is *richer* than the LM Studio catalog, which had only name and sizes.
Cache TTL and the user-agent config carry over unchanged.

### `sources.py`

- `settings()` reads `systemctl show ollama --property=Environment` and parses
  the `OLLAMA_*` vars — a real replacement for `settings.json`, surfacing
  actual tuning (`KV_CACHE_TYPE`, `FLASH_ATTENTION`, `MAX_LOADED_MODELS`).
  Values arrive shell-quoted (`"OLLAMA_ORIGINS=*"`); the parser must unquote.
  The current whitelist inverts to a **denylist** on names matching
  `KEY|TOKEN|SECRET|PASSWORD`, since the set of vars a user may add later
  cannot be enumerated in advance.
- `models_root()` reads Ollama's own `OLLAMA_MODELS`, defaulting to
  `/usr/share/ollama/.ollama/models`.
- `service_info()` targets the **system** unit `ollama`; `SYSTEMD_USER`
  defaults to `False`.
- `engine_info()` comes from `GET /api/version` plus the existing nvidia-smi
  driver/CUDA versions, replacing `lms runtime ls` parsing.
- `lms_ok` becomes `ollama_ok`, an HTTP reachability check rather than a test
  for an executable bit.

**Disk needs explicit handling.** `/usr/share/ollama` is `0750 ollama:ollama`,
so both `statvfs` and `du` fail for a process without the `ollama` group.
`dubthecoder` was added to that group today but existing sessions have not
picked it up. Therefore:

- Filesystem totals: `statvfs` on the models root, walking up to the nearest
  traversable ancestor on `PermissionError`. Same filesystem, same numbers.
- Model store size: `du -sb` when permitted; otherwise sum `/api/tags` sizes
  and set an explicit `approximate: true` flag. The UI must show that flag —
  the gap is real and large (67.5 GB actual vs 48.9 GB summed).
- The difference between the two, when both are available, is orphaned blob
  space and is surfaced as reclaimable.

No `sudo` anywhere. The dashboard degrades rather than escalating.

### `config.py`

Every dashboard-owned variable is prefixed **`OLLAMA_DASHBOARD_*`**, not bare
`OLLAMA_*`. That namespace belongs to Ollama itself, and a dashboard
`OLLAMA_MODELS` would collide with the real one. Ollama's own variables are
read as inputs and never shadowed. `OLLAMA_URL` is the one exception, being
unambiguous and not an Ollama variable.

Port stays **11435**, clear of Ollama's 11434.

Dropped: `LMS_BIN`, `SETTINGS_PATH`, `MODEL_INDEX_PATH`, `HUB_MODELS_DIR`,
`LOG_DIR`, `LOG_TAIL_BYTES`, `HAYSTACK_PATH`, `HAYSTACK_WORDS`.
Added: `OLLAMA_DASHBOARD_JOURNAL_UNIT`, `OLLAMA_DASHBOARD_JOURNAL_BACKFILL`,
`OLLAMA_DASHBOARD_PS_TIMELINE_LEN`.

### `server.py` and templates

Routes are unchanged except that `POST /api/control/test` and
`POST /api/control/load/estimate` are removed. `_load_opts` narrows to
`context`, `gpu`, `ttl`.

`templates/index.html` (556 lines) and `templates/control.html` (466 lines)
change moderately: the benchmark panel is removed, the settings card is rebuilt
around systemd environment variables, the request table gains status and
latency columns, a client-IP breakdown is added, and loaded-model rows gain the
VRAM/CPU split badge. `PRODUCT.md` is unchanged apart from replacing the
product name — the design principles it states are not affected by which model
server sits underneath.

## Testing

150 tests pass today and the suite must stay green at every commit.

| File | Action |
|---|---|
| `test_lmstudio.py` | → `test_ollama.py`, rewritten for `/api/ps` and `/api/tags` |
| `test_logs.py` | rewritten: GIN parsing, duration units, percentiles, client stats |
| `test_jobs.py` | rewritten for JSON-stream pull progress and derived rate/ETA |
| `test_catalog.py` | rewritten for `ollama.com/library` HTML |
| `test_delete.py` | shrinks hard — most of it covers filesystem rails that no longer exist |
| `test_scenarios.py` | deleted |
| `test_sources.py` | systemd env parsing, disk fallback, permission-denied paths |
| `test_config.py`, `test_samplers.py`, `test_server.py` | moderate edits |

New fixtures captured from the live instance: `/api/tags`, `/api/ps`,
`/api/show`, a real `/api/pull` stream, a journalctl excerpt containing GIN
lines from both clients, and the `ollama.com/library` HTML.

Cases that must be covered because they are where this port can silently go
wrong: latency suffix parsing across all five units, shell-quoted systemd
environment values, `statvfs`/`du` permission denial, `size_vram < size`
partial offload, `/api/tags` rows missing `context_length`, and model
attribution when the timeline has no observation covering a request.

Two pull-progress cases deserve named tests, since both produce a plausible
but wrong progress bar rather than an obvious failure: a **multi-layer pull**
must report monotonically increasing summed progress across digests, and an
**indeterminate status** (`verifying sha256 digest`) arriving mid-pull must
leave the existing percentage untouched instead of zeroing it.

## Migration

Single branch, whole port, cut over at the end. The dashboard's user unit is
currently inactive, so nothing is disrupted while this is in flight.

1. `git mv` the repo to `~/GitHub/ollama-dashboard`; history is preserved.
2. Port module by module on the branch, suite green throughout.
3. `git mv lmstudio.py ollama.py`. Safe: the project is stdlib-only and runs
   from its own directory, so the module cannot shadow a pip `ollama` package.
4. Rewrite `README.md`; update `PRODUCT.md`'s product name.
5. Install `ollama-dashboard.service` as a user unit
   (`After=network-online.target`, no dependency on `ollama.service` since that
   is a system unit). Remove `lmstudio-dashboard.service`.
6. Deploy to `/opt/ollama-dashboard`; remove `/opt/lmstudio-dashboard`.

The unit keeps `HOST=0.0.0.0`, and the existing comment warning that this
exposes an unauthenticated control panel — one that can load, download, and
now delete models via a real API — to the whole LAN must be carried over
verbatim. It is more true after this port, not less.

## Risks

- **Journal streaming is a long-lived subprocess.** If `journalctl -f` dies the
  request window silently stops updating. It must restart on exit, and staleness
  must be visible in the UI rather than presenting a frozen window as current.
- **Model attribution is inferred, not observed.** Correct under
  `MAX_LOADED_MODELS=1`; must be labelled as inferred so raising that limit
  does not turn a soft approximation into a confident lie.
- **Disk size is approximate without the `ollama` group.** Flagged in the
  payload and shown in the UI.
- **Catalog scraping remains fragile**, as it was for LM Studio. Failure falls
  back to the cached copy and surfaces the error, as today.
