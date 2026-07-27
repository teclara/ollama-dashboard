# LM Studio Dashboard — Migration Design

**Date:** 2026-07-26
**Status:** Approved
**Scope:** Replace Ollama with LM Studio as the dashboard's backend. Hard cutover — no dual-backend support.

## Context

The dashboard currently targets Ollama on `http://localhost:11434`, reading state from Ollama's REST API, its systemd journal (GIN-format access logs), and its on-disk model store.

The target machine no longer runs Ollama. It runs LM Studio 0.4.19 headless:

- Server on `http://localhost:1234` (OpenAI-compatible `/v1` plus native `/api/v0`)
- Managed by the **user** systemd unit `lmstudio-server.service`
- CLI at `~/.lmstudio/bin/lms` (not on `PATH` — `cliInstalled` is `false` in settings)
- Models under `~/.lmstudio/models`
- Server logs under `~/.lmstudio/server-logs/YYYY-MM/YYYY-MM-DD.N.log`, daily rotation
- Settings at `~/.lmstudio/settings.json`

## Goals

1. Every panel that has an LM Studio equivalent keeps working, using the richest available source.
2. Panels with no data source behind them are removed, not faked with zeros.
3. Gain the capabilities LM Studio offers that Ollama did not — notably explicit model loading with tuning parameters.
4. Stay dependency-free: Python 3 standard library only.

## Non-goals

- Supporting Ollama and LM Studio simultaneously.
- Renaming the repository directory or git remote (left to the operator; it touches their systemd unit paths).
- Proxying inference traffic through the dashboard to recover request telemetry (considered and rejected — see "Request telemetry" below).

## Architecture

Unchanged. The existing four-module split is a good fit and survives the migration:

```
config.py          environment-driven settings
sources.py         read-only state (models, gpu, logs, disk, service, tailscale)
control.py         load/unload, downloads, deletes, scenarios, catalog scrape
server.py          HTTP routing + main
templates/
  index.html       dashboard UI
  control.html     control panel UI
```

`subprocess` calls to `lms` keep the zero-dependency property intact.

## Data source mapping

| Panel | Ollama source (old) | LM Studio source (new) | Direction |
|---|---|---|---|
| Loaded models | `GET /api/ps` | `lms ps --json` | Richer |
| Model library | `GET /api/tags` | `lms ls --json` + `GET /api/v0/models` for load state | Richer |
| Service info | `systemctl show ollama`, `GET /api/version` | `systemctl --user show lmstudio-server`, `lms runtime ls` | Changed |
| Requests / stats | `journalctl -u ollama`, GIN regex | `~/.lmstudio/server-logs/*.log` | Degraded |
| Server config | systemd override file | `~/.lmstudio/settings.json` (whitelisted keys) | Changed |
| Disk | `OLLAMA_MODEL_DIRS` guess-list | `downloadsFolder` from settings, plus bundled-models root | Improved |
| GPU, host, PCIe, tailscale | nvidia-smi, /proc, tailscale | unchanged | — |

### Loaded models

`lms ps --json` returns, per loaded instance: `identifier`, `modelKey`, `displayName`, `publisher`, `architecture`, `quantization{name,bits}`, `paramsString`, `sizeBytes`, `contextLength`, `maxContextLength`, `ttlMs`, `status`, `queued`, `parallel`, `lastUsedTime`, `vision`, `trainedForToolUse`.

This is strictly more than Ollama's `/api/ps` provided. New fields surfaced on the card: loaded vs. max context, TTL, in-flight queue depth, parallelism, and vision/tool-use capability badges.

Normalize into a stable internal shape so the template does not depend on LM Studio's field names directly.

### Model library

`lms ls --json` gives every model on disk with `sizeBytes`, `architecture`, `quantization`, `paramsString`, `maxContextLength`, `vision`, `trainedForToolUse`. `GET /api/v0/models` gives `state` (`loaded` / `not-loaded`) and `type` (`llm` / `vlm` / `embeddings`). Join on model key.

Note the old library rows carried `modified_at`; LM Studio exposes no equivalent. Drop the column.

`lms ls --json` already collapses LM Studio's internal index down to user-facing models (5 rows on this machine, against 9 raw index entries), so the library panel uses it directly and does not need to deduplicate virtual and concrete entries itself. Only the delete path needs the raw index — see below.

### Service info

The unit is a **user** unit, so all systemd calls need `--user`. Config exposes `LMSTUDIO_SYSTEMD_USER` (default `1`) to control this, because a system-level install is possible on other machines.

There is no `/api/v0/version` endpoint (verified: returns `Unexpected endpoint or method`). `lms version` reports only a CLI commit hash, which is not useful. Instead report the **selected inference engine** from `lms runtime ls` — e.g. `llama.cpp-linux-x86_64-nvidia-cuda12-avx2 @ 2.27.1` — which is what actually determines inference behavior. The same string also appears in the `runtime` block of `/api/v0/chat/completions` responses and can be cross-checked there.

PID, RSS, and uptime continue to come from `/proc` as before.

### Request telemetry (degraded)

Ollama's GIN logs provided timestamp, HTTP status, latency, client IP, method, and path. LM Studio's logs provide substantially less:

```
[2026-07-26 22:24:10][DEBUG] Received request: POST to /v1/chat/completions with body {
[2026-07-24 22:16:08][INFO][qwen/qwen3.6-35b-a3b] Running chat completion on conversation with 2 messages.
[2026-07-24 22:16:09][INFO][qwen/qwen3.6-35b-a3b] Model generated tool calls
[2026-07-24 22:16:11][INFO][qwen/qwen3.6-35b-a3b] Generated prediction
[2026-07-24 22:16:12][INFO][qwen/qwen3.6-35b-a3b] Finished streaming response
```

No status code, no latency, no client IP.

**Decision:** parse what exists and delete the panels that have no data.

Kept, rebuilt on the new parser:
- Recent request list (timestamp, method, path)
- Request count and requests/sec over the rolling window
- Top endpoints by count
- **New:** per-model activity counts (completions started, predictions generated, tool calls emitted, streams finished)

Removed entirely:
- Latency average / p50 / p95
- Error count and error rate
- Top clients by IP

A proxy mode (dashboard fronting LM Studio, recording real metrics) would recover all of this, but makes the dashboard load-bearing for inference traffic and requires repointing every client. Rejected as disproportionate.

**Log discovery:** logs live in `LMSTUDIO_LOG_DIR/YYYY-MM/YYYY-MM-DD.N.log`. Read the most recently modified file, tailing the last `LMSTUDIO_LOG_WINDOW_LINES` lines. When the window would span a date rollover, also read the previous day's newest file so the window is not truncated at midnight. Filter noise paths (`/api/v0/models`, `/v1/models`, `/lmstudio-greeting`) exactly as the Ollama version filtered its own polling paths.

### Server config

`~/.lmstudio/settings.json` is mostly desktop UI preferences. Surface only these:

| Key | Meaning |
|---|---|
| `downloadsFolder` | Models root — also drives the disk panel |
| `defaultContextLength` | Default context for newly loaded models |
| `modelLoadingGuardrails` | Load-size guardrail mode and threshold |
| `enableLocalService` | Whether the headless service is enabled |
| `useHFProxy` | Whether downloads route through LM Studio's HF proxy |

**Security requirement:** the current implementation puts the entire override file into a `raw` field rendered in the UI. `settings.json` contains `hfSearchToken` and `hfDownloadToken` — credential fields, empty on this machine but not in general. **Do not port the `raw` field.** Whitelist the keys above and emit nothing else. No `raw`, no passthrough of unrecognized keys.

## Control panel

### Load

New capability — the Ollama version had no load control.

Form: model picker (from library), plus optional context length, GPU offload ratio (`off` / `max` / 0–1), TTL seconds, parallel count, and identifier. Shells to:

```
lms load -y <model-key> [-c N] [--gpu R] [--ttl S] [--parallel N] [--identifier ID]
```

Before loading, run the same command with `--estimate-only` and show the resource estimate, so the user sees the VRAM cost before committing. Load runs in a background thread with status tracked in the same in-memory job map the pull code uses today.

### Unload

`lms unload <identifier>`, plus an unload-all action (`lms unload --all`).

### Download

Replaces the pull panel. Text input takes an LM Studio model identifier (`qwen/qwen3.5-9b@q8_0`, or a full Hugging Face URL). Shells to `lms get -y <name>` in a daemon thread, parsing progress from stdout into the existing `{status, completed, total, rate_bps, done, error}` job shape so the progress UI carries over with minimal change.

`lms get` progress output format is not documented and must be determined empirically during implementation. If a percentage or byte count cannot be parsed reliably, fall back to an indeterminate in-progress state with the raw last line shown — a download that reports "running" is acceptable; one that appears stuck is not.

### Catalog

Scrape `https://lmstudio.ai/models`, cached under the existing TTL logic. The page is a Next.js app, but the model cards render in the server HTML as anchors:

```html
href="/models/qwen3.6">…Qwen3.6…27B…35B…
```

Yields ~41 staff-pick entries with slug, display name, and available parameter sizes. Each row's slug feeds the download box directly (`lms get qwen3.6`).

This is thinner than the ollama.com/library panel, which carried descriptions, pull counts, capability tags, and update dates. That page does not expose them. There is no JSON catalog API — `https://lmstudio.ai/api/models` returns HTML, not JSON (verified).

### Delete

Removes a model's files from disk, since LM Studio exposes no delete API and `lms` has no remove command.

**The `path` field from `lms ls --json` must not be used for this.** It is only sometimes a real relative path. Verified on this machine:

| `lms ls` `modelKey` | `lms ls` `path` | Real location |
|---|---|---|
| `cyberpal2.0-20b-i1` | `mradermacher/CyberPal2.0-20B-i1-GGUF/…gguf` | matches — real path |
| `google/gemma-4-31b` | `google/gemma-4-31b` | **does not match** — weights are at `models/lmstudio-community/gemma-4-31B-it-GGUF/` |

The authoritative source is `~/.lmstudio/.internal/model-index-cache.json`, whose `models[]` entries carry `indexedModelIdentifier`, `containingDirAbsolutePath`, and `sourceDirectoryType`.

**Three storage classes**, which the delete logic must distinguish:

| `sourceDirectoryType` | Location | Meaning |
|---|---|---|
| `user` | `~/.lmstudio/models/<publisher>/<repo>/` | Real GGUF weights. This is what reclaims disk. |
| `hub` | `~/.lmstudio/hub/models/<owner>/<name>/` | A virtual-model **pointer**, not weights. Resolves to a concrete `user` model. |
| `bundled` | `~/.lmstudio/.internal/bundled-models/…` | Ships with LM Studio. Never delete. |

Hub entries appear in the index twice — once bare (`google/gemma-4-31b`) and once suffixed with the concrete target (`google/gemma-4-31b@lmstudio-community/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf`). The suffixed form after `@` is the resolution mapping from virtual key to concrete `user` model.

**Resolution algorithm:**

1. Look up the model key in the index.
2. If `sourceDirectoryType` is `bundled`, refuse — return a clear "bundled model, cannot delete" error.
3. If `user`, the target is its `containingDirAbsolutePath`.
4. If `hub`, find the index entry whose id is `<key>@<concrete>`, take the part after `@`, and look that up to get the concrete `user` entry's `containingDirAbsolutePath`. Delete **both** the concrete weights directory and the hub stub directory. If no `@` entry exists, refuse rather than guess.

**Safety requirements, all mandatory:**

1. `os.path.realpath` every target and confirm it is strictly inside one of the two permitted roots — the models root (`downloadsFolder`) or the hub models root. Anything resolving elsewhere, including via symlink, is refused. This guard is non-negotiable and gets a direct unit test.
2. Require the client to send the exact model key as a typed confirmation; mismatch is a 400.
3. Refuse to delete a model that is currently loaded — unload first.
4. Refuse if the resolved directory would be a permitted root itself, not a subdirectory of one.

## Benchmarks

All six scenarios (`baseline`, `reasoning`, `coding`, `needle29k`, `tool_call`, `abliter`) plus custom prompts port to `POST /api/v0/chat/completions`.

Request changes:
- `options.num_predict` → `max_tokens`
- `tools` block is already OpenAI-shaped; unchanged

Response changes:
- `message.thinking` → `choices[0].message.reasoning_content`
- `message.content` → `choices[0].message.content`
- `message.tool_calls` → `choices[0].message.tool_calls`
- Manual tok/s math from `eval_count` / `eval_duration` is replaced by the server-provided `stats` block:

```json
"stats": {
  "tokens_per_second": 65.95,
  "time_to_first_token": 0.136,
  "generation_time": 0.591,
  "stop_reason": "maxPredictedTokensReached"
}
```

Prompt token count comes from `usage.prompt_tokens`; completion from `usage.completion_tokens`, with `usage.completion_tokens_details.reasoning_tokens` surfaced separately for reasoning models.

**Gain:** time-to-first-token, stop reason, and the `model_info` / `runtime` blocks (arch, quant, context length, engine version) were all unavailable from Ollama. Surface TTFT alongside tok/s in the results card.

The haystack corpus configuration carries over unchanged.

## Configuration

All `OLLAMA_*` variables become `LMSTUDIO_*`. The dashboard's own host and port defaults are unchanged.

| Variable | Default | Notes |
|---|---|---|
| `LMSTUDIO_DASHBOARD_HOST` | `127.0.0.1` | |
| `LMSTUDIO_DASHBOARD_PORT` | `11435` | Unchanged from the Ollama version |
| `LMSTUDIO_URL` | `http://localhost:1234` | |
| `LMSTUDIO_LMS_BIN` | `~/.lmstudio/bin/lms` | Not on `PATH` by default |
| `LMSTUDIO_SYSTEMD_UNIT` | `lmstudio-server` | |
| `LMSTUDIO_SYSTEMD_USER` | `1` | Use `systemctl --user` / `journalctl --user` |
| `LMSTUDIO_LOG_DIR` | `~/.lmstudio/server-logs` | |
| `LMSTUDIO_SETTINGS_PATH` | `~/.lmstudio/settings.json` | |
| `LMSTUDIO_MODEL_INDEX` | `~/.lmstudio/.internal/model-index-cache.json` | Authoritative model→path index; required for delete |
| `LMSTUDIO_HUB_MODELS_DIR` | `~/.lmstudio/hub/models` | Virtual-model stubs; second permitted delete root |
| `LMSTUDIO_CATALOG_URL` | `https://lmstudio.ai/models` | |
| `LMSTUDIO_CATALOG_TTL` | `3600` | Was `OLLAMA_LIBRARY_TTL` |
| `LMSTUDIO_CATALOG_UA` | `lmstudio-dashboard/1.0` | Was `OLLAMA_LIBRARY_UA` |
| `LMSTUDIO_HAYSTACK_PATH` | `/tmp/moby.txt` | |
| `LMSTUDIO_HAYSTACK_WORDS` | `21000` | |
| `LMSTUDIO_GPU_HISTORY_LEN` | `60` | |
| `LMSTUDIO_PCIE_HISTORY_LEN` | `60` | |
| `LMSTUDIO_LOG_WINDOW_LINES` | `600` | |
| `LMSTUDIO_STATS_WINDOW_SEC` | `300` | |

Removed: `OLLAMA_MODEL_DIRS` (models root now comes from `downloadsFolder` in settings, with `~/.lmstudio/models` as fallback if settings are unreadable), `OLLAMA_SYSTEMD_OVERRIDE` (replaced by `LMSTUDIO_SETTINGS_PATH`).

No backward compatibility shim for the old variable names. This is a clean cutover on a single-operator machine; a shim would outlive its usefulness.

## Error handling

Preserve the existing philosophy: every source function catches broadly and returns an empty or error-tagged result so one failing panel never takes down `/api/state`.

Specific additions:

- **`lms` binary missing or non-executable** — surface a single explicit "LM Studio CLI not found at `<path>`" banner rather than silently empty model panels, since it disables four panels at once.
- **LM Studio server down** — model state falls back to `lms ls --json` (which reads disk, not the server) so the library still renders with everything marked not-loaded.
- **Log directory missing** — request panels render empty with an explanatory note.
- **Catalog scrape failure** — serve the last cached result with its age, exactly as the current library scrape does.
- **Subprocess timeouts** — every `lms` invocation gets an explicit timeout. Reads use short timeouts (2–5s); `lms get` and `lms load` run in threads without a wall-clock timeout, as they legitimately take minutes.

## Testing

The repo has no tests today. Add a stdlib `unittest` suite covering the pure functions where breakage is silent and likely, run against captured fixtures with no live server required:

| Target | What it must prove |
|---|---|
| Log parser | Extracts requests and per-model events from a real captured log excerpt; correctly ignores noise paths; handles the date-rollover window |
| `lms ps --json` normalization | Maps a real captured payload to the internal shape; tolerates missing optional fields (`ttlMs: null`, absent `quantization`) |
| `lms ls --json` + `/api/v0/models` join | Load state lands on the right rows; models present in one source but not the other do not crash the join |
| Catalog scraper | Extracts slug/name/sizes from a captured copy of the page; returns `[]` rather than raising on unrecognized markup |
| Delete path guard | Rejects traversal, absolute paths outside the permitted roots, symlinks escaping them, and a target equal to a root itself |
| Delete resolution | A `user` key resolves to its own dir; a `hub` key resolves through its `@`-suffixed index entry to the concrete weights dir *and* the stub dir; a `bundled` key is refused; a `hub` key with no `@` entry is refused rather than guessed |
| Settings whitelist | Emits only whitelisted keys; a fixture containing a populated `hfDownloadToken` must not appear anywhere in the output |
| Benchmark response mapping | A captured `/api/v0/chat/completions` response maps to the results shape, including reasoning content and the stats block |

Fixtures are captured from this machine during implementation and committed under `tests/fixtures/`.

Live verification: run the dashboard against the actual LM Studio instance and confirm every panel renders, plus one real load, unload, download, and benchmark run.

## Risks

- **Unversioned surfaces.** `lms ps`/`ls` JSON shapes, the server log format, and the catalog page markup are all undocumented and will break on some future LM Studio release. Mitigation: every parser degrades to an empty panel rather than raising; fixture tests make a break obvious and localized.
- **`lms get` progress format is unknown.** Determined empirically during implementation; fallback to indeterminate progress is specified above.
- **Delete is genuinely destructive** with no API-level undo, and its path resolution is the most intricate logic in the migration — three storage classes and a virtual-to-concrete indirection. Mitigated by sourcing paths from LM Studio's own index rather than reconstructing them, refusing on any unresolved case instead of guessing, the two-root path guard, typed confirmation, loaded-model refusal, and dedicated tests. Implement delete last, after everything else is verified working.
- **`model-index-cache.json` is a cache, not an API.** It could go stale or move. Delete must re-read it fresh on every request (never cache it in-process) and refuse cleanly if it is missing or unparseable.

## Out of scope / operator follow-ups

- Renaming the working directory and git remote to `lmstudio-dashboard`.
- A stale `lms load cyberpal2.0-20b-i1 --identifier cpal` process (PID 298888, started Jul 24) is sitting on the machine. Unrelated to this work but worth clearing.
