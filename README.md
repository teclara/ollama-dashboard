# Ollama Dashboard

A local dashboard for [Ollama](https://ollama.com). Live view of loaded models, the on-disk library, GPU and host metrics, plus controls for loading, benchmarking, pulling, and deleting models.

No dependencies — Python 3 standard library only.

## Run

```bash
git clone https://github.com/teclara/ollama-dashboard.git
cd ollama-dashboard
python3 server.py
```

Defaults to `http://127.0.0.1:11435` and expects Ollama on `http://localhost:11434`.

- Dashboard: `http://localhost:11435/`

Everything lives on one page. A fixed left rail carries the live instrument — a plain-language verdict (`Serving` / `Ready` / `Idle` / `Degraded` / `Offline`), the resident model, and compute, memory, i/o, request-rate and host meters — and never scrolls or reflows. Beside it, four addressable sections hold the work: **Models** (`#models`), **Activity** (`#activity`), **Host** (`#host`) and **Bench** (`#bench`). `⌘K` / `Ctrl-K` opens a command palette for jumping and for running actions directly.

`/control` is no longer a page. It redirects to `/#models` so bookmarks from the older two-page layout keep working, as do the old hashes (`#overview`, `#performance`, `#system`, `#settings`).

Below roughly 1120px the rail becomes a horizontal instrument strip above the workbench rather than shrinking, because a narrow rail stops being legible at a distance.

Ollama must be running (`systemctl status ollama`). Everything goes over Ollama's HTTP API — there is no CLI dependency.

## Permissions

Two group memberships change how much the dashboard can see. It degrades rather than failing without either.

| Group | Grants | Without it |
|---|---|---|
| `adm` | Reading the journal, which is where Ollama logs | The request panels stay empty |
| `ollama` | Reading `/usr/share/ollama` (mode `0750`) | The model-store size falls back to summing `/api/tags`, which counts only *referenced* blobs and so understates the total. The UI marks it `≥` and labels it a lower bound. |

```bash
sudo usermod -aG adm,ollama "$USER"   # then log out and back in
```

With the `ollama` group the dashboard also reports **reclaimable** space — the gap between what `du` sees and what any manifest references, i.e. orphaned blobs left by model churn.

## How it refreshes

The monitoring interface polls three cache-only endpoints:

- **`/api/live`** — GPU, host CPU/RAM, and PCIe throughput. About 1 KB, polled 4×/second.
- **`/api/state`** — everything else: model lists, request panels, service info, settings. Polled every 3 seconds.
- **`/api/control/jobs`** — active and failed background work, shown in the rail and on Models. Polled every second.

None of these endpoints does source work on the request path. Every source is sampled by a background thread on its own cadence and served from memory.

Ollama's API is fast enough that this is not about latency. It exists because polling from the request path would flood the GIN access log that the dashboard exists to display — every refresh would appear as a logged, timed, client-attributed request, crowding out the real traffic — and because `du -sb` over a 60+ GB model store must never block a response.

Requests the dashboard makes to `/api/tags`, `/api/ps` and `/api/version` are filtered from the statistics, but **only when they come from loopback**. Other clients hitting the same endpoints — an Open WebUI container, say — are real consumers and stay visible.

## What it shows

**The verdict is the headline.** The rail's largest line answers "is this box healthy" before you focus your eyes on anything else. It has five states, each carrying its own one-line reason, so the colour is never the only thing telling you what is going on:

| State | Means |
|---|---|
| `Serving` | A model is resident and requests arrived in the window |
| `Ready` | A model is resident, no recent requests |
| `Idle` | Ollama is reachable, nothing resident |
| `Degraded` | Ollama answers, but something is wrong: the unit is not active, the journal follower died, GPU telemetry is unavailable, ≥5% of requests are failing, or a job failed |
| `Offline` | Ollama is not answering at all |

**Resident models** carry a placement badge derived from `size` vs `size_vram`. When they differ, layers spilled to the CPU; the badge shows how many bytes. On a single card running ~30B Q4 models this is the load-health signal that matters most.

**Request statistics** come from Ollama's GIN access lines, which carry an HTTP status, a latency, and a client address. That gives error rate, p50/p95/p99 latency, per-endpoint p95, and a per-client breakdown.

**The model column is inferred, and labelled as such.** GIN lines carry no model name, and the only place the journal names a model is by weights-blob SHA, which does not match the manifest digest in `/api/tags`. So the dashboard builds residency intervals from `/api/ps` — including each model's `expires_at` — and asks which model was loaded for the whole time a request was in flight.

Because GIN logs a line when a request *completes*, the span matters: a request is matched over `[completed - latency, completed]`, not at its end timestamp. A request whose span crosses a model swap reports no model rather than naming the one that happened to be loaded when it finished. Residency is never carried past the keep_alive expiry, so a stalled sampler cannot keep attributing new traffic to a stale model. Exact under `OLLAMA_MAX_LOADED_MODELS=1`; above that it records "unknown" rather than guessing.

**Follower health is visible.** The journal follower is a long-lived `journalctl -f`. If the process dies the dashboard says so, rather than presenting a frozen window as current. It keys on the subprocess rather than on line arrival — a healthy follower watching an idle server reads nothing for minutes, and warning on that is a false alarm.

## Interface

The dashboard is built to be read from a few feet away on an always-on second monitor, and worked in up close when something needs doing. Most of the decisions below follow from that one fact.

**The rail never moves.** Its DOM is built once and the poll loop writes only text values and bar widths — it never inserts, removes or reorders a node. Numerics are fixed-width tabular figures, and any text whose length varies with the data truncates rather than wraps. Without that, a GPU going from `11W` to `111W` rewraps a line and shoves every meter below it down the page, which is exactly what fixed geometry exists to prevent. The full untruncated value is always on **Host**.

**Telemetry is smoothed, faults are not.** Continuous readings are damped before display, because a raw value at poll rate is visual noise in peripheral vision. Anything that signals a fault — service state, error counts, follower health — is written exactly as it arrives, undamped and immediately.

**Colour means something specific.** Four data roles are pinned to what they measure and are identical in every bar, sparkline and label:

| Role | Carries |
|---|---|
| compute | GPU utilization, host CPU |
| memory | VRAM, host RAM |
| i/o | PCIe rx and tx |
| requests | request rate, latency percentiles, job progress |

Semantic colour (good / warning / bad) sits **outside** those four hues, so an alert can never be mistaken for a data series. Hue placement is forced by adjacency: compute and memory share the GPU sparkline so they sit 95° apart, and requests sits beside error counts so it sits 120° from bad. There is deliberately **no chromatic accent** — primary actions and selection use ink, because a fifth hue would read as a fifth data role.

**Both themes are first class.** An always-on monitor follows the room, so `prefers-color-scheme` decides by default and the rail's Theme button cycles auto → light → dark, persisting your choice. Data hues shift lightness between themes to hold contrast at small sizes.

**Type has a hard 14px floor.** At viewing distance the usual 10–12px label sizes are unreadable. The floor compresses the bottom of the scale, so the smallest steps separate by weight, case and colour rather than size.

**Percentage charts keep a fixed 0–100% scale** with gridlines, rather than auto-scaling. Auto-scaling would make 27% VRAM look full.

**Keyboard and accessibility.** `⌘K` / `Ctrl-K` opens the command palette, which can jump between sections and run actions (load, unload, download, clear jobs, switch theme) directly. Sections are a proper tablist with arrow-key navigation. Focus is always visible, wide tables become keyboard-scrollable only when they actually overflow, status is never carried by colour alone, and `prefers-reduced-motion` removes transitions.

## Operations

- **Load** with explicit `num_ctx`, `num_gpu` and `keep_alive`. Ollama keys a distinct runner per option set, so these genuinely apply per load. `OLLAMA_NUM_PARALLEL` and `OLLAMA_MAX_LOADED_MODELS` are server-wide and shown read-only beside the form.
- **Check fit** compares the model's on-disk size to free VRAM. It is not an estimate — Ollama has no equivalent of LM Studio's `--estimate-only` — and it ignores KV cache and context, so treat it as a smell test.
- **Benchmark** runs local completion models sequentially with configurable warm-up and measured runs. It reports median time to first token, prompt tokens/second, generation tokens/second, total time, and per-run evidence using Ollama's own token counts and nanosecond timings. Benchmark jobs are exclusive of dashboard-initiated model operations; other clients must be kept idle for clean results. Embedding models are excluded because they require a different workload and metrics.
- **Pull** streams progress from `/api/pull`. Progress is reported per blob digest and summed across layers, so the bar does not walk backwards at layer boundaries.
- **Delete** calls `DELETE /api/delete`. It refuses to delete a loaded model, and requires the model name typed back verbatim.

## Configuration

Every dashboard setting is prefixed `OLLAMA_DASHBOARD_*`. Bare `OLLAMA_*` is Ollama's own namespace and is never shadowed — `OLLAMA_MODELS` is read as an input so the dashboard follows the server. `OLLAMA_URL` is the one exception, since Ollama does not define it.

| Variable | Default |
|---|---|
| `OLLAMA_DASHBOARD_HOST` | `127.0.0.1` |
| `OLLAMA_DASHBOARD_PORT` | `11435` |
| `OLLAMA_URL` | `http://localhost:11434` |
| `OLLAMA_DASHBOARD_SYSTEMD_UNIT` | `ollama` |
| `OLLAMA_DASHBOARD_SYSTEMD_USER` | `0` (Ollama ships a *system* unit) |
| `OLLAMA_MODELS` | `/usr/share/ollama/.ollama/models` |
| `OLLAMA_DASHBOARD_JOURNAL_UNIT` | `ollama` |
| `OLLAMA_DASHBOARD_JOURNAL_BACKFILL` | `2000` |
| `OLLAMA_DASHBOARD_CATALOG_URL` | `https://ollama.com/library` |
| `OLLAMA_DASHBOARD_CATALOG_TTL` | `3600` |
| `OLLAMA_DASHBOARD_STATS_WINDOW_SEC` | `300` |
| `OLLAMA_DASHBOARD_LOG_WINDOW_LINES` | `2000` |
| `OLLAMA_DASHBOARD_PS_TIMELINE_LEN` | `900` |
| `OLLAMA_DASHBOARD_GPU_SAMPLE_MS` | `100` |
| `OLLAMA_DASHBOARD_LOADED_SAMPLE_SEC` | `2` |
| `OLLAMA_DASHBOARD_SLOW_SAMPLE_SEC` | `15` |

The **Service environment** section on Host reads `systemctl show ollama --property=Environment` — Ollama has no settings file, its configuration *is* the unit's environment. Variables whose names match `KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL` are redacted before the value ever reaches a response body.

## Requirements

- Python 3, standard library only
- `systemd` (for service state and the Ollama environment)
- `journalctl` and membership of `adm` (for request statistics)
- `nvidia-smi` for the GPU panel and PCIe throughput (degrades gracefully without a GPU)

## Install as a service

```bash
sudo mkdir -p /opt/ollama-dashboard
sudo cp -r *.py templates README.md /opt/ollama-dashboard/
cp ollama-dashboard.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ollama-dashboard.service
```

It runs as a *user* unit because it needs your group memberships (`adm`, `ollama`). Ollama itself is a system unit, so the dashboard cannot `Require` it — it degrades gracefully when Ollama is down, which is a state it is meant to display.

> **The shipped unit binds `0.0.0.0`.** That exposes the operational controls — which can load, download and delete models — to anything that can reach this host on the LAN, with no authentication. Change `OLLAMA_DASHBOARD_HOST` to `127.0.0.1` if you do not want that.

## Tests

```bash
python3 -m pytest -q
```

225 tests, standard library only, no running Ollama required. Fixtures are captured from a real instance: `/api/tags`, `/api/ps`, a journald excerpt with real GIN lines, and the `ollama.com/library` HTML.

## Layout

```
config.py          environment-driven settings
ollama.py          the Ollama HTTP API and payload normalization
logs.py            journald following, GIN parsing, and aggregation
samplers.py        background sampling; keeps the request path off Ollama
sources.py         read-only state (gpu, host, disk, service, tailscale)
control.py         load, unload, pull, delete, catalog scrape
server.py          HTTP routing + main
templates/
  app.css          shared shell and design tokens
  app.js           whole client runtime: rail, views, actions, palette
  favicon.svg      dashboard browser icon
  index.html       the dashboard, single page
  control.html     redirect shim for old /control bookmarks
tests/             unittest suite + captured fixtures
```

## License

MIT
