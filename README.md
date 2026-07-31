# Ollama Dashboard

A local dashboard for [Ollama](https://ollama.com). Live view of loaded models, the on-disk library, GPU and host metrics, plus a control panel for loading, benchmarking, pulling, and deleting models.

No dependencies — Python 3 standard library only.

## Run

```bash
python3 server.py
```

Defaults to `http://127.0.0.1:11435` and expects Ollama on `http://localhost:11434`.

- Dashboard: `http://localhost:11435/`
- Control panel: `http://localhost:11435/control`

The interface is organized into six addressable workspaces rather than one long dashboard: **Overview**, **Activity**, **Models**, **Performance**, **System**, and **Settings**. Desktop uses a persistent workspace rail; narrow screens use an accessible navigation drawer.

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

- **`/api/live`** — GPU, host CPU/RAM, and PCIe throughput. About 1 KB, polled 10×/second.
- **`/api/state`** — everything else: model lists, request panels, service info, settings. Polled every 2 seconds.
- **`/api/control/jobs`** — active and failed background work shown on Overview. Polled every second.

None of these endpoints does source work on the request path. Every source is sampled by a background thread on its own cadence and served from memory.

Ollama's API is fast enough that this is not about latency. It exists because polling from the request path would flood the GIN access log that the dashboard exists to display — every refresh would appear as a logged, timed, client-attributed request, crowding out the real traffic — and because `du -sb` over a 60+ GB model store must never block a response.

Requests the dashboard makes to `/api/tags`, `/api/ps` and `/api/version` are filtered from the statistics, but **only when they come from loopback**. Other clients hitting the same endpoints — an Open WebUI container, say — are real consumers and stay visible.

## What the panels show

**Loaded models** carry a placement badge derived from `size` vs `size_vram`. When they differ, layers spilled to the CPU; the badge shows how many bytes. On a single card running ~30B Q4 models this is the load-health signal that matters most.

**Request statistics** come from Ollama's GIN access lines, which carry an HTTP status, a latency, and a client address. That gives error rate, p50/p95/p99 latency, per-endpoint p95, and a per-client breakdown.

**The model column is inferred, and labelled as such.** GIN lines carry no model name, and the only place the journal names a model is by weights-blob SHA, which does not match the manifest digest in `/api/tags`. So the dashboard builds residency intervals from `/api/ps` — including each model's `expires_at` — and asks which model was loaded for the whole time a request was in flight.

Because GIN logs a line when a request *completes*, the span matters: a request is matched over `[completed - latency, completed]`, not at its end timestamp. A request whose span crosses a model swap reports no model rather than naming the one that happened to be loaded when it finished. Residency is never carried past the keep_alive expiry, so a stalled sampler cannot keep attributing new traffic to a stale model. Exact under `OLLAMA_MAX_LOADED_MODELS=1`; above that it records "unknown" rather than guessing.

**Follower health is visible.** The journal follower is a long-lived `journalctl -f`. If the process dies the dashboard says so, rather than presenting a frozen window as current. It keys on the subprocess rather than on line arrival — a healthy follower watching an idle server reads nothing for minutes, and warning on that is a false alarm.

## Control panel

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

The **Configuration** panel reads `systemctl show ollama --property=Environment` — Ollama has no settings file, its configuration *is* the unit's environment. Variables whose names match `KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL` are redacted before the value ever reaches a response body.

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

> **The shipped unit binds `0.0.0.0`.** That exposes the control panel — which can load, download and delete models — to anything that can reach this host on the LAN, with no authentication. Change `OLLAMA_DASHBOARD_HOST` to `127.0.0.1` if you do not want that.

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
  app.js           workspace routing and responsive navigation
  index.html       dashboard UI
  control.html     control panel UI
tests/             unittest suite + captured fixtures
```

## License

MIT
