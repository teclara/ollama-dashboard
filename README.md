# LM Studio Dashboard

A local dashboard for [LM Studio](https://lmstudio.ai). Live view of loaded models, the on-disk library, GPU and host metrics, plus a control panel for loading and unloading models, downloading from the LM Studio catalog, and running quick benchmarks.

No dependencies — Python 3 standard library only.

## Run

```bash
python3 server.py
```

Defaults to `http://127.0.0.1:11435` and expects LM Studio on `http://localhost:1234`.

- Dashboard: `http://localhost:11435/`
- Control panel: `http://localhost:11435/control`

LM Studio's server must be running (`lms server start`, or the desktop app's server toggle). The `lms` CLI is used for model operations; it ships at `~/.lmstudio/bin/lms` and does not need to be on your `PATH`.

## How it refreshes

The dashboard polls two endpoints:

- **`/api/live`** — GPU, host CPU/RAM, and PCIe throughput. About 1 KB, polled 10×/second.
- **`/api/state`** — everything else: model lists, request panels, service info, settings. Polled every 2 seconds.

Neither endpoint does any work on the request path. Every source is sampled by a background thread on its own cadence and served from memory, so both respond in well under a millisecond. GPU samples come from a single long-lived `nvidia-smi --loop-ms` process rather than one spawn per request.

This matters because shelling out per request made fast refresh impossible: each `lms` invocation costs roughly 200 ms of Node startup, and a single aggregate call spent ~840 ms of its ~935 ms in subprocess spawns — capping the whole dashboard at about 1.4 Hz regardless of the browser's poll interval. Sampling in the background also stops the dashboard from flooding LM Studio's own logs with the polling traffic it is trying to report on.

Sustained 10 Hz polling costs about 2.6% of one core. The live loop pauses while the browser tab is hidden.

## What the request panel does and doesn't show

Requests are read from LM Studio's own server logs under `~/.lmstudio/server-logs/`. Those logs record the timestamp, HTTP method, and path of each request, plus per-model inference events — completions started, predictions generated, tool calls emitted, streams finished.

They do **not** record HTTP status codes, latency, or client IP addresses. So there are no p50/p95 latency figures, no error rate, and no per-client breakdown. Those panels are absent rather than showing zeros. If you are comparing against an Ollama-based dashboard, this is the one real capability loss in the migration — Ollama's GIN access logs carried all three.

## Model deletion

LM Studio exposes no delete API and `lms` has no remove command, so deleting a model removes its files from disk directly. To make that safe, the dashboard:

- resolves paths from LM Studio's own model index rather than reconstructing them,
- distinguishes the three storage classes (`user` weights, `hub` virtual pointers, and `bundled` models that ship with LM Studio and are never deletable),
- removes both the concrete weights and the hub stub when you delete a virtual model,
- refuses any target that resolves outside the models or hub roots,
- refuses to delete a loaded model, and
- requires the model key typed back verbatim before the confirm button enables.

## Run as a service (systemd)

For a persistent install on Linux, run it as a user systemd unit. This survives logouts (`loginctl enable-linger $USER`) and restarts on failure.

Create `~/.config/systemd/user/lmstudio-dashboard.service`:

```ini
[Unit]
Description=LM Studio Dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/lmstudio-dashboard
Environment=LMSTUDIO_DASHBOARD_HOST=0.0.0.0
ExecStart=/usr/bin/python3 %h/lmstudio-dashboard/server.py
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
```

Adjust `WorkingDirectory` / `ExecStart` to wherever you cloned the repo, then:

```bash
loginctl enable-linger $USER          # one-time, so it runs without you logged in
systemctl --user daemon-reload
systemctl --user enable --now lmstudio-dashboard
```

Common operations:

```bash
systemctl --user status lmstudio-dashboard
systemctl --user restart lmstudio-dashboard
journalctl --user -u lmstudio-dashboard -f
```

To change config, edit the `Environment=` lines in the unit file, then `systemctl --user daemon-reload && systemctl --user restart lmstudio-dashboard`.

## Configuration

All settings are environment variables with sensible defaults:

| Variable | Default | Notes |
|---|---|---|
| `LMSTUDIO_DASHBOARD_HOST` | `127.0.0.1` | Set to `0.0.0.0` to expose on the LAN |
| `LMSTUDIO_DASHBOARD_PORT` | `11435` | |
| `LMSTUDIO_URL` | `http://localhost:1234` | Upstream LM Studio server |
| `LMSTUDIO_LMS_BIN` | `~/.lmstudio/bin/lms` | The `lms` CLI; not on `PATH` by default |
| `LMSTUDIO_SYSTEMD_UNIT` | `lmstudio-server` | LM Studio's own unit, not this dashboard's |
| `LMSTUDIO_SYSTEMD_USER` | `1` | Use `systemctl --user`; set `0` for a system-level install |
| `LMSTUDIO_LOG_DIR` | `~/.lmstudio/server-logs` | Source for the request panels |
| `LMSTUDIO_SETTINGS_PATH` | `~/.lmstudio/settings.json` | Read-only, whitelisted keys shown on the control panel |
| `LMSTUDIO_MODEL_INDEX` | `~/.lmstudio/.internal/model-index-cache.json` | Authoritative model→path index; required for delete |
| `LMSTUDIO_HUB_MODELS_DIR` | `~/.lmstudio/hub/models` | Virtual-model stubs |
| `LMSTUDIO_MODELS_DIR` | `~/.lmstudio/models` | Fallback only; normally read from `downloadsFolder` in settings |
| `LMSTUDIO_CATALOG_URL` | `https://lmstudio.ai/models` | |
| `LMSTUDIO_CATALOG_TTL` | `3600` | Seconds to cache the scraped catalog |
| `LMSTUDIO_CATALOG_UA` | `lmstudio-dashboard/1.0` | |
| `LMSTUDIO_HAYSTACK_PATH` | `/tmp/moby.txt` | Corpus for the long-context benchmark |
| `LMSTUDIO_HAYSTACK_WORDS` | `21000` | |
| `LMSTUDIO_GPU_HISTORY_LEN` | `60` | Sparkline buffer length |
| `LMSTUDIO_PCIE_HISTORY_LEN` | `60` | |
| `LMSTUDIO_LOG_WINDOW_LINES` | `600` | Max parsed *events* retained (not raw lines) |
| `LMSTUDIO_LOG_TAIL_BYTES` | `4194304` | How much of each log file's tail to read per poll |
| `LMSTUDIO_STATS_WINDOW_SEC` | `300` | Stats aggregation window |
| `LMSTUDIO_GPU_SAMPLE_MS` | `100` | GPU stream cadence (`nvidia-smi --loop-ms`) |
| `LMSTUDIO_HOST_SAMPLE_MS` | `100` | Host CPU/RAM sampling |
| `LMSTUDIO_LOGS_SAMPLE_SEC` | `1` | Server-log re-read cadence |
| `LMSTUDIO_LOADED_SAMPLE_SEC` | `2` | `lms ps` cadence |
| `LMSTUDIO_SLOW_SAMPLE_SEC` | `15` | `lms ls`, engine, disk, service, tailscale |
| `LMSTUDIO_HISTORY_INTERVAL_SEC` | `1` | Sparkline sample spacing, throttled apart from the GPU sample rate |

`LMSTUDIO_LOG_TAIL_BYTES` exists because LM Studio logs full request bodies at DEBUG level, so raw log lines vastly outnumber actual events — a few hundred lines can span only a few seconds. The dashboard reads log tails by byte count and stops once the parsed events actually reach back past the stats window. Raise it only if your window looks truncated under heavy traffic.

### Setting variables

**One-off, command line:**

```bash
LMSTUDIO_DASHBOARD_HOST=0.0.0.0 LMSTUDIO_DASHBOARD_PORT=8080 python3 server.py
```

**Persistent, current shell session:**

```bash
export LMSTUDIO_URL=http://localhost:1234
python3 server.py
```

**Persistent, all shells** — append to `~/.bashrc` (or `~/.zshrc`):

```bash
export LMSTUDIO_DASHBOARD_HOST=0.0.0.0
export LMSTUDIO_URL=http://localhost:1234
```

**Under systemd** — add `Environment=` lines to the `[Service]` section of the unit file, one per variable:

```ini
[Service]
Environment=LMSTUDIO_DASHBOARD_HOST=0.0.0.0
Environment=LMSTUDIO_DASHBOARD_PORT=8080
Environment=LMSTUDIO_URL=http://localhost:1234
Environment=LMSTUDIO_HAYSTACK_PATH=/srv/corpora/moby.txt
```

For many variables, point `EnvironmentFile=` at a `.env`-style file instead:

```ini
[Service]
EnvironmentFile=%h/lmstudio-dashboard.env
```

```
# ~/lmstudio-dashboard.env
LMSTUDIO_DASHBOARD_HOST=0.0.0.0
LMSTUDIO_URL=http://localhost:1234
```

After any unit-file change: `systemctl --user daemon-reload && systemctl --user restart lmstudio-dashboard`.

## Requirements

- Python 3.9+
- LM Studio with its server running, and the `lms` CLI available
- Linux + systemd for service info (degrades gracefully elsewhere)
- `nvidia-smi` for GPU panel and PCIe throughput (degrades gracefully without a GPU)
- Optional: a copy of *Moby-Dick* at `LMSTUDIO_HAYSTACK_PATH` for the long-context needle benchmark

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The suite runs against captured fixtures in `tests/fixtures/` and needs no running LM Studio instance.

## Layout

```
config.py          environment-driven settings
lmstudio.py        the lms CLI, the /api/v0 API, and payload normalization
logs.py            server-log discovery, parsing, and aggregation
samplers.py        background sampling; keeps the request path free of subprocesses
sources.py         read-only state (gpu, host, disk, service, tailscale)
control.py         load, unload, download, delete, scenarios, catalog scrape
server.py          HTTP routing + main
templates/
  index.html       dashboard UI
  control.html     control panel UI
tests/             unittest suite + captured fixtures
```

## License

MIT
