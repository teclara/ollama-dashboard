# Ollama Dashboard

A local dashboard for [Ollama](https://ollama.com). Live view of loaded models, library, pulls, and request stats, plus a control panel for loading/unloading models, pulling from `ollama.com/library`, and running quick benchmarks.

No dependencies — Python 3 standard library only.

## Run

```bash
python3 server.py
```

Defaults to `http://127.0.0.1:11435` and expects Ollama on `http://localhost:11434`.

- Dashboard: `http://localhost:11435/`
- Control panel: `http://localhost:11435/control`

## Run as a service (systemd)

For a persistent install on Linux, run it as a user systemd unit. This survives logouts (`loginctl enable-linger $USER`) and restarts on failure.

Create `~/.config/systemd/user/ollama-dashboard.service`:

```ini
[Unit]
Description=Ollama Dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/ollama-dashboard
Environment=OLLAMA_DASHBOARD_HOST=0.0.0.0
ExecStart=/usr/bin/python3 %h/ollama-dashboard/server.py
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
```

Adjust `WorkingDirectory` / `ExecStart` to wherever you cloned the repo, then:

```bash
loginctl enable-linger $USER          # one-time, so it runs without you logged in
systemctl --user daemon-reload
systemctl --user enable --now ollama-dashboard
```

Common operations:

```bash
systemctl --user status ollama-dashboard
systemctl --user restart ollama-dashboard
journalctl --user -u ollama-dashboard -f
```

To change config, edit the `Environment=` lines in the unit file, then `systemctl --user daemon-reload && systemctl --user restart ollama-dashboard`.

## Configuration

All settings are environment variables with sensible defaults:

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_DASHBOARD_HOST` | `127.0.0.1` | Set to `0.0.0.0` to expose on the LAN |
| `OLLAMA_DASHBOARD_PORT` | `11435` | |
| `OLLAMA_URL` | `http://localhost:11434` | Upstream Ollama server |
| `OLLAMA_LIBRARY_URL` | `https://ollama.com/library` | |
| `OLLAMA_LIBRARY_TTL` | `3600` | Seconds to cache the scraped library |
| `OLLAMA_LIBRARY_UA` | `ollama-dashboard/1.0` | |
| `OLLAMA_HAYSTACK_PATH` | `/tmp/moby.txt` | Corpus for the long-context benchmark |
| `OLLAMA_HAYSTACK_WORDS` | `21000` | |
| `OLLAMA_SYSTEMD_UNIT` | `ollama` | systemd unit name |
| `OLLAMA_SYSTEMD_OVERRIDE` | `/etc/systemd/system/ollama.service.d/override.conf` | Read-only, shown on the control panel |
| `OLLAMA_MODEL_DIRS` | `/usr/share/ollama/.ollama/models:~/.ollama/models` | Colon-separated; first existing wins |
| `OLLAMA_GPU_HISTORY_LEN` | `60` | Sparkline buffer length |
| `OLLAMA_PCIE_HISTORY_LEN` | `60` | |
| `OLLAMA_LOG_WINDOW_LINES` | `600` | journalctl window for request stats |
| `OLLAMA_STATS_WINDOW_SEC` | `300` | Stats aggregation window |

### Setting variables

**One-off, command line:**

```bash
OLLAMA_DASHBOARD_HOST=0.0.0.0 OLLAMA_DASHBOARD_PORT=8080 python3 server.py
```

**Persistent, current shell session:**

```bash
export OLLAMA_URL=http://localhost:11434
python3 server.py
```

**Persistent, all shells** — append to `~/.bashrc` (or `~/.zshrc`):

```bash
export OLLAMA_DASHBOARD_HOST=0.0.0.0
export OLLAMA_URL=http://localhost:11434
```

**Under systemd** — add `Environment=` lines to the `[Service]` section of the unit file, one per variable:

```ini
[Service]
Environment=OLLAMA_DASHBOARD_HOST=0.0.0.0
Environment=OLLAMA_DASHBOARD_PORT=8080
Environment=OLLAMA_URL=http://localhost:11434
Environment=OLLAMA_HAYSTACK_PATH=/srv/corpora/moby.txt
```

For many variables, point `EnvironmentFile=` at a `.env`-style file instead:

```ini
[Service]
EnvironmentFile=%h/ollama-dashboard.env
```

```
# ~/ollama-dashboard.env
OLLAMA_DASHBOARD_HOST=0.0.0.0
OLLAMA_URL=http://localhost:11434
```

After any unit-file change: `systemctl --user daemon-reload && systemctl --user restart ollama-dashboard`.

## Requirements

- Python 3.9+
- Ollama running locally on port 11434
- Linux + systemd for journal-based request stats and service info (degrades gracefully elsewhere)
- `nvidia-smi` for GPU panel and PCIe throughput (degrades gracefully without a GPU)
- Optional: a copy of *Moby-Dick* at `OLLAMA_HAYSTACK_PATH` for the long-context needle benchmark

## Layout

```
config.py          environment-driven settings
sources.py         read-only state (gpu, logs, disk, service, tailscale)
control.py         pulls, deletes, scenarios, library scrape
server.py          HTTP routing + main
templates/
  index.html       dashboard UI
  control.html     control panel UI
```

## License

MIT
