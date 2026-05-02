"""Read-only state sources: GPU, logs, disk, service, tailscale, and the aggregate state() call."""
import json, os, re, subprocess, threading, time, urllib.request
from collections import defaultdict, deque
from datetime import datetime

from config import (
    GPU_HISTORY_LEN, LOG_WINDOW_LINES, MODEL_DIRS, NOISE_PATHS, OLLAMA_URL,
    PCIE_HISTORY_LEN, STATS_WINDOW_SEC, SYSTEMD_OVERRIDE_PATH, SYSTEMD_UNIT,
)

START = time.time()

GIN_RE = re.compile(
    r'\[GIN\]\s+(?P<ts>\S+\s+-\s+\S+)\s+\|\s+(?P<status>\d+)\s+\|\s+(?P<lat>\S+)\s+\|\s+(?P<ip>\S+)\s+\|\s+(?P<method>\S+)\s+"(?P<path>[^"]+)"'
)
LAT_RE = re.compile(r'^([\d.]+)\s*(µs|us|ms|s|m)$')


def parse_latency_ms(s):
    m = LAT_RE.match(s)
    if not m: return None
    n, u = float(m.group(1)), m.group(2)
    return n / 1000 if u in ("µs", "us") else n if u == "ms" else n * 1000 if u == "s" else n * 60000


def parse_log_ts(ts):  # "2026/05/01 - 11:08:32"
    try: return datetime.strptime(ts, "%Y/%m/%d - %H:%M:%S").timestamp()
    except Exception: return 0


def gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
             "--format=csv,noheader,nounits"], text=True, timeout=2).strip()
        n, mu, mt, u, t, p, pl, lg, lgm, lw, lwm = [x.strip() for x in out.split(",")]
        return {"name": n, "mem_used": int(mu), "mem_total": int(mt), "util": int(u), "temp": int(t),
                "power": float(p), "power_limit": float(pl),
                "pcie_gen": int(lg), "pcie_gen_max": int(lgm),
                "pcie_width": int(lw), "pcie_width_max": int(lwm)}
    except Exception as e:
        return {"error": str(e)}


def loaded_models():
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/ps", timeout=2) as r:
            return json.loads(r.read()).get("models", [])
    except Exception:
        return []


def all_models():
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as r:
            data = json.loads(r.read())
        return sorted(
            [{"name": m["name"], "size": m.get("size", 0), "modified_at": m.get("modified_at")}
             for m in data.get("models", [])],
            key=lambda x: x["name"],
        )
    except Exception:
        return []


def parse_logs(window_lines=None):
    window_lines = window_lines or LOG_WINDOW_LINES
    try:
        out = subprocess.check_output(
            ["journalctl", "-u", SYSTEMD_UNIT, "-n", str(window_lines), "--no-pager", "-o", "cat"],
            text=True, timeout=4,
        )
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        m = GIN_RE.search(line)
        if not m or m.group("path") in NOISE_PATHS: continue
        rows.append({
            "ts": m.group("ts"),
            "epoch": parse_log_ts(m.group("ts")),
            "status": int(m.group("status")),
            "latency": m.group("lat"),
            "lat_ms": parse_latency_ms(m.group("lat")),
            "ip": m.group("ip"),
            "method": m.group("method"),
            "path": m.group("path"),
        })
    return rows


def stats(rows, window_sec=None):
    window_sec = window_sec or STATS_WINDOW_SEC
    now = time.time()
    recent = [r for r in rows if r["epoch"] >= now - window_sec]
    if not recent:
        return {"window_sec": window_sec, "count": 0, "rps": 0, "avg_ms": None,
                "p50_ms": None, "p95_ms": None, "errors": 0, "error_rate": 0}
    lats = sorted([r["lat_ms"] for r in recent if r["lat_ms"] is not None])
    n = len(lats)
    p = lambda q: lats[min(n - 1, int(q * n))] if n else None
    errs = sum(1 for r in recent if r["status"] >= 400)
    return {
        "window_sec": window_sec,
        "count": len(recent),
        "rps": round(len(recent) / window_sec, 2),
        "avg_ms": round(sum(lats) / n, 1) if n else None,
        "p50_ms": round(p(0.5), 1) if n else None,
        "p95_ms": round(p(0.95), 1) if n else None,
        "errors": errs,
        "error_rate": round(errs / len(recent), 3),
    }


def top_clients(rows, window_sec=None, top=5):
    window_sec = window_sec or STATS_WINDOW_SEC
    now = time.time()
    recent = [r for r in rows if r["epoch"] >= now - window_sec]
    by_ip = defaultdict(lambda: {"count": 0, "last": 0, "errors": 0})
    for r in recent:
        b = by_ip[r["ip"]]
        b["count"] += 1
        b["last"] = max(b["last"], r["epoch"])
        if r["status"] >= 400: b["errors"] += 1
    return sorted(
        [{"ip": ip, **v, "ago_s": int(now - v["last"])} for ip, v in by_ip.items()],
        key=lambda x: -x["count"],
    )[:top]


def top_endpoints(rows, window_sec=None, top=8):
    window_sec = window_sec or STATS_WINDOW_SEC
    now = time.time()
    recent = [r for r in rows if r["epoch"] >= now - window_sec]
    by_path = defaultdict(lambda: {"count": 0, "lats": []})
    for r in recent:
        b = by_path[r["path"]]
        b["count"] += 1
        if r["lat_ms"] is not None: b["lats"].append(r["lat_ms"])
    out = []
    for path, b in by_path.items():
        lats = sorted(b["lats"])
        out.append({"path": path, "count": b["count"],
                    "p50_ms": round(lats[len(lats) // 2], 1) if lats else None})
    return sorted(out, key=lambda x: -x["count"])[:top]


def disk():
    info = {"models_dir": None, "models_size": 0, "fs_used": 0, "fs_total": 0, "fs_free": 0}
    for d in MODEL_DIRS:
        if not os.path.isdir(d): continue
        info["models_dir"] = d
        try:
            out = subprocess.check_output(["du", "-sb", d], text=True, timeout=10).split()[0]
            info["models_size"] = int(out)
        except Exception: pass
        try:
            st = os.statvfs(d)
            info["fs_total"] = st.f_blocks * st.f_frsize
            info["fs_free"] = st.f_bavail * st.f_frsize
            info["fs_used"] = info["fs_total"] - info["fs_free"]
        except Exception: pass
        break
    return info


def service_info():
    info = {"uptime_s": None, "pid": None, "rss_kb": None, "active": "unknown", "version": None}
    try:
        out = subprocess.check_output(
            ["systemctl", "show", SYSTEMD_UNIT,
             "--property=ActiveState,MainPID,ActiveEnterTimestampMonotonic"],
            text=True, timeout=2,
        )
        kv = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
        info["active"] = kv.get("ActiveState", "unknown")
        pid = int(kv.get("MainPID", "0") or 0)
        info["pid"] = pid or None
        if pid:
            try:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            info["rss_kb"] = int(line.split()[1])
                            break
                with open(f"/proc/{pid}/stat") as f:
                    starttime = int(f.read().split()[21])
                clk = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
                with open("/proc/uptime") as f:
                    uptime = float(f.read().split()[0])
                info["uptime_s"] = int(uptime - starttime / clk)
            except Exception: pass
    except Exception: pass
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=1) as r:
            info["version"] = json.loads(r.read()).get("version")
    except Exception: pass
    return info


def tailscale():
    try:
        out = subprocess.check_output(["tailscale", "status", "--json"], text=True, timeout=2)
        d = json.loads(out)
        self_node = d.get("Self", {})
        peers = d.get("Peer", {}) or {}
        ips = self_node.get("TailscaleIPs", []) or []
        return {
            "up": d.get("BackendState") == "Running",
            "hostname": self_node.get("HostName"),
            "dnsname": self_node.get("DNSName", "").rstrip("."),
            "ip": ips[0] if ips else None,
            "peers_online": sum(1 for p in peers.values() if p.get("Online")),
            "peers_total": len(peers),
            "tailnet": d.get("CurrentTailnet", {}).get("Name"),
        }
    except Exception as e:
        return {"up": False, "error": str(e)}


# GPU sample history --------------------------------------------------------

_HIST = deque(maxlen=GPU_HISTORY_LEN)
_HIST_LOCK = threading.Lock()


def push_history(g):
    if "error" in g: return
    with _HIST_LOCK:
        _HIST.append({
            "t": int(time.time()),
            "vram_pct": round(g["mem_used"] / g["mem_total"] * 100, 1),
            "util": g["util"],
            "temp": g["temp"],
        })


def get_history():
    with _HIST_LOCK:
        return list(_HIST)


# PCIe throughput from a continuous nvidia-smi dmon stream ------------------

_PCIE_LATEST = {"rx_mbs": 0, "tx_mbs": 0, "ts": 0}
_PCIE_HIST = deque(maxlen=PCIE_HISTORY_LEN)
_PCIE_LOCK = threading.Lock()


def _pcie_dmon_loop():
    while True:
        try:
            proc = subprocess.Popen(
                ["nvidia-smi", "dmon", "-s", "t", "-d", "1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
            for line in proc.stdout:
                line = line.strip()
                if not line or line.startswith("#"): continue
                parts = line.split()
                if len(parts) < 3: continue
                try:
                    rx, tx = int(parts[1]), int(parts[2])
                    with _PCIE_LOCK:
                        _PCIE_LATEST.update({"rx_mbs": rx, "tx_mbs": tx, "ts": int(time.time())})
                        _PCIE_HIST.append({"t": int(time.time()), "rx": rx, "tx": tx})
                except ValueError:
                    continue
            proc.wait()
        except Exception:
            time.sleep(2)
        time.sleep(1)


def start_pcie_monitor():
    threading.Thread(target=_pcie_dmon_loop, daemon=True).start()


def pcie():
    with _PCIE_LOCK:
        return {"latest": dict(_PCIE_LATEST), "history": list(_PCIE_HIST)}


def server_config():
    info = {"path": SYSTEMD_OVERRIDE_PATH, "env": {}, "raw": "", "readable": False}
    try:
        with open(SYSTEMD_OVERRIDE_PATH) as f:
            raw = f.read()
        info["raw"] = raw
        info["readable"] = True
        for line in raw.splitlines():
            m = re.match(r'^Environment="?([^=]+)=([^"]*)"?$', line.strip())
            if m: info["env"][m.group(1)] = m.group(2)
    except Exception as e:
        info["error"] = str(e)
    return info


def state():
    g = gpu()
    push_history(g)
    rows = parse_logs()
    return {
        "now": datetime.now().isoformat(timespec="seconds"),
        "dash_uptime_s": int(time.time() - START),
        "gpu": g,
        "gpu_history": get_history(),
        "loaded": loaded_models(),
        "library": all_models(),
        "requests": rows[-30:][::-1],
        "stats_5m": stats(rows),
        "top_clients": top_clients(rows),
        "top_endpoints": top_endpoints(rows),
        "disk": disk(),
        "service": service_info(),
        "tailscale": tailscale(),
        "pcie": pcie(),
    }
