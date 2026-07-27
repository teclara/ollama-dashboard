"""Read-only state sources: models, GPU, logs, disk, service, tailscale, and the aggregate state() call."""
import json, os, re, subprocess, threading, time
from collections import deque
from datetime import datetime

import logs
import lmstudio
from config import (
    GPU_HISTORY_LEN, LMS_BIN, MODELS_DIR_FALLBACK, PCIE_HISTORY_LEN,
    SETTINGS_PATH, SYSTEMD_UNIT, SYSTEMD_USER,
)

START = time.time()

_THROTTLE_BITS = [
    (0x0000000000000001, "gpu_idle"),
    (0x0000000000000002, "applications_clocks"),
    (0x0000000000000004, "sw_power_cap"),
    (0x0000000000000008, "hw_slowdown"),
    (0x0000000000000010, "sync_boost"),
    (0x0000000000000020, "sw_thermal_slowdown"),
    (0x0000000000000040, "hw_thermal_slowdown"),
    (0x0000000000000080, "hw_power_brake"),
    (0x0000000000000100, "display_clock_setting"),
]


def _decode_throttle(hex_str):
    try: bits = int(hex_str, 16)
    except (ValueError, TypeError): return []
    return [name for mask, name in _THROTTLE_BITS if bits & mask and name != "gpu_idle"]


def gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,temperature.memory,fan.speed,power.draw,power.limit,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,clocks_event_reasons.active",
             "--format=csv,noheader,nounits"], text=True, timeout=2).strip()
        n, mu, mt, u, t, tm, fan, p, pl, lg, lgm, lw, lwm, throttle = [x.strip() for x in out.split(",")]

        def _maybe_int(s):
            try: return int(s)
            except ValueError: return None  # [N/A]

        return {"name": n, "mem_used": int(mu), "mem_total": int(mt), "util": int(u), "temp": int(t),
                "temp_mem": _maybe_int(tm),
                "fan": _maybe_int(fan),
                "power": float(p), "power_limit": float(pl),
                "pcie_gen": int(lg), "pcie_gen_max": int(lgm),
                "pcie_width": int(lw), "pcie_width_max": int(lwm),
                "throttle_reasons": _decode_throttle(throttle)}
    except Exception as e:
        return {"error": str(e)}


_NVIDIA_VERSIONS = {}


def nvidia_versions():
    if _NVIDIA_VERSIONS: return _NVIDIA_VERSIONS
    try:
        out = subprocess.check_output(["nvidia-smi", "--version"], text=True, timeout=2)
    except Exception:
        return {}
    for line in out.splitlines():
        m = re.match(r"^\s*(.+?)\s*:\s*(.+?)\s*$", line)
        if not m: continue
        k, v = m.group(1).lower(), m.group(2)
        if "driver version" in k: _NVIDIA_VERSIONS["driver"] = v
        elif "cuda version" in k: _NVIDIA_VERSIONS["cuda"] = v
    return _NVIDIA_VERSIONS


def gpu_processes():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], text=True, timeout=2).strip()
    except Exception:
        return []
    procs = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",", 2)]
        if len(parts) != 3: continue
        try:
            procs.append({"pid": int(parts[0]), "name": parts[1], "vram_mb": int(parts[2])})
        except ValueError:
            continue
    return sorted(procs, key=lambda p: -p["vram_mb"])


# Settings and on-disk model store ------------------------------------------

# Only these keys are ever exposed. settings.json also holds hfSearchToken and
# hfDownloadToken; nothing outside this list may reach a response body.
_SETTINGS_WHITELIST = ("downloadsFolder", "defaultContextLength",
                       "modelLoadingGuardrails", "enableLocalService", "useHFProxy")


def settings(path=None):
    path = path or SETTINGS_PATH
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        return {"path": path, "error": str(e)}
    out = {"path": path}
    out.update({k: raw[k] for k in _SETTINGS_WHITELIST if k in raw})
    return out


def models_root(settings_dict):
    return (settings_dict or {}).get("downloadsFolder") or MODELS_DIR_FALLBACK


def disk(root=None):
    root = root or models_root(settings())
    info = {"models_dir": None, "models_size": 0, "fs_used": 0, "fs_total": 0, "fs_free": 0}
    if not os.path.isdir(root): return info
    info["models_dir"] = root
    try:
        info["models_size"] = int(
            subprocess.check_output(["du", "-sb", root], text=True, timeout=10).split()[0])
    except Exception: pass
    try:
        st = os.statvfs(root)
        info["fs_total"] = st.f_blocks * st.f_frsize
        info["fs_free"] = st.f_bavail * st.f_frsize
        info["fs_used"] = info["fs_total"] - info["fs_free"]
    except Exception: pass
    return info


def _systemctl(*args):
    cmd = ["systemctl"] + (["--user"] if SYSTEMD_USER else []) + list(args)
    return subprocess.check_output(cmd, text=True, timeout=2)


def service_info():
    info = {"uptime_s": None, "pid": None, "rss_kb": None, "active": "unknown",
            "engine": {"name": None, "version": None}}
    try:
        out = _systemctl("show", SYSTEMD_UNIT,
                         "--property=ActiveState,MainPID,ActiveEnterTimestampMonotonic")
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
    info["engine"] = lmstudio.engine_info()
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


# Host CPU/RAM ----------------------------------------------------------------

_CPU_LAST = {"idle": 0, "total": 0}
_CPU_LOCK = threading.Lock()


def _read_cpu_totals():
    with open("/proc/stat") as f:
        parts = f.readline().split()  # cpu user nice system idle iowait irq softirq steal ...
    nums = [int(x) for x in parts[1:]]
    idle = nums[3] + (nums[4] if len(nums) > 4 else 0)  # idle + iowait
    return idle, sum(nums)


def host():
    info = {"cpu_pct": None, "mem_used": 0, "mem_total": 0, "mem_pct": 0, "load_1": None, "ncpu": os.cpu_count() or 1}
    try:
        idle, total = _read_cpu_totals()
        with _CPU_LOCK:
            d_idle = idle - _CPU_LAST["idle"]
            d_total = total - _CPU_LAST["total"]
            _CPU_LAST["idle"], _CPU_LAST["total"] = idle, total
        if d_total > 0:
            info["cpu_pct"] = round(max(0.0, min(100.0, (1 - d_idle / d_total) * 100)), 1)
    except Exception: pass
    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                mem[k.strip()] = int(v.strip().split()[0]) * 1024  # kB → B
        total_b = mem.get("MemTotal", 0)
        avail_b = mem.get("MemAvailable", mem.get("MemFree", 0))
        used_b = max(0, total_b - avail_b)
        info["mem_total"] = total_b
        info["mem_used"] = used_b
        info["mem_pct"] = round(used_b / total_b * 100, 1) if total_b else 0
    except Exception: pass
    try:
        info["load_1"] = round(os.getloadavg()[0], 2)
    except Exception: pass
    return info


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


def state():
    g = gpu()
    push_history(g)
    rows = logs.read_window()
    cfg = settings()
    return {
        "now": datetime.now().isoformat(timespec="seconds"),
        "dash_uptime_s": int(time.time() - START),
        "gpu": g,
        "gpu_processes": gpu_processes(),
        "gpu_versions": nvidia_versions(),
        "gpu_history": get_history(),
        "loaded": lmstudio.loaded_models(),
        "library": lmstudio.library(),
        "requests": rows[-30:][::-1],
        "stats_5m": logs.stats(rows),
        "top_endpoints": logs.top_endpoints(rows),
        "model_activity": logs.model_activity(rows),
        "disk": disk(models_root(cfg)),
        "service": service_info(),
        "tailscale": tailscale(),
        "pcie": pcie(),
        "host": host(),
        "settings": cfg,
        "lms_ok": os.access(LMS_BIN, os.X_OK),
    }
