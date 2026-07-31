"""Read-only state sources: models, GPU, logs, disk, service, tailscale, and the aggregate state() call."""
import json, os, re, subprocess, threading, time
from collections import deque
from datetime import datetime

import logs
import ollama
from config import (
    GPU_HISTORY_LEN, HISTORY_INTERVAL_SEC, MODELS_DIR_FALLBACK,
    PCIE_HISTORY_LEN, SYSTEMD_UNIT, SYSTEMD_USER,
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


GPU_QUERY_FIELDS = (
    "name,memory.used,memory.total,utilization.gpu,temperature.gpu,"
    "temperature.memory,fan.speed,power.draw,power.limit,"
    "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,"
    "pcie.link.width.max,clocks_event_reasons.active"
)


def parse_gpu_csv(line):
    """One `--query-gpu` CSV row -> the GPU dict. Shared by the one-shot call
    and the streaming sampler, so both produce an identical shape."""
    n, mu, mt, u, t, tm, fan, p, pl, lg, lgm, lw, lwm, throttle = [
        x.strip() for x in line.split(",")]

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


def gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY_FIELDS}",
             "--format=csv,noheader,nounits"], text=True, timeout=2).strip()
        return parse_gpu_csv(out)
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
#
# Ollama has no settings file. Its configuration is the systemd unit's
# environment, which is also where this machine's real tuning lives
# (KV_CACHE_TYPE, FLASH_ATTENTION, MAX_LOADED_MODELS).

# A denylist, not a whitelist. Under LM Studio we could enumerate the settings
# worth exposing; here the user may add any OLLAMA_* variable at any time, so
# anything secret-shaped is redacted and everything else passes through.
_SECRET_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)

_ENV_TOKEN_RE = re.compile(r'"([^"]*)"|(\S+)')


def parse_systemd_environment(text):
    """`systemctl show <unit> --property=Environment` -> {name: value}.

    systemd emits one space-separated line and quotes only the values that
    need it, e.g. `OLLAMA_HOST=http://0.0.0.0:11434 "OLLAMA_ORIGINS=*"`.
    """
    out = {}
    for line in (text or "").splitlines():
        if not line.startswith("Environment="):
            continue
        for quoted, bare in _ENV_TOKEN_RE.findall(line[len("Environment="):]):
            token = quoted or bare
            name, sep, value = token.partition("=")
            if not sep or not name.startswith("OLLAMA_"):
                continue
            out[name] = "<redacted>" if _SECRET_RE.search(name) else value
    return out


def settings():
    try:
        raw = _systemctl("show", SYSTEMD_UNIT, "--property=Environment")
    except Exception as e:
        return {"unit": SYSTEMD_UNIT, "error": str(e)}
    env = parse_systemd_environment(raw)
    return {"unit": SYSTEMD_UNIT, **env}


def models_root(settings_dict):
    return (settings_dict or {}).get("OLLAMA_MODELS") or MODELS_DIR_FALLBACK


def _statvfs_walk_up(root):
    """statvfs on `root`, walking toward / past unreadable ancestors.

    /usr/share/ollama is 0750 ollama:ollama, so a dashboard process outside
    that group cannot stat the model dir. Any ancestor on the same mount
    reports identical filesystem totals.
    """
    path = os.path.abspath(root)
    while True:
        try:
            return os.statvfs(path)
        except OSError:
            parent = os.path.dirname(path)
            if parent == path:
                return None
            path = parent


def _du(root):
    """Exact bytes via `du -sb`, or None when not permitted."""
    try:
        return int(subprocess.check_output(
            ["du", "-sb", root], text=True, timeout=30,
            stderr=subprocess.DEVNULL).split()[0])
    except Exception:
        return None


def _readable_dir(path):
    """(exists, readable). os.path.isdir is False for BOTH a missing directory
    and one we lack traverse permission on, and those need opposite handling:
    missing means "nothing there", unreadable means "67 GB we cannot see".
    Reporting the second as an exact zero would be a confident lie."""
    if os.path.isdir(path):
        return True, True
    parent = os.path.dirname(path.rstrip(os.sep))
    while parent and parent != os.sep:
        if os.path.exists(parent):
            # A parent resolves but the target does not: either genuinely
            # absent, or hidden behind a mode we cannot traverse.
            return not os.access(parent, os.X_OK), False
        parent = os.path.dirname(parent)
    return False, False


def disk(root=None):
    root = root or models_root(settings())
    info = {"models_dir": None, "models_size": 0, "approximate": False,
            "orphan_bytes": None, "fs_used": 0, "fs_total": 0, "fs_free": 0}
    exists, readable = _readable_dir(root)
    if not exists:
        return info
    info["models_dir"] = root
    if not readable:
        # Permission-denied: fall through to the /api/tags sum below, which
        # counts only referenced blobs and therefore understates the truth.
        info["approximate"] = True
        info["models_size"] = sum(m.get("size") or 0 for m in ollama.library())
        st = _statvfs_walk_up(root)
        if st is not None:
            info["fs_total"] = st.f_blocks * st.f_frsize
            info["fs_free"] = st.f_bavail * st.f_frsize
            info["fs_used"] = info["fs_total"] - info["fs_free"]
        return info

    referenced = sum(m.get("size") or 0 for m in ollama.library())
    exact = _du(root)
    if exact is None:
        # No read access to the store. Summing /api/tags counts only
        # referenced blobs, so this understates the total — flag it.
        info["models_size"] = referenced
        info["approximate"] = True
    else:
        info["models_size"] = exact
        # Everything du sees that no manifest references: reclaimable.
        info["orphan_bytes"] = max(0, exact - referenced)

    st = _statvfs_walk_up(root)
    if st is not None:
        info["fs_total"] = st.f_blocks * st.f_frsize
        info["fs_free"] = st.f_bavail * st.f_frsize
        info["fs_used"] = info["fs_total"] - info["fs_free"]
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
    info["engine"] = {"name": "ollama", "version": ollama.version().get("version")}
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


_LAST_HIST_PUSH = [0.0]


def push_history(g, now=None, min_interval=None):
    """Append a sparkline sample, throttled independently of the sample rate.

    GPU samples arrive ~10x/second; without throttling a 60-slot buffer would
    cover six seconds instead of a minute.
    """
    if "error" in g: return False
    now = time.time() if now is None else now
    interval = HISTORY_INTERVAL_SEC if min_interval is None else min_interval
    with _HIST_LOCK:
        if now - _LAST_HIST_PUSH[0] < interval: return False
        _LAST_HIST_PUSH[0] = now
        _HIST.append({
            "t": int(now),
            "vram_pct": round(g["mem_used"] / g["mem_total"] * 100, 1),
            "util": g["util"],
            "temp": g["temp"],
        })
        return True


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


def live():
    """The small, fast-moving slice: everything that visibly moves.

    Served entirely from the sampler cache, so this is safe to poll many times
    a second. Kept deliberately small — shipping the model lists at that rate
    would be pure waste, since they change on the order of minutes.
    """
    import samplers
    return {
        "now": datetime.now().isoformat(timespec="seconds"),
        "dash_uptime_s": int(time.time() - START),
        "gpu": samplers.GPU.get(),
        "gpu_history": get_history(),
        "pcie": pcie(),
        "host": samplers.HOST.get(),
    }


def state():
    """The full payload. Also served from the sampler cache."""
    import samplers
    rows = samplers.attribute(samplers.LOGS.get())
    return {
        **live(),
        "gpu_processes": samplers.GPU_PROCS.get(),
        "gpu_versions": nvidia_versions(),
        "loaded": samplers.LOADED.get(),
        "library": samplers.LIBRARY.get(),
        "requests": rows[-30:][::-1],
        "stats_5m": logs.stats(rows),
        "top_endpoints": logs.top_endpoints(rows),
        "by_client": logs.by_client(rows),
        "problems": logs.problems(rows),
        "log_age_s": logs.follower_age(),
        "disk": samplers.DISK.get(),
        "service": samplers.SERVICE.get(),
        "tailscale": samplers.TAILSCALE.get(),
        "settings": samplers.SETTINGS.get(),
        # Sampled, never called live — state() must not touch Ollama.
        "ollama_ok": samplers.PING.get(),
    }
