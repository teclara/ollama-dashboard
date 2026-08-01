/* Ollama Dashboard — client runtime. No framework, no build, no dependencies.
 *
 * Three polling streams, all served from the server's sampler cache so none of
 * them costs a subprocess or touches Ollama:
 *
 *   /api/live          fast slice: GPU, host, PCIe
 *   /api/state         model lists, request evidence, service, disk, settings
 *   /api/control/jobs  loads, downloads, benchmark progress
 *
 * The rail is the reason this file is structured the way it is. It is read from
 * across a room on an always-on monitor, so its DOM is built once by the HTML
 * and this file only ever writes text-node values and bar widths into it.
 * Nothing in the rail is created, removed, or reordered by a poll. See RAIL.
 */
(function(){
'use strict';

/* ------------------------------------------------------------------ utils */

const $ = id => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const EMDASH = '—';

function ce(tag, opts, kids){
  const e = document.createElement(tag);
  if(opts){
    if(opts.cls) e.className = opts.cls;
    if(opts.text != null) e.textContent = opts.text;
    if(opts.attrs) for(const k in opts.attrs) e.setAttribute(k, opts.attrs[k]);
    if(opts.on) for(const k in opts.on) e.addEventListener(k, opts.on[k]);
  }
  if(kids) for(const c of kids){ if(c) e.appendChild(c); }
  return e;
}
function svg(tag, attrs){
  const e = document.createElementNS(SVG, tag);
  for(const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
function put(host, kids){
  if(!host) return;
  host.replaceChildren.apply(host, Array.isArray(kids) ? kids.filter(Boolean) : (kids ? [kids] : []));
}
function setText(node, text){ if(node && node.textContent !== text) node.textContent = text; }

function fmtBytes(b){
  if(b == null || !isFinite(b) || b <= 0) return EMDASH;
  const u = ['B','KB','MB','GB','TB'];
  let i = 0;
  while(b >= 1024 && i < u.length - 1){ b /= 1024; i++; }
  return (i === 0 ? b.toFixed(0) : b.toFixed(1)) + ' ' + u[i];
}
// GIN latencies arrive in seconds and span microseconds to minutes.
function fmtDur(s){
  if(s == null) return EMDASH;
  if(s < 1e-3) return (s * 1e6).toFixed(0) + 'µs';
  if(s < 1) return (s * 1e3).toFixed(1) + 'ms';
  if(s < 60) return s.toFixed(2) + 's';
  return Math.floor(s / 60) + 'm' + Math.round(s % 60) + 's';
}
function fmtSpan(s){
  if(s == null) return EMDASH;
  s = Math.round(s);
  const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60);
  if(d) return d + 'd ' + h + 'h';
  if(h) return h + 'h ' + m + 'm';
  if(m) return m + 'm ' + (s % 60) + 's';
  return s + 's';
}
function fmtRate(bps){
  if(!bps || bps < 1) return EMDASH;
  if(bps < 1024) return bps.toFixed(0) + ' B/s';
  if(bps < 1024 * 1024) return (bps / 1024).toFixed(0) + ' KB/s';
  return (bps / (1024 * 1024)).toFixed(1) + ' MB/s';
}
function fmtCtx(n){ return n ? (n / 1024).toFixed(0) + 'K' : EMDASH; }
function plural(n, one, many){ return n + ' ' + (n === 1 ? one : many); }

function pill(text, tone){ return ce('span', {cls: 'pill' + (tone ? ' ' + tone : ''), text}); }
function capPills(m){
  const out = [];
  for(const [flag, label] of [['vision','vision'],['tools','tools'],['thinking','thinking'],['embedding','embed']]){
    if(m[flag]) out.push(pill(label, 'quiet'));
  }
  return out.length ? ce('span', {cls:'pills'}, out) : document.createTextNode(EMDASH);
}

/* --------------------------------------------------------------- api ---- */

async function api(method, path, body, signal){
  const r = await fetch(path, {
    method, signal,
    headers: body ? {'Content-Type':'application/json'} : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await r.text();
  let data = null;
  try { data = JSON.parse(text); } catch(e){ /* non-JSON error body */ }
  if(!r.ok) throw new Error(data && data.error ? data.error : method + ' ' + path + ' ' + r.status);
  return data;
}

/* One timer, one in-flight request and one generation counter per stream. That
   prevents overlap, and lets a response that was invalidated while the tab was
   hidden be dropped instead of rendered stale. */
function pollStream(name, fetcher, render, intervalMs){
  let timer = null, request = null, generation = 0, pending = false;
  function schedule(delay){
    if(document.hidden || timer !== null || request) return;
    timer = setTimeout(run, delay);
  }
  async function run(){
    timer = null;
    if(document.hidden) return;
    if(request){ pending = true; return; }
    const current = ++generation;
    const controller = new AbortController();
    request = controller;
    try {
      const data = await fetcher(controller.signal);
      if(current !== generation) return;
      render(data);
      setPollError(name, null);
    } catch(e){
      if(current === generation && e.name !== 'AbortError') setPollError(name, e.message);
    } finally {
      if(request === controller) request = null;
      if(current !== generation){
        if(pending && !document.hidden){ pending = false; schedule(0); }
        return;
      }
      const delay = pending ? 0 : intervalMs;
      pending = false;
      schedule(delay);
    }
  }
  return {
    start(){ schedule(0); },
    refresh(){
      if(document.hidden) return;
      if(request){ pending = true; generation++; request.abort(); }
      else { if(timer !== null){ clearTimeout(timer); timer = null; } schedule(0); }
    },
    pause(){
      pending = false; generation++;
      if(timer !== null){ clearTimeout(timer); timer = null; }
      if(request) request.abort();
    },
    resume(){ if(request) pending = true; else schedule(0); },
  };
}

/* ------------------------------------------------------------ app state -- */

let STATE = null;      // last /api/state payload
let LIVE = null;       // last /api/live payload
let JOBS = {};         // last /api/control/jobs payload
let CATALOG = [], CATALOG_ERROR = null;
let ollamaAvailable = true;
let benchInitialised = false;
let peakRps = 1;       // scale for the rail's requests gauge, stated in its detail line
const POLL_ERRORS = {live:null, state:null, jobs:null};
const BENCH_KEY = 'benchmark:suite';

/* --------------------------------------------------------------- alerts -- */

function setPollError(name, message){
  POLL_ERRORS[name] = message;
  renderAlerts();
}

/* One composed alert region. Every entry here exists so a broken input is never
   silently rendered as real data. */
function renderAlerts(){
  const items = [];
  if(STATE && STATE.ollama_ok === false){
    items.push(['Ollama is unreachable.',
      'Model data is unavailable and operational controls are disabled. Check OLLAMA_URL and the state of ollama.service.']);
  }
  // Keyed on the subprocess, not on silence: a healthy follower watching an
  // idle server reads nothing for minutes, which is not a fault.
  if(STATE && STATE.log_follower_ok === false){
    items.push(['The journal follower is not running.',
      'Request statistics, client attribution and problem lines have stopped updating.']);
  }
  const failed = Object.keys(POLL_ERRORS).filter(k => POLL_ERRORS[k]);
  if(failed.length){
    items.push(['The dashboard cannot reach its own server.',
      failed.map(k => k + ': ' + POLL_ERRORS[k]).join(' · ')]);
  }
  put($('alerts'), items.map(([title, detail]) => ce('div', {cls:'alert bad'}, [
    ce('span', {cls:'alert-mark', text:'!', attrs:{'aria-hidden':'true'}}),
    ce('div', {}, [ce('p', {text:title}), ce('p', {text:detail})]),
  ])));
}

let toastTimer = null;
function toast(message, tone){
  tone = tone || 'bad';
  if(toastTimer){ clearTimeout(toastTimer); toastTimer = null; }
  put($('toast'), ce('div', {cls:'alert ' + tone}, [
    ce('span', {cls:'alert-mark', text: tone === 'good' ? '✓' : '!', attrs:{'aria-hidden':'true'}}),
    ce('p', {text:message}),
  ]));
  if(tone !== 'bad'){
    toastTimer = setTimeout(() => { put($('toast'), []); toastTimer = null; }, 6000);
  }
}

/* Controls that mutate Ollama are disabled while it is unreachable, and the
   bookkeeping flag makes sure we only re-enable what we ourselves disabled. */
function setOllamaAvailable(available){
  ollamaAvailable = available;
  for(const c of document.querySelectorAll('[data-needs-ollama]')){
    if(!available && !c.disabled){
      c.dataset.offDisabled = 'true';
      c.disabled = true;
    } else if(available && c.dataset.offDisabled){
      delete c.dataset.offDisabled;
      c.disabled = false;
    }
  }
}

/* ---------------------------------------------------------------- tables - */

function tbl(id, headers, rows){
  const head = ce('tr', {}, headers.map(h =>
    ce('th', {text: typeof h === 'string' ? h : h.label, attrs:{scope:'col'}})));
  return ce('table', {attrs:{id}}, [
    ce('thead', {}, [head]),
    ce('tbody', {}, rows),
  ]);
}
function td(content, cls){
  const cell = ce('td', {cls});
  if(content instanceof Node) cell.appendChild(content);
  else if(Array.isArray(content)) content.forEach(c => c && cell.appendChild(c));
  else cell.textContent = content == null ? EMDASH : String(content);
  return cell;
}
function tr(cells){ return ce('tr', {}, cells); }

function empty(title, detail, actions){
  return ce('div', {cls:'empty'}, [
    title ? ce('strong', {text:title}) : null,
    detail ? ce('span', {text:detail}) : null,
    actions && actions.length ? ce('div', {cls:'btn-row'}, actions) : null,
  ]);
}

function kv(rows){
  return rows.filter(Boolean).map(([k, v, flag]) => ce('div', {cls:'kv-row'}, [
    ce('span', {cls:'kv-k', text:k}),
    v instanceof Node ? ce('span', {cls:'kv-v'}, [v])
                      : ce('span', {cls:'kv-v' + (flag ? ' flag' : ''), text: v == null ? EMDASH : String(v)}),
  ]));
}

/* Wide operational tables stay complete rather than losing columns. Make the
   scroller keyboard-reachable only when it actually overflows, and show an
   edge cue until the last column is reached. */
function syncScroller(region){
  const over = region.scrollWidth > region.clientWidth + 1;
  region.dataset.overflow = String(over);
  region.dataset.end = String(!over || region.scrollLeft + region.clientWidth >= region.scrollWidth - 1);
  if(over){
    region.tabIndex = 0;
    if(!region.hasAttribute('aria-label')){
      const heading = region.closest('.panel');
      const h = heading && heading.querySelector('h2');
      region.setAttribute('aria-label', 'Scrollable ' + ((h && h.textContent.trim()) || 'table'));
      region.dataset.autoLabel = 'true';
    }
  } else {
    region.removeAttribute('tabindex');
    if(region.dataset.autoLabel){
      region.removeAttribute('aria-label');
      delete region.dataset.autoLabel;
    }
  }
}
function initScrollers(){
  const regions = Array.from(document.querySelectorAll('.tablewrap'));
  const sync = () => regions.forEach(syncScroller);
  for(const region of regions){
    region.addEventListener('scroll', () => syncScroller(region), {passive:true});
    new MutationObserver(() => requestAnimationFrame(() => syncScroller(region)))
      .observe(region, {childList:true, subtree:true});
  }
  if('ResizeObserver' in window){
    const ro = new ResizeObserver(sync);
    regions.forEach(r => ro.observe(r));
  }
  window.addEventListener('resize', sync, {passive:true});
  sync();
}

/* ----------------------------------------------------------------- theme - */

const THEME_KEY = 'ollama-dash-theme';
function themeCycle(){
  // auto -> light -> dark -> auto. Auto is the default because an always-on
  // monitor should follow the room unless it is told otherwise.
  const now = document.documentElement.dataset.theme || 'auto';
  const next = now === 'auto' ? 'light' : now === 'light' ? 'dark' : 'auto';
  if(next === 'auto'){
    delete document.documentElement.dataset.theme;
    try { localStorage.removeItem(THEME_KEY); } catch(e){}
  } else {
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem(THEME_KEY, next); } catch(e){}
  }
  syncThemeButton();
}
function syncThemeButton(){
  const mode = document.documentElement.dataset.theme || 'auto';
  const label = mode === 'auto' ? 'Theme: auto' : mode === 'light' ? 'Theme: light' : 'Theme: dark';
  const btn = $('btn-theme');
  setText(btn, label);
  btn.setAttribute('aria-label', label + '. Activate to change.');
}

/* ------------------------------------------------------------------ tabs - */

const VIEWS = ['models','activity','host','bench'];
// Old bookmarks from the two-page layout keep working.
const LEGACY = {overview:'models', performance:'bench', system:'host', settings:'host'};
let currentView = 'models';

function showView(name, pushHash){
  if(!VIEWS.includes(name)) name = 'models';
  currentView = name;
  for(const v of VIEWS){
    const tab = $('tab-' + v), panel = $('v-' + v), on = v === name;
    tab.setAttribute('aria-selected', String(on));
    tab.tabIndex = on ? 0 : -1;
    panel.hidden = !on;
  }
  document.title = name[0].toUpperCase() + name.slice(1) + ' · Ollama Operations';
  $('main').scrollTop = 0;
  if(pushHash && location.hash.slice(1) !== name) history.replaceState(null, '', '#' + name);
}
function viewFromHash(){
  const raw = location.hash.slice(1);
  return LEGACY[raw] || (VIEWS.includes(raw) ? raw : 'models');
}
function initTabs(){
  VIEWS.forEach((v, i) => {
    const tab = $('tab-' + v);
    tab.addEventListener('click', () => showView(v, true));
    tab.addEventListener('keydown', e => {
      let next = null;
      if(e.key === 'ArrowRight') next = VIEWS[(i + 1) % VIEWS.length];
      else if(e.key === 'ArrowLeft') next = VIEWS[(i - 1 + VIEWS.length) % VIEWS.length];
      else if(e.key === 'Home') next = VIEWS[0];
      else if(e.key === 'End') next = VIEWS[VIEWS.length - 1];
      if(next){ e.preventDefault(); showView(next, true); $('tab-' + next).focus(); }
    });
  });
  window.addEventListener('hashchange', () => showView(viewFromHash(), false));
  showView(viewFromHash(), false);
}

/* ====================================================================== RAIL
 *
 * Anti-strobe contract:
 *   - every node below already exists in index.html and is never replaced
 *   - only text-node values and bar widths are written
 *   - all numerics are fixed-width tabular figures, so a digit change cannot
 *     reflow a neighbour
 *   - continuous telemetry is smoothed before display, because a raw reading
 *     at poll rate is visual noise in peripheral vision
 *   - nothing that signals a fault is smoothed: service state, error counts and
 *     follower health are written exactly as they arrive
 */

const RAIL = {};
function bindRail(){
  for(const k of ['compute','memory','io','req','cpu','ram']){
    const host = $('m-' + k);
    RAIL[k] = {
      // firstChild is the value text node, ahead of the unit <span>.
      value: host.firstChild,
      gauge: $('g-' + k),
      detail: $('d-' + k) || null,
    };
  }
  RAIL.dot = $('verdict-dot');
  RAIL.word = $('verdict-word');
  RAIL.why = $('verdict-why');
  RAIL.residentName = $('resident-name');
  RAIL.residentMeta = $('resident-meta');
  RAIL.residentExtra = $('resident-extra');
  RAIL.job = $('rail-job');
  RAIL.jobName = $('rail-job-name');
  RAIL.jobBar = $('rail-job-bar');
  RAIL.jobLine = $('rail-job-line');
  RAIL.clock = $('clock');
}
function setVal(slot, text){ if(slot.value.nodeValue !== text) slot.value.nodeValue = text; }
function setBar(slot, pct){
  const w = (pct == null ? 0 : clamp(pct, 0, 100)).toFixed(1) + '%';
  if(slot.gauge.style.width !== w) slot.gauge.style.width = w;
}
function setDetail(slot, text){ if(slot.detail) setText(slot.detail, text); }
function setRole(node, role){
  const cls = 'r-' + role;
  if(node.dataset.role !== role){
    node.classList.remove('r-good','r-warn','r-bad','r-neutral');
    node.classList.add(cls);
    node.dataset.role = role;
  }
}

// Exponential smoothing, ~0.4s settling at the live poll rate. Telemetry only.
const SMOOTH = new Map();
function smooth(key, value){
  if(value == null || !isFinite(value)){ SMOOTH.delete(key); return null; }
  const prev = SMOOTH.get(key);
  const next = prev == null ? value : prev + 0.3 * (value - prev);
  SMOOTH.set(key, next);
  return next;
}

function renderRailLive(s){
  const g = s.gpu || {};
  if(g.error){
    setVal(RAIL.compute, EMDASH); setBar(RAIL.compute, 0);
    setDetail(RAIL.compute, g.error);
    setVal(RAIL.memory, EMDASH); setBar(RAIL.memory, 0);
    setDetail(RAIL.memory, 'GPU sample unavailable');
  } else {
    const util = smooth('util', g.util);
    setVal(RAIL.compute, util == null ? EMDASH : String(Math.round(util)));
    setBar(RAIL.compute, util);
    setDetail(RAIL.compute, [
      g.name || 'GPU',
      g.temp != null ? g.temp + '°C' : null,
      g.power != null ? Math.round(g.power) + 'W' : null,
    ].filter(Boolean).join(' · '));

    const vramPct = g.mem_total ? (g.mem_used / g.mem_total * 100) : null;
    const vram = smooth('vram', vramPct);
    setVal(RAIL.memory, vram == null ? EMDASH : String(Math.round(vram)));
    setBar(RAIL.memory, vram);
    setDetail(RAIL.memory, g.mem_total
      ? fmtBytes(g.mem_used * 1048576) + ' / ' + fmtBytes(g.mem_total * 1048576)
      : 'no VRAM reading');
  }

  // PCIe telemetry rides on the GPU sample, so do not invent a link when that
  // sample failed.
  const lat = (s.pcie && s.pcie.latest) || {};
  if(g.error || lat.rx_mbs == null){
    setVal(RAIL.io, EMDASH); setBar(RAIL.io, 0);
    setDetail(RAIL.io, 'no link reading');
  } else {
    const total = (lat.rx_mbs || 0) + (lat.tx_mbs || 0);
    const shown = smooth('io', total);
    const cap = linkCapMbs(g);
    setVal(RAIL.io, shown == null ? EMDASH : String(Math.round(shown)));
    setBar(RAIL.io, cap ? (total / cap * 100) : 0);
    setDetail(RAIL.io, 'rx ' + (lat.rx_mbs || 0) + ' · tx ' + (lat.tx_mbs || 0)
      + ' MB/s' + (cap ? ' · cap ' + Math.round(cap) + ' MB/s' : ''));
  }

  const h = s.host || {};
  const cpu = smooth('cpu', h.cpu_pct);
  setVal(RAIL.cpu, cpu == null ? EMDASH : String(Math.round(cpu)));
  setBar(RAIL.cpu, cpu);
  const ram = smooth('ram', h.mem_pct);
  setVal(RAIL.ram, ram == null ? EMDASH : String(Math.round(ram)));
  setBar(RAIL.ram, ram);

  setText(RAIL.clock, String(s.now || '').slice(11) || EMDASH);
}

function linkCapMbs(g){
  // Rough per-lane throughput by PCIe generation, in MB/s.
  const perLane = {1:250, 2:500, 3:985, 4:1969, 5:3938};
  return (perLane[g.pcie_gen] || 0) * (g.pcie_width || 0);
}

/* The verdict. Five states, each carrying its own reason, so colour is never
   the only thing saying what is going on. */
function verdictOf(s){
  if(!s) return {word:'Starting', why:'Waiting for the first sample.', role:'neutral'};
  if(s.ollama_ok === false){
    return {word:'Offline', role:'bad',
            why:'Ollama is not answering. Nothing can be served or loaded.'};
  }
  const faults = [];
  const svc = s.service || {};
  if(svc.active && svc.active !== 'active') faults.push('ollama.service is ' + svc.active);
  if(s.log_follower_ok === false) faults.push('the journal follower stopped');
  if((s.gpu || {}).error) faults.push('GPU telemetry is unavailable');
  const st = s.stats_5m || {};
  if(st.error_count && st.error_rate >= 5) faults.push(st.error_rate + '% of requests are failing');
  const brokenJobs = Object.keys(JOBS).filter(k => JOBS[k].error).length;
  if(brokenJobs) faults.push(plural(brokenJobs, 'job has failed', 'jobs have failed'));

  if(faults.length){
    return {word:'Degraded', role:'warn',
            why: faults[0][0].toUpperCase() + faults[0].slice(1) + '.'
                 + (faults.length > 1 ? ' ' + plural(faults.length - 1, 'other fault', 'other faults') + '.' : '')};
  }
  const loaded = s.loaded || [];
  if(!loaded.length){
    return {word:'Idle', role:'neutral',
            why:'No model is resident. Ollama is reachable and ready to load one.'};
  }
  if(st.count){
    return {word:'Serving', role:'good',
            why: plural(st.count, 'request', 'requests') + ' in the last 5 minutes, p95 ' + fmtDur(st.p95_s) + '.'};
  }
  return {word:'Ready', role:'good',
          why: plural(loaded.length, 'model is', 'models are') + ' resident. No requests in the last 5 minutes.'};
}

function renderRailState(s){
  const v = verdictOf(s);
  setText(RAIL.word, v.word);
  setText(RAIL.why, v.why);
  setRole(RAIL.dot, v.role);
  setRole(RAIL.word, v.role);

  const loaded = (s && s.loaded) || [];
  if(!loaded.length){
    setText(RAIL.residentName, 'Nothing resident');
    put(RAIL.residentMeta, []);
    setText(RAIL.residentExtra, s && s.ollama_ok === false ? 'Ollama is unreachable' : '');
  } else {
    const m = loaded[0];
    setText(RAIL.residentName, m.model_key || EMDASH);
    put(RAIL.residentMeta, [
      // size vs size_vram is the placement signal: when they differ, layers
      // spilled to the CPU. Colour is paired with text, never used alone.
      m.fully_gpu ? pill('GPU', 'good') : pill('CPU +' + fmtBytes(m.cpu_bytes), 'warn'),
      pill(fmtCtx(m.context) + ' ctx', 'quiet'),
      pill(m.ttl_s == null ? 'no ttl' : fmtSpan(m.ttl_s) + ' left', 'quiet'),
    ]);
    setText(RAIL.residentExtra, loaded.length > 1
      ? plural(loaded.length - 1, 'other model resident', 'other models resident')
      : (m.quant ? m.quant + ' · ' + fmtBytes(m.size) : fmtBytes(m.size)));
  }

  const st = (s && s.stats_5m) || {};
  if(st.rps != null && st.rps > peakRps) peakRps = st.rps;
  setVal(RAIL.req, st.rps == null ? EMDASH : String(st.rps));
  setBar(RAIL.req, st.rps == null ? 0 : (st.rps / peakRps * 100));
  setDetail(RAIL.req, st.count == null
    ? 'no journal evidence'
    : 'p95 ' + fmtDur(st.p95_s) + ' · ' + (st.error_count || 0) + ' err · peak '
      + peakRps.toFixed(peakRps < 10 ? 1 : 0) + ' rps');
}

function renderRailJob(jobs){
  const entries = Object.entries(jobs);
  const active = entries.filter(([, j]) => !j.done).sort((a, b) => (b[1].started || 0) - (a[1].started || 0));
  if(!active.length){ RAIL.job.hidden = true; return; }
  RAIL.job.hidden = false;
  const [name, job] = active[0];
  setText(RAIL.jobName, job.kind === 'benchmark' ? 'Benchmark suite' : name);
  // A pull can carry byte telemetry while its model-wide percentage is still
  // indeterminate. Hide the bar rather than draw a zero that reads as progress.
  if(job.pct == null){
    RAIL.jobBar.parentNode.style.visibility = 'hidden';
  } else {
    RAIL.jobBar.parentNode.style.visibility = 'visible';
    const w = clamp(job.pct, 0, 100).toFixed(1) + '%';
    if(RAIL.jobBar.style.width !== w) RAIL.jobBar.style.width = w;
  }
  const bits = [job.status || job.kind];
  if(job.pct != null) bits.push(job.pct.toFixed(0) + '%');
  if(job.rate_bps) bits.push(fmtRate(job.rate_bps));
  if(job.eta_s != null) bits.push('eta ' + fmtSpan(job.eta_s));
  setText(RAIL.jobLine, bits.join(' · '));
  if(active.length > 1) setText(RAIL.jobName, active.length + ' jobs running');
}

/* ------------------------------------------------------------ sparklines - */

function drawSpark(node, series, opts){
  const W = 600, H = 130, pad = 3;
  node.replaceChildren();
  if(!series.length || !series[0].points.length){
    node.appendChild(svg('text', {x:6, y:70, fill:'currentColor', 'font-size':'15',
      opacity:'.55', 'font-family':'ui-monospace, monospace'}));
    node.lastChild.textContent = 'collecting samples';
    return;
  }
  const n = Math.max(series[0].points.length, 2);
  const step = (W - 2 * pad) / (n - 1);
  const max = opts && opts.max ? opts.max : 100;
  // Percentage charts keep a fixed 0-100 scale, because auto-scaling would make
  // 27% VRAM look full. Gridlines make the resulting headroom read as scale
  // rather than as empty space, and give the eye something to measure against.
  if(opts && opts.grid){
    for(const frac of [0.25, 0.5, 0.75, 1]){
      const y = (pad + (H - 2 * pad) * (1 - frac)).toFixed(1);
      node.appendChild(svg('line', {
        x1:pad, x2:W - pad, y1:y, y2:y,
        stroke:'var(--line)', 'stroke-width':'1', 'vector-effect':'non-scaling-stroke',
      }));
    }
  }
  for(const s of series){
    const pts = s.points.map((v, i) =>
      (pad + i * step).toFixed(1) + ',' + (pad + (H - 2 * pad) * (1 - clamp(v / max, 0, 1))).toFixed(1)
    ).join(' ');
    if(s.fill){
      node.appendChild(svg('polygon', {
        points: pad + ',' + (H - pad) + ' ' + pts + ' ' + (pad + (n - 1) * step) + ',' + (H - pad),
        fill: s.color, opacity: '0.16',
      }));
    }
    node.appendChild(svg('polyline', {
      points: pts, fill:'none', stroke:s.color, 'stroke-width':'1.6',
      'vector-effect':'non-scaling-stroke',
      'stroke-linejoin':'round', 'stroke-linecap':'round',
      'stroke-dasharray': s.dashed ? '3 3' : '',
    }));
  }
}
function legend(items){
  return items.map(([role, label, value, area]) => ce('span', {cls:'legend-key r-' + role}, [
    ce('span', {cls:'legend-swatch' + (area ? ' area' : ''), attrs:{'aria-hidden':'true'}}),
    ce('span', {text:label + ' '}),
    ce('b', {text:value}),
  ]));
}

/* ======================================================== view: MODELS ==== */

function renderModels(s){
  // Never rebuild a table the user is currently inside: a poll landing mid-tab
  // or mid-click would move the focus out from under them.
  const residentHost = $('resident');
  if(!residentHost.contains(document.activeElement)){
    const loaded = s.loaded || [];
    setText($('resident-note'), loaded.length
      ? plural(loaded.length, 'model held in memory', 'models held in memory')
      : 'Nothing is held in memory');
    if(!loaded.length){
      put(residentHost, empty(
        s.ollama_ok === false ? 'No data' : 'No model is resident',
        s.ollama_ok === false
          ? 'Ollama is unreachable, so the dashboard cannot tell what is loaded.'
          : 'Ollama loads a model on the first request, or you can load one now and keep it warm.',
        s.ollama_ok === false ? null : [ce('button', {cls:'btn sm', text:'Go to the load form',
          on:{click: () => { $('load-model').focus(); $('load-model').scrollIntoView({block:'center'}); }}})]));
    } else {
      put(residentHost, tbl('t-resident',
        ['model','arch','quant','size','placement','ctx','ttl','capabilities',''],
        loaded.map(m => tr([
          td(m.model_key, 'name'),
          td(m.arch), td(m.quant),
          td(fmtBytes(m.size), 'num'),
          td(m.fully_gpu ? pill('GPU', 'good') : pill('CPU +' + fmtBytes(m.cpu_bytes), 'warn')),
          td(m.max_context ? fmtCtx(m.context) + ' / ' + fmtCtx(m.max_context) : fmtCtx(m.context), 'num'),
          td(m.ttl_s == null ? 'none' : fmtSpan(m.ttl_s), 'num'),
          td(capPills(m)),
          td(ce('button', {cls:'btn danger sm', text:'Unload',
            attrs:{'data-needs-ollama':'', 'aria-label':'Unload ' + m.model_key},
            on:{click: e => unloadModel(m.model_key, e.currentTarget)}}), 'act'),
        ]))));
    }
  }

  const inventoryHost = $('inventory');
  if(!inventoryHost.contains(document.activeElement) && !$('bench-models').contains(document.activeElement)){
    const lib = s.library || [];
    setText($('inventory-note'), lib.length
      ? plural(lib.length, 'model on disk', 'models on disk') : 'Nothing on disk');
    syncLoadSelect(lib);
    syncBenchList(lib);
    if(!lib.length){
      put(inventoryHost, empty(
        s.ollama_ok === false ? 'No data' : 'No models yet',
        s.ollama_ok === false
          ? 'Ollama is unreachable, so the local inventory cannot be read.'
          : 'Nothing has been pulled to this machine. Search the catalog above and download a model to get started.'));
    } else {
      put(inventoryHost, tbl('t-inventory',
        ['model','arch','params','quant','size','max ctx','capabilities',''],
        lib.map(m => tr([
          td([document.createTextNode(m.model_key), m.loaded ? ce('span', {cls:'pill good', text:'loaded',
            attrs:{style:'margin-left:7px'}}) : null], 'name'),
          td(m.arch), td(m.params), td(m.quant),
          td(fmtBytes(m.size), 'num'),
          td(fmtCtx(m.max_context), 'num'),
          td(capPills(m)),
          td(ce('button', {cls:'btn danger sm', text:'Delete',
            attrs:{'data-needs-ollama':'', 'aria-label':'Delete ' + m.model_key},
            on:{click: e => openDelete(m, e.currentTarget)}}), 'act'),
        ]))));
    }
  }

  const available = s.ollama_ok !== false;
  setOllamaAvailable(available);
  $('btn-unload-all').disabled = !available || !(s.loaded || []).length;
  $('btn-load').disabled = !available || !(s.library || []).length;
  $('btn-fit').disabled = !available || !(s.library || []).length;
  syncDownloadButton();

  const count = (s.loaded || []).length;
  const badge = $('count-models');
  badge.hidden = !count;
  setText(badge, String(count));
}

function syncLoadSelect(lib){
  const sel = $('load-model');
  const keep = sel.value;
  put(sel, lib.map(m => ce('option', {text:m.model_key, attrs:{value:m.model_key}})));
  if(keep && Array.from(sel.options).some(o => o.value === keep)) sel.value = keep;
}

function syncBenchList(lib){
  const host = $('bench-models');
  const checked = new Set(Array.from(host.querySelectorAll('input:checked'), i => i.value));
  put(host, lib.map(m => {
    const box = ce('input', {attrs:{type:'checkbox', value:m.model_key}});
    // Embedding models never produce a completion, so they cannot be timed by
    // the same workload.
    box.disabled = !!m.embedding;
    box.checked = !m.embedding && checked.has(m.model_key);
    return ce('label', {cls:'check'}, [box,
      ce('span', {text:m.model_key + (m.embedding ? ' · embedding' : '')})]);
  }));
  if(!benchInitialised && lib.length){
    const first = host.querySelector('input:not(:disabled)');
    if(first) first.checked = true;
    benchInitialised = true;
  }
}

/* ------------------------------------------------------------------ jobs - */

function renderJobs(jobs){
  JOBS = jobs || {};
  renderRailJob(JOBS);
  renderBench(JOBS[BENCH_KEY]);
  if(STATE) renderRailState(STATE);   // job failures feed the verdict

  const entries = Object.entries(JOBS).sort((a, b) => (b[1].started || 0) - (a[1].started || 0));
  let active = 0, done = 0, failed = 0;
  for(const [, j] of entries){
    if(j.error) failed++; else if(j.done) done++; else active++;
  }
  setText($('jobs-note'), entries.length
    ? active + ' active · ' + done + ' done · ' + failed + ' failed'
    : 'No history');
  $('btn-clear-jobs').disabled = !entries.some(([, j]) => j.done);

  const host = $('jobs');
  if(host.contains(document.activeElement)) return;
  if(!entries.length){
    put(host, empty('No jobs yet',
      'Loads, downloads and benchmarks appear here while they run, and stay until you clear them.'));
    return;
  }
  put(host, ce('div', {cls:'joblist'}, entries.map(([name, j]) => {
    const tone = j.error ? 'bad' : j.done ? 'good' : 'warn';
    const label = j.error ? 'failed' : j.done ? 'done'
      : j.kind === 'load' ? 'loading' : j.kind === 'download' ? 'downloading' : 'running';
    const metrics = [];
    if(!j.done){
      if(j.pct != null) metrics.push(['progress', j.pct.toFixed(1) + '%']);
      if(j.total) metrics.push(['transferred', fmtBytes(j.completed) + ' / ' + fmtBytes(j.total)]);
      if(j.rate_bps) metrics.push(['rate', fmtRate(j.rate_bps)]);
      if(j.eta_s != null) metrics.push(['eta', fmtSpan(j.eta_s)]);
    }
    return ce('div', {cls:'job'}, [
      ce('div', {cls:'job-top'}, [
        ce('span', {cls:'job-name', text: j.kind === 'benchmark' ? 'Benchmark suite' : name}),
        ce('span', {cls:'pills'}, [pill(j.kind, 'quiet'), pill(label, tone)]),
      ]),
      ce('p', {cls:'job-line', text: j.error || j.last_line || j.status || EMDASH}),
      (!j.done && j.pct != null)
        ? ce('div', {cls:'gauge r-req'}, [ce('i', {attrs:{style:'width:' + clamp(j.pct, 0, 100) + '%'}})])
        : null,
      metrics.length ? ce('div', {cls:'job-metrics'}, metrics.map(([k, v]) =>
        ce('span', {}, [document.createTextNode(k + ' '), ce('b', {text:v})]))) : null,
    ]);
  })));
}

/* ====================================================== view: ACTIVITY ==== */

function renderActivity(s){
  const st = s.stats_5m || {};
  const errTone = st.error_count ? 'bad' : 'good';
  put($('figures'), [
    figure('requests', st.count, 'req', 'in the window'),
    figure('rate', st.rps, 'req', st.rps != null ? 'per second' : ''),
    figure('errors', st.error_count, errTone,
      st.error_count ? st.error_rate + '% of traffic' : 'none in the window'),
    figure('p50', fmtDur(st.p50_s), 'req', 'median latency'),
    figure('p95', fmtDur(st.p95_s), 'req', 'slow tail'),
    figure('p99', fmtDur(st.p99_s), 'req', 'slowest tail'),
  ]);

  const badge = $('count-activity');
  badge.hidden = !st.error_count;
  badge.className = 'tab-count bad';
  setText(badge, String(st.error_count || 0));

  setText($('req-freshness'), s.log_follower_ok === false
    ? 'The follower stopped, so this list is frozen'
    // Age grows without bound on an idle server, which is not a fault. Never
    // alarm on it by itself.
    : s.log_age_s == null ? 'No request evidence yet'
    : 'Newest line ' + fmtSpan(s.log_age_s) + ' old');

  const reqs = s.requests || [];
  put($('requests'), reqs.length
    ? tbl('t-requests', ['time','method','endpoint','status','latency','client','model (inferred)'],
        reqs.map(r => tr([
          td(r.ts),
          td(r.method || pill(r.kind, 'quiet')),
          td(r.path, 'wrap t-io'),
          td(r.status == null ? EMDASH : pill(String(r.status), r.status >= 400 ? 'bad' : 'good')),
          td(fmtDur(r.latency_s), 'num'),
          td(r.client, 'wrap t-dim'),
          td(r.model, 'wrap t-dim'),
        ])))
    : empty('No requests in the window',
        s.log_follower_ok === false
          ? 'The journal follower is not running, so no request evidence is being read.'
          : 'Ollama has served nothing in the last five minutes. On an idle server this is normal.'));

  const clients = s.by_client || [];
  put($('clients'), clients.length
    ? tbl('t-clients', ['client','requests','errors','last seen'], clients.map(c => tr([
        td(c.client, 'wrap t-io'),
        td(c.count, 'num'),
        td(c.errors ? pill(String(c.errors), 'bad') : '0', 'num'),
        td(fmtSpan(Math.max(0, Date.now() / 1000 - c.last_seen)) + ' ago', 'num'),
      ])))
    : empty(null, 'No clients have called in the window.'));

  const eps = s.top_endpoints || [];
  put($('endpoints'), eps.length
    ? tbl('t-endpoints', ['path','count','errors','p95'], eps.map(e => tr([
        td(e.path, 'wrap t-io'),
        td(e.count, 'num'),
        td(e.errors ? pill(String(e.errors), 'bad') : '0', 'num'),
        td(fmtDur(e.p95_s), 'num'),
      ])))
    : empty(null, 'No endpoints have been hit in the window.'));

  const problems = s.problems || [];
  put($('problems'), problems.length
    ? tbl('t-problems', ['time','level','message'], problems.map(p => tr([
        td(p.ts),
        td(pill(p.level || 'WARN', p.level === 'ERROR' ? 'bad' : 'warn')),
        td(p.message, 'wrap'),
      ])))
    : empty(null, 'No warnings or errors in the journal window.'));
}

function figure(label, value, role, sub){
  return ce('div', {cls:'figure r-' + role}, [
    ce('div', {cls:'figure-k', text:label}),
    ce('div', {cls:'figure-v on-role', text: value == null ? EMDASH : String(value)}),
    ce('div', {cls:'figure-sub', text: sub || ''}),
  ]);
}

/* ========================================================== view: HOST ==== */

// Throttle flags flip on and off between consecutive samples, so a raw read at
// poll rate would strobe. Latch each reason briefly past its last sighting so
// the indicator reads steady rather than flickering.
const THROTTLE_HOLD_MS = 2500;
const throttleSeen = new Map();
function heldThrottles(reasons){
  const now = Date.now();
  for(const r of reasons || []) throttleSeen.set(r, now);
  const out = [];
  for(const [r, t] of throttleSeen){
    if(now - t <= THROTTLE_HOLD_MS) out.push(r);
    else throttleSeen.delete(r);
  }
  return out;
}

function renderHostLive(s){
  const g = s.gpu || {};
  if(g.error){
    setText($('gpu-note'), 'Unavailable');
    put($('gpu-kv'), kv([['error', g.error, true]]));
    drawSpark($('gpu-spark'), []);
    put($('gpu-spark-legend'), []);
  } else {
    const versions = (STATE && STATE.gpu_versions) || {};
    const active = heldThrottles(g.throttle_reasons);
    const powerCapped = active.some(r => r === 'sw_power_cap' || r === 'hw_power_brake');
    const thermal = active.some(r => r.includes('thermal'));
    const other = active.filter(r => !r.includes('thermal') && r !== 'sw_power_cap' && r !== 'hw_power_brake');
    const vramPct = g.mem_total ? (g.mem_used / g.mem_total * 100) : null;

    setText($('gpu-note'), g.name || 'Sampled');
    put($('gpu-kv'), kv([
      ['device', g.name],
      (versions.driver || versions.cuda)
        ? ['driver', 'driver ' + (versions.driver || EMDASH) + ' · cuda ' + (versions.cuda || EMDASH)] : null,
      ['utilization', g.util + '%' + (other.length ? ' · ' + other.join(', ') : ''), other.length > 0],
      ['vram', fmtBytes(g.mem_used * 1048576) + ' / ' + fmtBytes(g.mem_total * 1048576)
        + (vramPct == null ? '' : ' (' + vramPct.toFixed(1) + '%)')],
      ['temperature', (g.temp_mem == null ? g.temp + '°C' : g.temp + '°C (mem ' + g.temp_mem + '°C)')
        + (thermal ? ' · thermal throttle' : ''), thermal],
      ['fan', g.fan == null ? EMDASH : g.fan + '%'],
      ['power', (g.power_limit
        ? g.power.toFixed(1) + ' / ' + g.power_limit.toFixed(0) + ' W ('
          + (g.power / g.power_limit * 100).toFixed(0) + '%)'
        : g.power.toFixed(1) + ' W') + (powerCapped ? ' · power capped' : ''), powerCapped],
    ]));

    const hist = s.gpu_history || [];
    drawSpark($('gpu-spark'), [
      {points: hist.map(p => p.vram_pct), color:'var(--c-memory)', fill:true},
      {points: hist.map(p => p.util), color:'var(--c-compute)'},
    ], {grid:true});
    const last = hist[hist.length - 1];
    put($('gpu-spark-legend'), last ? legend([
      ['memory', 'vram', last.vram_pct.toFixed(0) + '%', true],
      ['compute', 'util', last.util + '%'],
      ['neutral', 'scale', '0 to 100%'],
      ['neutral', '', hist.length + ' samples'],
    ]) : []);
  }

  // PCIe
  const pc = s.pcie || {};
  const lat = pc.latest || {};
  if(g.error){
    setText($('pcie-note'), 'Unavailable');
    put($('pcie-kv'), kv([['error', 'PCIe telemetry rides on the GPU sample', true]]));
    drawSpark($('pcie-spark'), []);
    put($('pcie-spark-legend'), []);
  } else {
    const negotiated = 'gen ' + g.pcie_gen + '.0 x' + g.pcie_width;
    const max = 'gen ' + g.pcie_gen_max + '.0 x' + g.pcie_width_max;
    const cap = linkCapMbs(g);
    const total = (lat.rx_mbs || 0) + (lat.tx_mbs || 0);
    setText($('pcie-note'), negotiated === max ? negotiated : negotiated + ', max ' + max);
    put($('pcie-kv'), kv([
      ['link', negotiated + (negotiated === max ? '' : ' (max ' + max + ')')],
      ['link cap', cap ? '~' + (cap / 1024).toFixed(2) + ' GB/s' : null],
      ['rx', (lat.rx_mbs || 0) + ' MB/s'],
      ['tx', (lat.tx_mbs || 0) + ' MB/s'],
      ['total', total + ' MB/s' + (cap ? ' (' + (total / cap * 100).toFixed(1) + '% of link)' : '')],
    ]));
    const hist = pc.history || [];
    const peak = Math.max(1, ...hist.flatMap(p => [p.rx, p.tx]));
    drawSpark($('pcie-spark'), [
      {points: hist.map(p => p.rx), color:'var(--c-io)', fill:true},
      {points: hist.map(p => p.tx), color:'var(--c-io)', dashed:true},
    ], {max: peak});
    put($('pcie-spark-legend'), hist.length ? legend([
      ['io', 'rx', (lat.rx_mbs || 0) + ' MB/s', true],
      ['io', 'tx (dashed)', (lat.tx_mbs || 0) + ' MB/s'],
      ['neutral', 'peak', peak + ' MB/s'],
    ]) : []);
  }

  const h = s.host || {};
  put($('host-kv'), kv([
    ['cpu', h.cpu_pct == null ? null : h.cpu_pct.toFixed(1) + '%'],
    ['cores', h.ncpu],
    ['load 1m', h.load_1],
    ['ram', h.mem_total ? fmtBytes(h.mem_used) + ' / ' + fmtBytes(h.mem_total)
      + ' (' + h.mem_pct.toFixed(1) + '%)' : null],
    ['dashboard up', fmtSpan(s.dash_uptime_s)],
  ]));
}

function renderHostState(s){
  const procs = s.gpu_processes || [];
  setText($('procs-note'), procs.length ? plural(procs.length, 'process', 'processes') : 'None');
  put($('procs'), procs.length
    ? tbl('t-procs', ['pid','process','vram'], procs.map(p => tr([
        td(p.pid, 'num'),
        td(p.name, 'wrap t-io'),
        td(fmtBytes(p.vram_mb * 1048576), 'num'),
      ])))
    : empty(null, 'Nothing is currently holding GPU memory.'));

  const svc = s.service || {}, dk = s.disk || {}, eng = svc.engine || {};
  put($('service-kv'), kv([
    ['ollama.service', svc.active === 'active' ? pill('active', 'good') : pill(svc.active || 'unknown', 'bad')],
    ['engine', eng.name ? eng.name + ' @ ' + eng.version : null],
    ['uptime', fmtSpan(svc.uptime_s)],
    ['rss', svc.rss_kb ? fmtBytes(svc.rss_kb * 1024) : null],
    // approximate means the store could not be read and we are summing
    // /api/tags instead, which counts referenced blobs only. Showing that as an
    // exact figure would understate real usage without saying so.
    ['model store', dk.models_size
      ? (dk.approximate ? '≥ ' + fmtBytes(dk.models_size) : fmtBytes(dk.models_size)) : null],
    dk.approximate ? ['', 'lower bound: the store is not readable'] : null,
    dk.orphan_bytes ? ['reclaimable', fmtBytes(dk.orphan_bytes)] : null,
    ['disk free', dk.fs_total ? fmtBytes(dk.fs_free) + ' / ' + fmtBytes(dk.fs_total) : null],
  ]));

  const ts = s.tailscale || {};
  put($('ts-kv'), ts.up ? kv([
    ['status', pill('up', 'good')],
    ['hostname', ts.hostname],
    ['dns', ts.dnsname],
    ['ip', ts.ip],
    ['tailnet', ts.tailnet],
    ['peers online', ts.peers_online + ' / ' + ts.peers_total],
  ]) : kv([['status', ts.error || 'down', true]]));

  const cfg = s.settings || {};
  setText($('env-unit'), cfg.unit ? 'unit ' + cfg.unit : '');
  if(cfg.error){
    put($('env'), empty('Settings unavailable', cfg.error));
  } else {
    // The key set is not fixed, since a user may add any OLLAMA_* variable, so
    // this renders whatever arrived. Secret-shaped names are redacted server side.
    const keys = Object.keys(cfg).filter(k => k.startsWith('OLLAMA_')).sort();
    put($('env'), keys.length
      ? tbl('t-env', ['variable','value'], keys.map(k => tr([
          td(k, 'name'), td(String(cfg[k]), 'wrap t-io'),
        ])))
      : empty(null, 'No OLLAMA_* variables are set for the service.'));
    // Server-wide settings with no per-load equivalent, named next to the load
    // form so their absence there reads as "not per-load" rather than missing.
    setText($('server-opts'), 'Server-wide: num_parallel='
      + (cfg.OLLAMA_NUM_PARALLEL != null ? cfg.OLLAMA_NUM_PARALLEL : 'default')
      + ', max_loaded=' + (cfg.OLLAMA_MAX_LOADED_MODELS != null ? cfg.OLLAMA_MAX_LOADED_MODELS : 'default')
      + '. Neither can be set per load.');
  }
}

/* ========================================================= view: BENCH ==== */

function fmtMetric(v, suffix, digits){
  return v == null ? EMDASH : Number(v).toFixed(digits == null ? 1 : digits) + (suffix || '');
}

function renderBench(p){
  const host = $('bench-results');
  const running = !!(p && !p.done);
  $('bench-start').disabled = running || !ollamaAvailable;
  setText($('bench-start'), running ? 'Benchmark running' : 'Run benchmark');
  if(!p){
    put(host, empty('No benchmark yet',
      'Pick one or more completion models and run the suite. Each model is warmed up, then timed over several runs, and the median is reported with every individual run kept as evidence.'));
    return;
  }
  const state = p.error ? 'failed' : p.done ? 'finished' : p.status;
  const tone = p.error ? 'bad' : p.done ? 'good' : 'warn';
  const line = running
    ? (p.current_model || 'preparing') + ' · model ' + (p.model_index || 0) + '/' + (p.model_total || 0)
      + ' · ' + (p.last_line || p.status)
    : (p.last_line || p.error || state);

  const kids = [
    ce('div', {cls:'job-top'}, [
      ce('span', {cls:'job-name', text:line}),
      pill(state, tone),
    ]),
  ];
  if(running && p.pct != null){
    kids.push(ce('div', {cls:'gauge r-req', attrs:{style:'margin-top:10px'}},
      [ce('i', {attrs:{style:'width:' + clamp(p.pct, 0, 100) + '%'}})]));
    kids.push(ce('p', {cls:'hint', text:p.pct.toFixed(1) + '% complete', attrs:{style:'margin-top:6px'}}));
  }

  const results = p.results || [];
  if(results.length){
    kids.push(ce('div', {cls:'tablewrap', attrs:{style:'margin-top:var(--space-4)'}}, [
      tbl('t-bench', ['model','status','output tok/s','prompt tok/s','TTFT','total'],
        results.map(r => tr([
          td(r.model, 'name'),
          td(pill(r.status, r.status === 'failed' ? 'bad' : 'good')),
          td(fmtMetric(r.generation_tps), 'num'),
          td(fmtMetric(r.prompt_tps), 'num'),
          td(fmtMetric(r.ttft_s == null ? null : r.ttft_s * 1000, ' ms', 0), 'num'),
          td(fmtMetric(r.total_s, ' s', 2), 'num'),
        ]))),
    ]));
    for(const r of results){
      const det = ce('details', {}, [ce('summary', {
        text: r.status === 'failed' ? r.model + ': ' + r.error : r.model + ': per-run evidence'})]);
      if((r.runs || []).length){
        det.appendChild(ce('div', {cls:'tablewrap'}, [
          tbl(null, ['run','output tok/s','prompt tok/s','TTFT','load','wall','tokens'],
            r.runs.map(run => tr([
              td(run.run, 'num'),
              td(fmtMetric(run.generation_tps), 'num'),
              td(fmtMetric(run.prompt_tps), 'num'),
              td(fmtMetric(run.ttft_s == null ? null : run.ttft_s * 1000, ' ms', 0), 'num'),
              td(fmtMetric(run.load_s, ' s', 2), 'num'),
              td(fmtMetric(run.wall_s, ' s', 2), 'num'),
              td(run.output_tokens, 'num'),
            ]))),
        ]));
      }
      kids.push(det);
    }
  }
  put(host, kids);
}

/* ======================================================== catalog ========= */

function syncDownloadButton(){
  $('btn-download').disabled = !ollamaAvailable || !$('dl-name').value.trim();
}
function useName(name){
  $('dl-name').value = name;
  syncDownloadButton();
  $('dl-name').scrollIntoView({block:'center'});
  $('dl-name').focus();
}
function renderCatalog(){
  const q = ($('cat-search').value || '').toLowerCase().trim();
  const items = CATALOG.filter(m => {
    if(!q) return true;
    const hay = (m.name + ' ' + m.slug + ' ' + m.description + ' ' + m.sizes.join(' ')
      + ' ' + (m.capabilities || []).join(' ')).toLowerCase();
    return q.split(/\s+/).every(t => hay.includes(t));
  });
  const host = $('catalog');
  if(!items.length){
    put(host, empty(null, CATALOG.length ? 'No models match that filter.'
      : CATALOG_ERROR ? 'The catalog could not be read. You can still type a model name above.'
      : 'The catalog is empty.'));
    return;
  }
  put(host, items.map(m => ce('div', {cls:'cat-item'}, [
    ce('div', {cls:'cat-body'}, [
      ce('div', {cls:'cat-name', text:m.name}),
      m.description ? ce('p', {cls:'cat-desc', text:m.description}) : null,
      ce('div', {cls:'cat-tags'}, [
        // Capabilities first, then parameter sizes: sizes are what you click
        // through to a concrete tag, capabilities are what you filter on.
        ...(m.capabilities || []).map(c => pill(c, 'quiet')),
        ...m.sizes.map(sz => ce('button', {cls:'tagbtn', text:sz,
          attrs:{'aria-label':'Use ' + m.slug + ':' + sz},
          on:{click: () => useName(m.slug + ':' + sz)}})),
      ]),
    ]),
    ce('button', {cls:'btn sm', text:'Use', attrs:{'aria-label':'Use ' + m.slug},
      on:{click: () => useName(m.slug)}}),
  ])));
}
async function refreshCatalog(force){
  try {
    const r = await api('GET', '/api/control/catalog' + (force ? '?refresh=1' : ''));
    CATALOG = r.data || [];
    CATALOG_ERROR = r.error || null;
    setText($('cat-age'), r.cached_age_s != null
      ? (r.cached_age_s === 0 ? 'fresh' : Math.round(r.cached_age_s / 60) + 'm old') : EMDASH);
  } catch(e){
    CATALOG_ERROR = e.message;
  }
  const status = $('cat-status');
  status.hidden = !CATALOG_ERROR;
  setText(status, CATALOG_ERROR ? 'Catalog refresh failed: ' + CATALOG_ERROR : '');
  renderCatalog();
}

/* ======================================================== actions ========= */

async function withBusy(btn, busyLabel, fn){
  const label = btn.textContent;
  btn.disabled = true;
  btn.setAttribute('aria-busy', 'true');
  setText(btn, busyLabel);
  try { await fn(); }
  finally {
    btn.removeAttribute('aria-busy');
    setText(btn, label);
    btn.disabled = false;
  }
}

async function unloadModel(name, btn){
  await withBusy(btn, 'Unloading', async () => {
    try {
      await api('POST', '/api/control/unload', {model:name});
      toast(name + ' unloaded.', 'good');
    } catch(e){ toast('Could not unload ' + name + ': ' + e.message); }
    statePoll.refresh();
  });
}

let unloadAllArmed = null;
function initUnloadAll(){
  const btn = $('btn-unload-all');
  btn.addEventListener('click', async () => {
    if(!unloadAllArmed){
      unloadAllArmed = setTimeout(() => {
        unloadAllArmed = null;
        setText(btn, 'Unload all');
      }, 6000);
      setText(btn, 'Confirm unload all');
      toast('Choose "Confirm unload all" again to evict every resident model.', 'warn');
      return;
    }
    clearTimeout(unloadAllArmed);
    unloadAllArmed = null;
    setText(btn, 'Unload all');
    await withBusy(btn, 'Unloading', async () => {
      try {
        await api('POST', '/api/control/unload', {all:true});
        toast('Every resident model was unloaded.', 'good');
      } catch(e){ toast('Could not unload all models: ' + e.message); }
      statePoll.refresh();
    });
  });
}

function loadOpts(){
  const v = id => { const s = $(id).value.trim(); return s === '' ? null : s; };
  return {model: $('load-model').value, context: v('load-ctx'), gpu: v('load-gpu'), ttl: v('load-ttl')};
}
function loadInputsValid(){
  return $('load-ctx').reportValidity() && $('load-gpu').reportValidity();
}

function initLoad(){
  $('btn-load').addEventListener('click', async () => {
    if(!loadInputsValid()) return;
    const model = $('load-model').value;
    await withBusy($('btn-load'), 'Starting', async () => {
      try {
        const r = await api('POST', '/api/control/load', loadOpts());
        if(r.started) toast('Load started for ' + model + '.', 'good');
        else toast('The load did not start. It may already be running, or the active-job limit was reached.', 'warn');
      } catch(e){ toast('Could not start the load: ' + e.message); }
      jobsPoll.refresh();
    });
  });

  // Not an estimate. Ollama has no equivalent of a dry-run load, so this
  // compares on-disk weights with free VRAM and ignores KV cache and context.
  $('btn-fit').addEventListener('click', async () => {
    if(!loadInputsValid()) return;
    const out = $('fit-out');
    put(out, ce('p', {cls:'hint', text:'Comparing model weights with free VRAM'}));
    await withBusy($('btn-fit'), 'Checking', async () => {
      try {
        const r = await api('POST', '/api/control/load/fit', {model: $('load-model').value});
        if(!r.ok){ put(out, ce('p', {cls:'hint bad', text:'Fit check failed: ' + r.error})); return; }
        put(out, ce('p', {cls:'hint'}, [
          r.fits ? pill('fits', 'good') : pill('may not fit', 'warn'),
          document.createTextNode(' ' + fmtBytes(r.model_bytes) + ' of weights against '
            + fmtBytes(r.free_bytes) + ' free VRAM. This excludes KV cache and context.'),
        ]));
      } catch(e){ put(out, ce('p', {cls:'hint bad', text:'Fit check failed: ' + e.message})); }
    });
  });
}

function initDownload(){
  $('dl-name').addEventListener('input', syncDownloadButton);
  $('btn-download').addEventListener('click', async () => {
    const name = $('dl-name').value.trim();
    if(!name) return;
    await withBusy($('btn-download'), 'Starting', async () => {
      try {
        const r = await api('POST', '/api/control/download', {name});
        if(r.started){
          toast('Download started for ' + name + '.', 'good');
          $('dl-name').value = '';
        } else {
          toast('The download did not start. It may already be running, or the active-job limit was reached.', 'warn');
        }
      } catch(e){ toast('Could not start the download: ' + e.message); }
      jobsPoll.refresh();
    });
    syncDownloadButton();
  });
  $('cat-search').addEventListener('input', renderCatalog);
  $('btn-cat-refresh').addEventListener('click', () => refreshCatalog(true));
}

/* Deleting removes files with no undo, which is the one case where a modal is
   the right answer rather than the lazy one. */
let pendingDelete = null;
function openDelete(model, trigger){
  pendingDelete = {model, trigger};
  setText($('del-copy'), 'Deleting ' + model.model_key + ' removes '
    + fmtBytes(model.size) + ' of files from this Ollama server. There is no undo.');
  $('del-input').value = '';
  setText($('del-status'), '');
  $('del-confirm').disabled = true;
  setText($('del-confirm'), 'Delete model');
  $('confirm-delete').showModal();
  $('del-input').focus();
}
function initDelete(){
  $('del-input').addEventListener('input', () => {
    $('del-confirm').disabled = !pendingDelete || $('del-input').value !== pendingDelete.model.model_key;
  });
  $('del-cancel').addEventListener('click', () => $('confirm-delete').close());
  $('confirm-delete').addEventListener('close', () => {
    if(pendingDelete && pendingDelete.trigger && pendingDelete.trigger.isConnected){
      pendingDelete.trigger.focus();
    }
    pendingDelete = null;
  });
  $('del-confirm').addEventListener('click', async () => {
    if(!pendingDelete) return;
    const {model} = pendingDelete;
    const btn = $('del-confirm');
    btn.disabled = true;
    setText(btn, 'Deleting');
    try {
      await api('DELETE', '/api/control/model', {name:model.model_key, confirm:$('del-input').value});
      $('confirm-delete').close();
      toast(model.model_key + ' deleted.', 'good');
      statePoll.refresh();
    } catch(e){
      setText($('del-status'), 'Delete failed: ' + e.message);
      setText(btn, 'Delete model');
      btn.disabled = $('del-input').value !== model.model_key;
    }
  });
}

function initJobsClear(){
  $('btn-clear-jobs').addEventListener('click', async () => {
    await withBusy($('btn-clear-jobs'), 'Clearing', async () => {
      try {
        await api('POST', '/api/control/jobs/clear', {});
        toast('Finished job history cleared.', 'good');
      } catch(e){ toast('Could not clear job history: ' + e.message); }
      jobsPoll.refresh();
    });
  });
}

function initBench(){
  $('bench-all').addEventListener('click', () => {
    for(const i of $('bench-models').querySelectorAll('input')) i.checked = !i.disabled;
  });
  $('bench-none').addEventListener('click', () => {
    for(const i of $('bench-models').querySelectorAll('input')) i.checked = false;
  });
  $('bench-start').addEventListener('click', async () => {
    const numeric = ['bench-warmups','bench-runs','bench-tokens','bench-context'];
    if(!numeric.every(id => $(id).reportValidity())) return;
    const models = Array.from($('bench-models').querySelectorAll('input:checked'), i => i.value);
    const msg = $('bench-message');
    if(!models.length){ setText(msg, 'Select at least one completion model.'); return; }
    const prompt = $('bench-prompt').value.trim();
    if(!prompt){ setText(msg, 'Enter a benchmark prompt.'); return; }
    const ctx = $('bench-context').value.trim();
    $('bench-start').disabled = true;
    setText(msg, 'Starting');
    try {
      const r = await api('POST', '/api/control/benchmark', {
        models, prompt,
        warmups: Number($('bench-warmups').value),
        runs: Number($('bench-runs').value),
        num_predict: Number($('bench-tokens').value),
        context: ctx === '' ? null : Number(ctx),
      });
      setText(msg, r.started ? 'Benchmark queued.' : 'Another benchmark or model job is already running.');
      if(!r.started) $('bench-start').disabled = false;
    } catch(e){
      setText(msg, 'Could not start: ' + e.message);
      $('bench-start').disabled = false;
    }
    jobsPoll.refresh();
  });
}

/* ======================================================== palette ========= */
/* The reason six workspaces across two pages could become four sections on one
   page without burying anything: every action is still one keystroke away. */

let palItems = [], palIndex = 0;

function buildPalette(){
  const items = [];
  const go = (name, label) => items.push({group:'Go to', name:label,
    hint:'section', run:() => { showView(name, true); $('tab-' + name).focus(); }});
  go('models', 'Models');
  go('activity', 'Activity');
  go('host', 'Host');
  go('bench', 'Bench');

  items.push({group:'Actions', name:'Switch colour theme', hint:'light, dark or auto', run:themeCycle});
  items.push({group:'Actions', name:'Download a model', hint:'jump to the field', run:() => {
    showView('models', true); $('dl-name').focus(); $('dl-name').scrollIntoView({block:'center'});
  }});
  items.push({group:'Actions', name:'Refresh the catalog', hint:'re-scrape the library', run:() => refreshCatalog(true)});
  if(ollamaAvailable && (STATE && (STATE.loaded || []).length)){
    items.push({group:'Actions', name:'Unload all models', hint:'needs confirming', run:() => {
      showView('models', true); $('btn-unload-all').focus(); $('btn-unload-all').click();
    }});
  }
  if(Object.values(JOBS).some(j => j.done)){
    items.push({group:'Actions', name:'Clear finished jobs', hint:'job history', run:() => $('btn-clear-jobs').click()});
  }

  for(const m of (STATE && STATE.loaded) || []){
    items.push({group:'Resident', name:'Unload ' + m.model_key, hint:'evict from memory', run:() => {
      showView('models', true);
      const btn = document.querySelector('#t-resident button[aria-label="Unload ' + CSS.escape(m.model_key) + '"]');
      if(btn) btn.click(); else unloadModel(m.model_key, $('btn-unload-all'));
    }});
  }
  if(ollamaAvailable){
    for(const m of (STATE && STATE.library) || []){
      if(m.loaded) continue;
      items.push({group:'Load a model', name:m.model_key,
        hint: (m.params ? m.params + ' · ' : '') + fmtBytes(m.size),
        run:() => { showView('models', true); $('load-model').value = m.model_key; $('btn-load').click(); }});
    }
  }
  return items;
}

function renderPalette(){
  const q = $('pal-input').value.toLowerCase().trim();
  const matched = palItems.filter(i => !q ||
    q.split(/\s+/).every(t => (i.name + ' ' + i.group + ' ' + (i.hint || '')).toLowerCase().includes(t)));
  const list = $('pal-list');
  if(!matched.length){
    put(list, ce('div', {cls:'pal-empty', text:'Nothing matches that.'}));
    palIndex = 0;
    return;
  }
  palIndex = clamp(palIndex, 0, matched.length - 1);
  const kids = [];
  let group = null;
  matched.forEach((item, i) => {
    if(item.group !== group){
      group = item.group;
      kids.push(ce('div', {cls:'pal-group', text:group}));
    }
    kids.push(ce('div', {
      cls:'pal-item',
      attrs:{role:'option', id:'pal-opt-' + i, 'aria-selected':String(i === palIndex), 'data-active':String(i === palIndex)},
      on:{click: () => { closePalette(); item.run(); },
          mousemove: () => { if(palIndex !== i){ palIndex = i; markActive(); } }},
    }, [
      ce('span', {cls:'pal-item-name', text:item.name}),
      item.hint ? ce('span', {cls:'pal-item-hint', text:item.hint}) : null,
    ]));
  });
  put(list, kids);
  list.dataset.count = String(matched.length);
  palMatched = matched;
  markActive();
}
let palMatched = [];
function markActive(){
  const nodes = $('pal-list').querySelectorAll('.pal-item');
  nodes.forEach((n, i) => {
    const on = i === palIndex;
    n.dataset.active = String(on);
    n.setAttribute('aria-selected', String(on));
    if(on){
      n.scrollIntoView({block:'nearest'});
      $('pal-input').setAttribute('aria-activedescendant', n.id);
    }
  });
}
function openPalette(){
  palItems = buildPalette();
  palIndex = 0;
  $('pal-input').value = '';
  $('palette').showModal();
  renderPalette();
  $('pal-input').focus();
}
function closePalette(){ if($('palette').open) $('palette').close(); }

function initPalette(){
  $('btn-palette').addEventListener('click', openPalette);
  $('pal-input').addEventListener('input', () => { palIndex = 0; renderPalette(); });
  $('pal-input').addEventListener('keydown', e => {
    if(e.key === 'ArrowDown'){ e.preventDefault(); palIndex = Math.min(palIndex + 1, palMatched.length - 1); markActive(); }
    else if(e.key === 'ArrowUp'){ e.preventDefault(); palIndex = Math.max(palIndex - 1, 0); markActive(); }
    else if(e.key === 'Home'){ e.preventDefault(); palIndex = 0; markActive(); }
    else if(e.key === 'End'){ e.preventDefault(); palIndex = palMatched.length - 1; markActive(); }
    else if(e.key === 'Enter'){
      e.preventDefault();
      const item = palMatched[palIndex];
      if(item){ closePalette(); item.run(); }
    }
  });
  document.addEventListener('keydown', e => {
    if((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')){
      e.preventDefault();
      if($('palette').open) closePalette(); else openPalette();
    }
  });
  if(!/Mac|iPhone|iPad/.test(navigator.platform || '')) setText($('kbd-hint'), 'Ctrl K');
}

/* ========================================================== render roots == */

function renderLive(s){
  LIVE = s;
  renderRailLive(s);
  renderHostLive(s);
}
function renderState(s){
  STATE = s;
  renderAlerts();
  renderRailState(s);
  renderModels(s);
  renderActivity(s);
  renderHostState(s);
}

const livePoll = pollStream('live', signal => api('GET', '/api/live', null, signal), renderLive, 250);
const statePoll = pollStream('state', signal => api('GET', '/api/state', null, signal), renderState, 3000);
const jobsPoll = pollStream('jobs', signal => api('GET', '/api/control/jobs', null, signal), renderJobs, 1000);

/* ---------------------------------------------------------------- start -- */

function start(){
  bindRail();
  initTabs();
  initScrollers();
  initUnloadAll();
  initLoad();
  initDownload();
  initDelete();
  initJobsClear();
  initBench();
  initPalette();
  syncThemeButton();
  $('btn-theme').addEventListener('click', themeCycle);

  const pollers = [livePoll, statePoll, jobsPoll];
  document.addEventListener('visibilitychange', () => {
    for(const p of pollers) document.hidden ? p.pause() : p.resume();
  });
  for(const p of pollers) p.start();
  refreshCatalog(false);
}

if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
else start();

})();
