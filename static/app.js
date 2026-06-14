"use strict";

const HISTORY_KEY = "anastomosis_history_v1";
const HISTORY_CAP = 60;

const state = {
  file: null,        // File or Blob to score
  filename: null,
  thumb: null,       // small dataURL for history
  meta: null,        // {features, thresholds}
  fullById: {},      // id -> full analyze() result (images), in-memory only
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};

/* ----------------------------------------------------------- init */
window.addEventListener("DOMContentLoaded", async () => {
  bindStaticEvents();
  await loadMeta();
  loadSamples();
  renderHistory();
});

async function loadMeta() {
  const res = await fetch("/api/meta");
  state.meta = await res.json();
  buildFeatureList();
  buildThresholds();
}

function buildFeatureList() {
  const list = $("#feature-list");
  list.innerHTML = "";
  state.meta.features.forEach((f) => {
    const item = el("label", "feature-item");
    item.innerHTML = `
      <input type="checkbox" class="feat-cb" value="${f.key}" checked />
      <span class="fi-label">${f.label}</span>
      <span class="fi-kind">${f.kind === "count" ? "count" : "category"}</span>
      <span class="info" title="${escapeAttr(f.help)}">?</span>`;
    list.appendChild(item);
  });
  list.querySelectorAll(".feat-cb").forEach((cb) =>
    cb.addEventListener("change", syncSelectAll)
  );
  // Prevent the tooltip "?" from toggling the checkbox.
  list.querySelectorAll(".info").forEach((i) =>
    i.addEventListener("click", (e) => e.preventDefault())
  );
}

function buildThresholds() {
  const list = $("#threshold-list");
  list.innerHTML = "";
  state.meta.thresholds.forEach((t) => {
    const item = el("div", "threshold-item");
    item.innerHTML = `
      <label for="thr-${t.key}">${t.label}</label>
      <input id="thr-${t.key}" type="number" data-key="${t.key}"
             value="${t.default}" min="${t.min}" max="${t.max}" step="${t.step}" />`;
    list.appendChild(item);
  });
}

async function loadSamples() {
  try {
    const res = await fetch("/api/samples");
    const samples = await res.json();
    const strip = $("#sample-strip");
    strip.innerHTML = "";
    samples.forEach((s) => {
      const btn = el("button", "sample-thumb");
      btn.type = "button";
      btn.title = `Sample ${s.name}`;
      btn.innerHTML = `<img src="${s.url}" alt="Sample ${s.name}" />`;
      btn.addEventListener("click", () => pickSample(s, btn));
      strip.appendChild(btn);
    });
  } catch (_) { /* samples are optional */ }
}

/* ----------------------------------------------------------- events */
function bindStaticEvents() {
  const dz = $("#dropzone");
  const input = $("#file-input");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => {
    if (input.files[0]) setFile(input.files[0], input.files[0].name);
  });
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); })
  );
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) setFile(f, f.name);
  });

  $("#select-all").addEventListener("change", (e) => {
    document.querySelectorAll(".feat-cb").forEach((cb) => (cb.checked = e.target.checked));
  });

  $("#advanced-toggle").addEventListener("click", (e) => {
    const panel = $("#advanced-panel");
    const open = panel.hasAttribute("hidden");
    panel.toggleAttribute("hidden", !open);
    e.currentTarget.setAttribute("aria-expanded", String(open));
  });
  $("#reset-thresholds").addEventListener("click", resetThresholds);

  $("#score-btn").addEventListener("click", runAnalysis);

  $("#history-toggle").addEventListener("click", () => toggleDrawer(true));
  $("#history-close").addEventListener("click", () => toggleDrawer(false));
  $("#drawer-scrim").addEventListener("click", () => toggleDrawer(false));
  $("#export-csv").addEventListener("click", exportCsv);
  $("#clear-history").addEventListener("click", clearHistory);
}

function syncSelectAll() {
  const boxes = [...document.querySelectorAll(".feat-cb")];
  $("#select-all").checked = boxes.every((cb) => cb.checked);
}

function resetThresholds() {
  state.meta.thresholds.forEach((t) => {
    const inp = document.getElementById(`thr-${t.key}`);
    if (inp) inp.value = t.default;
  });
}

/* ----------------------------------------------------------- file handling */
async function setFile(file, name) {
  state.file = file;
  state.filename = name || "image";
  const url = URL.createObjectURL(file);
  const preview = $("#preview");
  preview.src = url;
  preview.hidden = false;
  $("#dropzone-empty").hidden = true;
  $("#score-btn").disabled = false;
  state.thumb = await makeThumb(url);
}

async function pickSample(sample, btn) {
  document.querySelectorAll(".sample-thumb").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  const res = await fetch(sample.url);
  const blob = await res.blob();
  await setFile(blob, `sample-${sample.name}.png`);
}

function makeThumb(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const max = 72;
      const scale = Math.min(max / img.width, max / img.height, 1);
      const c = document.createElement("canvas");
      c.width = Math.max(1, Math.round(img.width * scale));
      c.height = Math.max(1, Math.round(img.height * scale));
      c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
      try { resolve(c.toDataURL("image/jpeg", 0.6)); }
      catch (_) { resolve(null); }
    };
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

/* ----------------------------------------------------------- analysis */
function getSelectedFeatures() {
  return [...document.querySelectorAll(".feat-cb:checked")].map((cb) => cb.value);
}
function getOverrides() {
  const o = {};
  document.querySelectorAll("#threshold-list input").forEach((inp) => {
    if (inp.value !== "") o[inp.dataset.key] = parseFloat(inp.value);
  });
  return o;
}

async function runAnalysis() {
  if (!state.file) return;
  const features = getSelectedFeatures();
  if (features.length === 0) { showError("Select at least one feature to score."); return; }

  showView("loading");
  const fd = new FormData();
  fd.append("image", state.file, state.filename);
  fd.append("features", JSON.stringify(features));
  fd.append("overrides", JSON.stringify(getOverrides()));

  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) { showError(data.error || `Request failed (${res.status}).`); return; }
    data.filename = state.filename;
    renderResults(data);
    addHistory(data);
  } catch (err) {
    showError("Could not reach the server. Is the app still running?");
  }
}

function showView(which) {
  $("#empty-state").hidden = which !== "empty";
  $("#loading").hidden = which !== "loading";
  $("#results").hidden = which !== "results";
  $("#error-box").hidden = which !== "error";
}

function showError(msg) {
  $("#error-box").textContent = msg;
  showView("error");
}

function renderResults(data) {
  $("#result-filename").textContent = data.filename;
  const errors = data.results.filter((r) => r.status === "error").length;
  $("#result-meta").textContent =
    `${data.num_stitches} stitches detected · ${data.results.length} feature${data.results.length === 1 ? "" : "s"} scored · ` +
    `${errors} flagged`;
  $("#overview-img").src = data.overview;

  const grid = $("#result-grid");
  grid.innerHTML = "";
  data.results.forEach((r) => grid.appendChild(resultCard(r)));
  showView("results");
}

function resultCard(r) {
  const card = el("div", "result-card");
  const feat = state.meta.features.find((f) => f.key === r.key) || {};
  const head = el("div", "rc-head");
  head.innerHTML = `<span class="rc-title">${r.label}</span>
                    <span class="rc-badge ${r.status}">${r.display}</span>`;
  card.appendChild(head);

  if (r.image) {
    const wrap = el("div", "rc-img");
    wrap.innerHTML = `<img src="${r.image}" alt="${r.label} diagnostic overlay" loading="lazy" />`;
    card.appendChild(wrap);
  } else if (r.error) {
    card.appendChild(el("div", "rc-error", r.error));
  }
  if (feat.help) card.appendChild(el("div", "rc-help", feat.help));
  return card;
}

/* ----------------------------------------------------------- history */
function readHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch (_) { return []; }
}
function writeHistory(arr) {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(arr)); }
  catch (_) { /* quota — drop silently */ }
}

function addHistory(data) {
  const id = `${Date.now()}-${Math.round(Math.random() * 1e6)}`;
  state.fullById[id] = data;
  const entry = {
    id,
    filename: data.filename,
    ts: new Date().toISOString(),
    numStitches: data.num_stitches,
    thumb: state.thumb,
    items: data.results.map((r) => ({
      key: r.key, label: r.label, kind: r.kind, display: r.display, status: r.status,
    })),
  };
  const arr = readHistory();
  arr.unshift(entry);
  writeHistory(arr.slice(0, HISTORY_CAP));
  renderHistory();
}

function renderHistory() {
  const arr = readHistory();
  $("#history-count").textContent = arr.length;
  const list = $("#history-list");
  list.innerHTML = "";
  $("#history-empty").hidden = arr.length > 0;
  arr.forEach((entry) => {
    const errs = entry.items.filter((i) => i.status === "error").length;
    const item = el("div", "history-item");
    item.innerHTML = `
      <img src="${entry.thumb || ""}" alt="" />
      <div class="hi-body">
        <div class="hi-name">${escapeHtml(entry.filename)}</div>
        <div class="hi-time">${formatTime(entry.ts)} · ${entry.numStitches} stitches</div>
      </div>
      <span class="hi-badge ${errs ? "has-errors" : ""}">${errs ? errs + " flagged" : "clean"}</span>`;
    item.addEventListener("click", () => openHistory(entry));
    list.appendChild(item);
  });
}

function openHistory(entry) {
  toggleDrawer(false);
  const full = state.fullById[entry.id];
  if (full) { renderResults(full); return; }
  // Reloaded session: images are gone, show a summary table.
  $("#result-filename").textContent = entry.filename;
  $("#result-meta").textContent =
    `${entry.numStitches} stitches · scored ${formatTime(entry.ts)} · overlays not stored — re-score to regenerate`;
  $("#overview-img").removeAttribute("src");
  const grid = $("#result-grid");
  grid.innerHTML = "";
  entry.items.forEach((i) => grid.appendChild(resultCard({ ...i, image: null })));
  showView("results");
}

function clearHistory() {
  if (!confirm("Clear all scored images from this session?")) return;
  localStorage.removeItem(HISTORY_KEY);
  state.fullById = {};
  renderHistory();
}

function exportCsv() {
  const arr = readHistory();
  if (arr.length === 0) { alert("No history to export yet."); return; }
  const featCols = state.meta.features.map((f) => f.label);
  const featKeys = state.meta.features.map((f) => f.key);
  const header = ["timestamp", "filename", "stitches", ...featCols];
  const rows = [header];
  arr.slice().reverse().forEach((entry) => {
    const byKey = {};
    entry.items.forEach((i) => (byKey[i.key] = i.display));
    rows.push([
      entry.ts, entry.filename, entry.numStitches,
      ...featKeys.map((k) => (k in byKey ? byKey[k] : "")),
    ]);
  });
  const csv = rows.map((r) => r.map(csvCell).join(",")).join("\r\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const a = el("a");
  a.href = URL.createObjectURL(blob);
  a.download = `anastomosis-scores-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ----------------------------------------------------------- drawer + utils */
function toggleDrawer(open) {
  $("#history-drawer").hidden = !open;
  $("#drawer-scrim").hidden = !open;
}

function formatTime(iso) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}
function csvCell(v) {
  const s = String(v ?? "");
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function escapeAttr(s) { return escapeHtml(s).replace(/\n/g, " "); }
