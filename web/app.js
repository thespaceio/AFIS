// AFIS browser demo: UI only. The engine runs in worker.js.
"use strict";

const VERSION = document.documentElement.dataset.version || "dev";
const $ = (id) => document.getElementById(id);

const HELP = {
  identify: "Search all enrolled people for the closest match to one print (1 : N).",
  verify: "Check one print against a single claimed identity (1 : 1).",
  analyse: "See every stage of the pipeline and the minutiae (ridge endings and bifurcations) AFIS extracts.",
};
const ACTION = { identify: "Identify", verify: "Verify", analyse: "Analyse" };

const state = { mode: "identify", print: null, info: null, busy: false };

// ---------------------------------------------------------------- worker RPC
const worker = new Worker(`worker.js?v=${encodeURIComponent(VERSION)}`);
const pending = new Map();
let nextId = 1;

worker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "status") return showStatus(msg.text, msg.step);
  const p = pending.get(msg.id);
  if (!p) return;
  pending.delete(msg.id);
  msg.type === "error" ? p.reject(new Error(msg.error)) : p.resolve(msg.result);
};
worker.onerror = (e) => failBoot(e.message || "The engine stopped unexpectedly.");

function rpc(cmd, ...args) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, cmd, args });
  });
}

// ---------------------------------------------------------------- boot
function showStatus(text, step) {
  $("load-text").textContent = text;
  [...$("steps").children].forEach((li, k) => {
    li.className = k + 1 < step ? "done" : k + 1 === step ? "active" : "";
  });
}

function failBoot(text) {
  $("loading").hidden = true;
  $("load-error").hidden = false;
  $("load-error-text").textContent = text;
}

async function boot() {
  $("version").textContent = VERSION;
  if (typeof WebAssembly !== "object") return failBoot("This browser does not support WebAssembly.");
  try {
    state.info = await rpc("boot", VERSION);
  } catch (err) {
    return failBoot(err.message);
  }
  $("loading").hidden = true;
  $("app").hidden = false;
  $("version").textContent = state.info.version;
  renderPeople();
  renderClaim();
  const t = $("threshold");
  t.value = state.info.threshold;
  $("threshold-value").textContent = Number(t.value).toFixed(2);
  setMode("identify");
}

// ---------------------------------------------------------------- rendering
function thumb(name) {
  return `prints/${name}.png`;
}

function renderPeople() {
  $("people").innerHTML = state.info.people.map((p) => `
    <figure class="person">
      <img src="${thumb(p.enrolled)}" alt="Enrolled print of ${p.id}" loading="lazy">
      <figcaption><b>${p.id}</b><span>enrolled from ${p.enrolled}</span></figcaption>
    </figure>`).join("");
}

function renderClaim() {
  $("claim").innerHTML = state.info.people.map((p) => `<option value="${p.id}">${p.id}</option>`).join("");
}

function enrolledNames() {
  return new Set(state.info.people.map((p) => p.enrolled));
}

function renderPrints() {
  const enrolled = enrolledNames();
  // Identify / Verify use the prints that were not enrolled; Analyse can look at any of them.
  const names = state.mode === "analyse" ? state.info.prints : state.info.prints.filter((n) => !enrolled.has(n));
  if (!names.includes(state.print)) state.print = names[0];
  $("pick-hint").textContent = state.mode === "analyse"
    ? "All 16 demo prints."
    : "Other impressions of the enrolled fingers, taken at a different angle, position and pressure.";
  $("prints").innerHTML = names.map((n) => `
    <button class="print" role="option" data-name="${n}" aria-selected="${n === state.print}">
      <img src="${thumb(n)}" alt="" loading="lazy"><span>${n}${enrolled.has(n) ? " · enrolled" : ""}</span>
    </button>`).join("");
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".tab").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
  $("mode-help").textContent = HELP[mode];
  $("claim-control").hidden = mode !== "verify";
  $("threshold-control").hidden = mode === "analyse";
  $("run").textContent = ACTION[mode];
  renderPrints();
  updateRun();
}

function updateRun() {
  $("run").disabled = state.busy || !state.print;
  $("busy").hidden = !state.busy;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function scoreBar(score, threshold) {
  const pct = (v) => `${Math.min(100, Math.max(0, v * 100)).toFixed(1)}%`;
  return `
    <div class="bar" role="img" aria-label="Score ${score.toFixed(3)}, threshold ${threshold.toFixed(2)}">
      <div class="fill ${score >= threshold ? "ok" : "bad"}" style="width:${pct(score)}"></div>
      <div class="mark" style="left:${pct(threshold)}"><span>threshold ${threshold.toFixed(2)}</span></div>
    </div>`;
}

function truthLine(predicted, truth, expectedMatch) {
  // The demo knows which finger each print came from, so it can say whether AFIS got it right.
  const right = expectedMatch === undefined ? predicted === truth : expectedMatch;
  return `<p class="truth ${right ? "right" : "wrong"}">${right ? "✓ Correct" : "✗ Wrong"}:
    this print really belongs to <b>${esc(truth)}</b>.</p>`;
}

function note() {
  return `<p class="note">${esc(state.info.note)}</p>`;
}

function showIdentify(r) {
  const top = r.candidates[0];
  const head = r.candidate
    ? `<h2 class="verdict ok">✓ Candidate: ${esc(r.candidate)}</h2>`
    : `<h2 class="verdict bad">✗ No candidate above the threshold</h2>`;
  const rows = r.candidates.map((c, k) => `
    <tr class="${c.is_match ? "hit" : ""}"><td>${k + 1}</td><td>${esc(c.person)}</td>
      <td class="num">${c.score.toFixed(3)}</td><td class="num">${c.matched}</td>
      <td>${c.is_match ? "above threshold" : "below"}</td></tr>`).join("");
  $("result").innerHTML = `
    ${head}
    ${truthLine(r.candidate, r.truth, r.candidate ? r.candidate === r.truth : false)}
    ${top ? scoreBar(top.score, r.threshold) : ""}
    <div class="table-wrap"><table>
      <thead><tr><th>Rank</th><th>Person</th><th class="num">Score</th><th class="num">Matched minutiae</th><th></th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    ${r.picture ? figure(r.picture, `Left: ${r.probe} (query). Right: ${r.reference} (enrolled ${esc(top.person)}). Lines join minutiae AFIS paired up.`) : ""}
    ${note()}`;
}

function showVerify(r) {
  if (r.error) {
    $("result").innerHTML = `<h2 class="verdict bad">${esc(r.error)}</h2>`;
    return;
  }
  const head = r.is_match
    ? `<h2 class="verdict ok">✓ Match: consistent with ${esc(r.person)}</h2>`
    : `<h2 class="verdict bad">✗ No match: does not match ${esc(r.person)}</h2>`;
  $("result").innerHTML = `
    ${head}
    ${truthLine(null, r.truth, r.is_match === (r.truth === r.person))}
    ${scoreBar(r.score, r.threshold)}
    <p class="muted">Score ${r.score.toFixed(3)} · ${r.matched} matched minutiae · threshold ${r.threshold.toFixed(2)}</p>
    ${r.picture ? figure(r.picture, `Left: ${r.probe} (query). Right: ${esc(r.person)}'s enrolled print.`) : ""}
    ${note()}`;
}

function showAnalyse(r) {
  const stat = (k, v) => `<div class="stat"><span>${k}</span><b>${v}</b></div>`;
  $("result").innerHTML = `
    <h2>${esc(r.name)}</h2>
    <div class="stats">
      ${stat("Quality", r.quality.toFixed(2))}
      ${stat("Minutiae", r.minutiae)}
      ${stat("Endings", r.endings)}
      ${stat("Bifurcations", r.bifurcations)}
      ${stat("Pattern", esc(r.pattern[0].toUpperCase() + r.pattern.slice(1)))}
      ${stat("Cores / deltas", `${r.cores} / ${r.deltas}`)}
    </div>
    ${figure(r.overlay, "Green circles: ridge endings. Blue: bifurcations. Red square: core. Yellow triangle: delta. Ticks show ridge direction.", "overlay")}
    <h3>Pipeline stages</h3>
    <div class="stages">${r.stages.map((s) => `
      <figure><img src="${s.image}" alt="${esc(s.name)}"><figcaption>${esc(s.name)}</figcaption></figure>`).join("")}
    </div>`;
}

function figure(src, caption, cls = "") {
  return `<figure class="pic ${cls}"><img src="${src}" alt=""><figcaption>${esc(caption)}</figcaption></figure>`;
}

// ---------------------------------------------------------------- actions
async function run() {
  if (state.busy || !state.print) return;
  state.busy = true;
  updateRun();
  try {
    let r;
    if (state.mode === "identify") showIdentify((r = await rpc("identify", state.print)));
    else if (state.mode === "verify") showVerify((r = await rpc("verify", state.print, $("claim").value)));
    else showAnalyse((r = await rpc("analyse", state.print)));
    $("result").hidden = false;
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    $("result").hidden = false;
    $("result").innerHTML = `<h2 class="verdict bad">Something went wrong</h2><p>${esc(err.message)}</p>`;
  } finally {
    state.busy = false;
    updateRun();
  }
}

document.querySelector(".tabs").addEventListener("click", (e) => {
  const tab = e.target.closest(".tab");
  if (tab && !state.busy) {
    setMode(tab.dataset.mode);
    $("result").hidden = true;
  }
});

$("prints").addEventListener("click", (e) => {
  const b = e.target.closest(".print");
  if (!b) return;
  state.print = b.dataset.name;
  document.querySelectorAll(".print").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  updateRun();
});

$("run").addEventListener("click", run);

$("threshold").addEventListener("input", (e) => {
  $("threshold-value").textContent = Number(e.target.value).toFixed(2);
});
$("threshold").addEventListener("change", async (e) => {
  try {
    await rpc("set_threshold", Number(e.target.value));
  } catch (err) {
    console.warn(err);
  }
});

boot();
