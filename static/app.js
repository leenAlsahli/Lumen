const $ = (s, r = document) => r.querySelector(s);
const EMOJI = { buildings: "🏢", forest: "🌲", glacier: "🧊", mountain: "⛰️", sea: "🌊", street: "🛣️" };
const pct = v => (v * 100).toFixed(1) + "%";

/* ---------- nav + reveal ---------- */
const nav = $("#nav");
addEventListener("scroll", () => nav.classList.toggle("scrolled", scrollY > 12), { passive: true });
const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: .15 });
const observe = () => document.querySelectorAll(".reveal:not(.in)").forEach(el => io.observe(el));

/* ---------- static content from metrics.json ---------- */
function counter(el, target, decimals, suffix) {
  const t0 = performance.now(), dur = 1600;
  const step = t => { const k = Math.min((t - t0) / dur, 1), e = 1 - Math.pow(1 - k, 3);
    el.textContent = (target * e).toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) + suffix;
    if (k < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}

function lineChart(el, title, caption, series, { min, max, ticks, fmt }) {
  const W = 540, H = 290, p = { l: 44, r: 10, t: 12, b: 30 }, n = series[0].data.length;
  const x = i => p.l + i * (W - p.l - p.r) / (n - 1), y = v => p.t + (1 - (v - min) / (max - min)) * (H - p.t - p.b);
  let g = "";
  ticks.forEach(t => g += `<line class="grid" x1="${p.l}" x2="${W - p.r}" y1="${y(t)}" y2="${y(t)}"/><text x="${p.l - 8}" y="${y(t) + 4}" text-anchor="end">${fmt(t)}</text>`);
  [1, 10, 20, 30, 40, n].forEach(e => g += `<text x="${x(e - 1)}" y="${H - 8}" text-anchor="middle">${e}</text>`);
  const paths = series.map(s => `<path class="draw" stroke="${s.color}" d="${s.data.map((v, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1)).join("")}"/>`).join("");
  el.innerHTML = `<h3>${title}</h3><svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}">${g}${paths}</svg>
    <div class="legend">${series.map(s => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("")}</div><figcaption>${caption}</figcaption>`;
  el.querySelectorAll(".draw").forEach(p => p.style.setProperty("--len", Math.ceil(p.getTotalLength()) + 2));
}

function confusion(m) {
  const cm = m.confusion_matrix, cls = m.classes;
  let h = `<span></span>` + cls.map(c => `<span class="h col">${c}</span>`).join("");
  cm.forEach((row, r) => {
    const tot = row.reduce((a, b) => a + b, 0);
    h += `<span class="h">${cls[r]}</span>`;
    row.forEach((v, c) => {
      const k = v / tot, a = Math.pow(k, .55);
      const bg = `rgb(${Math.round(251 - 227 * a)},${Math.round(249 - 195 * a)},${Math.round(244 - 197 * a)})`;
      h += `<span class="c" style="--i:${r * 6 + c};background:${bg};color:${k > .45 ? "#fdf2de" : "#18362f"}" title="True ${cls[r]} → predicted ${cls[c]}: ${v} (${(k * 100).toFixed(1)}%)">${v}</span>`;
    });
  });
  $("#cm").innerHTML = h + `<span class="ax">Predicted class</span>`;
  $("#rep").innerHTML = cls.map(c => { const r = m.report[c];
    return `<div><span class="n">${c}</span><span class="t"><span class="f" style="--w:${r.f1 * 100}%"></span></span><small>${r.f1.toFixed(2)}</small></div>`; }).join("");
}

async function loadMetrics() {
  const m = await (await fetch("/static/data/metrics.json")).json();
  const s = m.stats;
  $("#stats").innerHTML = [[s.images, 0, "K+", "Images in dataset"], [s.classes, 0, "", "Scene categories"], [s.accuracy, 2, "%", "Test accuracy"]]
    .map(([v, d, suf, l], i) => `<div class="stat"><b data-v="${i === 0 ? s.images / 1000 : v}" data-d="${d}" data-s="${suf}">0</b><span>${l}</span></div>`).join("");
  new IntersectionObserver((es, o) => es.forEach(e => { if (e.isIntersecting) {
    e.target.querySelectorAll("b").forEach(b => counter(b, +b.dataset.v, +b.dataset.d, b.dataset.s)); o.disconnect(); } }), { threshold: .5 }).observe($("#stats"));

  $("#chips").innerHTML = m.classes.map(c => `<span>${EMOJI[c]} ${c}</span>`).join("");
  $("#arch").innerHTML = m.architecture.map(a => `<div class="stage"><i style="--h:${a.h}px"></i><div><b>${a.t}</b><span>${a.d}</span><span>${a.s}</span></div></div>`).join("");
  $("#info").innerHTML = Object.entries(m.info).map(([t, rows]) => `<div><h3>${t}</h3><dl>${rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl></div>`).join("");

  const h = m.history, last = h.acc.length - 1;
  const ax = $("#accChart"), lx = $("#lossChart");
  lineChart(ax, "Accuracy", `Final epoch — train ${pct(h.acc[last])}, validation ${pct(h.val_acc[last])}`,
    [{ name: "Train", color: "#a89a84", data: h.acc }, { name: "Validation", color: "#18362f", data: h.val_acc }],
    { min: .1, max: 1, ticks: [.2, .4, .6, .8, 1], fmt: t => Math.round(t * 100) + "%" });
  lineChart(lx, "Loss", `Final epoch — train ${h.loss[last].toFixed(3)}, validation ${h.val_loss[last].toFixed(3)} (label smoothing 0.1)`,
    [{ name: "Train", color: "#a89a84", data: h.loss }, { name: "Validation", color: "#18362f", data: h.val_loss }],
    { min: .4, max: 4.5, ticks: [.5, 1, 2, 3, 4], fmt: t => t.toFixed(1) });
  try {
    const r = await fetch("/static/data/evaluation.json");
    if (!r.ok) throw 0;
    const ev = await r.json();
    m.confusion_matrix = ev.confusion_matrix; m.report = ev.report;
    if (ev.macro_f1) { const rows = $("#info").querySelectorAll("dl")[2]; rows.insertAdjacentHTML("beforeend", `<div><dt>Macro F1</dt><dd>${ev.macro_f1.toFixed(2)}</dd></div>`); }
    confusion(m);
  } catch (e) { $(".cm-wrap").hidden = true; }
  observe();
}

/* ---------- demo ---------- */
const drop = $("#drop"), fileIn = $("#file"), panel = $("#panel");
let camUrl = null, origUrl = null;

const setView = v => {
  document.querySelectorAll("#seg button").forEach(b => b.classList.toggle("on", b.dataset.v === v));
  $("#prevImg").src = v === "cam" && camUrl ? camUrl : origUrl;
};
$("#seg").addEventListener("click", e => { if (e.target.dataset.v) { e.preventDefault(); setView(e.target.dataset.v); } });

["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => e.dataTransfer.files[0] && handle(e.dataTransfer.files[0]));
drop.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileIn.click(); } });
fileIn.addEventListener("change", () => fileIn.files[0] && handle(fileIn.files[0]));
addEventListener("paste", e => { const f = [...(e.clipboardData?.files || [])].find(f => f.type.startsWith("image/")); if (f) handle(f); });

function showError(msg) {
  $("#resEmpty").hidden = true; $("#resBox").hidden = true;
  const el = $("#err"); el.textContent = msg; el.hidden = false;
}

async function handle(file) {
  if (!file.type.startsWith("image/")) return showError("Please choose an image file (JPG, PNG or WEBP).");
  if (file.size > 10 * 1024 * 1024) return showError("Image is larger than 10 MB.");
  if (origUrl) URL.revokeObjectURL(origUrl);
  origUrl = URL.createObjectURL(file); camUrl = null;
  $("#prevImg").src = origUrl; $("#dropEmpty").hidden = true; $("#preview").hidden = false;
  $("#seg").hidden = true; $("#camHint").hidden = true; setView("orig");
  $("#err").hidden = true; $("#resBox").hidden = true; $("#resEmpty").hidden = false;
  $("#resEmpty .muted").textContent = "Analysing scene…";
  drop.classList.add("scanning");
  const fd = new FormData(); fd.append("file", file);
  try {
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction failed.");
    render(data);
  } catch (err) { showError(err.message); }
  finally { drop.classList.remove("scanning"); fileIn.value = ""; }
}

function render(d) {
  $("#resEmpty").hidden = true; $("#err").hidden = true;
  $("#resEmpty .muted").textContent = "Your result will appear here.";
  $("#rEmoji").textContent = EMOJI[d.label] || "";
  $("#rLabel").textContent = d.label + (d.preview ? " (simulated)" : "");
  $("#rConf").textContent = d.confidence.toFixed(1) + "%";
  $("#bars").innerHTML = d.top3.map(t => `<li><span class="n">${t.label}</span><span class="t"><span class="f" data-w="${t.probability}"></span></span><span class="p">${t.probability.toFixed(1)}%</span></li>`).join("");
  $("#rMeta").textContent = d.preview ? "Simulated · no model loaded" : `Model ${d.model} · inference ${d.inference_ms} ms`;
  const note = $("#rNote");
  note.hidden = d.confidence >= 60 && !d.preview;
  note.textContent = d.preview ? "Preview mode — simulated result for interface testing. Add your model file for real predictions."
    : "Low confidence — this image may not be a typical natural or urban scene.";
  $("#resBox").hidden = false;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $("#ringVal").style.strokeDashoffset = 264 * (1 - d.confidence / 100);
    document.querySelectorAll("#bars .f").forEach(f => f.style.width = f.dataset.w + "%");
  }));
  camUrl = d.gradcam;
  $("#seg").hidden = $("#camHint").hidden = !camUrl;
}

fetch("/api/health").then(r => r.json()).then(h => {
  if (h.status === "no_model") { const n = $("#resEmpty .muted"); n.textContent = "Model file not found — place your .keras or .h5 file in the model/ folder and restart the server."; }
}).catch(() => {});
loadMetrics();
observe();
