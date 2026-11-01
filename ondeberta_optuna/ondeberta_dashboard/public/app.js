// OnDeBERTa dashboard front-end: cursor trail + click bursts, mood-reactive
// background, and SVG charts built from /api/metrics.

const COLORS = {
  neutral: "#3987e5",
  positive: "#22c55e",
  negative: "#f0555a",
  accent2: "#9085e9",
  warn: "#fab219",
  critical: "#d03b3b",
  good: "#0ca30c",
  muted: "#8a8a86",
};

/* ---------------------------- mood state ---------------------------- */
let mood = "neutral"; // 'neutral' | 'positive' | 'negative'
function setMood(next) {
  mood = next;
  document.body.dataset.mood = next;
  const color = COLORS[next] || COLORS.neutral;
  document.documentElement.style.setProperty("--mood-color", color);
  document.documentElement.style.setProperty("--mood-glow", hexToRgba(color, 0.35));
}
function hexToRgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  return `rgba(${r},${g},${b},${a})`;
}

/* ---------------------------- custom cursor ---------------------------- */
const cursorDot = document.getElementById("cursorDot");
window.addEventListener("mousemove", (e) => {
  cursorDot.style.left = e.clientX + "px";
  cursorDot.style.top = e.clientY + "px";
});
window.addEventListener("mousedown", () => cursorDot.classList.add("press"));
window.addEventListener("mouseup", () => cursorDot.classList.remove("press"));

/* ---------------------------- particle trail + bursts ---------------------------- */
const canvas = document.getElementById("particles");
const ctx = canvas.getContext("2d");
let W, H;
function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
resize();
window.addEventListener("resize", resize);

let particles = [];
const MAX_TRAIL_PARTICLES = 140;

function currentMoodColor() {
  return COLORS[mood] || COLORS.neutral;
}

function spawnTrailParticle(x, y) {
  particles.push({
    x, y,
    vx: (Math.random() - 0.5) * 0.6,
    vy: (Math.random() - 0.5) * 0.6,
    r: Math.random() * 2.2 + 1.2,
    life: 1,
    decay: 0.018 + Math.random() * 0.012,
    color: currentMoodColor(),
    kind: "trail",
  });
  if (particles.length > MAX_TRAIL_PARTICLES) particles.shift();
}

function spawnBurst(x, y, color, count = 42) {
  for (let i = 0; i < count; i++) {
    const angle = (Math.PI * 2 * i) / count + Math.random() * 0.3;
    const speed = Math.random() * 4.2 + 1.6;
    particles.push({
      x, y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      r: Math.random() * 2.6 + 1.6,
      life: 1,
      decay: 0.012 + Math.random() * 0.01,
      color,
      kind: "burst",
      gravity: 0.02,
    });
  }
}

let lastTrailAt = 0;
window.addEventListener("mousemove", (e) => {
  const now = performance.now();
  if (now - lastTrailAt > 24) {
    spawnTrailParticle(e.clientX, e.clientY);
    lastTrailAt = now;
  }
});
window.addEventListener("click", (e) => {
  spawnBurst(e.clientX, e.clientY, currentMoodColor(), 22);
});

function tick() {
  ctx.clearRect(0, 0, W, H);
  particles = particles.filter((p) => p.life > 0);
  for (const p of particles) {
    p.x += p.vx;
    p.y += p.vy;
    if (p.gravity) p.vy += p.gravity;
    p.life -= p.decay;
    const alpha = Math.max(p.life, 0);
    ctx.beginPath();
    ctx.fillStyle = hexToRgba(p.color, p.kind === "trail" ? alpha * 0.55 : alpha * 0.9);
    ctx.arc(p.x, p.y, p.r * (p.kind === "burst" ? alpha + 0.3 : 1), 0, Math.PI * 2);
    ctx.fill();
  }
  requestAnimationFrame(tick);
}
tick();

/* ---------------------------- data loading ---------------------------- */
let METRICS = null;

async function loadMetrics() {
  const res = await fetch("/api/metrics");
  METRICS = await res.json();
  renderStats();
  renderModules();
  renderCurveChart();
  renderTrialChart();
  renderParamsTable();
  document.getElementById("modelStatus").textContent = METRICS.modelConnected
    ? "Trained model connected — predictions use model.pt."
    : "No trained model connected — predictions use the demo heuristic (see README.md).";
}

function pct(x) { return (x * 100).toFixed(2) + "%"; }

function renderStats() {
  const t = METRICS.test;
  const grid = document.getElementById("statGrid");
  const items = [
    ["Accuracy", pct(t.accuracy)],
    ["Precision", pct(t.precision)],
    ["Recall", pct(t.recall)],
    ["F1 score", pct(t.f1)],
  ];
  grid.innerHTML = items.map(([label, value]) => `
    <div class="stat-card">
      <div class="label">${label}</div>
      <div class="value">${value}</div>
    </div>
  `).join("");
}

function renderModules() {
  const chips = document.getElementById("moduleChips");
  chips.innerHTML = METRICS.modules.map((m) => `<span class="chip">${m.name}</span>`).join("");

  const grid = document.getElementById("moduleGrid");
  grid.innerHTML = METRICS.modules.map((m, i) => `
    <div class="module-card">
      <div class="n">MODULE ${i + 1}</div>
      <div class="t">${m.name}</div>
      <div class="d">${m.detail}</div>
    </div>
  `).join("");
}

/* ---------------------------- SVG line chart: training curves ---------------------------- */
function renderCurveChart() {
  const svg = document.getElementById("curveChart");
  const epochs = METRICS.epochs;
  const W = 560, H = 260, padL = 42, padR = 16, padT = 16, padB = 30;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  const xs = epochs.map((e) => e.epoch);
  const lossVals = epochs.flatMap((e) => [e.trainLoss, e.valLoss]);
  const yMax = Math.ceil(Math.max(...lossVals) * 10) / 10 + 0.05;
  const yMin = 0;

  const x = (v) => padL + ((v - 1) / (xs.length - 1)) * plotW;
  const y = (v) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  function pathFor(key) {
    return epochs.map((e, i) => `${i === 0 ? "M" : "L"} ${x(e.epoch).toFixed(1)} ${y(e[key]).toFixed(1)}`).join(" ");
  }

  const gridLines = [];
  const steps = 4;
  for (let i = 0; i <= steps; i++) {
    const v = yMin + (yMax - yMin) * (i / steps);
    gridLines.push(`<line class="gridline" x1="${padL}" x2="${W - padR}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}" />`);
    gridLines.push(`<text x="${padL - 8}" y="${(y(v) + 3).toFixed(1)}" text-anchor="end">${v.toFixed(2)}</text>`);
  }

  const xLabels = epochs.map((e) => `<text x="${x(e.epoch).toFixed(1)}" y="${H - 8}" text-anchor="middle">E${e.epoch}</text>`).join("");

  const dotsTrain = epochs.map((e) => `<circle cx="${x(e.epoch).toFixed(1)}" cy="${y(e.trainLoss).toFixed(1)}" r="3.2" fill="${COLORS.neutral}" />`).join("");
  const dotsVal = epochs.map((e) => `<circle cx="${x(e.epoch).toFixed(1)}" cy="${y(e.valLoss).toFixed(1)}" r="3.2" fill="${COLORS.accent2}" />`).join("");

  svg.innerHTML = `
    ${gridLines.join("")}
    <line class="axis-line" x1="${padL}" x2="${padL}" y1="${padT}" y2="${H - padB}" />
    <line class="axis-line" x1="${padL}" x2="${W - padR}" y1="${H - padB}" y2="${H - padB}" />
    <path d="${pathFor("trainLoss")}" fill="none" stroke="${COLORS.neutral}" stroke-width="2" />
    <path d="${pathFor("valLoss")}" fill="none" stroke="${COLORS.accent2}" stroke-width="2" stroke-dasharray="5 4" />
    ${dotsTrain}
    ${dotsVal}
    ${xLabels}
  `;

  document.getElementById("curveLegend").innerHTML = `
    <div class="legend-item"><span class="legend-swatch" style="background:${COLORS.neutral}"></span>Train loss</div>
    <div class="legend-item"><span class="legend-swatch" style="background:${COLORS.accent2}"></span>Validation loss</div>
    <div class="legend-item"><span class="legend-swatch" style="background:${COLORS.good}"></span>Best epoch: ${METRICS.bestEpoch} (val F1 ${pct(epochs[METRICS.bestEpoch - 1].valF1)})</div>
  `;
}

/* ---------------------------- SVG bar chart: Optuna trials ---------------------------- */
function statusColor(status) {
  return { best: COLORS.good, complete: COLORS.neutral, pruned: COLORS.warn, collapsed: COLORS.critical }[status] || COLORS.muted;
}

function renderTrialChart() {
  const svg = document.getElementById("trialChart");
  const trials = METRICS.trials;
  const W = 420, H = 260, padL = 34, padR = 10, padT = 16, padB = 30;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const barW = plotW / trials.length * 0.6;
  const gap = plotW / trials.length;

  const y = (v) => padT + plotH - v * plotH;

  const gridLines = [0, 0.25, 0.5, 0.75, 1].map((v) => `
    <line class="gridline" x1="${padL}" x2="${W - padR}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}" />
    <text x="${padL - 8}" y="${(y(v) + 3).toFixed(1)}" text-anchor="end">${v.toFixed(2)}</text>
  `).join("");

  const bars = trials.map((t, i) => {
    const cx = padL + gap * i + gap / 2;
    const x0 = cx - barW / 2;
    const h = Math.max(t.f1, 0.01) * plotH;
    const y0 = padT + plotH - h;
    return `
      <rect x="${x0.toFixed(1)}" y="${y0.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}"
            rx="3" fill="${statusColor(t.status)}" opacity="${t.status === "pruned" || t.status === "collapsed" ? 0.75 : 1}" />
      <text x="${cx.toFixed(1)}" y="${H - 8}" text-anchor="middle">T${t.trial}</text>
    `;
  }).join("");

  svg.innerHTML = `
    ${gridLines}
    <line class="axis-line" x1="${padL}" x2="${padL}" y1="${padT}" y2="${H - padB}" />
    <line class="axis-line" x1="${padL}" x2="${W - padR}" y1="${H - padB}" y2="${H - padB}" />
    ${bars}
  `;

  const table = document.getElementById("trialTable");
  table.innerHTML = `
    <tr><th>Trial</th><th>Val F1</th><th>Status</th></tr>
    ${trials.map((t) => `
      <tr>
        <td>T${t.trial}</td>
        <td>${t.f1.toFixed(4)}</td>
        <td><span class="status-pill ${t.status}">${t.status}</span></td>
      </tr>
    `).join("")}
  `;
}

function renderParamsTable() {
  const p = METRICS.bestParams;
  const rows = [
    ["Encoder learning rate", p.encoder_lr?.toExponential ? p.encoder_lr.toExponential(2) : p.encoder_lr],
    ["Head learning rate", p.head_lr?.toExponential ? p.head_lr.toExponential(2) : p.head_lr],
    ["Weight decay", p.weight_decay],
    ["Warm-up ratio", p.warmup_ratio],
    ["Dropout", p.dropout],
    ["ON-LSTM hidden size", p.onlstm_hidden],
    ["ON-LSTM chunk size", p.chunk_size],
    ["Adversarial ε", p.adv_epsilon],
    ["Batch size", p.batch_size],
    ["Max sequence length", p.max_len],
  ];
  document.getElementById("paramsTable").innerHTML = `
    <tr><th>Hyperparameter</th><th>Value</th></tr>
    ${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}
  `;
}

/* ---------------------------- predict ---------------------------- */
const analyzeBtn = document.getElementById("analyze");
const reviewBox = document.getElementById("review");
const resultEl = document.getElementById("result");
const resultBig = document.getElementById("resultBig");
const resultConf = document.getElementById("resultConf");
const probeFill = document.getElementById("probeFill");
const attnRow = document.getElementById("attnRow");
const sourceTag = document.getElementById("sourceTag");

analyzeBtn.addEventListener("click", async (e) => {
  const text = reviewBox.value.trim();
  if (!text) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing…";
  setMood("neutral");

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const isPositive = data.positive >= data.negative;
    const nextMood = isPositive ? "positive" : "negative";
    setMood(nextMood);

    const rect = analyzeBtn.getBoundingClientRect();
    spawnBurst(rect.left + rect.width / 2, rect.top + rect.height / 2, currentMoodColor(), 60);

    resultEl.classList.add("show");
    resultBig.textContent = isPositive ? "Positive" : "Negative";
    resultBig.className = "big " + (isPositive ? "positive" : "negative");
    resultConf.textContent = `confidence ${(Math.max(data.positive, data.negative) * 100).toFixed(1)}%`;
    probeFill.style.width = (data.positive * 100).toFixed(1) + "%";

    sourceTag.textContent = data.source === "model" ? "● model prediction" : "○ demo heuristic (no trained model connected)";
    sourceTag.className = "source-tag " + (data.source === "model" ? "model" : "");

    if (data.attention && data.attention.length) {
      attnRow.innerHTML = data.attention
        .filter((a) => !["[CLS]", "[SEP]", "[PAD]"].includes(a.token))
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 12)
        .map((a) => `<span class="attn-tok" title="weight ${a.weight.toFixed(3)}">${a.token.replace(/^▁/, "")}</span>`)
        .join("");
    } else {
      attnRow.innerHTML = "";
    }
  } catch (err) {
    resultEl.classList.add("show");
    resultBig.textContent = "Error";
    resultBig.className = "big negative";
    resultConf.textContent = String(err.message || err);
    setMood("neutral");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze sentiment";
  }
});

loadMetrics();
