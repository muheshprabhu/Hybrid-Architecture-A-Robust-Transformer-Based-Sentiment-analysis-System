// OnDeBERTa + Optuna dashboard server.
// Serves the dashboard UI and two APIs:
//   GET  /api/metrics  -> training/Optuna results (from ../ondeberta_optuna/outputs if present, else baked defaults)
//   POST /api/predict  -> runs predict.py against your trained model if configured, else a labeled demo heuristic
const express = require("express");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const PORT = process.env.PORT || 3000;

// --- Optional real-model wiring -------------------------------------------------
// Point these at your downloaded project if you want live predictions from the
// actual trained model instead of the built-in demo heuristic. See README.md.
const PROJECT_DIR = process.env.ONDEBERTA_PROJECT_DIR || path.join(__dirname, "..", "ondeberta_optuna");
const CKPT_DIR = process.env.ONDEBERTA_CKPT_DIR || path.join(PROJECT_DIR, "outputs", "final");
const PYTHON_BIN = process.env.PYTHON_BIN || "python3";

function modelIsAvailable() {
  return fs.existsSync(path.join(CKPT_DIR, "model.pt")) && fs.existsSync(path.join(PROJECT_DIR, "predict.py"));
}

// --- Metrics ---------------------------------------------------------------------
const DEFAULT_METRICS = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "metrics.json"), "utf8"));

function loadMetrics() {
  const metrics = JSON.parse(JSON.stringify(DEFAULT_METRICS)); // clone
  try {
    const resultsPath = path.join(CKPT_DIR, "results.json");
    if (fs.existsSync(resultsPath)) {
      const r = JSON.parse(fs.readFileSync(resultsPath, "utf8"));
      if (r.test) metrics.test = r.test;
    }
  } catch (e) { /* keep defaults */ }
  try {
    const bestParamsPath = path.join(PROJECT_DIR, "outputs", "optuna", "best_params.json");
    if (fs.existsSync(bestParamsPath)) {
      metrics.bestParams = { ...metrics.bestParams, ...JSON.parse(fs.readFileSync(bestParamsPath, "utf8")) };
    }
  } catch (e) { /* keep defaults */ }
  metrics.modelConnected = modelIsAvailable();
  return metrics;
}

app.get("/api/metrics", (req, res) => {
  res.json(loadMetrics());
});

// --- Predict -----------------------------------------------------------------------
// Small, transparent fallback lexicon so the demo works with zero setup.
// This is NOT the trained model — the UI labels it clearly as a heuristic.
const POS_WORDS = ["good","great","excellent","amazing","love","loved","wonderful","best","fantastic","brilliant",
  "perfect","enjoyable","enjoyed","masterpiece","beautiful","superb","funny","touching","recommend","favorite",
  "outstanding","charming","delight","delightful","impressive","engaging","captivating","solid","fun","great"];
const NEG_WORDS = ["bad","terrible","awful","worst","hate","hated","boring","waste","poor","horrible","disappointing",
  "disappointed","dull","mess","weak","annoying","stupid","bland","predictable","cliche","clichéd","slow","mediocre",
  "flat","forgettable","cheesy","overrated","fails","fail","lacking"];

function heuristicPredict(text) {
  const words = (text.toLowerCase().match(/[a-z']+/g) || []);
  let score = 0;
  for (const w of words) {
    if (POS_WORDS.includes(w)) score += 1;
    if (NEG_WORDS.includes(w)) score -= 1;
    if (w === "not" || w === "n't") score -= 0.5; // crude negation nudge
  }
  const p = 1 / (1 + Math.exp(-score / 2)); // squash to (0,1)
  return { positive: p, negative: 1 - p };
}

app.post("/api/predict", async (req, res) => {
  const text = (req.body && req.body.text || "").trim();
  if (!text) return res.status(400).json({ error: "text is required" });

  if (modelIsAvailable()) {
    const py = spawn(PYTHON_BIN, ["predict.py", "--ckpt", CKPT_DIR, text], {
  cwd: PROJECT_DIR,
  env: { ...process.env, PYTHONIOENCODING: "utf-8" },
});
    let out = "", err = "";
    py.stdout.on("data", (d) => (out += d));
    py.stderr.on("data", (d) => (err += d));
    py.on("close", (code) => {
      if (code !== 0) {
        return res.status(500).json({ error: "predict.py failed", detail: err.slice(-2000) });
      }
      const negMatch = out.match(/Negative:\s*([\d.]+)/);
      const posMatch = out.match(/Positive:\s*([\d.]+)/);
      if (!negMatch || !posMatch) {
        return res.status(500).json({ error: "could not parse predict.py output", raw: out });
      }
      const attention = [];
      const lineRe = /^\s*(\S+)\s+([\d.]+)\s+#*/gm;
      let m;
      while ((m = lineRe.exec(out)) !== null) {
        attention.push({ token: m[1], weight: parseFloat(m[2]) });
      }
      res.json({
        source: "model",
        negative: parseFloat(negMatch[1]),
        positive: parseFloat(posMatch[1]),
        attention: attention.slice(0, 40),
      });
    });
  } else {
    const { positive, negative } = heuristicPredict(text);
    res.json({ source: "heuristic", positive, negative, attention: [] });
  }
});

app.listen(PORT, () => {
  console.log(`\nOnDeBERTa dashboard running: http://localhost:${PORT}`);
  console.log(modelIsAvailable()
    ? `Live model detected at ${CKPT_DIR} — predictions use your trained model.`
    : `No trained model found at ${CKPT_DIR} — predictions use the built-in demo heuristic (see README.md to connect your model).`);
});
