# OnDeBERTa Sentiment Dashboard

A local Node.js dashboard for the OnDeBERTa + Optuna project: a "try a review" demo
with a sentiment-reactive cursor trail, click bursts and background glow, plus the
real training/Optuna metrics from your run (test accuracy 92.02%, F1 92.07%, etc.).

## Run it

Requires Node.js 18+ (no Python needed for the dashboard-only mode).

```bash
npm install
npm start
```

Then open **http://localhost:3000**.

## What you get out of the box

- The metrics panels (test accuracy/precision/recall/F1, training curves, Optuna
  trial comparison, best hyperparameters) are pre-filled with the real numbers
  from your Kaggle run — no setup needed.
- The "Try a review" box works immediately using a small built-in keyword
  heuristic (clearly labeled "demo heuristic" in the UI) — it is **not** your
  trained model, just enough to make the animations reactive without any setup.

## Connecting your real trained model (optional)

To get actual predictions from `model.pt` instead of the demo heuristic:

1. Put your downloaded `ondeberta_optuna` project (the one with `predict.py`,
   `outputs/final/model.pt`, `outputs/final/tokenizer/`) next to this folder, so
   you have:
   ```
   some-folder/
     ondeberta_dashboard/   <- this project
     ondeberta_optuna/      <- your trained project
   ```
2. Make sure `python3` on your machine has `torch` and `transformers` installed
   (the same environment you used on Kaggle, or `pip install torch transformers
   --break-system-packages`).
3. Start the dashboard again: `npm start`. The terminal will print
   `Live model detected — predictions use your trained model.` if it found it.
   The "Try a review" result will then show `● model prediction` instead of
   `○ demo heuristic`, and will include the attention weights per token.

If your project lives somewhere else, set environment variables before starting:

```bash
ONDEBERTA_PROJECT_DIR=/path/to/ondeberta_optuna \
ONDEBERTA_CKPT_DIR=/path/to/ondeberta_optuna/outputs/final \
PYTHON_BIN=python3 \
npm start
```

Also drop a fresh `outputs/final/results.json` and `outputs/optuna/best_params.json`
in that project folder and the dashboard's stat cards will pick up those numbers
automatically on the next page load, instead of the baked-in defaults in
`data/metrics.json`.

## What's inside

| File | Purpose |
|---|---|
| `server.js` | Express server: serves the UI, `/api/metrics`, `/api/predict` |
| `data/metrics.json` | Baked-in default metrics (your actual run's results) |
| `public/index.html` | Page structure |
| `public/styles.css` | Dark UI, mood-reactive background layers, chart styling |
| `public/app.js` | Cursor trail/burst particle system, chart rendering, predict call |

No build step, no framework — plain Express + vanilla JS/CSS/SVG, so it just runs.
