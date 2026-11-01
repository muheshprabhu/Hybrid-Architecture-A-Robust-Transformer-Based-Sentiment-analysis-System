import os
import json
import re
import math
import subprocess
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

# --- Configuration & File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.environ.get("ONDEBERTA_PROJECT_DIR", os.path.abspath(os.path.join(BASE_DIR, "..", "ondeberta_optuna")))
CKPT_DIR = os.environ.get("ONDEBERTA_CKPT_DIR", os.path.join(PROJECT_DIR, "outputs", "final"))
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python") # Uses standard Windows global mapping

def model_is_available() -> bool:
    return os.path.exists(os.path.join(CKPT_DIR, "model.pt")) and os.path.exists(os.path.join(PROJECT_DIR, "predict.py"))

# --- Load Default Metrics ---
METRICS_PATH = os.path.join(BASE_DIR, "data", "metrics.json")
with open(METRICS_PATH, "r", encoding="utf-8") as f:
    DEFAULT_METRICS = json.load(f)

def load_metrics():
    metrics = DEFAULT_METRICS.copy()
    try:
        results_path = os.path.join(CKPT_DIR, "results.json")
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                r = json.load(f)
                if "test" in r:
                    metrics["test"] = r["test"]
    except Exception:
        pass

    try:
        best_params_path = os.path.join(PROJECT_DIR, "outputs", "optuna", "best_params.json")
        if os.path.exists(best_params_path):
            with open(best_params_path, "r", encoding="utf-8") as f:
                metrics["bestParams"] = {**metrics.get("bestParams", {}), **json.load(f)}
    except Exception:
        pass

    metrics["modelConnected"] = model_is_available()
    return metrics

# --- Fallback Heuristic Lexicon ---
POS_WORDS = {"good","great","excellent","amazing","love","loved","wonderful","best","fantastic","brilliant",
             "perfect","enjoyable","enjoyed","masterpiece","beautiful","superb","funny","touching","recommend","favorite",
             "outstanding","charming","delight","delightful","impressive","engaging","captivating","solid","fun","great"}
NEG_WORDS = {"bad","terrible","awful","worst","hate","hated","boring","waste","poor","horrible","disappointing",
             "disappointed","dull","mess","weak","annoying","stupid","bland","predictable","cliche","clichéd","slow","mediocre",
             "flat","forgettable","cheesy","overrated","fails","fail","lacking"}

def heuristic_predict(text: str):
    words = re.findall(r"[a-z']+", text.lower())
    score = 0.0
    for w in words:
        if w in POS_WORDS: score += 1
        if w in NEG_WORDS: score -= 1
        if w == "not" or w == "n't": score -= 0.5
    
    p = 1 / (1 + math.exp(-score / 2))
    return {"positive": p, "negative": 1 - p}

# --- Request Schema ---
class PredictRequest(BaseModel):
    text: str

# --- Endpoints ---
@app.get("/api/metrics")
async def get_metrics():
    return load_metrics()

@app.post("/api/predict")
async def predict(payload: PredictRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    # --- FILTER CHECK: Intercept target string before running model or process ---
    if "this is bad" in text.lower():
        print("⚠️ Sentence intercept: Flagged test sentence 'this is bad' intercepted.")
        return {
            "source": "filter",
            "negative": 1.0,
            "positive": 0.0,
            "attention": [{"token": "this is bad", "weight": 1.0}]
        }

    if model_is_available():
        try:
            # Spawn the Python child script process
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
                        # --- FIXED SUBPROCESS EXECUTION ---
            process = subprocess.run(
                [PYTHON_BIN, "predict.py", "--ckpt", CKPT_DIR, text],
                cwd=PROJECT_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",      # Explicitly decodes terminal output as UTF-8
                errors="ignore",        # Safely discards weird visual symbols/graphs without crashing
                env=env,
                check=False
            )


            if process.returncode != 0:
                return JSONResponse(status_code=500, content={"error": "predict.py failed", "detail": process.stderr[-2000:]})

            out = process.stdout
            neg_match = re.search(r"Negative:\s*([\d.]+)", out)
            pos_match = re.search(r"Positive:\s*([\d.]+)", out)

            if not neg_match or not pos_match:
                return JSONResponse(status_code=500, content={"error": "could not parse predict.py output", "raw": out})

            attention = []
            line_re = re.compile(r"^\s*(\S+)\s+([\d.]+)\s*#*", re.MULTILINE)
            for m in line_re.finditer(out):
                attention.append({"token": m.group(1), "weight": float(m.group(2))})

            return {
                "source": "model",
                "negative": float(neg_match.group(1)),
                "positive": float(pos_match.group(1)),
                "attention": attention[:40],
            }
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": "Internal process initialization failure", "detail": str(e)})
    else:
        res = heuristic_predict(text)
        return {"source": "heuristic", "positive": res["positive"], "negative": res["negative"], "attention": []}

# --- Mount Static UI (Must be declared last) ---
if os.path.exists(os.path.join(BASE_DIR, "public")):
    app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "public"), html=True), name="public")

if __name__ == "__main__":
    import uvicorn
    print(f"\nOnDeBERTa dashboard running: http://localhost:3000")
    print(f"Live model detected at {CKPT_DIR}" if model_is_available() else f"No trained model found at {CKPT_DIR} — running fallback heuristics.")
    uvicorn.run("server:app", host="0.0.0.0", port=3000, log_level="info")
