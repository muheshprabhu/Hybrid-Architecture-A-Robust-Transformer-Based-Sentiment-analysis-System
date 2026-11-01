"""Optuna hyper-parameter optimisation (the "OPTUNA : HYPERPARAMETER OPTIMIZER" block).

  Objective Function -> validation F1 of a short OnDeBERTa training run
  Suggest Algo       -> TPESampler (Tree-structured Parzen Estimator)
  Prune Algo         -> MedianPruner (stops unpromising trials after each epoch)
  OPTUNA API         -> optuna.create_study / study.optimize

Usage:  python tune.py --n_trials 20 --subset 0.3 --epochs 3
"""
import argparse
import json
import os

import optuna
from transformers import AutoTokenizer

from ondeberta.data import EncodedDataset, load_imdb_splits, make_collate
from ondeberta.engine import HParams, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--csv", default=None, help="optional local IMDB csv (review,sentiment)")
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=3, help="max epochs per trial")
    ap.add_argument("--subset", type=float, default=0.3, help="fraction of train/val used per trial")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--out", default="outputs/optuna")
    ap.add_argument("--study_name", default="ondeberta")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.encoder)
    splits = load_imdb_splits(args.seed, args.csv)

    def sub(xy, frac):
        x, y = xy
        n = max(64, int(len(x) * frac))
        return x[:n], y[:n]  # splits are already shuffled/stratified

    tr = EncodedDataset(*sub(splits["train"], args.subset), tok, args.max_len)
    va = EncodedDataset(*sub(splits["val"], args.subset), tok, args.max_len)
    collate = make_collate(tok.pad_token_id)

    def objective(trial: optuna.Trial) -> float:  # "Objective Function"
        hp = HParams(
            encoder_name=args.encoder,
            lr=trial.suggest_float("lr", 1e-5, 5e-5, log=True),
            head_lr=trial.suggest_float("head_lr", 1e-4, 2e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
            warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.15),
            dropout=trial.suggest_float("dropout", 0.05, 0.3),
            onlstm_hidden=trial.suggest_categorical("onlstm_hidden", [128, 256, 384]),
            chunk_size=trial.suggest_categorical("chunk_size", [8, 16, 32]),
            onlstm_layers=trial.suggest_int("onlstm_layers", 1, 2),
            adv_epsilon=trial.suggest_float("adv_epsilon", 0.1, 1.5),
            batch_size=trial.suggest_categorical("batch_size", [16, 32]),
            max_len=args.max_len,
            epochs=args.epochs,
            patience=args.epochs,
            seed=args.seed,
        )
        best, _ = fit(hp, tr, va, collate, trial=trial)
        return best["f1"]

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),                       # Suggest Algo
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),  # Prune Algo
        storage=f"sqlite:///{args.out}/study.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = {"encoder_name": args.encoder, "max_len": args.max_len, **study.best_params}
    with open(f"{args.out}/best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    study.trials_dataframe().to_csv(f"{args.out}/trials.csv", index=False)
    print("Best val F1:", study.best_value)
    print("Best params:", json.dumps(best, indent=2))
    print(f"Saved to {args.out}/best_params.json  -> now run: python train.py --params {args.out}/best_params.json")


if __name__ == "__main__":
    main()
