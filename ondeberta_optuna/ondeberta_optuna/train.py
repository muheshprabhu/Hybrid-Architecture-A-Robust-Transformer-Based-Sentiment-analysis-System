"""Final training on the full 70/20/10 split (optionally with Optuna's best params), then test evaluation.

  python train.py                                   # defaults
  python train.py --params outputs/optuna/best_params.json
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ondeberta.data import EncodedDataset, load_imdb_splits, make_collate
from ondeberta.engine import HParams, build_model, evaluate, fit, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None, help="best_params.json from tune.py")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--adv_epsilon", type=float, default=None, help="0 disables adversarial training")
    ap.add_argument("--out", default="outputs/final")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    hp = HParams(encoder_name=args.encoder, epochs=args.epochs, max_len=args.max_len, seed=args.seed)
    if args.params:
        for k, v in json.load(open(args.params)).items():
            setattr(hp, k, v)
        hp.epochs = args.epochs
    if args.adv_epsilon is not None:
        hp.adv_epsilon = args.adv_epsilon

    tok = AutoTokenizer.from_pretrained(hp.encoder_name)
    s = load_imdb_splits(args.seed, args.csv)
    print({k: len(v[0]) for k, v in s.items()})
    ds = {k: EncodedDataset(*v, tok, hp.max_len) for k, v in s.items()}
    collate = make_collate(tok.pad_token_id)

    best_val, state = fit(hp, ds["train"], ds["val"], collate)
    print("best val:", best_val)

    device = get_device()
    model = build_model(hp).to(device)
    model.load_state_dict(state)
    test = evaluate(model, DataLoader(ds["test"], batch_size=hp.batch_size * 2, collate_fn=collate), device)
    print("TEST:", {k: round(v, 4) for k, v in test.items()})

    torch.save({"state_dict": state, "hparams": hp.to_dict()}, f"{args.out}/model.pt")
    tok.save_pretrained(f"{args.out}/tokenizer")
    json.dump({"hparams": hp.to_dict(), "val": best_val, "test": test}, open(f"{args.out}/results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
