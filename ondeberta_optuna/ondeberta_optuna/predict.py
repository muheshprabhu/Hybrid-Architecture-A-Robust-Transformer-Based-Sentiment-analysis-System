"""Predict sentiment for a review and show the attention-pooling weight of each token.

  python predict.py --ckpt outputs/final "There is a smile on his face, but he is not actually good with movie"
"""
import argparse

import torch
from transformers import AutoTokenizer

from ondeberta.data import clean_text
from ondeberta.engine import HParams, build_model, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/final")
    ap.add_argument("text")
    a = ap.parse_args()

    ck = torch.load(f"{a.ckpt}/model.pt", map_location="cpu")
    hp = HParams(**ck["hparams"])
    tok = AutoTokenizer.from_pretrained(f"{a.ckpt}/tokenizer")
    dev = get_device()
    model = build_model(hp).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    enc = tok(clean_text(a.text), truncation=True, max_length=hp.max_len, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"], return_attention=True)
    p = torch.softmax(out["logits"].float(), -1)[0].cpu()
    print(f"Negative: {p[0]:.3f}   Positive: {p[1]:.3f}  ->  {'Positive' if p[1] > p[0] else 'Negative'}")
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    for t, w in zip(toks, out["attention"][0].cpu().tolist()):
        print(f"{t:>15s}  {w:.3f}  {'#' * int(w * 60)}")


if __name__ == "__main__":
    main()
