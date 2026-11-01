"""IMDB Movie Reviews: 50,000 reviews, 2 classes (Positive / Negative).

Split (matches the "Dataset Details" slide): 70 % train (35,000) /
20 % test (10,000) / 10 % validation (5,000), stratified.
"""
from __future__ import annotations

import html
import re
import unicodedata

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_WS = re.compile(r"\s+")


def clean_text(t: str) -> str:
    """Algorithm 1 - text cleaning: strip HTML/URLs, NFKC-normalise, squeeze whitespace."""
    t = html.unescape(t)
    t = _TAG.sub(" ", t)
    t = _URL.sub(" ", t)
    t = unicodedata.normalize("NFKC", t)
    return _WS.sub(" ", t).strip()


def load_imdb_splits(seed: int = 42, csv_path: str | None = None):
    """Returns dict(train/val/test -> (texts, labels)).

    Uses HuggingFace `datasets` ("imdb"), or a CSV with columns `review,sentiment`
    (the Kaggle 'IMDB Dataset of 50K Movie Reviews') if csv_path is given.
    """
    if csv_path:
        import pandas as pd

        df = pd.read_csv(csv_path)
        texts = df["review"].tolist()
        labels = (df["sentiment"].str.lower() == "positive").astype(int).tolist()
    else:
        from datasets import load_dataset

        ds = load_dataset("imdb")
        texts = list(ds["train"]["text"]) + list(ds["test"]["text"])
        labels = list(ds["train"]["label"]) + list(ds["test"]["label"])

    texts = [clean_text(t) for t in texts]
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(texts, labels, test_size=0.30, stratify=labels, random_state=seed)
    x_te, x_va, y_te, y_va = train_test_split(x_tmp, y_tmp, test_size=1 / 3, stratify=y_tmp, random_state=seed)
    return {"train": (x_tr, y_tr), "val": (x_va, y_va), "test": (x_te, y_te)}


class EncodedDataset(Dataset):
    """Pre-tokenised with the DeBERTa tokenizer ([CLS] ... [SEP])."""

    def __init__(self, texts, labels, tokenizer, max_len=256):
        enc = tokenizer(texts, truncation=True, max_length=max_len)
        self.ids = enc["input_ids"]
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def make_collate(pad_id: int):
    def collate(batch):
        L = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (ids, _) in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            mask[i, : len(ids)] = 1
        labels = torch.tensor([y for _, y in batch], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": mask, "labels": labels}

    return collate
