"""Training / evaluation engine shared by train.py and tune.py."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .adversarial import FGM
from .model import OnDeBERTa


@dataclass
class HParams:
    encoder_name: str = "microsoft/deberta-v3-base"
    lr: float = 2e-5              # encoder learning rate
    head_lr: float = 1e-3         # ON-LSTM / pooling / FC learning rate
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    dropout: float = 0.1
    onlstm_hidden: int = 256
    chunk_size: int = 16
    onlstm_layers: int = 1
    adv_epsilon: float = 1.0      # 0 disables adversarial training
    batch_size: int = 32
    max_len: int = 256
    epochs: int = 5
    patience: int = 2
    grad_clip: float = 1.0
    seed: int = 42

    def to_dict(self):
        return asdict(self)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(hp: HParams, encoder_config=None):
    return OnDeBERTa(
        encoder_name=hp.encoder_name,
        onlstm_hidden=hp.onlstm_hidden,
        chunk_size=hp.chunk_size,
        onlstm_layers=hp.onlstm_layers,
        dropout=hp.dropout,
        encoder_config=encoder_config,
    )


def compute_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f1}


@torch.no_grad()
def evaluate(model, loader, device, amp=True):
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = model(b["input_ids"], b["attention_mask"], b["labels"])
        loss_sum += out["loss"].item() * len(b["labels"])
        n += len(b["labels"])
        ps += out["logits"].argmax(-1).tolist()
        ys += b["labels"].tolist()
    m = compute_metrics(ys, ps)
    m["loss"] = loss_sum / max(n, 1)
    return m


def fit(hp: HParams, train_ds, val_ds, collate, trial=None, encoder_config=None, log=print, amp=True):
    """Train with FGM adversarial training; returns (best_val_metrics, best_state_dict)."""
    set_seed(hp.seed)
    device = get_device()
    model = build_model(hp, encoder_config).to(device)
    #get model to memory
    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=hp.batch_size * 2, shuffle=False, collate_fn=collate, num_workers=2)
    #split the data to batches as we cant provide the entire 50k reviews
    no_decay = ("bias", "LayerNorm.weight", "layernorm", "norm")
    #in order not to decay the weights ie higher mean memorising and lower means learning
    head_ids = {id(p) for p in model.head_parameters()}
    #providing high values to remaining (other than deberta) because deberta already knows english. 
    groups = []
    for lr, is_head in ((hp.lr, False), (hp.head_lr, True)):
        for decay in (True, False):
            params = [
                p for n, p in model.named_parameters()
                if (id(p) in head_ids) == is_head and (not any(nd in n for nd in no_decay)) == decay
            ]
            if params:
                groups.append({"params": params, "lr": lr, "weight_decay": hp.weight_decay if decay else 0.0})
    #checks parameters and if they are in head_ids and if they are not in no_decay then it adds them to the groups list with the learning rate and weight decay values.
    optim = torch.optim.AdamW(groups)
    #changes the weights by line 120, 131, 134
    total = len(train_loader) * hp.epochs
    #no of train batch * epoch
    sched = get_linear_schedule_with_warmup(optim, int(hp.warmup_ratio * total), total)
    #to train with safe and fast and precise manner. It warms up the learning rate and then decays it linearly to zero over the course of training.
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    fgm = FGM(model, epsilon=hp.adv_epsilon) if hp.adv_epsilon > 0 else None

    best, best_state, bad = None, None, 0
    for epoch in range(hp.epochs):
        model.train()
        run = 0.0
        for step, b in enumerate(train_loader):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(b["input_ids"], b["attention_mask"], b["labels"])["loss"]
            scaler.scale(loss).backward()

            if fgm is not None:  # adversarial step
                fgm.attack()
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    loss_adv = model(b["input_ids"], b["attention_mask"], b["labels"])["loss"]
                scaler.scale(loss_adv).backward()
                fgm.restore()

            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            scaler.step(optim)
            scaler.update()
            sched.step()
            optim.zero_grad(set_to_none=True)
            run += loss.item()
            if step % 100 == 0:
                log(f"epoch {epoch+1} step {step}/{len(train_loader)} loss {loss.item():.4f}")

        m = evaluate(model, val_loader, device, amp)
        log(f"[epoch {epoch+1}] train_loss={run/len(train_loader):.4f} val: " +
            " ".join(f"{k}={v:.4f}" for k, v in m.items()))

        if trial is not None:  # Optuna: report intermediate value, allow pruning
            import optuna
            trial.report(m["f1"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if best is None or m["f1"] > best["f1"]:
            best, bad = m, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}) \
                if trial is None else None
        else:
            bad += 1
            if bad >= hp.patience:
                log("early stopping")
                break
    return best, best_state
