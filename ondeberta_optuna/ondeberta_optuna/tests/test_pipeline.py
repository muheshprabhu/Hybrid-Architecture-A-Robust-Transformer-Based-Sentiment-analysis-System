"""Offline end-to-end test: engine.fit + Optuna (TPE + MedianPruner) on fake data. Run: python tests/test_pipeline.py"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optuna
from transformers import DebertaV2Config
from ondeberta.data import make_collate
from ondeberta.engine import HParams, fit

cfg = DebertaV2Config(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                      intermediate_size=64, max_position_embeddings=64, relative_attention=True,
                      position_buckets=8, pos_att_type=["p2c", "c2p"], norm_rel_ebd="layer_norm")
random.seed(0)
def fake(n):
    return [([2] + [random.randint(5, 99) for _ in range(random.randint(5, 14))] + [3], random.randint(0, 1)) for _ in range(n)]
tr, va = fake(64), fake(32)
collate = make_collate(0)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    hp = HParams(lr=trial.suggest_float("lr", 1e-5, 5e-4, log=True), onlstm_hidden=16, chunk_size=4,
                 adv_epsilon=trial.suggest_float("adv_epsilon", 0.1, 1.0), batch_size=16, epochs=2, patience=2)
    return fit(hp, tr, va, collate, trial=trial, encoder_config=cfg, log=lambda *_: None)[0]["f1"]

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0))
study.optimize(objective, n_trials=3)
best, state = fit(HParams(onlstm_hidden=16, chunk_size=4, batch_size=16, epochs=1), tr, va, collate,
                  encoder_config=cfg, log=lambda *_: None)
assert state is not None
print("pipeline OK; best trial:", study.best_params, "states:", [t.state.name for t in study.trials])
