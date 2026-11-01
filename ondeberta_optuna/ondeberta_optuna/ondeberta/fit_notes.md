# `fit()` notes (engine.py)

Line numbers below match `engine.py` **with your comments added** (fit is lines 85–164).

---

## 0. The big picture

`fit()` = **set up everything once → train for several epochs → keep the best version → return it.**

Analogy: a student preparing for an exam.

| Real life | In `fit()` |
|---|---|
| Student | model |
| Practice book (35,000 questions) | training data |
| Mock test (5,000 questions) | validation data |
| One full pass through the book | 1 epoch |
| Doing 32 questions at a time | 1 batch |
| Checking how wrong the answers are | loss |
| Working out what went wrong | backward (gradients) |
| Fixing their understanding | optimizer step |
| How big each fix is | learning rate |

```
PART 1  SETUP          lines 87–118   runs ONCE, no learning yet
PART 2  EPOCH LOOP     lines 119–121  repeat hp.epochs times (4 in our run)
PART 3  BATCH LOOP     lines 122–143  1094 batches per epoch → learn
PART 4  END OF EPOCH   lines 145–163  validate → Optuna → save best / early stop
PART 5  RETURN         line 164       best score + best weights
```

**The 4 most important lines:**
- **125**: forward (predict, get the loss)
- **126**: backward (find the mistakes)
- **129–133**: adversarial training (FGM)
- **137**: update the weights (the model actually learns here)

---

## 1. Inputs (line 85)

```python
def fit(hp, train_ds, val_ds, collate, trial=None, encoder_config=None, log=print, amp=True):
```

| Argument | Meaning |
|---|---|
| `hp` | all settings (lr, dropout, epochs, ...), from `HParams` |
| `train_ds` | training reviews (already tokenized by `EncodedDataset`) |
| `val_ds` | validation reviews |
| `collate` | function that pads each batch (`make_collate` in data.py) |
| `trial` | Optuna trial. Given in tune.py, **None** in train.py |
| `encoder_config` | only for the tests (tiny random DeBERTa, no download) |
| `log` | how to print (the tests pass a "print nothing" function) |
| `amp` | allow fp16 mixed precision on the GPU |

---

## 2. PART 1: Setup (lines 87–118), runs once

| Line | Code | What it does | Why |
|---|---|---|---|
| 87 | `set_seed(hp.seed)` | fixes all randomness (seed 42) | same result every run |
| 88 | `device = get_device()` | GPU if available, else CPU | |
| 89 | `model = build_model(hp, encoder_config).to(device)` | **creates the model** and moves it to the GPU | "pre-start": the model is ready but has learned nothing |
| 91 | `train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, ...)` | plan to cut the training data into batches of 32, **shuffled** | shuffling stops the model learning the order |
| 92 | `val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, ...)` | same for validation, **not shuffled**, batch ×2 | no gradients during validation, so a bigger batch fits |
| 94 | `no_decay = ("bias", "LayerNorm.weight", ...)` | names of parameters that get **no weight decay** | see section 6 |
| 96 | `head_ids = {id(p) for p in model.head_parameters()}` | "name tags" (memory IDs) of every **head** parameter | used to tell head and DeBERTa apart |
| 98 | `groups = []` | empty list of optimizer groups | |
| 99–106 | the two `for` loops | sort **all** parameters into **4 groups** | see section 5 |
| 108 | `optim = torch.optim.AdamW(groups)` | creates the **optimizer** for ALL weights (DeBERTa + head) | it changes the weights at line 137 |
| 110 | `total = len(train_loader) * hp.epochs` | total steps = 1094 × 4 = **4376** | the scheduler needs to know when training ends |
| 112 | `sched = get_linear_schedule_with_warmup(...)` | learning-rate plan: warmup, then decay | see section 7 |
| 114 | `use_amp = amp and device.type == "cuda"` | use fp16 only on a GPU | see section 8 |
| 115 | `scaler = torch.amp.GradScaler(...)` | protects tiny fp16 gradients | see section 8 |
| 116 | `fgm = FGM(model, epsilon=hp.adv_epsilon) if hp.adv_epsilon > 0 else None` | creates adversarial training (ε = 0.558). `None` if ε = 0 | see section 9 |
| 118 | `best, best_state, bad = None, None, 0` | start values: no best score, no saved weights, 0 bad epochs | used for "keep best" and early stopping |

> The `DataLoader` lines don't make the batches yet. They only store the **recipe**. Each batch is actually made at line 122, one at a time.

---

## 3. PART 2 + PART 3: The training loop (lines 119–143)

Two loops, one inside the other:

```
for each epoch (4 times):                 ← outer, line 119
    for each batch of 32 (1094 times):    ← inner, line 122
        learn from this batch
    check on validation
```

The model learns 1094 × 4 = **4376 times** in total.

### Start of each epoch

| Line | Code | Meaning |
|---|---|---|
| 119 | `for epoch in range(hp.epochs):` | **training starts here** |
| 120 | `model.train()` | training mode: **dropout ON** |
| 121 | `run = 0.0` | reset the total loss for this epoch |

### Each batch: 4 steps

**Step ① normal training**

| Line | Code | Meaning |
|---|---|---|
| 122 | `for step, b in enumerate(train_loader):` | take the next 32 reviews (`collate` pads them here) |
| 123 | `b = {k: v.to(device) ...}` | move the batch to the GPU |
| 124 | `with torch.autocast(..., enabled=use_amp):` | run the next line in fp16 (faster) |
| 125 | `loss = model(...)["loss"]` | **FORWARD**: DeBERTa → ON-LSTM → attention → FC → loss (how wrong) |
| 126 | `scaler.scale(loss).backward()` | **BACKWARD**: work out how each weight should change (gradients) |

**Step ② adversarial training (FGM)**

| Line | Code | Meaning |
|---|---|---|
| 128 | `if fgm is not None:` | only if adversarial training is on |
| 129 | `fgm.attack()` | slightly change the **word embeddings** in the "worst" direction (a harder input) |
| 131 | `loss_adv = model(...)["loss"]` | forward again on the harder input |
| 132 | `scaler.scale(loss_adv).backward()` | backward again. Gradients **ADD** to those from line 126 |
| 133 | `fgm.restore()` | put the original embeddings back |

**Step ③ update the weights**

| Line | Code | Meaning |
|---|---|---|
| 135 | `scaler.unscale_(optim)` | bring gradients back to their real size (undo the fp16 scaling) |
| 136 | `clip_grad_norm_(model.parameters(), 1.0)` | if gradients are too big, shrink them to 1.0 (no big jumps) |
| 137 | `scaler.step(optim)` | ★ **UPDATE THE WEIGHTS**: the model learns here. Skipped if a gradient is inf/NaN |
| 138 | `scaler.update()` | adjust the fp16 scale number for the next batch |
| 139 | `sched.step()` | move the learning rate one step along the plan |
| 140 | `optim.zero_grad(set_to_none=True)` | clear the gradients (otherwise they add up across batches) |

**Step ④ print progress**

| Line | Code | Meaning |
|---|---|---|
| 141 | `run += loss.item()` | add this batch's loss to the epoch total |
| 142–143 | `if step % 100 == 0: log(...)` | print the loss every 100 batches |

> **Note:** your comment on line 109 says "changes the weights by line 120, 131, 134". Those were the old line numbers. With your comments added, they are now **126 (backward), 137 (step), 140 (zero_grad)**.

---

## 4. PART 4: End of each epoch (lines 145–163)

| Line | Code | Meaning | Why |
|---|---|---|---|
| 145 | `m = evaluate(model, val_loader, device, amp)` | test on the 5,000 validation reviews → accuracy, precision, recall, F1, loss | data the model does NOT learn from |
| 146–147 | `log(...)` | print the epoch results | |
| 149 | `if trial is not None:` | **only in tune.py** (skipped in train.py) | |
| 151 | `trial.report(m["f1"], epoch)` | tell Optuna this epoch's F1 | |
| 152–153 | `if trial.should_prune(): raise optuna.TrialPruned()` | MedianPruner: worse than the median of earlier trials → stop now | saves time |
| 155 | `if best is None or m["f1"] > best["f1"]:` | first epoch, or a **better** F1? | F1 balances precision and recall |
| 156 | `best, bad = m, 0` | remember the score, reset the bad counter | |
| 157–158 | `best_state = copy.deepcopy(...) if trial is None else None` | **save a copy of the weights** (on the CPU). Not during tuning | deepcopy so it doesn't change later. Skipped in tuning to save memory |
| 159–160 | `else: bad += 1` | F1 did not improve → count a bad epoch | |
| 161–163 | `if bad >= hp.patience: break` | 2 bad epochs in a row → **early stopping** | avoids overfitting and wasted time |

### PART 5: Return (line 164)

```python
return best, best_state
```
→ goes back to train.py as `best_val, state`.

### Our real run (Kaggle, train.py)

```
epoch 1 → val F1 0.9060 → best → save
epoch 2 → val F1 0.9126 → best → save
epoch 3 → val F1 0.9184 → best → save   ← final model
epoch 4 → val F1 0.9184 → not higher (bad = 1)
→ return epoch 3's weights → test: accuracy 0.9202, F1 0.9207
```

---

## 5. The two `for` loops (lines 99–106): 4 boxes

**Goal:** put every parameter into the right box, because each box has its own rules.

Two yes/no questions:
1. Is it **DeBERTa** (pretrained) or the **head** (new: ON-LSTM, attention, FC)?
2. Is it a **normal weight** or a **bias / norm**?

```
                  normal weight     bias / norm
   DeBERTa          BOX 1             BOX 2
   Head             BOX 3             BOX 4
```

| Box | Learning rate | Weight decay |
|---|---|---|
| 1. DeBERTa, normal | 3.3e-5 (slow) | 0.0028 |
| 2. DeBERTa, bias/norm | 3.3e-5 (slow) | 0 |
| 3. Head, normal | 2.7e-4 (fast) | 0.0028 |
| 4. Head, bias/norm | 2.7e-4 (fast) | 0 |

- **Outer loop** picks the row (DeBERTa, then head).
- **Inner loop** picks the column (normal, then bias/norm).
- In each of the 4 rounds, the code looks at **all** parameters and keeps only those that pass **both** checks on line 103:
  - `(id(p) in head_ids) == is_head`: is it in the right part?
  - `(not any(nd in n for nd in no_decay)) == decay`: is it the right type?
- `n` = the parameter's name (used for the check), `p` = the actual numbers (what is kept).
- `params` = the list of parameters picked for the current box.

**ALL parameters still learn.** The boxes only decide **how** they change (fast or slow, with or without weight decay), not **whether** they change.

**Why two learning rates?** DeBERTa already knows English, so it gets small careful changes. The head knows nothing yet, so it gets bigger changes.

---

## 6. Weight decay and `no_decay`

- **Weight decay** gently pulls every weight toward 0 at each update.
- **Big** weights → the model reacts to tiny details → **memorizes** → bad on new reviews.
- **Small** weights → the model learns the main patterns → **generalizes**.
- Weights **don't become 0**. Learning pushes useful weights back, and they balance.
- `no_decay` (bias, LayerNorm): these **still learn**. Only the pull to 0 is switched off, because shrinking them doesn't help and LayerNorm should stay around 1.

---

## 7. Scheduler (`sched`, line 112)

```
lr
max ┤     ╱╲
    │    ╱  ╲
    │   ╱    ╲
  0 ┼──╱──────╲──► step
    0  594    4376
    warmup   decay
```

- **Warmup** (first 13.6% ≈ 594 steps): lr rises from 0 → protects pretrained DeBERTa while the head is still random.
- **Decay**: lr falls to 0 → careful fine-tuning at the end.
- Controls **all 4 groups**. It multiplies each group's own lr by the same factor (0 → 1 → 0), so the head is always about 8× DeBERTa.
- Called every batch at line 139.

---

## 8. AMP (`use_amp`) and `scaler`

- **AMP = Automatic Mixed Precision.** Calculations run in **fp16** (2 bytes, about 2× faster on a GPU), and weights stay in **fp32** (accurate).
- `use_amp` is True only on a GPU.
- **Problem:** fp16 can't hold very tiny numbers; small gradients become 0.
- **Scaler fix:** multiply the loss by a big number (line 126/132) → gradients are big enough → divide back (line 135) → update (line 137, skipped if inf/NaN) → adjust the scale (line 138).
- Unscale **before** clipping, so clipping sees the real gradient size.
- On Kaggle this caused the error `Attempting to unscale FP16 gradients`, fixed by `self.encoder.float()` in model.py.

---

## 9. FGM adversarial training: answer for the mentor

**Q: DeBERTa already has "adversarial", so why add another one?**

- **DeBERTa = Decoding-enhanced BERT with disentangled Attention.** The name and architecture have no adversarial part.
- The adversarial-style ideas linked to DeBERTa are:
  - **SiFT**: an optional fine-tuning method proposed in the DeBERTa paper. It is **not** part of the model; loading with `from_pretrained` doesn't run it.
  - **ELECTRA-style generator/discriminator** (DeBERTa-v3): used only during Microsoft's **pretraining**, already finished, and not real adversarial perturbation training.
- So during **our** fine-tuning on IMDB there is **no** adversarial training unless we add it → FGM.
- FGM also trains our **new head** (ON-LSTM, attention, FC), which never had any pretraining.
- It makes the model more robust to small input changes and acts as regularization.
- ε is tuned by Optuna (0.558). `--adv_epsilon 0` turns it off.

**Likely follow-up: "Does it help?"** Run the ablation and compare:
```bash
python train.py --params outputs/optuna/best_params.json --epochs 4 --max_len 128 --adv_epsilon 0
```

---

## 10. Where the Optuna params enter training: answer for the mentor

1. **tune.py** saves the best trial's params → `outputs/optuna/best_params.json` (tune.py lines 78–80).
2. **train.py** with `--params outputs/optuna/best_params.json`:
   ```python
   hp = HParams(...)                                   # defaults from engine.py
   if args.params:
       for k, v in json.load(open(args.params)).items():
           setattr(hp, k, v)                           # ← overwrite defaults with Optuna values
   ```
   (train.py lines 31–35)
3. That `hp` goes into `fit(hp, ...)` at train.py line 45.

- **Without `--params`** → the `HParams` defaults in engine.py are used (except encoder, epochs, max_len and seed, which come from the command line).
- Anything not in the JSON (patience, grad_clip, and in our run onlstm_layers and batch_size) keeps its default.
- In **tune.py**, Optuna's values go **directly** into `HParams(...)` inside `objective`. The JSON file is only the bridge from tune.py to train.py.

---

## 11. Call chain (who calls whom)

```
train.py / tune.py
 └── fit()                                 engine.py
      ├── build_model() → OnDeBERTa         model.py   (create)
      └── model(...) → OnDeBERTa.forward()  model.py   (lines 125, 131, and inside evaluate)
           ├── DeBERTa encoder              transformers library (not our code)
           ├── ONLSTMDecoder.forward()      onlstm.py
           │    └── ONLSTMLayer → ONLSTMCell → cumsoftmax
           ├── AttentionPooling.forward()   model.py
           └── classifier (FC) → loss
```

- `fit()` is **training code**, not DeBERTa code. DeBERTa itself lives in the HuggingFace `transformers` library.
- `fit()` never calls the ON-LSTM directly. It calls the model, and the model calls the ON-LSTM.

---

## 12. One-paragraph summary to say out loud

> "`fit()` first sets up the model, the data loaders, an AdamW optimizer with two learning rates (small for pretrained DeBERTa, larger for the new ON-LSTM head) split into decay and no-decay groups, a warmup scheduler, an fp16 grad scaler, and FGM. Then for each epoch and each batch, it does a normal forward and backward pass, then an FGM adversarial forward and backward pass, then unscales and clips the gradients and updates the weights. After each epoch it evaluates on validation, reports F1 to Optuna during tuning so bad trials can be pruned, saves the weights if F1 improved, and stops early if F1 doesn't improve for two epochs. It returns the best validation metrics and the best weights."
