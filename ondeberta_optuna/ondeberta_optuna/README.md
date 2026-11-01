# OnDeBERTa + Optuna: Hybrid Sentiment Classifier (IMDB)

Implements the **System Diagram** slide: 

| Diagram block | Code |
|---|---|
| Input review (e.g. "There is a smile on his face, but he is not actually good with movie") | `ondeberta/data.py` (`clean_text`, NFKC/HTML/URL cleaning) |
| DeBERTa Tokenizer (`[CLS] ... [SEP]`) | `AutoTokenizer` (microsoft/deberta-v3-base) in `train.py` / `tune.py` |
| DeBERTa Encoder (Transformer blocks 1..N) | `OnDeBERTa.encoder` in `ondeberta/model.py` |
| Adversarial Training | `ondeberta/adversarial.py` (FGM on word embeddings), used in `engine.fit` |
| ON-LSTM Decoder (Memory Layer 1..K) | `ondeberta/onlstm.py` (`cumax` master gates, K = `onlstm_layers`) |
| Attention Pooling (a1..an, sum) | `AttentionPooling` in `ondeberta/model.py` (Eq. 5) |
| Fully Connected Layer -> Softmax -> Positive / Negative | `OnDeBERTa.classifier`, softmax in loss / `predict_proba` |
| OPTUNA Hyperparameter Optimizer (Objective Function, Suggest Algo, Prune Algo, Optuna API) | `tune.py` (objective = val F1, `TPESampler`, `MedianPruner`, `optuna.create_study`) |

Dataset (per the "Dataset Details" slide): IMDB Movie Reviews, 50,000 reviews, 2 classes,
split 70/20/10 -> 35,000 train / 10,000 test / 5,000 validation (stratified).

## Run

```bash
pip install -r requirements.txt
python tests/test_smoke.py && python tests/test_pipeline.py   # offline sanity checks (no GPU/download)

python tune.py --n_trials 20 --subset 0.3 --epochs 3           # Optuna search -> outputs/optuna/best_params.json
python train.py --params outputs/optuna/best_params.json       # final training + TEST metrics
python predict.py --ckpt outputs/final "There is a smile on his face, but he is not actually good with movie"
```

* GPU strongly recommended (fp16 autocast is used automatically). 
* No internet for HF datasets? Download the Kaggle *IMDB Dataset of 50K Movie Reviews* CSV and pass `--csv IMDB\ Dataset.csv`.
* Ablation: `python train.py --adv_epsilon 0` turns adversarial training off.
* Searched hyper-parameters: encoder lr, head lr, weight decay, warmup, dropout, ON-LSTM hidden size / chunk size / layers, FGM epsilon, batch size.

## Notes
* Architecture follows OnDeBERTa (DeBERTa -> ON-LSTM -> attention pooling -> classifier); Optuna tuning and adversarial training are added per your diagram.
* Metrics reported: Accuracy, Precision, Recall, F1. I have not trained on the real dataset here, so no result numbers are claimed. Fill them into your slides after running `train.py` (`outputs/final/results.json`).
