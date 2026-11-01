"""Offline smoke test: tiny random DeBERTa, no downloads. Run: python tests/test_smoke.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import DebertaV2Config

from ondeberta.adversarial import FGM
from ondeberta.data import clean_text
from ondeberta.model import OnDeBERTa
from ondeberta.onlstm import cumsoftmax


def main():
    assert clean_text("<br />Great&amp;fine http://x.com  movie") == "Great&fine movie"
    x = cumsoftmax(torch.randn(3, 8))
    assert torch.allclose(x[:, -1], torch.ones(3)) and (x.diff(dim=-1) >= 0).all()

    cfg = DebertaV2Config(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                          intermediate_size=64, max_position_embeddings=64, relative_attention=True,
                          position_buckets=8, pos_att_type=["p2c", "c2p"], norm_rel_ebd="layer_norm")
    model = OnDeBERTa(onlstm_hidden=16, chunk_size=4, onlstm_layers=2, encoder_config=cfg)
    ids = torch.randint(1, 100, (4, 12))
    mask = torch.ones(4, 12, dtype=torch.long)
    mask[0, 8:] = 0
    labels = torch.tensor([0, 1, 0, 1])

    out = model(ids, mask, labels, return_attention=True)
    assert out["logits"].shape == (4, 2)
    assert torch.allclose(out["attention"].sum(-1), torch.ones(4), atol=1e-5)
    assert out["attention"][0, 8:].sum() == 0  # padding gets no attention

    fgm = FGM(model, 0.5)
    out["loss"].backward()
    w0 = model.encoder.embeddings.word_embeddings.weight.data.clone()
    fgm.attack()
    assert not torch.equal(w0, model.encoder.embeddings.word_embeddings.weight.data)
    model(ids, mask, labels)["loss"].backward()
    fgm.restore()
    assert torch.equal(w0, model.encoder.embeddings.word_embeddings.weight.data)
    print("smoke test OK, loss =", float(out["loss"]))


if __name__ == "__main__":
    main()
