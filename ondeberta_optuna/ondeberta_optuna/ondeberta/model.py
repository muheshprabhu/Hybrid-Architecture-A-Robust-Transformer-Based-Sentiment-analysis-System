"""Hybrid model: DeBERTa encoder -> ON-LSTM decoder -> Attention pooling -> FC -> Softmax."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification  # noqa: F401

from .onlstm import ONLSTMDecoder


class AttentionPooling(nn.Module):
    """s = sum_t alpha_t z_t, alpha = softmax(v^T tanh(W z_t))  (Eq. 5)."""

    def __init__(self, dim: int, attn_dim: int | None = None):
        super().__init__()
        attn_dim = attn_dim or dim
        self.proj = nn.Linear(dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, z, mask):
        scores = self.v(torch.tanh(self.proj(z))).squeeze(-1)  # (B, T)
        scores = scores.float().masked_fill(mask == 0, float("-inf"))
        alpha = torch.softmax(scores, dim=-1).to(z.dtype)
        s = torch.bmm(alpha.unsqueeze(1), z).squeeze(1)  # (B, D)
        return s, alpha


class OnDeBERTa(nn.Module):
    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 2,
        onlstm_hidden: int = 256,
        chunk_size: int = 16,
        onlstm_layers: int = 1,
        dropout: float = 0.1,
        encoder_config=None,
    ):
        super().__init__()
        if encoder_config is not None:  # random-init (used by tests, no download)
            self.encoder = AutoModel.from_config(encoder_config)
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name)
            self.encoder.float()
        d = self.encoder.config.hidden_size
        self.enc_drop = nn.Dropout(dropout)
        self.decoder = ONLSTMDecoder(d, onlstm_hidden, chunk_size, onlstm_layers, dropout)
        self.pool = AttentionPooling(onlstm_hidden)
        # Fully connected layer (hidden + output), followed by softmax
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(onlstm_hidden, onlstm_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(onlstm_hidden // 2, num_labels),
        )

    def head_parameters(self):
        for m in (self.decoder, self.pool, self.classifier):
            yield from m.parameters()

    def forward(self, input_ids, attention_mask, labels=None, return_attention=False):
        H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        Z = self.decoder(self.enc_drop(H), attention_mask)
        s, alpha = self.pool(Z, attention_mask)
        logits = self.classifier(s)  # softmax applied in the loss / predict_proba
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits.float(), labels)
        if return_attention:
            out["attention"] = alpha
        return out

    @torch.no_grad()
    def predict_proba(self, input_ids, attention_mask):
        return torch.softmax(self.forward(input_ids, attention_mask)["logits"].float(), dim=-1)
