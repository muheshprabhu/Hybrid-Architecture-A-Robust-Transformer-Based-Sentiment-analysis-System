"""Ordered-Neuron LSTM (Shen et al., 2019, arXiv:1810.09536).

Implements the "ON-LSTM Decoder" block of the system diagram: K stacked
"Memory Layers", each one an ON-LSTM layer that uses the cumax() activation
(Eq. 4 of the OnDeBERTa paper) to build master forget / input gates that give
the neurons a hierarchical (high-level = long-term, low-level = short-term)
ordering.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cumsoftmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """cumax(x)_k = sum_{j<=k} softmax(x)_j"""
    return torch.cumsum(F.softmax(x, dim=dim), dim=dim)


class ONLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, chunk_size: int, dropconnect: float = 0.0):
        super().__init__()
        if hidden_size % chunk_size != 0:
            raise ValueError("hidden_size must be divisible by chunk_size")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_chunk = hidden_size // chunk_size

        # 4 standard gates (per neuron) + 2 master gates (per chunk)
        out = self.n_chunk * 2 + hidden_size * 4
        self.ih = nn.Linear(input_size, out)
        self.hh = nn.Linear(hidden_size, out)
        self.drop_weight = nn.Dropout(dropconnect) if dropconnect > 0 else nn.Identity()

    def init_state(self, batch: int, device, dtype):
        h = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(batch, self.n_chunk, self.chunk_size, device=device, dtype=dtype)
        return h, c

    def forward(self, x, state):
        hx, cx = state
        gates = self.ih(x) + self.hh(self.drop_weight(hx))

        cin, cforget = gates[:, : self.n_chunk * 2].chunk(2, 1)
        rest = gates[:, self.n_chunk * 2 :].reshape(-1, self.n_chunk * 4, self.chunk_size)
        outgate, cell, ingate, forgetgate = rest.chunk(4, 1)

        # master gates (ordered neurons)
        cin = (1.0 - cumsoftmax(cin)).unsqueeze(-1)  # master input gate (descending)
        cforget = cumsoftmax(cforget).unsqueeze(-1)  # master forget gate (ascending)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cell = torch.tanh(cell)
        outgate = torch.sigmoid(outgate)

        overlap = cforget * cin
        forgetgate = forgetgate * overlap + (cforget - overlap)
        ingate = ingate * overlap + (cin - overlap)

        cy = forgetgate * cx + ingate * cell
        hy = outgate * torch.tanh(cy)
        return hy.reshape(-1, self.hidden_size), cy


class ONLSTMLayer(nn.Module):
    """One ON-LSTM "Memory Layer" unrolled over time with padding-mask support."""

    def __init__(self, input_size, hidden_size, chunk_size, dropconnect=0.0):
        super().__init__()
        self.cell = ONLSTMCell(input_size, hidden_size, chunk_size, dropconnect)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (B, T, D)  mask: (B, T) with 1 for real tokens
        B, T, _ = x.shape
        h, c = self.cell.init_state(B, x.device, x.dtype)
        outs = []
        for t in range(T):
            h_new, c_new = self.cell(x[:, t], (h, c))
            m = mask[:, t].to(x.dtype).unsqueeze(1)
            h = m * h_new + (1 - m) * h  # freeze state on padding
            c = m.unsqueeze(-1) * c_new + (1 - m.unsqueeze(-1)) * c
            outs.append(h_new * m)
        return torch.stack(outs, dim=1)  # (B, T, H)


class ONLSTMDecoder(nn.Module):
    """K stacked memory layers (Memory Layer 1 ... Memory Layer K)."""

    def __init__(self, input_size, hidden_size=256, chunk_size=16, num_layers=1, dropout=0.1, dropconnect=0.0):
        super().__init__()
        sizes = [input_size] + [hidden_size] * num_layers
        self.layers = nn.ModuleList(
            [ONLSTMLayer(sizes[i], sizes[i + 1], chunk_size, dropconnect) for i in range(num_layers)]
        )
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden_size

    def forward(self, x, mask):
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            if i < len(self.layers) - 1:
                x = self.drop(x)
        return x
