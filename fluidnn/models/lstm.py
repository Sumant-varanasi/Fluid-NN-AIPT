"""Bidirectional LSTM equalizer, the strong discrete-time recurrent baseline.

Mirrors the window-based BiLSTM equalizers of the performance-vs-complexity
literature: run over the received window, read the center time step of both
directions, and regress the clean center symbol.
"""

from __future__ import annotations

import torch
from torch import nn


class BiLSTMEqualizer(nn.Module):
    def __init__(self, window_len: int, hidden: int = 32):
        super().__init__()
        self.window_len = window_len
        self.hidden = hidden
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (B, T, 2H)
        center = self.window_len // 2
        return x[:, center, :] + self.head(out[:, center, :])

    def macs_per_symbol(self) -> int:
        per_step_per_dir = 4 * self.hidden * (2 + self.hidden)  # W_ih + W_hh for 4 gates
        recurrent = 2 * self.window_len * per_step_per_dir
        return recurrent + self.head.in_features * self.head.out_features
