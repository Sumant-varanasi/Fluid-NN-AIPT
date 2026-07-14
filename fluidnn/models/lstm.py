"""Bidirectional LSTM equalizer, the strong discrete-time recurrent baseline.

Mirrors the window-based BiLSTM equalizers of the performance-vs-complexity
literature: run over the received window, read the center time step of both
directions, and regress the clean center symbol.
"""

from __future__ import annotations

import torch
from torch import nn


class BiLSTMEqualizer(nn.Module):
    def __init__(self, window_len: int, hidden: int = 32, in_channels: int = 2):
        super().__init__()
        self.window_len = window_len
        self.hidden = hidden
        self.lstm = nn.LSTM(
            input_size=in_channels, hidden_size=hidden, batch_first=True, bidirectional=True
        )
        self.head = nn.Linear(2 * hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (B, T, 2H)
        center = self.window_len // 2
        return x[:, center, :2] + self.head(out[:, center, :])

    def macs_per_symbol(self) -> int:
        in_size = self.lstm.input_size
        per_step_per_dir = 4 * self.hidden * (in_size + self.hidden)  # W_ih + W_hh, 4 gates
        recurrent = 2 * self.window_len * per_step_per_dir
        return recurrent + self.head.in_features * self.head.out_features


class StreamingLSTMEqualizer(nn.Module):
    """Causal streaming LSTM: the discrete-time counterpart of the streaming CfC.

    Same contract as ``StreamingCfCEqualizer``: consumes ``warmup + L`` chunks
    (see ``make_stream_chunks``), one LSTM step per symbol, tapped readout
    [h_t, h_{t-delay}], residual on the delayed input. O(1) per symbol.
    """

    def __init__(self, hidden: int = 32, in_channels: int = 2, delay: int = 8, warmup: int = 32):
        super().__init__()
        if delay > warmup:
            raise ValueError("delay must be <= warmup so every output has an input")
        self.hidden = hidden
        self.delay = delay
        self.warmup = warmup
        self.lstm = nn.LSTM(in_channels, hidden, batch_first=True)
        self.head = nn.Linear(2 * hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (B, T, H); out[:, s] is the state after step s
        idx = torch.arange(self.warmup, x.shape[1], device=x.device)
        z = torch.cat([out[:, idx], out[:, idx - self.delay]], dim=-1)
        return x[:, idx - self.delay, :2] + self.head(z)

    def macs_per_symbol(self) -> int:
        step = 4 * self.hidden * (self.lstm.input_size + self.hidden)
        return step + self.head.in_features * self.head.out_features
