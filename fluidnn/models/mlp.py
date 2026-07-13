"""Feed-forward baseline: the window is flattened and mapped to the center symbol."""

from __future__ import annotations

import torch
from torch import nn


class MLPEqualizer(nn.Module):
    def __init__(self, window_len: int, hidden: tuple[int, ...] = (128, 64)):
        super().__init__()
        self.window_len = window_len
        dims = [2 * window_len, *hidden, 2]
        layers: list[nn.Module] = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(d_in, d_out))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))

    def macs_per_symbol(self) -> int:
        return sum(
            m.in_features * m.out_features for m in self.net if isinstance(m, nn.Linear)
        )
