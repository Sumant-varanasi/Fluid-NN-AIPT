"""Feed-forward baseline: the window is flattened and mapped to the center symbol.

All equalizers here output a *residual*: the network predicts the correction to
the received center symbol rather than the clean symbol from scratch. The
identity part of the mapping (received ~ transmitted) is free, so learning only
has to model the distortion.
"""

from __future__ import annotations

import torch
from torch import nn


class MLPEqualizer(nn.Module):
    def __init__(self, window_len: int, hidden: tuple[int, ...] = (128, 64), in_channels: int = 2):
        super().__init__()
        self.window_len = window_len
        dims = [in_channels * window_len, *hidden, 2]
        layers: list[nn.Module] = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(d_in, d_out))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.window_len // 2, :2] + self.net(x.flatten(1))

    def macs_per_symbol(self) -> int:
        return sum(
            m.in_features * m.out_features for m in self.net if isinstance(m, nn.Linear)
        )
