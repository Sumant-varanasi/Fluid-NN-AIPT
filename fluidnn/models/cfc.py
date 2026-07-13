"""Closed-form Continuous-time (CfC) equalizer.

Implements the gated CfC cell of Hasani, Lechner et al., "Closed-form
continuous-time neural networks" (Nature Machine Intelligence, 2022), following
the reference `ncps` formulation. The hidden state obeys liquid time-constant
dynamics whose ODE solution is replaced by an explicit closed-form expression --
no numerical solver in the loop:

    z        = backbone([x_t, h])            (optional shared trunk)
    f1       = tanh(W1 z)                    candidate A
    f2       = tanh(W2 z)                    candidate B
    g        = sigmoid(Wa z * dt + Wb z)     learned, input-dependent time gate
    h_new    = (1 - g) * f1 + g * f2

The gate g plays the role of the closed-form solution's time-dependent
interpolation sigma(-f(x,h) * dt): the effective time constant is a function of
the input and state, which is what lets the cell adapt its dynamics to the
signal -- the "liquid" property.

The cell is written from scratch (rather than importing `ncps`) so that every
multiply is visible for hardware-oriented MAC accounting.
"""

from __future__ import annotations

import torch
from torch import nn


class CfCCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, backbone_units: int = 0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.backbone_units = backbone_units
        cat = input_size + hidden_size
        if backbone_units > 0:
            self.backbone = nn.Sequential(nn.Linear(cat, backbone_units), nn.Tanh())
            trunk = backbone_units
        else:
            self.backbone = None
            trunk = cat
        self.ff1 = nn.Linear(trunk, hidden_size)
        self.ff2 = nn.Linear(trunk, hidden_size)
        self.time_a = nn.Linear(trunk, hidden_size)
        self.time_b = nn.Linear(trunk, hidden_size)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        z = torch.cat([x_t, h], dim=-1)
        if self.backbone is not None:
            z = self.backbone(z)
        f1 = torch.tanh(self.ff1(z))
        f2 = torch.tanh(self.ff2(z))
        gate = torch.sigmoid(self.time_a(z) * dt + self.time_b(z))
        return (1.0 - gate) * f1 + gate * f2

    def macs_per_step(self) -> int:
        macs = 0
        if self.backbone is not None:
            macs += self.backbone[0].in_features * self.backbone[0].out_features
        for lin in (self.ff1, self.ff2, self.time_a, self.time_b):
            macs += lin.in_features * lin.out_features
        return macs


class CfCEqualizer(nn.Module):
    """Window-mode CfC: sweep the received window once, read out the final state.

    This is the apples-to-apples counterpart of the window-based baselines. The
    cell also supports streaming (one step per new symbol, state carried
    forward), which is the low-latency real-time deployment mode -- see
    ``step()``.
    """

    def __init__(self, window_len: int, hidden: int = 24, backbone_units: int = 0):
        super().__init__()
        self.window_len = window_len
        self.hidden = hidden
        self.cell = CfCCell(input_size=2, hidden_size=hidden, backbone_units=backbone_units)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.new_zeros(x.shape[0], self.hidden)
        for t in range(x.shape[1]):
            h = self.cell(x[:, t, :], h)
        return x[:, self.window_len // 2, :] + self.head(h)

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Streaming mode: one cell update per incoming symbol, O(1) per symbol.

        The residual connection uses the current input, so in streaming mode the
        prediction corrects the symbol that just arrived.
        """
        h = self.cell(x_t, h)
        return x_t + self.head(h), h

    def macs_per_symbol(self) -> int:
        return (
            self.window_len * self.cell.macs_per_step()
            + self.head.in_features * self.head.out_features
        )

    def macs_per_symbol_streaming(self) -> int:
        return self.cell.macs_per_step() + self.head.in_features * self.head.out_features
