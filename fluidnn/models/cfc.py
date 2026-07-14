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
    """Window-mode CfC equalizer.

    Bidirectional by default: a forward and a backward cell sweep the window and
    both are read out *at the center step*, mirroring the BiLSTM readout. (A
    final-state readout demonstrably fails here: the center symbol's information
    has to survive half a window of gated updates, and the network converges to
    the identity -- see docs/notes/spike_v1_diagnosis.md and spike v2 logs.)

    The cell also supports streaming (one step per new symbol, state carried
    forward), the low-latency real-time deployment mode -- see ``step()``.
    """

    def __init__(
        self,
        window_len: int,
        hidden: int = 24,
        backbone_units: int = 0,
        bidirectional: bool = True,
        in_channels: int = 2,
    ):
        super().__init__()
        self.window_len = window_len
        self.hidden = hidden
        self.bidirectional = bidirectional
        self.cell = CfCCell(in_channels, hidden_size=hidden, backbone_units=backbone_units)
        if bidirectional:
            self.cell_bw = CfCCell(in_channels, hidden_size=hidden, backbone_units=backbone_units)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center = self.window_len // 2
        h = x.new_zeros(x.shape[0], self.hidden)
        for t in range(center + 1):  # forward sweep up to and including the center
            h = self.cell(x[:, t, :], h)
        if self.bidirectional:
            hb = x.new_zeros(x.shape[0], self.hidden)
            for t in range(x.shape[1] - 1, center - 1, -1):  # backward sweep to center
                hb = self.cell_bw(x[:, t, :], hb)
            h = torch.cat([h, hb], dim=-1)
        return x[:, center, :2] + self.head(h)

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Streaming mode: one cell update per incoming symbol, O(1) per symbol.

        The residual connection uses the current input, so in streaming mode the
        prediction corrects the symbol that just arrived. Streaming uses the
        forward cell only; a bidirectional model's head expects both states, so
        streaming applies to unidirectional models.
        """
        if self.bidirectional:
            raise RuntimeError("streaming requires bidirectional=False")
        h = self.cell(x_t, h)
        return x_t[:, :2] + self.head(h), h

    def macs_per_symbol(self) -> int:
        center = self.window_len // 2
        steps = center + 1  # forward sweep
        if self.bidirectional:
            steps += self.window_len - center  # backward sweep down to the center
        return (
            steps * self.cell.macs_per_step()
            + self.head.in_features * self.head.out_features
        )

    def macs_per_symbol_streaming(self) -> int:
        return self.cell.macs_per_step() + self.head.in_features * self.head.out_features


class StreamingCfCEqualizer(nn.Module):
    """Causal streaming CfC: one cell update per received symbol, state carried.

    The model consumes the symbol stream in order and, at each step, outputs the
    equalized symbol received ``delay`` steps earlier -- its decision latency,
    which gives it ``delay`` symbols of lookahead context relative to the symbol
    it corrects. There is no window and no re-computation: the per-symbol cost
    is exactly one cell update plus the head (see ``macs_per_symbol()``).

    Input follows ``make_stream_chunks``: chunks of ``warmup + L`` steps, the
    first ``warmup`` steps only prime the state and produce no output.

    With ``tapped=True`` (default) the head reads [h_t, h_{t-delay}]: the state
    captured the moment the target symbol arrived (a one-step path from that
    input, which is what makes training escape the identity -- a head reading
    only h_t leaves the target's information ``delay`` gated updates old, the
    exact failure mode measured for final-state window readouts) plus the
    current state carrying ``delay`` symbols of lookahead. In hardware this is
    a FIFO of ``delay`` states; the per-symbol cost stays one cell update +
    head.
    """

    def __init__(
        self,
        hidden: int = 32,
        backbone_units: int = 0,
        in_channels: int = 2,
        delay: int = 8,
        warmup: int = 32,
        tapped: bool = True,
    ):
        super().__init__()
        if delay > warmup:
            raise ValueError("delay must be <= warmup so every output has an input")
        self.hidden = hidden
        self.delay = delay
        self.warmup = warmup
        self.tapped = tapped
        self.cell = CfCCell(in_channels, hidden_size=hidden, backbone_units=backbone_units)
        self.head = nn.Linear(hidden * (2 if tapped else 1), 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, warmup+L, C) -> (B, L, 2); output j equalizes input step warmup+j-delay."""
        h = x.new_zeros(x.shape[0], self.hidden)
        history: list[torch.Tensor] = []
        outs = []
        for s in range(x.shape[1]):
            h = self.cell(x[:, s, :], h)
            history.append(h)
            if s >= self.warmup:
                z = torch.cat([h, history[s - self.delay]], dim=-1) if self.tapped else h
                outs.append(x[:, s - self.delay, :2] + self.head(z))
        return torch.stack(outs, dim=1)

    def macs_per_symbol(self) -> int:
        return self.cell.macs_per_step() + self.head.in_features * self.head.out_features
