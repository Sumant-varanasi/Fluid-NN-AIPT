"""End-to-end coherent link simulation: bits -> QAM -> RRC -> SSFM link -> Rx DSP.

Output symbols are ready for an equalizer: chromatic dispersion is compensated,
matched-filtered, symbol-spaced, and corrected by a single data-aided complex
scalar. What remains is nonlinear signal-signal/signal-noise distortion plus ASE
-- exactly what the neural equalizers are asked to undo.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from fluidnn.channel.modulation import QAM
from fluidnn.channel.pulse import filt_circular, rrc_taps, upsample
from fluidnn.channel.receiver import cdc, cpe_sliding, ls_correct, matched_filter_downsample
from fluidnn.channel.ssfm import FiberParams, propagate_link


@dataclass
class LinkConfig:
    mod_order: int = 16
    n_symbols: int = 2**16
    symbol_rate: float = 32e9  # Bd
    sps: int = 4  # simulation samples per symbol
    rolloff: float = 0.1
    rrc_span: int = 64  # RRC length in symbols (long enough that truncation ISI is negligible)
    launch_power_dbm: float = 2.0
    n_spans: int = 12
    steps_per_span: int = 50
    nf_db: float = 4.5
    ase: bool = True
    cpe_window: int | None = 32  # genie sliding-window CPE; None disables
    fiber: FiberParams = field(default_factory=FiberParams)
    seed: int = 0

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["fiber"] = dataclasses.asdict(self.fiber)
        return d


def simulate_link(cfg: LinkConfig) -> dict:
    """Run one end-to-end simulation; returns tx/rx symbol-rate sequences."""
    rng = np.random.default_rng(cfg.seed)
    qam = QAM(cfg.mod_order)
    fs = cfg.symbol_rate * cfg.sps

    tx_symbols, tx_bits = qam.random_symbols(cfg.n_symbols, rng)

    taps = rrc_taps(cfg.sps, cfg.rrc_span, cfg.rolloff)
    waveform = filt_circular(upsample(tx_symbols, cfg.sps), taps)

    launch_w = 1e-3 * 10 ** (cfg.launch_power_dbm / 10)
    waveform *= np.sqrt(launch_w / np.mean(np.abs(waveform) ** 2))

    rx_waveform = propagate_link(
        waveform,
        fs,
        cfg.fiber,
        cfg.n_spans,
        cfg.steps_per_span,
        cfg.nf_db,
        rng if cfg.ase else None,
    )

    total_length_m = cfg.n_spans * cfg.fiber.length_km * 1e3
    rx_waveform = cdc(rx_waveform, fs, cfg.fiber.beta2_s2_m, total_length_m)
    rx_symbols = matched_filter_downsample(rx_waveform, taps, cfg.sps)
    rx_symbols = ls_correct(rx_symbols, tx_symbols)
    if cfg.cpe_window is not None:
        rx_symbols = cpe_sliding(rx_symbols, tx_symbols, cfg.cpe_window)

    return {
        "tx_bits": tx_bits,
        "tx_symbols": tx_symbols,
        "rx_symbols": rx_symbols,
        "qam": qam,
        "config": cfg,
    }
