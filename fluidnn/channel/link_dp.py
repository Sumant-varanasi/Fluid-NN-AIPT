"""End-to-end dual-polarization coherent link with RSOP drift.

Pipeline: two independent QAM streams -> RRC -> Manakov propagation -> EDFA/ASE
-> time-varying polarization rotation (RSOP) -> CDC -> matched filter ->
block-wise genie 2x2 demux -> per-pol genie CPE.

The RSOP drift is the *time-varying* impairment: with ``rsop_drift_deg_per_ksym``
> 0 the channel rotates continuously during the sequence, and whatever rotation
accumulates within one demux block is left for the equalizer under test.
Launch power is the total across both polarizations.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from fluidnn.channel.manakov import (
    apply_rsop,
    genie_block_demux,
    propagate_link_manakov,
    rsop_angles,
)
from fluidnn.channel.modulation import QAM
from fluidnn.channel.pulse import filt_circular, rrc_taps, upsample
from fluidnn.channel.receiver import cdc, cpe_sliding
from fluidnn.channel.ssfm import FiberParams


@dataclass
class DPLinkConfig:
    mod_order: int = 16
    n_symbols: int = 2**16  # per polarization
    symbol_rate: float = 32e9
    sps: int = 4
    rolloff: float = 0.1
    rrc_span: int = 64
    launch_power_dbm: float = 3.0  # total, both polarizations
    n_spans: int = 12
    steps_per_span: int = 50
    nf_db: float = 4.5
    ase: bool = True
    rsop_theta0_deg: float = 20.0
    rsop_drift_deg_per_ksym: float = 0.0  # 0 = static rotation
    demux_block: int = 256
    cpe_window: int | None = 32
    fiber: FiberParams = field(default_factory=FiberParams)
    seed: int = 0

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["fiber"] = dataclasses.asdict(self.fiber)
        return d


def simulate_dp_link(cfg: DPLinkConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    qam = QAM(cfg.mod_order)
    fs = cfg.symbol_rate * cfg.sps

    tx_symbols = np.empty((2, cfg.n_symbols), dtype=complex)
    tx_bits = []
    for p in range(2):
        tx_symbols[p], bits = qam.random_symbols(cfg.n_symbols, rng)
        tx_bits.append(bits)
    tx_bits = np.stack(tx_bits)

    taps = rrc_taps(cfg.sps, cfg.rrc_span, cfg.rolloff)
    waveform = np.stack([filt_circular(upsample(tx_symbols[p], cfg.sps), taps) for p in range(2)])

    launch_w = 1e-3 * 10 ** (cfg.launch_power_dbm / 10)
    total_power = np.mean(np.abs(waveform[0]) ** 2 + np.abs(waveform[1]) ** 2)
    waveform *= np.sqrt(launch_w / total_power)

    rx_wave = propagate_link_manakov(
        waveform, fs, cfg.fiber, cfg.n_spans, cfg.steps_per_span, cfg.nf_db,
        rng if cfg.ase else None,
    )

    theta = rsop_angles(
        rx_wave.shape[1], cfg.rsop_drift_deg_per_ksym, cfg.sps, cfg.rsop_theta0_deg
    )
    rx_wave = apply_rsop(rx_wave, theta)

    total_length_m = cfg.n_spans * cfg.fiber.length_km * 1e3
    rx_symbols = np.empty_like(tx_symbols)
    for p in range(2):
        w = cdc(rx_wave[p], fs, cfg.fiber.beta2_s2_m, total_length_m)
        rx_symbols[p] = filt_circular(w, taps)[:: cfg.sps]

    rx_symbols = genie_block_demux(rx_symbols, tx_symbols, cfg.demux_block)
    if cfg.cpe_window is not None:
        for p in range(2):
            rx_symbols[p] = cpe_sliding(rx_symbols[p], tx_symbols[p], cfg.cpe_window)

    return {
        "tx_bits": tx_bits,        # (2, n_bits)
        "tx_symbols": tx_symbols,  # (2, N)
        "rx_symbols": rx_symbols,  # (2, N)
        "qam": qam,
        "config": cfg,
    }


def dp_report(rx: np.ndarray, tx_symbols: np.ndarray, tx_bits: np.ndarray, qam: QAM) -> dict:
    """Metrics averaged over both polarizations (bits concatenated)."""
    from fluidnn.metrics import equalizer_report

    per_pol = [equalizer_report(rx[p], tx_symbols[p], tx_bits[p], qam) for p in range(2)]
    both = equalizer_report(
        np.concatenate([rx[0], rx[1]]),
        np.concatenate([tx_symbols[0], tx_symbols[1]]),
        np.concatenate([tx_bits[0], tx_bits[1]]),
        qam,
    )
    both["per_pol_q_db"] = [per_pol[0]["q_db"], per_pol[1]["q_db"]]
    return both
