"""Signal-quality metrics: BER, SER, EVM, Q-factor, and AWGN theory references."""

from __future__ import annotations

import numpy as np
from scipy.special import erfc, erfcinv

from fluidnn.channel.modulation import QAM


def ber(bits_hat: np.ndarray, bits_ref: np.ndarray) -> float:
    return float(np.mean(bits_hat != bits_ref))


def ser(symbols_hat: np.ndarray, symbols_ref: np.ndarray) -> float:
    return float(np.mean(~np.isclose(symbols_hat, symbols_ref, atol=1e-9)))


def evm_percent(rx: np.ndarray, tx: np.ndarray) -> float:
    return float(100 * np.sqrt(np.mean(np.abs(rx - tx) ** 2) / np.mean(np.abs(tx) ** 2)))


def q_factor_db(ber_value: float, n_bits: int | None = None) -> float:
    """Gaussian-equivalent Q-factor from BER: Q_dB = 20 log10(sqrt(2) erfcinv(2 BER)).

    If BER is 0 and ``n_bits`` is given, it is floored at 1/n_bits (a lower bound).
    """
    if ber_value <= 0:
        if n_bits is None:
            return float("inf")
        ber_value = 1.0 / n_bits
    return float(20 * np.log10(np.sqrt(2) * erfcinv(2 * ber_value)))


def equalizer_report(rx: np.ndarray, tx_symbols: np.ndarray, tx_bits: np.ndarray, qam: QAM) -> dict:
    """Hard-decision metrics of an equalized symbol sequence against ground truth."""
    bits_hat = qam.decide_bits(rx)
    b = ber(bits_hat, np.asarray(tx_bits))
    return {
        "ber": b,
        "q_db": q_factor_db(b, n_bits=len(tx_bits)),
        "evm_percent": evm_percent(rx, tx_symbols),
        "mse": float(np.mean(np.abs(rx - tx_symbols) ** 2)),
    }


# ----------------------------------------------------------------- AWGN theory
def theory_ser_qam_awgn(order: int, snr_db: float) -> float:
    """Exact SER of Gray square M-QAM with min-distance detection over AWGN."""
    side = int(round(np.sqrt(order)))
    snr = 10 ** (snr_db / 10)
    arg = np.sqrt(3 * snr / (order - 1))
    p_axis = 2 * (1 - 1 / side) * 0.5 * erfc(arg / np.sqrt(2))
    return float(1 - (1 - p_axis) ** 2)
