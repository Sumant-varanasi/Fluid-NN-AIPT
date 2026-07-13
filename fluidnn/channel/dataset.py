"""Turn symbol-rate sequences into supervised windows for equalizer training."""

from __future__ import annotations

import numpy as np


def make_windows(
    rx_symbols: np.ndarray, tx_symbols: np.ndarray, half_window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding window of ``2*half_window + 1`` received symbols per target symbol.

    The simulation chain is fully circular, so windows wrap around the sequence
    ends without edge artifacts. Returns (X, y): X complex (n, 2W+1), y complex (n,).
    """
    n = len(rx_symbols)
    offsets = np.arange(-half_window, half_window + 1)
    idx = (np.arange(n)[:, None] + offsets[None, :]) % n
    return rx_symbols[idx], tx_symbols.copy()


def to_real_features(x_complex: np.ndarray) -> np.ndarray:
    """(n, T) complex -> (n, T, 2) float32 with channels [real, imag]."""
    return np.stack([x_complex.real, x_complex.imag], axis=-1).astype(np.float32)
