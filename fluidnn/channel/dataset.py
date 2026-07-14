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


def make_stream_chunks(
    rx_symbols: np.ndarray,
    tx_symbols: np.ndarray,
    chunk_len: int = 256,
    warmup: int = 32,
    delay: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Chunk a symbol sequence for streaming (sequence-to-sequence) training.

    Chunk k covers received symbols [k*chunk_len - warmup, k*chunk_len + chunk_len)
    (circular). A streaming model consumes the chunk step by step; after the
    ``warmup`` steps its state is valid, and at step s it predicts the symbol
    received ``delay`` steps earlier (its decision latency). Targets are aligned
    accordingly: y[k, j] = tx at position k*chunk_len + j - delay.

    Returns (X complex (n_chunks, warmup+chunk_len), Y complex (n_chunks, chunk_len)).
    Concatenating all chunks' predictions covers every symbol exactly once.
    """
    n = len(rx_symbols)
    if n % chunk_len != 0:
        raise ValueError("chunk_len must divide the sequence length")
    starts = np.arange(0, n, chunk_len)
    x_idx = (starts[:, None] + np.arange(-warmup, chunk_len)[None, :]) % n
    y_idx = (starts[:, None] + np.arange(chunk_len)[None, :] - delay) % n
    return rx_symbols[x_idx], tx_symbols[y_idx]


def make_dp_windows(
    rx_symbols: np.ndarray, tx_symbols: np.ndarray, half_window: int, power_feature: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Joint dual-polarization windows.

    rx/tx: (2, N) complex. Returns (X, Y): X float32 (n, 2W+1, C) with channels
    [Ix, Qx, Iy, Qy] (+ [Px, Py] if power_feature), Y float32 (n, 4). The first
    four feature channels always match the target layout so residual models can
    add the center symbol directly.
    """
    xw, y0 = make_windows(rx_symbols[0], tx_symbols[0], half_window)
    yw, y1 = make_windows(rx_symbols[1], tx_symbols[1], half_window)
    feats = [xw.real, xw.imag, yw.real, yw.imag]
    if power_feature:
        feats += [np.abs(xw) ** 2, np.abs(yw) ** 2]
    x = np.stack(feats, axis=-1).astype(np.float32)
    y = np.stack([y0.real, y0.imag, y1.real, y1.imag], axis=-1).astype(np.float32)
    return x, y


def make_dp_stream_chunks(
    rx_symbols: np.ndarray,
    tx_symbols: np.ndarray,
    chunk_len: int = 64,
    warmup: int = 32,
    delay: int = 8,
    power_feature: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Dual-polarization streaming chunks (see ``make_stream_chunks`` for the
    alignment contract). rx/tx: (2, N) complex.

    Returns X float32 (n_chunks, warmup+chunk_len, C) with channels
    [Ix, Qx, Iy, Qy] (+ [Px, Py]), Y float32 (n_chunks, chunk_len, 4).
    """
    xa, ya = make_stream_chunks(rx_symbols[0], tx_symbols[0], chunk_len, warmup, delay)
    xb, yb = make_stream_chunks(rx_symbols[1], tx_symbols[1], chunk_len, warmup, delay)
    feats = [xa.real, xa.imag, xb.real, xb.imag]
    if power_feature:
        feats += [np.abs(xa) ** 2, np.abs(xb) ** 2]
    x = np.stack(feats, axis=-1).astype(np.float32)
    y = np.stack([ya.real, ya.imag, yb.real, yb.imag], axis=-1).astype(np.float32)
    return x, y


def to_real_features(x_complex: np.ndarray, power_feature: bool = False) -> np.ndarray:
    """(n, T) complex -> (n, T, C) float32.

    Channels are [real, imag] and, when ``power_feature`` is set, additionally
    the instantaneous power |x|^2. The Kerr nonlinearity is driven by power, so
    exposing it directly is a physics-informed input that spares every model
    from having to synthesize input-times-input products internally.
    """
    feats = [x_complex.real, x_complex.imag]
    if power_feature:
        feats.append(np.abs(x_complex) ** 2)
    return np.stack(feats, axis=-1).astype(np.float32)
