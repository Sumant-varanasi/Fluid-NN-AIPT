"""Root-raised-cosine pulse shaping and delay-free circular filtering.

The whole simulation chain is circular (the SSFM uses FFTs, hence periodic
boundary conditions), so pulse shaping uses circular convolution too: there are
no filter edge effects and no group-delay bookkeeping anywhere in the chain.
"""

from __future__ import annotations

import numpy as np


def rrc_taps(sps: int, span: int, rolloff: float) -> np.ndarray:
    """Root-raised-cosine impulse response, unit energy, odd length ``span*sps + 1``.

    Args:
        sps: samples per symbol.
        span: filter length in symbol durations (even recommended).
        rolloff: roll-off factor beta in (0, 1].
    """
    if not 0 < rolloff <= 1:
        raise ValueError("rolloff must be in (0, 1]")
    n = span * sps
    t = np.arange(-(n // 2), n // 2 + 1) / sps  # in symbol durations
    h = np.zeros_like(t)

    # Regular points
    denom = np.pi * t * (1 - (4 * rolloff * t) ** 2)
    regular = np.abs(denom) > 1e-12
    tr = t[regular]
    h[regular] = (
        np.sin(np.pi * tr * (1 - rolloff))
        + 4 * rolloff * tr * np.cos(np.pi * tr * (1 + rolloff))
    ) / (np.pi * tr * (1 - (4 * rolloff * tr) ** 2))

    # t = 0
    h[np.abs(t) < 1e-12] = 1 - rolloff + 4 * rolloff / np.pi

    # t = +/- 1/(4 rolloff)
    singular = np.abs(np.abs(t) - 1 / (4 * rolloff)) < 1e-12
    h[singular] = (rolloff / np.sqrt(2)) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
    )

    return h / np.sqrt(np.sum(h**2))


def filt_circular(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Zero-delay circular convolution of ``x`` with odd-length symmetric ``taps``."""
    n, L = len(x), len(taps)
    if L > n:
        raise ValueError("filter longer than signal")
    center = L // 2
    h = np.zeros(n, dtype=complex)
    h[: L - center] = taps[center:]
    h[n - center :] = taps[:center]
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(h))


def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    """Zero-insertion upsampling: symbol k lands on sample k*sps."""
    out = np.zeros(len(symbols) * sps, dtype=complex)
    out[::sps] = symbols
    return out
