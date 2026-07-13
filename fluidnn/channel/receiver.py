"""Coherent receiver DSP: chromatic dispersion compensation, matched filtering,
downsampling, and data-aided complex scaling."""

from __future__ import annotations

import numpy as np

from fluidnn.channel.pulse import filt_circular


def cdc(field: np.ndarray, fs: float, beta2_s2_m: float, total_length_m: float) -> np.ndarray:
    """Ideal frequency-domain compensation of the accumulated chromatic dispersion."""
    n = len(field)
    w = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / fs)
    return np.fft.ifft(
        np.fft.fft(field) * np.exp(-1j * (beta2_s2_m / 2) * w**2 * total_length_m)
    )


def matched_filter_downsample(field: np.ndarray, taps: np.ndarray, sps: int) -> np.ndarray:
    """RRC matched filter followed by symbol-rate decimation (symbols sit on k*sps)."""
    return filt_circular(field, taps)[::sps]


def ls_correct(rx: np.ndarray, tx: np.ndarray) -> np.ndarray:
    """Data-aided single complex scalar (amplitude + constant phase) correction.

    This removes the mean nonlinear phase rotation and any residual gain, the
    standard genie normalization used in equalizer studies; per-symbol distortion
    is untouched and left for the equalizer under test.
    """
    a = np.vdot(rx, tx) / np.vdot(rx, rx)
    return a * rx
