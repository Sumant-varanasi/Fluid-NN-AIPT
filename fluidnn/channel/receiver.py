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


def cpe_sliding(rx: np.ndarray, tx: np.ndarray, window: int) -> np.ndarray:
    """Genie sliding-window carrier phase estimation (circular moving average).

    Removes the slowly wandering common phase (laser/Gordon-Mollenauer type)
    exactly the way an ideal decision-directed CPE would, using the known data --
    the standard idealization in equalizer studies. Fast per-symbol distortion
    is untouched.
    """
    n = len(rx)
    if not 0 < window < n:
        raise ValueError("window must be in (0, len(rx))")
    kernel = np.zeros(n)
    kernel[:window] = 1.0 / window
    kernel = np.roll(kernel, -(window // 2))  # center the average on each symbol
    smoothed = np.fft.ifft(np.fft.fft(rx * np.conj(tx)) * np.fft.fft(kernel))
    return rx * np.exp(-1j * np.angle(smoothed))


def ls_correct(rx: np.ndarray, tx: np.ndarray) -> np.ndarray:
    """Data-aided single complex scalar (amplitude + constant phase) correction.

    This removes the mean nonlinear phase rotation and any residual gain, the
    standard genie normalization used in equalizer studies; per-symbol distortion
    is untouched and left for the equalizer under test.
    """
    a = np.vdot(rx, tx) / np.vdot(rx, rx)
    return a * rx
