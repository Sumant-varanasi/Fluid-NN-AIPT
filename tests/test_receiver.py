"""Receiver DSP stages: CPE removes slow phase wander, leaves fast distortion."""

import numpy as np

from fluidnn.channel.modulation import QAM
from fluidnn.channel.receiver import cpe_sliding, ls_correct


def test_cpe_removes_slow_phase_walk():
    rng = np.random.default_rng(0)
    qam = QAM(16)
    tx, _ = qam.random_symbols(8192, rng)
    # slow random-walk phase + mild AWGN
    phase = np.cumsum(rng.standard_normal(8192) * 0.02)
    noise = 0.02 * (rng.standard_normal(8192) + 1j * rng.standard_normal(8192))
    rx = tx * np.exp(1j * phase) + noise

    before = np.mean(np.abs(rx - tx) ** 2)
    after = np.mean(np.abs(cpe_sliding(rx, tx, window=32) - tx) ** 2)
    assert after < before / 5
    # essentially only the AWGN should remain
    assert after < 3 * np.mean(np.abs(noise) ** 2)


def test_cpe_does_not_touch_white_error():
    """With no phase wander, CPE must be (nearly) a no-op."""
    rng = np.random.default_rng(1)
    qam = QAM(16)
    tx, _ = qam.random_symbols(8192, rng)
    noise = 0.05 * (rng.standard_normal(8192) + 1j * rng.standard_normal(8192))
    rx = tx + noise
    out = cpe_sliding(rx, tx, window=32)
    assert np.mean(np.abs(out - rx) ** 2) < 0.05 * np.mean(np.abs(noise) ** 2)


def test_ls_correct_undoes_complex_gain():
    rng = np.random.default_rng(2)
    qam = QAM(16)
    tx, _ = qam.random_symbols(1024, rng)
    rx = 0.8 * np.exp(1j * 0.7) * tx
    assert np.allclose(ls_correct(rx, tx), tx)
