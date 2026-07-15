"""Real-capture alignment: recover delay, gain, conjugation, and pol swap."""

import numpy as np
import pytest

from fluidnn.channel.modulation import QAM
from fluidnn.realdata import align_dual_pol, align_single, find_delay, load_capture


def _sequence(n, seed):
    rng = np.random.default_rng(seed)
    qam = QAM(16)
    tx, _ = qam.random_symbols(n, rng)
    return tx, rng


def test_find_delay_exact():
    tx, rng = _sequence(4096, 0)
    for true_delay in (0, 1, 137, 4000):
        rx = np.roll(tx, true_delay) * (0.7 * np.exp(1j * 0.9))
        rx += 0.01 * (rng.standard_normal(4096) + 1j * rng.standard_normal(4096))
        k, a = find_delay(rx, tx)
        assert k == true_delay
        assert abs(a - 0.7 * np.exp(1j * 0.9)) < 0.01


@pytest.mark.parametrize("conjugated", [False, True])
def test_align_single_recovers_convention(conjugated):
    tx, rng = _sequence(4096, 1)
    rx = np.roll(tx, 731) * (1.3 * np.exp(-1j * 2.1))
    rx += 0.02 * (rng.standard_normal(4096) + 1j * rng.standard_normal(4096))
    if conjugated:
        rx = np.conj(rx)
    out = align_single(rx, tx)
    assert out["delay"] == 731
    assert out["conjugated"] == conjugated
    assert out["nmse_db"] < -30
    # after alignment, rx and tx agree up to the injected noise
    assert np.mean(np.abs(out["rx"] - out["tx"]) ** 2) < 1e-3


@pytest.mark.parametrize("swap", [False, True])
def test_align_dual_pol_resolves_swap(swap):
    tx0, rng = _sequence(4096, 2)
    tx1, _ = _sequence(4096, 3)
    tx = np.stack([tx0, tx1])
    rx = np.stack([
        np.roll(tx[0], 55) * np.exp(1j * 0.4),
        np.roll(tx[1], 55) * np.exp(-1j * 1.0),
    ])
    rx += 0.02 * (rng.standard_normal(rx.shape) + 1j * rng.standard_normal(rx.shape))
    if swap:
        rx = rx[::-1]
    out = align_dual_pol(rx, tx)
    assert out["swapped"] == swap
    for p in range(2):
        assert out["per_pol"][p]["delay"] == 55
        assert np.mean(np.abs(out["rx"][p] - out["tx"][p]) ** 2) < 1e-3


def test_load_capture_npz_roundtrip(tmp_path):
    tx, _ = _sequence(256, 4)
    f = tmp_path / "capture.npz"
    np.savez(f, tx=tx, rx=tx * 2.0, meta=np.array([1.0, 2.0]))
    arrays = load_capture(f)
    assert set(arrays) == {"tx", "rx", "meta"}
    assert np.allclose(arrays["rx"], tx * 2.0)


def test_load_capture_mat_roundtrip(tmp_path):
    from scipy.io import savemat

    tx, _ = _sequence(256, 5)
    f = tmp_path / "capture.mat"
    savemat(f, {"txSig": tx, "rxSig": tx * 1j})
    arrays = load_capture(f)
    assert "txSig" in arrays and "rxSig" in arrays
    assert np.allclose(np.ravel(arrays["rxSig"]), tx * 1j)
