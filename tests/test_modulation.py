"""Modulation sanity: bit round-trip, unit energy, Gray adjacency."""

import numpy as np
import pytest

from fluidnn.channel.modulation import QAM


@pytest.mark.parametrize("order", [4, 16, 64])
def test_bit_roundtrip(order):
    rng = np.random.default_rng(1)
    qam = QAM(order)
    bits = rng.integers(0, 2, size=6000 * qam.bits_per_symbol)
    symbols = qam.bits_to_symbols(bits)
    assert np.array_equal(qam.decide_bits(symbols), bits)


@pytest.mark.parametrize("order", [4, 16, 64])
def test_unit_energy(order):
    qam = QAM(order)
    assert np.isclose(np.mean(np.abs(qam.constellation) ** 2), 1.0)


def test_gray_neighbours_differ_by_one_bit():
    """Horizontally/vertically adjacent points must differ in exactly one bit."""
    qam = QAM(16)
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, size=4000 * qam.bits_per_symbol)
    symbols = qam.bits_to_symbols(bits)
    step = 2 * qam.scale  # distance between adjacent levels
    for shift in (step, 1j * step):
        moved = symbols + shift
        # keep only points that stay inside the constellation
        inside = (np.abs(moved.real) <= 3.05 * qam.scale) & (
            np.abs(moved.imag) <= 3.05 * qam.scale
        )
        a = qam.decide_bits(symbols[inside]).reshape(-1, 4)
        b = qam.decide_bits(moved[inside]).reshape(-1, 4)
        assert np.all(np.sum(a != b, axis=1) == 1)


def test_decide_symbols_matches_constellation():
    qam = QAM(16)
    rng = np.random.default_rng(3)
    noisy = qam.constellation + 0.01 * (rng.standard_normal(16) + 1j * rng.standard_normal(16))
    decided = qam.decide_symbols(noisy)
    assert np.allclose(decided, qam.constellation)
