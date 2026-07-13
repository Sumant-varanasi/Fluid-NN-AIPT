"""End-to-end statistical check: measured SER over AWGN matches exact theory."""

import numpy as np
import pytest

from fluidnn.channel.modulation import QAM
from fluidnn.metrics import ser, theory_ser_qam_awgn


@pytest.mark.parametrize("order,snr_db", [(4, 8.0), (16, 15.0), (64, 20.0)])
def test_awgn_ser_matches_theory(order, snr_db):
    rng = np.random.default_rng(42)
    qam = QAM(order)
    n = 400_000
    tx, _ = qam.random_symbols(n, rng)
    n0 = 10 ** (-snr_db / 10)  # Es = 1
    noise = np.sqrt(n0 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    rx = tx + noise
    measured = ser(qam.decide_symbols(rx), qam.decide_symbols(tx))
    expected = theory_ser_qam_awgn(order, snr_db)
    assert measured == pytest.approx(expected, rel=0.08)
