"""RRC pulse shaping: unit energy, Nyquist (ISI-free) matched cascade, circular filtering."""

import numpy as np

from fluidnn.channel.pulse import filt_circular, rrc_taps, upsample


def test_unit_energy():
    taps = rrc_taps(sps=4, span=32, rolloff=0.1)
    assert np.isclose(np.sum(taps**2), 1.0)
    assert len(taps) % 2 == 1


def test_matched_cascade_is_nyquist():
    """RRC -> RRC must be ISI-free at symbol-spaced sampling points."""
    sps, span = 4, 32
    taps = rrc_taps(sps, span, rolloff=0.1)
    rc = np.convolve(taps, taps)  # raised cosine, peak at center
    center = len(rc) // 2
    symbol_points = rc[center % sps :: sps]
    peak = np.argmax(np.abs(symbol_points))
    isi = np.delete(symbol_points, peak)
    assert np.isclose(symbol_points[peak], 1.0, atol=1e-3)
    assert np.max(np.abs(isi)) < 5e-3


def test_circular_shaping_roundtrip_recovers_symbols():
    """upsample -> RRC -> RRC -> downsample == identity up to truncation ISI.

    The residual error comes only from truncating the RRC impulse response, so
    it must fall monotonically (and fast) as the filter span grows.
    """
    rng = np.random.default_rng(0)
    symbols = rng.standard_normal(512) + 1j * rng.standard_normal(512)
    sps = 4
    errors = []
    for span in (32, 64, 128):
        taps = rrc_taps(sps, span, rolloff=0.1)
        shaped = filt_circular(upsample(symbols, sps), taps)
        recovered = filt_circular(shaped, taps)[::sps]
        errors.append(np.max(np.abs(recovered - symbols)))
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] < 5e-3  # span 64 (the simulator default) is already clean
    assert errors[2] < 2e-3
