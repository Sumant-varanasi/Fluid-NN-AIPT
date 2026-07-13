"""SSFM physics validation against analytically known behaviour."""

import numpy as np

from fluidnn.channel.link import LinkConfig, simulate_link
from fluidnn.channel.pulse import rrc_taps, filt_circular, upsample
from fluidnn.channel.receiver import cdc
from fluidnn.channel.ssfm import FiberParams, propagate_span
from fluidnn.metrics import equalizer_report


def _gaussian_pulse(n, fs, t0):
    t = (np.arange(n) - n / 2) / fs
    return np.exp(-(t**2) / (2 * t0**2)).astype(complex)


def test_linear_propagation_is_inverted_by_cdc():
    """gamma = 0, no loss, no noise: CDC must exactly undo the fiber."""
    fs = 128e9
    field = _gaussian_pulse(4096, fs, t0=50e-12)
    fiber = FiberParams(length_km=80, alpha_db_km=0.0, gamma_w_km=0.0)
    out = propagate_span(field, fs, fiber, n_steps=20)
    back = cdc(out, fs, fiber.beta2_s2_m, fiber.length_km * 1e3)
    assert np.max(np.abs(back - field)) < 1e-10


def test_dispersion_broadens_gaussian_by_theory():
    """RMS width of a Gaussian grows as T1 = T0 sqrt(1 + (beta2 z / T0^2)^2)."""
    fs = 256e9
    n = 8192
    t0 = 25e-12
    field = _gaussian_pulse(n, fs, t0)
    fiber = FiberParams(length_km=100, alpha_db_km=0.0, gamma_w_km=0.0)
    out = propagate_span(field, fs, fiber, n_steps=10)

    t = (np.arange(n) - n / 2) / fs
    power = np.abs(out) ** 2
    t1_meas = np.sqrt(np.sum(t**2 * power) / np.sum(power))  # RMS width = T1/sqrt(2)
    z = fiber.length_km * 1e3
    t1_theory = t0 * np.sqrt(1 + (fiber.beta2_s2_m * z / t0**2) ** 2) / np.sqrt(2)
    assert abs(t1_meas - t1_theory) / t1_theory < 0.01


def test_energy_conserved_without_loss():
    """Lossless nonlinear propagation must conserve energy (unitary evolution)."""
    rng = np.random.default_rng(0)
    field = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)) * np.sqrt(0.5e-3)
    fiber = FiberParams(length_km=80, alpha_db_km=0.0, gamma_w_km=1.3)
    out = propagate_span(field, 128e9, fiber, n_steps=100)
    e_in, e_out = np.sum(np.abs(field) ** 2), np.sum(np.abs(out) ** 2)
    assert abs(e_out - e_in) / e_in < 1e-9


def test_nonlinear_phase_of_cw_matches_theory():
    """A CW field acquires exactly phi_NL = gamma * P * L_eff (loss included)."""
    fs = 128e9
    p0 = 5e-3  # W
    field = np.full(2048, np.sqrt(p0), dtype=complex)
    fiber = FiberParams(length_km=80, alpha_db_km=0.2, gamma_w_km=1.3)
    out = propagate_span(field, fs, fiber, n_steps=400)
    alpha, L = fiber.alpha_np_m, fiber.length_km * 1e3
    l_eff = (1 - np.exp(-alpha * L)) / alpha
    phi_theory = fiber.gamma_w_m * p0 * l_eff
    phi_meas = np.angle(out[1024] / field[1024])
    assert abs(phi_meas - phi_theory) < 1e-3
    # amplitude decayed by exp(-alpha L / 2)
    assert np.allclose(np.abs(out), np.sqrt(p0) * np.exp(-alpha * L / 2), rtol=1e-6)


def test_linear_link_with_noise_free_rx_is_error_free():
    """Full pipeline, gamma = 0, ASE off: BER must be exactly 0."""
    cfg = LinkConfig(
        n_symbols=4096,
        ase=False,
        launch_power_dbm=0.0,
        n_spans=4,
        steps_per_span=10,
        fiber=FiberParams(gamma_w_km=0.0),
    )
    r = simulate_link(cfg)
    rep = equalizer_report(r["rx_symbols"], r["tx_symbols"], r["tx_bits"], r["qam"])
    assert rep["ber"] == 0.0
    assert rep["evm_percent"] < 1.0


def test_nonlinearity_degrades_high_power():
    """With ASE off, higher launch power must give worse EVM (pure Kerr distortion)."""
    reports = {}
    for p_dbm in (0.0, 6.0):
        cfg = LinkConfig(
            n_symbols=4096,
            ase=False,
            launch_power_dbm=p_dbm,
            n_spans=8,
            steps_per_span=25,
            seed=7,
        )
        r = simulate_link(cfg)
        reports[p_dbm] = equalizer_report(r["rx_symbols"], r["tx_symbols"], r["tx_bits"], r["qam"])
    assert reports[6.0]["evm_percent"] > reports[0.0]["evm_percent"]
