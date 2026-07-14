"""Dual-polarization physics: Manakov propagation, RSOP, block demux."""

import numpy as np

from fluidnn.channel.link_dp import DPLinkConfig, dp_report, simulate_dp_link
from fluidnn.channel.manakov import (
    apply_rsop,
    genie_block_demux,
    propagate_span_manakov,
    rsop_angles,
)
from fluidnn.channel.ssfm import FiberParams, propagate_span


def test_single_pol_limit_matches_scalar_with_manakov_factor():
    """With the y-pol empty, Manakov == scalar NLSE with gamma * 8/9."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(2048) + 1j * rng.standard_normal(2048)) * np.sqrt(1e-3)
    dual = np.stack([x, np.zeros_like(x)])
    fiber = FiberParams(length_km=80, gamma_w_km=1.3)
    out_dual = propagate_span_manakov(dual, 128e9, fiber, n_steps=50)
    fiber_scaled = FiberParams(length_km=80, gamma_w_km=1.3 * 8 / 9)
    out_scalar = propagate_span(x, 128e9, fiber_scaled, n_steps=50)
    assert np.max(np.abs(out_dual[0] - out_scalar)) < 1e-12
    assert np.max(np.abs(out_dual[1])) == 0.0


def test_energy_conserved_lossless_dual():
    rng = np.random.default_rng(1)
    field = (rng.standard_normal((2, 2048)) + 1j * rng.standard_normal((2, 2048))) * np.sqrt(5e-4)
    fiber = FiberParams(length_km=80, alpha_db_km=0.0, gamma_w_km=1.3)
    out = propagate_span_manakov(field, 128e9, fiber, n_steps=100)
    e_in = np.sum(np.abs(field) ** 2)
    e_out = np.sum(np.abs(out) ** 2)
    assert abs(e_out - e_in) / e_in < 1e-9


def test_rsop_is_unitary_and_demux_inverts_static_rotation():
    rng = np.random.default_rng(2)
    tx = rng.standard_normal((2, 4096)) + 1j * rng.standard_normal((2, 4096))
    theta = rsop_angles(4096, drift_deg_per_ksym=0.0, sps=1, theta0_deg=35.0)
    rx = apply_rsop(tx, theta)
    assert np.allclose(np.abs(rx[0]) ** 2 + np.abs(rx[1]) ** 2,
                       np.abs(tx[0]) ** 2 + np.abs(tx[1]) ** 2)
    recovered = genie_block_demux(rx, tx, block=256)
    assert np.max(np.abs(recovered - tx)) < 1e-9


def test_faster_drift_leaves_more_residual_after_block_demux():
    """Within-block rotation is uncorrectable by a per-block matrix: residual
    error must grow with the drift rate."""
    rng = np.random.default_rng(3)
    tx = rng.standard_normal((2, 8192)) + 1j * rng.standard_normal((2, 8192))
    residuals = []
    for rate in (5.0, 50.0):
        theta = rsop_angles(8192, drift_deg_per_ksym=rate, sps=1)
        rx = genie_block_demux(apply_rsop(tx, theta), tx, block=256)
        residuals.append(np.mean(np.abs(rx - tx) ** 2))
    assert residuals[1] > 5 * residuals[0]


def test_linear_noisefree_dp_link_is_error_free():
    cfg = DPLinkConfig(
        n_symbols=4096,
        ase=False,
        launch_power_dbm=0.0,
        n_spans=4,
        steps_per_span=10,
        rsop_theta0_deg=25.0,
        fiber=FiberParams(gamma_w_km=0.0),
    )
    r = simulate_dp_link(cfg)
    rep = dp_report(r["rx_symbols"], r["tx_symbols"], r["tx_bits"], r["qam"])
    assert rep["ber"] == 0.0
    assert rep["evm_percent"] < 1.0


def test_dp_nonlinearity_degrades_high_power():
    reports = {}
    for p in (0.0, 7.0):
        cfg = DPLinkConfig(
            n_symbols=4096, ase=False, launch_power_dbm=p,
            n_spans=8, steps_per_span=25, seed=7,
        )
        r = simulate_dp_link(cfg)
        reports[p] = dp_report(r["rx_symbols"], r["tx_symbols"], r["tx_bits"], r["qam"])
    assert reports[7.0]["evm_percent"] > reports[0.0]["evm_percent"]
