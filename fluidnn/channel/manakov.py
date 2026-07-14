"""Dual-polarization propagation: the Manakov equation, plus receiver-side
time-varying polarization rotation (RSOP drift).

The Manakov model (averaged over fast random birefringence) for the Jones
vector A = [Ax, Ay]:

    dA/dz = -(alpha/2) A - j (beta2/2) d^2A/dT^2 + j (8/9) gamma (|Ax|^2+|Ay|^2) A

i.e. both polarizations share the dispersion/loss operator and a *common*
nonlinear phase driven by the total instantaneous power, with the 8/9 Manakov
factor. First-order PMD is deliberately omitted in this first version; the
time-varying impairment is a receiver-side rotation of the state of
polarization (RSOP), the slow endless rotation real coherent receivers must
track.
"""

from __future__ import annotations

import numpy as np

from fluidnn.channel.ssfm import FiberParams, _PLANCK, _C


def propagate_span_manakov(
    field: np.ndarray, fs: float, fiber: FiberParams, n_steps: int
) -> np.ndarray:
    """Propagate a (2, N) Jones field through one span (symmetric split-step)."""
    if field.ndim != 2 or field.shape[0] != 2:
        raise ValueError("field must have shape (2, N)")
    n = field.shape[1]
    w = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / fs)
    h = fiber.length_km * 1e3 / n_steps
    half_linear = np.exp(
        (1j * fiber.beta2_s2_m / 2) * w**2 * (h / 2) - (fiber.alpha_np_m / 2) * (h / 2)
    )
    gamma_eff = (8.0 / 9.0) * fiber.gamma_w_m

    a = np.fft.fft(field, axis=1)
    for _ in range(n_steps):
        a *= half_linear
        t = np.fft.ifft(a, axis=1)
        total_power = np.abs(t[0]) ** 2 + np.abs(t[1]) ** 2
        t *= np.exp(1j * gamma_eff * total_power * h)
        a = np.fft.fft(t, axis=1)
        a *= half_linear
    return np.fft.ifft(a, axis=1)


def edfa_dual(
    field: np.ndarray,
    fs: float,
    fiber: FiberParams,
    nf_db: float,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Span-loss-compensating EDFA with independent ASE on each polarization."""
    gain = np.exp(fiber.alpha_np_m * fiber.length_km * 1e3)
    out = field * np.sqrt(gain)
    if rng is not None:
        nu = _C / (fiber.wavelength_nm * 1e-9)
        n_sp = 10 ** (nf_db / 10) / 2
        p_ase = (gain - 1) * n_sp * _PLANCK * nu * fs
        noise = np.sqrt(p_ase / 2) * (
            rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape)
        )
        out = out + noise
    return out


def propagate_link_manakov(
    field: np.ndarray,
    fs: float,
    fiber: FiberParams,
    n_spans: int,
    n_steps_per_span: int,
    nf_db: float = 4.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    out = field
    for _ in range(n_spans):
        out = propagate_span_manakov(out, fs, fiber, n_steps_per_span)
        out = edfa_dual(out, fs, fiber, nf_db, rng)
    return out


# --------------------------------------------------------------- RSOP drift
def rsop_angles(n: int, drift_deg_per_ksym: float, sps: int, theta0_deg: float = 0.0) -> np.ndarray:
    """Rotation angle per sample for a linear RSOP drift.

    ``drift_deg_per_ksym`` is the rotation rate in degrees per 1000 *symbols*
    (so it is symbol-rate independent); 0 gives a static rotation theta0.
    """
    theta0 = np.deg2rad(theta0_deg)
    rate = np.deg2rad(drift_deg_per_ksym) / (1000.0 * sps)  # rad per sample
    return theta0 + rate * np.arange(n)


def apply_rsop(field: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Apply a (possibly time-varying) polarization rotation R(theta_k) per sample:

        [x']   [ cos t  -sin t ] [x]
        [y'] = [ sin t   cos t ] [y]
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * field[0] - s * field[1], s * field[0] + c * field[1]])


def genie_block_demux(
    rx: np.ndarray, tx: np.ndarray, block: int = 256
) -> np.ndarray:
    """Block-wise data-aided 2x2 demux (the dual-pol analogue of genie CPE).

    For each block of ``block`` symbols, the least-squares 2x2 Jones matrix
    mapping rx -> tx is estimated from the known data and applied. Models an
    ideal MIMO tracker with an update period of one block; within-block drift
    is untouched -- exactly the residual a downstream equalizer must handle.
    rx, tx: shape (2, N) symbol-rate sequences.
    """
    n = rx.shape[1]
    out = np.empty_like(rx)
    for start in range(0, n, block):
        sl = slice(start, min(start + block, n))
        r, t = rx[:, sl], tx[:, sl]
        # LS solve J r ~= t  ->  J = (t r^H) (r r^H)^-1
        j = (t @ r.conj().T) @ np.linalg.inv(r @ r.conj().T)
        out[:, sl] = j @ r
    return out
