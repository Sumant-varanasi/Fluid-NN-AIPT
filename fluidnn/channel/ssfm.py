"""Split-step Fourier propagation of the scalar nonlinear Schroedinger equation.

Model (single polarization, field envelope A(z, t), power in W):

    dA/dz = -(alpha/2) A - j (beta2/2) d^2A/dT^2 + j gamma |A|^2 A

Linear step applied in the frequency domain as exp(+j (beta2/2) w^2 h) together
with loss; nonlinear step as exp(+j gamma |A|^2 h). Symmetric (Strang) splitting.
Each span is followed by an EDFA that exactly compensates the span loss and adds
ASE noise (white complex Gaussian over the simulation bandwidth).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_PLANCK = 6.62607015e-34  # J s
_C = 299792458.0  # m/s


@dataclass
class FiberParams:
    """Standard single-mode fiber by default."""

    length_km: float = 80.0
    alpha_db_km: float = 0.2
    dispersion_ps_nm_km: float = 17.0  # D; beta2 derived from it
    gamma_w_km: float = 1.3
    wavelength_nm: float = 1550.0

    @property
    def alpha_np_m(self) -> float:  # power attenuation, 1/m
        return self.alpha_db_km * np.log(10) / 10 / 1e3

    @property
    def beta2_s2_m(self) -> float:
        lam = self.wavelength_nm * 1e-9
        d_si = self.dispersion_ps_nm_km * 1e-6  # s/m^2
        return -d_si * lam**2 / (2 * np.pi * _C)

    @property
    def gamma_w_m(self) -> float:
        return self.gamma_w_km / 1e3


def propagate_span(field: np.ndarray, fs: float, fiber: FiberParams, n_steps: int) -> np.ndarray:
    """Propagate ``field`` (sampled at ``fs``) through one span with symmetric SSFM."""
    n = len(field)
    w = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / fs)
    h = fiber.length_km * 1e3 / n_steps
    half_linear = np.exp(
        (1j * fiber.beta2_s2_m / 2) * w**2 * (h / 2) - (fiber.alpha_np_m / 2) * (h / 2)
    )
    gamma = fiber.gamma_w_m

    a = np.fft.fft(field)
    for _ in range(n_steps):
        a *= half_linear
        t_domain = np.fft.ifft(a)
        t_domain *= np.exp(1j * gamma * np.abs(t_domain) ** 2 * h)
        a = np.fft.fft(t_domain)
        a *= half_linear
    return np.fft.ifft(a)


def edfa(
    field: np.ndarray,
    fs: float,
    fiber: FiberParams,
    nf_db: float,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Amplify by the exact span loss and add ASE noise (skipped if rng is None)."""
    gain = np.exp(fiber.alpha_np_m * fiber.length_km * 1e3)  # power gain
    out = field * np.sqrt(gain)
    if rng is not None:
        nu = _C / (fiber.wavelength_nm * 1e-9)
        n_sp = 10 ** (nf_db / 10) / 2  # high-gain approximation
        p_ase = (gain - 1) * n_sp * _PLANCK * nu * fs  # W in simulation bandwidth
        noise = np.sqrt(p_ase / 2) * (
            rng.standard_normal(len(field)) + 1j * rng.standard_normal(len(field))
        )
        out = out + noise
    return out


def propagate_link(
    field: np.ndarray,
    fs: float,
    fiber: FiberParams,
    n_spans: int,
    n_steps_per_span: int,
    nf_db: float = 4.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Multi-span link: [fiber span -> EDFA(+ASE)] x n_spans."""
    out = field
    for _ in range(n_spans):
        out = propagate_span(out, fs, fiber, n_steps_per_span)
        out = edfa(out, fs, fiber, nf_db, rng)
    return out
