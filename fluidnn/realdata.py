"""Ingestion and alignment for experimental captures.

Lab captures arrive with unknown conventions: arbitrary file layout (.mat /
.npz / .csv), an unknown integer delay between the transmitted and received
sequences, arbitrary complex gain (amplitude + phase, including 90-degree
constellation rotations), possible spectral inversion (conjugated field), and
-- for dual-polarization captures -- possibly swapped polarizations. This
module resolves all of that with data-aided estimators so that the equalizer
pipeline sees the same clean (rx, tx) convention the simulator produces.
"""

from __future__ import annotations

import pathlib

import numpy as np


# --------------------------------------------------------------------- loading
def load_capture(path: str | pathlib.Path) -> dict[str, np.ndarray]:
    """Load a capture file into {name: array}, format inferred from suffix.

    Supports .npz, .mat (MATLAB, both pre-7.3 and HDF5-based v7.3), and
    .csv/.txt (single array). Complex data stored as separate real/imag
    columns is NOT auto-merged -- inspect with ``describe_capture`` first.
    """
    path = pathlib.Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {k: np.asarray(z[k]) for k in z.files}
    if suffix == ".mat":
        try:
            from scipy.io import loadmat

            raw = loadmat(path)
            return {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
        except NotImplementedError:  # MATLAB v7.3 = HDF5
            import h5py

            out = {}
            with h5py.File(path, "r") as f:
                def walk(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        arr = np.asarray(obj)
                        if arr.dtype.names and set(arr.dtype.names) >= {"real", "imag"}:
                            arr = arr["real"] + 1j * arr["imag"]
                        out[name] = arr
                f.visititems(walk)
            return out
    if suffix in (".csv", ".txt"):
        return {"data": np.loadtxt(path, delimiter="," if suffix == ".csv" else None)}
    raise ValueError(f"unsupported capture format: {suffix}")


def describe_capture(arrays: dict[str, np.ndarray]) -> str:
    """One line per array: name, shape, dtype, complex-ness — for a first look."""
    lines = []
    for k, v in arrays.items():
        kind = "complex" if np.iscomplexobj(v) else str(v.dtype)
        lines.append(f"{k:30s} shape={str(v.shape):18s} {kind}")
    return "\n".join(lines)


# ------------------------------------------------------------------- alignment
def find_delay(rx: np.ndarray, tx: np.ndarray) -> tuple[int, complex]:
    """Circular cross-correlation delay estimate.

    Returns (delay k, complex gain a) such that rx[n] ~= a * tx[n - k]
    (indices modulo the sequence length).
    """
    if len(rx) != len(tx):
        n = min(len(rx), len(tx))
        rx, tx = rx[:n], tx[:n]
    corr = np.fft.ifft(np.fft.fft(rx) * np.conj(np.fft.fft(tx)))
    k = int(np.argmax(np.abs(corr)))
    a = corr[k] / np.sum(np.abs(tx) ** 2)
    return k, complex(a)


def align_single(rx: np.ndarray, tx: np.ndarray) -> dict:
    """Align one polarization: resolve delay, complex gain, and conjugation.

    Returns {rx, tx, delay, gain, conjugated, nmse_db}: ``tx`` is rolled to sit
    time-aligned under ``rx``, and ``rx`` is scaled by 1/gain so both sequences
    share the simulator convention (unit-ish power, zero mean phase).
    """
    best = None
    for conj in (False, True):
        r = np.conj(rx) if conj else rx
        k, a = find_delay(r, tx)
        tx_aligned = np.roll(tx, k)
        residual = r / a - tx_aligned
        nmse = np.mean(np.abs(residual) ** 2) / np.mean(np.abs(tx_aligned) ** 2)
        if best is None or nmse < best["nmse"]:
            best = dict(rx=r / a, tx=tx_aligned, delay=k, gain=a, conjugated=conj, nmse=nmse)
    best["nmse_db"] = float(10 * np.log10(max(best.pop("nmse"), 1e-30)))
    return best


def align_dual_pol(rx: np.ndarray, tx: np.ndarray) -> dict:
    """Align a (2, N) dual-pol capture, additionally resolving polarization swap.

    Tries both rx-to-tx polarization assignments, aligns each pol independently
    (per-pol delay/gain/conjugation), and keeps the assignment with the lower
    combined NMSE. Residual polarization *mixing* (RSOP) is deliberately left
    untouched -- that is channel impairment for the receiver DSP/equalizer,
    not a capture-convention artifact.
    """
    best = None
    for swap in (False, True):
        r = rx[::-1] if swap else rx
        per_pol = [align_single(r[p], tx[p]) for p in range(2)]
        nmse = sum(p["nmse_db"] for p in per_pol)
        if best is None or nmse < best["_score"]:
            best = dict(
                rx=np.stack([p["rx"] for p in per_pol]),
                tx=np.stack([p["tx"] for p in per_pol]),
                per_pol=[{k: v for k, v in p.items() if k not in ("rx", "tx")} for p in per_pol],
                swapped=swap,
                _score=nmse,
            )
    best.pop("_score")
    return best
