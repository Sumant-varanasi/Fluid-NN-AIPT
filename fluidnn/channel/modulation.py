"""Gray-coded square M-QAM mapping, hard-decision demapping, and bit round-trip."""

from __future__ import annotations

import numpy as np


def _gray(n: np.ndarray | int):
    return n ^ (n >> 1)


class QAM:
    """Square M-QAM (M = 4, 16, 64, 256) with independent Gray coding per I/Q axis.

    The constellation is normalized to unit average symbol energy.
    Bit convention: each symbol carries ``bits_per_symbol`` bits, the first half
    (MSB first) selects the I level, the second half the Q level.
    """

    def __init__(self, order: int):
        side = int(round(np.sqrt(order)))
        if side * side != order or order < 4 or (side & (side - 1)) != 0:
            raise ValueError(f"order must be 4, 16, 64, 256, ... got {order}")
        self.order = order
        self.side = side
        self.bits_per_symbol = int(np.log2(order))
        self.bits_per_axis = self.bits_per_symbol // 2

        positions = np.arange(side)
        amplitudes = (2 * positions - (side - 1)).astype(float)  # -3,-1,+1,+3 ...
        # Gray label of the level at position i is gray(i); build label -> amplitude.
        self._amp_of_label = np.empty(side)
        self._amp_of_label[_gray(positions)] = amplitudes
        self._label_of_position = _gray(positions)

        es = 2.0 * np.mean(amplitudes**2)  # I and Q are independent PAM
        self.scale = 1.0 / np.sqrt(es)

        labels = np.arange(order)
        i_lab, q_lab = labels >> self.bits_per_axis, labels & (side - 1)
        self.constellation = self.scale * (
            self._amp_of_label[i_lab] + 1j * self._amp_of_label[q_lab]
        )

    # ------------------------------------------------------------------ mapping
    def bits_to_symbols(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.int64).reshape(-1, self.bits_per_symbol)
        weights = 1 << np.arange(self.bits_per_axis - 1, -1, -1)
        i_label = bits[:, : self.bits_per_axis] @ weights
        q_label = bits[:, self.bits_per_axis :] @ weights
        return self.scale * (self._amp_of_label[i_label] + 1j * self._amp_of_label[q_label])

    def random_symbols(self, n: int, rng: np.random.Generator):
        bits = rng.integers(0, 2, size=n * self.bits_per_symbol)
        return self.bits_to_symbols(bits), bits

    # ----------------------------------------------------------------- demapping
    def _nearest_positions(self, values: np.ndarray) -> np.ndarray:
        pos = np.round((values / self.scale + (self.side - 1)) / 2.0)
        return np.clip(pos, 0, self.side - 1).astype(np.int64)

    def decide_bits(self, symbols: np.ndarray) -> np.ndarray:
        """Minimum-distance decision, returns the Gray-decoded bit stream."""
        symbols = np.asarray(symbols)
        i_lab = self._label_of_position[self._nearest_positions(symbols.real)]
        q_lab = self._label_of_position[self._nearest_positions(symbols.imag)]
        shifts = np.arange(self.bits_per_axis - 1, -1, -1)
        i_bits = (i_lab[:, None] >> shifts) & 1
        q_bits = (q_lab[:, None] >> shifts) & 1
        return np.concatenate([i_bits, q_bits], axis=1).reshape(-1)

    def decide_symbols(self, symbols: np.ndarray) -> np.ndarray:
        """Minimum-distance decision, returns the nearest constellation points."""
        symbols = np.asarray(symbols)
        i_pos = self._nearest_positions(symbols.real)
        q_pos = self._nearest_positions(symbols.imag)
        amps = 2 * np.arange(self.side) - (self.side - 1)
        return self.scale * (amps[i_pos] + 1j * amps[q_pos])
