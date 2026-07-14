"""Equalizer models. All share one interface:

    forward(x) with x of shape (batch, T, 2)  ->  (batch, 2)

where T = 2*half_window + 1 received symbols centered on the symbol to recover,
channels are [real, imag], and the output is the equalized center symbol.
Every model reports weight MACs per recovered symbol via ``macs_per_symbol()``.
"""

from fluidnn.models.mlp import MLPEqualizer
from fluidnn.models.lstm import BiLSTMEqualizer, StreamingLSTMEqualizer
from fluidnn.models.cfc import CfCEqualizer, StreamingCfCEqualizer
