"""Coherent optical channel simulation: modulation, pulse shaping, fiber propagation, receiver DSP."""

from fluidnn.channel.modulation import QAM
from fluidnn.channel.pulse import rrc_taps, filt_circular, upsample
from fluidnn.channel.ssfm import FiberParams, propagate_span, propagate_link
from fluidnn.channel.receiver import cdc, matched_filter_downsample, ls_correct
from fluidnn.channel.link import LinkConfig, simulate_link
from fluidnn.channel.dataset import make_windows
