"""Streaming CfC: chunk alignment and delayed-passthrough identity."""

import numpy as np
import torch

from fluidnn.channel.dataset import make_stream_chunks, to_real_features
from fluidnn.models import StreamingCfCEqualizer


def test_chunks_cover_sequence_exactly_once():
    n, L, W, D = 1024, 128, 32, 8
    rx = np.arange(n) + 0j
    tx = 1000.0 + np.arange(n) + 0j
    x, y = make_stream_chunks(rx, tx, chunk_len=L, warmup=W, delay=D)
    assert x.shape == (n // L, W + L) and y.shape == (n // L, L)
    # every tx symbol appears exactly once across all targets
    assert sorted(v - 1000 for v in y.reshape(-1).real) == list(range(n))
    # alignment: y[k, j] is tx at the position of x[k, W + j - D]
    assert np.allclose(y.real - 1000, x[:, W - D : W - D + L].real % 1024)


def test_zero_weight_model_is_delayed_passthrough():
    torch.manual_seed(0)
    W, D, L = 16, 4, 64
    m = StreamingCfCEqualizer(hidden=8, delay=D, warmup=W)
    for p in m.parameters():
        torch.nn.init.zeros_(p)
    x = torch.randn(3, W + L, 2)
    out = m(x)
    assert out.shape == (3, L, 2)
    assert torch.allclose(out, x[:, W - D : W + L - D, :2])


def test_streaming_matches_channel_error_when_untrained():
    """Zero-weight streaming model must reproduce the channel MSE exactly."""
    rng = np.random.default_rng(0)
    n = 2048
    tx = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    rx = tx + 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    xc, yc = make_stream_chunks(rx, tx, chunk_len=256, warmup=32, delay=8)
    x = torch.from_numpy(to_real_features(xc))
    y = np.stack([yc.real, yc.imag], -1)
    m = StreamingCfCEqualizer(hidden=8, delay=8, warmup=32)
    for p in m.parameters():
        torch.nn.init.zeros_(p)
    with torch.no_grad():
        pred = m(x).numpy()
    assert np.isclose(np.mean((pred - y) ** 2), np.mean(np.abs(rx - tx) ** 2) / 2, rtol=1e-5)