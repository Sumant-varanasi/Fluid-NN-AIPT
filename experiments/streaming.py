"""Streaming CfC benchmark: the O(1)-per-symbol real-time operating mode.

Trains causal streaming CfC equalizers (state carried across the whole symbol
stream, ``delay`` symbols of decision latency, no windows, no recomputation)
and compares them against the window-based results on identical test data.
The per-symbol cost is one cell update + head: this is the point of the
continuous-time approach for real-time DSP.

Run from the repo root:  python experiments/streaming.py
Outputs: results/streaming.json, docs/figures/streaming_frontier.png
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from power_sweep import FIGURES_DIR, RESULTS_DIR, link_config

from fluidnn.channel.dataset import make_stream_chunks, to_real_features
from fluidnn.channel.link import simulate_link
from fluidnn.metrics import equalizer_report
from fluidnn.models import StreamingCfCEqualizer
from fluidnn.training.harness import train_equalizer

POWER_DBM = 3.0
CHUNK, WARMUP, DELAY = 256, 32, 8
HIDDEN_SIZES = [16, 32, 64]
COLORS = {"CDC+CPE": "#666666", "MLP": "#2a78d6", "BiLSTM": "#1baf7a", "CfC": "#eda100"}


def build(n_symbols: int, seed: int) -> dict:
    r = simulate_link(link_config(n_symbols, seed, POWER_DBM))
    xc, yc = make_stream_chunks(r["rx_symbols"], r["tx_symbols"], CHUNK, WARMUP, DELAY)
    r["x"] = to_real_features(xc, power_feature=True)  # (n_chunks, W+L, 3)
    r["y"] = np.stack([yc.real, yc.imag], axis=-1).astype(np.float32)  # (n_chunks, L, 2)
    return r


def assemble_predictions(pred: np.ndarray, n: int) -> np.ndarray:
    """(n_chunks, L, 2) predictions -> complex sequence aligned to tx positions."""
    eq = np.zeros(n, dtype=complex)
    starts = np.arange(0, n, CHUNK)
    y_idx = (starts[:, None] + np.arange(CHUNK)[None, :] - DELAY) % n
    eq[y_idx.reshape(-1)] = (pred[..., 0] + 1j * pred[..., 1]).reshape(-1)
    return eq


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Simulating links ...")
    train = build(2**17, seed=11)
    test = build(2**15, seed=22)
    n_val = max(len(train["x"]) // 10, 8)

    results = {}
    base = equalizer_report(test["rx_symbols"], test["tx_symbols"], test["tx_bits"], test["qam"])
    base.update(params=0, macs_per_symbol=0)
    results["CDC+CPE"] = base
    print(f"CDC+CPE: BER={base['ber']:.2e}  Q={base['q_db']:.2f}")

    for hidden in HIDDEN_SIZES:
        name = f"StreamCfC-h{hidden}"
        print(f"-- training {name} ...")
        model = StreamingCfCEqualizer(
            hidden=hidden, backbone_units=64, in_channels=3, delay=DELAY, warmup=WARMUP
        )
        train_equalizer(
            model,
            train["x"][:-n_val], train["y"][:-n_val],
            train["x"][-n_val:], train["y"][-n_val:],
            epochs=60, batch_size=32, lr=3e-3, lr_min=1e-4, seed=0, verbose=False,
        )
        model.eval()
        with torch.no_grad():
            pred = torch.cat(
                [model(torch.from_numpy(test["x"][i : i + 32])) for i in range(0, len(test["x"]), 32)]
            ).numpy()
        eq = assemble_predictions(pred, len(test["tx_symbols"]))
        rep = equalizer_report(eq, test["tx_symbols"], test["tx_bits"], test["qam"])
        rep["params"] = sum(p.numel() for p in model.parameters())
        rep["macs_per_symbol"] = model.macs_per_symbol()
        rep["delay_symbols"] = DELAY
        results[name] = rep
        print(f"{name}: BER={rep['ber']:.2e}  Q={rep['q_db']:.2f}  MACs/sym={rep['macs_per_symbol']}")

    payload = {
        "config": link_config(0, 0, POWER_DBM).to_dict(),
        "chunk": CHUNK, "warmup": WARMUP, "delay": DELAY,
        "results": results,
    }
    (RESULTS_DIR / "streaming.json").write_text(json.dumps(payload, indent=2))
    make_figure(results)
    print(f"Saved: {RESULTS_DIR / 'streaming.json'} and {FIGURES_DIR / 'streaming_frontier.png'}")


def make_figure(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # window-based reference points at the same operating point (from the sweep)
    sweep_path = RESULTS_DIR / "power_sweep.json"
    window_pts = {}
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())["sweep"]
        if "+3" in sweep:
            window_pts = {k: v for k, v in sweep["+3"].items() if v["macs_per_symbol"] > 0}

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for name, r in window_pts.items():
        ax.scatter(r["macs_per_symbol"], r["q_db"], s=56, color=COLORS[name], zorder=3)
        ax.annotate(f"{name} (window)", xy=(r["macs_per_symbol"], r["q_db"]),
                    xytext=(6, 5), textcoords="offset points", fontsize=9)
    for name, r in results.items():
        if r["macs_per_symbol"] == 0:
            ax.axhline(r["q_db"], color=COLORS["CDC+CPE"], linewidth=1.2, linestyle="--")
            ax.annotate("CDC+CPE (no equalizer)", xy=(0.02, r["q_db"]),
                        xycoords=("axes fraction", "data"), va="bottom",
                        fontsize=9, color="#444444")
            continue
        ax.scatter(r["macs_per_symbol"], r["q_db"], s=56, color=COLORS["CfC"],
                   marker="D", zorder=3)
        ax.annotate(name, xy=(r["macs_per_symbol"], r["q_db"]),
                    xytext=(6, -11), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("MACs per recovered symbol")
    ax.set_ylabel("Q-factor [dB]")
    ax.set_title("Streaming CfC (diamonds) vs window equalizers\n16QAM, 32 GBd, 12 x 80 km @ 3 dBm", fontsize=10)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "streaming_frontier.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
