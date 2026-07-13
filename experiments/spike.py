"""Spike experiment: first end-to-end accuracy-vs-complexity comparison.

Single-pol 16QAM, 32 GBd, 12 x 80 km SSMF at 3 dBm launch power, receiver chain
CDC + genie CPE. A strongly nonlinear operating point; residual distortion is ~97%
deterministic (see docs/notes/spike_v1_diagnosis.md). Every equalizer sees
identical data, optimizer budget, and metrics.

Run from the repo root:  python experiments/spike.py
Outputs: results/spike_results.json, docs/figures/spike_*.png
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fluidnn.channel.dataset import make_windows, to_real_features
from fluidnn.channel.link import LinkConfig, simulate_link
from fluidnn.metrics import equalizer_report
from fluidnn.models import BiLSTMEqualizer, CfCEqualizer, MLPEqualizer
from fluidnn.training.harness import evaluate_equalizer, train_equalizer

HALF_WINDOW = 20  # -> 41-symbol windows
EPOCHS = 50
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures"

# Validated categorical palette (dataviz skill, light mode) + neutral baseline.
COLORS = {"CDC+CPE": "#666666", "MLP": "#2a78d6", "BiLSTM": "#1baf7a", "CfC": "#eda100"}


def link_config(n_symbols: int, seed: int) -> LinkConfig:
    return LinkConfig(
        mod_order=16,
        n_symbols=n_symbols,
        symbol_rate=32e9,
        launch_power_dbm=3.0,
        n_spans=12,
        steps_per_span=40,
        seed=seed,
    )


def build_dataset(n_symbols: int, seed: int) -> dict:
    r = simulate_link(link_config(n_symbols, seed))
    x_c, y_c = make_windows(r["rx_symbols"], r["tx_symbols"], HALF_WINDOW)
    r["x"] = to_real_features(x_c, power_feature=True)  # (n, T, 3): I, Q, |x|^2
    r["y"] = np.stack([y_c.real, y_c.imag], axis=-1).astype(np.float32)
    return r


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Simulating training link (2^17 symbols) ...")
    train = build_dataset(2**17, seed=11)
    print("Simulating test link (2^15 symbols, independent data + noise) ...")
    test = build_dataset(2**15, seed=22)

    n_val = len(train["x"]) // 10
    x_tr, y_tr = train["x"][:-n_val], train["y"][:-n_val]
    x_val, y_val = train["x"][-n_val:], train["y"][-n_val:]

    results: dict[str, dict] = {}

    cdc_report = equalizer_report(
        test["rx_symbols"], test["tx_symbols"], test["tx_bits"], test["qam"]
    )
    cdc_report.update(params=0, macs_per_symbol=0)
    results["CDC+CPE"] = cdc_report
    print(f"\nCDC+CPE: BER={cdc_report['ber']:.2e}  Q={cdc_report['q_db']:.2f} dB")

    window_len = 2 * HALF_WINDOW + 1
    models = {
        "MLP": MLPEqualizer(window_len, hidden=(256, 128), in_channels=3),
        "BiLSTM": BiLSTMEqualizer(window_len, hidden=48, in_channels=3),
        "CfC": CfCEqualizer(window_len, hidden=32, backbone_units=64, in_channels=3),
    }
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        history = train_equalizer(
            model, x_tr, y_tr, x_val, y_val, epochs=EPOCHS, batch_size=1024,
            lr=3e-3, lr_min=1e-4, seed=0, verbose=True
        )
        report = evaluate_equalizer(
            model, test["x"], test["tx_symbols"], test["tx_bits"], test["qam"]
        )
        report["train_seconds"] = round(history["train_seconds"], 1)
        results[name] = report
        print(
            f"{name}: BER={report['ber']:.2e}  Q={report['q_db']:.2f} dB"
            f"  params={report['params']}  MACs/sym={report['macs_per_symbol']}"
        )

    print("\n=== Summary (test set) ===")
    print(f"{'model':10s} {'BER':>10s} {'Q [dB]':>8s} {'EVM %':>7s} {'params':>8s} {'MACs/sym':>9s}")
    for name, r in results.items():
        print(
            f"{name:10s} {r['ber']:10.2e} {r['q_db']:8.2f} {r['evm_percent']:7.1f}"
            f" {r['params']:8d} {r['macs_per_symbol']:9d}"
        )
    payload = {"config": link_config(0, 0).to_dict(), "half_window": HALF_WINDOW, "results": results}
    (RESULTS_DIR / "spike_results.json").write_text(json.dumps(payload, indent=2))

    make_figures(results, test, models)
    print(f"\nSaved: {RESULTS_DIR / 'spike_results.json'} and figures in {FIGURES_DIR}")


def make_figures(results: dict, test: dict, models: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    # ---- constellations: CDC-only vs CfC-equalized -------------------------
    with torch.no_grad():
        pred = models["CfC"](torch.from_numpy(test["x"][:20000])).numpy()
    eq = pred[:, 0] + 1j * pred[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharex=True, sharey=True)
    for ax, data, title in (
        (axes[0], test["rx_symbols"][:20000], "CDC + CPE (input to equalizer)"),
        (axes[1], eq, "CfC equalized"),
    ):
        ax.scatter(data.real, data.imag, s=1, alpha=0.12, color="#2a78d6", linewidths=0)
        c = test["qam"].constellation
        ax.scatter(c.real, c.imag, s=14, color="#1a1a19", marker="x", linewidths=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("In-phase")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Quadrature")
    fig.suptitle("16QAM after 12x80 km at 3 dBm", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "spike_constellations.png", dpi=160)
    plt.close(fig)

    # ---- accuracy vs complexity --------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, r in results.items():
        if r["macs_per_symbol"] == 0:
            ax.axhline(r["q_db"], color=COLORS[name], linewidth=1.2, linestyle="--")
            ax.annotate(
                f"{name} (no equalizer)",
                xy=(0.02, r["q_db"]),
                xycoords=("axes fraction", "data"),
                va="bottom",
                fontsize=9,
                color="#444444",
            )
            continue
        ax.scatter(r["macs_per_symbol"], r["q_db"], s=64, color=COLORS[name], zorder=3)
        ax.annotate(
            name,
            xy=(r["macs_per_symbol"], r["q_db"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
        )
    # Streaming-mode CfC complexity is reported in the JSON only: its Q-factor
    # must be measured with a trained seq-to-seq model before it can be plotted.
    ax.set_xscale("log")
    ax.set_xlabel("MACs per recovered symbol")
    ax.set_ylabel("Q-factor [dB]")
    ax.set_title("Equalization quality vs computational cost", fontsize=12)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "spike_q_vs_macs.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
