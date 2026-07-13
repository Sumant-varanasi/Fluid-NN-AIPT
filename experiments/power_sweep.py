"""Launch-power sweep: the standard Q-vs-power benchmark, per-model best recipes.

Each model is trained and tested independently at every launch power (single-pol
16QAM, 32 GBd, 12 x 80 km, CDC + genie CPE). Recipes differ per model -- each
gets its best-known input encoding and optimizer settings, all probed under the
same budget (see docs/RESEARCH_LOG.md for the probes):

- MLP:    I/Q inputs, lr 1e-3 cosine, weight decay 1e-4 (power feature and hot
          LR both measurably hurt its BER)
- BiLSTM: I/Q + |x|^2 inputs, lr 3e-3 cosine, 80 epochs
- CfC:    bidirectional, backbone 64; input encoding set by IQ_ONLY_CFC below
Trained checkpoints are saved per (model, power) for the adaptivity study.

Run from the repo root:  python experiments/power_sweep.py
Outputs: results/power_sweep.json, results/checkpoints/*.pt,
         docs/figures/power_sweep_q.png
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fluidnn.channel.dataset import make_windows, to_real_features
from fluidnn.channel.link import LinkConfig, simulate_link
from fluidnn.metrics import equalizer_report
from fluidnn.models import BiLSTMEqualizer, CfCEqualizer, MLPEqualizer
from fluidnn.training.harness import evaluate_equalizer, train_equalizer

HALF_WINDOW = 20
POWERS_DBM = [-1.0, 1.0, 3.0, 5.0]
IQ_ONLY_CFC = True  # set from the feature probe result
ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CKPT_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR = ROOT / "docs" / "figures"

COLORS = {"CDC+CPE": "#666666", "MLP": "#2a78d6", "BiLSTM": "#1baf7a", "CfC": "#eda100"}


def link_config(n_symbols: int, seed: int, power_dbm: float) -> LinkConfig:
    return LinkConfig(
        mod_order=16,
        n_symbols=n_symbols,
        symbol_rate=32e9,
        launch_power_dbm=power_dbm,
        n_spans=12,
        steps_per_span=40,
        seed=seed,
    )


def build(n_symbols: int, seed: int, power_dbm: float) -> dict:
    r = simulate_link(link_config(n_symbols, seed, power_dbm))
    x_c, y_c = make_windows(r["rx_symbols"], r["tx_symbols"], HALF_WINDOW)
    r["x_iq"] = to_real_features(x_c, power_feature=False)
    r["x_iqp"] = to_real_features(x_c, power_feature=True)
    r["y"] = np.stack([y_c.real, y_c.imag], axis=-1).astype(np.float32)
    return r


def model_zoo(window_len: int) -> dict:
    cfc_channels = 2 if IQ_ONLY_CFC else 3
    return {
        "MLP": dict(
            model=MLPEqualizer(window_len, hidden=(256, 128), in_channels=2),
            features="x_iq",
            train=dict(epochs=50, batch_size=1024, lr=1e-3, lr_min=1e-5, weight_decay=1e-4),
        ),
        "BiLSTM": dict(
            model=BiLSTMEqualizer(window_len, hidden=48, in_channels=3),
            features="x_iqp",
            train=dict(epochs=80, batch_size=1024, lr=3e-3, lr_min=1e-4),
        ),
        "CfC": dict(
            model=CfCEqualizer(
                window_len, hidden=32, backbone_units=64, in_channels=cfc_channels
            ),
            features="x_iq" if IQ_ONLY_CFC else "x_iqp",
            train=dict(epochs=80, batch_size=1024, lr=3e-3, lr_min=1e-4),
        ),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sweep: dict[str, dict] = {}

    for power in POWERS_DBM:
        print(f"\n########## {power:+.0f} dBm ##########")
        train = build(2**17, seed=11, power_dbm=power)
        test = build(2**15, seed=22, power_dbm=power)
        n_val = len(train["y"]) // 10

        entry: dict[str, dict] = {}
        base = equalizer_report(
            test["rx_symbols"], test["tx_symbols"], test["tx_bits"], test["qam"]
        )
        base.update(params=0, macs_per_symbol=0)
        entry["CDC+CPE"] = base
        print(f"CDC+CPE: BER={base['ber']:.2e}  Q={base['q_db']:.2f} dB")

        for name, spec in model_zoo(2 * HALF_WINDOW + 1).items():
            model, feats = spec["model"], spec["features"]
            print(f"-- training {name} ...")
            train_equalizer(
                model,
                train[feats][:-n_val],
                train["y"][:-n_val],
                train[feats][-n_val:],
                train["y"][-n_val:],
                seed=0,
                verbose=False,
                **spec["train"],
            )
            report = evaluate_equalizer(
                model, test[feats], test["tx_symbols"], test["tx_bits"], test["qam"]
            )
            entry[name] = report
            torch.save(model.state_dict(), CKPT_DIR / f"{name}_p{power:+.0f}dBm.pt")
            print(f"{name}: BER={report['ber']:.2e}  Q={report['q_db']:.2f} dB")

        sweep[f"{power:+.0f}"] = entry
        # persist incrementally so a crash loses nothing
        payload = {
            "config": link_config(0, 0, 0.0).to_dict(),
            "half_window": HALF_WINDOW,
            "iq_only_cfc": IQ_ONLY_CFC,
            "powers_dbm": POWERS_DBM,
            "sweep": sweep,
        }
        (RESULTS_DIR / "power_sweep.json").write_text(json.dumps(payload, indent=2))

    make_figure(sweep)
    print(f"\nSaved: {RESULTS_DIR / 'power_sweep.json'} and {FIGURES_DIR / 'power_sweep_q.png'}")


def make_figure(sweep: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    powers = [float(p) for p in sweep]
    for name in ("CDC+CPE", "MLP", "BiLSTM", "CfC"):
        qs = [sweep[p][name]["q_db"] for p in sweep]
        style = dict(linestyle="--", linewidth=1.4) if name == "CDC+CPE" else dict(linewidth=1.8)
        ax.plot(powers, qs, marker="o", markersize=5, color=COLORS[name], label=name, **style)
    ax.set_xlabel("Launch power [dBm]")
    ax.set_ylabel("Q-factor [dB]")
    ax.set_title("Equalizer gain across the nonlinear regime\n16QAM, 32 GBd, 12 x 80 km", fontsize=11)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "power_sweep_q.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
