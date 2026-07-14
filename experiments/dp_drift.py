"""Dual-polarization RSOP-drift study: the time-varying channel test.

This is the experiment the liquid-network hypothesis actually rides on: a
channel that changes *during* the sequence (endless polarization rotation at a
controlled rate), not just between train and test. Two questions:

1. Models trained on the static channel, evaluated frozen while the channel
   rotates under them at increasing rates -- who degrades most gracefully?
   Streaming models carry state and can in principle *track* the rotation;
   window models must infer it from 41 symbols.
2. Models trained *on* the drifting channel (matched at the highest rate) --
   who learns to be rotation-robust?

Setup: dual-pol 16QAM, 32 GBd, 12 x 80 km Manakov, +5 dBm total launch power,
receiver CDC + genie block demux (block 256) + per-pol genie CPE. The demux
removes all static rotation, so drift-rate 0 is the rotation-free reference;
within-block drift is what remains for the equalizer.

Run from the repo root:  python experiments/dp_drift.py
Outputs: results/dp_drift.json, results/checkpoints_dp/*.pt,
         docs/figures/dp_drift_q.png
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fluidnn.channel.dataset import make_dp_stream_chunks, make_dp_windows
from fluidnn.channel.link_dp import DPLinkConfig, dp_report, simulate_dp_link
from fluidnn.models import (
    BiLSTMEqualizer,
    CfCEqualizer,
    MLPEqualizer,
    StreamingCfCEqualizer,
    StreamingLSTMEqualizer,
)
from fluidnn.training.harness import train_equalizer

HALF_WINDOW = 20
CHUNK, WARMUP, DELAY = 64, 32, 8
POWER_DBM = 5.0
DRIFT_RATES = [0.0, 10.0, 30.0, 100.0]
MATCHED_RATE = 100.0
RESULTS_DIR = ROOT / "results"
CKPT_DIR = RESULTS_DIR / "checkpoints_dp"
FIGURES_DIR = ROOT / "docs" / "figures"

COLORS = {
    "CDC+demux+CPE": "#666666",
    "MLP": "#2a78d6",
    "BiLSTM": "#1baf7a",
    "CfC": "#eda100",
    "StreamLSTM": "#4a3aa7",
    "StreamCfC": "#e34948",
}


def link_config(n_symbols: int, seed: int, drift: float) -> DPLinkConfig:
    return DPLinkConfig(
        n_symbols=n_symbols,
        launch_power_dbm=POWER_DBM,
        steps_per_span=40,
        rsop_drift_deg_per_ksym=drift,
        seed=seed,
    )


def build(n_symbols: int, seed: int, drift: float) -> dict:
    r = simulate_dp_link(link_config(n_symbols, seed, drift))
    rx, tx = r["rx_symbols"], r["tx_symbols"]
    r["w_iq"], r["wy"] = make_dp_windows(rx, tx, HALF_WINDOW, power_feature=False)
    r["w_iqp"], _ = make_dp_windows(rx, tx, HALF_WINDOW, power_feature=True)
    r["s_iqp"], r["sy"] = make_dp_stream_chunks(rx, tx, CHUNK, WARMUP, DELAY, power_feature=True)
    return r


def model_zoo() -> dict:
    """Fresh models with their probed recipes (dual-pol: out_channels=4)."""
    T = 2 * HALF_WINDOW + 1
    return {
        "MLP": dict(
            model=MLPEqualizer(T, hidden=(256, 128), in_channels=4, out_channels=4),
            data=("w_iq", "wy"),
            train=dict(epochs=50, batch_size=1024, lr=1e-3, lr_min=1e-5, weight_decay=1e-4),
        ),
        "BiLSTM": dict(
            model=BiLSTMEqualizer(T, hidden=48, in_channels=6, out_channels=4),
            data=("w_iqp", "wy"),
            train=dict(epochs=50, batch_size=1024, lr=3e-3, lr_min=1e-4),
        ),
        "CfC": dict(
            model=CfCEqualizer(T, hidden=32, backbone_units=64, in_channels=6, out_channels=4),
            data=("w_iqp", "wy"),
            train=dict(epochs=50, batch_size=1024, lr=3e-3, lr_min=1e-4),
        ),
        "StreamLSTM": dict(
            model=StreamingLSTMEqualizer(
                hidden=32, in_channels=6, delay=DELAY, warmup=WARMUP, out_channels=4
            ),
            data=("s_iqp", "sy"),
            train=dict(epochs=200, batch_size=32, lr=3e-3, lr_min=1e-4),
        ),
        "StreamCfC": dict(
            model=StreamingCfCEqualizer(
                hidden=32, backbone_units=64, in_channels=6,
                delay=DELAY, warmup=WARMUP, out_channels=4,
            ),
            data=("s_iqp", "sy"),
            train=dict(epochs=200, batch_size=32, lr=3e-3, lr_min=1e-4),
        ),
    }


def eval_model(name: str, spec: dict, test: dict) -> dict:
    model = spec["model"]
    xk, _ = spec["data"]
    x = test[xk]
    model.eval()
    with torch.no_grad():
        batch = 32 if xk.startswith("s_") else 4096
        pred = torch.cat(
            [model(torch.from_numpy(x[i : i + batch])) for i in range(0, len(x), batch)]
        ).numpy()

    n = test["tx_symbols"].shape[1]
    if xk.startswith("s_"):  # streaming: reassemble chunk outputs into sequences
        starts = np.arange(0, n, CHUNK)
        y_idx = (starts[:, None] + np.arange(CHUNK)[None, :] - DELAY) % n
        eq = np.zeros((2, n), dtype=complex)
        eq[0][y_idx.reshape(-1)] = (pred[..., 0] + 1j * pred[..., 1]).reshape(-1)
        eq[1][y_idx.reshape(-1)] = (pred[..., 2] + 1j * pred[..., 3]).reshape(-1)
    else:
        eq = np.stack([pred[:, 0] + 1j * pred[:, 1], pred[:, 2] + 1j * pred[:, 3]])

    rep = dp_report(eq, test["tx_symbols"], test["tx_bits"], test["qam"])
    rep["params"] = sum(p.numel() for p in model.parameters())
    rep["macs_per_symbol"] = model.macs_per_symbol()
    return rep


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results: dict = {"power_dbm": POWER_DBM, "drift_rates": DRIFT_RATES,
                     "matched_rate": MATCHED_RATE, "frozen": {}, "matched": {}, "baseline": {}}
    out_path = RESULTS_DIR / "dp_drift.json"

    print("== simulating test sets ==")
    tests = {rate: build(2**15, seed=22, drift=rate) for rate in DRIFT_RATES}
    for rate, t in tests.items():
        rep = dp_report(t["rx_symbols"], t["tx_symbols"], t["tx_bits"], t["qam"])
        results["baseline"][f"{rate:.0f}"] = rep
        print(f"  baseline @ drift {rate:.0f}: Q={rep['q_db']:.2f}")

    for cond, train_drift in (("frozen", 0.0), ("matched", MATCHED_RATE)):
        print(f"== training on drift={train_drift:.0f} deg/ksym ({cond}) ==")
        train = build(2**17, seed=11, drift=train_drift)
        for name, spec in model_zoo().items():
            xk, yk = spec["data"]
            x, y = train[xk], train[yk]
            n_val = max(len(x) // 10, 8)
            print(f"-- training {name} ...")
            train_equalizer(
                spec["model"], x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:],
                seed=0, verbose=False, **spec["train"],
            )
            torch.save(spec["model"].state_dict(), CKPT_DIR / f"{name}_{cond}.pt")
            results[cond][name] = {}
            for rate in DRIFT_RATES:
                rep = eval_model(name, spec, tests[rate])
                results[cond][name][f"{rate:.0f}"] = rep
                print(f"   {name} @ drift {rate:.0f}: Q={rep['q_db']:.2f}")
            out_path.write_text(json.dumps(results, indent=2, default=float))

    make_figure(results)
    print(f"Saved: {out_path} and {FIGURES_DIR / 'dp_drift_q.png'}")


def make_figure(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    rates = [float(r) for r in DRIFT_RATES]

    for ax, cond, title in (
        (axes[0], "frozen", "Trained on static channel (frozen)"),
        (axes[1], "matched", f"Trained on drift {MATCHED_RATE:.0f} deg/ksym"),
    ):
        qs = [results["baseline"][f"{r:.0f}"]["q_db"] for r in rates]
        ax.plot(rates, qs, marker="o", ms=4, color=COLORS["CDC+demux+CPE"],
                linestyle="--", label="CDC+demux+CPE")
        for name in ("MLP", "BiLSTM", "CfC", "StreamLSTM", "StreamCfC"):
            if name not in results[cond]:
                continue
            qs = [results[cond][name][f"{r:.0f}"]["q_db"] for r in rates]
            ax.plot(rates, qs, marker="o", ms=4, color=COLORS[name], label=name)
        ax.set_xlabel("RSOP drift rate [deg / 1000 symbols]")
        ax.set_title(title, fontsize=10)
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Q-factor [dB]")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Dual-pol 16QAM, 32 GBd, 12 x 80 km @ +5 dBm total", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dp_drift_q.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
