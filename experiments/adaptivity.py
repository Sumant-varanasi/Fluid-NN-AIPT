"""Adaptivity study: how equalizers trained at one channel condition survive drift.

Two questions, run after power_sweep.py has produced checkpoints:

1. **Frozen robustness.** Take each model trained at +3 dBm / 12 spans and
   evaluate it, frozen, across launch-power drift (-1..+5 dBm) and distance
   drift (10..14 spans). Compare against the matched (trained-at-that-power)
   results from the sweep. The gap curve is the robustness result.
2. **Cost of adaptation.** At the drift point where the frozen model loses the
   most, fine-tune on a small number of pilot symbols (1k/4k/16k) and measure
   how much of the matched performance returns. This is the "how many pilots
   does re-tuning cost" number that matters for deployment.

Run from the repo root:  python experiments/adaptivity.py
Outputs: results/adaptivity.json, docs/figures/adaptivity_drift.png
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

from power_sweep import HALF_WINDOW, CKPT_DIR, FIGURES_DIR, RESULTS_DIR, link_config, model_zoo

from fluidnn.channel.dataset import make_windows, to_real_features
from fluidnn.channel.link import simulate_link
from fluidnn.metrics import equalizer_report
from fluidnn.training.harness import evaluate_equalizer, train_equalizer

TRAIN_POWER = 3.0  # the reference checkpoint
POWER_GRID = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
SPAN_GRID = [10, 11, 12, 13, 14]
PILOT_SIZES = [1024, 4096, 16384]
COLORS = {"CDC+CPE": "#666666", "MLP": "#2a78d6", "BiLSTM": "#1baf7a", "CfC": "#eda100"}


def build_test(power_dbm: float, n_spans: int, seed: int = 22, n_symbols: int = 2**15) -> dict:
    cfg = link_config(n_symbols, seed, power_dbm)
    cfg.n_spans = n_spans
    r = simulate_link(cfg)
    x_c, y_c = make_windows(r["rx_symbols"], r["tx_symbols"], HALF_WINDOW)
    r["x_iq"] = to_real_features(x_c, power_feature=False)
    r["x_iqp"] = to_real_features(x_c, power_feature=True)
    r["y"] = np.stack([y_c.real, y_c.imag], axis=-1).astype(np.float32)
    return r


def load_models() -> dict:
    zoo = model_zoo(2 * HALF_WINDOW + 1)
    for name, spec in zoo.items():
        state = torch.load(CKPT_DIR / f"{name}_p{TRAIN_POWER:+.0f}dBm.pt", weights_only=True)
        spec["model"].load_state_dict(state)
    return zoo


def frozen_eval(zoo: dict, test: dict) -> dict:
    out = {}
    base = equalizer_report(test["rx_symbols"], test["tx_symbols"], test["tx_bits"], test["qam"])
    out["CDC+CPE"] = base
    for name, spec in zoo.items():
        out[name] = evaluate_equalizer(
            spec["model"], test[spec["features"]], test["tx_symbols"], test["tx_bits"], test["qam"]
        )
    return out


def main() -> None:
    results = {"train_power_dbm": TRAIN_POWER, "power_drift": {}, "span_drift": {}, "adaptation": {}}

    print("== frozen evaluation: power drift ==")
    for p in POWER_GRID:
        zoo = load_models()  # fresh load each condition (no state bleed)
        rep = frozen_eval(zoo, build_test(p, 12))
        results["power_drift"][f"{p:+.0f}"] = {k: v for k, v in rep.items()}
        print(f"  {p:+.0f} dBm: " + "  ".join(f"{k} Q={v['q_db']:.2f}" for k, v in rep.items()))

    print("== frozen evaluation: distance drift ==")
    for s in SPAN_GRID:
        zoo = load_models()
        rep = frozen_eval(zoo, build_test(TRAIN_POWER, s))
        results["span_drift"][str(s)] = {k: v for k, v in rep.items()}
        print(f"  {s} spans: " + "  ".join(f"{k} Q={v['q_db']:.2f}" for k, v in rep.items()))

    # ---- cost of adaptation at the worst power-drift point --------------------
    sweep = json.loads((RESULTS_DIR / "power_sweep.json").read_text())["sweep"]
    worst_p, worst_gap = None, -1.0
    for p_str, entry in sweep.items():
        p = float(p_str)
        if f"{p:+.0f}" in results["power_drift"] and p != TRAIN_POWER:
            for name in ("MLP", "BiLSTM", "CfC"):
                gap = entry[name]["q_db"] - results["power_drift"][f"{p:+.0f}"][name]["q_db"]
                if gap > worst_gap:
                    worst_gap, worst_p = gap, p
    print(f"== adaptation study at {worst_p:+.0f} dBm (largest frozen gap {worst_gap:.2f} dB) ==")

    pilots_full = build_test(worst_p, 12, seed=33, n_symbols=2**15)
    test = build_test(worst_p, 12, seed=22)
    for n_pilot in PILOT_SIZES:
        zoo = load_models()
        entry = {}
        for name, spec in zoo.items():
            feats = spec["features"]
            xp, yp = pilots_full[feats][:n_pilot], pilots_full["y"][:n_pilot]
            n_val = max(n_pilot // 8, 128)
            train_equalizer(
                spec["model"], xp[:-n_val], yp[:-n_val], xp[-n_val:], yp[-n_val:],
                epochs=15, batch_size=256, lr=3e-4, seed=0, verbose=False,
            )
            entry[name] = evaluate_equalizer(
                spec["model"], test[feats], test["tx_symbols"], test["tx_bits"], test["qam"]
            )
            print(f"  {n_pilot:5d} pilots  {name}: Q={entry[name]['q_db']:.2f}")
        results["adaptation"][str(n_pilot)] = entry
    results["adaptation_power_dbm"] = worst_p

    (RESULTS_DIR / "adaptivity.json").write_text(json.dumps(results, indent=2, default=float))
    make_figure(results, sweep)
    print(f"Saved: {RESULTS_DIR / 'adaptivity.json'} and {FIGURES_DIR / 'adaptivity_drift.png'}")


def make_figure(results: dict, sweep: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    powers = sorted(float(p) for p in results["power_drift"])
    for name in ("CDC+CPE", "MLP", "BiLSTM", "CfC"):
        qs = [results["power_drift"][f"{p:+.0f}"][name]["q_db"] for p in powers]
        ls = "--" if name == "CDC+CPE" else "-"
        ax.plot(powers, qs, marker="o", ms=4, color=COLORS[name], linestyle=ls, label=f"{name} (frozen)")
    for name in ("MLP", "BiLSTM", "CfC"):  # matched reference from the sweep
        ps = sorted(float(p) for p in sweep)
        qs = [sweep[f"{p:+.0f}"][name]["q_db"] for p in ps]
        ax.plot(ps, qs, marker="s", ms=4, color=COLORS[name], linestyle=":", alpha=0.6)
    ax.axvline(results["train_power_dbm"], color="#999999", linewidth=0.8)
    ax.set_xlabel("Test launch power [dBm]")
    ax.set_ylabel("Q-factor [dB]")
    ax.set_title("Power drift: frozen (solid) vs matched retrain (dotted)", fontsize=10)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    spans = sorted(int(s) for s in results["span_drift"])
    for name in ("CDC+CPE", "MLP", "BiLSTM", "CfC"):
        qs = [results["span_drift"][str(s)][name]["q_db"] for s in spans]
        ls = "--" if name == "CDC+CPE" else "-"
        ax.plot(spans, qs, marker="o", ms=4, color=COLORS[name], linestyle=ls, label=name)
    ax.axvline(12, color="#999999", linewidth=0.8)
    ax.set_xlabel("Test link length [spans of 80 km]")
    ax.set_title("Distance drift (trained at 12 spans, frozen)", fontsize=10)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "adaptivity_drift.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
