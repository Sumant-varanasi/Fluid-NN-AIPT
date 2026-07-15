"""Train mixed-condition ("robust") checkpoints for transfer to real captures.

Real experimental data arrives with an unknown operating point (launch power,
effective distance, OSNR). A model trained at one simulated condition transfers
with a gap; a model trained on data *pooled across conditions* learns
condition-robust features and is the better initialization for fine-tuning on
a small real dataset (the expected regime: the group shares "a little" data).

Pools single-pol training links at {+1, +3, +5} dBm and dual-pol static links
at {+3, +5} dBm total. Window models only (fine-tuning targets).

Run from the repo root:  python experiments/robust_checkpoints.py
Outputs: results/checkpoints_robust/*.pt, results/robust_checkpoints.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fluidnn.channel.dataset import make_dp_windows, make_windows, to_real_features
from fluidnn.channel.link import LinkConfig, simulate_link
from fluidnn.channel.link_dp import DPLinkConfig, simulate_dp_link
from fluidnn.metrics import equalizer_report
from fluidnn.models import BiLSTMEqualizer, CfCEqualizer, MLPEqualizer
from fluidnn.training.harness import evaluate_equalizer, train_equalizer

HALF_WINDOW = 20
T = 2 * HALF_WINDOW + 1
CKPT_DIR = ROOT / "results" / "checkpoints_robust"


def pooled_single_pol(n_per_power: int, seed0: int) -> dict:
    xs_iq, xs_iqp, ys, tests = [], [], [], {}
    for i, p in enumerate((1.0, 3.0, 5.0)):
        r = simulate_link(
            LinkConfig(n_symbols=n_per_power, launch_power_dbm=p, n_spans=12,
                       steps_per_span=40, seed=seed0 + i)
        )
        xc, yc = make_windows(r["rx_symbols"], r["tx_symbols"], HALF_WINDOW)
        xs_iq.append(to_real_features(xc, False))
        xs_iqp.append(to_real_features(xc, True))
        ys.append(np.stack([yc.real, yc.imag], -1).astype(np.float32))
        tests[p] = r
    rng = np.random.default_rng(0)
    order = rng.permutation(sum(len(y) for y in ys))
    return {
        "x_iq": np.concatenate(xs_iq)[order],
        "x_iqp": np.concatenate(xs_iqp)[order],
        "y": np.concatenate(ys)[order],
    }


def main() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {"single_pol": {}, "dual_pol": {}}

    # ---------------- single-pol robust models ----------------
    print("== pooling single-pol data ({+1,+3,+5} dBm) ==")
    pool = pooled_single_pol(2**16, seed0=41)
    n_val = len(pool["y"]) // 10
    zoo = {
        "MLP": (MLPEqualizer(T, (256, 128), in_channels=2), "x_iq",
                dict(epochs=50, batch_size=1024, lr=1e-3, lr_min=1e-5, weight_decay=1e-4)),
        "BiLSTM": (BiLSTMEqualizer(T, 48, in_channels=3), "x_iqp",
                   dict(epochs=50, batch_size=1024, lr=3e-3, lr_min=1e-4)),
        "CfC": (CfCEqualizer(T, 32, backbone_units=64, in_channels=3), "x_iqp",
                dict(epochs=50, batch_size=1024, lr=3e-3, lr_min=1e-4)),
    }
    for name, (model, feats, recipe) in zoo.items():
        print(f"-- training robust {name} (single-pol) ...")
        x = pool[feats]
        train_equalizer(model, x[:-n_val], pool["y"][:-n_val], x[-n_val:], pool["y"][-n_val:],
                        seed=0, verbose=False, **recipe)
        torch.save(model.state_dict(), CKPT_DIR / f"{name}_robust_sp.pt")
        # sanity: evaluate on a fresh +3 dBm test link
        t = simulate_link(LinkConfig(n_symbols=2**15, launch_power_dbm=3.0, n_spans=12,
                                     steps_per_span=40, seed=22))
        xc, _ = make_windows(t["rx_symbols"], t["tx_symbols"], HALF_WINDOW)
        xt = to_real_features(xc, feats == "x_iqp")
        rep = evaluate_equalizer(model, xt, t["tx_symbols"], t["tx_bits"], t["qam"])
        summary["single_pol"][name] = rep["q_db"]
        print(f"   robust {name} @ +3 dBm test: Q={rep['q_db']:.2f}")

    # ---------------- dual-pol robust models ----------------
    print("== pooling dual-pol data ({+3,+5} dBm total, static RSOP) ==")
    xs, ys = [], []
    for i, p in enumerate((3.0, 5.0)):
        r = simulate_dp_link(DPLinkConfig(n_symbols=2**16, launch_power_dbm=p,
                                          steps_per_span=40, seed=51 + i))
        x, y = make_dp_windows(r["rx_symbols"], r["tx_symbols"], HALF_WINDOW, power_feature=True)
        xs.append(x)
        ys.append(y)
    rng = np.random.default_rng(0)
    order = rng.permutation(sum(len(y) for y in ys))
    x, y = np.concatenate(xs)[order], np.concatenate(ys)[order]
    n_val = len(y) // 10
    dp_zoo = {
        "BiLSTM": BiLSTMEqualizer(T, 48, in_channels=6, out_channels=4),
        "CfC": CfCEqualizer(T, 32, backbone_units=64, in_channels=6, out_channels=4),
    }
    for name, model in dp_zoo.items():
        print(f"-- training robust {name} (dual-pol) ...")
        train_equalizer(model, x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:],
                        epochs=50, batch_size=1024, lr=3e-3, lr_min=1e-4, seed=0, verbose=False)
        torch.save(model.state_dict(), CKPT_DIR / f"{name}_robust_dp.pt")
        summary["dual_pol"][name] = "saved"
        print(f"   saved {name}_robust_dp.pt")

    (ROOT / "results" / "robust_checkpoints.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )
    print("done")


if __name__ == "__main__":
    main()
