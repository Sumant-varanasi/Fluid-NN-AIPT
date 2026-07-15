"""Evaluate equalizers on an experimental capture.

Usage (from the repo root):

  1. Inspect an unknown file first:
       python experiments/real_eval.py path/to/capture.mat
     (prints every array with shape/dtype so the rx/tx keys can be identified)

  2. Run the evaluation:
       python experiments/real_eval.py path/to/capture.mat --rx rxSig --tx txSig
     Optional: --dual-pol if arrays are (2, N); --mod 16; --half-window 20;
               --no-cpe; --checkpoint results/checkpoints/CfC_p+3dBm.pt

Pipeline: load -> align (delay/gain/conjugation/pol-swap) -> optional sliding
CPE -> chronological 70/30 train/test split -> train window equalizers on the
train part -> report BER/Q on the held-out part, next to the no-equalizer
baseline. Time-ordered split, no shuffling across the boundary: the test data
is strictly later in time than anything trained on.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fluidnn.channel.dataset import make_dp_windows, make_windows, to_real_features
from fluidnn.channel.modulation import QAM
from fluidnn.channel.receiver import cpe_sliding
from fluidnn.metrics import equalizer_report
from fluidnn.models import BiLSTMEqualizer, CfCEqualizer, MLPEqualizer
from fluidnn.realdata import align_dual_pol, align_single, describe_capture, load_capture
from fluidnn.training.harness import train_equalizer


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("capture")
    p.add_argument("--rx", help="array key of the received sequence")
    p.add_argument("--tx", help="array key of the transmitted sequence")
    p.add_argument("--dual-pol", action="store_true")
    p.add_argument("--mod", type=int, default=16, help="QAM order (default 16)")
    p.add_argument("--half-window", type=int, default=20)
    p.add_argument("--no-cpe", action="store_true")
    p.add_argument("--cpe-window", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--checkpoints-dir",
        default="results/checkpoints_robust",
        help="if robust checkpoints exist here, each model is ALSO fine-tuned "
        "from its checkpoint (small captures rarely support from-scratch training)",
    )
    p.add_argument("--out", default="results/real_eval.json")
    return p.parse_args()


def main() -> None:
    args = get_args()
    arrays = load_capture(args.capture)
    if not args.rx or not args.tx:
        print(f"Arrays in {args.capture}:\n{describe_capture(arrays)}")
        print("\nRe-run with --rx <key> --tx <key>.")
        return

    rx = np.squeeze(np.asarray(arrays[args.rx], dtype=complex))
    tx = np.squeeze(np.asarray(arrays[args.tx], dtype=complex))
    qam = QAM(args.mod)

    # ---- align capture conventions -----------------------------------------
    if args.dual_pol:
        aligned = align_dual_pol(rx.reshape(2, -1), tx.reshape(2, -1))
        print(f"alignment: swapped={aligned['swapped']}  per-pol={aligned['per_pol']}")
        rx, tx = aligned["rx"], aligned["tx"]
        if not args.no_cpe:
            for p in range(2):
                rx[p] = cpe_sliding(rx[p], tx[p], args.cpe_window)
    else:
        aligned = align_single(rx, tx)
        print(
            f"alignment: delay={aligned['delay']}  conjugated={aligned['conjugated']}"
            f"  gain={aligned['gain']:.3f}  residual NMSE={aligned['nmse_db']:.1f} dB"
        )
        rx, tx = aligned["rx"], aligned["tx"]
        if not args.no_cpe:
            rx = cpe_sliding(rx, tx, args.cpe_window)

    # regenerate the bit labels from the aligned tx symbols
    tx_bits = (
        np.stack([qam.decide_bits(tx[p]) for p in range(2)])
        if args.dual_pol
        else qam.decide_bits(tx)
    )

    # ---- windows + chronological split -------------------------------------
    T = 2 * args.half_window + 1
    if args.dual_pol:
        x, y = make_dp_windows(rx, tx, args.half_window, power_feature=True)
        models = {
            "BiLSTM": BiLSTMEqualizer(T, 48, in_channels=6, out_channels=4),
            "CfC": CfCEqualizer(T, 32, backbone_units=64, in_channels=6, out_channels=4),
        }
    else:
        xc, yc = make_windows(rx, tx, args.half_window)
        x_iq = to_real_features(xc, power_feature=False)
        x_iqp = to_real_features(xc, power_feature=True)
        y = np.stack([yc.real, yc.imag], -1).astype(np.float32)
        models = {
            "MLP": MLPEqualizer(T, (256, 128), in_channels=2),
            "BiLSTM": BiLSTMEqualizer(T, 48, in_channels=3),
            "CfC": CfCEqualizer(T, 32, backbone_units=64, in_channels=3),
        }
        feats = {"MLP": x_iq, "BiLSTM": x_iqp, "CfC": x_iqp}

    n = len(y)
    split = int(0.7 * n)
    n_val = max(split // 10, 64)

    def test_slice(seq):  # symbol-level slices for reporting
        return seq[..., split:] if args.dual_pol else seq[split:]

    results = {}
    if args.dual_pol:
        from fluidnn.channel.link_dp import dp_report

        base = dp_report(test_slice(rx), test_slice(tx), tx_bits[..., split * qam.bits_per_symbol :], qam)
    else:
        base = equalizer_report(
            rx[split:], tx[split:], tx_bits[split * qam.bits_per_symbol :], qam
        )
    results["no equalizer"] = base
    print(f"no equalizer: BER={base['ber']:.2e}  Q={base['q_db']:.2f} dB")

    import torch

    recipes = {
        "MLP": dict(lr=1e-3, lr_min=1e-5, weight_decay=1e-4),
        "BiLSTM": dict(lr=3e-3, lr_min=1e-4),
        "CfC": dict(lr=3e-3, lr_min=1e-4),
    }
    ckpt_dir = pathlib.Path(args.checkpoints_dir)
    ckpt_suffix = "_robust_dp.pt" if args.dual_pol else "_robust_sp.pt"

    def run_variant(label, model, xf, recipe):
        print(f"-- {label} on the first 70% ...")
        train_equalizer(
            model,
            xf[: split - n_val], y[: split - n_val],
            xf[split - n_val : split], y[split - n_val : split],
            epochs=args.epochs, batch_size=1024, seed=0, verbose=False, **recipe,
        )
        model.eval()
        with torch.no_grad():
            pred = torch.cat(
                [model(torch.from_numpy(xf[i : i + 4096])) for i in range(split, len(xf), 4096)]
            ).numpy()
        if args.dual_pol:
            eq = np.stack([pred[:, 0] + 1j * pred[:, 1], pred[:, 2] + 1j * pred[:, 3]])
            rep = dp_report(eq, test_slice(tx), tx_bits[..., split * qam.bits_per_symbol :], qam)
        else:
            eq = pred[:, 0] + 1j * pred[:, 1]
            rep = equalizer_report(eq, tx[split:], tx_bits[split * qam.bits_per_symbol :], qam)
        rep["params"] = sum(p.numel() for p in model.parameters())
        rep["macs_per_symbol"] = model.macs_per_symbol()
        results[label] = rep
        print(f"{label}: BER={rep['ber']:.2e}  Q={rep['q_db']:.2f} dB")

    for name, model in models.items():
        xf = x if args.dual_pol else feats[name]
        run_variant(f"{name} (scratch)", model, xf, recipes[name])
        ckpt = ckpt_dir / f"{name}{ckpt_suffix}"
        if ckpt.exists():
            import copy

            ft = copy.deepcopy(model)
            ft.load_state_dict(torch.load(ckpt, weights_only=True))
            gentle = {**recipes[name], "lr": recipes[name]["lr"] / 3}
            run_variant(f"{name} (fine-tuned)", ft, xf, gentle)

    out = pathlib.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"capture": str(args.capture), "results": results}, indent=2, default=float))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
