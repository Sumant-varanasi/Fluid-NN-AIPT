# Research log

Running record of experiments, failures, diagnoses, and fixes. Newest at the bottom.
Numbers are test-set results unless stated otherwise; full configs in
`results/spike_results.json` at the corresponding commit.

---

## Spike v1 — pipeline works, nobody beats the baseline

**Setup.** 16QAM, 32 GBd, 20x80 km @ 2 dBm, no CPE, 21-symbol window,
59k train symbols, models predict symbols from scratch.

| model    | BER      | Q [dB] | MACs/sym |
|----------|----------|--------|----------|
| CDC only | 2.47e-02 | 5.87   | 0        |
| MLP      | 2.49e-02 | 5.86   | 13.7k    |
| BiLSTM   | 2.35e-02 | 5.96   | 183k     |
| CfC      | 2.83e-02 | 5.61   | 52k      |

**Diagnosis (measured):** 97% of the residual error is deterministic (ASE-off rerun,
error correlation 0.973) so the task is winnable; a missing CPE stage costs 0.8 dB;
the +/-10 window is far smaller than the +/-100-symbol channel memory at 1600 km.
Details: `notes/spike_v1_diagnosis.md`.

## Spike v2 — MLP and BiLSTM deliver; CfC frozen at the identity

**Changes.** CPE stage added (tested); residual outputs (zero-weight network ==
passthrough, verified); 12x80 km @ 3 dBm; 41-symbol window; 2^17 train symbols;
bigger models; 40 epochs.

| model    | BER      | Q [dB] | MACs/sym |
|----------|----------|--------|----------|
| CDC+CPE  | 8.72e-03 | 7.52   | 0        |
| MLP      | 3.24e-03 | 8.70   | 54k      |
| BiLSTM   | 2.75e-03 | 8.87   | 787k     |
| CfC      | 8.80e-03 | 7.51   | 178k     |

The receiver-chain and task-setup fixes worked: BER cut ~3x by the learned
equalizers. But the CfC's train MSE never moved off the passthrough value.

**Diagnosis.** Two compounding causes:
1. *Readout:* the CfC read only its final state, so the center symbol's
   information had to survive 21 gated updates; the optimizer settled on the
   identity. (BiLSTM reads the center time step and does not have this path.)
2. *Input representation:* a 4-epoch probe grid (backbone / learning rate /
   physics-informed |x|^2 input) showed the |x|^2 input channel was the only
   lever that moved the loss. Kerr distortion is power-driven; forcing the cell
   to synthesize input-times-input products from I/Q alone is what it is worst at.

Also resolved a false alarm: torch MSELoss averages over the two real components,
so its numbers are exactly half the complex-error power. The apparent train/test
MSE mismatch was unit bookkeeping, not a simulator bug (verified: link MSE is
independent of sequence length and seed to <3%).

## Spike v3 — bidirectional CfC + physics-informed inputs (running)

**Changes.**
- Bidirectional CfC, both directions read out **at the center step**; forward and
  backward sweeps meet at the center so the cost equals one full sweep.
- All models receive [I, Q, |x|^2] per symbol (same inputs for everyone: fair).
- CfC gets its backbone layer (64 units) as in the original CfC paper.
- Cosine LR 3e-3 -> 1e-4, 50 epochs, batch 1024.

Pre-flight (3 epochs, quarter-size data): CfC val MSE moving from epoch 1,
already past its entire 40-epoch v2 trajectory. Full results pending.
