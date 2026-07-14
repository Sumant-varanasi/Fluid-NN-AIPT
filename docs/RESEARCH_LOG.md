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
already past its entire 40-epoch v2 trajectory.

**Results.**

| model    | BER      | Q [dB] | EVM % | params | MACs/sym |
|----------|----------|--------|-------|--------|----------|
| CDC+CPE  | 8.72e-03 | 7.52   | 18.2  | 0      | 0        |
| MLP      | 6.55e-03 | 7.89   | 9.7   | 64.9k  | 64.5k    |
| BiLSTM   | 2.37e-03 | 9.02   | 13.4  | 20.5k  | 803k     |
| CfC      | 4.10e-03 | 8.44   | 7.6   | 21.4k  | 438k     |

**The CfC equalizer now works**: +0.92 dB Q over the baseline, better BER than the
MLP, within 0.6 dB of the BiLSTM at 1.8x fewer MACs/symbol and equal parameter
count. Constellation figure: docs/figures/spike_constellations.png.

**Observations for the next iteration.**
1. *MSE and BER rank models differently.* The CfC has by far the lowest EVM/MSE
   (7.6% vs BiLSTM 13.4%) yet a 1.7x worse BER: its error distribution is
   heavier-tailed -- rare large errors flip decisions while the bulk of symbols
   sit tighter than the BiLSTM's. Visible as faint bridges between constellation
   points in the figure. Worth a dedicated study (loss shaping / robust losses).
2. *MLP overfit hard* under the hotter v3 recipe (train MSE 0.0005 vs val 0.005;
   BER regressed vs v2's 3.24e-3). Per-model regularization/early-stopping needed
   -- one shared recipe is no longer the fair choice once models differ this much.
3. *BiLSTM was still improving* at epoch 50 (val MSE falling monotonically);
   its ceiling is not yet reached. Same for CfC (train/val gap still small).

**Next:** per-model training protocols, then the launch-power sweep and the
adaptivity (channel-drift) study; a causal/streaming CfC variant for the
real-time complexity story (368 MACs/symbol at h=8 -- three orders below the
window-based BiLSTM).

## Input-encoding probes -- the |x|^2 feature is model-dependent, and the MLP wakes up

All at 3 dBm, 12x80 km, same data and budgets as v3 unless noted.

| model | inputs | recipe | BER | Q [dB] |
|-------|--------|--------|-----|--------|
| MLP | I,Q,\|x\|^2 | any of 4 probed | 6.6-6.9e-03 | ~7.85 |
| MLP | I,Q only | lr 1e-3 cos, wd 1e-4 | **1.63e-03** | **9.37** |
| CfC | I,Q only | v3 recipe, 50 ep | 5.91e-03 | 8.02 |
| CfC | I,Q,\|x\|^2 | v3 recipe, 50 ep | 4.10e-03 | 8.44 |

Findings:
1. **The power feature is a trap for the MLP but a help for the CfC.** With
   \|x\|^2 the MLP reaches *lower MSE but 4x worse BER*: received power acts as
   a prior on the transmitted ring, so borderline symbols get confidently pulled
   to the wrong ring (heavy tails). The recurrent CfC integrates power over the
   window instead of reading it pointwise and net-benefits. Encoding choices are
   therefore per-model, each probed under equal budgets.
2. **A properly regularized MLP currently leads the board** (Q 9.37 vs BiLSTM
   9.02 at 13x fewer MACs). Humbling and worth reporting honestly: at this
   operating point, window MLPs are strong. The recurrent models have not had
   an equivalent tuning round yet -- BiLSTM encoding probe running; CfC capacity/
   depth tuning still to do.
3. MSE ranks models differently from BER throughout -- any model selection or
   early stopping should eventually switch to a BER-aligned criterion.
4. BiLSTM addendum (80-epoch probe): encoding-insensitive -- IQ-only Q 8.95 vs
   IQ+|x|^2 Q 8.99, a tie within run-to-run noise. Sweep keeps |x|^2 for it.
   Also: 80 epochs at lr 3e-3 does not beat the 50-epoch v3 number (9.02) --
   the BiLSTM's remaining headroom is smaller than its epoch-50 slope suggested.

## Launch-power sweep -- gains across the whole nonlinear regime

Per-model best recipes, trained and tested independently at each power
(figure: docs/figures/power_sweep_q.png; checkpoints saved per condition).

| Q [dB] | -1 dBm | +1 dBm | +3 dBm | +5 dBm |
|--------|--------|--------|--------|--------|
| CDC+CPE | 12.40 | 10.24 | 7.52 | 4.29 |
| MLP     | 12.72 | 12.40 | 9.42 | 6.16 |
| BiLSTM  | 12.72 | 11.46 | 9.02 | 5.80 |
| CfC     | 12.72 | 10.30 | 8.38 | 5.52 |

- Every learned equalizer improves on the baseline at every power; the tuned
  window MLP leads throughout, peaking at **+2.2 dB** at +1 dBm.
- At -1 dBm the link is ASE-limited and everyone saturates at the same Q
  (0-1 bit errors in 131k bits -- differences there are not statistically
  meaningful).
- **Open item:** the CfC's gain nearly vanishes at +1 dBm (10.30 vs 10.24)
  while holding +0.9/+1.2 dB at +3/+5 dBm. Its current recipe was tuned at
  +3 dBm; the milder-distortion regime likely needs its own (or the |x|^2
  feature hurts it there just as it does the MLP everywhere). Not yet probed.
