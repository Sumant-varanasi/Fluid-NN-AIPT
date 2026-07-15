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

## Adaptivity study -- honest verdict: no LNN advantage demonstrated (yet)

Models trained at +3 dBm / 12 spans, evaluated frozen under drift
(figure: docs/figures/adaptivity_drift.png; full numbers results/adaptivity.json).

**Frozen power drift** (gap = matched retrain minus frozen, at +5 dBm):
MLP 0.34 dB, BiLSTM 0.18 dB, CfC 0.60 dB. All models degrade gracefully and
keep beating the baseline everywhere, but the CfC's frozen gap is the largest,
not the smallest -- the continuous-time adaptivity hypothesis is **not**
supported by frozen-transfer behaviour at this operating point.

**Distance drift** (10-14 spans, frozen): all models hold their ranking and
their gains; nothing dramatic. Equalizers trained at 12 spans stay useful
over +/-160 km without retraining.

**Pilot-based adaptation at +5 dBm** (fine-tune 15 epochs, lr 3e-4):
| pilots | MLP | BiLSTM | CfC |
|--------|-----|--------|-----|
| frozen | 5.82 | 5.62 | 4.92 |
| 1k     | 5.73 | 5.63 | 4.93 |
| 4k     | 5.70 | 5.68 | 4.98 |
| 16k    | 5.91 | 5.80 | 5.10 |
| matched| 6.16 | 5.80 | 5.52 |

BiLSTM fully recovers its matched performance with 16k pilots; CfC recovers
only 0.18 of its 0.60 dB gap; the MLP *loses* Q at small pilot counts
(overfits the pilots -- consistent with its known regularization sensitivity).

**Surprise worth chasing:** the +3 dBm-trained CfC evaluated frozen at +1 dBm
scores Q 11.29 -- a full dB *better* than the CfC trained directly at +1 dBm
(10.30). The +1 dBm matched training run was pathological, not the regime
itself. CfC training stability across operating points is now the top open
thread, together with tail-aware losses.

## Streaming equalizers -- open problem, and the CfC is exonerated

The streaming benchmark (state carried across the stream, delay-8 lookahead,
O(1)/symbol) failed identically for CfC at h=16/32/64: all converge to the
identity (Q 7.51 vs baseline 7.52). Probes since:

1. **Tapped state readout** (head reads [h_t, h_{t-delay}] so the target symbol
   is one step from a readout, mirroring the fix that unstuck the window CfC):
   no effect at a 560-optimizer-step budget.
2. **Control: a causal streaming LSTM under the identical protocol also
   flatlines** (Q 7.58). The failure is the streaming training protocol, not
   the CfC cell -- important negative result for the comparison's fairness.
3. Suspect: optimizer-step starvation. Streaming chunks give ~10-60x fewer
   optimizer steps per epoch than window training, and every recurrent model
   in this project needed >1000 steps before leaving the identity.
4. **Confirmed.** Same streaming LSTM, 200 epochs (~2800 steps): val MSE
   0.0163 -> 0.0116 and still falling, Q 7.5 -> 8.08. The streaming objective
   trains fine given the budget; the earlier flatline was optimizer steps, not
   architecture. Full benchmark (StreamCfC-h32 vs StreamLSTM-h32, 200 epochs,
   full data) re-running.

Until streaming numbers land, the honest complexity story compares window
models only; streaming MAC numbers are stated as potential, never with a Q
attached.

**Results (200 epochs, full data; figure docs/figures/streaming_frontier.png):**

| model | BER | Q [dB] | MACs/sym | delay |
|-------|-----|--------|----------|-------|
| CDC+CPE | 8.72e-03 | 7.52 | 0 | - |
| StreamLSTM-h32 | 4.10e-03 | 8.44 | 4,608 | 8 sym |
| StreamCfC-h32 | 4.97e-03 | 8.22 | 10,560 | 8 sym |
| (window CfC)   | 4.33e-03 | 8.38 | 438k | offline |
| (window MLP)   | 1.54e-03 | 9.42 | 54k | offline |

- **The O(1)-per-symbol regime works**: a streaming equalizer with 8 symbols of
  decision latency recovers most of the window CfC's gain at **~95x fewer MACs**
  (StreamLSTM 4.6k vs window CfC 438k, equal Q within 0.06 dB).
- Within streaming, the discrete LSTM cell currently edges the CfC cell
  (Q 8.44 vs 8.22) at 2.3x fewer MACs -- the CfC's backbone is its cost. A
  leaner CfC (no backbone) and longer budgets remain unprobed in streaming.
- Both streaming losses were still falling at epoch 200.

## Dual-pol RSOP drift -- the time-varying channel test (first CfC-positive signal)

Dual-pol 16QAM via Manakov, +5 dBm total, receiver CDC + genie block demux
(256) + per-pol CPE; endless polarization rotation at 0-100 deg/ksym *during*
the sequence. Figure: docs/figures/dp_drift_q.png; full numbers
results/dp_drift.json.

**Q [dB] at drift rate 100 deg/ksym** (baseline 6.43):

| model | trained static (frozen) | trained on drift 100 (matched) |
|-------|------|------|
| MLP        | 6.44 | 6.68 |
| BiLSTM     | 6.92 | **7.93** |
| CfC        | 6.54 | 7.67 |
| StreamLSTM | 6.54 | 7.32 |
| StreamCfC  | 6.53 | **7.44** |

Findings:
1. **Training on the drifting channel teaches drift-tracking.** Matched models
   hold nearly flat Q as drift increases while the baseline collapses; at
   100 deg/ksym the matched BiLSTM keeps +1.50 dB over baseline vs +0.49 for
   its static-trained twin. This is the study's clearest new result.
2. **First head-to-head the CfC wins, in the predicted setting:** with drift in
   training, StreamCfC edges StreamLSTM (7.44 vs 7.32) -- the reverse of every
   static-channel comparison -- and the window CfC closes its gap to the BiLSTM
   from 0.61 dB (static) to 0.26 dB. *Caveat: single seed, ~0.1-dB scale;
   directional until a multi-seed repeat confirms it.*
   **Multi-seed verdict (4 seeds): NOT confirmed.** StreamCfC mean 7.36
   (7.12/7.40/7.44/7.48, std 0.16) vs StreamLSTM mean 7.29 (7.22/7.27/7.32/
   7.35, std 0.06): the means overlap within seed noise, and the CfC's
   seed-to-seed variance is ~3x the LSTM's -- consistent with the CfC
   training-stability issue seen at +1 dBm. The honest claim is parity in the
   drifting setting (vs a clear LSTM edge in static settings), not a CfC win.
   The window CfC's narrowed gap (0.26 dB) remains single-seed / unconfirmed.
3. Robustness-accuracy tradeoff: drift-trained models give up ~0.3-0.9 dB on
   the static channel.
4. **Anomaly / open item:** the window MLP -- the single-pol accuracy leader --
   sits exactly at baseline in every dual-pol condition (identity output). Its
   recipe (IQ-only, wd 1e-4) has not been re-probed for dual-pol; the BiLSTM's
   +0.83 dB at drift 0 proves learnable structure exists.
   **Probe result: not a recipe problem.** Three variants (lr 1e-3/3e-3, with/
   without weight decay, with/without |x|^2) all land within 0.03 dB of the
   baseline. At this size, the flat MLP genuinely cannot extract the dual-pol
   residual structure that the recurrent models can -- a real architectural
   effect and an open research question (capacity? cross-pol feature geometry?).

## Where the thesis stands after the first full benchmark cycle

Supported: learned equalizers give up to +2.2 dB Q across the nonlinear regime;
streaming models make real-time complexity plausible (~5-10k MACs/symbol at
+0.7-0.9 dB gain). Not yet supported: a regime where the CfC *dominates* --
the window MLP wins offline accuracy, the streaming LSTM edges the streaming
CfC, and frozen-drift robustness favours the BiLSTM. Open threads, in order:
CfC training stability across powers, tail-aware losses (the MSE/BER
divergence), leaner streaming CfC, dual-polarization + PMD drift (a channel
that *actually* varies in time, where the liquid-network hypothesis gets its
fair test -- static drift here may simply not be the setting it pays off in).
