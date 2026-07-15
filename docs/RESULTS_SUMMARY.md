# Fluid-NN: Results Summary

**Continuous-time neural equalizers for coherent optical links — first benchmark cycle**
Repository: https://github.com/Sumant-varanasi/Fluid-NN-AIPT (code, figures, experiment log)

## Objective

Test whether closed-form continuous-time (CfC / liquid) neural networks can equalize
nonlinear fiber distortion at a fraction of the compute cost of BiLSTM-class equalizers,
and whether they adapt better to channel drift — the performance-versus-complexity
question, extended to continuous-time models.

## Testbed (all Python, open-source)

Split-step Fourier simulation of the NLSE: single-pol 16QAM, 32 GBd, 12 x 80 km SSMF,
EDFA (NF 4.5 dB), RRC 0.1; receiver: CDC + ideal CPE; equalizers see 41-symbol windows.
The physics is validated by 26 unit tests against analytic results (dispersion law, CW
nonlinear phase, energy conservation, AWGN SER). Every model trains on identical data
and budget; complexity is reported as weight-MACs per recovered symbol.

## Key results

**1. Learned equalizers deliver across the nonlinear regime** — up to **+2.2 dB Q**
over CDC+CPE (power sweep, -1 to +5 dBm). At the +3 dBm operating point:

| model | Q [dB] | MACs/symbol | mode |
|---|---|---|---|
| CDC+CPE (no equalizer) | 7.52 | 0 | — |
| MLP (tuned) | **9.42** | 54k | window, offline |
| BiLSTM | 9.02 | 803k | window, offline |
| CfC (bidirectional) | 8.38 | 438k | window, offline |
| **Streaming LSTM** | 8.44 | **4.6k** | causal, 8-symbol latency |
| **Streaming CfC** | 8.22 | **10.6k** | causal, 8-symbol latency |

**2. The real-time operating mode works.** A causal streaming equalizer (state carried,
one cell update per symbol, 8 symbols of decision latency) matches the window CfC's
quality at **~95x fewer MACs/symbol**. This is the regime where per-symbol hardware
cost becomes plausible for real-time DSP.

**3. Drift robustness (trained +3 dBm, frozen):** all equalizers degrade gracefully over
+/-2 dB power drift and +/-160 km distance drift, retaining gains without retraining;
the BiLSTM fully recovers matched performance from 16k pilot symbols.

**4. Time-varying channel (dual-pol Manakov + endless polarization rotation):**
training *on* the drifting channel teaches drift-tracking -- at 100 deg/ksym rotation
the drift-trained BiLSTM retains **+1.5 dB** over the baseline where its static-trained
twin keeps only +0.5 dB (large effect, consistent across all five models). In this
time-varying setting the streaming CfC reaches **statistical parity** with the streaming
LSTM (4 seeds; the LSTM cell wins clearly on every static channel), though the
hypothesized CfC *advantage* did not survive multi-seed testing.

## Honest negatives and methodological findings (full diagnoses in the repo log)

- No regime found where the CfC *dominates*: a well-regularized window MLP leads
  offline single-pol accuracy; the LSTM cell edges the CfC in streaming mode on static
  channels (parity under drift); frozen-drift robustness favours the BiLSTM. Every
  claim in the repo log carries its multi-seed confirmation status.
- The window MLP — single-pol accuracy leader — cannot learn the dual-pol residual at
  all (flat at baseline across every recipe probed), while recurrent models extract up
  to +0.8 dB from the same data: an unexplained architectural effect worth discussing.
- A physics-informed |x|^2 input is a trap for some architectures: it lowers MSE but
  raises BER (received power acts as a wrong-ring prior on borderline symbols). MSE
  and BER rank models differently throughout — loss design matters.
- Streaming training suffers optimizer-step starvation and needs ~10x longer schedules
  than window training before any model escapes the identity.

## Next steps

Validation on experimental captures (ingestion / alignment / fine-tuning pipeline built
and rehearsed end to end; simulation-pretrained transfer checkpoints ready — see
docs/DATA_REQUEST.md for the capture formats we ingest); CfC training stability across
operating points; tail-aware (BER-aligned) losses; leaner streaming CfC variants;
first-order PMD in the dual-pol channel. We would value the group's guidance on
prioritization.
