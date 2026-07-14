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

## Honest negatives and methodological findings (full diagnoses in the repo log)

- No regime found yet where the CfC *dominates*: a well-regularized window MLP leads
  offline accuracy; the discrete LSTM cell currently edges the CfC in streaming mode;
  frozen-drift robustness favours the BiLSTM. The adaptivity hypothesis is untested in
  its fairest setting — a channel that varies *in time* (see next steps).
- A physics-informed |x|^2 input is a trap for some architectures: it lowers MSE but
  raises BER (received power acts as a wrong-ring prior on borderline symbols). MSE
  and BER rank models differently throughout — loss design matters.
- Streaming training suffers optimizer-step starvation and needs ~10x longer schedules
  than window training before any model escapes the identity.

## Next steps

Dual-polarization channel with **time-varying PMD** — a genuinely drifting channel, the
fair test of the liquid-network adaptivity claim; tail-aware (BER-aligned) losses;
leaner streaming CfC variants; CfC training stability across operating points. We would
value the group's guidance on which of these to prioritize, and on validation against
experimental captures when the simulation results warrant it.
