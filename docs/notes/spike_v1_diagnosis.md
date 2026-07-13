# Spike v1: results and diagnosis

**Setup.** Single-pol 16QAM, 32 GBd, 20 x 80 km SSMF @ 2 dBm, half-window 10
(21-symbol input), 59k training symbols, 30 epochs. Full config in
`experiments/spike.py` at commit `6fa8788` and `results/spike_results.json`.

## Result: no equalizer beat CDC-only

| model    | BER      | Q [dB] | EVM % | params | MACs/sym |
|----------|----------|--------|-------|--------|----------|
| CDC only | 2.47e-02 | 5.87   | 23.3  | 0      | 0        |
| MLP      | 2.49e-02 | 5.86   | 20.6  | 13,890 | 13,696   |
| BiLSTM   | 2.35e-02 | 5.96   | 21.9  | 9,346  | 182,912  |
| CfC      | 2.83e-02 | 5.61   | 22.9  | 2,642  | 52,464   |

Models halved the MSE but BER did not move: the error they removed was not the
error that flips decisions.

## Diagnosis (measured, not guessed)

1. **The distortion is overwhelmingly deterministic, hence learnable.** Re-running
   the identical link with ASE off: deterministic EVM 22.3% vs 5.5% stochastic;
   correlation between full error and noise-free error = **0.973**. A sufficiently
   good equalizer could approach the ~5.5% stochastic floor (BER ~1e-6 territory).
   The task is winnable; v1 models were not equipped to win it.
2. **Carrier phase wander accounts for part of it.** Genie sliding-window CPE
   (window 32) on the raw output: BER 2.38e-2 -> 1.50e-2, Q +0.8 dB. The receiver
   chain lacked a CPE stage, which every comparable study includes before the NN.
3. **The input window was far too short for the channel memory.** At 1600 km the
   accumulated-dispersion interaction spread is order +/-100 symbols; the models saw
   +/-10. They physically could not see most of the interfering symbols.
4. Secondary: models were small, trained on 59k symbols for 30 epochs (literature
   uses ~1M symbols), and predicted symbols from scratch rather than corrections.

## Fixes for spike v2

- Add a **CPE stage** (genie sliding-window, window 32) to the receiver pipeline.
- Shorten the link to **12 x 80 km @ 3 dBm** (memory ~ +/-50 symbols) and widen the
  window to **half-window 20** (41 symbols) so the window covers the strong part
  of the interaction kernel.
- **Residual output**: every model predicts a correction added to the received
  center symbol (also removes the CfC tanh saturation visible in the v1
  constellation figure).
- More data (2^17 train symbols), more capacity (BiLSTM h=48, CfC h=32), 40 epochs.
