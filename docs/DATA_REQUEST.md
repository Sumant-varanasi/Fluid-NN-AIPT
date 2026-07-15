# Experimental data: what we can ingest, and what helps

Short note on the capture format our pipeline handles, so the transfer is
painless on both sides. (We can work with less than the list below -- these
just turn reverse-engineering time into analysis time.)

## Formats

`.mat` (any MATLAB version incl. v7.3), `.npz`, or `.csv`. Complex arrays
preferred; separate real/imag arrays are fine too.

## Ideal contents per capture

1. **Received symbols** (or the sampled waveform, with the sample rate stated)
   -- after coherent detection; before or after standard DSP both work, but
   please say which stages were applied (CD compensation? MIMO? CPE?).
2. **Transmitted symbol sequence** (or the PRBS/seed that generated it) --
   required for supervised training and BER counting. Our alignment handles
   unknown delay, gain, phase, conjugation, and polarization swap
   automatically.
3. **Modulation format and symbol rate** (e.g. DP-16QAM, 34.4 GBd).
4. **Link description** -- fiber type, span length x span count, EDFA/Raman,
   launch power (and ROADM/WDM configuration if relevant).
5. If several captures exist: **a sweep** (launch power or distance) is far
   more valuable than one point -- it lets us reproduce our
   performance-vs-power benchmark on real data.
6. Sequence length: anything from ~30k symbols per condition is usable;
   ~100k+ lets us also train from scratch rather than only fine-tune.

## What happens on our side

Same-day: ingestion, automatic alignment, no-equalizer baseline (BER/Q), then
our equalizer suite (MLP / BiLSTM / CfC liquid networks) trained from scratch
and fine-tuned from simulation-pretrained checkpoints, reported side by side
with constellation figures. Everything lands in the repository with the exact
processing scripted and reproducible.
