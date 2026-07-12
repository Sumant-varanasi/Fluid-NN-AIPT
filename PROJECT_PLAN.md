# Fluid-NN: Liquid Neural Networks for Real-Time Nonlinear Equalization in Coherent Optical Systems

**Host:** Aston Institute of Photonic Technologies (AIPT)
**Direction:** Continuous-time machine learning for high-speed coherent optical signal processing
**Implementation language:** Python

---

## 1. One-paragraph pitch

Neural-network equalizers (BiLSTM, Volterra, deep MLPs) can reverse nonlinear fiber
distortions very effectively, but their real-time hardware cost — latency, power,
multiply-accumulate (MAC) count — is the wall that keeps them out of production DSP.
This project asks whether **continuous-time Liquid Neural Networks (LNNs)**, and in
particular the **Closed-form Continuous-time (CfC)** architecture, can match the
equalization quality of these models at a fraction of the computational cost, while
*additionally* adapting on-the-fly to time-varying channel conditions. Rather than
compressing a discrete model after the fact (pruning/quantization/distillation), we
change the underlying representation: model the received symbol stream as a continuous
dynamical system whose hidden state evolves by differential equations, and solve those
dynamics analytically instead of with a slow numerical ODE solver.

---

## 2. Background & motivation

- **The distortion problem.** In coherent optical links, the signal is corrupted by
  chromatic dispersion (CD), polarization-mode dispersion (PMD), and — critically —
  the **Kerr nonlinearity**, which couples the signal to its own intensity and to
  neighboring WDM channels. Linear equalizers (CD compensation + adaptive filters)
  cannot undo the nonlinear part.
- **The ML answer, and its cost.** Volterra-series equalizers and recurrent nets
  (LSTM/BiLSTM) model these nonlinear, memory-bearing effects well. But they treat
  the waveform as a sequence of discrete timestamps and carry large parameter/MAC
  budgets — exactly the metric that governs ASIC/FPGA feasibility. The Turitsyn
  group's own *performance-versus-complexity* line of work has made this trade-off
  the central design axis for optical NN equalizers.
- **Why continuous time.** The physical channel *is* continuous. Discrete RNNs
  approximate a continuous process with fixed time steps; LNNs model it directly with
  learnable, input-dependent time constants. This tends to give (a) better sample
  efficiency and memory behaviour, (b) graceful handling of irregular/variable-rate
  sampling, and (c) natural adaptivity when the channel drifts.
- **Why CfC specifically.** Liquid Time-Constant (LTC) networks need a numerical ODE
  solver at every step — accurate but slow, and awkward for hardware. The CfC network
  replaces the solver with an explicit closed-form approximation of the ODE solution,
  giving 1–2+ orders of magnitude faster inference while preserving the expressive,
  adaptive dynamics. That is precisely the property that could move continuous-time
  models from "interesting" to "real-time deployable."

---

## 3. Central hypothesis

> A CfC/LNN equalizer can achieve equalization quality (Q-factor / BER) on par with a
> tuned BiLSTM or Volterra equalizer, at materially lower complexity (parameters,
> MACs per recovered symbol, inference latency), **and** retain accuracy under
> time-varying channel conditions with little or no retraining.

Two things must both hold for the project to be a win: **(i)** comparable accuracy at
lower complexity (the efficiency claim), and **(ii)** superior adaptivity to drift
(the continuous-time claim). We will design experiments that can confirm or falsify
each independently.

---

## 4. Objectives

1. Build a reproducible, configurable **coherent-channel simulation testbed** in Python
   (split-step Fourier propagation) producing labelled, impairment-controlled datasets.
2. Implement and tune a **baseline suite** of equalizers: linear (CDC + adaptive),
   Volterra, MLP, and (Bi)LSTM.
3. Implement the **continuous-time family**: Neural-ODE, Liquid Time-Constant (LTC),
   and Closed-form Continuous-time (CfC), adapted for complex-valued / dual-polarization
   signals.
4. Produce a rigorous **performance-versus-complexity benchmark** across launch power,
   distance, and symbol rate.
5. Characterize **adaptivity**: accuracy under channel drift and the cost of online
   adaptation.
6. Assess **hardware realizability**: fixed-point/quantized inference, MAC/latency
   modelling, and an FPGA/neuromorphic implementation pathway.
7. Package results as an **open-source repository + a publication-grade study**.

---

## 5. Technical approach

### 5.1 Channel & data generation
- **Propagation:** split-step Fourier method (SSFM) solving the nonlinear Schrödinger
  equation; single-channel first, then WDM and dual-polarization (Manakov).
- **Configurable knobs:** launch power (to sweep into the nonlinear regime), fiber
  length / span count, modulation format (QPSK, 16-/64-QAM), symbol rate, ASE noise,
  laser phase noise.
- **Pipeline:** transmit DSP → channel → coherent Rx front-end → the equalizer under
  test → symbol decision → BER/Q. Datasets versioned and seed-controlled for
  reproducibility.
- **Python stack:** NumPy/SciPy for DSP and SSFM; optionally GPU acceleration
  (CuPy / PyTorch tensors) for large sweeps.

### 5.2 Baselines
| Baseline | Role |
|---|---|
| CD compensation + adaptive (LMS/CMA) | linear reference floor |
| Volterra series equalizer | classical nonlinear reference |
| MLP | simple learned nonlinear reference |
| LSTM / BiLSTM | strong SOTA recurrent reference (the target to match) |

All baselines implemented in **PyTorch** with a shared training/eval harness so the
comparison is apples-to-apples (same data, same optimizer budget, same metrics).

### 5.3 Continuous-time models (the core)
- **Neural-ODE** — establishes the continuous-time reference (accurate, slow).
- **LTC** — liquid time-constant dynamics with an ODE solver.
- **CfC** — the headline model; closed-form, solver-free, hardware-friendly.
- **Adaptations for optics:**
  - complex-valued I/Q handling (dual real channels or complex layers),
  - dual-polarization input,
  - a receptive field / tapped-delay input window matched to channel memory,
  - a bidirectional or lookahead variant for offline comparison, and a strictly
    causal variant for the realistic real-time case.
- **Libraries:** the `ncps` package (LTC/CfC reference implementation) on PyTorch;
  `torchdiffeq` for Neural-ODE/LTC baselines.

### 5.4 Evaluation methodology
- **Quality metrics:** BER, Q-factor, MSE/EVM — measured vs. launch power (the
  nonlinearity sweep) and vs. distance.
- **Complexity metrics:** trainable parameters, **MACs per recovered symbol**,
  measured inference latency, and an energy/throughput estimate. MACs/symbol is the
  primary efficiency axis because it maps most directly to hardware.
- **The headline plot:** Q-factor (or BER) **versus complexity** — every model as a
  point on the accuracy/cost frontier; the thesis is that CfC sits on a better frontier.
- **Adaptivity experiments:** train on one channel condition, test under drifted
  conditions (changed launch power / distance / added phase noise); measure degradation
  and the samples/updates needed for online adaptation. This is where LNNs should
  distinguish themselves from frozen discrete models.
- **Ablations:** input window length, hidden size, solver vs. closed-form (LTC vs CfC),
  causal vs. bidirectional, quantization bit-width.

### 5.5 Hardware & neuromorphic pathway
- Post-training and quantization-aware **fixed-point** inference; report the
  accuracy/bit-width curve.
- **MAC/latency model** for an FPGA datapath; identify the real-time bottleneck.
- Discuss mapping the continuous-time dynamics onto **neuromorphic / photonic**
  substrates (a natural fit for AIPT and an explicit extension hook, §8).

---

## 6. Novelty & contributions

1. A systematic study of **CfC/Liquid networks for nonlinear optical equalization** —
   extending the group's performance-vs-complexity programme from discrete RNNs into
   solver-free continuous-time models.
2. A clean, quantified **accuracy-vs-complexity frontier** placing CfC against Volterra
   and (Bi)LSTM on identical data and harness.
3. An **adaptivity result**: evidence for (or against) LNNs tracking time-varying
   channels with less retraining than discrete equalizers.
4. An **open-source Python testbed** (channel sim + baselines + LNN equalizers +
   benchmark scripts) that others can build on.

---

## 7. Phased plan (flexible timeline)

> Week ranges assume roughly a one-semester / summer internship; each phase can
> compress or extend. Milestones are the checkpoints, not the calendar.

| Phase | Focus | Key milestone |
|---|---|---|
| **P0 — Setup & literature** | Reproduce the group's NN-equalizer framing; lock scope, metrics, repo, and reading list | Annotated bibliography + agreed evaluation protocol |
| **P1 — Testbed & data** | SSFM channel simulator; data pipeline; BER/Q measurement; sanity vs. theory | Reproducible datasets + validated metric pipeline |
| **P2 — Baselines** | Linear, Volterra, MLP, (Bi)LSTM in a shared harness | Baseline accuracy-vs-complexity numbers reproduced |
| **P3 — Core LNN models** | Neural-ODE → LTC → CfC, adapted for complex/dual-pol | CfC equalizer matching a baseline's BER |
| **P4 — Benchmark & adaptivity** | Full sweeps; the headline frontier plot; drift/online-adaptation study | Complete performance-vs-complexity + adaptivity results |
| **P5 — Hardware-aware** | Quantization, MAC/latency modelling, FPGA/neuromorphic feasibility | Quantized CfC + complexity/latency report |
| **P6 — Write-up** | Paper draft, cleaned repo, reproducibility pass | Submission-ready manuscript + tagged release |

**Suggested early "spike" (de-risk fast):** a minimal end-to-end slice — small SSFM
dataset → one BiLSTM baseline → one CfC model → a single accuracy-vs-MACs comparison —
before scaling to full sweeps. If CfC is in the right ballpark on the toy problem, the
rest of the plan is justified; if not, we learn it in week 2, not month 3.

---

## 8. Extension / additional project ideas

These are adjacent directions (some raised as "other ideas" in the outreach) that
either fit as stretch goals or as follow-on projects:

- **Physics-informed LNN** — embed the structure of the nonlinear Schrödinger equation
  into the model (loss or architecture), so the network learns fewer free parameters
  and generalizes better across conditions.
- **Reservoir computing / photonic reservoir equalizer** — a strongly neuromorphic
  angle; a fixed random recurrent substrate with only a trained readout, well matched
  to photonic hardware at AIPT.
- **Liquid networks for optical performance monitoring (OPM)** — reuse the same
  continuous-time backbone to *estimate* channel parameters (OSNR, launch power, reach)
  rather than equalize, exploiting LNN adaptivity.
- **Digital twin of the link** — a continuous-time surrogate of the channel for fast
  what-if simulation and controller design.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| CfC doesn't reach baseline BER | Keep LTC/Neural-ODE as fallbacks; treat "smaller model, near-equal BER" as a valid result; the frontier plot is informative either way |
| Complex-valued / dual-pol adaptation is fiddly | Start real-valued single-pol; add complexity incrementally with tests at each step |
| Simulation-only results questioned | Validate SSFM against known analytical/asymptotic results; if available, test on experimental captures from the group |
| Complexity comparison seen as unfair | Fix the harness, data, optimizer budget, and metric definitions up front; report MACs/symbol with an explicit counting method |
| Scope creep from §8 ideas | Treat §8 strictly as stretch goals gated on P4 success |

---

## 10. Deliverables

- **Open-source Python repository**: channel simulator, baseline + LNN equalizers,
  training/eval harness, benchmark and plotting scripts, reproducibility instructions.
- **Benchmark report**: the accuracy-vs-complexity frontier and the adaptivity study.
- **Hardware-feasibility note**: quantization results + MAC/latency model.
- **Publication-grade manuscript** targeting an optical-communications / ML-for-photonics
  venue (e.g. a JLT-style journal or a top optics/ML conference — venue to be chosen
  with the supervisors).

---

## 11. Core references (verify exact citations before use)

- Hasani, Lechner, Amini, Rus et al., **"Liquid Time-constant Networks,"** AAAI 2021.
- Hasani, Lechner et al., **"Closed-form Continuous-time Neural Networks,"** Nature
  Machine Intelligence, 2022.
- Chen et al., **"Neural Ordinary Differential Equations,"** NeurIPS 2018.
- Freire, Turitsyn et al., **performance-versus-complexity studies of neural-network
  equalizers in coherent optical systems** (IEEE/OSA *Journal of Lightwave Technology*
  and related venues) — the group's line of work this project extends.
- Agrawal, *Nonlinear Fiber Optics* — reference for the channel model / SSFM.
- `ncps` (Neural Circuit Policies) library — reference LTC/CfC implementation.

> The optical-equalizer references above should be pinned to the supervisors' actual
> papers once confirmed; they anchor the project in the host group's existing programme.

---

*This is a living document — scope, phases, and metrics will be refined with the
supervisors after the first literature and testbed phase.*
