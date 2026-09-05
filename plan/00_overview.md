# AI Weather Forecast for Perth — Project Plan

**Goal:** Stand up a high-resolution AI weather forecast for Perth, Western Australia, built on
NVIDIA **earth2studio**, then extend it to a generative **ensemble forecast with bootstrapped
uncertainties**.

**Owner:** ct · Built on NVIDIA **earth2studio**; Australian km-scale data (BARRA) via the
**BoM / ACCESS** ecosystem.

---

## Strategy in one paragraph

Global AI weather models (FourCastNet3, GraphCast, AIFS, Pangu-class) run at ~0.25° (~25 km).
That is *not* "very high res" for a city. The path to km-scale over Perth is **generative
downscaling** (NVIDIA's CorrDiff): run a global model, then downscale a Perth window to ~2–4 km.
There is **no public Australia/Perth CorrDiff model**, so km-scale is a *training* effort whose
natural target is the BoM's **BARRA-C2 (4.4 km)** reanalysis. We therefore stage the work:
global forecast first (this gives a real, validated Perth forecast quickly), then downscaling,
then the ensemble/latent-space research.

## Decisions locked (from kickoff)

| Decision | Choice |
|---|---|
| Resolution ambition | **Global first (~25 km), then km-scale downscaling** |
| Use case | **Research/reproducible first, then real-time/operational** |
| Ensemble approach | **Recommendation made below** (built-in perturbations first → latent/generative research) |
| Package | **earth2studio primary; PyEarthTools noted for AU data/BARRA** |

## Hardware reality (drives everything)

- **2× Tesla P40, 24 GB each (48 GB total)** — Pascal `sm_61`, **no tensor cores**, FP16 ≈ 1/64 FP32.
- CUDA driver supports up to CUDA 13; local toolkit 12.4. 61 GB RAM, 40 cores, ~84 GB free disk.
- **Implication:** run models in **FP32**. Global inference is very feasible. CorrDiff *training*
  and diffusion *sampling* will be slow (diffusion = many forward passes); plan long jobs, small
  batch sizes, and consider using both GPUs. See [04_hardware_notes.md](04_hardware_notes.md).

## Phases

| Phase | Outcome | Doc |
|---|---|---|
| 0 | **✅ DONE** — pixi env + torch 2.5.1+cu124 + earth2studio 0.18 verified on P40; ERA5 fetch works | [01_setup.md](01_setup.md) |
| 1 | Reproducible **global** AI forecast over a Perth window (from ERA5/GFS), validated & plotted | [02_phase1_global_forecast.md](02_phase1_global_forecast.md) |
| 2 | **km-scale downscaling** over Perth (evaluate existing CorrDiff; scope training vs BARRA-C2) | [03_phase2_downscaling.md](03_phase2_downscaling.md) |
| 3 | **Ensemble** forecast — built-in perturbations → **latent/generative** research (your idea) | [05_phase3_ensembles.md](05_phase3_ensembles.md) |
| — | Real-time/operational wiring (after research validates) | [06_operationalisation.md](06_operationalisation.md) |

Supporting docs: [07_data_sources.md](07_data_sources.md), [08_open_questions.md](08_open_questions.md),
[09_machine_facts.md](09_machine_facts.md), [10_model_landscape.md](10_model_landscape.md).

## Two intents (reconciled — see [10_model_landscape.md](10_model_landscape.md))

This project sits inside a broader vault plan to explore the AI weather/climate model landscape on
the dual-P40 box. **Intent A** = the Perth forecast product (this plan). **Intent B** = the vault's
landscape tour (earth2studio → GraphCast → NeuralGCM → FourCastNeXt). They share one spine —
**earth2studio-first, local pixi env, inference/eval/light-tune** (the P40 sweet spot). Because
earth2studio 0.18 already ships GraphCast/Pangu/FCN3/Aurora/GenCastMini, the vault's "GraphCast
inference" step collapses into **Phase 1** here. NeuralGCM/FourCastNeXt/ClimaX remain separate lanes.

## Definition of done (v1)

A single command produces a validated, plotted 10-day forecast for the Perth domain from a chosen
initial time, reproducible from ERA5, with a written comparison against ERA5/observations — and a
documented, honest read on what the P40s can and can't do for the downscaling and ensemble phases.

## Local-model delegation

A `/qwen` skill (and cloneable `/muse`) delegates self-contained sub-tasks (boilerplate, docstrings,
bulk summarisation of docs/log files) to on-device Ollama models — useful for grinding through
earth2studio/BARRA documentation without leaving the machine. Outputs are always reviewed before use.
