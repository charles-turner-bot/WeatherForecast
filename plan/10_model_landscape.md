# Model landscape & how it maps onto this project

Distilled from Charles's Obsidian vault (the "AI weather on dual P40" cluster) and reconciled with
what **earth2studio 0.18.0 already gives us** (installed, Phase 0 ✅).

## Two intents, one spine

- **Intent A — Perth forecast product** (this project's original goal): a high-res AI forecast for
  Perth + a generative ensemble. Docs [00](00_overview.md)–[08](08_open_questions.md).
- **Intent B — local model-landscape exploration** (the vault's framing): a pragmatic tour of the
  AI weather/climate model space on the dual-P40 box — inference, evaluation, light adaptation;
  **not** flagship retraining.

They share a spine: **earth2studio-first**, on **local pixi env**, targeting **inference/eval/light
fine-tuning** (the P40 sweet spot both the vault and [04_hardware_notes.md](04_hardware_notes.md)
land on). The Perth forecast is the concrete deliverable that *forces* real end-to-end success;
the landscape lanes are breadth we can pursue through, or alongside, it.

## The three lanes (don't blur them) — vault's split

| Lane | Models | Job | On our box |
|---|---|---|---|
| **AI weather forecasting** | GraphCast, Pangu, FourCastNet/FCN3, FuXi, FuXi-ENS | global medium-range | **Mostly via earth2studio already** |
| **Nowcasting / storm-scale** | MetNet-3, StormCast | short-range, obs-heavy | `StormCast`/`StormScope*` in earth2studio; different job from Perth medium-range |
| **Climate / hybrid / emulator** | NeuralGCM, ClimaX, Aurora, Samudra, SamudrACE | hybrid physics, climate, coupled | Aurora + SamudrACE in earth2studio; **NeuralGCM/ClimaX = separate repos** |

## Key reconciliation: earth2studio subsumes most of the "forecasting" lane

The vault's bring-up sequence was *earth2studio → GraphCast → NeuralGCM → FourCastNeXt*, treating
GraphCast as a separate JAX repo to stand up (vault Phase 3, "known pain: JAX/CUDA + data plumbing").
**We can now skip that pain for inference:** earth2studio 0.18 ships `GraphCastOperational`,
`GraphCastSmall`, `Pangu3/6/24`, `FCN`/`FCN3`, `SFNO`, `FuXi`, `Aurora`, and `GenCastMini` as
loadable models with weight download + a unified `run.deterministic/ensemble` API. So **vault Phase 3
(GraphCast inference) collapses into our Phase 1** — running GraphCast over Perth is a config switch,
no separate JAX bring-up.

What is **genuinely separate** (own repos, JAX, not in earth2studio):
- **NeuralGCM** (`neuralgcm/neuralgcm`) — hybrid ML + differentiable physics; the vault's most
  intellectually-aligned target and the closest atmosphere-side analogue to Charles's
  emulator/latent instinct (ties to [05_phase3_ensembles.md](05_phase3_ensembles.md)). JAX repo →
  likely a **container fallback** candidate ([09_machine_facts.md](09_machine_facts.md)).
- **FourCastNeXt** (`nci/FourCastNeXt`) — reduced-compute *training* line; NCI/HPC-shaped, so treat
  as methodological reference for the P40s, not turnkey.
- **ClimaX** (`microsoft/ClimaX`) — foundation/transfer-learning infra; keep in view.

## Runnable repo shortlist (vault)

1. `NVIDIA/earth2studio` — ✅ installed & working.
2. `google-deepmind/graphcast` — inference reachable via earth2studio; standalone repo only if we
   need training/finetuning or the native data pipeline.
3. `neuralgcm/neuralgcm` — separate; inspection/inference lane (Intent B research).
4. `microsoft/ClimaX` — reference/foundation-model lane.
5. `nci/FourCastNeXt` — reduced-compute training breadcrumb.

## Physics framing (vault "AI weather physics memo")

Three-way "how much real physics" split: data-driven forecasters (GraphCast) · nowcasting
(MetNet-3, StormCast) · hybrid ML+physics (NeuralGCM). For a *physics-shaped atmosphere model*,
NeuralGCM is the cleanest name; for *practical forecast skill*, GraphCast/FCN3. Our Perth product
lives in the first lane; the NeuralGCM curiosity is the bridge to the hybrid lane.

## Papers & people (from vault, for outreach/reading)

- **GraphCast** — arXiv 2212.12794 (10-day global @0.25°).
- **NeuralGCM** — arXiv 2311.07222 (hybrid; deterministic + ensemble + climate).
- **Samudra** — arXiv 2412.03795 (ocean emulator; century rollouts, ~150× speedup; forcing-trend
  stability caveats). **SamudrACE** = coupled atmos–ocean breadcrumb.
- **Key authors (of the papers above), as a reading/citation aid:** DeepMind GraphCast/NeuralGCM —
  Battaglia, Sanchez-Gonzalez, Hoyer, Ravuri, Mohamed, Willson. Foundation weather/climate —
  Brandstetter, Kapoor, Grover, Gupta, Bodnar, Perdikaris. FourCastNet/HPC — Pathak, Mardani, Kurth,
  Subramanian. Ocean — Zanna.
- **Tooling & data:** NVIDIA **earth2studio**; Australian km-scale data (BARRA) via **BoM / ACCESS**.

## Vault sources (in `charles-obsidian-vault`)

`wiki/syntheses/`: *AI weather on Charles's dual P40 server*, *dual P40 AI weather bring-up plan*,
*dual P40 AI weather model options*, *runnable AI weather repo shortlist*, *AI weather and climate
models cluster*, *AI weather physics memo*, *AI climate emulators cluster*.
`raw/articles/`: *2026-07-05-AI-weather-and-climate-models*, *2026-07-05-Samudra-ocean-emulator*.
