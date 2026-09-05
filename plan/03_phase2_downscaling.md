# Phase 2 — Km-scale downscaling over Perth

**Goal:** turn the ~25 km global forecast into a **~2–4 km** forecast over the Perth domain — the
"very high res" part.

## The honest situation

- NVIDIA's downscaler is **CorrDiff** (a two-stage generative model: a regression "mean" model
  corrected by a diffusion model), publicly available **trained for Taiwan (~3 km), a US GEFS
  variant, and COSMO/Europe** — **not Australia**.
- So there is no drop-in Perth downscaler. Two routes, cheapest first:

### Route A — Evaluate/adapt an existing CorrDiff (fast, low commitment)

- Run the earth2studio CorrDiff example end-to-end to learn the API and the compute cost of
  diffusion sampling **on the P40s** (this is the real feasibility question — many denoising steps).
- Assess whether a pretrained regional CorrDiff transfers *at all* to SW WA (likely poor — trained
  on other regions' orography/climate — but it calibrates effort and tooling).
- **Output:** a working downscaling pipeline + an honest measurement of per-forecast diffusion cost
  on Pascal. Decision gate for Route B.

### Route B — Train a Perth/WA CorrDiff (the real deliverable)

- **Target (hi-res truth):** BoM **BARRA-C2 (4.4 km, convection-permitting)** over WA — the ideal
  training target. BARRA-R2 (12 km) is the intermediate/parent. This is exactly where
  the **BoM / ACCESS** community and **PyEarthTools** help: data access, variables, licensing, tiling.
- **Input (coarse):** the global model output (or ERA5) regridded to CorrDiff's expected input grid
  over the WA window.
- **Recipe:** pair coarse↔BARRA-C2 over a multi-year period for the Perth domain; train CorrDiff
  (regression stage then diffusion stage) following NVIDIA's CorrDiff training (Modulus) recipe.
- **Reality check:** training a diffusion downscaler on 2× P40 is heavy. Budget weeks of wall-clock,
  or use it to build/validate the data pipeline and a small-domain prototype, and seek a bigger GPU
  (cloud/HPC — NCI Gadi via ACCESS?) for the full train. **Raise scope with NVIDIA/earth2studio early.**

## Data engineering (shared by both routes)

- Define the exact Perth training/inference tile (extent, target grid, orography).
- Regridding coarse→fine grids with **xesmf** (already in env); align to BARRA-C2 grid + static
  fields (topography, land-sea mask) as conditioning inputs.
- Store paired samples as Zarr for efficient training IO.

## Open questions → [08_open_questions.md](08_open_questions.md)

- BARRA-C2 access path & license (NCI? BoM direct? which variables/levels?) — **for BoM / ACCESS.**
- Does NVIDIA have an AU CorrDiff in progress or a recommended transfer-learning path? — **for NVIDIA / earth2studio.**
- Is a smaller/cheaper downscaler (e.g. a deterministic U-Net/regression-only, or a diffusion model
  with few sampling steps) an acceptable v1 given P40 limits?

## Exit criteria

Either (A) a working, cost-characterised downscaling pipeline on an existing CorrDiff over the Perth
window, or (B) a trained/prototyped WA CorrDiff on BARRA-C2 with skill vs BARRA-C2 truth on held-out
case-study dates — plus a clear compute plan for full training.
