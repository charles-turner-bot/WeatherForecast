# WeatherForecast

AI weather forecasting for **Perth, Western Australia**, built on NVIDIA
[earth2studio](https://github.com/NVIDIA/earth2studio): a global AI forecast → km-scale
downscaling → generative ensemble with bootstrapped uncertainties.

Runs locally on a dual **Tesla P40** box (Pascal, FP32) — inference / evaluation / light
fine-tuning, not train-from-scratch.

## Status

- **Phase 0 ✅** — `pixi` env, `torch 2.5.1+cu124` on the P40s, `earth2studio 0.18.0`, verified
  ERA5 data fetch over Perth.
- **Phase 1 ⬜** — global AI forecast (FCN3/GraphCast) over the Perth window, validated vs ERA5/obs.
- Later: km-scale downscaling (CorrDiff vs BARRA-C2), ensembles, real-time.

## The plan

Full plan in [`plan/`](plan/) — start at [`plan/00_overview.md`](plan/00_overview.md).

## Quickstart

```bash
pixi install          # reproduce the env from pixi.lock
pixi run gpu-check    # confirm both P40s are visible to torch
pixi run lab          # JupyterLab
```

Model checkpoints and data cache go under `.cache/` (git-ignored);
set via `EARTH2STUDIO_CACHE` in `pixi.toml`.

## Context

Part of a broader local AI-weather exploration (see
[`plan/10_model_landscape.md`](plan/10_model_landscape.md)). Built on NVIDIA earth2studio, with
Australian km-scale data (BARRA) from the BoM / ACCESS ecosystem.
