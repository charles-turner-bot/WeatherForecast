# Phase 1 — task breakdown & delegation map

Overall size: **~2–4 focused sessions**. Bulk of the *effort* is first-run debugging (weight
downloads, model-specific extras) and validation — not lines of code.

## Task graph

| # | Task | Size | Owner | Depends on |
|---|---|---|---|---|
| T1 | Config module — Perth domain, variable set, model registry | S | qwen draft → I finalize | — |
| T2 | Forecast driver — earth2studio `data → model → io`, `run.deterministic` rollout | M | **me** (API-exact) | T1 |
| T3 | First **FCN3** run → Zarr output (weights/extras/debug, 10-day) | M | **me** (runtime) | T2 |
| T4 | **GraphCast** as 2nd model (config switch, its deps + weights) | S–M | **me** | T2, T3 |
| T5 | Metrics module — RMSE, bias, MAE, ACC, CRPS, rank hist, spread–skill | M | **qwen** → I unit-test | — |
| T6 | Plotting module — cartopy maps of t2m/mslp/wind over Perth | M | **qwen** draft → I refine | T3 |
| T7 | Obs verification — station obs source (IEM_ASOS/ISD/GHCN) matched to forecast | M | **me** (data API) | T3, T5 |
| T8 | Validation notebook + written readout | M | me integrate, qwen drafts prose | T5–T7 |

## Delegation rationale

- **qwen is good for:** pure, self-contained, spec-able code (T5 metrics = textbook numpy/xarray,
  unit-testable) and prose/boilerplate drafts (T1 config, T6 plotting scaffold, T8 write-up).
- **qwen is NOT good for:** anything needing the exact earth2studio 0.18 API, live data, weight
  downloads, or runtime debugging (T2–T4, T7). It's stateless and can't run the stack.
- **Rule:** every qwen output is treated as a draft — reviewed, unit-tested, and integrated by me
  before it lands. See the `/qwen` skill.

## Definition of done (Phase 1)

Reproducible 10-day forecast over the Perth window from ERA5 (FCN3 + GraphCast), written to Zarr,
plotted, and scored vs ERA5/persistence and Perth station obs — with a short written readout of
model choice and skill.

## Package layout

```
src/perthwx/
  __init__.py
  config.py      # T1
  forecast.py    # T2  (me)
  metrics.py     # T5  (qwen-drafted, verified)
  plotting.py    # T6
  verification.py# T7  (me)
```
