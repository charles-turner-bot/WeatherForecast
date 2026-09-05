# Phase 1 — Global AI forecast over a Perth window

**Goal:** a reproducible, validated global AI forecast, cropped/plotted over Perth. This is the
first "real forecast" milestone and the backbone everything else hangs off.

## Perth domain

- **Point:** Perth ≈ 31.95 °S, 115.86 °E.
- **Window (suggested):** lat −20 → −40 °S, lon 108 → 125 °E — covers Perth, the SW WA corner,
  the adjacent Indian Ocean (where WA's weather comes from), and enough margin for downscaling
  boundary conditions later. Tune once we see outputs.

## Model choice (global)

earth2studio's zoo evolves; confirm what's available in the installed version, but the sensible
shortlist and why:

| Model | Res | Notes for us |
|---|---|---|
| **FourCastNet3 / SFNO** | 0.25° | NVIDIA-native, well integrated, spherical — good default, robust long rollouts. |
| **GraphCast (operational)** | 0.25° | Strong deterministic skill; heavier deps; great baseline to compare. |
| **Pangu-Weather** | 0.25° | ONNX; fixed 1/3/6/24 h steps; solid, but ONNX path on Pascal needs checking. |
| **AIFS (ECMWF)** | 0.25° | ECMWF's ML model; good AU-relevant comparison. |

**Plan:** start with **FourCastNet3/SFNO** (best-integrated, FP32-friendly), then add **GraphCast**
as a second opinion. Keep the model behind a config switch so swapping is a one-liner.

## Pipeline (earth2studio building blocks)

earth2studio composes **data source → prognostic model → IO/output**, with an
`earth2studio.run` deterministic workflow:

1. **Initial conditions** — `data.GFS` (real-time) or `data.ARCO`/ERA5 (reproducible research).
   Start with ERA5 for a fixed historical date so runs are byte-reproducible.
2. **Model** — load prognostic model (auto-downloads checkpoint) in FP32.
3. **Run** — `run.deterministic([time], nsteps, model, data, io)` for a 10-day rollout (6-h steps
   → 40 steps).
4. **Output** — write to a **Zarr** store (`io.ZarrBackend`); crop to the Perth window on read.

Wrap this in `src/forecast_global.py` with a small YAML/CLI config: `init_time`, `model`,
`lead_time`, `domain`, `variables`, `out_path`.

## Variables (v1)

Surface: `t2m`, `u10m`/`v10m`, `msl`, `tp` (precip if the model provides it). Upper air:
`z500`, `t850`, `q` as available. (Precip is model-dependent — note which models output it.)

## Validation (this is what makes it "real", not just pretty)

- **Sanity:** plot t2m/msl/10m-wind sequences over the Perth window; check a cold front sweeping
  the SW corner looks physical.
- **Quantitative:** for a set of past cases, score the forecast vs **ERA5** (RMSE/ACC of z500/t2m)
  and against **BoM station obs** for Perth Airport / Perth Metro (t2m, wind, mslp).
- **Baselines:** compare against ERA5 persistence and, if feasible, GFS — is the AI model adding
  skill over the Perth region specifically?
- Pick 3–5 **case studies**: a summer heat event, a winter frontal passage, a trough/thunderstorm
  day. These double as the downscaling test cases in Phase 2.

## Deliverables

- [ ] `src/forecast_global.py` + config; `pixi run forecast` task.
- [ ] Zarr output for ≥3 case-study dates.
- [ ] A validation notebook: maps, station-obs comparison, skill scores vs ERA5/persistence.
- [ ] Short written readout: which model, what skill over Perth, known failure modes.

## Exit criteria

A validated, reproducible 10-day Perth forecast from ERA5 with quantified skill vs ERA5 and Perth
station obs, and a documented model choice.
