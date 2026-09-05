# T7 — Station verification

Implemented 2026-09-05 in `src/perthwx/verification.py`, with CLI
`scripts/verify_forecast.py` / `pixi run verify`.

## Source and matching contract

- Perth Airport: **YPPH**, latitude −31.92751, longitude 115.97610, elevation 13 m,
  from the [IEM station catalog](https://mesonet.agron.iastate.edu/sites/site.php?station=YPPH&network=AU__ASOS).
  Perth Metro is not included; it needs a separate station source.
- Uses the installed earth2studio 0.18 `IEM_ASOS` long-form observation API.
  Its lexicon converts temperature to K, wind to m/s and sea-level pressure to Pa.
  Wind components follow meteorological direction conventions; forecast speed
  is derived from u/v at the selected grid point.
- Select the nearest grid cell (record its coordinates); reject stations outside
  the grid. No elevation, exposure, or coastal representativeness adjustment.
- Valid time = initialization + lead, in UTC. Match each variable separately to
  the nearest finite report within ±30 minutes (`--tolerance-minutes` changes this).
  Equal-distance ties use the earlier report; duplicate station/variable/time
  reports are averaged. Multiple initializations retain separate forecast rows.
- Lead zero is kept in pairs for sanity checks but excluded from all scores.
  RMSE, bias (forecast minus observation), and MAE use finite pairs only;
  each summary includes expected, matched, and missing counts. No-pair scores
  are blank/NaN, never zero. A run with no positive-lead pairs exits nonzero
  after writing diagnostic outputs.
- Future valid times are not requested. Fetches refresh recent windows on rerun,
  avoiding permanently cached partial observations. The saved SI observation CSV
  is the reproducible snapshot; `--observations` replays it without network calls.

[IEM documents minimal quality control](https://mesonet.agron.iastate.edu/request/download.phtml).
These are preliminary station comparisons, not a BoM-quality verified product.
MSLP is used only when reported as `mslp`; altimeter/QNH is not substituted.
Temperature and wind are instantaneous-report comparisons, not daily extrema
or gust verification. Scores pooled over a single trajectory do not establish
climatological skill; scores by lead have only one sample per lead for one run.

## Outputs and use

```bash
pixi run verify --zarr outputs/fcn_gfs_7day.zarr
pixi run verify --zarr outputs/fcn_gfs_7day.zarr \
  --observations outputs/verification/observations.csv \
  --outdir outputs/verification-replay
```

The output directory contains `observations.csv`, `pairs.csv`, `scores.csv`,
`scores_by_lead.csv`, and `metadata.json`. Pairs include actual observation times,
time offsets, grid coordinates and SI units. Outputs remain git-ignored.
Offline tests cover grid selection, valid-time arithmetic, tolerance boundaries,
time ties, duplicate reports, nonfinite data, wrong stations, empty coverage,
multiple initializations, wind speed and known-answer scores.

## First real-data check

FCN/GFS initialized **2026-09-04 00 UTC**, seven-day output, checked on
**2026-09-05**. Nearest airport grid point: **−32°, 116°**.
Five positive leads (+6 through +30 h) matched for temperature and wind out of
28 forecast leads; most remaining leads are in the future. No usable MSLP
observations were returned. First-run scores:

| Variable | Matched | RMSE | Bias | MAE | Units |
|---|---:|---:|---:|---:|---|
| t2m | 5 | 2.651 | −2.420 | 2.420 | K |
| u10m | 5 | 1.522 | 0.290 | 1.138 | m/s |
| v10m | 5 | 1.452 | −1.163 | 1.163 | m/s |
| wind speed | 5 | 1.536 | −0.179 | 1.334 | m/s |

This demonstrates working data access and pairing; **it does not validate seven-day
skill**. No ERA5 or persistence comparison has been run yet. Next priority is
historical ERA5-compatible FCN initialization (relative humidity derivation), then
multiple historical cases and T8's written skill comparison. The original
10-day/ERA5/station-validation Phase 1 exit criteria remain unmet.
