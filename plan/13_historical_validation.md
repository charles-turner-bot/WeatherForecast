# Historical FCN/ERA5 validation — T8

Three ten-day FCN forecasts initialized from ERA5 establish a working Perth research
baseline. Across these cases, regional temperature RMSE improves on persistence by
about **42% at day five**, but is **19% worse at day ten**. Airport temperature beats
constant persistence in all three cases; wind-speed results are mixed, with a
negative forecast bias in each case. The results justify broader validation before
operational use or km-scale downscaling.

## Reproduce

```bash
# One ten-day forecast, with small Perth-domain output (global inference):
pixi run forecast --source arco --init 2022-01-01T00:00:00 --lead-days 10 \
  --crop-output --out outputs/my_historical_forecast.zarr

# Run/resume all three cases, fetch truth and observations, write scores and figures:
pixi run validate

# Regenerate the scorecard from complete local snapshots; fail if anything is missing:
pixi run validate --offline

pixi run test-era5
pixi run test-validation
```

Read the [validation notebook](../notebooks/phase1_validation.ipynb) and
[generated report artifacts](../reports/phase1/). Raw forecasts, observation pairs,
and per-valid-time ERA5 snapshots are under `outputs/historical/` (git-ignored).
`--report-dir` selects a separate output for replay comparisons. A forecast completion
marker records the configuration and library versions; incomplete or mismatched
outputs are rejected. Existing forecast outputs are never silently overwritten,
and `forecast --dry-run` does not create or modify a Zarr store.

## Initial conditions

`ERA5WithRH` requests ARCO temperature and specific humidity for each requested
pressure-level relative humidity field, and invokes the installed Earth2Studio
`DerivedRH` diagnostic on CPU in FP32. FCN needs r500 and r850; all other fields
pass through in the model's requested order. Temperature is K, specific humidity
kg/kg, pressure hPa converted internally to Pa, and output RH percent.

The diagnostic follows the [ECMWF mixed-phase saturation formulation](https://www.ecmwf.int/sites/default/files/elibrary/2016/17117-part-iv-physical-processes.pdf)
and [NVIDIA's DerivedRH API](https://nvidia.github.io/earth2studio/main/modules/generated/models/dx/DerivedRH/).
The installed implementation clips the liquid-water blend to [0,1] and the output
RH to [0,100] percent. Saturation clipping can differ from archived supersaturated
ERA5 RH, so this is a derived initialization, not an assertion of bitwise identity
to archived relative humidity. Offline reference tests cover dry/saturated air,
ice, mixed phase, warm air, variable order, and passthrough fields.

The [FCN model card](https://huggingface.co/nvidia/fourcastnet1) lists training in
1980–2015, testing in 2016–2017, and evaluation in 2018–2019. The installed FCN
loader pins checkpoint revision `c67a63995f6c8e0e557eb3d791f32f437e9b02d5`.
The report manifest includes checkpoint, code, lockfile, observation and truth
snapshot SHA-256 hashes. Numerical bitwise reproducibility across different GPU
architectures or library stacks is not claimed.

## Cases and scoring

Dates were fixed before looking at scores: **2022-01-01**, **2022-07-01**, and
**2022-10-01**, each initialized at 00 UTC and run to +240 h at six-hour intervals.
They sample Perth summer, winter, and spring; they are not asserted to be confirmed
heatwave, frontal, or thunderstorm events. Three seasonal cases are a first baseline,
not a climatological validation or a statistical significance test.

- ERA5: exact valid-time and grid alignment, all six output fields, cosine-latitude
  weighted RMSE/bias/MAE over the 81×69 Perth domain. z500 is geopotential in m²/s²,
  not geopotential height. Persistence holds each initial ERA5 field unchanged.
- Airport: YPPH observations and unit conventions from [T7](12_station_verification.md).
  Observation persistence uses the most recent finite report at or before the
  initialization within 30 minutes. A report after initialization cannot seed it.
- Both comparisons exclude lead zero and score FCN and persistence on identical
  finite samples. Coverage is explicit, including missing station baselines.
- Lead-zero fields that come directly from ARCO must agree with the independent
  verification fetch within 0.01 SI units. Nonfinite forecast or ERA5 fields fail
  the validation run.
- The overview plot pools cases as sqrt(mean(case RMSE²)), equally weighting cases.
  Per-case and per-lead CSVs retain the detail needed to examine individual failures.
- No multi-year climatology was fetched, so **ACC is not reported**. Spatial
  correlation without climatological anomalies would be a different score.

## Results

All three ten-day forecasts and validation cases completed on 2026-09-05.
The six directly supplied initial fields agreed **exactly** with the independent
ERA5 verification fetch (maximum absolute error zero for each case). All 5,589
regional grid points were finite for every scored field and lead.

The equally weighted three-case regional RMSE is:

| Variable | Lead | FCN | ERA5 persistence | Units |
|---|---:|---:|---:|---|
| t2m | 1 day | 0.762 | 1.250 | K |
| t2m | 5 days | 1.580 | 2.728 | K |
| t2m | 10 days | 2.893 | 2.432 | K |
| z500 | 1 day | 66.790 | 397.285 | m²/s² |
| z500 | 5 days | 149.376 | 1035.667 | m²/s² |
| z500 | 10 days | 415.933 | 973.724 | m²/s² |

![Regional ERA5 RMSE](../reports/phase1/era5_rmse.png)

**FCN adds useful skill in these cases, but the ten-day temperature result is a
failure against even this simple baseline.** It reduces pooled day-five temperature
RMSE by about 42%, while day-ten temperature RMSE is about 19% higher than persistence.
The day-ten regional temperature loss occurs in the January and July cases; the
October case improves on persistence. The z500 comparison remains substantially
better than persistence at the reported leads. These results support a working
research backbone, not a blanket claim of useful ten-day surface-weather skill.

Airport results pool the 40 positive leads in each case, with identical sample
counts for FCN and observation persistence:

| Initialization | t2m FCN / persistence RMSE (K) | Wind speed FCN / persistence RMSE (m/s) | FCN t2m bias (K) | FCN wind-speed bias (m/s) |
|---|---:|---:|---:|---:|
| 2022-01-01 | 3.356 / 6.721 | 2.983 / 2.232 | +2.392 | −1.471 |
| 2022-07-01 | 3.104 / 5.670 | 2.231 / 2.242 | −1.167 | −0.495 |
| 2022-10-01 | 2.171 / 5.182 | 2.665 / 3.284 | −0.397 | −1.007 |

Temperature, u/v wind, and wind speed each have **120/120** positive-lead matched
reports over the three cases. MSLP has **0/120** and is not assigned a station
skill score. The airport temperature comparison beats constant persistence in
all three cases; wind speed loses in summer, nearly ties in winter, and improves
in spring. Wind-speed bias is negative in every case. Airport exposure, grid-cell
representativeness, and forecast smoothing are hypotheses to investigate, not
causes established by this experiment.

The regional domain contains substantial ocean area, so its temperature RMSE is
not a substitute for the airport scores. Constant persistence also ignores the
diurnal cycle, which makes it a weak temperature baseline when all six-hour leads
are pooled. The lead-specific curves and same-time-of-day comparisons at whole
days are more informative than the pooled airport temperature win alone.

## Verification and next priority

The new tests cover RH reference states, safe dry runs and output protection,
exact temporal/spatial alignment, area weighting, common finite masks, station
persistence without future observations, missing snapshots, and deterministic
score row ordering. Existing metrics, plotting, and T7 tests also pass. The report
was replayed from saved data: all 1,440 regional score rows, 30 pooled station
score rows, and 1,200 station-by-lead rows reproduced exactly after sorting by
their identifiers. All four notebook code cells executed successfully with tables and embedded figures.

Observed GPU rollout time was approximately **36–37 seconds per ten-day case**
on one P40, excluding model loading and initial-condition retrieval. ERA5/IEM data
retrieval dominated the complete workflow. Regional output and snapshot storage
is small; global source chunks are not retained during truth retrieval.

Next priority is a broader validation set with a diurnally aware baseline,
climatological ACC, land/ocean separation, confirmed extreme events, and another
Perth station. Investigate wind-speed underprediction and late-lead temperature
errors before using downscaling to produce apparently more precise city forecasts.
GraphCast remains blocked on the current installed GPU stack; km-scale downscaling,
ensembles, station MSLP verification, and operational readiness are not delivered
by this milestone.
