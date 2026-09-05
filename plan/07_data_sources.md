# Data sources & packages

## earth2studio built-in data sources

| Source | Use | Notes |
|---|---|---|
| **ERA5** (ARCO / CDS) | Research ICs, verification truth | Reproducible; ARCO-ERA5 is a public analysis-ready Zarr on GCS (via `gcsfs`). CDS needs an API key. |
| **GFS** (NOAA) | Real-time ICs | Free, ~0.25°, 6-h cycles. Phase → operational. |
| **IFS** (ECMWF) | Real-time ICs (alt.) | Higher quality; check access/licensing. |
| **GEFS** | Ensemble ICs / CorrDiff input | Used by the US CorrDiff variant. |

Confirm exact source class names against the installed earth2studio version's docs.

## Australian high-res data (for Phase 2 downscaling)

| Dataset | Res | Role | Access |
|---|---|---|---|
| **BARRA-R2** | ~12 km | Intermediate / parent reanalysis | BoM / **NCI** (ACCESS community). |
| **BARRA-C2** | **4.4 km** | **Km-scale training target** (convection-permitting) | BoM / NCI. **Confirm license + variables with BoM / ACCESS.** |
| Perth station obs | point | Verification (t2m, wind, mslp, rain) | BoM. Perth Airport / Perth Metro. |

BARRA-C2 at 4.4 km is the natural truth for a WA downscaler. Data volume is large — plan storage
(see disk note in [04_hardware_notes.md](04_hardware_notes.md)) and tile to the Perth domain.

## PyEarthTools (noted, per decision — earth2studio stays primary)

- `ACCESS-Community-Hub/PyEarthTools` — ML framework for Earth-system science, modular sub-packages,
  runs laptop→HPC. Docs: pyearthtools.readthedocs.io.
- **Where it likely helps us:** AU-centric **data access (BARRA/ACCESS on NCI)**, indexing,
  regridding/patching pipelines, and NCI Gadi workflows — i.e. the *data-engineering* side of
  Phase 2, complementing (not replacing) earth2studio's inference stack.
- **Action:** evaluate PyEarthTools specifically as the BARRA-C2 access/tiling layer once Phase 2
  starts; check whether BoM / ACCESS already have BARRA loaders we should reuse.

## Formats & tooling (installed)

Zarr for outputs/paired samples; cfgrib/eccodes for GRIB (GFS/IFS); xesmf for regridding;
s3fs/gcsfs for cloud stores; xarray/dask throughout.
