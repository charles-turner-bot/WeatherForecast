# Phase 0 — Environment setup

**Status:** base pixi env ✅ installed & verified. GPU DL stack ⬜ to do (this doc).

## What's already done

- `pixi.toml` with a conda-forge base: xarray, dask, zarr, netcdf4, cfgrib/eccodes (GRIB),
  **xesmf** (regridding), cartopy, s3fs/gcsfs (cloud data), jupyterlab. Verified importing.
- Two tasks: `pixi run lab` (JupyterLab), `pixi run gpu-check` (torch + CUDA visibility).

## The one real risk: PyTorch on Pascal (sm_61)

earth2studio ≥0.14 default wheels target **CUDA 13**. The P40s are **Pascal `sm_61`**. We must
install a torch build that (a) is CUDA-12.x compatible with the local stack and (b) still ships
`sm_61` kernels. Do this **before** installing earth2studio.

### Step 1 — Install a Pascal-compatible PyTorch

Add torch as a **pypi dependency with an explicit CUDA index**, pinned to a version known to
include `sm_61`. CUDA 12.4 wheels are the safe target for a P40 + driver 580.

```bash
# Option A (preferred): pip index for cu124 wheels, pinned.
pixi add --pypi "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu124
# (repeat for torchvision if a model needs it)
```

Then **verify sm_61 is present and the GPU actually computes** (availability alone isn't enough
on Pascal — confirm a real matmul runs):

```bash
pixi run gpu-check
pixi run python - <<'PY'
import torch
print("arch list:", torch.cuda.get_arch_list())        # must contain 'sm_61'
x = torch.randn(4096, 4096, device="cuda")
print("matmul ok:", (x @ x).sum().item() is not None)   # real kernel launch
print("bf16 supported:", torch.cuda.is_bf16_supported()) # expected False on P40
PY
```

**Fallbacks if `sm_61` is missing or matmul fails:**
1. Try `torch==2.5.1` / `2.3.1` cu124 wheels (arch coverage shifts between releases).
2. Use conda-forge `pytorch-gpu` (conda builds historically cover Pascal) instead of pip wheels.
3. Last resort: build torch from source with `TORCH_CUDA_ARCH_LIST=6.1`.

### Step 2 — Install earth2studio

Once torch imports and computes on the GPU:

```bash
# Start minimal; add model/data extras as Phase 1/2 require them.
pixi add --pypi earth2studio
# Model- and data-specific extras are installed per model — consult:
#   https://nvidia.github.io/earth2studio/main/userguide/about/install/
# NVIDIA also ship an install helper skill: `npx skills add NVIDIA/skills --skill earth2studio-install`
```

Some earth2studio models pull heavy/custom deps (e.g. spherical-harmonics `torch-harmonics`,
`nvidia-modulus`, ONNX runtime for Pangu). Install extras **per chosen model** rather than "all"
to keep the env solvable on this stack. Record exactly what each model needs in
[07_data_sources.md](07_data_sources.md) as we go.

### Step 3 — Model checkpoint cache

Model weights download on first use. Set a cache dir on the big disk and keep it out of git:

```bash
# add to pixi [activation.env] so every `pixi run` sees it
EARTH2STUDIO_CACHE = "/home/ct/WeatherForecast/.cache/earth2studio"
```

Some checkpoints require **NGC / Hugging Face auth**; note any token setup here when hit.

## Exit criteria for Phase 0 — ✅ ALL MET

- [x] `torch.cuda.is_available() == True`, both **Tesla P40s** visible (torch **2.5.1+cu124**).
- [x] CUDA matmul returns a real result at **~9.2 TFLOP/s FP32**. Note: arch list has `sm_60`
      (not `sm_61`) but runs on the P40 via major-version binary compatibility — no workaround needed.
- [x] `import earth2studio` succeeds — **earth2studio 0.18.0**; `data/run/perturbation/io/models.px/
      models.dx` all import.
- [x] End-to-end smoke test: fetched ERA5 `t2m` via `earth2studio.data.ARCO`, cropped to the Perth
      window (81×69 grid), value near Perth 2022-01-01 00Z = **20.0 °C** (sensible summer morning).
- [x] Checkpoint cache configured: `EARTH2STUDIO_CACHE=.cache/earth2studio` via `[activation.env]`,
      git-ignored.

### Notes for later phases (from the 0.18.0 install)

- **Downscaling (Phase 2):** available CorrDiff variants are `CorrDiff`, `CorrDiffTaiwan`,
  `CorrDiffCosmoEra5`, `CorrDiffCMIP6` — **no Australia model**, confirming the train-vs-BARRA-C2 plan.
- **Ensembles (Phase 3):** `GenCastMini` (diffusion ensemble) ships in `models.px` — a ready-made
  generative-ensemble baseline to benchmark the autoencoder/latent idea against.
- **Verification:** obs data sources `GHCNHourly`, `IEM_ASOS`, `ISD` are available — Perth station
  obs without waiting on BoM.

## Model compatibility on the P40 (empirical, 2026-09-05)

Hard-won from getting Phase 1 running. The P40 (Pascal, 24 GB) + CUDA-13-era package
ecosystem is hostile to most of the zoo. What we found:

| Model | Status | Why |
|---|---|---|
| **FCN** (classic FourCastNet, AFNO) | ✅ **works on GPU**, ~1 s/step, 0.25° | pure-torch via `nvidia-physicsnemo`. **Load-bearing pins:** `nvidia-physicsnemo<2.0` (2.0 imports `warp.context`, removed in warp-lang 1.17 → ImportError). Needs `r500`/`r850` **relative humidity** inputs → **use GFS/IFS**; ARCO/ERA5 lacks RH; the implemented `ERA5WithRH` adapter derives it via `DerivedRH` for historical runs (see [validation](13_historical_validation.md)). |
| **FCN3 / SFNO** | ❌ blocked | require `makani`, not pip-installable in this env. |
| **Pangu** (24/6/3) | ⚠️ installs (ONNX) & runs on GPU but **OOMs** | single-forward-pass peak ~2 GB over 24 GB (a 1.85 GB tensor). Runs on **CPU** (slow). Can't shard one ONNX model across the 2 P40s. **Pin `onnxruntime-gpu==1.22.0`** (CUDA 12; latest 1.29 targets CUDA 13, which the P40 can't use and whose libs aren't present). |
| **GraphCast** | ❌ blocked | needs `jax[cuda13]`; **CUDA 13 dropped Pascal**. |
| **DLWP** | (untried) | pure-torch, should fit — coarse/HEALPix fallback if needed. |

**Takeaway:** on this box, **classic FCN + GFS is the working GPU path** for a 0.25° global backbone.
Anything heavier (Pangu, FCN3, GraphCast) needs a bigger/newer GPU. Config default is FCN.
