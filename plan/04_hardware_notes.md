# Hardware notes — 2× Tesla P40

## The machine

| | |
|---|---|
| GPU | 2× **Tesla P40**, 24 GB GDDR5 each (**48 GB** total) |
| Arch | **Pascal, `sm_61`** — no tensor cores |
| Precision | FP32 **~9.2 TFLOP/s measured** (torch 2.5.1); **FP16 ≈ 1/64 FP32** (Pascal penalty) |
| Driver / CUDA | 580.173.02 (supports ≤ CUDA 13); local toolkit 12.4 |
| Host | 61 GB RAM, 40 cores, ~84 GB free on `/` |

### Verified install (Phase 0 ✅)

`torch==2.5.1+cu124` from the pytorch cu124 wheel index works on the P40s:
- `torch.cuda.is_available() == True`, both P40s visible.
- Arch list is `sm_50, sm_60, sm_70…sm_90` — **`sm_61` is not listed, but `sm_60` is**, and CUDA
  binaries are forward-compatible within a major version (6.0 → 6.1). A real CUDA matmul runs at
  **~9.2 TFLOP/s FP32**, confirming compute works. No source build or arch-list workaround needed.
- Caveat: `torch.cuda.is_bf16_supported()` returns **True**, but that's CUDA-version/emulation, not
  hardware tensor cores — BF16 will be *correct but slow*. **Stick to FP32.**

## What this means for AI weather

**Run everything in FP32.** Do **not** use `autocast`/FP16 — on Pascal it's slower, not faster.
BF16 is unavailable. Watch for model code that hard-assumes tensor-core mixed precision.

- **Global inference (Phase 1): fine.** A 0.25° global model in FP32 fits comfortably in 24 GB and
  a 10-day rollout is minutes, not hours. This phase is not hardware-limited.
- **Diffusion downscaling (Phase 2): the pinch point.** CorrDiff sampling runs the network many
  times per field (denoising steps). In FP32 on Pascal this is slow — measure it early (Route A).
  Levers: fewer sampling steps, smaller tiles, batch across the two GPUs.
- **Training a downscaler (Phase 2B): the hard part.** Diffusion training wants tensor cores and
  lots of memory bandwidth — both weak here. Feasible for a *small-domain prototype*; a full
  multi-year WA train likely needs a bigger GPU (cloud A100/H100, or **NCI Gadi via ACCESS**).
  Plan the data pipeline here; plan the heavy train elsewhere.
- **Ensembles (Phase 3): embarrassingly parallel.** Perturbation ensembles are just N global runs —
  cheap here, and the **two GPUs** let us run members concurrently.

## Practical

- **Two GPUs, no NVLink** → use them as independent workers (one member/tile per GPU via
  `CUDA_VISIBLE_DEVICES`), not model-parallel. Simple and effective for ensembles/tiling.
- **Disk:** ERA5/BARRA + checkpoints + Zarr outputs are large; only ~84 GB free. Put caches/outputs
  under `WeatherForecast/.cache` and prune; consider an external/mounted volume before Phase 2B.
- **Verify, don't assume:** confirm `sm_61` in `torch.cuda.get_arch_list()` and that a real CUDA
  matmul runs (see [01_setup.md](01_setup.md)) — the most likely install failure mode here.
