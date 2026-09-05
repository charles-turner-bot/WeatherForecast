# Machine facts sheet + runtime decision

*(Deliverable requested by the vault's bring-up Phase 0. Measured 2026-09-05.)*

## Facts

| Item | Value |
|---|---|
| GPUs | **2× NVIDIA Tesla P40**, 24 GB GDDR5 each (48 GB total), Pascal `sm_61`, no tensor cores |
| GPU driver | **580.173.02** (supports up to CUDA 13) |
| CUDA toolkit (local) | 12.4 (`nvcc` r12.4) |
| Working torch | **2.5.1+cu124** — runs on P40s, **~9.2 TFLOP/s FP32** measured |
| CPU / RAM | 40 cores / **61 GB** |
| Disk | 218 GB total, **~84 GB free** on `/` (watch this before BARRA/ERA5 pulls) |
| Python | system 3.14 (too new); **pixi env pinned to 3.12** |
| pixi | 0.77.1 |
| **Container runtime** | **NONE installed** — no docker, no podman, no nvidia-container-toolkit |
| Ollama | present (local LLMs; see `/qwen` skill) |

## Decision: **local-env-first (pixi)** ✅

The vault (rightly) flagged that old GPU stacks are often easier to stabilise in **containers** than
via ad-hoc Python/JAX/PyTorch surgery, and recommended container-first. Two facts overrode that here:

1. **There is no container runtime on the box.** Container-first would first require installing
   Docker/Podman + the NVIDIA container toolkit + configuring GPU passthrough — all needing sudo and
   setup — before any model work could start.
2. **Local-env-first did not become painful.** A pinned `torch 2.5.1+cu124` runs on the P40s
   (major-version binary compat covers `sm_61` via `sm_60`), and earth2studio 0.18 installed with a
   single version-pin fix (`netcdf4 <1.7.3`). See [01_setup.md](01_setup.md).

**So:** proceed local-env-first with **pixi** as the one reproducible per-project env (exact
versions recorded in `pixi.toml` / `pixi.lock`). Keep containers as a **fallback** for a specific
repo that fights the local stack — most likely candidates are the **JAX** repos (GraphCast standalone,
NeuralGCM) and NVIDIA's **CorrDiff/earth2 NIM** images. If we hit that, install the NVIDIA container
toolkit then and use the vendor image for that repo only.

## Container fallback — when and how (deferred)

Trigger: a target repo needs a CUDA/JAX/cuDNN combo that won't co-exist with the pixi env.
Then: `nvidia-container-toolkit` install → `nvidia-smi` inside a CUDA base image to confirm
passthrough → use the vendor image (e.g. NVIDIA NGC earth2/modulus, or a JAX CUDA image) for that
one workflow. Record the working image tag here.
