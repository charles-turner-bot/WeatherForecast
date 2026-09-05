# Open questions & decisions to confirm

## For NVIDIA / earth2studio maintainers

- [ ] Recommended **global model** for a Southern-Hemisphere / WA regional focus (FCN3 vs GraphCast
      vs AIFS) — any known SH biases?
- [ ] Any **Australia CorrDiff** in progress, or a recommended **transfer-learning / fine-tuning**
      path from an existing regional CorrDiff to a new domain?
- [ ] **CorrDiff on Pascal (P40, FP32, no tensor cores):** realistic sampling cost, and the minimum
      sensible diffusion step count / any low-step samplers?
- [ ] Recommended **CorrDiff training recipe & compute** (Modulus) for a new ~4 km domain, and
      whether a regression-only (deterministic) downscaler is a reasonable v1.
- [ ] earth2studio **install extras** that are known-fiddly on CUDA 12.4 / older archs.

## For the BoM / ACCESS community

- [ ] **BARRA-C2 (4.4 km)** access path (NCI project? direct?), **license** for ML training, and
      which **variables/levels** are available at what cadence over WA.
- [ ] Existing **ACCESS/PyEarthTools BARRA loaders** we should reuse rather than rebuild.
- [ ] Best **Perth station observations** dataset for verification, and any BoM-preferred
      verification metrics/conventions.
- [ ] Is there **NCI Gadi** compute available to us for the heavy CorrDiff training (the P40s won't
      cut it for a full train)?
- [ ] Does BoM have an internal AI-downscaling effort we should align with / not duplicate?

## Internal / to decide as we go

- [ ] Exact **Perth domain** extent + target grid (draft in [02](02_phase1_global_forecast.md)).
- [ ] **Variable set** for v1 products (surface + which upper-air levels).
- [ ] **Ensemble size** vs cost on 2× P40 (start 16–50; scale per [05](05_phase3_ensembles.md)).
- [ ] **Storage plan** before Phase 2 (BARRA + paired samples will exceed the current ~84 GB free).
- [ ] Reproducibility: pin model checkpoints + earth2studio version; record per-model extras in
      [07_data_sources.md](07_data_sources.md).

## Known risks (tracked)

1. **PyTorch/earth2studio install on Pascal** — most likely early blocker. Mitigation in
   [01_setup.md](01_setup.md).
2. **Diffusion cost on P40** — may make km-scale slow; measure early (Phase 2 Route A).
3. **CorrDiff training compute** — likely needs off-box GPU; raise with NVIDIA and BoM/ACCESS early.
4. **Disk** — plan storage before pulling BARRA.
