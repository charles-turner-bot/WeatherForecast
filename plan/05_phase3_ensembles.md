# Phase 3 — Ensemble forecasting (with my recommendation)

**Your goal:** a really nice ensemble forecast with well-characterised uncertainties. You floated an
**autoencoder → latent space → sample many realisations → bootstrap uncertainties** approach.

Here's my honest recommendation on how to get there, and where the AE idea fits.

## Recommendation: two stages, baseline first

### 3a — Built-in perturbation ensembles (do this first, it's cheap here)

earth2studio ships perturbation methods designed exactly for this:
`Gaussian`, `Brown`, `Spherical Gaussian`, **`Correlated Spherical Gaussian`** (spatial + AR(1)
temporal correlation), **`Bred Vector`** and `Hemispheric Centred Bred Vector`. Bred vectors sample
the atmosphere's fastest-growing modes — the physically-motivated way to seed an IC ensemble.
The ensemble workflow (`run.ensemble(...)`) runs N perturbed members and writes to Zarr.

**Why first:** it gives a *working, defensible* ensemble in days, it's **embarrassingly parallel**
(the two P40s run members concurrently — see [04_hardware_notes.md](04_hardware_notes.md)), and it
gives us the **scoring harness** we need to judge *any* fancier method. The reference for scaling
this well is NVIDIA's **Huge Ensembles (HENS)** work (SFNO + bred-vector/noise).

**Deliverable 3a:** an N-member (start ~16–50) ensemble over Perth with a proper probabilistic
scorecard: **CRPS, spread–skill ratio, rank histograms, reliability**. This is what "well
characterised" actually means — calibration matters far more than member count or pretty spaghetti.

### 3b — Generative / latent-space ensemble (your research idea, grounded)

Your AE-latent idea is really a **generative ensemble**: learn a distribution over weather states
and sample members from it. That's a live, credible research direction — but let me flag the
subtleties honestly so we build the right version:

- **A plain autoencoder latent space is not a probability distribution.** Sampling it naively lands
  on arbitrary points that need not decode to physical states. To sample meaningfully you need
  either a **VAE** (probabilistic latent), a **density/flow/diffusion fit in latent space**, or to
  encode an existing ensemble/reanalysis population and sample *within that manifold*.
- **Be explicit about which uncertainty you're modelling** — initial-condition vs model error vs
  the full forecast distribution. "Bootstrapping" (resampling encoded states/analyses) mainly
  captures sampling uncertainty of the population you feed it; it is *not* the same as forecast
  growth of errors. Bred vectors/GenCast-style methods target the latter. We should say which we
  claim, and validate it.
- **Prior art to stand on, not reinvent:** **GenCast** (diffusion model where each sample *is* an
  ensemble member) is the strongest modern "generative ensemble"; latent-diffusion emulators and
  HENS are others. The AE-latent approach is one point in this space — worth trying, but judged
  against these.

**A clean framing that keeps your idea:** train a (V)AE on ERA5/BARRA states to get a latent, fit a
sampler in latent space (Gaussian in latent, or a small latent diffusion), decode to perturbed
initial states, run the global model per sample. Compare its **CRPS/reliability** head-to-head with
the 3a bred-vector ensemble. If it wins or adds cheap spread, it's real; if not, we learned that
cheaply.

## Why this ordering

The perturbation ensemble is low-risk, uses the hardware well, and — crucially — **builds the
scoring harness** that tells us whether the latent/generative approach is actually better rather
than just novel. Research thrust preserved; baseline protects us from chasing a dead end.

## Deliverables

- [ ] `run.ensemble` pipeline, config for method + N members, Zarr output.
- [ ] Scoring module: CRPS, spread–skill, rank histograms, reliability (vs ERA5 + Perth obs).
- [ ] 3a scorecard for bred-vector / correlated-Gaussian ensembles.
- [ ] 3b prototype: (V)AE + latent sampler, decoded IC perturbations, scored against 3a.
- [ ] Written comparison + recommendation.

## Exit criteria

A calibrated Perth ensemble with a full probabilistic scorecard, and an evidence-based verdict on
whether the latent/generative approach beats the perturbation baseline.
