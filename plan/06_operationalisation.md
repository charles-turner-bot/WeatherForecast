# Operationalisation — real-time forecasts (after research validates)

Deferred by decision (research-first). Captured here so we don't re-derive it later.

## What changes vs research mode

- **Initial conditions from live feeds:** switch the data source from ERA5 to **`data.GFS`**
  (NOAA, ~free, real-time) or **IFS** for the freshest analysis. Handle the ~4–6 h data latency
  and occasional missing cycles.
- **Scheduling:** run on each new IC cycle (GFS: 00/06/12/18 UTC). Use a simple `systemd` timer or
  cron on this box; the earth2studio pipeline itself is already a single command.
- **Idempotency & backfill:** key outputs by init-time; skip if present; backfill missed cycles.

## Outputs & delivery

- Write Zarr per cycle; render a standard Perth product set (t2m, wind, mslp, precip, ensemble
  spaghetti + probabilities) to PNG/HTML.
- Optional: a small dashboard (static site or lightweight app) showing latest deterministic +
  ensemble products for Perth. Could be published as an Artifact for easy sharing.

## Monitoring & reliability

- Log runtime, GPU memory, data-fetch success per cycle; alert on failure.
- Continuous verification: auto-score each cycle against Perth station obs as they arrive; track
  rolling skill so drift is visible.

## Hardware note

Real-time deterministic + a modest ensemble per 6-h cycle is within the 2× P40 budget **as long as
downscaling cost is controlled** (see [04_hardware_notes.md](04_hardware_notes.md)). If km-scale
CorrDiff proves too slow per cycle on Pascal, options: downscale only the deterministic run (not
every member), fewer diffusion steps, or offload to a bigger GPU.

## Not now

Auth/secrets management, alerting channels, and any public-facing publishing are out of scope until
Phases 1–3 validate. Listed so scope creep is a conscious choice, not an accident.
