# Compute and storage

Measured on the synthetic 4-degree test dataset and scaled to the
0.5-degree production grid (a factor of 64 in cells). Treat these as
order-of-magnitude estimates.

## Storage

| What | Size |
|---|---|
| ISIMIP3a `water_global`, 5 models x 7 variables x 2 climate x 2 soc, monthly | ~260 GB |
| ISIMIP3a `lakes_global`, 5 models x 4 variables | ~90 GB |
| Climate forcing (`pr`, `tas`, `rsds`) | ~40 GB |
| `data/interim` (harmonised + anomalies + indices) | ~180 GB |
| `data/processed` (composites, significance, contrasts) | ~20 GB |
| **Total** | **~590 GB** |

Halve it by dropping the counterclim and nosoc scenarios (at the cost of
Section 3.5 of the paper); halve it again by restricting to
`required_variables`.

## Time, 0.5-degree production grid

| Step | Cost | Notes |
|---|---|---|
| 03 download | 6-24 h | network-bound; restartable |
| 05 harmonise | ~2 h | I/O-bound |
| 06 anomalies | ~1.5 h | the moving climatology is O(n) per cell |
| 06 standardised indices | 6-20 h | **the bottleneck**: a gamma fit per cell per calendar month. Restrict `processing.indices.variables` |
| 07 composites | ~1 h | |
| 08 significance | ~5 min per variable with `--seasons-only`; ~30 min for the full lag axis | 10 000 iterations |
| 09-17 analysis | ~2 h total | |
| 20-31 figures and tables | ~15 min | |

A workstation with 8 cores, 32 GB of RAM and 700 GB of disk runs the
whole thing in about two days, most of it download and the gamma fits.

## Memory

The largest single object is one variable's anomaly field:
119 years x 12 months x 360 x 720 x 8 bytes = ~3 GB in float64. The
moving climatology processes the spatial axis in blocks, so peak usage
stays near 6-8 GB per variable. `processing.chunks` in the config
controls the dask chunking for everything else.

## Running on a cluster

Each numbered step is a separate process with no shared state, so the
natural parallelisation is one job per (variable, scenario):

```bash
#!/bin/bash
#SBATCH --array=0-27 --cpus-per-task=4 --mem=16G --time=8:00:00
VARS=(dis qtot qr tws rootmoist groundwstor surftemp)
SCEN=(obsclim_histsoc obsclim_nosoc counterclim_histsoc)
V=${VARS[$((SLURM_ARRAY_TASK_ID % 7))]}
S=${SCEN[$((SLURM_ARRAY_TASK_ID / 7))]}
python workflow/06_anomalies_and_indices.py --variable "$V"
```

Steps 08 onwards need all models of a variable present, so run them
after the per-variable array jobs have finished.
