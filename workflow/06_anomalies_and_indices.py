#!/usr/bin/env python3
"""Step 06 - anomalies and standardised drought/flood indices.

For every harmonised field this produces

* **standardised anomalies** (sigma units) against a centred moving
  30-year climatology, which removes the 1901-2019 drift and the
  seasonal cycle while leaving the 2-7 year ENSO band untouched; and
* **standardised indices** SRI/SSI/SGI at 1-, 3- and 12-month
  accumulations, gamma-fitted per calendar month, for the drought and
  flood statistics of step 13.

Both are written to ``data/interim/`` and are the only inputs the
compositing steps need.

Why standardise? A composite that averages the Amazon (10^5 m3/s) with
the Limpopo (10^2 m3/s) in physical units is just a map of the Amazon.
Sigma units make the *relative* hydrological perturbation comparable
across climates, which is the quantity the paper is about. Absolute
anomalies are kept alongside for the basin-scale water-balance numbers.

Run:  python workflow/06_anomalies_and_indices.py [--variable dis]
"""

from __future__ import annotations

import warnings
from pathlib import Path

import xarray as xr

# The configured dask chunks are a memory hint and need not line up
# with how a given model wrote its netCDF; xarray warns about that
# on every file, which drowns the log for a purely cosmetic cost.
warnings.filterwarnings(
    "ignore", message=".*chunks separate the stored chunks.*",
    category=UserWarning)

from _common import skip_existing, step_setup

from enso_inwaters import isimip_io as IO
from enso_inwaters.climatology import anomalies, standardized_index

# Which variables get a standardised drought/flood index, and its name.
INDEX_FOR = {
    "dis": "ssi",          # Standardized Streamflow Index
    "qtot": "sri",         # Standardized Runoff Index
    "qr": "sgri",          # Standardized Groundwater Recharge Index
    "groundwstor": "sgi",  # Standardized Groundwater level/storage Index
    "rootmoist": "ssmi",   # Standardized Soil Moisture Index
}


def add_args(p):
    p.add_argument("--sector", default=None)
    p.add_argument("--variable", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--skip-indices", action="store_true",
                   help="anomalies only (the gamma fits are the slow part)")


def parse_name(path: Path) -> dict:
    model, variable, clim, soc = path.stem.split("_", 3)
    soc = soc.split("_")[0]
    return {"model": model, "variable": variable,
            "climate_scenario": clim, "soc_scenario": soc}


def main() -> int:
    cfg, log, args = step_setup("06_anomalies_and_indices", __doc__, extra=add_args)
    src_root = cfg.path("interim", "harmonized")
    anom_root = cfg.path("interim", "anomalies", mkdir=True)
    idx_root = cfg.path("interim", "indices", mkdir=True)

    files = sorted(src_root.rglob("*.nc"))
    if not files:
        log.error("nothing in %s - run step 05 first", src_root)
        return 1

    opt = cfg["processing.anomaly"]
    iopt = cfg["processing.indices"]
    chunks = dict(cfg["processing.chunks"])
    window = 30 if str(opt["climatology"]).startswith("moving") else None
    n_done = n_fail = 0

    for f in files:
        meta = parse_name(f)
        sector = f.parent.name
        if args.sector and sector != args.sector:
            continue
        if args.variable and meta["variable"] != args.variable:
            continue
        if args.model and meta["model"] != args.model:
            continue

        out_anom = anom_root / sector / (f.stem + "_anom.nc")
        if not skip_existing(out_anom, args, log):
            try:
                da = xr.open_dataarray(f, chunks=chunks) \
                    if len(xr.open_dataset(f).data_vars) == 1 \
                    else xr.open_dataset(f, chunks=chunks)[meta["variable"]]
                anom = anomalies(
                    da, method=opt["method"],
                    climatology=opt["climatology"],
                    window_years=window or 30,
                    min_years=int(opt["min_years_for_std"]),
                    base=(int(cfg["period.climatology_start"]),
                          int(cfg["period.climatology_end"])),
                    detrend=opt["detrend"],
                    min_std_relative=float(opt["min_std_relative"]))
                valid = float(anom.notnull().mean())
                log.info("%-48s anomalies  valid=%.1f%%", f.name, 100 * valid)
                if valid < 0.05:
                    log.warning("  -> almost everything is masked; check the "
                                "units and the land mask for %s", f.name)
                if not args.dry_run:
                    IO.save_netcdf(anom.compute(), out_anom,
                                   "06_anomalies_and_indices", cfg, **meta)
                n_done += 1
            except Exception as exc:
                log.error("anomalies failed for %s: %s", f.name, exc)
                n_fail += 1
                continue

        index_name = INDEX_FOR.get(meta["variable"])
        wanted = iopt.get("variables") or list(INDEX_FOR)
        main_only = bool(iopt.get("main_scenario_only", True))
        is_main = (meta["climate_scenario"] == cfg["isimip.main_scenario.climate_scenario"]
                   and meta["soc_scenario"] == cfg["isimip.main_scenario.soc_scenario"])
        if (args.skip_indices or index_name is None
                or meta["variable"] not in wanted
                or (main_only and not is_main)):
            continue
        for scale in iopt["sri_scales"]:
            out_idx = idx_root / sector / (
                f"{f.stem}_{index_name}{scale}.nc")
            if skip_existing(out_idx, args, log):
                continue
            try:
                da = xr.open_dataset(f, chunks=chunks)[meta["variable"]]
                si = standardized_index(da, scale=int(scale),
                                        distribution=iopt["distribution"])
                si.name = f"{index_name}{scale}"
                if not args.dry_run:
                    IO.save_netcdf(si.compute(), out_idx,
                                   "06_anomalies_and_indices", cfg,
                                   index=index_name, scale=scale, **meta)
                log.info("%-48s %s-%d", f.name, index_name.upper(), scale)
            except Exception as exc:
                log.error("%s-%s failed for %s: %s", index_name, scale, f.name, exc)
                n_fail += 1

    log.info("done: %d field(s) processed, %d failure(s)", n_done, n_fail)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
