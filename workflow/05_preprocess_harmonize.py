#!/usr/bin/env python3
"""Step 05 - harmonise the raw simulations onto a common analysis grid.

Different ISIMIP models ship different calendars, coordinate names,
longitude conventions and units. Nothing downstream should have to care,
so this step does all of it once:

* time axis -> pandas month-start timestamps (noleap/360-day/Gregorian all
  collapse onto the same monthly index);
* coordinates -> ``lat`` ascending, ``lon`` in -180..180;
* units -> fluxes to mm/day, temperatures to degrees Celsius, storages
  left in kg m-2 (= mm);
* the first ``period.discard_spinup_years`` years are dropped;
* a common land mask is applied so that every model covers the same
  cells (otherwise the multi-model mean is computed over a different
  domain for each variable);
* the field is written to ``data/interim/harmonized/`` as compressed
  float32 with provenance attributes.

The set of (sector, model, variable, scenario) combinations is taken
from the files actually on disk, not from the config, so a partial
download is handled gracefully.

Run:  python workflow/05_preprocess_harmonize.py [--sector water_global]
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import xarray as xr

# The configured dask chunks are a memory hint and need not line up
# with how a given model wrote its netCDF; xarray warns about that
# on every file, which drowns the log for a purely cosmetic cost.
warnings.filterwarnings(
    "ignore", message=".*chunks separate the stored chunks.*",
    category=UserWarning)

from _common import skip_existing, step_setup, table_path

from enso_inwaters import isimip_io as IO
from enso_inwaters.isimip_io import save_table
from enso_inwaters.regions import land_fraction_mask


def add_args(p):
    p.add_argument("--sector", default=None)
    p.add_argument("--variable", default=None)
    p.add_argument("--model", default=None)


def inventory(raw_root: Path) -> pd.DataFrame:
    """Everything ISIMIP-shaped that is actually on disk."""
    rows = []
    for f in sorted(raw_root.rglob("*.nc")):
        try:
            meta = IO.parse_filename(f)
        except ValueError:
            continue
        meta["path"] = str(f)
        meta["sector"] = f.parent.name
        meta["size_gb"] = round(f.stat().st_size / 1e9, 3)
        rows.append(meta)
    return pd.DataFrame(rows)


def harmonise(da: xr.DataArray, cfg, land: xr.DataArray | None) -> xr.DataArray:
    """Units, period, spin-up and land mask."""
    var = da.name
    if var in ("qtot", "qr", "evap", "potevap", "pr"):
        da = IO.to_mm_per_day(da)
    elif var in ("surftemp", "watertemp", "tas", "lakebottemp"):
        da = IO.kelvin_to_celsius(da)

    y0 = int(cfg["period.start"]) + int(cfg["period.discard_spinup_years"])
    y1 = int(cfg["period.end"])
    da = da.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))

    if land is not None and {"lat", "lon"} <= set(da.dims):
        da = da.where(land)
    return da


def main() -> int:
    cfg, log, args = step_setup("05_preprocess_harmonize", __doc__, extra=add_args)
    raw_root = cfg.path("raw", "isimip")
    out_root = cfg.path("interim", "harmonized", mkdir=True)

    inv = inventory(raw_root)
    if inv.empty:
        log.error("no ISIMIP-style netCDF files under %s - run step 03 "
                  "(or tests/make_synthetic_dataset.py)", raw_root)
        return 1
    save_table(inv, table_path(cfg, "table_S3_file_inventory.csv"),
               "05_preprocess_harmonize", cfg)
    log.info("%d input files, %.1f GB, %d model(s), %d variable(s)",
             len(inv), inv.size_gb.sum(), inv.model.nunique(), inv.variable.nunique())

    for col, val in (("sector", args.sector), ("variable", args.variable),
                     ("model", args.model)):
        if val:
            inv = inv[inv[col] == val]
    combos = (inv.groupby(["sector", "model", "variable", "climate_scenario",
                           "soc_scenario", "forcing", "timestep"])
                 .size().reset_index(name="n_files"))
    log.info("%d (sector, model, variable, scenario) combination(s) to process",
             len(combos))

    land = None
    chunks = dict(cfg["processing.chunks"])
    n_done = n_fail = 0

    for row in combos.itertuples():
        out = out_root / row.sector / (
            f"{row.model}_{row.variable}_{row.climate_scenario}_"
            f"{row.soc_scenario}.nc")
        if skip_existing(out, args, log):
            n_done += 1
            continue
        try:
            da = IO.open_model_data(
                raw_root, model=row.model, variable=row.variable,
                climate_scenario=row.climate_scenario,
                soc_scenario=row.soc_scenario, forcing=row.forcing,
                timestep=row.timestep, chunks=chunks)
        except Exception as exc:
            log.error("could not open %s/%s: %s", row.model, row.variable, exc)
            n_fail += 1
            continue

        if land is None and {"lat", "lon"} <= set(da.dims):
            land = land_fraction_mask(cfg.get("processing.land_mask"), da.isel(time=0))
            log.info("land mask: %d of %d cells",
                     int(land.sum()), int(land.size))

        out_da = harmonise(da, cfg, land)
        n_t = out_da.sizes.get("time", 0)
        expected = (int(cfg["period.end"]) - int(cfg["period.start"])
                    - int(cfg["period.discard_spinup_years"]) + 1) * 12
        if n_t < 0.9 * expected:
            log.warning("%s/%s covers only %d months (expected ~%d)",
                        row.model, row.variable, n_t, expected)

        if args.dry_run:
            log.info("would write %s (%d months)", out.name, n_t)
            continue

        out_da.attrs.update({
            "sector": row.sector, "model": row.model,
            "climate_scenario": row.climate_scenario,
            "soc_scenario": row.soc_scenario,
        })
        IO.save_netcdf(out_da.compute(), out, "05_preprocess_harmonize", cfg,
                       source_files=int(row.n_files))
        log.info("wrote %-58s %5d months  %s", out.name, n_t,
                 out_da.attrs.get("units", ""))
        n_done += 1

    log.info("harmonised %d combination(s), %d failure(s)", n_done, n_fail)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
