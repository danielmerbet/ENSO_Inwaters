#!/usr/bin/env python3
"""Step 12 - lake response: surface temperature, ice and stratification.

Lakes integrate the atmosphere differently from rivers. Their surface
temperature responds within weeks to the air-temperature and radiation
anomalies of an El Nino, but deep lakes carry that heat for seasons, and
the ice season of a boreal lake can shift by weeks. This step extracts,
from the ISIMIP ``lakes_global`` ensemble:

* composite lake-surface-water-temperature (LSWT) anomalies by lag and
  season, with the same Monte-Carlo significance treatment as the
  hydrological fields;
* **ice phenology**: ice-covered months per year, and the shift in ice
  duration during and after a super El Nino;
* a **latitude/depth breakdown**, because the LSWT response is expected
  to scale with the inverse of mixed-layer depth, and a super El Nino
  should therefore show up most strongly in small, shallow lakes;
* the largest named lakes individually, when a lake polygon file is
  configured.

Run:  python workflow/12_lake_analysis.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from _common import composite_path, step_setup

from enso_inwaters import regions as R
from enso_inwaters import stats as S
from enso_inwaters.isimip_io import save_table

LAT_BANDS = {
    "tropical (23S-23N)": (-23.5, 23.5),
    "subtropical N (23-35N)": (23.5, 35.0),
    "subtropical S (35-23S)": (-35.0, -23.5),
    "midlatitude N (35-55N)": (35.0, 55.0),
    "midlatitude S (55-35S)": (-55.0, -35.0),
    "boreal (55-72N)": (55.0, 72.0),
}


def add_args(p):
    p.add_argument("--classes", default="super,reference")


def ice_metrics(icefrac: xr.DataArray, threshold: float = 0.5) -> xr.DataArray:
    """Ice-covered months per calendar year."""
    iced = (icefrac > threshold)
    return iced.groupby("time.year").sum("time").rename("ice_months")


def main() -> int:
    cfg, log, args = step_setup("12_lake_analysis", __doc__, extra=add_args)
    clim = cfg["isimip.main_scenario.climate_scenario"]
    soc = cfg["isimip.main_scenario.soc_scenario"]
    events = pd.read_csv(cfg.path("processed", "enso", "enso_events.csv"))
    seasons = [s for g in cfg["composite.key_seasons"].values() for s in g]

    lake_vars = list(cfg.get("isimip.sectors.lakes_global.variables", {}))
    log.info("lake variables in the configuration: %s", lake_vars)

    rows = []
    for variable in lake_vars:
        for klass in [c.strip() for c in args.classes.split(",")]:
            cpath = composite_path(cfg, variable, klass, clim, soc)
            if not cpath.exists():
                log.warning("no composite for %s/%s", variable, klass)
                continue
            ds = xr.open_dataset(cpath)
            comp = ds["composite"].mean("model", skipna=True)
            per_event = ds["seasonal"].mean("model", skipna=True)
            n_models = ds["composite"].sizes["model"]
            agreement = S.sign_agreement(ds["composite"], "model")

            for band, (lo, hi) in LAT_BANDS.items():
                sel = comp.sel(lat=slice(lo, hi))
                if sel.sizes.get("lat", 0) == 0 or not bool(sel.notnull().any()):
                    continue
                prof = R.global_mean(sel)
                k = int(np.nanargmax(np.abs(prof.values)))
                for season in seasons:
                    if season not in [str(s) for s in per_event["season"].values]:
                        continue
                    vals = R.global_mean(
                        per_event.sel(season=season).sel(lat=slice(lo, hi))).values
                    ci = S.bootstrap_ci(np.atleast_1d(vals), n_boot=5000, ci=90,
                                        seed=int(cfg["statistics.bootstrap.seed"]))
                    rows.append({
                        "variable": variable, "event_class": klass,
                        "band": band, "season": season,
                        "mean_anomaly": float(np.nanmean(vals)),
                        "ci90_low": ci[0], "ci90_high": ci[1],
                        "peak_lag": int(comp["lag"].values[k]),
                        "peak_amplitude": float(prof.values[k]),
                        "n_models": int(n_models),
                        "mean_sign_agreement": float(
                            agreement.sel(lat=slice(lo, hi)).mean()),
                    })
            log.info("%-12s %-10s summarised over %d latitude band(s)",
                     variable, klass, len(LAT_BANDS))

    # --- ice phenology -------------------------------------------------
    ice_rows = []
    # ice duration is counted on the raw fraction, not on the anomaly
    harmon = cfg.path("interim", "harmonized", "lakes_global")
    ice_files = sorted(harmon.glob(f"*_lakeicefrac_{clim}_{soc}.nc"))
    super_years = set(E_years(events, "super"))
    ref_years = set(E_years(events, "reference"))
    for f in ice_files:
        model = f.stem.split("_")[0]
        ds = xr.open_dataset(f)
        ice = ds[next(iter(ds.data_vars))]
        months = ice_metrics(ice)
        clim_mean = months.mean("year", skipna=True)
        for label, yrs in (("super", super_years), ("reference", ref_years)):
            yy = [y for y in yrs if y in months["year"].values]
            if not yy:
                continue
            anom = (months.sel(year=yy).mean("year", skipna=True) - clim_mean)
            ice_rows.append({
                "model": model, "event_class": label, "n_events": len(yy),
                "mean_ice_month_anomaly": float(R.global_mean(anom)),
                "min_ice_month_anomaly": float(anom.min()),
                "frac_cells_less_ice": float((anom < 0).mean()),
            })
        log.info("%-14s ice phenology computed", model)

    if rows and not args.dry_run:
        df = pd.DataFrame(rows)
        save_table(df, cfg.path("tables", "table_5_lake_response.csv"),
                   "12_lake_analysis", cfg)
        log.info("\n%s", df[df.season == "DJF01"].to_string(index=False))
    if ice_rows and not args.dry_run:
        save_table(pd.DataFrame(ice_rows),
                   cfg.path("tables", "table_S6_lake_ice.csv"),
                   "12_lake_analysis", cfg)
    return 0


def E_years(events: pd.DataFrame, klass: str) -> list[int]:
    from enso_inwaters.enso import select_event_class
    return select_event_class(events, klass)["year0"].astype(int).tolist()


if __name__ == "__main__":
    raise SystemExit(main())
