#!/usr/bin/env python3
"""Step 11 - basin-scale response of the world's major rivers.

Grid-cell composites answer "where"; basins answer "how much water, and
for whom". For every major river basin this computes

* the area-weighted composite of each variable, lag by lag;
* the composite at the basin **outlet cell** for discharge, which is the
  number a downstream user actually experiences;
* the seasonal-mean anomaly in the developing, mature and decaying
  phases, with a bootstrap confidence interval over events;
* the same for the moderate/strong reference sample, so the basin table
  can report the super-event amplification directly.

The basin definition comes from ``auxiliary.basins`` in the config: a
polygon file (GeoPackage/shapefile, read with geopandas) or a
pre-rasterised basin-ID netCDF. Both paths end in the same integer
region-ID field on the analysis grid.

Run:  python workflow/11_basin_aggregation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from _common import composite_path, step_setup

from enso_inwaters import isimip_io as IO
from enso_inwaters import regions as R
from enso_inwaters import stats as S
from enso_inwaters.isimip_io import save_table


def add_args(p):
    p.add_argument("--classes", default="super,reference")
    p.add_argument("--variables", default=None)
    p.add_argument("--top-n", type=int, default=40,
                   help="report the N largest basins")


def load_basins(cfg, like: xr.DataArray, log):
    """Basin-ID field on the analysis grid, plus id -> name mapping."""
    spec = cfg["auxiliary.basins"]
    path = Path(cfg.root) / spec["file"]
    if not path.exists():
        log.warning("basin file %s not found - falling back to a "
                    "latitude/longitude tiling so the step still runs; the "
                    "resulting 'basins' are NOT river basins.", path)
        lat2d, lon2d = xr.broadcast(like["lat"], like["lon"])
        rid = (np.floor((lat2d + 90) / 30) * 12 +
               np.floor((lon2d + 180) / 30)).where(like.notnull())
        return rid.rename("region_id"), None

    if path.suffix in (".nc", ".nc4"):
        rid = R.masks_from_raster(path, like)
        return rid, None

    import geopandas as gpd
    gdf = gpd.read_file(path)
    name_field = next((c for c in ("RIVER_BASI", "MRBASIN", "NAME", "name",
                                   "River_Basi", "BASIN")
                       if c in gdf.columns), gdf.columns[0])
    gdf = gdf.to_crs("EPSG:4326")
    if "AREA_CALC" in gdf.columns:
        gdf = gdf.sort_values("AREA_CALC", ascending=False)
    log.info("%d basin polygons from %s (name field %r)",
             len(gdf), path.name, name_field)
    rid = R.masks_from_polygons(gdf, like, name_field=name_field)
    names = {i: str(n) for i, n in enumerate(gdf[name_field].astype(str))}
    return rid, names


def main() -> int:
    cfg, log, args = step_setup("11_basin_aggregation", __doc__, extra=add_args)
    comp_root = cfg.path("processed", "composites")
    out_root = cfg.path("processed", "basins", mkdir=True)
    clim = cfg["isimip.main_scenario.climate_scenario"]
    soc = cfg["isimip.main_scenario.soc_scenario"]
    seasons_of_interest = [s for g in cfg["composite.key_seasons"].values() for s in g]

    files = sorted(comp_root.glob(f"composite_*_super_{clim}_{soc}.nc"))
    if not files:
        log.error("no composites - run step 07 first")
        return 1
    variables = [f.stem.split("_")[1] for f in files]
    if args.variables:
        wanted = {v.strip() for v in args.variables.split(",")}
        variables = [v for v in variables if v in wanted]

    like = xr.open_dataset(files[0])["composite"].isel(model=0, lag=0, drop=True)
    region_id, names = load_basins(cfg, like, log)
    n_regions = int(np.unique(region_id.values[np.isfinite(region_id.values)]).size)
    log.info("%d region(s) on the analysis grid", n_regions)

    area = R.cell_area(like["lat"].values, like["lon"].values)
    rows, profiles = [], []

    for variable in variables:
        for klass in [c.strip() for c in args.classes.split(",")]:
            cpath = composite_path(cfg, variable, klass, clim, soc)
            if not cpath.exists():
                continue
            ds = xr.open_dataset(cpath)
            comp = ds["composite"].mean("model", skipna=True)      # (lag, lat, lon)
            per_event = ds["seasonal"].mean("model", skipna=True)  # (season, event, ...)

            agg = R.aggregate_regions(comp, region_id, names, weights=area)
            agg.name = "composite"
            profiles.append(agg.assign_coords(variable=variable,
                                              event_class=klass))

            for season in seasons_of_interest:
                if season not in [str(s) for s in per_event["season"].values]:
                    continue
                sea = per_event.sel(season=season)
                sea_agg = R.aggregate_regions(sea, region_id, names, weights=area)
                for region in sea_agg["region"].values:
                    vals = sea_agg.sel(region=region).values
                    lo, hi = S.bootstrap_ci(vals, n_boot=5000, ci=90,
                                            seed=int(cfg["statistics.bootstrap.seed"]))
                    rows.append({
                        "region": str(region), "variable": variable,
                        "event_class": klass, "season": season,
                        "mean_anomaly_sigma": float(np.nanmean(vals)),
                        "ci90_low": lo, "ci90_high": hi,
                        "n_events": int(np.isfinite(vals).sum()),
                        "sign_consistent": bool(np.all(np.sign(vals[np.isfinite(vals)])
                                                       == np.sign(np.nanmean(vals)))),
                    })
            log.info("%-12s %-10s aggregated over %d region(s)",
                     variable, klass, agg.sizes["region"])

    if not rows:
        log.error("nothing aggregated")
        return 1

    df = pd.DataFrame(rows)
    # amplification: super vs reference, same region/variable/season
    piv = df.pivot_table(index=["region", "variable", "season"],
                         columns="event_class", values="mean_anomaly_sigma")
    if {"super", "reference"} <= set(piv.columns):
        piv["amplification"] = piv["super"] / piv["reference"].where(
            np.abs(piv["reference"]) > 0.05)
        piv = piv.reset_index()
        save_table(piv, cfg.path("tables", "table_4_basin_amplification.csv"),
                   "11_basin_aggregation", cfg)
        log.info("median super/reference amplification: %.2f",
                 float(piv["amplification"].median(skipna=True)))

    if not args.dry_run:
        save_table(df, cfg.path("tables", "table_S5_basin_response.csv"),
                   "11_basin_aggregation", cfg)
        prof = xr.concat(profiles, dim="case") if profiles else None
        if prof is not None:
            IO.save_netcdf(prof.to_dataset(name="composite"),
                           out_root / f"basin_profiles_{clim}_{soc}.nc",
                           "11_basin_aggregation", cfg)
        region_id.to_dataset(name="region_id").to_netcdf(
            out_root / "region_id.nc")

    top = (df[(df.event_class == "super") & (df.season == "DJF01")]
           .reindex(df["mean_anomaly_sigma"].abs().sort_values(ascending=False).index)
           .head(15))
    log.info("largest mature-phase (DJF01) super-El-Nino basin anomalies:\n%s",
             top[["region", "variable", "mean_anomaly_sigma",
                  "ci90_low", "ci90_high"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
