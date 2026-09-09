#!/usr/bin/env python3
"""Step 15 - who is exposed.

Translates the composite fields into the numbers a policy reader needs:
land area, population and irrigated cropland in regions where a super
El Nino drives a robust hydrological anomaly, and how those numbers
change relative to moderate/strong events.

Population comes from HYDE 3.2, sampled at the year of each event, so
the 1972 event is scored against the 1972 population and 2015 against
the 2015 population; the aggregate "people affected by super El Ninos"
therefore reflects both the hydrology and the growth of the exposed
population over the century. Both are reported separately, because
conflating them is a common way to overstate a climate signal.

Run:  python workflow/15_exposure.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr
from _common import step_setup

from enso_inwaters import enso as E
from enso_inwaters import regions as R
from enso_inwaters.isimip_io import normalise_coords, save_table


def add_args(p):
    p.add_argument("--variables", default="dis,qtot,qr,tws")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="|anomaly| in sigma that counts as 'affected'")


def load_population(cfg, like: xr.DataArray, log) -> xr.DataArray | None:
    path = Path(cfg.root) / cfg["auxiliary.population.file"]
    if not path.exists():
        log.warning("population file %s not found - exposure will be reported "
                    "as area only", path)
        return None
    ds = xr.open_dataset(path)
    pop = normalise_coords(ds[next(iter(ds.data_vars))])
    if "time" in pop.dims:
        pop = pop.rename({"time": "year"}) if pop["time"].dtype.kind in "iu" else pop
    pop = pop.interp(lat=like["lat"], lon=like["lon"], method="linear")
    # densities (people per km2) become counts; counts are left alone
    units = (pop.attrs.get("units") or "").lower()
    if "km" in units or "density" in units:
        pop = pop * R.cell_area(like["lat"].values, like["lon"].values) / 1e6
        pop.attrs["units"] = "people"
    log.info("population field: %s, global total %.2f billion",
             path.name, float(pop.sum()) / 1e9
             if "year" not in pop.dims else float(pop.isel(year=-1).sum()) / 1e9)
    return pop


def main() -> int:
    cfg, log, args = step_setup("15_exposure", __doc__, extra=add_args)
    sig_root = cfg.path("processed", "significance")
    clim = cfg["isimip.main_scenario.climate_scenario"]
    soc = cfg["isimip.main_scenario.soc_scenario"]
    events = pd.read_csv(cfg.path("processed", "enso", "enso_events.csv"))
    thr = args.threshold

    files = sorted(sig_root.glob(f"significance_*_{clim}_{soc}.nc"))
    if not files:
        log.error("no significance files - run step 08 first")
        return 1

    like = xr.open_dataset(files[0])["composite"]
    like = like.isel({d: 0 for d in like.dims if d not in ("lat", "lon")}, drop=True)
    pop = load_population(cfg, like, log)

    rows = []
    wanted = {v.strip() for v in args.variables.split(",")}
    for f in files:
        parts = f.stem.split("_")
        variable, klass = parts[1], "_".join(parts[2:-2])
        if variable not in wanted:
            continue
        ds = xr.open_dataset(f)
        comp = ds["composite"]
        robust = ds["robust"].astype(bool)
        dim = "season" if "season" in comp.dims else "lag"
        for value in comp[dim].values:
            c = comp.sel({dim: value})
            rb = robust.sel({dim: value})
            dry = (c <= -thr) & rb
            wet = (c >= thr) & rb
            row = {"variable": variable, "event_class": klass,
                   dim: str(value),
                   "area_dry_1e6km2": R.exposed_area_km2(dry) / 1e6,
                   "area_wet_1e6km2": R.exposed_area_km2(wet) / 1e6}
            if pop is not None:
                p = pop
                if "year" in p.dims:
                    yrs = E.select_event_class(events, klass)["year0"].astype(int)
                    avail = p["year"].values
                    pick = [min(avail, key=lambda a, y=y: abs(a - y)) for y in yrs]
                    p = p.sel(year=pick).mean("year")
                row["pop_dry_millions"] = R.exposed_population(dry, p) / 1e6
                row["pop_wet_millions"] = R.exposed_population(wet, p) / 1e6
            rows.append(row)
        log.info("%-12s %-12s exposure computed over %d %s",
                 variable, klass, comp.sizes[dim], dim)

    if not rows:
        log.error("nothing to report")
        return 1
    df = pd.DataFrame(rows)

    # super vs reference amplification of exposure
    key = [c for c in ("season", "lag") if c in df.columns][0]
    piv = df.pivot_table(index=["variable", key], columns="event_class",
                         values=[c for c in df.columns
                                 if c.startswith(("area_", "pop_"))])
    if not args.dry_run:
        save_table(df, cfg.path("tables", "table_8_exposure.csv"),
                   "15_exposure", cfg)
        piv.reset_index().to_csv(
            cfg.path("tables", "table_S7_exposure_by_class.csv"), index=False)

    mature = df[df.get(key).astype(str).isin(["DJF01", "0", "1"])]
    if not mature.empty:
        log.info("mature-phase exposure:\n%s", mature.to_string(index=False))
    log.warning("Exposure numbers count people living in cells with a robust "
                "anomaly; they are not an impact assessment. Say so in the "
                "paper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
