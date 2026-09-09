#!/usr/bin/env python3
"""Step 13 - hydrological extremes: droughts and floods during super events.

Composite means understate what matters to people. A basin whose mean
anomaly is -0.6 sigma may be in outright drought for a year. This step
therefore works on the standardised indices from step 06 (SSI for
streamflow, SRI for runoff, SGI for groundwater) and asks, for the
event windows:

* how many **drought months** (index < -1) and **flood months**
  (index > +1) occur in the 24 months following the event peak, against
  the climatological expectation;
* the **maximum event duration** and the **minimum index value**
  reached, which is the severity metric a water manager uses;
* the **odds ratio** of drought/flood occurrence in super-El-Nino
  windows relative to neutral windows, with a Monte-Carlo confidence
  interval -- a cleaner effect measure than a difference of counts when
  the base rate varies enormously between climates;
* the land area and the number of basins that flip from "no drought" to
  "drought" in the super composite but not in the reference composite.

Run:  python workflow/13_extremes.py [--index ssi3]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from _common import skip_existing, step_setup

from enso_inwaters import enso as E
from enso_inwaters import isimip_io as IO
from enso_inwaters import regions as R
from enso_inwaters import stats as S
from enso_inwaters.isimip_io import save_table


def add_args(p):
    p.add_argument("--window", default="0,23",
                   help="lag window in months relative to the event peak")
    p.add_argument("--classes", default="super,reference,all_el_nino")


def window_counts(index: xr.DataArray, years, lo: int, hi: int,
                  drought_thr: float, flood_thr: float) -> xr.Dataset:
    """Drought/flood month counts in the [lo, hi] window of each event."""
    drought, flood, minimum, maximum, n_used = [], [], [], [], 0
    for y in years:
        anchor = pd.Timestamp(year=int(y), month=12, day=1)
        start = anchor + pd.DateOffset(months=int(lo))
        end = anchor + pd.DateOffset(months=int(hi))
        sub = index.sel(time=slice(start, end))
        if sub.sizes.get("time", 0) < (hi - lo + 1) // 2:
            continue
        drought.append((sub < drought_thr).sum("time"))
        flood.append((sub > flood_thr).sum("time"))
        minimum.append(sub.min("time"))
        maximum.append(sub.max("time"))
        n_used += 1
    if not n_used:
        raise ValueError("no event window falls inside the record")
    dim = pd.Index(range(n_used), name="event")
    return xr.Dataset({
        "drought_months": xr.concat(drought, dim=dim).mean("event"),
        "flood_months": xr.concat(flood, dim=dim).mean("event"),
        "min_index": xr.concat(minimum, dim=dim).mean("event"),
        "max_index": xr.concat(maximum, dim=dim).mean("event"),
    }, attrs={"n_events": n_used, "window": f"{lo}..{hi}"})


def main() -> int:
    cfg, log, args = step_setup("13_extremes", __doc__, extra=add_args)
    idx_root = cfg.path("interim", "indices")
    out_root = cfg.path("processed", "extremes", mkdir=True)
    clim = cfg["isimip.main_scenario.climate_scenario"]
    soc = cfg["isimip.main_scenario.soc_scenario"]
    events = pd.read_csv(cfg.path("processed", "enso", "enso_events.csv"))
    lo, hi = (int(v) for v in args.window.split(","))
    dthr = float(cfg["processing.indices.drought_threshold"])
    fthr = float(cfg["processing.indices.flood_threshold"])
    n_boot = int(cfg["statistics.bootstrap.n_iterations"])
    seed = int(cfg["statistics.bootstrap.seed"])

    files = sorted(idx_root.rglob(f"*_{clim}_{soc}_*.nc"))
    if not files:
        log.error("no standardised indices in %s - run step 06 "
                  "(without --skip-indices)", idx_root)
        return 1

    # group index files by (variable, index name) across models
    groups: dict[tuple[str, str], list] = {}
    for f in files:
        parts = f.stem.split("_")
        model, variable, index_name = parts[0], parts[1], parts[-1]
        groups.setdefault((variable, index_name), []).append((model, f))
    log.info("index fields: %s", sorted(groups))

    all_years = np.unique(pd.DatetimeIndex(
        xr.open_dataset(files[0])["time"].values).year)
    event_years_all = events.loc[events.in_main_sample, "year0"].astype(int).values
    rng = np.random.default_rng(seed)
    rows = []

    for (variable, index_name), members in sorted(groups.items()):
        arrays = {}
        for model, f in members:
            ds = xr.open_dataset(f)
            arrays[model] = ds[next(iter(ds.data_vars))]
        ens = IO.concat_models(arrays).mean("model", skipna=True)
        log.info("%s/%s: %d model(s)", variable, index_name, len(arrays))

        # climatological baseline: the same window drawn from neutral years
        neutral = S.eligible_anchor_years(all_years, event_years_all, 1)
        neutral = [y for y in neutral if y + (hi // 12) + 1 <= all_years.max()]
        base = window_counts(ens, neutral, lo, hi, dthr, fthr)

        results = {}
        for klass in [c.strip() for c in args.classes.split(",")]:
            yrs = E.select_event_class(events, klass)["year0"].astype(int).tolist()
            yrs = [y for y in yrs if y + (hi // 12) + 1 <= all_years.max()]
            if len(yrs) < 2:
                log.warning("  %s: too few events inside the record", klass)
                continue
            wc = window_counts(ens, yrs, lo, hi, dthr, fthr)
            results[klass] = wc

            n_months = hi - lo + 1
            # odds ratio of a drought month, event vs neutral windows
            pe = (wc["drought_months"] / n_months).clip(1e-4, 1 - 1e-4)
            pn = (base["drought_months"] / n_months).clip(1e-4, 1 - 1e-4)
            odds = (pe / (1 - pe)) / (pn / (1 - pn))
            # Most land cells never reach the drought threshold in either
            # sample, so an odds ratio pooled over all of them is pinned at
            # 1.0 and says nothing. Summarise only where drought actually
            # occurs in the neutral baseline or in the event windows.
            occurs = (base["drought_months"] > 0) | (wc["drought_months"] > 0)
            odds_where_relevant = odds.where(occurs)

            # bootstrap the neutral baseline to get a CI on the odds ratio
            boots = []
            for _ in range(min(n_boot, 1000)):
                pick = rng.choice(neutral, size=len(yrs), replace=False)
                bw = window_counts(ens, pick, lo, hi, dthr, fthr)
                pb = (bw["drought_months"] / n_months).clip(1e-4, 1 - 1e-4)
                boots.append(float(R.global_mean(pb)))
            pe_g = float(R.global_mean(pe))
            lo_ci, hi_ci = np.percentile(boots, [5, 95]) if boots else (np.nan, np.nan)

            ds_out = xr.merge([wc.rename({v: f"{v}_event" for v in wc.data_vars}),
                               base.rename({v: f"{v}_neutral" for v in base.data_vars})])
            ds_out["drought_odds_ratio"] = odds_where_relevant
            ds_out.attrs.update({"variable": variable, "index": index_name,
                                 "event_class": klass, "window": f"{lo}..{hi}",
                                 "n_events": len(yrs)})
            out = out_root / f"extremes_{variable}_{index_name}_{klass}_{clim}_{soc}.nc"
            if not skip_existing(out, args, log) and not args.dry_run:
                IO.save_netcdf(ds_out, out, "13_extremes", cfg)

            rows.append({
                "variable": variable, "index": index_name, "event_class": klass,
                "n_events": len(yrs), "window_months": f"{lo}..{hi}",
                "mean_drought_months_event": float(R.global_mean(wc["drought_months"])),
                "mean_drought_months_neutral": float(R.global_mean(base["drought_months"])),
                "mean_flood_months_event": float(R.global_mean(wc["flood_months"])),
                "mean_flood_months_neutral": float(R.global_mean(base["flood_months"])),
                "drought_frac_event": pe_g,
                "neutral_baseline_ci90": f"{lo_ci:.3f}-{hi_ci:.3f}",
                "median_odds_ratio": float(odds_where_relevant.median()),
                "frac_cells_with_drought": float(occurs.mean()),
                "land_area_drought_1e6km2": R.exposed_area_km2(
                    wc["drought_months"] > base["drought_months"] + 1) / 1e6,
            })
            log.info("  %-12s drought months %.2f (neutral %.2f), "
                     "median odds ratio %.2f", klass,
                     rows[-1]["mean_drought_months_event"],
                     rows[-1]["mean_drought_months_neutral"],
                     rows[-1]["median_odds_ratio"])

        # super-minus-reference contrast: where does a super event add drought?
        if {"super", "reference"} <= set(results):
            extra = (results["super"]["drought_months"]
                     - results["reference"]["drought_months"])
            out = out_root / f"extra_drought_{variable}_{index_name}_{clim}_{soc}.nc"
            if not args.dry_run:
                IO.save_netcdf(extra.rename("extra_drought_months"), out,
                               "13_extremes", cfg, variable=variable)
            log.info("  super adds %.2f drought months on average vs "
                     "moderate/strong events", float(R.global_mean(extra)))

    if rows and not args.dry_run:
        df = pd.DataFrame(rows)
        save_table(df, cfg.path("tables", "table_6_extremes.csv"),
                   "13_extremes", cfg)
        log.info("\n%s", df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
