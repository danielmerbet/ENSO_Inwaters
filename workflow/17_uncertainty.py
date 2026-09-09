#!/usr/bin/env python3
"""Step 17 - where the uncertainty comes from.

Every number in the paper carries three separable sources of spread:
which impact model produced it, which of the four super events it came
from, and internal variability. A two-way ANOVA on the
(model x event) array of seasonal responses partitions the variance
between them at every grid cell.

The result matters for how the findings should be read. Where the model
term dominates, the signal is a modelling problem and the honest
statement is "models disagree". Where the event term dominates, super
El Ninos are genuinely diverse and no composite -- however significant --
describes the next one well. Only where the residual is small and both
terms are small is a composite a reliable predictor.

Also reported: the ensemble spread as a function of lag, and a
leave-one-model-out and leave-one-event-out sensitivity of the headline
numbers, which is the cheapest possible guard against a result that
rests on a single model or a single event.

Run:  python workflow/17_uncertainty.py
"""

from __future__ import annotations

import pandas as pd
import xarray as xr
from _common import skip_existing, step_setup

from enso_inwaters import isimip_io as IO
from enso_inwaters import regions as R
from enso_inwaters import stats as S
from enso_inwaters.isimip_io import save_table


def add_args(p):
    p.add_argument("--classes", default="super,reference")


def main() -> int:
    cfg, log, args = step_setup("17_uncertainty", __doc__, extra=add_args)
    comp_root = cfg.path("processed", "composites")
    out_root = cfg.path("processed", "uncertainty", mkdir=True)
    clim = cfg["isimip.main_scenario.climate_scenario"]
    soc = cfg["isimip.main_scenario.soc_scenario"]

    rows, loo_rows = [], []
    for klass in [c.strip() for c in args.classes.split(",")]:
        files = sorted(comp_root.glob(f"composite_*_{klass}_{clim}_{soc}.nc"))
        for f in files:
            variable = f.stem.split("_")[1]
            ds = xr.open_dataset(f)
            seas = ds["seasonal"]                       # (model, season, event, ...)
            if "model" not in seas.dims or "event" not in seas.dims:
                log.warning("%s: needs both a model and an event dimension", f.name)
                continue

            vp = S.variance_partition(seas, "model", "event")
            out = out_root / f"variance_partition_{variable}_{klass}.nc"
            if not skip_existing(out, args, log) and not args.dry_run:
                IO.save_netcdf(vp, out, "17_uncertainty", cfg,
                               variable=variable, event_class=klass)

            for season in seas["season"].values:
                v = vp.sel(season=season)
                rows.append({
                    "variable": variable, "event_class": klass,
                    "season": str(season),
                    "n_models": int(seas.sizes["model"]),
                    "n_events": int(seas.sizes["event"]),
                    "frac_model": float(R.global_mean(v["frac_model"])),
                    "frac_event": float(R.global_mean(v["frac_event"])),
                    "frac_residual": float(R.global_mean(v["frac_residual"])),
                })

            # leave-one-out sensitivity of the area-mean mature response
            if "DJF01" in [str(s) for s in seas["season"].values]:
                mature = seas.sel(season="DJF01")
                full = float(R.global_mean(mature.mean(["model", "event"])))
                for m in mature["model"].values:
                    v = float(R.global_mean(
                        mature.drop_sel(model=m).mean(["model", "event"])))
                    loo_rows.append({"variable": variable, "event_class": klass,
                                     "left_out": f"model:{m}", "full": full,
                                     "without": v, "change": v - full})
                for e in mature["event"].values:
                    v = float(R.global_mean(
                        mature.drop_sel(event=e).mean(["model", "event"])))
                    loo_rows.append({"variable": variable, "event_class": klass,
                                     "left_out": f"event:{e}", "full": full,
                                     "without": v, "change": v - full})
            log.info("%-12s %-10s variance partition done (%d models, %d events)",
                     variable, klass, seas.sizes["model"], seas.sizes["event"])

    if rows and not args.dry_run:
        df = pd.DataFrame(rows)
        save_table(df, cfg.path("tables", "table_10_variance_partition.csv"),
                   "17_uncertainty", cfg)
        log.info("\n%s", df.to_string(index=False))
    if loo_rows and not args.dry_run:
        ldf = pd.DataFrame(loo_rows)
        save_table(ldf, cfg.path("tables", "table_S9_leave_one_out.csv"),
                   "17_uncertainty", cfg)
        worst = ldf.reindex(ldf["change"].abs().sort_values(ascending=False).index).head(8)
        log.info("largest leave-one-out sensitivities:\n%s",
                 worst.to_string(index=False))
        big = ldf[ldf["change"].abs() > 0.5 * ldf["full"].abs()]
        if not big.empty:
            log.warning("%d headline number(s) change by more than half when a "
                        "single model or event is removed - report this.", len(big))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
