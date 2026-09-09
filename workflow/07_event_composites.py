#!/usr/bin/env python3
"""Step 07 - superposed-epoch composites of the hydrological response.

For each variable and each ENSO amplitude class this builds

* ``composite``  (model, lag, lat, lon)          -- the event-mean anomaly
  from 18 months before to 30 months after the event peak; and
* ``seasonal``   (model, event, season, lat, lon) -- per-event seasonal
  means (JJA0 ... DJF12), which steps 08, 09 and 17 need for the
  significance test, the nonlinearity test and the variance partition.

Epochs are anchored on **December of the developing year** rather than
on each event's own peak month, so that the ``lag`` axis keeps a fixed
seasonal meaning and the season codes (JJA0, DJF01, MAM1, ...) are exact
for every event. Anchoring on the observed peak is available with
``--anchor peak`` as a sensitivity test.

Run:  python workflow/07_event_composites.py [--variable dis]
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr
from _common import composite_path, skip_existing, step_setup

from enso_inwaters import enso as E
from enso_inwaters import isimip_io as IO


def add_args(p):
    p.add_argument("--variable", default=None)
    p.add_argument("--sector", default=None)
    p.add_argument("--classes", default="super,strong,moderate,reference,all_el_nino,all_la_nina",
                   help="comma-separated event classes to composite")
    p.add_argument("--anchor", default="december", choices=["december", "peak"])
    p.add_argument("--climate-scenario", default=None)
    p.add_argument("--soc-scenario", default=None)


def anomaly_files(root: Path, sector: str | None, variable: str | None,
                  clim: str | None, soc: str | None) -> list[Path]:
    files = sorted(root.rglob("*_anom.nc"))
    out = []
    for f in files:
        parts = f.stem[:-len("_anom")].split("_")
        if len(parts) < 4:
            continue
        var, cs, ss = parts[1], parts[2], parts[3]
        if sector and f.parent.name != sector:
            continue
        if variable and var != variable:
            continue
        if clim and cs != clim:
            continue
        if soc and ss != soc:
            continue
        out.append(f)
    return out


def main() -> int:
    cfg, log, args = step_setup("07_event_composites", __doc__, extra=add_args)
    anom_root = cfg.path("interim", "anomalies")
    events = pd.read_csv(cfg.path("processed", "enso", "enso_events.csv"))
    lag_min, lag_max = int(cfg["composite.lag_min"]), int(cfg["composite.lag_max"])
    seasons = [s for group in cfg["composite.key_seasons"].values() for s in group]
    log.info("seasons reported: %s", seasons)

    clim = args.climate_scenario or cfg["isimip.main_scenario.climate_scenario"]
    soc = args.soc_scenario or cfg["isimip.main_scenario.soc_scenario"]
    files = anomaly_files(anom_root, args.sector, args.variable, clim, soc)
    if not files:
        log.error("no anomaly files matching %s/%s under %s - run step 06",
                  clim, soc, anom_root)
        return 1

    # group by variable so that all models of one variable end up in one file
    by_var: dict[str, list[Path]] = {}
    for f in files:
        by_var.setdefault(f.stem.split("_")[1], []).append(f)
    log.info("variables: %s", sorted(by_var))

    n_written = 0
    for variable, vfiles in sorted(by_var.items()):
        for klass in args.classes.split(","):
            klass = klass.strip()
            sel = E.select_event_class(events, klass)
            if sel.empty:
                log.warning("no events in class %r - skipped", klass)
                continue
            out = composite_path(cfg, variable, klass, clim, soc)
            if skip_existing(out, args, log):
                continue

            anchors = [E.event_reference({"year0": int(r.year0),
                                          "peak_time": r.peak_time},
                                         anchor=args.anchor)
                       for r in sel.itertuples()]
            log.info("%-12s %-14s %d event(s): %s", variable, klass, len(sel),
                     ", ".join(sel["label"].astype(str)))

            comps, seas, models = [], [], []
            for f in vfiles:
                model = f.stem.split("_")[0]
                try:
                    da = xr.open_dataset(f)[variable]
                except KeyError:
                    ds = xr.open_dataset(f)
                    da = ds[next(iter(ds.data_vars))]
                try:
                    epochs = E.superposed_epoch(da, anchors, lag_min, lag_max,
                                                keep_events=True)
                except ValueError as exc:
                    log.warning("  %s skipped: %s", model, exc)
                    continue
                epochs = epochs.assign_coords(
                    event=("event", [str(lab) for lab in sel["label"]]
                           [:epochs.sizes["event"]]))
                comps.append(epochs.mean("event", skipna=True))
                seas.append(xr.concat(
                    [E.season_mean(epochs, s).assign_coords(season=s)
                     for s in seasons], dim="season"))
                models.append(model)
                log.info("  %-16s epochs=%d  lag range %d..%d", model,
                         epochs.sizes["event"], lag_min, lag_max)

            if not comps:
                log.warning("  no usable model for %s/%s", variable, klass)
                continue

            ds_out = xr.Dataset({
                "composite": xr.concat(comps, dim=pd.Index(models, name="model")),
                "seasonal": xr.concat(seas, dim=pd.Index(models, name="model")),
            })
            ds_out["composite"].attrs.update({
                "long_name": f"{klass} composite anomaly of {variable}",
                "units": "sigma", "n_events": len(sel),
                "events": ";".join(sel["label"].astype(str)),
                "epoch_anchor": args.anchor})
            ds_out.attrs.update({
                "event_class": klass, "variable": variable,
                "n_events": len(sel),
                "mean_peak_oni": float(sel["peak_oni"].mean()),
                "climate_scenario": clim, "soc_scenario": soc})
            if args.dry_run:
                log.info("  would write %s", out.name)
                continue
            IO.save_netcdf(ds_out, out, "07_event_composites", cfg,
                           event_class=klass, variable=variable)
            log.info("  wrote %s", out.name)
            n_written += 1

    log.info("wrote %d composite file(s)", n_written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
