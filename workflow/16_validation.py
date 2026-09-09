#!/usr/bin/env python3
"""Step 16 - do the models get the observed ENSO response right?

A multi-model composite is only worth reporting if the ensemble can
reproduce the ENSO signal where observations exist. Three independent
evaluations, each covering a different part of the water cycle:

* **GRDC station discharge** -- composite anomalies at gauges with >= 40
  years of record, compared with the simulated composite at the nearest
  land cell. Reported as the correlation of the composite patterns
  across stations, the bias in composite amplitude, and the fraction of
  stations where the model and the observation agree on sign.
* **GRACE/GRACE-FO total water storage** -- covers 2002-2019, hence the
  2015/16 super event. This is the only direct observational constraint
  on the simulated storage response, and it is the variable where the
  models differ most.
* **Satellite lake surface water temperature** (ESA CCI Lakes) -- for
  the lake sector.

Where an observational file is absent the corresponding evaluation is
skipped with a warning rather than silently omitted, so the manuscript
can state exactly which claims are observationally supported.

Run:  python workflow/16_validation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from _common import step_setup

from enso_inwaters import climatology as C
from enso_inwaters import enso as E
from enso_inwaters import isimip_io as IO
from enso_inwaters.isimip_io import normalise_coords, save_table


def add_args(p):
    p.add_argument("--classes", default="super,all_el_nino")


def composite_of(da: xr.DataArray, years, lag_min: int, lag_max: int
                 ) -> xr.DataArray:
    anchors = [pd.Timestamp(year=int(y), month=12, day=1) for y in years]
    return E.superposed_epoch(da, anchors, lag_min, lag_max)


def evaluate_grdc(cfg, log, events, klass, lag_min, lag_max) -> pd.DataFrame:
    path = Path(cfg.root) / cfg["auxiliary.observations.grdc_discharge.file"]
    if not path.exists():
        log.warning("GRDC file %s not found - station evaluation skipped", path)
        return pd.DataFrame()
    obs = xr.open_dataset(path)
    qname = next((v for v in ("discharge", "dis", "runoff", "Q") if v in obs), None)
    if qname is None:
        log.warning("no discharge variable in %s (found %s)", path.name,
                    list(obs.data_vars))
        return pd.DataFrame()
    q = IO.normalise_time(obs[qname])
    min_years = int(cfg["auxiliary.observations.grdc_discharge.min_years"])
    max_missing = float(cfg["auxiliary.observations.grdc_discharge.max_missing_frac"])
    keep = ((q.notnull().sum("time") >= min_years * 12) &
            (q.isnull().mean("time") <= max_missing))
    q = q.where(keep, drop=True)
    log.info("GRDC: %d station(s) pass the record-length filter",
             int(q.sizes.get("station", 0)))

    q_anom = C.anomalies(q, method="standardized", climatology="moving_30yr")
    years = E.select_event_class(events, klass)["year0"].astype(int)
    obs_comp = composite_of(q_anom, years, lag_min, lag_max)

    sim_path = cfg.path("processed", "composites") / (
        f"composite_dis_{klass}_{cfg['isimip.main_scenario.climate_scenario']}_"
        f"{cfg['isimip.main_scenario.soc_scenario']}.nc")
    if not sim_path.exists():
        log.warning("no simulated discharge composite to compare against")
        return pd.DataFrame()
    sim = xr.open_dataset(sim_path)["composite"].mean("model", skipna=True)

    rows = []
    for i in range(obs_comp.sizes.get("station", 0)):
        st = obs_comp.isel(station=i)
        lat = float(st["lat"]) if "lat" in st.coords else np.nan
        lon = float(st["lon"]) if "lon" in st.coords else np.nan
        if not np.isfinite(lat) or not np.isfinite(lon):
            continue
        sm = sim.sel(lat=lat, lon=lon, method="nearest")
        both = np.isfinite(st.values) & np.isfinite(sm.values)
        if both.sum() < 6:
            continue
        r = float(np.corrcoef(st.values[both], sm.values[both])[0, 1])
        rows.append({
            "station": str(st["station"].values) if "station" in st.coords else i,
            "lat": lat, "lon": lon, "lag_correlation": r,
            "obs_peak": float(st.values[both][np.argmax(np.abs(st.values[both]))]),
            "sim_peak": float(sm.values[both][np.argmax(np.abs(sm.values[both]))]),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["sign_agrees"] = np.sign(df["obs_peak"]) == np.sign(df["sim_peak"])
        log.info("GRDC evaluation: median lag-profile correlation %.2f, "
                 "sign agreement at %.0f%% of %d stations",
                 df["lag_correlation"].median(),
                 100 * df["sign_agrees"].mean(), len(df))
    return df


def evaluate_gridded(cfg, log, events, klass, lag_min, lag_max,
                     obs_key: str, sim_variable: str, label: str) -> dict:
    path = Path(cfg.root) / cfg[f"auxiliary.observations.{obs_key}.file"]
    if not path.exists():
        log.warning("%s file %s not found - %s evaluation skipped",
                    label, path, label)
        return {}
    ds = xr.open_dataset(path)
    obs = normalise_coords(IO.normalise_time(ds[next(iter(ds.data_vars))]))
    obs_anom = C.anomalies(obs, method="standardized", climatology="fixed",
                           base=(int(pd.DatetimeIndex(obs.time.values).year.min()),
                                 int(pd.DatetimeIndex(obs.time.values).year.max())))
    years = [y for y in E.select_event_class(events, klass)["year0"].astype(int)
             if y >= pd.DatetimeIndex(obs.time.values).year.min()]
    if not years:
        log.warning("%s record does not overlap any %s event", label, klass)
        return {}
    obs_comp = composite_of(obs_anom, years, lag_min, lag_max)

    sim_path = cfg.path("processed", "composites") / (
        f"composite_{sim_variable}_{klass}_"
        f"{cfg['isimip.main_scenario.climate_scenario']}_"
        f"{cfg['isimip.main_scenario.soc_scenario']}.nc")
    if not sim_path.exists():
        log.warning("no simulated %s composite", sim_variable)
        return {}
    sim = xr.open_dataset(sim_path)["composite"].mean("model", skipna=True)
    sim_i = sim.interp(lat=obs_comp["lat"], lon=obs_comp["lon"])

    a = obs_comp.values.ravel()
    b = sim_i.values.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100:
        log.warning("%s: too little overlap for a pattern correlation", label)
        return {}
    r = float(np.corrcoef(a[ok], b[ok])[0, 1])
    out = {"evaluation": label, "event_class": klass, "n_events": len(years),
           "years": ";".join(map(str, years)), "pattern_correlation": r,
           "sim_over_obs_amplitude": float(np.nanstd(b[ok]) / np.nanstd(a[ok])),
           "sign_agreement": float(np.mean(np.sign(a[ok]) == np.sign(b[ok])))}
    log.info("%s: pattern r = %.2f, amplitude ratio %.2f, sign agreement %.0f%%",
             label, r, out["sim_over_obs_amplitude"], 100 * out["sign_agreement"])
    return out


def main() -> int:
    cfg, log, args = step_setup("16_validation", __doc__, extra=add_args)
    events = pd.read_csv(cfg.path("processed", "enso", "enso_events.csv"))
    lag_min, lag_max = int(cfg["composite.lag_min"]), int(cfg["composite.lag_max"])
    summary = []

    for klass in [c.strip() for c in args.classes.split(",")]:
        df = evaluate_grdc(cfg, log, events, klass, lag_min, lag_max)
        if not df.empty and not args.dry_run:
            save_table(df, cfg.path("tables",
                                    f"table_S8_grdc_evaluation_{klass}.csv"),
                       "16_validation", cfg)
            summary.append({
                "evaluation": "GRDC discharge", "event_class": klass,
                "n_stations": len(df),
                "median_lag_correlation": float(df["lag_correlation"].median()),
                "sign_agreement": float(df["sign_agrees"].mean())})
        for obs_key, sim_var, label in (
                ("grace_tws", "tws", "GRACE TWS"),
                ("lswt_satellite", "surftemp", "satellite LSWT"),
                ("era5_land", "qtot", "ERA5-Land runoff")):
            res = evaluate_gridded(cfg, log, events, klass, lag_min, lag_max,
                                   obs_key, sim_var, label)
            if res:
                summary.append(res)

    if summary and not args.dry_run:
        save_table(pd.DataFrame(summary),
                   cfg.path("tables", "table_9_model_evaluation.csv"),
                   "16_validation", cfg)
        log.info("\n%s", pd.DataFrame(summary).to_string(index=False))
    if not summary:
        log.warning("no observational dataset was available. The manuscript "
                    "must then state that the ensemble is unevaluated, or "
                    "step 04's manual downloads must be completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
