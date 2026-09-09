#!/usr/bin/env python3
"""Step 10 - the propagation of the signal from rain to groundwater.

An El Nino precipitation anomaly does not reach the water table the
month it falls. It is routed through the soil, the river network and the
aquifer, each of which adds delay and memory. This step quantifies that
cascade from the composites:

* **lag of peak response** -- the lag (months from the event peak) at
  which |composite| is largest, per grid cell and per variable;
* **response duration** -- the number of consecutive months around that
  peak for which the composite stays beyond 0.5 sigma;
* **recovery time** -- months from the peak until the composite falls
  back inside +-0.5 sigma, i.e. how long after a super El Nino the
  system is still perturbed;
* **memory timescale** -- the AR(1) e-folding time of the anomalies
  themselves, which sets the ceiling on how long a perturbation *can*
  persist in each model;
* **ONI cross-correlation** -- lag of maximum correlation between the
  ONI and the area-averaged anomaly, as an independent estimate that
  does not depend on the event sample.

The expected ordering -- runoff, then discharge, then recharge, then
groundwater storage -- is the paper's mechanistic backbone, and the
numbers here are what supports it.

Run:  python workflow/10_lag_cascade.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from _common import skip_existing, step_setup

from enso_inwaters import isimip_io as IO
from enso_inwaters.isimip_io import save_table
from enso_inwaters.regions import global_mean

THRESHOLD = 0.5     # sigma; "still perturbed" threshold


def add_args(p):
    p.add_argument("--event-class", default="super")
    p.add_argument("--variables", default=None,
                   help="comma-separated subset (default: all available)")


def peak_lag(comp: xr.DataArray) -> xr.Dataset:
    """Lag, sign and magnitude of the strongest composite response."""
    absc = np.abs(comp)
    k = absc.fillna(-1).argmax("lag")
    lag = comp["lag"].values[k.values]
    amp = comp.isel(lag=k)
    lag_da = xr.DataArray(lag, dims=k.dims,
                          coords={d: k[d] for d in k.dims if d in k.coords})
    valid = comp.notnull().any("lag")
    return xr.Dataset({
        "peak_lag": lag_da.where(valid),
        "peak_amplitude": amp.where(valid),
    })


def duration_and_recovery(comp: xr.DataArray, threshold: float = THRESHOLD
                          ) -> xr.Dataset:
    """Months beyond threshold, and months until the anomaly subsides."""
    strong = np.abs(comp) > threshold
    duration = strong.sum("lag")

    after = comp.sel(lag=slice(0, None))
    strong_after = (np.abs(after) > threshold).values      # (lag, ...)
    n_lag = strong_after.shape[0]
    # first lag >= 0 at which the anomaly is back inside the threshold
    # *and stays there*: scan from the end backwards
    still = np.zeros(strong_after.shape[1:], dtype=int)
    for i in range(n_lag - 1, -1, -1):
        still = np.where(strong_after[i], i + 1, still)
    rec = xr.DataArray(still.astype("float64"), dims=after.dims[1:],
                       coords={d: after[d] for d in after.dims[1:]
                               if d in after.coords})
    valid = comp.notnull().any("lag")
    return xr.Dataset({"duration_months": duration.where(valid),
                       "recovery_months": rec.where(valid)})


def ar1_memory(anom: xr.DataArray) -> xr.DataArray:
    """AR(1) e-folding memory in months, from the lag-1 autocorrelation."""
    x = anom - anom.mean("time", skipna=True)
    num = (x * x.shift(time=1)).mean("time", skipna=True)
    den = (x * x).mean("time", skipna=True)
    r1 = (num / den.where(den > 0)).clip(-0.99, 0.99)
    tau = -1.0 / np.log(r1.where(r1 > 0))
    tau.attrs.update({"units": "months", "long_name": "AR(1) e-folding memory"})
    return tau


def cross_correlation(series: xr.DataArray, oni: pd.Series,
                      max_lag: int = 24) -> pd.DataFrame:
    """Correlation of an area-mean anomaly with the ONI, lag by lag."""
    s = pd.Series(series.values, index=pd.DatetimeIndex(series.time.values))
    joined = pd.concat([s.rename("y"), oni.rename("oni")], axis=1).dropna()
    rows = []
    for lag in range(-6, max_lag + 1):
        r = joined["y"].corr(joined["oni"].shift(lag))
        rows.append({"lag": lag, "correlation": r})
    return pd.DataFrame(rows)


def main() -> int:
    cfg, log, args = step_setup("10_lag_cascade", __doc__, extra=add_args)
    comp_root = cfg.path("processed", "composites")
    anom_root = cfg.path("interim", "anomalies")
    out_root = cfg.path("processed", "lag_cascade", mkdir=True)
    clim = cfg["isimip.main_scenario.climate_scenario"]
    soc = cfg["isimip.main_scenario.soc_scenario"]
    klass = args.event_class

    oni = pd.read_csv(cfg.path("processed", "enso", "oni.csv"),
                      parse_dates=["time"]).set_index("time")["oni"]

    files = sorted(comp_root.glob(f"composite_*_{klass}_{clim}_{soc}.nc"))
    if not files:
        log.error("no %s composites - run step 07 first", klass)
        return 1
    variables = [f.stem.split("_")[1] for f in files]
    if args.variables:
        wanted = [v.strip() for v in args.variables.split(",")]
        files = [f for f, v in zip(files, variables, strict=True) if v in wanted]
        variables = [v for v in variables if v in wanted]

    rows, xcorr_rows = [], []
    for f, variable in zip(files, variables, strict=True):
        out = out_root / f"lag_cascade_{variable}_{klass}_{clim}_{soc}.nc"
        ds = xr.open_dataset(f)
        comp = ds["composite"].mean("model", skipna=True)

        pk = peak_lag(comp)
        dr = duration_and_recovery(comp)

        # area-mean profile and its cross-correlation with the ONI
        area_mean = global_mean(comp) if {"lat", "lon"} <= set(comp.dims) else comp
        profile = area_mean.values
        lag_axis = comp["lag"].values
        k = int(np.nanargmax(np.abs(profile)))

        # AR(1) memory from the underlying anomalies (ensemble mean)
        afiles = sorted(anom_root.rglob(f"*_{variable}_{clim}_{soc}_anom.nc"))
        tau_med = np.nan
        if afiles:
            arrays = {}
            for af in afiles:
                d = xr.open_dataset(af)
                arrays[af.stem.split("_")[0]] = d[variable if variable in d
                                                  else next(iter(d.data_vars))]
            ens = IO.concat_models(arrays).mean("model", skipna=True)
            tau = ar1_memory(ens)
            tau_med = float(tau.median())
            xc = cross_correlation(global_mean(ens) if {"lat", "lon"} <= set(ens.dims)
                                   else ens, oni)
            xc["variable"] = variable
            xcorr_rows.append(xc)
            best = xc.iloc[xc["correlation"].abs().idxmax()]
            log.info("%-12s peak composite at lag %+3d (%.2f sigma) | "
                     "ONI xcorr peak at lag %+3d (r=%.2f) | AR(1) memory %.1f mo",
                     variable, lag_axis[k], profile[k], int(best["lag"]),
                     best["correlation"], tau_med)
        else:
            tau = None
            best = {"lag": np.nan, "correlation": np.nan}
            log.info("%-12s peak composite at lag %+3d (%.2f sigma)",
                     variable, lag_axis[k], profile[k])

        ds_out = xr.merge([pk, dr])
        if tau is not None:
            ds_out["ar1_memory_months"] = tau
        ds_out.attrs.update({"variable": variable, "event_class": klass})
        if not skip_existing(out, args, log) and not args.dry_run:
            IO.save_netcdf(ds_out, out, "10_lag_cascade", cfg)

        rows.append({
            "variable": variable,
            "peak_lag_area_mean": int(lag_axis[k]),
            "peak_amplitude_area_mean": float(profile[k]),
            "median_peak_lag_grid": float(pk["peak_lag"].median()),
            "median_duration_months": float(dr["duration_months"].median()),
            "median_recovery_months": float(dr["recovery_months"].median()),
            "p90_recovery_months": float(dr["recovery_months"].quantile(0.9)),
            "median_ar1_memory_months": tau_med,
            "oni_xcorr_peak_lag": float(best["lag"]),
            "oni_xcorr_peak_r": float(best["correlation"]),
        })

    if rows and not args.dry_run:
        df = pd.DataFrame(rows).sort_values("peak_lag_area_mean")
        save_table(df, cfg.path("tables", "table_3_lag_cascade.csv"),
                   "10_lag_cascade", cfg)
        if xcorr_rows:
            save_table(pd.concat(xcorr_rows, ignore_index=True),
                       cfg.path("processed", "lag_cascade",
                                "oni_cross_correlation.csv"),
                       "10_lag_cascade", cfg)
        log.info("\n%s", df.to_string(index=False))
        order = " -> ".join(df["variable"])
        log.info("response ordering (earliest to latest peak): %s", order)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
