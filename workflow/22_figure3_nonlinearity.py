#!/usr/bin/env python3
"""Step 22 - Figure 3: is a super El Nino more than a scaled-up strong one?

(a, b) Maps of the nonlinear excess -- the super composite minus the
       moderate/strong composite scaled by the ratio of peak ONI -- for
       discharge and total water storage in the mature season. Positive
       (green) means the super event does more than proportionally.
(c)    Map of the hinge coefficient ratio h/b1: the extra sensitivity of
       the response beyond ONI = +1.5, relative to the sensitivity below
       it.
(d)    Response-versus-amplitude scatter for a set of hotspot regions,
       with the fitted piecewise-linear curve, which is the clearest
       single picture of the paper's central claim.

Run:  python workflow/22_figure3_nonlinearity.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from _common import figure_path, step_setup

from enso_inwaters import isimip_io as IO
from enso_inwaters import plotting as P
from enso_inwaters import regions as R

# Regions where ENSO teleconnections to inland waters are well documented.
HOTSPOTS = {
    "maritime continent": (-10, 10, 95, 140),
    "eastern Australia": (-38, -15, 138, 154),
    "southern Africa": (-30, -15, 15, 35),
    "Amazon": (-15, 5, -70, -50),
    "southern USA / N Mexico": (25, 37, -115, -95),
    "La Plata": (-35, -20, -65, -50),
}


def add_args(p):
    p.add_argument("--variables", default="dis,tws")
    p.add_argument("--season", default="DJF01")


def main() -> int:
    cfg, log, args = step_setup("22_figure3_nonlinearity", __doc__, extra=add_args)
    P.set_style(int(cfg["figures.font_size"]))
    import matplotlib.pyplot as plt

    nl_root = cfg.path("processed", "nonlinearity")
    clim = cfg["isimip.main_scenario.climate_scenario"]
    soc = cfg["isimip.main_scenario.soc_scenario"]
    variables = [v.strip() for v in args.variables.split(",")]
    season = args.season

    files = {v: nl_root / f"nonlinearity_{v}_{clim}_{soc}.nc" for v in variables}
    files = {v: f for v, f in files.items() if f.exists()}
    if not files:
        log.error("no nonlinearity results - run step 09 first")
        return 1

    fig = plt.figure(figsize=(183 * P.MM, 165 * P.MM))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.25], hspace=0.40,
                          wspace=0.16)
    panel = iter("abcdef")

    scale = None
    for i, (variable, f) in enumerate(list(files.items())[:2]):
        ds = xr.open_dataset(f)
        scale = ds.attrs.get("scaling_factor", scale)
        ax = fig.add_subplot(gs[0, i],
                             projection=P.ccrs.Robinson() if P.HAS_CARTOPY else None)
        ex = ds["excess"].sel(season=season)
        sig = (ds["excess_significant"].sel(season=season).astype(bool)
               if "excess_significant" in ds else None)
        P.plot_map(ex, ax=ax, vmin=-0.6, vmax=0.6,
                   cmap=cfg["figures.colormaps.anomaly"], stipple=sig,
                   add_colorbar=True, projection=cfg["figures.projection"],
                   cbar_label="nonlinear excess (sigma)")
        ax.set_title(f"{variable}: super - {scale:.2f} x (moderate+strong)"
                     if scale else variable, fontsize=7, loc="left", pad=6)
        P.panel_label(ax, next(panel), x=-0.02, y=1.0)

    # --- hinge ratio map ------------------------------------------------
    first = list(files)[0]
    ds = xr.open_dataset(files[first])
    if "steepening_ratio" in ds:
        ax = fig.add_subplot(gs[1, 0],
                             projection=P.ccrs.Robinson() if P.HAS_CARTOPY else None)
        ratio = ds["steepening_ratio"].sel(season=season).clip(-1.5, 1.5)
        strong = np.abs(ds["b1_linear"].sel(season=season)) > 0.2
        P.plot_map(ratio.where(strong), ax=ax, vmin=-1, vmax=1, cmap="PuOr_r",
                   add_colorbar=True, projection=cfg["figures.projection"],
                   cbar_label="extra sensitivity beyond ONI +1.5 (h / b1)")
        ax.set_title(f"{first}: amplitude steepening", fontsize=7,
                     loc="left", pad=6)
        P.panel_label(ax, next(panel), x=-0.02, y=1.0)

        # --- histogram of the steepening ratio --------------------------
        ax = fig.add_subplot(gs[1, 1])
        vals = ratio.where(strong).values.ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.hist(vals, bins=40, color="#4292c6", edgecolor="none")
            ax.axvline(0, color="0.3", lw=0.7)
            ax.axvline(np.median(vals), color="#b2182b", lw=1.0,
                       label=f"median {np.median(vals):+.2f}")
            ax.set_xlabel("h / b1 (extra sensitivity beyond ONI +1.5)")
            ax.set_ylabel("grid cells")
            ax.legend(fontsize=6, frameon=False)
            frac = float((vals > 0).mean())
            ax.set_title(f"{100*frac:.0f}% of cells steepen with amplitude",
                         fontsize=7, loc="left")
        P.panel_label(ax, next(panel), x=-0.10)

    # --- response vs amplitude for hotspot regions ----------------------
    ax = fig.add_subplot(gs[2, :])
    oni = pd.read_csv(cfg.path("processed", "enso", "oni.csv"),
                      parse_dates=["time"]).set_index("time")["oni"]
    anom_root = cfg.path("interim", "anomalies")
    afiles = sorted(anom_root.rglob(f"*_{first}_{clim}_{soc}_anom.nc"))
    if afiles:
        arrays = {}
        for af in afiles:
            d = xr.open_dataset(af)
            arrays[af.stem.split("_")[0]] = d[first if first in d
                                              else next(iter(d.data_vars))]
        ens = IO.concat_models(arrays).mean("model", skipna=True)
        colours = plt.cm.tab10(np.linspace(0, 1, 10))
        for c, (name, (la0, la1, lo0, lo1)) in zip(colours, HOTSPOTS.items(), strict=False):
            box = ens.sel(lat=slice(la0, la1), lon=slice(lo0, lo1))
            if box.sizes.get("lat", 0) == 0 or box.sizes.get("lon", 0) == 0:
                continue
            series = R.global_mean(box)
            djf = pd.Series(series.values,
                            index=pd.DatetimeIndex(series.time.values))
            djf = djf.rolling(3, center=True).mean()
            years = sorted({t.year for t in djf.index})[1:-1]
            xs, ys = [], []
            for y in years:
                t = pd.Timestamp(year=y, month=1, day=1)
                if t in djf.index and t in oni.index:
                    xs.append(oni[t])
                    ys.append(djf[t])
            xs, ys = np.array(xs), np.array(ys)
            ok = np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[ok], ys[ok]          # the moving climatology leaves
            if xs.size < 20:                 # NaNs at both ends of the record
                continue
            ax.scatter(xs, ys, s=5, color=c, alpha=0.45, lw=0)
            # piecewise-linear fit with knots at +-1.5
            X = np.column_stack([np.ones_like(xs), xs,
                                 np.maximum(xs - 1.5, 0),
                                 np.minimum(xs + 1.5, 0)])
            beta = np.linalg.lstsq(X, ys, rcond=None)[0]
            grid = np.linspace(xs.min(), xs.max(), 100)
            G = np.column_stack([np.ones_like(grid), grid,
                                 np.maximum(grid - 1.5, 0),
                                 np.minimum(grid + 1.5, 0)])
            ax.plot(grid, G @ beta, color=c, lw=1.3,
                    label=f"{name} (h/b1 = {beta[2]/beta[1]:+.2f})"
                    if abs(beta[1]) > 1e-6 else name)
        ax.axvline(2.0, color="0.3", lw=0.6, ls="--")
        ax.text(2.03, ax.get_ylim()[1] * 0.9, "super", fontsize=6, color="0.3")
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xlabel("DJF ONI (degC)")
        ax.set_ylabel(f"{first} anomaly (sigma)")
        ax.legend(fontsize=5.5, frameon=False, ncol=3, loc="upper left")
        P.panel_label(ax, next(panel), x=-0.045)
        ax.set_title("Regional response vs. event amplitude - a single "
                     "straight line would mean super events are simply "
                     "scaled-up moderate ones", fontsize=6.5, loc="left",
                     color="0.3", pad=4)

    out = figure_path(cfg, "fig03_nonlinearity")
    paths = P.save_figure(fig, out, formats=cfg["figures.format"],
                          dpi=int(cfg["figures.dpi"]))
    log.info("wrote %s", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
