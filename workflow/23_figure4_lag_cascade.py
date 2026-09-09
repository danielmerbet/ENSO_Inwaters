#!/usr/bin/env python3
"""Step 23 - Figure 4: the cascade from rainfall to groundwater.

(a) Composite lag profiles, area-averaged over the regions where the
    response is robust, one line per variable: runoff, discharge,
    recharge, soil moisture, total water storage, groundwater storage.
    The successive delay of the peaks is the mechanistic core of the
    paper.
(b) The same for the moderate/strong reference sample, on identical
    axes, so amplitude and persistence can be compared directly.
(c) Map of the lag at which total water storage peaks -- where the
    system is slow and where it is fast.
(d) Recovery time: months for which the composite anomaly stays beyond
    0.5 sigma after the event peak, by variable, as a distribution over
    grid cells.

Run:  python workflow/23_figure4_lag_cascade.py
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from _common import composite_path, figure_path, step_setup

from enso_inwaters import plotting as P
from enso_inwaters import regions as R

ORDER = ["qtot", "dis", "rootmoist", "qr", "tws", "groundwstor"]
LABELS = {"qtot": "total runoff", "dis": "river discharge",
          "rootmoist": "root-zone soil moisture", "qr": "groundwater recharge",
          "tws": "total water storage", "groundwstor": "groundwater storage"}


def add_args(p):
    p.add_argument("--map-variable", default="tws")


def profile(ds: xr.Dataset, mask: xr.DataArray | None = None) -> xr.DataArray:
    """Area-weighted **magnitude** of the composite response, lag by lag.

    The teleconnection is a dipole -- wet in the southern US, dry over
    the maritime continent -- so a signed area mean cancels itself out
    and says nothing. Averaging |anomaly| over the response region
    measures how strongly the land surface is perturbed, whichever way.
    """
    comp = ds["composite"].mean("model", skipna=True)
    if mask is not None:
        comp = comp.where(mask)
    return R.global_mean(np.abs(comp))


def main() -> int:
    cfg, log, args = step_setup("23_figure4_lag_cascade", __doc__, extra=add_args)
    P.set_style(int(cfg["figures.font_size"]))
    import matplotlib.pyplot as plt

    clim = cfg["isimip.main_scenario.climate_scenario"]
    soc = cfg["isimip.main_scenario.soc_scenario"]
    sig_root = cfg.path("processed", "significance")
    lag_root = cfg.path("processed", "lag_cascade")

    fig = plt.figure(figsize=(183 * P.MM, 135 * P.MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.60, wspace=0.45)
    colours = plt.cm.viridis(np.linspace(0.05, 0.9, len(ORDER)))

    peaks = {}
    for panel_i, (klass, title) in enumerate(
            (("super", "super El Nino"), ("reference", "moderate + strong El Nino"))):
        ax = fig.add_subplot(gs[0, panel_i])
        for colour, variable in zip(colours, ORDER, strict=True):
            path = composite_path(cfg, variable, klass, clim, soc)
            if not path.exists():
                continue
            ds = xr.open_dataset(path)
            # restrict to where the super composite is robust, so the
            # profile is not diluted by the two-thirds of the land with
            # no teleconnection at all
            sig = sig_root / f"significance_{variable}_super_{clim}_{soc}.nc"
            mask = None
            if sig.exists():
                sds = xr.open_dataset(sig)
                if "season" in sds["robust"].dims:
                    mask = sds["robust"].any("season").astype(bool)
                else:
                    mask = sds["robust"].any("lag").astype(bool)
            prof = profile(ds, mask)
            ax.plot(prof["lag"], prof.values, color=colour, lw=1.3,
                    label=LABELS.get(variable, variable))
            k = int(np.nanargmax(np.abs(prof.values)))
            ax.plot(prof["lag"].values[k], prof.values[k], "o", ms=3,
                    color=colour)
            if klass == "super":
                peaks[variable] = int(prof["lag"].values[k])
        ax.axvline(0, color="0.4", lw=0.5, ls=":")
        ax.set_ylim(bottom=0)
        ax.axvspan(-6, -1, color="0.93", zorder=0)
        ax.axvspan(0, 2, color="0.86", zorder=0)
        ax.set_xlabel("lag from December of the developing year (months)")
        ax.set_ylabel("response magnitude over the\nresponse region (sigma)")
        ax.set_title(title, fontsize=7, loc="left")
        if panel_i == 0:
            ax.legend(fontsize=5.5, frameon=False, ncol=2)
        P.panel_label(ax, "ab"[panel_i], x=-0.12)

    if peaks:
        log.info("peak lag by variable (super): %s",
                 ", ".join(f"{k}={v:+d}" for k, v in
                           sorted(peaks.items(), key=lambda kv: kv[1])))

    # --- (c) map of the peak lag ---------------------------------------
    ax = fig.add_subplot(gs[1, 0],
                         projection=P.ccrs.Robinson() if P.HAS_CARTOPY else None)
    lc = lag_root / f"lag_cascade_{args.map_variable}_super_{clim}_{soc}.nc"
    if lc.exists():
        ds = xr.open_dataset(lc)
        sig = sig_root / f"significance_{args.map_variable}_super_{clim}_{soc}.nc"
        field = ds["peak_lag"]
        if sig.exists():
            sds = xr.open_dataset(sig)
            dim = "season" if "season" in sds["robust"].dims else "lag"
            field = field.where(sds["robust"].any(dim).astype(bool))
        P.plot_map(field, ax=ax, vmin=-6, vmax=18, cmap="magma_r",
                   add_colorbar=True, projection=cfg["figures.projection"],
                   cbar_label="lag of the peak response (months)")
        ax.set_title(f"{LABELS.get(args.map_variable, args.map_variable)}: "
                     f"when the anomaly peaks", fontsize=7, loc="left", pad=6)
    else:
        ax.axis("off")
        log.warning("no lag-cascade file for %s - panel c empty", args.map_variable)
    P.panel_label(ax, "c", x=-0.02, y=1.0)

    # --- (d) recovery time distribution --------------------------------
    ax = fig.add_subplot(gs[1, 1])
    data, labels, cols = [], [], []
    for colour, variable in zip(colours, ORDER, strict=True):
        lc = lag_root / f"lag_cascade_{variable}_super_{clim}_{soc}.nc"
        if not lc.exists():
            continue
        ds = xr.open_dataset(lc)
        vals = ds["recovery_months"].values.ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            data.append(vals)
            labels.append(LABELS.get(variable, variable))
            cols.append(colour)
    if data:
        bp = ax.boxplot(data, orientation="horizontal", widths=0.6,
                        showfliers=False, patch_artist=True,
                        medianprops={"color": "k", "lw": 0.8}) \
            if "orientation" in ax.boxplot.__doc__ else \
            ax.boxplot(data, vert=False, widths=0.6, showfliers=False,
                       patch_artist=True, medianprops={"color": "k", "lw": 0.8})
        for patch, colour in zip(bp["boxes"], cols, strict=True):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)
            patch.set_linewidth(0.4)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("months after the peak with |anomaly| > 0.5 sigma")
        ax.set_title("how long the perturbation lasts", fontsize=7, loc="left")
    P.panel_label(ax, "d", x=-0.42)

    out = figure_path(cfg, "fig04_lag_cascade")
    paths = P.save_figure(fig, out, formats=cfg["figures.format"],
                          dpi=int(cfg["figures.dpi"]))
    log.info("wrote %s", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
