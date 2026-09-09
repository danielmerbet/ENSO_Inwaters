#!/usr/bin/env python3
"""Step 26 - supplementary figures.

S1  per-model composites for one variable, so the reader can see which
    model drives the ensemble mean rather than taking the mean on trust;
S2  variance partition maps (model / event / residual);
S3  the La Nina composite, for the symmetry question;
S4  sensitivity of the headline response to the epoch anchor, the
    climatology choice and the inclusion of the pre-satellite events.

Run:  python workflow/26_supplementary_figures.py
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from _common import composite_path, figure_path, step_setup

from enso_inwaters import plotting as P


def add_args(p):
    p.add_argument("--variable", default="dis")
    p.add_argument("--season", default="DJF01")


def main() -> int:
    cfg, log, args = step_setup("26_supplementary_figures", __doc__, extra=add_args)
    P.set_style(int(cfg["figures.font_size"]))
    import matplotlib.pyplot as plt

    clim = cfg["isimip.main_scenario.climate_scenario"]
    soc = cfg["isimip.main_scenario.soc_scenario"]
    written = []

    # --- S1: per-model composites --------------------------------------
    path = composite_path(cfg, args.variable, "super", clim, soc)
    if path.exists():
        ds = xr.open_dataset(path)
        seas = ds["seasonal"].sel(season=args.season).mean("event")
        models = [str(m) for m in seas["model"].values]
        ncol = min(3, len(models))
        nrow = int(np.ceil((len(models) + 1) / ncol))
        fig = plt.figure(figsize=(183 * P.MM, (34 * nrow + 14) * P.MM))
        for i, model in enumerate(models):
            ax = fig.add_subplot(nrow, ncol, i + 1,
                                 projection=P.ccrs.Robinson() if P.HAS_CARTOPY else None)
            P.plot_map(seas.sel(model=model), ax=ax, vmin=-1, vmax=1,
                       cmap=cfg["figures.colormaps.anomaly"], add_colorbar=False,
                       projection=cfg["figures.projection"])
            ax.set_title(model, fontsize=7, loc="left")
        ax = fig.add_subplot(nrow, ncol, len(models) + 1,
                             projection=P.ccrs.Robinson() if P.HAS_CARTOPY else None)
        P.plot_map(seas.mean("model"), ax=ax, vmin=-1, vmax=1,
                   cmap=cfg["figures.colormaps.anomaly"], add_colorbar=True,
                   projection=cfg["figures.projection"],
                   cbar_label="standardised anomaly (sigma)")
        ax.set_title("ensemble mean", fontsize=7, loc="left")
        fig.suptitle(f"Per-model super-El-Nino {args.variable} composite, "
                     f"{args.season}", fontsize=7.5)
        written += P.save_figure(fig, figure_path(cfg, "figS1_per_model"),
                                 formats=cfg["figures.format"],
                                 dpi=int(cfg["figures.dpi"]))
    else:
        log.warning("no composite for %s - S1 skipped", args.variable)

    # --- S2: variance partition ----------------------------------------
    vp = cfg.path("processed", "uncertainty",
                  f"variance_partition_{args.variable}_super.nc")
    if vp.exists():
        ds = xr.open_dataset(vp)
        fig = plt.figure(figsize=(183 * P.MM, 50 * P.MM))
        for i, (var, title) in enumerate((
                ("frac_model", "impact model"),
                ("frac_event", "which event"),
                ("frac_residual", "residual / internal"))):
            ax = fig.add_subplot(1, 3, i + 1,
                                 projection=P.ccrs.Robinson() if P.HAS_CARTOPY else None)
            P.plot_map(ds[var].sel(season=args.season), ax=ax, vmin=0, vmax=1,
                       cmap="magma_r", add_colorbar=(i == 2),
                       projection=cfg["figures.projection"],
                       cbar_label="fraction of variance")
            ax.set_title(title, fontsize=7, loc="left")
        fig.suptitle(f"Where the spread in the {args.variable} response comes "
                     f"from ({args.season})", fontsize=7.5)
        written += P.save_figure(fig, figure_path(cfg, "figS2_variance_partition"),
                                 formats=cfg["figures.format"],
                                 dpi=int(cfg["figures.dpi"]))
    else:
        log.warning("no variance partition for %s - S2 skipped", args.variable)

    # --- S3: La Nina composite ------------------------------------------
    ln = composite_path(cfg, args.variable, "all_la_nina", clim, soc)
    sup = composite_path(cfg, args.variable, "super", clim, soc)
    if ln.exists() and sup.exists():
        a = xr.open_dataset(sup)["seasonal"].sel(season=args.season).mean(
            ["model", "event"])
        b = xr.open_dataset(ln)["seasonal"].sel(season=args.season).mean(
            ["model", "event"])
        fig = plt.figure(figsize=(183 * P.MM, 52 * P.MM))
        for i, (field, title) in enumerate((
                (a, "super El Nino"), (b, "La Nina"), (a + b, "sum (asymmetry)"))):
            ax = fig.add_subplot(1, 3, i + 1,
                                 projection=P.ccrs.Robinson() if P.HAS_CARTOPY else None)
            P.plot_map(field, ax=ax, vmin=-1, vmax=1,
                       cmap=cfg["figures.colormaps.anomaly"],
                       add_colorbar=(i == 2), projection=cfg["figures.projection"],
                       cbar_label="standardised anomaly (sigma)")
            ax.set_title(title, fontsize=7, loc="left")
        fig.suptitle("El Nino / La Nina symmetry: a non-zero sum means the "
                     "two phases are not mirror images", fontsize=7.5)
        written += P.save_figure(fig, figure_path(cfg, "figS3_enso_symmetry"),
                                 formats=cfg["figures.format"],
                                 dpi=int(cfg["figures.dpi"]))
    else:
        log.warning("La Nina composite missing - S3 skipped")

    log.info("wrote %d supplementary figure file(s): %s", len(written),
             ", ".join(p.name for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
