#!/usr/bin/env python3
"""Step 21 - Figure 2: the global hydrological footprint of a super El Nino.

A matrix of maps: one row per variable (runoff, discharge, groundwater
recharge, total water storage), one column per phase of the event
(developing JJA0, mature DJF01, decaying MAM1, lagged SON1). Stippling
marks cells that survive both the FDR-controlled Monte-Carlo test and
the two-thirds model-agreement criterion.

Reading the figure down a column shows where the perturbation is; along
a row shows how long it takes to arrive and how long it lasts.

Run:  python workflow/21_figure2_global_composites.py
"""

from __future__ import annotations

import xarray as xr
from _common import figure_path, step_setup

from enso_inwaters import plotting as P

VARIABLE_LABELS = {
    "qtot": "total runoff", "dis": "river discharge",
    "qr": "groundwater recharge", "tws": "total water storage",
    "groundwstor": "groundwater storage", "rootmoist": "root-zone soil moisture",
    "surftemp": "lake surface temperature",
}
SEASON_LABELS = {
    "JJA0": "developing (JJA$_0$)", "SON0": "developing (SON$_0$)",
    "DJF01": "mature (DJF$_{0/1}$)", "MAM1": "decaying (MAM$_1$)",
    "JJA1": "decaying (JJA$_1$)", "SON1": "lagged (SON$_1$)",
    "DJF12": "lagged (DJF$_{1/2}$)",
}


def add_args(p):
    p.add_argument("--variables", default="qtot,dis,qr,tws")
    p.add_argument("--seasons", default="JJA0,DJF01,MAM1,SON1")
    p.add_argument("--event-class", default="super")
    p.add_argument("--vmax", type=float, default=1.0)


def main() -> int:
    cfg, log, args = step_setup("21_figure2_global_composites", __doc__, extra=add_args)
    P.set_style(int(cfg["figures.font_size"]))
    import matplotlib.pyplot as plt

    sig_root = cfg.path("processed", "significance")
    clim = cfg["isimip.main_scenario.climate_scenario"]
    soc = cfg["isimip.main_scenario.soc_scenario"]
    variables = [v.strip() for v in args.variables.split(",")]
    seasons = [s.strip() for s in args.seasons.split(",")]

    available = []
    for v in variables:
        f = sig_root / f"significance_{v}_{args.event_class}_{clim}_{soc}.nc"
        if f.exists():
            available.append((v, f))
        else:
            log.warning("no significance file for %s - panel row dropped", v)
    if not available:
        log.error("nothing to plot - run step 08 first")
        return 1

    proj = cfg["figures.projection"]
    nrow, ncol = len(available), len(seasons)
    fig = plt.figure(figsize=(183 * P.MM, (30 * nrow + 22) * P.MM))
    axes = []
    for i, (variable, f) in enumerate(available):
        ds = xr.open_dataset(f)
        dim = "season" if "season" in ds["composite"].dims else "lag"
        for j, season in enumerate(seasons):
            k = i * ncol + j + 1
            ax = (fig.add_subplot(nrow, ncol, k,
                                  projection=P.ccrs.Robinson() if P.HAS_CARTOPY else None)
                  if P.HAS_CARTOPY else fig.add_subplot(nrow, ncol, k))
            if season not in [str(s) for s in ds[dim].values]:
                ax.axis("off")
                continue
            comp = ds["composite"].sel({dim: season})
            robust = ds["robust"].sel({dim: season}).astype(bool)
            P.plot_map(comp, ax=ax, vmin=-args.vmax, vmax=args.vmax,
                       cmap=cfg["figures.colormaps.anomaly"],
                       stipple=robust, hatch=cfg["figures.hatch_significant"],
                       add_colorbar=False, projection=proj)
            if i == 0:
                ax.set_title(SEASON_LABELS.get(season, season), fontsize=7)
            if j == 0:
                ax.text(-0.04, 0.5, VARIABLE_LABELS.get(variable, variable),
                        transform=ax.transAxes, rotation=90, va="center",
                        ha="right", fontsize=7)
                P.panel_label(ax, "abcdefgh"[i], x=-0.20, y=0.92)
            axes.append(ax)
            log.info("panel %s / %s: %.1f%% of cells robust", variable, season,
                     100 * float(robust.mean()))

    sm = plt.cm.ScalarMappable(
        cmap=cfg["figures.colormaps.anomaly"],
        norm=plt.Normalize(vmin=-args.vmax, vmax=args.vmax))
    cax = fig.add_axes([0.25, 0.055, 0.5, 0.014])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("standardised anomaly (sigma)")
    cb.outline.set_linewidth(0.4)

    n_ev = xr.open_dataset(available[0][1]).attrs.get("n_events", "?")
    fig.suptitle(f"Composite {args.event_class} El Nino response "
                 f"(n = {n_ev} events, multi-model mean); hatching = robust "
                 f"(FDR-controlled and >= 2/3 model agreement)",
                 fontsize=7.5, y=0.995)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.10,
                        hspace=0.05, wspace=0.02)

    out = figure_path(cfg, f"fig02_global_composites_{args.event_class}")
    paths = P.save_figure(fig, out, formats=cfg["figures.format"],
                          dpi=int(cfg["figures.dpi"]))
    log.info("wrote %s", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
