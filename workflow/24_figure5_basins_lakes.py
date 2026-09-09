#!/usr/bin/env python3
"""Step 24 - Figure 5: basins and lakes.

(a) Forest plot of the mature-season discharge anomaly for the major
    river basins, super events against the moderate/strong reference,
    with 90% bootstrap intervals over events. Basins are ordered by the
    super response, so the reader sees immediately which rivers carry
    the strongest signal and whether the super and reference intervals
    overlap.
(b) Basin amplification: the ratio of the super to the reference
    response, against the linear expectation (the ratio of mean peak
    ONI). Points above the line are basins where a super El Nino does
    more than a scaled-up strong one.
(c) Lake surface water temperature response by latitude band and season.
(d) Lake ice: change in ice-covered months per year during and after
    super events.

Run:  python workflow/24_figure5_basins_lakes.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import figure_path, step_setup

from enso_inwaters import plotting as P


def add_args(p):
    p.add_argument("--variable", default="dis")
    p.add_argument("--season", default="DJF01")
    p.add_argument("--top-n", type=int, default=22)


def main() -> int:
    cfg, log, args = step_setup("24_figure5_basins_lakes", __doc__, extra=add_args)
    P.set_style(int(cfg["figures.font_size"]))
    import matplotlib.pyplot as plt

    tables = cfg.path("tables")
    fig = plt.figure(figsize=(183 * P.MM, 135 * P.MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.45, wspace=0.35)

    # --- (a) basin forest plot -----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    basins = tables / "table_S5_basin_response.csv"
    if basins.exists():
        df = pd.read_csv(basins)
        df = df[(df.variable == args.variable) & (df.season == args.season)]
        sup = df[df.event_class == "super"].set_index("region")
        ref = df[df.event_class == "reference"].set_index("region")
        order = sup["mean_anomaly_sigma"].abs().sort_values(ascending=False)
        order = order.head(args.top_n).index[::-1]
        y = np.arange(len(order))
        for i, region in enumerate(order):
            r = sup.loc[region]
            ax.plot([r.ci90_low, r.ci90_high], [i, i], color="#08306b", lw=1.1)
            ax.plot(r.mean_anomaly_sigma, i, "o", ms=3.2, color="#08306b")
            if region in ref.index:
                q = ref.loc[region]
                ax.plot([q.ci90_low, q.ci90_high], [i - 0.28, i - 0.28],
                        color="#9ecae1", lw=1.1)
                ax.plot(q.mean_anomaly_sigma, i - 0.28, "o", ms=3.2,
                        color="#9ecae1")
        ax.set_yticks(y)
        ax.set_yticklabels([str(r)[:26] for r in order], fontsize=5)
        ax.axvline(0, color="0.4", lw=0.6)
        ax.set_xlabel(f"{args.variable} anomaly, {args.season} (sigma)")
        ax.legend(handles=[plt.Line2D([], [], color="#08306b", marker="o", ms=3,
                                      lw=1.1, label="super"),
                           plt.Line2D([], [], color="#9ecae1", marker="o", ms=3,
                                      lw=1.1, label="moderate + strong")],
                  fontsize=5.5, frameon=False, loc="upper left",
                  bbox_to_anchor=(0.0, -0.14), ncol=2)
        ax.set_title("basin response, 90% bootstrap interval over events",
                     fontsize=7, loc="left")
    else:
        ax.axis("off")
        log.warning("no basin table - run step 11")
    P.panel_label(ax, "a", x=-0.42)

    # --- (b) amplification ---------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    amp_file = tables / "table_4_basin_amplification.csv"
    if amp_file.exists():
        adf = pd.read_csv(amp_file)
        adf = adf[(adf.variable == args.variable) & (adf.season == args.season)]
        adf = adf.dropna(subset=["amplification", "reference"])
        ax.scatter(adf["reference"], adf["super"], s=10, color="#2171b5",
                   alpha=0.8, lw=0)
        lim = np.nanmax(np.abs(np.concatenate(
            [adf["reference"].values, adf["super"].values]))) * 1.1 or 1
        grid = np.linspace(-lim, lim, 10)
        # the linear expectation is the ratio of mean peak ONI
        events = pd.read_csv(cfg.path("processed", "enso", "enso_events.csv"))
        from enso_inwaters.enso import select_event_class
        k = (select_event_class(events, "super")["peak_oni"].mean() /
             select_event_class(events, "reference")["peak_oni"].mean())
        ax.plot(grid, grid * k, color="#b2182b", lw=1.0,
                label=f"linear scaling ({k:.2f}x)")
        ax.plot(grid, grid, color="0.6", lw=0.7, ls="--", label="1:1")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("moderate + strong response (sigma)")
        ax.set_ylabel("super response (sigma)")
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")
        above = float(np.mean(np.abs(adf["super"]) > k * np.abs(adf["reference"])))
        ax.set_title(f"{100*above:.0f}% of basins exceed the linear scaling",
                     fontsize=7, loc="left")
    else:
        ax.axis("off")
        log.warning("no basin amplification table - run step 11")
    P.panel_label(ax, "b", x=-0.20)

    # --- (c) lake temperature by latitude band --------------------------
    ax = fig.add_subplot(gs[1, 0])
    lake_file = tables / "table_5_lake_response.csv"
    if lake_file.exists():
        ldf = pd.read_csv(lake_file)
        ldf = ldf[(ldf.variable == "surftemp")]
        seasons = [s for s in ["JJA0", "SON0", "DJF01", "MAM1", "JJA1", "SON1"]
                   if s in set(ldf.season)]
        bands = list(dict.fromkeys(ldf["band"]))
        width = 0.8 / max(len(bands), 1)
        colours = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(bands)))
        for i, (band, colour) in enumerate(zip(bands, colours, strict=True)):
            sub = ldf[(ldf.band == band) & (ldf.event_class == "super")]
            sub = sub.set_index("season").reindex(seasons)
            ax.bar(np.arange(len(seasons)) + i * width - 0.4, sub["mean_anomaly"],
                   width=width, color=colour, label=band)
        ax.set_xticks(np.arange(len(seasons)))
        ax.set_xticklabels(seasons, fontsize=6)
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set_ylabel("lake surface temperature\nanomaly (sigma)")
        ax.legend(fontsize=4.8, frameon=False, ncol=2)
        ax.set_title("lake surface response by latitude band (super events)",
                     fontsize=7, loc="left")
    else:
        ax.axis("off")
        log.warning("no lake table - run step 12")
    P.panel_label(ax, "c", x=-0.20)

    # --- (d) lake ice ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ice_file = tables / "table_S6_lake_ice.csv"
    if ice_file.exists():
        idf = pd.read_csv(ice_file)
        piv = idf.pivot_table(index="model", columns="event_class",
                              values="mean_ice_month_anomaly")
        piv.plot.bar(ax=ax, color={"super": "#08306b", "reference": "#9ecae1"},
                     width=0.75, legend=True)
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set_ylabel("change in ice-covered\nmonths per year")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=25, labelsize=5.5)
        ax.legend(fontsize=5.5, frameon=False)
        ax.set_title("lake ice season", fontsize=7, loc="left")
    else:
        ax.axis("off")
        log.warning("no lake-ice table - run step 12")
    P.panel_label(ax, "d", x=-0.20)

    out = figure_path(cfg, "fig05_basins_lakes")
    paths = P.save_figure(fig, out, formats=cfg["figures.format"],
                          dpi=int(cfg["figures.dpi"]))
    log.info("wrote %s", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
