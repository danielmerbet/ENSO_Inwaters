#!/usr/bin/env python3
"""Step 20 - Figure 1: the ENSO record and the super-event sample.

(a) ONI 1901-2019 with the amplitude classes shaded and the super events
    labelled -- this is the figure that defines the paper's sample.
(b) Peak ONI of every El Nino event, ordered, coloured by class and
    marked EP/CP, so the reader can see how far the four super events
    sit from the rest of the distribution.
(c) The mean ONI evolution of the super composite against the
    moderate/strong reference composite, on the same lag axis used
    throughout the paper.

Run:  python workflow/20_figure1_enso_events.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import figure_path, step_setup

from enso_inwaters import enso as E
from enso_inwaters import plotting as P

CLASS_COLOURS = {"weak": "#c6dbef", "moderate": "#6baed6",
                 "strong": "#2171b5", "super": "#08306b"}


def main() -> int:
    cfg, log, args = step_setup("20_figure1_enso_events", __doc__)
    P.set_style(int(cfg["figures.font_size"]))
    import matplotlib.pyplot as plt

    oni = pd.read_csv(cfg.path("processed", "enso", "oni.csv"),
                      parse_dates=["time"]).set_index("time")["oni"]
    events = pd.read_csv(cfg.path("processed", "enso", "enso_events.csv"))
    el = events[events.phase == "el_nino"].sort_values("year0")

    fig = plt.figure(figsize=(183 * P.MM, 130 * P.MM))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1], hspace=0.55,
                          wspace=0.25)

    # ---- (a) the ONI record ------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(oni.index, 0, oni.values, where=oni.values > 0,
                    color="#b2182b", alpha=0.75, lw=0)
    ax.fill_between(oni.index, 0, oni.values, where=oni.values < 0,
                    color="#2166ac", alpha=0.75, lw=0)
    for thr, ls in ((0.5, ":"), (1.5, "--"), (2.0, "-")):
        ax.axhline(thr, color="0.35", lw=0.5, ls=ls)
        ax.axhline(-thr, color="0.35", lw=0.5, ls=ls)
    ax.text(oni.index[5], 2.05, "super El Nino threshold (+2.0 degC)",
            fontsize=6, va="bottom", color="0.25")
    for r in el[el.amplitude_class == "super"].itertuples():
        t = pd.Timestamp(year=int(r.year0) + 1, month=1, day=1)
        ax.annotate(r.label, xy=(t, r.peak_oni), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=6,
                    fontweight="bold" if not r.presatellite else "normal",
                    color="0.1" if not r.presatellite else "0.45")
    ax.set_ylabel("ONI (degC)")
    ax.set_xlim(oni.index[0], oni.index[-1])
    P.panel_label(ax, "a", x=-0.045)
    ax.set_title("Oceanic Nino Index, 3-month running mean, sliding 30-year "
                 "base periods", loc="left", fontsize=7, color="0.3")

    # ---- (b) the amplitude distribution -------------------------------
    ax = fig.add_subplot(gs[1, 0])
    order = el.sort_values("peak_oni")
    colours = [CLASS_COLOURS.get(c, "0.7") for c in order.amplitude_class]
    ax.barh(np.arange(len(order)), order.peak_oni.values, color=colours,
            height=0.75)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order.label, fontsize=4.5)
    ax.axvline(2.0, color="0.2", lw=0.7)
    ax.set_xlabel("peak ONI (degC)")
    for i, r in enumerate(order.itertuples()):
        if r.flavour in ("EP", "CP"):
            ax.text(r.peak_oni + 0.04, i, r.flavour, va="center", fontsize=4.5,
                    color="0.35")
    P.panel_label(ax, "b", x=-0.28)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLOURS.values()]
    ax.legend(handles, list(CLASS_COLOURS), fontsize=5, frameon=False,
              loc="lower right", title="amplitude class", title_fontsize=5)

    # ---- (c) composite ONI evolution ----------------------------------
    ax = fig.add_subplot(gs[1, 1])
    lag_min, lag_max = int(cfg["composite.lag_min"]), int(cfg["composite.lag_max"])
    lags = np.arange(lag_min, lag_max + 1)
    for klass, colour in (("super", "#08306b"), ("reference", "#6baed6")):
        yrs = E.select_event_class(events, klass)["year0"].astype(int)
        stack = []
        for y in yrs:
            anchor = pd.Timestamp(year=int(y), month=12, day=1)
            vals = [oni.get(anchor + pd.DateOffset(months=int(lg)), np.nan)
                    for lg in lags]
            stack.append(vals)
        if not stack:
            continue
        arr = np.array(stack, dtype="float64")
        m = np.nanmean(arr, axis=0)
        ax.plot(lags, m, color=colour, lw=1.3,
                label=f"{klass} (n={len(stack)})")
        ax.fill_between(lags, np.nanmin(arr, axis=0), np.nanmax(arr, axis=0),
                        color=colour, alpha=0.18, lw=0)
    ax.axhline(0, color="0.4", lw=0.5)
    ax.axvline(0, color="0.4", lw=0.5, ls=":")
    ax.set_xlabel("lag from December of the developing year (months)")
    ax.set_ylabel("ONI (degC)")
    ax.legend(fontsize=5.5, frameon=False)
    P.panel_label(ax, "c", x=-0.16)

    # ---- (d) class counts through the record --------------------------
    ax = fig.add_subplot(gs[2, :])
    decades = (el.year0 // 10 * 10)
    tab = pd.crosstab(decades, el.amplitude_class)
    for klass in ("weak", "moderate", "strong", "super"):
        if klass not in tab:
            tab[klass] = 0
    tab = tab[["weak", "moderate", "strong", "super"]]
    bottom = np.zeros(len(tab))
    for klass in tab.columns:
        ax.bar(tab.index, tab[klass].values, bottom=bottom, width=8,
               color=CLASS_COLOURS[klass], label=klass)
        bottom += tab[klass].values
    ax.set_xlabel("decade")
    ax.set_ylabel("El Nino events")
    ax.legend(fontsize=5.5, frameon=False, ncol=4)
    P.panel_label(ax, "d", x=-0.045)
    ax.set_title("El Nino events per decade by amplitude class "
                 "(pre-1950 events rest on sparser SST observations)",
                 loc="left", fontsize=6.5, color="0.3")

    out = figure_path(cfg, "fig01_enso_events")
    paths = P.save_figure(fig, out, formats=cfg["figures.format"],
                          dpi=int(cfg["figures.dpi"]))
    log.info("wrote %s", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
