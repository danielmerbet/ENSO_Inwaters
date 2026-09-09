#!/usr/bin/env python3
"""Step 25 - Figure 6: consequences, and what modulates them.

(a) Drought and flood months in the 24 months after the event peak,
    super events against moderate/strong and against the neutral
    baseline, by variable.
(b) Land area and population living where the response is robust, for
    super and reference events -- the exposure numbers, with the
    caveat that they count people in affected cells, not impacts.
(c) The obsclim - counterclim contrast: how much of the super-El-Nino
    response is attributable to the warmed background climate.
(d) The histsoc - nosoc contrast: how much reservoirs, irrigation and
    abstraction modify the signal before it reaches the river.

Run:  python workflow/25_figure6_impacts.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import figure_path, step_setup

from enso_inwaters import plotting as P

CLASS_COLOUR = {"super": "#08306b", "reference": "#9ecae1",
                "all_el_nino": "#4292c6"}


def add_args(p):
    p.add_argument("--season", default="DJF01")


def _bar_by_class(ax, df, value_col, label, log):
    variables = list(dict.fromkeys(df["variable"]))
    classes = [c for c in ("super", "reference") if c in set(df["event_class"])]
    width = 0.8 / max(len(classes), 1)
    for i, klass in enumerate(classes):
        sub = df[df.event_class == klass].set_index("variable").reindex(variables)
        ax.bar(np.arange(len(variables)) + i * width - 0.4 + width / 2,
               sub[value_col].values, width=width,
               color=CLASS_COLOUR.get(klass, "0.6"), label=klass)
    ax.set_xticks(np.arange(len(variables)))
    ax.set_xticklabels(variables, fontsize=6)
    ax.set_ylabel(label)
    ax.legend(fontsize=5.5, frameon=False)


def main() -> int:
    cfg, log, args = step_setup("25_figure6_impacts", __doc__, extra=add_args)
    P.set_style(int(cfg["figures.font_size"]))
    import matplotlib.pyplot as plt

    tables = cfg.path("tables")
    fig = plt.figure(figsize=(183 * P.MM, 125 * P.MM))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.35)

    # --- (a) drought / flood months ------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    f = tables / "table_6_extremes.csv"
    if f.exists():
        df = pd.read_csv(f)
        idx = list(dict.fromkeys(df["index"]))
        classes = [c for c in ("super", "reference") if c in set(df.event_class)]
        width = 0.8 / (len(classes) + 1)
        for i, klass in enumerate(classes):
            sub = df[df.event_class == klass].set_index("index").reindex(idx)
            ax.bar(np.arange(len(idx)) + i * width - 0.4,
                   sub["mean_drought_months_event"], width=width,
                   color=CLASS_COLOUR.get(klass, "0.6"), label=f"{klass} (drought)")
        sub = df.drop_duplicates("index").set_index("index").reindex(idx)
        ax.bar(np.arange(len(idx)) + len(classes) * width - 0.4,
               sub["mean_drought_months_neutral"], width=width,
               color="0.75", label="neutral baseline")
        ax.set_xticks(np.arange(len(idx)))
        ax.set_xticklabels([str(i).upper() for i in idx], fontsize=6)
        ax.set_ylabel("drought months in the\n24 months after the peak")
        ax.legend(fontsize=5.2, frameon=False)
        ax.set_title("hydrological drought", fontsize=7, loc="left")
    else:
        ax.axis("off")
        log.warning("no extremes table - run step 13")
    P.panel_label(ax, "a", x=-0.22)

    # --- (b) exposure ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    f = tables / "table_8_exposure.csv"
    if f.exists():
        df = pd.read_csv(f)
        key = "season" if "season" in df.columns else "lag"
        df = df[df[key].astype(str) == args.season]
        col = ("pop_dry_millions" if "pop_dry_millions" in df.columns
               else "area_dry_1e6km2")
        label = ("population in robustly drier cells\n(millions)"
                 if col.startswith("pop") else
                 "land area robustly drier\n(million km2)")
        _bar_by_class(ax, df, col, label, log)
        ax.set_title(f"exposure, {args.season}", fontsize=7, loc="left")
    else:
        ax.axis("off")
        log.warning("no exposure table - run step 15")
    P.panel_label(ax, "b", x=-0.22)

    # --- (c, d) scenario contrasts --------------------------------------
    f = tables / "table_7_scenario_contrasts.csv"
    contrasts = [("climate_change",
                  "warming contribution\n(obsclim - counterclim)"),
                 ("human_water_use",
                  "water-management contribution\n(histsoc - nosoc)")]
    for j, (name, label) in enumerate(contrasts):
        ax = fig.add_subplot(gs[1, j])
        if f.exists():
            df = pd.read_csv(f)
            df = df[(df.contrast == name) & (df.season == args.season)]
            if not df.empty:
                _bar_by_class(ax, df, "relative_to_response",
                              "|difference| as a fraction\nof the response", log)
                ax.axhline(0, color="0.4", lw=0.6)
                ax.set_title(label, fontsize=7, loc="left")
            else:
                ax.axis("off")
                log.warning("contrast %s has no rows for %s", name, args.season)
        else:
            ax.axis("off")
            log.warning("no scenario-contrast table - run step 14")
        P.panel_label(ax, "cd"[j], x=-0.22)

    out = figure_path(cfg, "fig06_impacts")
    paths = P.save_figure(fig, out, formats=cfg["figures.format"],
                          dpi=int(cfg["figures.dpi"]))
    log.info("wrote %s", ", ".join(p.name for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
