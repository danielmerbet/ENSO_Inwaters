#!/usr/bin/env python3
"""Step 31 - extract every number quoted in the manuscript.

Writes ``results/paper_numbers.json`` and a Markdown summary containing
exactly the values the text cites: how many super events, their mean
amplitude, the fraction of the land surface with a robust response, the
lag ordering, the nonlinear excess, the basin amplification, the
exposure figures, the model-evaluation scores and the variance
partition.

The point is that no number in the paper is typed by hand. If a
configuration changes, this file changes with it, and the manuscript
placeholders ({{n_super_events}} and friends) are refilled by
``40_render_manuscript.py``.

Run:  python workflow/31_paper_numbers.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from _common import step_setup

from enso_inwaters.utils import write_json


def read_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def main() -> int:
    cfg, log, args = step_setup("31_paper_numbers", __doc__)
    tdir = cfg.path("tables")
    pdir = cfg.path("processed")
    out: dict = {"_generated_by": "workflow/31_paper_numbers.py",
                 "_config": str(cfg.source)}

    # --- events ---------------------------------------------------------
    summary_file = pdir / "enso" / "event_summary.json"
    if summary_file.exists():
        s = json.loads(summary_file.read_text())
        out["events"] = {
            "period": s["period"],
            "n_el_nino": s["n_el_nino"],
            "n_la_nina": s["n_la_nina"],
            "class_counts": s["el_nino_class_counts"],
            "super_events": s["super_events"],
            "n_super": len(s["super_events"]),
            "mean_peak_oni_super": s.get("mean_peak_oni_super"),
            "matches_literature": s.get("matches_expectation"),
        }

    ev = read_csv(pdir / "enso" / "enso_events.csv")
    if ev is not None:
        ref = ev[(ev.phase == "el_nino") & ev.in_main_sample &
                 ev.amplitude_class.isin(["moderate", "strong"])]
        out.setdefault("events", {})["n_reference"] = int(len(ref))
        out["events"]["mean_peak_oni_reference"] = float(ref["peak_oni"].mean())
        if out["events"].get("mean_peak_oni_super"):
            out["events"]["linear_scaling_factor"] = float(
                out["events"]["mean_peak_oni_super"] /
                out["events"]["mean_peak_oni_reference"])

    # --- significance ---------------------------------------------------
    sig = read_csv(tdir / "table_S4_significance_summary.csv")
    if sig is not None:
        sup = sig[sig.event_class == "super"]
        out["significance"] = {
            "by_variable": {
                str(r.variable): {"frac_robust": float(r.frac_robust),
                                  "frac_significant": float(r.frac_significant),
                                  "max_abs_composite": float(r.max_abs_composite)}
                for r in sup.itertuples()},
            "max_frac_robust": float(sup["frac_robust"].max()) if len(sup) else None,
            "n_models": int(sup["n_models"].max()) if len(sup) else None,
        }

    # --- lag cascade ----------------------------------------------------
    lag = read_csv(tdir / "table_3_lag_cascade.csv")
    if lag is not None:
        lag = lag.sort_values("peak_lag_area_mean")
        out["lag_cascade"] = {
            "ordering": list(lag["variable"]),
            "peak_lag_months": {str(r.variable): int(r.peak_lag_area_mean)
                                for r in lag.itertuples()},
            "oni_xcorr_peak_lag": {str(r.variable): float(r.oni_xcorr_peak_lag)
                                   for r in lag.itertuples()},
            "median_recovery_months": {str(r.variable): float(r.median_recovery_months)
                                       for r in lag.itertuples()},
        }

    # --- nonlinearity ---------------------------------------------------
    nl = read_csv(tdir / "table_2_nonlinearity.csv")
    if nl is not None:
        mature = nl[nl.season == "DJF01"]
        out["nonlinearity"] = {
            "scaling_factor": float(nl["scaling_factor"].iloc[0]),
            "mature": {
                str(r.variable): {
                    "mean_abs_excess_sigma": float(r.mean_abs_excess),
                    "frac_area_significant": float(r.frac_area_significant),
                    "frac_steepening": (float(r.frac_steepening)
                                        if not pd.isna(r.frac_steepening) else None),
                    "median_steepening_ratio": (
                        float(r.median_steepening_ratio)
                        if hasattr(r, "median_steepening_ratio")
                        and not pd.isna(r.median_steepening_ratio) else None),
                } for r in mature.itertuples()},
        }

    # --- basins ---------------------------------------------------------
    amp = read_csv(tdir / "table_4_basin_amplification.csv")
    if amp is not None and "amplification" in amp:
        m = amp[amp.season == "DJF01"] if "season" in amp else amp
        out["basins"] = {
            "n_basins": int(m["region"].nunique()),
            "median_amplification": float(m["amplification"].median(skipna=True)),
            "iqr_amplification": [float(m["amplification"].quantile(0.25)),
                                  float(m["amplification"].quantile(0.75))],
        }

    # --- extremes -------------------------------------------------------
    ext = read_csv(tdir / "table_6_extremes.csv")
    if ext is not None:
        sup = ext[ext.event_class == "super"]
        ref = ext[ext.event_class == "reference"]
        out["extremes"] = {
            "super_drought_months": float(sup["mean_drought_months_event"].mean()),
            "reference_drought_months": float(ref["mean_drought_months_event"].mean())
            if len(ref) else None,
            "neutral_drought_months": float(sup["mean_drought_months_neutral"].mean()),
            "median_odds_ratio": float(sup["median_odds_ratio"].median()),
        }

    # --- exposure -------------------------------------------------------
    exp = read_csv(tdir / "table_8_exposure.csv")
    if exp is not None:
        key = "season" if "season" in exp.columns else "lag"
        m = exp[(exp[key].astype(str) == "DJF01") & (exp.event_class == "super")]
        out["exposure"] = {
            "area_dry_1e6km2": float(m["area_dry_1e6km2"].max()) if len(m) else None,
            "area_wet_1e6km2": float(m["area_wet_1e6km2"].max()) if len(m) else None,
        }
        if "pop_dry_millions" in m:
            out["exposure"]["pop_dry_millions"] = float(m["pop_dry_millions"].max())
            out["exposure"]["pop_wet_millions"] = float(m["pop_wet_millions"].max())

    # --- scenario contrasts ---------------------------------------------
    con = read_csv(tdir / "table_7_scenario_contrasts.csv")
    if con is not None:
        m = con[con.season == "DJF01"]
        out["contrasts"] = {
            name: {"median_relative_contribution":
                   float(g["relative_to_response"].median(skipna=True)),
                   "max_frac_area_significant": float(g["frac_area_significant"].max())}
            for name, g in m.groupby("contrast")}

    # --- evaluation ------------------------------------------------------
    ev_tab = read_csv(tdir / "table_9_model_evaluation.csv")
    if ev_tab is not None:
        out["evaluation"] = ev_tab.to_dict(orient="records")
    else:
        out["evaluation"] = None
        log.warning("no observational evaluation available - the manuscript "
                    "must state that the ensemble is unevaluated")

    # --- variance partition ----------------------------------------------
    vp = read_csv(tdir / "table_10_variance_partition.csv")
    if vp is not None:
        m = vp[(vp.season == "DJF01") & (vp.event_class == "super")]
        out["variance_partition"] = {
            str(r.variable): {"model": float(r.frac_model),
                              "event": float(r.frac_event),
                              "residual": float(r.frac_residual)}
            for r in m.itertuples()}

    if not args.dry_run:
        path = write_json(out, cfg.path("processed", "paper_numbers.json"))
        write_json(out, cfg.path("tables", "paper_numbers.json"))
        log.info("wrote %s", path)

    log.info("\n%s", json.dumps(out, indent=2, default=str)[:3000])
    missing = [k for k in ("events", "significance", "lag_cascade",
                           "nonlinearity", "basins", "extremes", "exposure",
                           "contrasts", "variance_partition")
               if k not in out or out[k] is None]
    if missing:
        log.warning("no numbers for: %s - the corresponding workflow steps "
                    "have not been run", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
