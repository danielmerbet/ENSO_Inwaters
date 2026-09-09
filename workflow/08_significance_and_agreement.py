#!/usr/bin/env python3
"""Step 08 - significance of the composites and multi-model robustness.

Three things a composite of four events needs before it can be believed:

1. **A Monte-Carlo null.** The composite is compared with 10 000
   composites built from the same number of randomly drawn *neutral*
   anchor years (years within one year of any ENSO event are excluded so
   the null cannot accidentally sample a real teleconnection). This
   respects the autocorrelation of the fields, because each null draw
   uses the same contiguous epoch windows as the real one.
2. **Field significance.** Testing 60 000 land cells at p < 0.05 yields
   3 000 false positives by construction. The maps therefore report
   Benjamini-Hochberg FDR-controlled significance (alpha_FDR = 0.10,
   Wilks 2016), not raw local p-values.
3. **Ensemble agreement.** A signal is called robust only when at least
   ``min_models`` models are available and >= 2/3 of them agree on the
   sign of the ensemble mean.

Outputs (per variable and event class), in
``data/processed/significance/``: the ensemble-mean composite, its
Monte-Carlo p-value, the FDR mask, the sign agreement and the combined
"robust" mask that the figures stipple.

Run:  python workflow/08_significance_and_agreement.py [--variable dis]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from _common import composite_path, skip_existing, step_setup

from enso_inwaters import enso as E
from enso_inwaters import isimip_io as IO
from enso_inwaters import stats as S
from enso_inwaters.isimip_io import save_table


def add_args(p):
    p.add_argument("--variable", default=None)
    p.add_argument("--classes", default="super,strong,moderate,reference,all_el_nino")
    p.add_argument("--n-iterations", type=int, default=None)
    p.add_argument("--seasons-only", action="store_true",
                   help="test seasonal means only (much faster on full grids)")


def ensemble_anomaly(anom_root: Path, variable: str, clim: str, soc: str,
                     log) -> tuple[xr.DataArray, list[str]]:
    """Multi-model mean anomaly time series (the field that is mapped)."""
    files = sorted(anom_root.rglob(f"*_{variable}_{clim}_{soc}_anom.nc"))
    arrays, models = {}, []
    for f in files:
        ds = xr.open_dataset(f)
        name = variable if variable in ds else next(iter(ds.data_vars))
        arrays[f.stem.split("_")[0]] = ds[name]
        models.append(f.stem.split("_")[0])
    if not arrays:
        raise FileNotFoundError(f"no anomaly files for {variable} {clim}/{soc}")
    ens = IO.concat_models(arrays)
    log.info("%s: %d model(s) %s", variable, len(models), sorted(models))
    return ens.mean("model", skipna=True), sorted(models)


def main() -> int:
    cfg, log, args = step_setup("08_significance_and_agreement", __doc__, extra=add_args)
    anom_root = cfg.path("interim", "anomalies")
    comp_root = cfg.path("processed", "composites")
    out_root = cfg.path("processed", "significance", mkdir=True)

    events = pd.read_csv(cfg.path("processed", "enso", "enso_events.csv"))
    clim = cfg["isimip.main_scenario.climate_scenario"]
    soc = cfg["isimip.main_scenario.soc_scenario"]
    n_iter = args.n_iterations or int(cfg["statistics.bootstrap.n_iterations"])
    seed = int(cfg["statistics.bootstrap.seed"])
    alpha = float(cfg["statistics.significance.alpha"])
    alpha_fdr = float(cfg["statistics.significance.alpha_fdr"])
    lags = np.arange(int(cfg["composite.lag_min"]), int(cfg["composite.lag_max"]) + 1)
    seasons = {s: E._season_to_lags(s)
               for group in cfg["composite.key_seasons"].values() for s in group}

    comps = sorted(comp_root.glob(f"composite_*_{clim}_{soc}.nc"))
    if not comps:
        log.error("no composites in %s - run step 07 first", comp_root)
        return 1

    all_event_years = events.loc[events.in_main_sample, "year0"].astype(int).values
    rows = []

    variables = sorted({p.stem.split("_")[1] for p in comps})
    if args.variable:
        variables = [args.variable]

    for variable in variables:
        try:
            ens_anom, models = ensemble_anomaly(anom_root, variable, clim, soc, log)
        except FileNotFoundError as exc:
            log.warning("%s", exc)
            continue

        years = np.unique(pd.DatetimeIndex(ens_anom.time.values).year)
        # anchors must have a full epoch inside the record
        usable = years[(years >= years.min() - int(lags.min()) // 12) &
                       (years <= years.max() - int(np.ceil(lags.max() / 12)))]
        reducer = (S.seasonal_reducer(lags, seasons) if args.seasons_only else None)
        log.info("building the anchor pool for %s (%d candidate years, %s)",
                 variable, len(usable), "seasonal" if args.seasons_only else "lags")
        pool, _ = S.build_anchor_pool(ens_anom, usable, lags, reducer=reducer)
        if args.seasons_only:
            pool = pool.assign_coords(season=list(seasons))

        for klass in args.classes.split(","):
            klass = klass.strip()
            cpath = composite_path(cfg, variable, klass, clim, soc)
            if not cpath.exists():
                log.debug("no composite for %s/%s", variable, klass)
                continue
            out = out_root / f"significance_{variable}_{klass}_{clim}_{soc}.nc"
            if skip_existing(out, args, log):
                continue

            ds_comp = xr.open_dataset(cpath)
            klass_years = E.select_event_class(events, klass)["year0"].astype(int)
            ev_years = [int(y) for y in klass_years
                        if y in pool["anchor_year"].values]
            if len(ev_years) < 2:
                log.warning("%s/%s: only %d event(s) inside the epoch-safe "
                            "period - skipped", variable, klass, len(ev_years))
                continue

            log.info("%s / %-12s testing %d event(s) with %d iterations",
                     variable, klass, len(ev_years), n_iter)
            res = S.mc_composite_test(pool, ev_years, n_iterations=n_iter,
                                      seed=seed, exclude_years=all_event_years)

            per_model = ds_comp["composite"]              # (model, lag|season, ...)
            if args.seasons_only:
                per_model = ds_comp["seasonal"].mean("event") \
                    if "event" in ds_comp["seasonal"].dims else ds_comp["seasonal"]
            agreement = S.sign_agreement(per_model, "model")
            robust = S.robust_mask(
                per_model, res.p_value, dim="model",
                agreement_fraction=float(cfg["statistics.multimodel.agreement_fraction"]),
                min_models=int(cfg["statistics.multimodel.min_models"]),
                alpha=alpha, alpha_fdr=alpha_fdr)
            fdr = S.fdr_mask(res.p_value, alpha_fdr)

            ds_out = xr.Dataset({
                "composite": res.composite,
                "p_value": res.p_value,
                "null_mean": res.null_mean,
                "null_std": res.null_std,
                "fdr_significant": fdr,
                "sign_agreement": agreement,
                "robust": robust,
            })
            ds_out.attrs.update({
                "variable": variable, "event_class": klass,
                "n_events": len(ev_years),
                "events": ";".join(str(y) for y in ev_years),
                "n_iterations": n_iter, "models": ";".join(models),
                "fdr_threshold": float(fdr.attrs.get("fdr_threshold", np.nan)),
                "alpha_fdr": alpha_fdr})
            if not args.dry_run:
                IO.save_netcdf(ds_out, out, "08_significance_and_agreement", cfg)

            frac_sig = float(fdr.mean())
            frac_robust = float(robust.mean())
            log.info("   FDR threshold p<=%.4g | %.1f%% of tested points "
                     "significant | %.1f%% robust",
                     fdr.attrs.get("fdr_threshold", np.nan),
                     100 * frac_sig, 100 * frac_robust)
            rows.append({"variable": variable, "event_class": klass,
                         "n_events": len(ev_years), "n_models": len(models),
                         "fdr_threshold": fdr.attrs.get("fdr_threshold", np.nan),
                         "frac_significant": frac_sig,
                         "frac_robust": frac_robust,
                         "max_abs_composite": float(np.nanmax(np.abs(res.composite)))})

    if rows and not args.dry_run:
        save_table(pd.DataFrame(rows),
                   cfg.path("tables", "table_S4_significance_summary.csv"),
                   "08_significance_and_agreement", cfg)
    log.info("tested %d variable/class combination(s)", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
