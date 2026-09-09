#!/usr/bin/env python3
"""Step 14 - what changes the response: warming, or water management?

ISIMIP3a is built for exactly this question. Two factorial contrasts
isolate the two drivers, using the *same* impact models and the same
ENSO events:

* **obsclim - counterclim** (with human influence held at histsoc):
  ``counterclim`` repeats the observed weather with the century-scale
  warming removed, so the difference is the part of the super-El-Nino
  hydrological response attributable to the warmed background state.
  A positive contrast for a dry-signal region means warming has made
  El-Nino droughts deeper.
* **histsoc - nosoc** (with climate held at obsclim): ``nosoc`` removes
  reservoirs, irrigation and industrial abstraction, so the difference
  is the extent to which water management damps -- or amplifies -- the
  climatic signal before it reaches the river.

Both contrasts are tested by bootstrapping over events, and reported
per variable, per season and per latitude band.

Run:  python workflow/14_scenario_contrasts.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from _common import composite_path, skip_existing, step_setup

from enso_inwaters import isimip_io as IO
from enso_inwaters import regions as R
from enso_inwaters import stats as S
from enso_inwaters.isimip_io import save_table

CONTRASTS = {
    "climate_change": {
        "a": ("obsclim", "histsoc"), "b": ("counterclim", "histsoc"),
        "question": "how much of the response is due to the warmed background"},
    "human_water_use": {
        "a": ("obsclim", "histsoc"), "b": ("obsclim", "nosoc"),
        "question": "how much do reservoirs and abstraction modify the response"},
}


def add_args(p):
    p.add_argument("--classes", default="super,reference")
    p.add_argument("--n-boot", type=int, default=2000)


def bootstrap_difference(a: xr.DataArray, b: xr.DataArray, n_boot: int,
                         seed: int) -> xr.Dataset:
    """Paired bootstrap over events of the per-event difference a - b."""
    common = [e for e in a["event"].values if e in set(b["event"].values)]
    a = a.sel(event=common)
    b = b.sel(event=common)
    d = (a - b).transpose("event", ...)
    vals = d.values
    n = vals.shape[0]
    obs = np.nanmean(vals, axis=0)
    rng = np.random.default_rng(seed)
    cross = np.zeros(obs.shape, dtype=np.int32)
    ok_total = np.zeros(obs.shape, dtype=np.int32)
    for _ in range(n_boot):
        with np.errstate(invalid="ignore"):
            m = np.nanmean(vals[rng.integers(0, n, n)], axis=0)
        ok = np.isfinite(m)
        cross += ok & (np.sign(m) != np.sign(obs))
        ok_total += ok
    p = np.clip(2.0 * (cross + 1.0) / (ok_total + 1.0), 0, 1)
    dims = d.dims[1:]
    coords = {k: d[k] for k in dims if k in d.coords}
    return xr.Dataset({
        "difference": xr.DataArray(obs, dims=dims, coords=coords),
        "p_value": xr.DataArray(p, dims=dims, coords=coords),
    }, attrs={"n_events": int(n), "events": ";".join(map(str, common))})


def main() -> int:
    cfg, log, args = step_setup("14_scenario_contrasts", __doc__, extra=add_args)
    out_root = cfg.path("processed", "contrasts", mkdir=True)
    comp_root = cfg.path("processed", "composites")
    seed = int(cfg["statistics.bootstrap.seed"])
    alpha_fdr = float(cfg["statistics.significance.alpha_fdr"])

    rows = []
    for name, spec in CONTRASTS.items():
        (ca, sa), (cb, sb) = spec["a"], spec["b"]
        variables = sorted({p.stem.split("_")[1]
                            for p in comp_root.glob(f"composite_*_{ca}_{sa}.nc")})
        log.info("--- contrast %s (%s) ---", name, spec["question"])
        for variable in variables:
            for klass in [c.strip() for c in args.classes.split(",")]:
                pa = composite_path(cfg, variable, klass, ca, sa)
                pb = composite_path(cfg, variable, klass, cb, sb)
                if not (pa.exists() and pb.exists()):
                    log.debug("  %s/%s: missing one side of the contrast",
                              variable, klass)
                    continue
                out = out_root / f"contrast_{name}_{variable}_{klass}.nc"
                if skip_existing(out, args, log):
                    continue

                da = xr.open_dataset(pa)["seasonal"].mean("model", skipna=True)
                db = xr.open_dataset(pb)["seasonal"].mean("model", skipna=True)
                res = bootstrap_difference(da, db, args.n_boot, seed)
                res["significant"] = S.fdr_mask(res["p_value"], alpha_fdr)
                res.attrs.update({"contrast": name, "variable": variable,
                                  "event_class": klass,
                                  "scenario_a": f"{ca}/{sa}",
                                  "scenario_b": f"{cb}/{sb}",
                                  "question": spec["question"]})
                if not args.dry_run:
                    IO.save_netcdf(res, out, "14_scenario_contrasts", cfg)

                for season in res["season"].values:
                    d = res["difference"].sel(season=season)
                    sig = res["significant"].sel(season=season)
                    ref_amp = float(np.abs(
                        xr.open_dataset(pa)["seasonal"].mean(["model", "event"])
                        .sel(season=season)).mean())
                    rows.append({
                        "contrast": name, "variable": variable,
                        "event_class": klass, "season": str(season),
                        "mean_difference": float(R.global_mean(d)),
                        "mean_abs_difference": float(R.global_mean(np.abs(d))),
                        "relative_to_response": (float(R.global_mean(np.abs(d))) / ref_amp
                                                 if ref_amp > 0 else np.nan),
                        "frac_area_significant": float(sig.mean()),
                    })
                log.info("  %-12s %-10s mean |difference| %.3f sigma "
                         "(%.0f%% of area significant, DJF01)",
                         variable, klass,
                         float(R.global_mean(np.abs(res["difference"]
                                                    .sel(season="DJF01"))))
                         if "DJF01" in [str(s) for s in res["season"].values] else np.nan,
                         100 * float(res["significant"].sel(season="DJF01").mean())
                         if "DJF01" in [str(s) for s in res["season"].values] else np.nan)

    if rows and not args.dry_run:
        df = pd.DataFrame(rows)
        save_table(df, cfg.path("tables", "table_7_scenario_contrasts.csv"),
                   "14_scenario_contrasts", cfg)
        log.info("\n%s", df.to_string(index=False))
    elif not rows:
        log.warning("no contrast could be computed - the counterclim and/or "
                    "nosoc composites are missing. Run step 07 with "
                    "--climate-scenario counterclim and --soc-scenario nosoc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
