#!/usr/bin/env python3
"""Step 09 - is a super El Nino more than a scaled-up strong El Nino?

This is the paper's central question, and it is testable two ways.

**(a) Excess over a linear scaling.** The response to moderate+strong
events is scaled by the ratio of mean peak ONI (super / reference) and
subtracted from the super composite. What is left is the *nonlinear
excess*. Its uncertainty comes from a bootstrap over events (events are
resampled with replacement within each group), which is the only
resampling unit that respects the tiny sample.

**(b) Quadratic response function.** Across all El Nino events, each
grid cell's seasonal response is regressed on peak ONI with a linear and
a quadratic term. A quadratic coefficient of the same sign as the linear
one means the response steepens with amplitude. The coefficient is
tested by bootstrapping the event sample.

Both are computed per season, so the answer can differ between the
developing, mature and decaying phases -- which it should, if the
nonlinearity comes from catchment storage rather than from the
atmospheric teleconnection.

Outputs: ``data/processed/nonlinearity/`` and a summary table.

Run:  python workflow/09_nonlinearity.py [--variable dis]
"""

from __future__ import annotations

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
    p.add_argument("--n-boot", type=int, default=2000)


def bootstrap_excess(super_seas: xr.DataArray, ref_seas: xr.DataArray,
                     scale: float, n_boot: int, seed: int) -> xr.Dataset:
    """Bootstrap the nonlinear excess over the two event samples."""
    rng = np.random.default_rng(seed)
    s = super_seas.values          # (event, ...)
    r = ref_seas.values
    n_s, n_r = s.shape[0], r.shape[0]
    obs = np.nanmean(s, axis=0) - scale * np.nanmean(r, axis=0)

    ge = np.zeros(obs.shape, dtype=np.int32)
    n_ok = np.zeros(obs.shape, dtype=np.int32)
    draws_sum = np.zeros(obs.shape)
    draws_sq = np.zeros(obs.shape)
    for _ in range(n_boot):
        si = rng.integers(0, n_s, n_s)
        ri = rng.integers(0, n_r, n_r)
        with np.errstate(invalid="ignore"):
            d = np.nanmean(s[si], axis=0) - scale * np.nanmean(r[ri], axis=0)
        ok = np.isfinite(d)
        draws_sum[ok] += d[ok]
        draws_sq[ok] += d[ok] ** 2
        n_ok += ok
        # p: how often does the bootstrap distribution cross zero?
        ge += ok & (np.sign(d) != np.sign(obs))
    with np.errstate(invalid="ignore", divide="ignore"):
        p = 2.0 * (ge + 1.0) / (n_ok + 1.0)     # two-sided
        p = np.clip(p, 0, 1)
        mean = draws_sum / np.where(n_ok > 0, n_ok, np.nan)
        sd = np.sqrt(np.clip(draws_sq / np.where(n_ok > 0, n_ok, np.nan)
                             - mean ** 2, 0, None))

    dims = super_seas.dims[1:]
    coords = {d: super_seas[d] for d in dims if d in super_seas.coords}
    mk = lambda a, n: xr.DataArray(a, dims=dims, coords=coords, name=n)
    return xr.Dataset({"excess": mk(obs, "excess"),
                       "excess_p": mk(p, "excess_p"),
                       "excess_sd": mk(sd, "excess_sd")})


def hinge_fit(seas: xr.DataArray, oni: np.ndarray, n_boot: int, seed: int,
              knot: float = 1.5, evaluate_at: float = 2.3) -> xr.Dataset:
    """Piecewise-linear response as a function of ENSO amplitude.

    The response of each grid cell is fitted, across **every year of the
    record**, as

    ``y = b0 + b1*ONI + h_pos*max(ONI-k, 0) + h_neg*min(ONI+k, 0)``

    with the knot ``k`` at the strong-event threshold (1.5 degC).

    * ``b1`` is the sensitivity over the ordinary ENSO range;
    * ``h_pos`` is the **extra** sensitivity beyond ONI = +1.5, i.e. the
      steepening that separates strong and super El Ninos from weaker
      ones. ``h_pos / b1`` is the paper's headline nonlinearity number;
    * ``h_neg`` is its La Nina counterpart, so the El Nino/La Nina
      symmetry of the nonlinearity can be reported rather than assumed.

    Why a hinge and not a quadratic. A quadratic in ONI cannot be
    identified here. Fitted on El Nino events alone, ONI and ONI^2
    correlate at r > 0.99 and the coefficients simply trade off; fitted
    over both phases, ONI^2 and ONI*|ONI| correlate at ~0.95 because the
    sample's large values are not symmetric. The hinge basis has none of
    that problem: its columns are zero over most of the sample, it
    localises the extra sensitivity exactly where the question is
    (large-amplitude events) and its coefficient has a direct physical
    reading.

    Statistics: coefficients are bootstrapped over years, and
    ``hinge_p`` is the two-sided proportion of resamples in which
    ``h_pos`` flips sign.
    """
    y = seas.values                              # (sample, ...)
    flat = y.reshape(y.shape[0], -1)

    # Anchor years covering far fewer cells than the rest are the ends of
    # the moving-climatology window; keeping them would void every cell
    # in the vectorised solve and force the slow per-cell path everywhere.
    coverage = np.isfinite(flat).sum(axis=1)
    rows_ok = coverage >= 0.5 * coverage.max()
    flat = flat[rows_ok]
    x = np.asarray(oni, dtype="float64")[rows_ok]
    n = int(rows_ok.sum())

    X = np.column_stack([np.ones(n), x,
                         np.maximum(x - knot, 0.0),
                         np.minimum(x + knot, 0.0)])
    n_terms = X.shape[1]
    n_above = int((x > knot).sum())
    n_below = int((x < -knot).sum())
    corr = np.corrcoef(X[:, 1:], rowvar=False)
    collinearity = float(np.max(np.abs(corr - np.eye(n_terms - 1))))
    min_samples = max(20, n_terms * 6)

    def _solve(Xm, Ym, fallback: bool = True):
        """Least squares per cell, tolerating ragged missing data.

        Complete cells are solved in one vectorised call; cells with
        scattered gaps fall back to a per-cell fit so they are not
        silently dropped. The fallback runs only for the point estimate --
        inside the bootstrap it would dominate the runtime for no gain,
        so there ragged cells are left NaN.
        """
        out = np.full((n_terms, Ym.shape[1]), np.nan)
        if np.linalg.matrix_rank(Xm) < n_terms:
            return out
        finite = np.isfinite(Ym)
        full = finite.all(axis=0)
        if full.any():
            out[:, full] = np.linalg.lstsq(Xm, Ym[:, full], rcond=None)[0]
        if not fallback:
            return out
        n_ok = finite.sum(axis=0)
        for c in np.nonzero(~full & (n_ok >= min_samples))[0]:
            m = finite[:, c]
            Xc = Xm[m]
            if np.linalg.matrix_rank(Xc) == n_terms:
                out[:, c] = np.linalg.lstsq(Xc, Ym[m, c], rcond=None)[0]
        return out

    beta = _solve(X, flat, fallback=True)
    b1, h_pos, h_neg = beta[1], beta[2], beta[3]

    with np.errstate(invalid="ignore", divide="ignore"):
        steepening_ratio = np.where(np.abs(b1) > 1e-6, h_pos / b1, np.nan)
    # how much a super event exceeds the linear prediction, in sigma
    extra = h_pos * max(float(evaluate_at) - knot, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        linear = b1 * float(evaluate_at)
        excess_frac = np.where(np.abs(linear) > 1e-6, extra / linear, np.nan)

    rng = np.random.default_rng(seed)
    same_sign = np.zeros(flat.shape[1], dtype=np.int32)
    n_used = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if (x[idx] > knot).sum() < 3 or (x[idx] < -knot).sum() < 3:
            continue
        b = _solve(X[idx], flat[idx], fallback=False)
        same_sign += (np.sign(b[2]) == np.sign(h_pos)).astype(np.int32)
        n_used += 1
    p_h = np.clip(2.0 * (1.0 - same_sign / max(n_used, 1)), 0, 1)

    shape = seas.shape[1:]
    dims = seas.dims[1:]
    coords = {d: seas[d] for d in dims if d in seas.coords}
    mk = lambda a: xr.DataArray(np.asarray(a, dtype="float64").reshape(shape),
                                dims=dims, coords=coords)
    valid = np.isfinite(b1) & np.isfinite(h_pos)
    steep = np.where(valid, np.sign(b1) == np.sign(h_pos), np.nan)

    ds = xr.Dataset({
        "b1_linear": mk(b1),
        "hinge_el_nino": mk(h_pos),
        "hinge_la_nina": mk(h_neg),
        "steepening_ratio": mk(steepening_ratio),
        "nonlinear_excess_frac": mk(excess_frac),
        "hinge_p": mk(p_h),
        "steepening": mk(steep),
    })
    ds.attrs.update({
        "n_samples": n,
        "knot_oni": float(knot),
        "n_years_above_knot": n_above,
        "n_years_below_knot": n_below,
        "oni_range": f"{np.nanmin(x):.2f}..{np.nanmax(x):.2f}",
        "max_predictor_collinearity": collinearity,
        "evaluated_at_oni": float(evaluate_at),
        "n_bootstrap_used": int(n_used),
        "terms": "b0 + b1*ONI + h_pos*max(ONI-k,0) + h_neg*min(ONI+k,0)",
    })
    return ds


def yearly_seasonal_responses(cfg, variable: str, clim: str, soc: str,
                              seasons: dict, log):
    """Seasonal-mean anomalies for **every** anchor year of the record.

    This is the sample the quadratic fit needs: composites of El Nino
    events alone cannot identify a curvature (see ``quadratic_fit``).
    """
    from enso_inwaters import stats as _S
    anom_root = cfg.path("interim", "anomalies")
    files = sorted(anom_root.rglob(f"*_{variable}_{clim}_{soc}_anom.nc"))
    if not files:
        return None, None
    arrays = {}
    for f in files:
        ds = xr.open_dataset(f)
        arrays[f.stem.split("_")[0]] = ds[variable if variable in ds
                                          else next(iter(ds.data_vars))]
    ens = IO.concat_models(arrays).mean("model", skipna=True)
    lags = np.arange(int(cfg["composite.lag_min"]), int(cfg["composite.lag_max"]) + 1)
    years = np.unique(pd.DatetimeIndex(ens.time.values).year)
    usable = years[(years >= years.min() + 2) & (years <= years.max() - 3)]
    pool, _ = _S.build_anchor_pool(ens, usable, lags,
                                   reducer=_S.seasonal_reducer(lags, seasons))
    pool = pool.assign_coords(season=list(seasons))
    log.info("  quadratic-fit sample: %d anchor years (%d model(s))",
             len(usable), len(arrays))
    return pool, usable


def main() -> int:
    cfg, log, args = step_setup("09_nonlinearity", __doc__, extra=add_args)
    comp_root = cfg.path("processed", "composites")
    out_root = cfg.path("processed", "nonlinearity", mkdir=True)
    events = pd.read_csv(cfg.path("processed", "enso", "enso_events.csv"))
    clim = cfg["isimip.main_scenario.climate_scenario"]
    soc = cfg["isimip.main_scenario.soc_scenario"]
    seed = int(cfg["statistics.bootstrap.seed"])
    alpha_fdr = float(cfg["statistics.significance.alpha_fdr"])

    oni_series = pd.read_csv(cfg.path("processed", "enso", "oni.csv"),
                             parse_dates=["time"]).set_index("time")["oni"]
    seasons_lags = {s: E._season_to_lags(s)
                    for g in cfg["composite.key_seasons"].values() for s in g}

    sup_ev = E.select_event_class(events, "super")
    ref_ev = E.select_event_class(events, "reference")
    if sup_ev.empty or ref_ev.empty:
        log.error("need both a super and a reference (moderate+strong) sample")
        return 1
    amp_super = float(sup_ev["peak_oni"].mean())
    amp_ref = float(ref_ev["peak_oni"].mean())
    scale = amp_super / amp_ref
    log.info("mean peak ONI: super %.2f (n=%d), reference %.2f (n=%d) "
             "-> linear scaling factor %.2f",
             amp_super, len(sup_ev), amp_ref, len(ref_ev), scale)

    variables = sorted({p.stem.split("_")[1]
                        for p in comp_root.glob(f"composite_*_super_{clim}_{soc}.nc")})
    if args.variable:
        variables = [args.variable]
    if not variables:
        log.error("no super composites found - run step 07 first")
        return 1

    rows = []
    for variable in variables:
        out = out_root / f"nonlinearity_{variable}_{clim}_{soc}.nc"
        if skip_existing(out, args, log):
            continue
        try:
            sup = xr.open_dataset(composite_path(cfg, variable, "super", clim, soc))
            ref = xr.open_dataset(composite_path(cfg, variable, "reference", clim, soc))
            alle = xr.open_dataset(composite_path(cfg, variable, "all_el_nino", clim, soc))
        except FileNotFoundError as exc:
            log.warning("%s: %s", variable, exc)
            continue

        # multi-model mean of the per-event seasonal responses
        s_seas = sup["seasonal"].mean("model", skipna=True).transpose("event", ...)
        r_seas = ref["seasonal"].mean("model", skipna=True).transpose("event", ...)
        a_seas = alle["seasonal"].mean("model", skipna=True).transpose("event", ...)

        log.info("%s: %d super / %d reference / %d total events",
                 variable, s_seas.sizes["event"], r_seas.sizes["event"],
                 a_seas.sizes["event"])

        ds_ex = bootstrap_excess(s_seas, r_seas, scale, args.n_boot, seed)

        # Quadratic response: fitted over EVERY year of the record, not
        # over the El Nino events alone, so that ONI spans both phases
        # and the curvature is identifiable.
        pool, anchor_years = yearly_seasonal_responses(
            cfg, variable, clim, soc, seasons_lags, log)
        if pool is None:
            log.warning("%s: no anomaly fields for the quadratic fit", variable)
            ds_quad = xr.Dataset()
        else:
            djf = oni_series.reindex(
                [pd.Timestamp(year=int(y) + 1, month=1, day=1)
                 for y in anchor_years]).values
            keep = np.isfinite(djf)
            if keep.sum() < 20:
                log.warning("%s: only %d usable years for the quadratic fit",
                            variable, int(keep.sum()))
                ds_quad = xr.Dataset()
            else:
                sample = pool.isel(anchor_year=np.nonzero(keep)[0]) \
                             .rename({"anchor_year": "sample"}) \
                             .transpose("sample", ...)
                ds_quad = hinge_fit(sample, djf[keep], args.n_boot, seed,
                                    knot=float(cfg["enso.classes.strong"][0]),
                                    evaluate_at=amp_super)
                log.info("  hinge fit: %d years (%d above +%.1f, %d below "
                         "-%.1f), max predictor collinearity r=%.2f",
                         ds_quad.attrs["n_samples"],
                         ds_quad.attrs["n_years_above_knot"],
                         ds_quad.attrs["knot_oni"],
                         ds_quad.attrs["n_years_below_knot"],
                         ds_quad.attrs["knot_oni"],
                         ds_quad.attrs["max_predictor_collinearity"])

        ds_out = xr.merge([ds_ex, ds_quad])
        ds_out["excess_significant"] = S.fdr_mask(ds_ex["excess_p"], alpha_fdr)
        ds_out.attrs.update({
            "variable": variable, "scaling_factor": scale,
            "mean_peak_oni_super": amp_super, "mean_peak_oni_reference": amp_ref,
            "n_super": int(len(sup_ev)), "n_reference": int(len(ref_ev)),
            "n_bootstrap": args.n_boot,
            "interpretation": "excess > 0 where the super-El-Nino response "
                              "exceeds a linear scaling of the moderate/strong "
                              "response"})
        if not args.dry_run:
            IO.save_netcdf(ds_out, out, "09_nonlinearity", cfg)
        log.info("  wrote %s", out.name)

        for season in ds_ex["season"].values:
            ex = ds_ex["excess"].sel(season=season)
            sig = ds_out["excess_significant"].sel(season=season)
            rows.append({
                "variable": variable, "season": str(season),
                "scaling_factor": round(scale, 3),
                "mean_excess": float(ex.mean()),
                "mean_abs_excess": float(np.abs(ex).mean()),
                "frac_area_significant": float(sig.mean()),
                "frac_area_amplified": float((ex > 0).where(sig).mean()),
                "frac_steepening": (float(ds_quad["steepening"].sel(season=season).mean())
                                    if "steepening" in ds_quad else np.nan),
                "median_nonlinear_excess_frac": (
                    float(ds_quad["nonlinear_excess_frac"].sel(season=season)
                          .median()) if "nonlinear_excess_frac" in ds_quad
                    else np.nan),
                "frac_nonlinear_significant": (
                    float((ds_quad["hinge_p"].sel(season=season) < 0.1).mean())
                    if "hinge_p" in ds_quad else np.nan),
                "median_steepening_ratio": (
                    float(ds_quad["steepening_ratio"].sel(season=season).median())
                    if "steepening_ratio" in ds_quad else np.nan),
            })

    if rows and not args.dry_run:
        df = pd.DataFrame(rows)
        save_table(df, cfg.path("tables", "table_2_nonlinearity.csv"),
                   "09_nonlinearity", cfg)
        log.info("\n%s", df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
