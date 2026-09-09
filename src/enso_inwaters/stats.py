"""Statistics: Monte-Carlo epoch tests, FDR, ensemble agreement, ANOVA.

The inference problem here is specific: we have a *sample of four*
super El Nino events in a 119-year record, and we ask whether the
composite anomaly at a given lag differs from what an arbitrary set of
four windows would give. That rules out parametric tests -- the sample
is tiny, the fields are autocorrelated in time and space, and the
anomalies are not Gaussian. The workflow therefore uses:

1. a **random-anchor (Monte-Carlo) superposed-epoch test** -- the null is
   built by re-drawing the same number of anchor dates from
   ENSO-neutral years, keeping each epoch window contiguous so that the
   serial correlation of the hydrological memory is preserved;
2. **Benjamini-Hochberg FDR control** for field significance across the
   many grid cells tested (Wilks, 2016, BAMS), which is far less
   conservative than Bonferroni and has a clear interpretation;
3. **ensemble agreement** thresholds (>= 2/3 of models agreeing on sign)
   so that a significant ensemble mean driven by one outlier model is
   not reported as robust.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------
# Monte-Carlo superposed-epoch test
# ---------------------------------------------------------------------
@dataclass
class SEATestResult:
    composite: xr.DataArray     # observed event-mean anomaly
    p_value: xr.DataArray       # two-sided Monte-Carlo p
    null_mean: xr.DataArray
    null_std: xr.DataArray
    n_events: int
    n_iterations: int

    def significant(self, alpha: float = 0.05) -> xr.DataArray:
        return self.p_value < alpha


def eligible_anchor_years(all_years: np.ndarray, exclude_years: np.ndarray,
                          buffer_years: int = 1) -> np.ndarray:
    """Years usable as null anchors: neutral, and not adjacent to an event.

    Excluding the years either side of a real event stops the null from
    accidentally sampling the tail of a genuine teleconnection, which
    would bias the test towards non-significance.
    """
    bad = set()
    for y in np.atleast_1d(exclude_years):
        for d in range(-buffer_years, buffer_years + 1):
            bad.add(int(y) + d)
    return np.array([int(y) for y in all_years if int(y) not in bad])


def _gather_epochs(values: np.ndarray, time_index: pd.DatetimeIndex,
                   anchors: pd.DatetimeIndex, lags: np.ndarray) -> np.ndarray:
    """(n_events, n_lags, ...) view of ``values`` around each anchor.

    ``values`` has time on axis 0. Positions outside the record are NaN.
    """
    pos = {(t.year, t.month): i for i, t in enumerate(time_index)}
    n_t = values.shape[0]
    out = np.full((len(anchors), len(lags)) + values.shape[1:], np.nan,
                  dtype="float64")
    for e, a in enumerate(anchors):
        base = pos.get((a.year, a.month))
        if base is None:
            continue
        idx = base + lags
        ok = (idx >= 0) & (idx < n_t)
        out[e, ok] = values[idx[ok]]
    return out


def sea_test(da: xr.DataArray, anchors, lags: np.ndarray,
             candidate_years: np.ndarray, n_iterations: int = 10000,
             seed: int = 0, anchor_month: int = 12,
             time_dim: str = "time") -> SEATestResult:
    """Composite ``da`` on ``anchors`` and test it against random anchors.

    Parameters
    ----------
    da
        Anomaly field with a monthly ``time`` dimension; any number of
        trailing dimensions (lat/lon, basin, lake, ...).
    anchors
        Event anchor dates (lag 0), e.g. December of each developing year.
    lags
        Month offsets to composite over.
    candidate_years
        Years from which the null anchors are drawn (see
        :func:`eligible_anchor_years`).
    """
    da = da.transpose(time_dim, ...)
    time_index = pd.DatetimeIndex(da[time_dim].values)
    values = np.asarray(da.values, dtype="float64")
    anchors = pd.DatetimeIndex([pd.Timestamp(a) for a in anchors])
    lags = np.asarray(lags, dtype=int)

    obs = np.nanmean(_gather_epochs(values, time_index, anchors, lags), axis=0)

    rng = np.random.default_rng(seed)
    n_ev = len(anchors)
    ge = np.zeros_like(obs)          # count of |null| >= |obs|
    n_valid = np.zeros_like(obs)
    run_sum = np.zeros_like(obs)
    run_sq = np.zeros_like(obs)

    cand = np.asarray(candidate_years, dtype=int)
    if len(cand) < n_ev:
        raise ValueError("not enough neutral years to build the null "
                         f"({len(cand)} available, {n_ev} needed)")

    for _ in range(n_iterations):
        yrs = rng.choice(cand, size=n_ev, replace=False)
        null_anchors = pd.DatetimeIndex(
            [pd.Timestamp(year=int(y), month=anchor_month, day=1) for y in yrs])
        with np.errstate(invalid="ignore"):
            nm = np.nanmean(
                _gather_epochs(values, time_index, null_anchors, lags), axis=0)
        finite = np.isfinite(nm)
        run_sum[finite] += nm[finite]
        run_sq[finite] += nm[finite] ** 2
        n_valid += finite
        ge += finite & (np.abs(nm) >= np.abs(obs))

    with np.errstate(invalid="ignore", divide="ignore"):
        p = (ge + 1.0) / (n_valid + 1.0)      # +1: never report p = 0
        mean = run_sum / np.where(n_valid > 0, n_valid, np.nan)
        var = run_sq / np.where(n_valid > 0, n_valid, np.nan) - mean ** 2
        std = np.sqrt(np.clip(var, 0, None))

    dims = ("lag",) + tuple(d for d in da.dims if d != time_dim)
    coords = {"lag": lags}
    coords.update({d: da[d] for d in dims[1:] if d in da.coords})
    mk = lambda arr, name: xr.DataArray(arr, dims=dims, coords=coords, name=name)
    return SEATestResult(
        composite=mk(obs, "composite"), p_value=mk(p, "p_value"),
        null_mean=mk(mean, "null_mean"), null_std=mk(std, "null_std"),
        n_events=n_ev, n_iterations=n_iterations)


# ---------------------------------------------------------------------
# Pool-based Monte-Carlo test (the fast path used on full grids)
# ---------------------------------------------------------------------
def build_anchor_pool(da: xr.DataArray, years, lags: np.ndarray,
                      anchor_month: int = 12, time_dim: str = "time",
                      reducer=None) -> tuple[xr.DataArray, np.ndarray]:
    """Pre-extract one epoch per candidate anchor year.

    Composite tests re-draw anchor years thousands of times. Extracting
    the epoch each time dominates the cost, yet every draw uses windows
    from the same fixed pool -- so extract the pool **once**
    (``n_years x n_lags x ...``) and let the bootstrap be a mean over
    rows. That turns a 10 000-iteration test on a global 0.5-degree grid
    from hours into minutes.

    ``reducer`` optionally collapses the lag axis first (e.g. into
    seasonal means), which shrinks the pool further.
    """
    da = da.transpose(time_dim, ...)
    time_index = pd.DatetimeIndex(da[time_dim].values)
    values = np.asarray(da.values, dtype="float32")
    lags = np.asarray(lags, dtype=int)

    years = np.asarray([int(y) for y in years], dtype=int)
    anchors = pd.DatetimeIndex(
        [pd.Timestamp(year=int(y), month=anchor_month, day=1) for y in years])
    epochs = _gather_epochs(values, time_index, anchors, lags).astype("float32")

    if reducer is not None:
        epochs = reducer(epochs)          # (n_years, n_reduced, ...)
        coord_name, coord_vals = "season", None
    else:
        coord_name, coord_vals = "lag", lags

    dims = ("anchor_year", coord_name) + tuple(d for d in da.dims if d != time_dim)
    coords = {"anchor_year": years}
    if coord_vals is not None:
        coords[coord_name] = coord_vals
    coords.update({d: da[d] for d in dims[2:] if d in da.coords})
    pool = xr.DataArray(epochs, dims=dims, coords=coords, name="pool")
    return pool, years


def mc_composite_test(pool: xr.DataArray, event_years, n_iterations: int = 10000,
                      seed: int = 0, exclude_years=None,
                      buffer_years: int = 1) -> SEATestResult:
    """Monte-Carlo test of a composite against random anchor years.

    ``pool`` comes from :func:`build_anchor_pool`; the observed
    composite is the mean over ``event_years`` and the null is the mean
    over the same number of randomly drawn neutral years.
    """
    years = np.asarray(pool["anchor_year"].values, dtype=int)
    ev = np.asarray([int(y) for y in event_years], dtype=int)
    ev_idx = np.array([int(np.nonzero(years == y)[0][0]) for y in ev
                       if (years == y).any()])
    if ev_idx.size == 0:
        raise ValueError("none of the event years are in the anchor pool")

    values = np.asarray(pool.values, dtype="float32")
    with warnings.catch_warnings():   # all-NaN ocean cells are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        obs = np.nanmean(values[ev_idx], axis=0)

    extra = ([] if exclude_years is None else
             np.asarray(exclude_years, dtype=int).ravel())
    excl = np.concatenate([ev, np.asarray(extra, dtype=int)])
    keep = eligible_anchor_years(years, excl, buffer_years)
    cand_idx = np.array([int(np.nonzero(years == y)[0][0]) for y in keep])
    if cand_idx.size < ev_idx.size:
        raise ValueError(f"only {cand_idx.size} neutral anchor years available, "
                         f"{ev_idx.size} needed")

    rng = np.random.default_rng(seed)
    n = ev_idx.size
    ge = np.zeros(obs.shape, dtype=np.int32)
    n_valid = np.zeros(obs.shape, dtype=np.int32)
    run_sum = np.zeros(obs.shape, dtype=np.float64)
    run_sq = np.zeros(obs.shape, dtype=np.float64)
    abs_obs = np.abs(obs)

    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    for _ in range(n_iterations):
        pick = rng.choice(cand_idx, size=n, replace=False)
        with np.errstate(invalid="ignore"):
            nm = np.nanmean(values[pick], axis=0)
        finite = np.isfinite(nm)
        run_sum[finite] += nm[finite]
        run_sq[finite] += nm[finite] ** 2
        n_valid += finite
        ge += finite & (np.abs(nm) >= abs_obs)

    with np.errstate(invalid="ignore", divide="ignore"):
        p = (ge + 1.0) / (n_valid + 1.0)
        mean = run_sum / np.where(n_valid > 0, n_valid, np.nan)
        var = run_sq / np.where(n_valid > 0, n_valid, np.nan) - mean ** 2
        std = np.sqrt(np.clip(var, 0, None))

    dims = tuple(d for d in pool.dims if d != "anchor_year")
    coords = {d: pool[d] for d in dims if d in pool.coords}
    mk = lambda arr, name: xr.DataArray(arr, dims=dims, coords=coords, name=name)
    return SEATestResult(composite=mk(obs, "composite"), p_value=mk(p, "p_value"),
                         null_mean=mk(mean, "null_mean"), null_std=mk(std, "null_std"),
                         n_events=int(n), n_iterations=int(n_iterations))


def seasonal_reducer(lags: np.ndarray, seasons: dict[str, list[int]]):
    """Reducer for :func:`build_anchor_pool` that averages lags into seasons."""
    lags = np.asarray(lags, dtype=int)
    index = {int(lg): i for i, lg in enumerate(lags)}
    picks = [[index[lg] for lg in ls if lg in index]
             for ls in seasons.values()]

    def _reduce(epochs: np.ndarray) -> np.ndarray:
        out = np.full((epochs.shape[0], len(picks)) + epochs.shape[2:],
                      np.nan, dtype="float32")
        for j, cols in enumerate(picks):
            if cols:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    out[:, j] = np.nanmean(epochs[:, cols], axis=1)
        return out
    return _reduce


# ---------------------------------------------------------------------
# Field significance
# ---------------------------------------------------------------------
def fdr_threshold(p: np.ndarray | xr.DataArray, alpha_fdr: float = 0.10) -> float:
    """Benjamini-Hochberg critical p-value for a field of tests.

    Returns the largest p that is still rejected; 0.0 if nothing is.
    Following Wilks (2016), alpha_FDR = 2 * alpha_global is a sensible
    default for spatially correlated geophysical fields.
    """
    arr = np.asarray(p.values if isinstance(p, xr.DataArray) else p, dtype="float64")
    vals = np.sort(arr[np.isfinite(arr)])
    if vals.size == 0:
        return 0.0
    n = vals.size
    crit = alpha_fdr * np.arange(1, n + 1) / n
    passed = np.nonzero(vals <= crit)[0]
    return float(vals[passed[-1]]) if passed.size else 0.0


def fdr_mask(p: xr.DataArray, alpha_fdr: float = 0.10) -> xr.DataArray:
    """Boolean 'locally significant after FDR control' mask."""
    thr = fdr_threshold(p, alpha_fdr)
    out = p <= thr if thr > 0 else xr.zeros_like(p, dtype=bool)
    out.attrs["fdr_threshold"] = thr
    out.attrs["alpha_fdr"] = alpha_fdr
    return out


# ---------------------------------------------------------------------
# Multi-model ensemble
# ---------------------------------------------------------------------
def sign_agreement(ens: xr.DataArray, dim: str = "model") -> xr.DataArray:
    """Fraction of members sharing the sign of the ensemble mean."""
    mean = ens.mean(dim, skipna=True)
    same = (np.sign(ens) == np.sign(mean)).where(ens.notnull())
    return same.sum(dim) / ens.notnull().sum(dim)


def robust_mask(ens: xr.DataArray, p_value: xr.DataArray | None = None,
                dim: str = "model", agreement_fraction: float = 0.66,
                min_models: int = 3, alpha: float = 0.05,
                alpha_fdr: float | None = 0.10) -> xr.DataArray:
    """Robust = enough members, agreeing on sign, and significant.

    This is the mask that decides what gets stippled in the paper's
    maps, so it is deliberately strict.
    """
    n = ens.notnull().sum(dim)
    ok = (n >= min_models) & (sign_agreement(ens, dim) >= agreement_fraction)
    if p_value is not None:
        sig = fdr_mask(p_value, alpha_fdr) if alpha_fdr else (p_value < alpha)
        ok = ok & sig
    ok.attrs.update({"agreement_fraction": agreement_fraction,
                     "min_models": min_models})
    return ok


# ---------------------------------------------------------------------
# Variance partitioning
# ---------------------------------------------------------------------
def variance_partition(ens: xr.DataArray, model_dim: str = "model",
                       event_dim: str = "event") -> xr.Dataset:
    """Two-way ANOVA decomposition of a (model x event) response array.

    Splits the total variance of the composite response into the part
    explained by the impact model, the part explained by which event it
    was, and the residual (model-event interaction plus internal
    variability). This is what tells the reader whether the spread in
    the answer is a modelling problem or genuine event-to-event
    diversity.
    """
    grand = ens.mean([model_dim, event_dim], skipna=True)
    m_mean = ens.mean(event_dim, skipna=True)
    e_mean = ens.mean(model_dim, skipna=True)
    n_m = ens.sizes[model_dim]
    n_e = ens.sizes[event_dim]

    ss_model = (n_e * (m_mean - grand) ** 2).sum(model_dim)
    ss_event = (n_m * (e_mean - grand) ** 2).sum(event_dim)
    ss_total = ((ens - grand) ** 2).sum([model_dim, event_dim])
    ss_resid = (ss_total - ss_model - ss_event).clip(min=0)

    denom = ss_total.where(ss_total > 0)
    return xr.Dataset({
        "frac_model": ss_model / denom,
        "frac_event": ss_event / denom,
        "frac_residual": ss_resid / denom,
        "ss_total": ss_total,
    }, attrs={"description": "two-way ANOVA variance partition of the "
                             "composite response"})


# ---------------------------------------------------------------------
# Nonlinearity
# ---------------------------------------------------------------------
def quadratic_response(y: xr.DataArray, oni: xr.DataArray,
                       dim: str = "time") -> xr.Dataset:
    """Fit ``y = b0 + b1*ONI + b2*ONI^2`` per grid cell.

    A significantly positive ``b2`` where ``b1`` is also positive means
    the response steepens with event amplitude -- the quantitative
    statement behind "super El Ninos do more than a scaled-up strong
    event".
    """
    x1 = oni
    x2 = oni ** 2
    X = xr.concat([xr.ones_like(x1), x1, x2], dim="coef").transpose(dim, "coef")

    def _fit(Y, XX):
        # Y: (n,), XX: (n, 3)
        ok = np.isfinite(Y) & np.isfinite(XX).all(axis=1)
        out = np.full(3, np.nan)
        r2 = np.nan
        if ok.sum() > 10:
            beta, *_ = np.linalg.lstsq(XX[ok], Y[ok], rcond=None)
            out = beta
            resid = Y[ok] - XX[ok] @ beta
            sst = ((Y[ok] - Y[ok].mean()) ** 2).sum()
            r2 = 1 - (resid ** 2).sum() / sst if sst > 0 else np.nan
        return out, r2

    beta, r2 = xr.apply_ufunc(
        _fit, y, X,
        input_core_dims=[[dim], [dim, "coef"]],
        output_core_dims=[["coef"], []],
        vectorize=True, dask="parallelized",
        output_dtypes=[np.float64, np.float64])
    beta = beta.assign_coords(coef=["b0", "b1", "b2"])
    return xr.Dataset({"beta": beta, "r2": r2})


def nonlinearity_metric(super_comp: xr.DataArray, ref_comp: xr.DataArray,
                        super_amp: float, ref_amp: float) -> xr.DataArray:
    """Super composite minus the amplitude-scaled reference composite.

    ``ref_comp * (super_amp / ref_amp)`` is what a purely linear
    teleconnection would predict for a super event given the response to
    moderate/strong events. The residual is the nonlinear excess.
    """
    scale = float(super_amp) / float(ref_amp)
    out = super_comp - scale * ref_comp
    out.attrs.update({"long_name": "nonlinear excess of the super-El-Nino "
                                   "response over a linearly scaled reference",
                      "scaling_factor": scale,
                      "units": super_comp.attrs.get("units", "sigma")})
    return out


def bootstrap_ci(samples: np.ndarray, n_boot: int = 10000, ci: float = 90,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of a small sample."""
    x = np.asarray(samples, dtype="float64")
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))
