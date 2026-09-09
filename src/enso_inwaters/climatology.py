"""Anomalies, standardisation and drought/flood indices.

Design choices, all configurable in ``config/config.yaml``:

* **Moving 30-year climatology.** Over 1901-2019 both the climate
  forcing and the simulated stores drift. Referencing every month to a
  30-year window centred on it removes that drift without imposing a
  linear trend model, and keeps the ENSO-band variability intact
  (ENSO periods 2-7 yr are far shorter than the 30-yr window).
* **Standardised anomalies** (divide by the moving standard deviation of
  the same calendar month) make grid cells with wildly different
  magnitudes -- Amazon discharge vs. a Sahelian wadi -- comparable, which
  is what a global composite needs.
* **Gamma-fitted standardised indices** (SRI/SSI/SGI) are used for the
  drought and flood statistics because monthly runoff and discharge are
  strongly right-skewed and a Gaussian z-score misstates their tails.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr

try:  # scipy is required for the gamma fit but not for plain anomalies
    from scipy import stats as _sstats
except Exception:  # pragma: no cover
    _sstats = None


# ---------------------------------------------------------------------
# Climatologies and anomalies
# ---------------------------------------------------------------------
def monthly_climatology(da: xr.DataArray, start: int | None = None,
                        end: int | None = None) -> xr.DataArray:
    """Fixed-base-period monthly mean climatology."""
    sub = da
    if start is not None:
        sub = sub.sel(time=slice(f"{start}-01-01", f"{end}-12-31"))
    return sub.groupby("time.month").mean("time", skipna=True)


def moving_climatology(da: xr.DataArray, window_years: int = 30,
                       min_years: int = 20, ddof: int = 1,
                       block_size: int = 8192) -> tuple[xr.DataArray, xr.DataArray]:
    """Centred moving mean and standard deviation per calendar month.

    Each month of the record is referenced to the ``window_years``-year
    window centred on it, computed separately for each calendar month so
    that the trend and the seasonal cycle are removed together.

    Implemented with cumulative sums along the year axis rather than a
    rolling reduction: the cost is O(n_years) instead of
    O(n_years x window), which is what makes this tractable on the full
    0.5-degree grid (~60 000 land cells x 119 years x 12 months). The
    spatial axis is processed in blocks so peak memory stays bounded.

    Returns ``(mean, std)`` on the original time axis; ``ddof=1`` gives
    the sample standard deviation used to standardise the anomalies.
    """
    if "time" not in da.dims:
        raise ValueError("moving_climatology needs a 'time' dimension")
    da = da.transpose("time", ...)
    times = pd.DatetimeIndex(da["time"].values)
    years, months = times.year.values, times.month.values
    uyears = np.arange(int(years.min()), int(years.max()) + 1)
    n_years = uyears.size
    if n_years < min_years:
        raise ValueError(
            f"only {n_years} years available; need >= {min_years} for a "
            f"{window_years}-year moving climatology")

    rest = da.shape[1:]
    n_space = int(np.prod(rest)) if rest else 1
    values = np.asarray(da.values, dtype="float64").reshape(len(times), n_space)

    yi = years - uyears[0]
    mi = months - 1
    grid = np.full((n_years, 12, n_space), np.nan)
    grid[yi, mi] = values

    mean_g = np.empty_like(grid)
    std_g = np.empty_like(grid)

    half = window_years // 2
    idx = np.arange(n_years)
    lo = np.clip(idx - half, 0, n_years)
    hi = np.clip(idx - half + window_years, 0, n_years)

    for b0 in range(0, n_space, block_size):
        b1 = min(b0 + block_size, n_space)
        blk = grid[:, :, b0:b1]
        valid = np.isfinite(blk)
        x0 = np.where(valid, blk, 0.0)
        zeros = np.zeros((1,) + x0.shape[1:])
        cs = np.concatenate([zeros, np.cumsum(x0, axis=0)], axis=0)
        cq = np.concatenate([zeros, np.cumsum(x0 ** 2, axis=0)], axis=0)
        cn = np.concatenate([zeros, np.cumsum(valid, axis=0)], axis=0)

        n = cn[hi] - cn[lo]
        ssum = cs[hi] - cs[lo]
        ssq = cq[hi] - cq[lo]
        enough = n >= min_years
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(enough, ssum / np.where(n > 0, n, np.nan), np.nan)
            var = np.where(enough,
                           (ssq - n * m ** 2) / np.where(n > ddof, n - ddof, np.nan),
                           np.nan)
        mean_g[:, :, b0:b1] = m
        std_g[:, :, b0:b1] = np.sqrt(np.clip(var, 0, None))

    def _back(g):
        arr = g[yi, mi].reshape((len(times),) + rest)
        return xr.DataArray(arr, dims=da.dims,
                            coords={d: da[d] for d in da.dims if d in da.coords})

    return _back(mean_g), _back(std_g)


def anomalies(da: xr.DataArray, method: str = "standardized",
              climatology: str = "moving_30yr", window_years: int = 30,
              min_years: int = 20, base: tuple[int, int] | None = None,
              detrend: str = "none",
              min_std_relative: float = 1e-6) -> xr.DataArray:
    """Deseasonalised anomalies in absolute, percent or sigma units.

    Cells whose month-of-year standard deviation is numerically zero
    (permanent ice, hyper-arid cells with no runoff in a model) are
    masked rather than allowed to explode the standardisation.
    """
    if climatology.startswith("moving"):
        clim, std = moving_climatology(da, window_years, min_years)
    elif climatology == "fixed":
        if base is None:
            raise ValueError("fixed climatology needs a base period")
        clim_m = monthly_climatology(da, *base)
        std_m = da.sel(time=slice(f"{base[0]}-01-01", f"{base[1]}-12-31")
                       ).groupby("time.month").std("time")
        clim = clim_m.sel(month=da["time.month"]).drop_vars("month")
        std = std_m.sel(month=da["time.month"]).drop_vars("month")
    elif climatology == "none":
        clim = xr.zeros_like(da)
        std = xr.ones_like(da)
    else:
        raise ValueError(f"unknown climatology option: {climatology}")

    anom = da - clim

    if detrend == "linear":
        anom = linear_detrend(anom)

    if method == "absolute":
        out = anom
    elif method == "standardized":
        # all-NaN (ocean) cells make numpy complain about the degrees of
        # freedom; they are masked out two lines below anyway.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scale = da.std("time", skipna=True).compute() \
                if hasattr(da.data, "compute") else da.std("time", skipna=True)
        valid = std > (min_std_relative * xr.where(scale > 0, scale, np.nan))
        out = anom / std.where(valid)
    elif method == "percent":
        out = 100.0 * anom / clim.where(np.abs(clim) > 0)
    else:
        raise ValueError(f"unknown anomaly method: {method}")

    out.attrs.update(da.attrs)
    out.attrs["anomaly_method"] = method
    out.attrs["climatology"] = climatology
    out.attrs["units"] = {"standardized": "sigma", "percent": "%"}.get(
        method, da.attrs.get("units", ""))
    return out.rename(da.name)


def linear_detrend(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Remove the full least-squares linear fit along ``dim`` (NaN-safe).

    Both the slope and the intercept are removed, matching
    ``scipy.signal.detrend(type='linear')``, so the result has zero mean.
    In the workflow this runs after the climatology has already been
    subtracted, where the mean is zero anyway.
    """
    x = xr.DataArray(np.arange(da.sizes[dim], dtype="float64"),
                     dims=dim, coords={dim: da[dim]})
    xm = x.where(da.notnull()).mean(dim)
    ym = da.mean(dim, skipna=True)
    cov = ((x - xm) * (da - ym)).mean(dim, skipna=True)
    var = ((x - xm) ** 2).where(da.notnull()).mean(dim)
    slope = cov / var.where(var > 0)
    return da - ym - (slope * (x - xm))


# ---------------------------------------------------------------------
# Standardised drought / flood indices
# ---------------------------------------------------------------------
def accumulate(da: xr.DataArray, scale: int) -> xr.DataArray:
    """Trailing ``scale``-month running mean, as used by SPI-like indices."""
    if scale <= 1:
        return da
    return da.rolling(time=scale, min_periods=scale).mean()


def standardized_index(da: xr.DataArray, scale: int = 1,
                       distribution: str = "gamma") -> xr.DataArray:
    """SRI / SSI / SGI: accumulate, fit per calendar month, map to z.

    ``gamma`` fits a two-parameter gamma to the non-zero values of each
    calendar month and handles the zero-flow mass with the standard
    mixed-distribution correction of Stagge et al. (2015);
    ``empirical`` uses the Gringorten plotting position instead, which
    makes no distributional assumption but is noisier in short records.
    """
    acc = accumulate(da, scale)
    if distribution == "empirical":
        out = acc.groupby("time.month").map(_empirical_z)
    elif distribution == "gamma":
        if _sstats is None:  # pragma: no cover
            raise ImportError("scipy is required for the gamma fit")
        out = acc.groupby("time.month").map(
            lambda g: xr.apply_ufunc(
                _gamma_z_1d, g,
                input_core_dims=[["time"]], output_core_dims=[["time"]],
                vectorize=True, dask="parallelized",
                output_dtypes=[np.float64]))
    else:
        raise ValueError(distribution)
    out = out.sortby("time")
    out.attrs.update({"units": "sigma", "index_scale_months": scale,
                      "distribution": distribution,
                      "long_name": f"standardized index ({scale}-month)"})
    return out


def _empirical_z(g: xr.DataArray) -> xr.DataArray:
    from scipy.stats import norm, rankdata
    def _f(v):
        v = np.asarray(v, dtype="float64")
        ok = np.isfinite(v)
        z = np.full(v.shape, np.nan)
        n = ok.sum()
        if n < 10:
            return z
        r = rankdata(v[ok])
        p = (r - 0.44) / (n + 0.12)          # Gringorten
        z[ok] = norm.ppf(p)
        return z
    return xr.apply_ufunc(_f, g, input_core_dims=[["time"]],
                          output_core_dims=[["time"]], vectorize=True,
                          dask="parallelized", output_dtypes=[np.float64])


def _gamma_z_1d(v: np.ndarray) -> np.ndarray:
    """Gamma-fit one series (single calendar month, one grid cell)."""
    from scipy.stats import gamma, norm
    v = np.asarray(v, dtype="float64")
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.sum() < 10:
        return out
    x = v[ok]
    # Mixed distribution: point mass at zero + gamma on the positive part
    zero = x <= 0
    q = zero.mean()
    pos = x[~zero]
    if pos.size < 5 or np.allclose(pos, pos[0]):
        return out
    try:
        a, loc, scale = gamma.fit(pos, floc=0)
    except Exception:
        return out
    p = np.where(zero, q, q + (1 - q) * gamma.cdf(x, a, loc=loc, scale=scale))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out[ok] = norm.ppf(p)
    return out


def event_counts(index: xr.DataArray, threshold: float,
                 below: bool = True) -> xr.DataArray:
    """Number of months beyond a drought (``below``) or flood threshold."""
    mask = index < threshold if below else index > threshold
    return mask.sum("time").where(index.notnull().any("time"))
