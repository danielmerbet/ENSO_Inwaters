"""Tests for the inference machinery.

These are the tests that matter most: a bug here would produce a
plausible-looking but wrong significance map, which is far harder to
catch by eye than a broken figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enso_inwaters import stats as S  # noqa: E402

EVENT_YEARS = [1972, 1982, 1997, 2015]


def _field(signal_at_lag0: float, n_sites: int = 4, seed: int = 0,
           noise: float = 1.0):
    """Noise everywhere, an injected DJF signal at site 0 only."""
    t = pd.date_range("1901-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(seed)
    x = rng.normal(0, noise, (len(t), n_sites))
    ti = pd.DatetimeIndex(t)
    for y in EVENT_YEARS:
        k = np.nonzero((ti.year == y) & (ti.month == 12))[0][0]
        x[k:k + 3, 0] += signal_at_lag0
    return xr.DataArray(x, dims=("time", "site"),
                        coords={"time": t,
                                "site": [f"s{i}" for i in range(n_sites)]})


def _direct_composite(da: xr.DataArray, years, lag: int) -> np.ndarray:
    """Composite computed the obvious way, to check the fast path."""
    ti = pd.DatetimeIndex(da["time"].values)
    rows = []
    for y in years:
        anchor = pd.Timestamp(year=int(y), month=12, day=1) + \
            pd.DateOffset(months=lag)
        k = np.nonzero((ti.year == anchor.year) & (ti.month == anchor.month))[0]
        if k.size:
            rows.append(da.values[k[0]])
    return np.mean(rows, axis=0)


def test_eligible_anchor_years_excludes_a_buffer():
    years = np.arange(1950, 1961)
    keep = S.eligible_anchor_years(years, np.array([1955]), buffer_years=1)
    assert 1955 not in keep and 1954 not in keep and 1956 not in keep
    assert 1953 in keep and 1957 in keep


def test_mc_composite_equals_the_direct_computation():
    """The pooled fast path must give exactly the naive composite."""
    da = _field(2.5)
    pool, _ = S.build_anchor_pool(da, np.arange(1903, 2017), np.arange(-18, 31))
    res = S.mc_composite_test(pool, EVENT_YEARS, n_iterations=50, seed=1)
    for lag in (-6, 0, 2, 12):
        assert np.allclose(res.composite.sel(lag=lag).values,
                           _direct_composite(da, EVENT_YEARS, lag), atol=1e-5)


def test_mc_composite_test_detects_a_real_signal():
    # noise=0.3 keeps the composite standard error at ~0.15, so this
    # tests the machinery rather than one lucky random draw
    da = _field(2.5, noise=0.3)
    pool, _ = S.build_anchor_pool(da, np.arange(1903, 2017), np.arange(-18, 31))
    res = S.mc_composite_test(pool, EVENT_YEARS, n_iterations=2000, seed=1)
    assert float(res.composite.sel(lag=0, site="s0")) == pytest.approx(2.5, abs=0.5)
    assert float(res.p_value.sel(lag=0, site="s0")) < 0.01
    assert abs(float(res.composite.sel(lag=-12, site="s0"))) < 0.5
    assert float(res.p_value.sel(lag=-12, site="s0")) > 0.05
    assert res.n_events == 4


def test_mc_composite_test_does_not_cry_wolf_on_noise():
    """A pure-noise field must not be significant more often than alpha."""
    da = _field(0.0, n_sites=40, seed=5)
    pool, _ = S.build_anchor_pool(da, np.arange(1903, 2017), np.arange(-6, 7))
    res = S.mc_composite_test(pool, EVENT_YEARS, n_iterations=2000, seed=2)
    false_rate = float((res.p_value < 0.05).mean())
    assert false_rate < 0.12, f"false-positive rate {false_rate:.3f} too high"
    # and FDR control must cut it down further
    assert float(S.fdr_mask(res.p_value, 0.10).mean()) <= false_rate


def test_seasonal_reducer_matches_a_direct_mean():
    da = _field(2.5)
    lags = np.arange(-18, 31)
    seasons = {"JJA0": [-6, -5, -4], "DJF01": [0, 1, 2]}
    full, _ = S.build_anchor_pool(da, np.arange(1903, 2017), lags)
    red, _ = S.build_anchor_pool(da, np.arange(1903, 2017), lags,
                                 reducer=S.seasonal_reducer(lags, seasons))
    direct = full.sel(lag=[0, 1, 2]).mean("lag")
    assert np.allclose(red[:, 1].values, direct.values, equal_nan=True, atol=1e-5)


def test_mc_test_rejects_too_few_neutral_years():
    da = _field(1.0)
    pool, _ = S.build_anchor_pool(da, np.array([1972, 1982, 1997, 2015, 1990]),
                                  np.arange(-3, 4))
    with pytest.raises(ValueError, match="neutral anchor years"):
        S.mc_composite_test(pool, EVENT_YEARS, n_iterations=10, seed=0)


def test_fdr_threshold_is_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.02, 0.2, 0.6])
    # BH at alpha=0.1: critical values 0.02, 0.04, 0.06, 0.08, 0.10
    assert S.fdr_threshold(p, 0.10) == pytest.approx(0.02)
    assert S.fdr_threshold(np.array([0.5, 0.6, 0.7]), 0.10) == 0.0


def test_sign_agreement_and_robust_mask():
    ens = xr.DataArray(np.array([[1.0], [1.0], [1.0], [-1.0]]),
                       dims=("model", "x"))
    assert float(S.sign_agreement(ens).isel(x=0)) == pytest.approx(0.75)
    p = xr.DataArray([0.001], dims="x")
    assert bool(S.robust_mask(ens, p, agreement_fraction=0.66,
                              min_models=3, alpha_fdr=None).isel(x=0))
    # too few models -> not robust, however significant
    assert not bool(S.robust_mask(ens.isel(model=slice(0, 2)), p,
                                  min_models=3, alpha_fdr=None).isel(x=0))


def test_variance_partition_sums_to_one():
    rng = np.random.default_rng(0)
    ens = xr.DataArray(rng.normal(size=(4, 5, 7)),
                       dims=("model", "event", "x"))
    vp = S.variance_partition(ens)
    total = (vp["frac_model"] + vp["frac_event"] + vp["frac_residual"])
    assert np.allclose(total.values, 1.0, atol=1e-9)


def test_variance_partition_attributes_a_model_offset():
    """A field that differs only between models must load onto frac_model."""
    base = np.zeros((4, 5, 3))
    base += np.array([0.0, 5.0, 10.0, 15.0])[:, None, None]
    ens = xr.DataArray(base, dims=("model", "event", "x"))
    vp = S.variance_partition(ens)
    assert float(vp["frac_model"].mean()) == pytest.approx(1.0, abs=1e-9)


def test_nonlinearity_metric_scales_the_reference():
    sup = xr.DataArray([2.0], dims="x")
    ref = xr.DataArray([1.0], dims="x")
    out = S.nonlinearity_metric(sup, ref, super_amp=2.4, ref_amp=1.2)
    assert float(out.isel(x=0)) == pytest.approx(0.0)   # exactly linear
    assert out.attrs["scaling_factor"] == pytest.approx(2.0)


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 1.0, 40)
    lo, hi = S.bootstrap_ci(x, n_boot=4000, ci=90, seed=0)
    assert lo < x.mean() < hi
    assert hi - lo < 1.0
