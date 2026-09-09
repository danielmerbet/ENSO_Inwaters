"""Tests for anomalies and the standardised indices."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enso_inwaters import climatology as C  # noqa: E402


@pytest.fixture
def series():
    t = pd.date_range("1901-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(3)
    seasonal = 2.0 * np.sin(2 * np.pi * t.month.values / 12)
    trend = 0.01 * (t.year.values - 1901)
    values = 10 + seasonal + trend + rng.normal(0, 1, len(t))
    return xr.DataArray(values, dims="time", coords={"time": t}, name="x")


def test_moving_climatology_matches_a_direct_window(series):
    """The cumulative-sum implementation must equal the naive computation."""
    mean, std = C.moving_climatology(series, window_years=30, min_years=20)
    t = pd.DatetimeIndex(series["time"].values)
    for year, month in ((1960, 6), (1980, 1), (2000, 11)):
        k = np.nonzero((t.year == year) & (t.month == month))[0][0]
        win = (t.year >= year - 15) & (t.year < year + 15) & (t.month == month)
        assert float(mean[k]) == pytest.approx(np.nanmean(series.values[win]), rel=1e-9)
        assert float(std[k]) == pytest.approx(
            np.nanstd(series.values[win], ddof=1), rel=1e-9)


def test_standardized_anomalies_are_unit_variance_and_deseasonalised(series):
    anom = C.anomalies(series, method="standardized", climatology="moving_30yr")
    assert abs(float(anom.mean())) < 0.05
    assert float(anom.std()) == pytest.approx(1.0, abs=0.1)
    monthly = anom.groupby("time.month").mean()
    assert float(np.abs(monthly).max()) < 0.2, "seasonal cycle not removed"


def test_moving_climatology_removes_the_trend(series):
    anom = C.anomalies(series, method="standardized", climatology="moving_30yr")
    early = float(anom.sel(time=slice("1920", "1940")).mean())
    late = float(anom.sel(time=slice("1995", "2015")).mean())
    assert abs(late - early) < 0.2


def test_anomalies_reject_too_short_a_record():
    t = pd.date_range("2000-01-01", "2005-12-01", freq="MS")
    da = xr.DataArray(np.arange(len(t), dtype=float), dims="time",
                      coords={"time": t})
    with pytest.raises(ValueError, match="years available"):
        C.anomalies(da, climatology="moving_30yr")


def test_linear_detrend_removes_a_known_slope():
    t = pd.date_range("1950-01-01", "2000-12-01", freq="MS")
    da = xr.DataArray(np.arange(len(t), dtype=float) * 0.5, dims="time",
                      coords={"time": t})
    out = C.linear_detrend(da)
    assert float(np.abs(out).max()) < 1e-8


def test_standardized_index_is_standard_normal():
    t = pd.date_range("1901-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(11)
    q = xr.DataArray(rng.gamma(2.0, 3.0, len(t)), dims="time",
                     coords={"time": t}, name="q")
    sri = C.standardized_index(q, scale=3, distribution="gamma")
    vals = sri.values[np.isfinite(sri.values)]
    assert abs(vals.mean()) < 0.1
    assert vals.std() == pytest.approx(1.0, abs=0.1)
    # roughly 16% of a standard normal lies below -1
    assert 0.10 < (vals < -1).mean() < 0.22


def test_accumulate_is_a_trailing_mean():
    t = pd.date_range("2000-01-01", periods=12, freq="MS")
    da = xr.DataArray(np.arange(12, dtype=float), dims="time", coords={"time": t})
    acc = C.accumulate(da, 3)
    assert np.isnan(acc[0]) and np.isnan(acc[1])
    assert float(acc[2]) == pytest.approx(1.0)
    assert float(acc[11]) == pytest.approx(10.0)
