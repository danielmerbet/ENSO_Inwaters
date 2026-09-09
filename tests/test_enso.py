"""Tests for the ENSO index and event machinery."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enso_inwaters import enso as E  # noqa: E402


def test_season_to_lags_are_exact():
    """Season codes must map to the lags the paper's figures assume."""
    assert E._season_to_lags("JJA0") == [-6, -5, -4]
    assert E._season_to_lags("SON0") == [-3, -2, -1]
    assert E._season_to_lags("DJF01") == [0, 1, 2]
    assert E._season_to_lags("MAM1") == [3, 4, 5]
    assert E._season_to_lags("JJA1") == [6, 7, 8]
    assert E._season_to_lags("DJF12") == [12, 13, 14]


def test_season_label_is_centred_on_the_month():
    assert E.season_label(1) == "DJF"     # January-centred
    assert E.season_label(7) == "JJA"
    assert E.season_label(12) == "NDJ"


def test_sliding_base_period_removes_a_trend():
    """1.2 degC of warming must survive as < 0.3 degC of residual."""
    t = pd.date_range("1901-01-01", "2019-12-01", freq="MS")
    sst = pd.Series(27.0 + 0.01 * (t.year.values - 1901), index=t)
    anom = E.sliding_base_period_anomalies(sst, 30, 5)
    raw_trend = sst.iloc[-1] - sst.iloc[0]
    assert raw_trend > 1.0
    assert anom.max() - anom.min() < 0.3 * raw_trend

    # In the interior of the record the removal is essentially exact ...
    interior = anom[(anom.index.year >= 1930) & (anom.index.year <= 1989)]
    assert abs(interior.mean()) < 0.01
    assert abs(interior).max() < 0.05

    # ... while the ends keep a small, bounded artefact, because the
    # 30-year window there cannot be centred. This is documented
    # behaviour, not a bug: see sliding_base_period_anomalies.
    ends = anom[(anom.index.year < 1916) | (anom.index.year > 2004)]
    assert abs(ends).max() < 0.16
    assert anom[anom.index.year < 1916].mean() < 0        # early deflated
    assert anom[anom.index.year > 2004].mean() > 0        # late inflated


def test_fixed_base_period_keeps_the_trend():
    """The contrast that motivates the sliding base period."""
    t = pd.date_range("1901-01-01", "2019-12-01", freq="MS")
    sst = pd.Series(27.0 + 0.01 * (t.year.values - 1901), index=t)
    anom = E.fixed_base_period_anomalies(sst, 1901, 2019)
    assert anom[anom.index.year > 1990].mean() - \
        anom[anom.index.year < 1930].mean() > 0.5


def _oni_with_events(peaks: dict[int, float]) -> pd.Series:
    t = pd.date_range("1901-01-01", "2019-12-01", freq="MS")
    x = np.zeros(len(t))
    idx = np.arange(len(t))
    for year, amp in peaks.items():
        c = np.nonzero((t.year == year) & (t.month == 12))[0]
        if c.size:
            x += amp * np.exp(-((idx - c[0]) ** 2) / (2 * 4.0 ** 2))
    return pd.Series(x, index=t)


def test_event_detection_and_classification():
    oni = _oni_with_events({1972: 2.2, 1982: 2.4, 1997: 2.6, 2015: 2.5,
                            1965: 1.6, 1991: 1.2, 2004: 0.7})
    events = E.detect_events(oni, threshold=0.5, min_consecutive_seasons=5)
    by_year = {e.year0: e for e in events}

    assert set(by_year) >= {1972, 1982, 1997, 2015, 1965, 1991}
    assert by_year[1997].amplitude_class == "super"
    assert by_year[1965].amplitude_class == "strong"
    assert by_year[1991].amplitude_class == "moderate"
    assert by_year[1972].label == "1972/73"
    # a 0.7 peak lasts fewer than five seasons above 0.5 -> not an event
    assert 2004 not in by_year or by_year[2004].amplitude_class == "weak"


def test_la_nina_detection_is_symmetric():
    oni = -_oni_with_events({1988: 1.9, 1999: 1.7})
    events = E.detect_events(oni, threshold=-0.5, min_consecutive_seasons=5,
                             phase="la_nina")
    assert {e.year0 for e in events} == {1988, 1999}
    assert all(e.peak_oni < 0 for e in events)
    assert all(e.amplitude_class == "strong" for e in events)


def test_event_reference_anchors_on_december():
    assert E.event_reference({"year0": 1997}) == pd.Timestamp("1997-12-01")
    assert E.event_reference({"year0": 1997, "peak_time": "1997-11-01"},
                             anchor="peak") == pd.Timestamp("1997-11-01")


def test_superposed_epoch_recovers_an_injected_signal():
    t = pd.date_range("1901-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.2, len(t))
    ti = pd.DatetimeIndex(t)
    for y in (1972, 1982, 1997, 2015):
        k = np.nonzero((ti.year == y) & (ti.month == 12))[0][0]
        x[k:k + 3] += 2.0
    da = xr.DataArray(x, dims="time", coords={"time": t})
    comp = E.superposed_epoch(
        da, [pd.Timestamp(y, 12, 1) for y in (1972, 1982, 1997, 2015)], -18, 30)
    assert float(comp.sel(lag=0)) == pytest.approx(2.0, abs=0.3)
    assert abs(float(comp.sel(lag=-12))) < 0.3
    assert float(E.season_mean(comp, "DJF01")) == pytest.approx(2.0, abs=0.3)
    assert abs(float(E.season_mean(comp, "JJA0"))) < 0.3


def test_select_event_class():
    df = pd.DataFrame({
        "year0": [1972, 1982, 1991, 1965, 1988],
        "phase": ["el_nino"] * 4 + ["la_nina"],
        "amplitude_class": ["super", "super", "moderate", "strong", "strong"],
        "in_main_sample": [True] * 5,
    })
    assert list(E.select_event_class(df, "super")["year0"]) == [1972, 1982]
    assert set(E.select_event_class(df, "reference")["year0"]) == {1991, 1965}
    assert len(E.select_event_class(df, "all_el_nino")) == 4
    assert len(E.select_event_class(df, "all_la_nina")) == 1
