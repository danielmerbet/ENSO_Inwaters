"""Tests for geometry, aggregation and the ISIMIP file conventions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enso_inwaters import isimip_io as IO  # noqa: E402
from enso_inwaters import regions as R  # noqa: E402

EARTH_AREA_KM2 = 510.1e6


def _grid(res: float = 2.0):
    lat = np.arange(-90 + res / 2, 90, res)
    lon = np.arange(-180 + res / 2, 180, res)
    return lat, lon


def test_cell_areas_sum_to_the_earths_surface():
    for res in (0.5, 1.0, 2.0, 5.0):
        lat, lon = _grid(res)
        total = float(R.cell_area(lat, lon).sum()) / 1e6
        assert total == pytest.approx(EARTH_AREA_KM2, rel=0.002)


def test_area_weighted_mean_of_a_constant_is_that_constant():
    lat, lon = _grid()
    da = xr.DataArray(np.full((lat.size, lon.size), 7.0), dims=("lat", "lon"),
                      coords={"lat": lat, "lon": lon})
    assert float(R.global_mean(da)) == pytest.approx(7.0)


def test_area_weighting_downweights_high_latitudes():
    """An unweighted mean would be badly wrong on a lat/lon grid."""
    lat, lon = _grid()
    values = np.where(np.abs(lat)[:, None] > 60, 10.0, 0.0)
    da = xr.DataArray(np.broadcast_to(values, (lat.size, lon.size)).copy(),
                      dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    weighted = float(R.global_mean(da))
    unweighted = float(da.mean())
    assert weighted < unweighted
    # poleward of 60 degrees is ~13.4% of the sphere
    assert weighted == pytest.approx(10 * 0.134, abs=0.1)


def test_aggregate_regions_recovers_per_region_values():
    lat, lon = _grid()
    lat2d = np.broadcast_to(lat[:, None], (lat.size, lon.size))
    rid = xr.DataArray(np.where(lat2d > 0, 1.0, 2.0), dims=("lat", "lon"),
                       coords={"lat": lat, "lon": lon})
    da = xr.DataArray(np.where(lat2d > 0, 3.0, -1.0), dims=("lat", "lon"),
                      coords={"lat": lat, "lon": lon})
    agg = R.aggregate_regions(da, rid, names={1: "north", 2: "south"})
    assert list(agg["region"].values) == ["north", "south"]
    assert float(agg.sel(region="north")) == pytest.approx(3.0)
    assert float(agg.sel(region="south")) == pytest.approx(-1.0)


def test_outlet_cell_is_the_maximum_discharge_cell():
    lat, lon = _grid(10.0)
    field = np.zeros((lat.size, lon.size))
    field[5, 7] = 99.0
    da = xr.DataArray(field, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    rid = xr.DataArray(np.ones_like(field), dims=("lat", "lon"),
                       coords={"lat": lat, "lon": lon})
    out = R.outlet_cells(da, rid)
    assert float(out.iloc[0]["lat"]) == pytest.approx(float(lat[5]))
    assert float(out.iloc[0]["lon"]) == pytest.approx(float(lon[7]))


def test_exposed_population_and_area():
    lat, lon = _grid(10.0)
    mask = xr.DataArray(np.zeros((lat.size, lon.size), dtype=bool),
                        dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    mask[9, 9] = True
    pop = xr.DataArray(np.full((lat.size, lon.size), 1000.0),
                       dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    assert R.exposed_population(mask, pop) == pytest.approx(1000.0)
    assert 0 < R.exposed_area_km2(mask) < EARTH_AREA_KM2


# --- ISIMIP file conventions ------------------------------------------
def test_filename_roundtrip():
    name = IO.build_filename("cwatm", "dis", climate_scenario="counterclim",
                             soc_scenario="nosoc")
    meta = IO.parse_filename(name)
    assert meta["model"] == "cwatm"
    assert meta["variable"] == "dis"
    assert meta["climate_scenario"] == "counterclim"
    assert meta["soc_scenario"] == "nosoc"
    assert meta["start"] == 1901 and meta["end"] == 2019


def test_parse_filename_rejects_a_foreign_name():
    with pytest.raises(ValueError):
        IO.parse_filename("some_random_file.nc")


def test_normalise_coords_rewraps_longitudes_and_flips_latitude():
    lat = np.arange(89.5, -90, -1.0)         # descending
    lon = np.arange(0.5, 360, 1.0)           # 0..360
    da = xr.DataArray(np.zeros((lat.size, lon.size)),
                      dims=("latitude", "longitude"),
                      coords={"latitude": lat, "longitude": lon})
    out = IO.normalise_coords(da)
    assert "lat" in out.dims and "lon" in out.dims
    assert float(out["lat"][0]) < float(out["lat"][-1])
    assert float(out["lon"].min()) >= -180 and float(out["lon"].max()) <= 180


def test_unit_conversions():
    da = xr.DataArray([1.0], dims="x", attrs={"units": "kg m-2 s-1"})
    mm = IO.to_mm_per_day(da)
    assert float(mm[0]) == pytest.approx(86400.0)
    assert mm.attrs["units"] == "mm day-1"

    k = xr.DataArray([300.0], dims="x", attrs={"units": "K"})
    c = IO.kelvin_to_celsius(k)
    assert float(c[0]) == pytest.approx(26.85)
    assert c.attrs["units"] == "degC"

    # an unrecognised unit must be left alone rather than silently scaled
    other = xr.DataArray([5.0], dims="x", attrs={"units": "m3 s-1"})
    assert float(IO.to_mm_per_day(other)[0]) == 5.0


def test_normalise_time_collapses_calendars():
    import cftime
    times = [cftime.DatetimeNoLeap(2000, m, 16) for m in range(1, 13)]
    da = xr.DataArray(np.arange(12.0), dims="time", coords={"time": times})
    out = IO.normalise_time(da)
    idx = pd.DatetimeIndex(out["time"].values)
    assert list(idx.month) == list(range(1, 13))
    assert set(idx.day) == {1}


def test_concat_models_aligns_on_the_union():
    a = xr.DataArray([1.0, 2.0], dims="x", coords={"x": [0, 1]}, name="v")
    b = xr.DataArray([3.0, 4.0], dims="x", coords={"x": [1, 2]}, name="v")
    out = IO.concat_models({"m1": a, "m2": b})
    assert list(out["model"].values) == ["m1", "m2"]
    assert list(out["x"].values) == [0, 1, 2]
    assert np.isnan(float(out.sel(model="m1", x=2)))
