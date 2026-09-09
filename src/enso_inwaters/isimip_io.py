"""ISIMIP3a file naming, discovery and netCDF I/O conventions.

ISIMIP3a output files follow::

    <model>_<climate-forcing>_<climate-scenario>_<soc-scenario>_
    <sens-scenario>_<variable>_<region>_<time-step>_<start>_<end>.nc

e.g. ``cwatm_gswp3-w5e5_obsclim_histsoc_default_dis_global_monthly_1901_2019.nc``

Everything in this module is filename-driven so the same code reads the
real archive, a local mirror, or the synthetic test dataset.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .utils import provenance

FILENAME_RE = re.compile(
    r"^(?P<model>[^_]+)_(?P<forcing>[^_]+)_(?P<climate_scenario>[^_]+)_"
    r"(?P<soc_scenario>[^_]+)_(?P<sens_scenario>[^_]+)_(?P<variable>[^_]+)_"
    r"(?P<region>[^_]+)_(?P<timestep>[^_]+)_(?P<start>\d{4})_(?P<end>\d{4})\.nc$"
)


def build_filename(model: str, variable: str, forcing: str = "gswp3-w5e5",
                   climate_scenario: str = "obsclim", soc_scenario: str = "histsoc",
                   sens_scenario: str = "default", region: str = "global",
                   timestep: str = "monthly", start: int = 1901,
                   end: int = 2019) -> str:
    return (f"{model}_{forcing}_{climate_scenario}_{soc_scenario}_"
            f"{sens_scenario}_{variable}_{region}_{timestep}_{start}_{end}.nc")


def parse_filename(path: str | Path) -> dict:
    m = FILENAME_RE.match(Path(path).name)
    if not m:
        raise ValueError(f"not an ISIMIP-style filename: {Path(path).name}")
    d = m.groupdict()
    d["start"], d["end"] = int(d["start"]), int(d["end"])
    return d


def find_files(root: str | Path, model: str = "*", variable: str = "*",
               climate_scenario: str = "*", soc_scenario: str = "*",
               forcing: str = "*", timestep: str = "*",
               recursive: bool = True) -> list[Path]:
    """Locate all files matching a selection, sorted by start year."""
    pattern = (f"{model}_{forcing}_{climate_scenario}_{soc_scenario}_*_"
               f"{variable}_*_{timestep}_*.nc")
    base = Path(root)
    hits = base.rglob(pattern) if recursive else base.glob(pattern)
    files = sorted(hits, key=lambda p: (p.name))
    return [p for p in files if FILENAME_RE.match(p.name)]


def open_model_data(root: str | Path, model: str, variable: str,
                    climate_scenario: str = "obsclim",
                    soc_scenario: str = "histsoc",
                    forcing: str = "gswp3-w5e5",
                    timestep: str = "monthly",
                    chunks: dict | None = None,
                    period: tuple[int, int] | None = None) -> xr.DataArray:
    """Open one (model, variable, scenario) combination as a DataArray.

    Multi-decade files are concatenated along time. Time axes are
    normalised to month-start timestamps so that different models'
    calendars (noleap, 360_day, proleptic_gregorian) become comparable.
    """
    files = find_files(root, model=model, variable=variable,
                       climate_scenario=climate_scenario,
                       soc_scenario=soc_scenario, forcing=forcing,
                       timestep=timestep)
    if not files:
        raise FileNotFoundError(
            f"no files for model={model} variable={variable} "
            f"{climate_scenario}/{soc_scenario} under {root}")
    # cftime keeps noleap/360-day calendars intact; normalise_time() then
    # collapses them onto a common monthly index.
    open_kw = dict(combine="by_coords", chunks=chunks or {},
                   decode_timedelta=False)
    try:  # xarray >= 2025.x
        open_kw["decode_times"] = xr.coders.CFDatetimeCoder(use_cftime=True)
    except AttributeError:  # pragma: no cover - older xarray
        open_kw["use_cftime"] = True
    ds = xr.open_mfdataset([str(f) for f in files], **open_kw)
    if variable not in ds:
        # some models name the variable differently inside the file
        candidates = [v for v in ds.data_vars if v.lower() == variable.lower()]
        if not candidates:
            raise KeyError(f"variable {variable!r} not in {files[0].name}; "
                           f"file contains {list(ds.data_vars)}")
        variable = candidates[0]
    da = ds[variable]
    da = normalise_time(da)
    if period:
        da = da.sel(time=slice(f"{period[0]}-01-01", f"{period[1]}-12-31"))
    da = normalise_coords(da)
    da.attrs.setdefault("model", model)
    da.attrs["climate_scenario"] = climate_scenario
    da.attrs["soc_scenario"] = soc_scenario
    return da.rename(variable)


def normalise_time(da: xr.DataArray) -> xr.DataArray:
    """Snap any monthly calendar onto pandas month-start timestamps."""
    t = da["time"].values
    try:
        years = np.array([x.year for x in t])
        months = np.array([x.month for x in t])
    except AttributeError:
        ti = pd.DatetimeIndex(t)
        years, months = ti.year.values, ti.month.values
    new = pd.to_datetime({"year": years, "month": months,
                          "day": np.ones_like(years)})
    return da.assign_coords(time=("time", new))


def normalise_coords(da: xr.DataArray) -> xr.DataArray:
    """Standard coordinate names, -180..180 longitudes, ascending latitude."""
    ren = {}
    for old, new in (("latitude", "lat"), ("longitude", "lon"),
                     ("Lat", "lat"), ("Lon", "lon")):
        if old in da.dims or old in da.coords:
            ren[old] = new
    if ren:
        da = da.rename(ren)
    if "lon" in da.coords and float(da.lon.max()) > 180.0:
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon")
    if "lat" in da.coords and da.lat.size > 1 and float(da.lat[0]) > float(da.lat[-1]):
        da = da.sortby("lat")
    return da


# ---------------------------------------------------------------------
# Unit handling
# ---------------------------------------------------------------------
SECONDS_PER_DAY = 86400.0


def to_mm_per_day(da: xr.DataArray) -> xr.DataArray:
    """Convert a flux in kg m-2 s-1 to mm day-1 (1 kg m-2 = 1 mm of water)."""
    units = (da.attrs.get("units") or "").replace(" ", "")
    if units in ("kgm-2s-1", "kg/m2/s", "kgm**-2s**-1"):
        out = da * SECONDS_PER_DAY
    elif units in ("mmday-1", "mm/day", "mmd-1"):
        out = da.copy()
    else:
        out = da.copy()
        out.attrs["unit_conversion"] = f"left unchanged (units={units!r})"
        return out
    out.attrs.update(da.attrs)
    out.attrs["units"] = "mm day-1"
    return out


def kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
    units = (da.attrs.get("units") or "").strip().upper()
    if units in ("K", "KELVIN"):
        out = da - 273.15
        out.attrs.update(da.attrs)
        out.attrs["units"] = "degC"
        return out
    return da


# ---------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------
def save_netcdf(obj: xr.Dataset | xr.DataArray, path: str | Path, step: str,
                cfg=None, complevel: int = 4, **extra_attrs) -> Path:
    """Write with compression and full provenance attributes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = obj.to_dataset() if isinstance(obj, xr.DataArray) else obj
    ds = ds.copy()
    ds.attrs.update(provenance(step, cfg, **extra_attrs))
    enc = {}
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.floating):
            enc[v] = {"zlib": True, "complevel": complevel,
                      "dtype": "float32", "_FillValue": np.float32(np.nan)}
        elif np.issubdtype(ds[v].dtype, np.bool_):
            ds[v] = ds[v].astype("int8")
            enc[v] = {"zlib": True, "complevel": complevel, "dtype": "int8"}
    ds.to_netcdf(path, encoding=enc)
    return path


def save_table(df: pd.DataFrame, path: str | Path, step: str, cfg=None) -> Path:
    """CSV plus a sibling ``.meta.json`` recording how it was made."""
    from .utils import write_json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    write_json(provenance(step, cfg, n_rows=len(df),
                          columns=list(df.columns)),
               path.with_suffix(".meta.json"))
    return path


def concat_models(arrays: dict[str, xr.DataArray]) -> xr.DataArray:
    """Stack per-model arrays along a ``model`` dimension (outer-joined)."""
    names = sorted(arrays)
    aligned = xr.align(*[arrays[n] for n in names], join="outer")
    return xr.concat(aligned, dim=pd.Index(names, name="model"))
