"""Spatial masks, area weighting and regional aggregation.

Two aggregation targets matter for the paper:

* **River basins** -- area-weighted means of the gridded fields, plus the
  discharge at the basin outlet cell (the quantity a water manager
  actually sees). Basin polygons come from the GRDC Major River Basins
  dataset; any polygon or raster-ID source works.
* **Lakes** -- ISIMIP ``lakes_global`` is gridded, so each lake is
  represented by the cells its polygon covers, weighted by lake area
  fraction from HydroLAKES.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

EARTH_RADIUS_M = 6_371_000.0


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------
def cell_area(lat: np.ndarray | xr.DataArray, lon: np.ndarray | xr.DataArray
              ) -> xr.DataArray:
    """Grid-cell area in m^2 for a regular lat/lon grid."""
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    dlat = np.abs(np.diff(lat)).mean() if lat.size > 1 else 1.0
    dlon = np.abs(np.diff(lon)).mean() if lon.size > 1 else 1.0
    dlam = np.deg2rad(dlon)
    area_lat = (EARTH_RADIUS_M ** 2) * dlam * (
        np.sin(np.deg2rad(lat + dlat / 2)) - np.sin(np.deg2rad(lat - dlat / 2)))
    area = np.repeat(area_lat[:, None], lon.size, axis=1)
    return xr.DataArray(np.abs(area), dims=("lat", "lon"),
                        coords={"lat": lat, "lon": lon}, name="cell_area",
                        attrs={"units": "m2"})


def area_weights(da: xr.DataArray) -> xr.DataArray:
    """Cosine-latitude weights aligned to ``da``."""
    return cell_area(da["lat"].values, da["lon"].values)


def global_mean(da: xr.DataArray, mask: xr.DataArray | None = None) -> xr.DataArray:
    """Area-weighted spatial mean over (optionally masked) land."""
    w = area_weights(da)
    if mask is not None:
        w = w.where(mask)
    return da.weighted(w.fillna(0)).mean(("lat", "lon"), skipna=True)


def land_fraction_mask(path: str | Path | None, like: xr.DataArray,
                       threshold: float = 0.5) -> xr.DataArray:
    """Land mask from file, or inferred from where the field is defined."""
    if path and Path(path).exists():
        ds = xr.open_dataset(path)
        var = next(iter(ds.data_vars))
        from .isimip_io import normalise_coords
        m = normalise_coords(ds[var])
        m = m.interp(lat=like["lat"], lon=like["lon"], method="nearest")
        return (m > threshold).rename("land_mask")
    inferred = like.notnull()
    if "time" in inferred.dims:
        inferred = inferred.any("time")
    return inferred.rename("land_mask")


# ---------------------------------------------------------------------
# Region masks
# ---------------------------------------------------------------------
def masks_from_polygons(gdf, like: xr.DataArray, name_field: str = "RIVER_BASI",
                        id_field: str | None = None) -> xr.DataArray:
    """Rasterise polygons onto the analysis grid as an integer ID field.

    Uses ``regionmask`` when available (handles antimeridian-crossing
    polygons correctly); falls back to ``rasterio.features.rasterize``.
    """
    lat = like["lat"].values
    lon = like["lon"].values
    try:
        import regionmask
        names = gdf[name_field].astype(str).tolist()
        numbers = (gdf[id_field].astype(int).tolist() if id_field
                   else list(range(len(gdf))))
        regions = regionmask.Regions(list(gdf.geometry), numbers=numbers,
                                     names=names, name="basins")
        m = regions.mask(lon, lat)
        m = m.rename({"lon": "lon", "lat": "lat"})
    except ImportError:  # pragma: no cover - optional dependency
        from affine import Affine
        from rasterio import features
        dlat = float(np.diff(lat).mean())
        dlon = float(np.diff(lon).mean())
        transform = Affine.translation(lon[0] - dlon / 2, lat[0] - dlat / 2) * \
            Affine.scale(dlon, dlat)
        shapes = ((geom, i) for i, geom in enumerate(gdf.geometry))
        arr = features.rasterize(shapes, out_shape=(lat.size, lon.size),
                                 transform=transform, fill=-1, dtype="int32")
        m = xr.DataArray(np.where(arr < 0, np.nan, arr), dims=("lat", "lon"),
                         coords={"lat": lat, "lon": lon})
    m.name = "region_id"
    m.attrs["region_names"] = ";".join(gdf[name_field].astype(str).tolist())
    return m


def masks_from_raster(path: str | Path, like: xr.DataArray,
                      var: str | None = None) -> xr.DataArray:
    """Load a pre-rasterised region-ID field and align it to the grid."""
    from .isimip_io import normalise_coords
    ds = xr.open_dataset(path)
    var = var or next(iter(ds.data_vars))
    m = normalise_coords(ds[var])
    return m.interp(lat=like["lat"], lon=like["lon"], method="nearest"
                    ).rename("region_id")


def aggregate_regions(da: xr.DataArray, region_id: xr.DataArray,
                      names: dict[int, str] | None = None,
                      weights: xr.DataArray | None = None,
                      how: str = "mean") -> xr.DataArray:
    """Area-weighted aggregation of ``da`` over each region ID.

    Returns an array with a ``region`` dimension labelled by name.
    """
    w = weights if weights is not None else area_weights(da)
    ids = np.unique(region_id.values[np.isfinite(region_id.values)]).astype(int)
    out, labels = [], []
    for rid in ids:
        sel = region_id == rid
        ww = w.where(sel)
        if float(ww.sum()) == 0:
            continue
        if how == "mean":
            val = da.where(sel).weighted(ww.fillna(0)).mean(("lat", "lon"), skipna=True)
        elif how == "sum":
            val = (da.where(sel) * ww).sum(("lat", "lon"), skipna=True)
        elif how == "max":
            val = da.where(sel).max(("lat", "lon"), skipna=True)
        else:
            raise ValueError(how)
        out.append(val)
        labels.append(names.get(int(rid), f"region_{rid}") if names
                      else f"region_{rid}")
    if not out:
        raise ValueError("no regions overlapped the field")
    return xr.concat(out, dim=pd.Index(labels, name="region"))


def outlet_cells(discharge_clim: xr.DataArray, region_id: xr.DataArray
                 ) -> pd.DataFrame:
    """Locate each basin's outlet as its cell of maximum mean discharge.

    Crude but robust, and independent of any particular flow-direction
    map -- which matters because the ISIMIP models do not share one.
    """
    rows = []
    ids = np.unique(region_id.values[np.isfinite(region_id.values)]).astype(int)
    for rid in ids:
        sel = discharge_clim.where(region_id == rid)
        if not bool(sel.notnull().any()):
            continue
        flat = sel.stack(cell=("lat", "lon"))
        k = int(flat.argmax("cell", skipna=True))
        rows.append({"region_id": int(rid),
                     "lat": float(flat["lat"][k]),
                     "lon": float(flat["lon"][k]),
                     "mean_discharge": float(flat[k])})
    return pd.DataFrame(rows)


def lake_cell_mask(lake_area_frac: xr.DataArray, min_fraction: float = 0.05
                   ) -> xr.DataArray:
    """Cells with enough lake area for the lake models to be meaningful."""
    return (lake_area_frac >= min_fraction).rename("lake_mask")


# ---------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------
def exposed_population(mask: xr.DataArray, population: xr.DataArray) -> float:
    """People living in cells flagged by ``mask`` (population per cell)."""
    return float(population.where(mask).sum())


def exposed_area_km2(mask: xr.DataArray) -> float:
    a = area_weights(mask.astype("float64"))
    return float(a.where(mask).sum()) / 1e6
