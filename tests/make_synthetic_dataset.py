#!/usr/bin/env python3
"""Build a small SYNTHETIC dataset that mimics the real inputs.

Purpose: exercise every step of the workflow end-to-end -- offline, in
about a minute -- so that the code can be validated before committing
hundreds of gigabytes and days of download to the real ISIMIP3a archive.

**These are not observations and not model output.** The files are
written with ``synthetic`` in their global attributes and land in
``data/raw/`` only when the synthetic configuration is used. Nothing
produced from them may appear in the paper.

What is imposed, so that the analysis has a known right answer:

* a Nino3.4 series with prescribed events, including four "super" ones
  (peak > 2 degC) and a warming trend that the sliding base period must
  remove;
* a fixed spatial teleconnection pattern (dry over the maritime
  continent / Amazon / southern Africa / Australia, wet over the
  southern US, eastern equatorial Pacific coast and the Parana);
* a variable-dependent response lag -- runoff responds within a month,
  discharge after 1-2, recharge after 3, storage after 5 -- which step 10
  must recover;
* a quadratic term, so the response to a super event is ~30% larger than
  a linear scaling of the moderate response, which step 09 must detect;
* AR(1) noise, a seasonal cycle, and ~30% missing (ocean) cells.

Run:  python tests/make_synthetic_dataset.py [--outdir data]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# (developing year, peak ONI) -- shaped after the historical record but
# entirely synthetic.
# NOTE: the amplitudes below are the *prescribed* SST bumps. The ONI that
# step 02 recovers is 0.3-0.4 degC smaller, because a 30-year base period
# necessarily contains other events and so has a warm December
# climatology -- exactly as in the real CPC index. The values are chosen
# so that 1972, 1982, 1997 and 2015 come out above the 2.0 degC "super"
# threshold after that reduction.
EVENTS = [
    (1905, 1.2), (1911, 1.0), (1918, 2.2), (1925, 2.2), (1930, 1.2),
    (1940, 2.2), (1951, 0.8), (1957, 1.7), (1963, 1.2), (1965, 1.6),
    (1968, 1.1), (1972, 2.6), (1976, 0.9), (1979, 0.6), (1982, 2.7),
    (1986, 1.2), (1987, 1.6), (1991, 1.7), (1994, 1.2), (1997, 2.9),
    (2002, 1.3), (2004, 0.7), (2006, 0.9), (2009, 1.4), (2014, 0.7),
    (2015, 2.6), (2018, 0.9),
    # La Nina (negative peaks)
    (1909, -1.6), (1916, -1.5), (1949, -1.2), (1954, -1.6), (1970, -1.2),
    (1973, -1.9), (1975, -1.6), (1988, -1.8), (1998, -1.6), (1999, -1.7),
    (2007, -1.5), (2010, -1.6), (2011, -1.0), (2017, -0.9),
]

MODELS_WATER = ["cwatm", "h08", "watergap2-2e", "jules-w1"]
MODELS_LAKES = ["albm", "gotm", "simstrat-uog"]

# variable -> (units, response lag in months, response gain, base value)
WATER_VARS = {
    "dis":         ("m3 s-1",      2, 0.9, 500.0),
    "qtot":        ("kg m-2 s-1",  0, 1.0, 2.0e-5),
    "qr":          ("kg m-2 s-1",  3, 0.7, 6.0e-6),
    "tws":         ("kg m-2",      5, 0.6, 300.0),
    "rootmoist":   ("kg m-2",      1, 0.5, 150.0),
    "groundwstor": ("kg m-2",      6, 0.5, 200.0),
}
LAKE_VARS = {
    "surftemp":    ("K",           1, 0.8, 288.0),
    "lakeicefrac": ("1",           1, 0.3, 0.1),
}

NONLINEAR_GAIN = 0.12   # coefficient of the ONI^2 term (the "super" excess)


# ---------------------------------------------------------------------
def synthetic_nino34(times: pd.DatetimeIndex, seed: int = 42) -> pd.Series:
    """Nino3.4 SST: prescribed event bumps + red noise + a warming trend."""
    rng = np.random.default_rng(seed)
    t = np.arange(len(times), dtype="float64")
    anom = np.zeros_like(t)
    for year0, peak in EVENTS:
        centre = np.where((times.year == year0) & (times.month == 12))[0]
        if centre.size == 0:
            continue
        anom += peak * np.exp(-((t - centre[0]) ** 2) / (2 * 4.0 ** 2))
    # AR(1) background so that neutral years are not perfectly flat
    noise = np.zeros_like(t)
    for i in range(1, len(t)):
        noise[i] = 0.75 * noise[i - 1] + rng.normal(0, 0.06)
    trend = 0.010 * (times.year.values - times.year.values[0])   # ~1.2 K/century
    seasonal = 0.4 * np.sin(2 * np.pi * (times.month.values - 3) / 12)
    sst = 27.0 + anom + noise + trend + seasonal
    return pd.Series(sst, index=times, name="nino34")


def teleconnection_pattern(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """A smooth, fixed 'El Nino wet/dry' pattern in sigma per degC of ONI."""
    LON, LAT = np.meshgrid(lon, lat)

    def blob(lat0, lon0, dlat, dlon, amp):
        return amp * np.exp(-(((LAT - lat0) / dlat) ** 2 +
                              (((LON - lon0 + 180) % 360 - 180) / dlon) ** 2))

    p = (blob(-2, 120, 18, 30, -0.55)     # maritime continent: dry
         + blob(-5, -60, 15, 22, -0.45)   # Amazon: dry
         + blob(-18, 26, 14, 20, -0.40)   # southern Africa: dry
         + blob(-25, 135, 14, 25, -0.50)  # eastern Australia: dry
         + blob(10, 5, 10, 25, -0.25)     # Sahel: dry
         + blob(32, -100, 12, 22, 0.45)   # southern US: wet
         + blob(-8, -78, 10, 8, 0.55)     # coastal Peru/Ecuador: wet
         + blob(-30, -60, 10, 14, 0.40)   # Parana: wet
         + blob(35, 60, 12, 25, 0.25))    # central Asia: wet
    return xr.DataArray(p, dims=("lat", "lon"),
                        coords={"lat": lat, "lon": lon}, name="pattern")


def land_mask(lat: np.ndarray, lon: np.ndarray, seed: int = 7) -> xr.DataArray:
    """A crude but stable land mask: broad continental blocks."""
    LON, LAT = np.meshgrid(lon, lat)
    land = np.zeros_like(LAT, dtype=bool)
    blocks = [  # (lat0, lat1, lon0, lon1)
        (5, 72, -170, -55), (-56, 12, -82, -35), (35, 71, -10, 60),
        (-35, 37, -18, 52), (5, 75, 60, 180), (-44, -11, 113, 154),
    ]
    for la0, la1, lo0, lo1 in blocks:
        land |= (LAT >= la0) & (LAT <= la1) & (LON >= lo0) & (LON <= lo1)
    land &= ~((LAT > 60) & (LAT < 90) & (LON > -60) & (LON < -20))  # keep it ragged
    return xr.DataArray(land, dims=("lat", "lon"),
                        coords={"lat": lat, "lon": lon}, name="land")


def make_field(oni: np.ndarray, pattern: xr.DataArray, mask: xr.DataArray,
               times: pd.DatetimeIndex, lag: int, gain: float, base: float,
               model_bias: float, seed: int) -> xr.DataArray:
    """base * (1 + response + seasonality + AR(1) noise), masked to land."""
    rng = np.random.default_rng(seed)
    nt = len(times)
    ny, nx = pattern.shape

    oni_lag = np.roll(oni, lag)
    oni_lag[:lag] = 0.0
    # linear + quadratic (the quadratic makes super events over-respond)
    drive = gain * model_bias * (oni_lag + NONLINEAR_GAIN * np.sign(oni_lag) * oni_lag ** 2)

    resp = drive[:, None, None] * pattern.values[None, :, :]

    noise = np.zeros((nt, ny, nx))
    innov = rng.normal(0, 0.35, size=(nt, ny, nx))
    for i in range(1, nt):
        noise[i] = 0.6 * noise[i - 1] + innov[i]

    season = 0.25 * np.sin(2 * np.pi * (times.month.values - 4) / 12)[:, None, None]
    season = season * np.where(pattern.values >= 0, 1.0, -1.0)[None, :, :]

    rel = resp + noise + season
    values = base * (1.0 + rel)
    if base > 0:
        values = np.clip(values, 0.0 if base < 1 else 1e-6, None)
    values = np.where(mask.values[None, :, :], values, np.nan)

    return xr.DataArray(values.astype("float32"), dims=("time", "lat", "lon"),
                        coords={"time": times, "lat": pattern.lat,
                                "lon": pattern.lon})


def write_nc(da: xr.DataArray, name: str, units: str, path: Path,
             model: str, sector: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    da = da.rename(name)
    da.attrs.update({"units": units, "long_name": name,
                     "model": model, "sector": sector})
    ds = da.to_dataset()
    ds.attrs.update({
        "title": "SYNTHETIC test data for the ENSO_Inwaters workflow",
        "synthetic": "yes - generated by tests/make_synthetic_dataset.py",
        "warning": "NOT observations and NOT ISIMIP model output; "
                   "for code testing only",
    })
    enc = {name: {"zlib": True, "complevel": 4, "dtype": "float32"}}
    ds.to_netcdf(path, encoding=enc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="data", help="root data directory")
    ap.add_argument("--start", type=int, default=1901)
    ap.add_argument("--end", type=int, default=2019)
    ap.add_argument("--resolution", type=float, default=4.0,
                    help="grid spacing in degrees (default 4 -> fast)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.outdir)
    times = pd.date_range(f"{args.start}-01-01", f"{args.end}-12-01", freq="MS")
    res = args.resolution
    lat = np.arange(-90 + res / 2, 90, res)
    lon = np.arange(-180 + res / 2, 180, res)
    print(f"grid {lat.size} x {lon.size}, {len(times)} months")

    # ---- ENSO indices -------------------------------------------------
    enso_dir = out / "raw" / "enso"
    enso_dir.mkdir(parents=True, exist_ok=True)
    n34 = synthetic_nino34(times, seed=args.seed)
    pd.DataFrame({"time": times, "nino34": n34.values}).to_csv(
        enso_dir / "nino34.csv", index=False)
    # Nino3 leads / Nino4 lags slightly -> most events classify as EP
    n3 = n34 + 0.25 * (n34 - n34.rolling(6, min_periods=1).mean())
    n4 = 26.5 + 0.55 * (n34 - 27.0)
    pd.DataFrame({"time": times, "nino3": n3.values}).to_csv(
        enso_dir / "nino3.csv", index=False)
    pd.DataFrame({"time": times, "nino4": n4.values}).to_csv(
        enso_dir / "nino4.csv", index=False)
    (enso_dir / "SYNTHETIC.txt").write_text(
        "Synthetic Nino indices generated by tests/make_synthetic_dataset.py.\n"
        "NOT observations. Delete before running the real workflow.\n")
    print(f"wrote synthetic Nino indices to {enso_dir}")

    # ONI used to drive the fields (same construction as step 02)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from enso_inwaters.enso import oni_from_nino34
    oni = oni_from_nino34(n34, method="sliding_30yr")
    oni = oni.reindex(times).fillna(0.0)

    pattern = teleconnection_pattern(lat, lon)
    mask = land_mask(lat, lon)
    print(f"land cells: {int(mask.sum())} / {mask.size}")

    # ---- impact-model output -----------------------------------------
    rng = np.random.default_rng(args.seed)
    n_files = 0
    for sector, models, variables in (
            ("water_global", MODELS_WATER, WATER_VARS),
            ("lakes_global", MODELS_LAKES, LAKE_VARS)):
        for mi, model in enumerate(models):
            bias = float(rng.uniform(0.7, 1.3))     # inter-model spread
            for var, (units, lag, gain, base) in variables.items():
                for clim, soc in (("obsclim", "histsoc"),
                                  ("obsclim", "nosoc"),
                                  ("counterclim", "histsoc")):
                    # counterclim: same ENSO, no warming-related amplification
                    g = gain * (0.85 if clim == "counterclim" else 1.0)
                    # nosoc: no direct human water use -> slightly larger stores
                    b = base * (1.15 if soc == "nosoc" else 1.0)
                    da = make_field(oni.values, pattern, mask, times, lag, g, b,
                                    bias, seed=args.seed + 1000 * mi + hash(var) % 997)
                    fn = (f"{model}_gswp3-w5e5_{clim}_{soc}_default_{var}_"
                          f"global_monthly_{args.start}_{args.end}.nc")
                    write_nc(da, var, units,
                             out / "raw" / "isimip" / sector / fn, model, sector)
                    n_files += 1
    print(f"wrote {n_files} synthetic ISIMIP-style files under "
          f"{out / 'raw' / 'isimip'}")

    # ---- auxiliary ----------------------------------------------------
    aux = out / "raw" / "masks"
    aux.mkdir(parents=True, exist_ok=True)
    mask.astype("int8").rename("landseamask").to_dataset().to_netcdf(
        aux / "landseamask_synthetic.nc")

    # a synthetic basin field and a population field, for steps 11 and 15
    basin_id = xr.DataArray(
        np.where(mask.values,
                 (np.digitize(np.meshgrid(lon, lat)[1], [-30, 0, 23, 45]) * 10
                  + np.digitize(np.meshgrid(lon, lat)[0], [-100, -30, 30, 100])),
                 np.nan),
        dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="basin_id")
    (out / "raw" / "basins").mkdir(parents=True, exist_ok=True)
    basin_id.to_dataset().to_netcdf(
        out / "raw" / "basins" / "basin_id_synthetic.nc")
    pop = xr.DataArray(
        np.where(mask.values,
                 np.abs(np.cos(np.deg2rad(np.meshgrid(lon, lat)[1]))) * 2.0e6
                 * np.random.default_rng(3).random(mask.shape), 0.0),
        dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="population")
    (out / "raw" / "population").mkdir(parents=True, exist_ok=True)
    pop.to_dataset().to_netcdf(out / "raw" / "population" / "population_synthetic.nc")
    print("wrote synthetic land mask, basin IDs and population")

    print("\nSynthetic dataset ready. Run the workflow with:")
    print("  ENSO_INWATERS_CONFIG=config/config_synthetic.yaml bash workflow/run_all.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
