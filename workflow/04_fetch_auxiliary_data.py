#!/usr/bin/env python3
"""Step 04 - auxiliary and observational datasets.

Some of these need a (free) registration or a manual click-through, so
this step downloads what it can and prints exact instructions for the
rest instead of failing silently. Every dataset lands under
``data/raw/`` with a ``sources.json`` next to it.

  basins       GRDC Major River Basins of the World (polygons)
  lakes        HydroLAKES v1.0 polygons + Global Lake Database depths
  population   HYDE 3.2 gridded population, 1901-2017 (exposure)
  land mask    ISIMIP 0.5-degree land-sea mask
  GRDC         monthly station discharge (evaluation of the models)
  GRACE/-FO    JPL mascon terrestrial water storage, 2002- (evaluation)
  ESA CCI      satellite lake surface water temperature (evaluation)
  ERA5-Land    monthly runoff/soil moisture (independent cross-check)

Run:  python workflow/04_fetch_auxiliary_data.py
"""

from __future__ import annotations

from _common import step_setup

from enso_inwaters.download import download_file
from enso_inwaters.utils import file_sha256, write_json

# Datasets that can be fetched without credentials.
DIRECT = {
    "landmask": (
        "https://files.isimip.org/ISIMIP3a/InputData/geo_conditions/landseamask/"
        "landseamask.nc", "masks/landseamask_0.5deg.nc"),
}

# Datasets requiring a manual step: (why, landing page, expected local path)
MANUAL = {
    "basins": (
        "GRDC Major River Basins of the World - free download after a "
        "one-click acceptance of the data policy.",
        "https://grdc.bafg.de/products/basin_layers/major_river_basins/",
        "basins/mrb_basins.gpkg"),
    "lakes": (
        "HydroLAKES v1.0 polygons (~600 MB zipped).",
        "https://www.hydrosheds.org/products/hydrolakes",
        "lakes/HydroLAKES_polys_v10.gdb"),
    "population": (
        "HYDE 3.2 gridded population density, needed for the exposure "
        "numbers. Regrid to 0.5 degrees with the helper in step 15.",
        "https://doi.org/10.17026/dans-25g-gez3",
        "population/hyde32_popd_halfdeg.nc"),
    "grdc": (
        "GRDC monthly discharge time series. Request the stations with "
        ">= 40 years of record; export as netCDF or CSV.",
        "https://portal.grdc.bafg.de/",
        "obs/grdc_monthly.nc"),
    "grace": (
        "GRACE/GRACE-FO JPL RL06.1M mascon terrestrial water storage "
        "(2002-), used to evaluate the simulated TWS response to the "
        "2015/16 event. Free Earthdata login.",
        "https://podaac.jpl.nasa.gov/dataset/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.1_V3",
        "obs/grace_jpl_mascon.nc"),
    "lswt": (
        "ESA CCI Lakes v2 lake surface water temperature, used to "
        "evaluate the simulated lake response.",
        "https://climate.esa.int/en/projects/lakes/data/",
        "obs/esacci_lakes_lswt.nc"),
    "era5_land": (
        "ERA5-Land monthly means (runoff, volumetric soil water). "
        "Retrieve with the CDS API; a request template is in "
        "docs/data_sources.md.",
        "https://cds.climate.copernicus.eu/",
        "obs/era5_land_monthly.nc"),
}


def main() -> int:
    cfg, log, args = step_setup("04_fetch_auxiliary_data", __doc__)
    raw = cfg.path("raw", mkdir=True)
    manifest = {}

    for name, (url, rel) in DIRECT.items():
        dest = raw / rel
        if dest.exists():
            log.info("already present: %s", rel)
        elif args.dry_run:
            log.info("would download %s -> %s", url, rel)
            continue
        else:
            try:
                download_file(url, dest)
                log.info("downloaded %s", rel)
            except Exception as exc:
                log.warning("could not fetch %s: %s", name, exc)
                continue
        if dest.exists():
            manifest[name] = {"url": url, "file": rel,
                              "sha256": file_sha256(dest)}

    missing = []
    for name, (why, page, rel) in MANUAL.items():
        dest = raw / rel
        if dest.exists():
            log.info("already present: %s", rel)
            manifest[name] = {"url": page, "file": rel, "manual": True}
        else:
            missing.append((name, why, page, dest))

    if manifest and not args.dry_run:
        write_json(manifest, raw / "auxiliary_sources.json")

    if missing:
        log.warning("=" * 72)
        log.warning("%d dataset(s) need a manual download:", len(missing))
        for name, why, page, dest in missing:
            log.warning("")
            log.warning("  [%s] %s", name, why)
            log.warning("      page: %s", page)
            log.warning("      save to: %s", dest)
        log.warning("")
        log.warning("The core composite analysis (steps 05-10) runs without "
                    "them. Basin/lake aggregation (11-12), exposure (15) and "
                    "evaluation (16) need the corresponding files.")
        log.warning("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
