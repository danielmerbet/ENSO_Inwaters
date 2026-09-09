#!/usr/bin/env python3
"""Step 00 - verify the environment before anything expensive runs.

Checks the Python packages, the directory layout, free disk space and
whether the external data hosts are reachable. It never fails on a
missing *optional* package; it fails loudly on a missing required one,
because discovering that halfway through a 400 GB download is worse.

Run:  python workflow/00_check_environment.py
"""

from __future__ import annotations

import importlib
import shutil
import socket
import sys
from urllib.parse import urlparse

from _common import step_setup

REQUIRED = ["numpy", "pandas", "xarray", "netCDF4", "scipy", "matplotlib", "yaml"]
OPTIONAL = {
    "dask": "parallel/out-of-core processing of the 0.5-degree fields",
    "cartopy": "map projections in the figures",
    "geopandas": "reading basin and lake polygons",
    "regionmask": "rasterising polygons onto the analysis grid",
    "statsmodels": "regression diagnostics",
    "requests": "downloading the input data",
    "isimip_client": "querying the ISIMIP repository API",
}
HOSTS = [
    ("ISIMIP repository API", "https://data.isimip.org"),
    ("ISIMIP file server", "https://files.isimip.org"),
    ("NOAA PSL (Nino3.4)", "https://psl.noaa.gov"),
    ("NOAA CPC (ONI)", "https://origin.cpc.ncep.noaa.gov"),
]
# Rough on-disk footprint of the full ISIMIP3a selection at 0.5 deg monthly.
DISK_ESTIMATE_GB = {
    "water_global (5 models x 7 vars x 2 climate x 2 soc)": 260,
    "lakes_global (5 models x 4 vars)": 90,
    "climate forcing (pr, tas, rsds)": 40,
    "interim anomalies + indices": 180,
    "processed composites + results": 20,
}


def check_packages(log) -> bool:
    ok = True
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            log.info("required  %-14s OK  %s", mod, getattr(m, "__version__", ""))
        except ImportError:
            log.error("required  %-14s MISSING", mod)
            ok = False
    for mod, why in OPTIONAL.items():
        try:
            m = importlib.import_module(mod)
            log.info("optional  %-14s OK  %s", mod, getattr(m, "__version__", ""))
        except ImportError:
            log.warning("optional  %-14s missing - needed for: %s", mod, why)
    return ok


def check_hosts(log) -> None:
    log.info("--- network reachability (data hosts) ---")
    for name, url in HOSTS:
        host = urlparse(url).netloc
        try:
            socket.setdefaulttimeout(8)
            socket.getaddrinfo(host, 443)
            import requests
            r = requests.head(url, timeout=15, allow_redirects=True)
            log.info("%-24s reachable (HTTP %s)", name, r.status_code)
        except Exception as exc:
            log.warning("%-24s NOT reachable: %s", name, type(exc).__name__)
            log.warning("    -> steps 01/03/04 must be run where %s is "
                        "accessible, or the files staged manually.", host)


def check_disk(cfg, log) -> None:
    total = sum(DISK_ESTIMATE_GB.values())
    usage = shutil.disk_usage(cfg.root)
    free_gb = usage.free / 1e9
    log.info("--- disk ---")
    for what, gb in DISK_ESTIMATE_GB.items():
        log.info("  %-52s ~%4d GB", what, gb)
    log.info("  %-52s ~%4d GB", "TOTAL (full analysis)", total)
    log.info("free on %s: %.0f GB", cfg.root, free_gb)
    if free_gb < total:
        log.warning("less free space than the full analysis needs. Options: "
                    "restrict isimip.sectors/models in the config, drop the "
                    "counterclim/nosoc scenarios, or stage data per variable.")


def main() -> int:
    cfg, log, args = step_setup("00_check_environment", __doc__)
    log.info("--- python ---")
    log.info("%s", sys.version.replace("\n", " "))
    log.info("--- packages ---")
    ok = check_packages(log)

    log.info("--- directories ---")
    for key in ("raw", "interim", "processed", "figures", "tables", "logs"):
        p = cfg.path(key, mkdir=True)
        log.info("%-10s %s", key, p)

    check_hosts(log)
    check_disk(cfg, log)

    if not ok:
        log.error("required packages missing - install with "
                  "`conda env create -f environment.yml` or `pip install -r requirements.txt`")
        return 1
    log.info("environment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
