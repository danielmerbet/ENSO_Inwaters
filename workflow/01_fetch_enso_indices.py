#!/usr/bin/env python3
"""Step 01 - download the SST-based ENSO indices.

Sources (all public, no registration):

* NOAA PSL ``nino34.long.data``     - Nino3.4 SST, ERSSTv5, 1854-present
* NOAA PSL ``nino3``/``nino4``      - needed for the EP/CP flavour indices
* NOAA CPC ``detrend.nino34.ascii`` - operational Nino3.4/ONI, 1950-present
* HadISST Nino3.4                   - independent SST product (sensitivity)

The raw files are written verbatim to ``data/raw/enso/`` together with a
``sources.json`` recording the URL, retrieval time and SHA-256 of each,
so the provenance of the index is auditable.

Nothing is fabricated: if a host is unreachable the script fails with
instructions for staging the file by hand.

Run:  python workflow/01_fetch_enso_indices.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import requests
from _common import step_setup

from enso_inwaters.utils import file_sha256, write_json

# name -> (url key in config, filename on disk, required?)
TARGETS = {
    "nino34_ersstv5":  ("enso.sources.psl_nino34_raw",    "nino34.long.data",       True),
    "nino34_anom":     ("enso.sources.psl_nino34_ersstv5", "nina34.anom.data",      False),
    "nino34_cpc":      ("enso.sources.cpc_nino34",        "detrend.nino34.ascii.txt", False),
    "nino34_hadisst":  ("enso.sources.hadisst_nino34",    "nino34.long.anom.data",  False),
}
# Nino3 and Nino4 are needed for the EP/CP classification.
EXTRA_URLS = {
    "nino3": "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino3.long.data",
    "nino4": "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino4.long.data",
}


def fetch(url: str, dest: Path, log, timeout: int = 60) -> bool:
    if dest.exists():
        log.info("already present: %s", dest.name)
        return True
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": "ENSO_Inwaters/0.1"})
        r.raise_for_status()
    except Exception as exc:
        log.warning("could not fetch %s (%s: %s)", url, type(exc).__name__, exc)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    log.info("downloaded %s (%d bytes)", dest.name, len(r.content))
    return True


def main() -> int:
    cfg, log, args = step_setup("01_fetch_enso_indices", __doc__)
    out_dir = cfg.path("raw", "enso", mkdir=True)

    urls = {name: (cfg[key], fname, req)
            for name, (key, fname, req) in TARGETS.items()}
    urls.update({k: (u, Path(u).name, k in ("nino3", "nino4"))
                 for k, u in EXTRA_URLS.items()})

    manifest, missing_required = {}, []
    for name, (url, fname, required) in urls.items():
        dest = out_dir / fname
        got = fetch(url, dest, log) if not args.dry_run else dest.exists()
        if got and dest.exists():
            manifest[name] = {
                "url": url, "file": str(dest.relative_to(cfg.root)),
                "sha256": file_sha256(dest), "bytes": dest.stat().st_size,
                "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        elif required:
            missing_required.append((name, url, dest))

    if manifest and not args.dry_run:
        write_json(manifest, out_dir / "sources.json")
        log.info("wrote %s", out_dir / "sources.json")

    if missing_required:
        log.error("=" * 70)
        log.error("Required ENSO index files are missing. Stage them by hand:")
        for _name, url, dest in missing_required:
            log.error("  curl -L -o %s \\\n       %s", dest, url)
        log.error("Then re-run this step (it skips files already on disk).")
        log.error("For an offline dry run of the pipeline use the synthetic "
                  "dataset: python tests/make_synthetic_dataset.py")
        log.error("=" * 70)
        return 1

    log.info("retrieved %d ENSO index file(s) into %s", len(manifest), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
