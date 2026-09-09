#!/usr/bin/env python3
"""Step 03 - discover and download the ISIMIP3a simulations.

Two phases, deliberately separated:

**discover** queries ``data.isimip.org/api/v1`` for every combination of
sector x model x variable x climate/soc scenario listed in the config and
writes ``data/raw/isimip_manifest.csv`` (one row per file, with URL, size
and checksum) plus an availability matrix. Review that matrix before
committing to the download - ISIMIP3a coverage is uneven, and knowing
which model is missing ``qr`` or ``groundwstor`` up front changes what
the paper can claim.

**download** streams every file in the manifest into
``data/raw/isimip/<sector>/``, skipping files already present with the
right size and verifying checksums. It is restartable.

Examples
--------
    python workflow/03_fetch_isimip3a.py --discover-only
    python workflow/03_fetch_isimip3a.py --sector water_global
    python workflow/03_fetch_isimip3a.py --max-files 5      # smoke test
"""

from __future__ import annotations

import pandas as pd
from _common import step_setup, table_path

from enso_inwaters import download as D
from enso_inwaters.isimip_io import save_table


def add_args(p):
    p.add_argument("--sector", default=None,
                   help="restrict to one sector (default: all in the config)")
    p.add_argument("--discover-only", action="store_true",
                   help="build the manifest without downloading")
    p.add_argument("--download-only", action="store_true",
                   help="use the existing manifest, skip the API query")
    p.add_argument("--max-files", type=int, default=None,
                   help="stop after N files (smoke test)")


def discover(cfg, log, sectors) -> pd.DataFrame:
    api = cfg["isimip.api"]
    frames = []
    for sector, spec in sectors.items():
        for model in spec["models"]:
            for variable in spec["variables"]:
                for clim in cfg["isimip.climate_scenario"]:
                    for soc in cfg["isimip.soc_scenario"]:
                        params = {
                            "simulation_round": cfg["isimip.simulation_round"],
                            "product": "OutputData",
                            "sector": sector,
                            "model": model,
                            "climate_forcing": cfg["isimip.climate_forcing"][0],
                            "climate_scenario": clim,
                            "soc_scenario": soc,
                            "variable": variable,
                            "time_step": cfg["isimip.time_step"],
                        }
                        try:
                            found = D.api_search(api, params)
                        except Exception as exc:
                            log.warning("API query failed for %s/%s/%s %s/%s: %s",
                                        sector, model, variable, clim, soc, exc)
                            continue
                        if not found:
                            log.debug("no data: %s %s %s %s/%s",
                                      sector, model, variable, clim, soc)
                            continue
                        df = D.datasets_to_manifest(found)
                        df["sector"] = sector
                        df["query_model"] = model
                        df["query_variable"] = variable
                        df["query_climate_scenario"] = clim
                        df["query_soc_scenario"] = soc
                        frames.append(df)
                        log.info("%-13s %-16s %-12s %s/%-8s -> %d file(s)",
                                 sector, model, variable, clim, soc, len(df))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("file_url")


def availability_matrix(man: pd.DataFrame) -> pd.DataFrame:
    """model x variable x scenario coverage - the table to eyeball first."""
    if man.empty:
        return man
    return (man.groupby(["sector", "query_model", "query_variable",
                         "query_climate_scenario", "query_soc_scenario"])
               .agg(n_files=("file_url", "size"),
                    total_gb=("size", lambda s: round(s.fillna(0).sum() / 1e9, 2)))
               .reset_index())


def main() -> int:
    cfg, log, args = step_setup("03_fetch_isimip3a", __doc__, extra=add_args)
    raw = cfg.path("raw", mkdir=True)
    manifest_path = raw / "isimip_manifest.csv"

    sectors = dict(cfg["isimip.sectors"])
    if args.sector:
        sectors = {args.sector: sectors[args.sector]}

    if args.download_only:
        if not manifest_path.exists():
            log.error("no manifest at %s - run the discovery phase first",
                      manifest_path)
            return 1
        man = pd.read_csv(manifest_path)
    else:
        log.info("querying the ISIMIP repository API ...")
        man = discover(cfg, log, sectors)
        if man.empty:
            log.error("the API returned nothing. Either the host is "
                      "unreachable from here, or the selection in the config "
                      "matches no ISIMIP3a data. Check %s manually.",
                      cfg["isimip.api"])
            return 1
        save_table(man, manifest_path, "03_fetch_isimip3a", cfg)
        avail = availability_matrix(man)
        save_table(avail, table_path(cfg, "table_S2_isimip_availability.csv"),
                   "03_fetch_isimip3a", cfg)
        log.info("manifest: %d files, %.1f GB total",
                 len(man), man["size"].fillna(0).sum() / 1e9)

        # flag gaps against the variables the analysis needs
        for sector, spec in sectors.items():
            have = set(man.loc[man.sector == sector, "query_variable"])
            for req in spec.get("required_variables", []):
                if req not in have:
                    log.warning("required variable %r is unavailable for "
                                "sector %s - the analysis will drop it",
                                req, sector)
        for sector, spec in sectors.items():
            sub = man[man.sector == sector]
            for model in spec["models"]:
                got = sorted(set(sub.loc[sub.query_model == model, "query_variable"]))
                missing = [v for v in spec["variables"] if v not in got]
                if missing:
                    log.warning("%s/%s missing: %s", sector, model, missing)

    if args.discover_only:
        log.info("discovery complete; review %s and %s before downloading",
                 manifest_path.name, "table_S2_isimip_availability.csv")
        return 0

    # ---------------- download ------------------------------------------
    rows = man if args.max_files is None else man.head(args.max_files)
    total = len(rows)
    ok = failed = 0
    for i, row in enumerate(rows.itertuples(), start=1):
        dest = raw / "isimip" / str(row.sector) / str(row.file_name)
        if args.dry_run:
            log.info("[%d/%d] would download %s", i, total, row.file_name)
            continue
        try:
            D.download_file(row.file_url, dest,
                            expected_size=int(row.size) if pd.notna(row.size) else None)
            if not D.verify(dest, getattr(row, "checksum", None),
                            getattr(row, "checksum_type", "sha512") or "sha512"):
                log.error("checksum mismatch: %s (removed)", dest.name)
                dest.unlink(missing_ok=True)
                failed += 1
                continue
            ok += 1
            log.info("[%d/%d] %s", i, total, row.file_name)
        except Exception as exc:
            failed += 1
            log.error("[%d/%d] FAILED %s: %s", i, total, row.file_name, exc)

    log.info("downloaded %d file(s), %d failure(s)", ok, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
