"""Shared boilerplate for the numbered workflow scripts.

Every script starts with::

    from _common import step_setup
    cfg, log, args = step_setup("07_event_composites", __doc__)

which puts ``src/`` on the path, parses ``--config`` and the common
flags, and opens a per-step log file under ``logs/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from enso_inwaters import load_config  # noqa: E402
from enso_inwaters.utils import Timer, setup_logging  # noqa: E402,F401


def build_parser(description: str = "") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="YAML configuration (default: config/config.yaml)")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute outputs that already exist")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be done without writing outputs")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def step_setup(step: str, description: str = "", extra=None):
    """Parse arguments, load the config and open the step's log."""
    parser = build_parser(description)
    if extra:
        extra(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    import logging
    log = setup_logging(step, cfg.path("logs", mkdir=True),
                        level=logging.DEBUG if args.verbose else logging.INFO)
    log.info("configuration: %s", cfg.source)
    log.info("repository root: %s", cfg.root)
    if args.dry_run:
        log.info("DRY RUN - no outputs will be written")
    return cfg, log, args


# ---------------------------------------------------------------------
# Canonical output locations (single source of truth for the file layout)
# ---------------------------------------------------------------------
def anomaly_path(cfg, sector: str, model: str, variable: str,
                 climate_scenario: str, soc_scenario: str) -> Path:
    return cfg.path("interim", "anomalies", sector,
                    f"{model}_{variable}_{climate_scenario}_{soc_scenario}_anom.nc")


def index_path(cfg, sector: str, model: str, variable: str, index: str,
               scale: int, climate_scenario: str, soc_scenario: str) -> Path:
    return cfg.path("interim", "indices", sector,
                    f"{model}_{variable}_{index}{scale}_"
                    f"{climate_scenario}_{soc_scenario}.nc")


def composite_path(cfg, variable: str, event_class: str,
                   climate_scenario: str = "obsclim",
                   soc_scenario: str = "histsoc") -> Path:
    return cfg.path("processed", "composites",
                    f"composite_{variable}_{event_class}_"
                    f"{climate_scenario}_{soc_scenario}.nc")


def result_path(cfg, *parts: str) -> Path:
    return cfg.path("processed", *parts)


def table_path(cfg, name: str) -> Path:
    return cfg.path("tables", name)


def figure_path(cfg, name: str) -> Path:
    return cfg.path("figures", name)


def skip_existing(path: Path, args, log) -> bool:
    """True when ``path`` exists and the user did not ask to overwrite."""
    if path.exists() and not args.overwrite:
        log.info("exists, skipping (use --overwrite): %s", path.name)
        return True
    return False
