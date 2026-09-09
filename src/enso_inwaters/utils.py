"""Logging, provenance and small shared helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(name: str, log_dir: str | Path = "logs",
                  level: int = logging.INFO) -> logging.Logger:
    """Console + file logger. One log file per workflow step."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter(LOG_FORMAT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / f"{name}.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def git_revision(repo: str | Path = ".") -> str:
    """Short git hash, or 'unknown' outside a repository."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        rev = out.stdout.strip()
        if not rev:
            return "unknown"
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:  # pragma: no cover - environment dependent
        return "unknown"


def provenance(step: str, cfg=None, **extra: Any) -> dict:
    """Attributes stamped onto every netCDF/CSV the workflow writes."""
    meta = {
        "history": f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                   f"created by ENSO_Inwaters step {step}",
        "source": "ENSO_Inwaters workflow",
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "step": step,
    }
    if cfg is not None and getattr(cfg, "source", None):
        meta["config_file"] = str(cfg.source)
        meta["config_sha256"] = file_sha256(cfg.source)[:16]
    meta.update({k: str(v) for k, v in extra.items()})
    return meta


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


class Timer:
    """``with Timer(logger, 'label'):`` -> logs elapsed wall time."""

    def __init__(self, logger: logging.Logger, label: str):
        self.logger, self.label = logger, label

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info("START %s", self.label)
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        self.logger.info("DONE  %s (%.1f s)", self.label, dt)
        return False


def require(condition: bool, message: str) -> None:
    """Fail loudly and early rather than producing a silently wrong figure."""
    if not condition:
        raise RuntimeError(message)
