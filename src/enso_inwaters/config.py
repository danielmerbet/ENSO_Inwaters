"""Configuration loading.

A single YAML file drives the whole workflow. Scripts do::

    from enso_inwaters import load_config
    cfg = load_config()                    # config/config.yaml
    cfg = load_config("config/config_synthetic.yaml")

``Config`` is a thin dict wrapper that supports dotted lookups
(``cfg["enso.classes.super"]``) so that scripts stay readable.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = "config/config.yaml"
ENV_VAR = "ENSO_INWATERS_CONFIG"


class Config(Mapping):
    """Read-only mapping with dotted-key access and path helpers."""

    def __init__(self, data: dict, source: Path | None = None):
        self._data = data
        self.source = Path(source) if source else None
        self.root = Path(data.get("paths", {}).get("root", ".")).resolve()

    # -- Mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        node: Any = self._data
        for part in str(key).split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise KeyError(f"config key not found: {key!r}")
            node = node[part]
        return node

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Config(source={self.source}, keys={list(self._data)})"

    # -- Convenience ------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def path(self, key: str, *parts: str, mkdir: bool = False) -> Path:
        """Resolve a configured directory (``paths.*``) plus optional parts."""
        base = self.root / str(self[f"paths.{key}"])
        p = base.joinpath(*parts) if parts else base
        if mkdir:
            (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
        return p

    def as_dict(self) -> dict:
        return copy.deepcopy(self._data)


def _deep_update(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_update(out[k], dict(v))
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike | None = None,
                overrides: dict | None = None) -> Config:
    """Load the YAML configuration.

    Resolution order: explicit ``path`` -> ``$ENSO_INWATERS_CONFIG`` ->
    ``config/config.yaml``. ``overrides`` is deep-merged on top, which is
    how the test suite swaps in the synthetic-data settings.
    """
    if path is None:
        path = os.environ.get(ENV_VAR, DEFAULT_CONFIG)
    path = Path(path)
    if not path.is_absolute():
        # Allow running the scripts from anywhere inside the repo.
        here = Path(__file__).resolve().parents[2]
        candidate = here / path
        path = candidate if candidate.exists() else path
    if not path.exists():
        raise FileNotFoundError(
            f"configuration file not found: {path}. Run from the repository "
            f"root or set ${ENV_VAR}."
        )
    with open(path) as fh:
        data = yaml.safe_load(fh)
    # A config may extend another one with a top-level `_base:` key, so
    # variants (synthetic test data, a cluster run) only state what differs.
    seen = {path.resolve()}
    while isinstance(data, dict) and data.get("_base"):
        base_path = Path(data.pop("_base"))
        if not base_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            for candidate in (path.parent / base_path.name,
                              repo_root / base_path,
                              Path.cwd() / base_path):
                if candidate.exists():
                    base_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"_base config {base_path} referenced by {path} not found")
        base_path = base_path.resolve()
        if base_path in seen:
            raise ValueError(f"circular _base chain at {base_path}")
        seen.add(base_path)
        with open(base_path) as fh:
            base = yaml.safe_load(fh)
        data = _deep_update(base, data)
    if overrides:
        data = _deep_update(data, overrides)
    # paths.root defaults to the repository root, not the CWD
    data.setdefault("paths", {})
    if data["paths"].get("root", ".") == ".":
        data["paths"]["root"] = str(Path(path).resolve().parents[1])
    return Config(data, source=path)
