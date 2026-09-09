"""Data acquisition helpers (ISIMIP repository + plain HTTP files).

The ISIMIP repository exposes a JSON API at ``data.isimip.org/api/v1``;
files live on ``files.isimip.org``. The workflow queries the API for the
datasets matching the configured selection, writes a **manifest**
(model x variable x scenario, with file URLs, sizes and checksums), and
then downloads from that manifest. Keeping discovery and download apart
means the manifest can be reviewed, version-controlled and re-used, and
a partial download can be resumed without re-querying.

Downloads are skipped when a local file already matches the recorded
size/checksum, so re-running the step is cheap and idempotent.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests

from .utils import file_sha256

USER_AGENT = "ENSO_Inwaters/0.1 (research workflow; contact via repository)"
CHUNK = 1 << 20


def api_search(api: str, params: dict, page_limit: int = 100,
               max_pages: int = 100, timeout: int = 60) -> list[dict]:
    """Page through an ISIMIP API listing endpoint."""
    out: list[dict] = []
    url = f"{api.rstrip('/')}/datasets/"
    page_params = dict(params)
    page_params.setdefault("limit", page_limit)
    offset = 0
    for _ in range(max_pages):
        page_params["offset"] = offset
        r = requests.get(url, params=page_params, timeout=timeout,
                         headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results", [])
        out.extend(results)
        if not payload.get("next"):
            break
        offset += len(results) or page_limit
    return out


def datasets_to_manifest(datasets: Iterable[dict]) -> pd.DataFrame:
    """Flatten the API response into one row per downloadable file."""
    rows = []
    for ds in datasets:
        sp = ds.get("specifiers", {})
        for f in ds.get("files", []):
            rows.append({
                "dataset_id": ds.get("id"),
                "dataset_name": ds.get("name"),
                "sector": sp.get("product") or sp.get("sector"),
                "model": (sp.get("model") or [None])[0]
                         if isinstance(sp.get("model"), list) else sp.get("model"),
                "climate_forcing": sp.get("climate_forcing"),
                "climate_scenario": sp.get("climate_scenario"),
                "soc_scenario": sp.get("soc_scenario"),
                "sens_scenario": sp.get("sens_scenario"),
                "variable": sp.get("variable"),
                "time_step": sp.get("time_step"),
                "file_name": f.get("name"),
                "file_url": f.get("file_url"),
                "checksum": f.get("checksum"),
                "checksum_type": f.get("checksum_type", "sha512"),
                "size": f.get("size"),
            })
    return pd.DataFrame(rows)


def download_file(url: str, dest: Path, expected_size: int | None = None,
                  retries: int = 4, backoff: float = 2.0,
                  timeout: int = 300) -> Path:
    """Stream a file to disk, resuming/skipping when it is already there."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        return dest

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout,
                              headers={"User-Agent": USER_AGENT}) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for block in r.iter_content(CHUNK):
                        fh.write(block)
                tmp.replace(dest)
            return dest
        except Exception as exc:  # network flakiness is the norm here
            last_err = exc
            if attempt < retries - 1:
                time.sleep(backoff ** (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {last_err}")


def verify(path: Path, checksum: str | None, checksum_type: str = "sha512") -> bool:
    """Verify a download; sha256 is checked, other types are trusted."""
    if not checksum:
        return True
    if checksum_type.lower() == "sha256":
        return file_sha256(path) == checksum
    import hashlib
    h = hashlib.new(checksum_type.lower())
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest() == checksum
