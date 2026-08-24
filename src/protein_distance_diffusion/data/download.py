"""Small PDB mmCIF downloader with cache-aware manifests."""

from __future__ import annotations

import hashlib
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd


def file_sha256(path: str | Path) -> str:
    """Return a file SHA256 hex digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_cache(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _download_one(
    pdb_id: str,
    output_dir: Path,
    *,
    force: bool,
    retries: int,
    dry_run: bool,
    delay_seconds: float,
    backoff_seconds: float,
) -> dict[str, Any]:
    code = pdb_id.strip().lower()
    dst = output_dir / f"{code}.cif.gz"
    url = f"https://files.rcsb.org/download/{code.upper()}.cif.gz"
    if not force and _valid_cache(dst):
        return {
            "pdb_id": code.upper(),
            "path": str(dst),
            "status": "cached",
            "bytes": dst.stat().st_size,
            "sha256": file_sha256(dst),
        }
    if dry_run:
        return {"pdb_id": code.upper(), "path": str(dst), "status": "dry_run", "bytes": 0, "sha256": None}
    output_dir.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(1, max(1, retries) + 1):
        try:
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            tmp = dst.with_name(f".{dst.name}.tmp")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dst)
            return {
                "pdb_id": code.upper(),
                "path": str(dst),
                "status": "downloaded",
                "bytes": dst.stat().st_size,
                "sha256": file_sha256(dst),
            }
        except Exception as exc:  # pragma: no cover - exact urllib exception type varies.
            last_error = str(exc)
            if attempt < retries and backoff_seconds > 0:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    return {
        "pdb_id": code.upper(),
        "path": str(dst),
        "status": "failed",
        "bytes": 0,
        "sha256": None,
        "error": last_error,
    }


def download_mmcif_ids(
    ids: list[str],
    output_dir: str | Path,
    *,
    force: bool = False,
    max_entries: int | None = None,
    retries: int = 3,
    dry_run: bool = False,
    delay_seconds: float = 0.0,
    workers: int = 4,
    backoff_seconds: float = 1.0,
) -> pd.DataFrame:
    """Download selected PDB IDs as mmCIF.gz files and return a manifest frame."""
    selected = [pdb_id.strip() for pdb_id in ids if pdb_id.strip()]
    if max_entries is not None:
        selected = selected[: int(max_entries)]
    out = Path(output_dir)
    if workers <= 1:
        rows = [
            _download_one(
                pdb_id,
                out,
                force=force,
                retries=retries,
                dry_run=dry_run,
                delay_seconds=delay_seconds,
                backoff_seconds=backoff_seconds,
            )
            for pdb_id in selected
        ]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            rows = list(
                executor.map(
                    lambda pdb_id: _download_one(
                        pdb_id,
                        out,
                        force=force,
                        retries=retries,
                        dry_run=dry_run,
                        delay_seconds=delay_seconds,
                        backoff_seconds=backoff_seconds,
                    ),
                    selected,
                )
            )
    return pd.DataFrame(rows)
