"""mmCIF downloader tests without network access."""

from __future__ import annotations

import urllib.error
from pathlib import Path

from protein_distance_diffusion.data import download as download_module


def test_download_mmcif_retries_and_records_sha256(tmp_path: Path, monkeypatch) -> None:
    """A transient failure is retried and successful downloads record file metadata."""
    calls = []

    def fake_urlretrieve(url, filename):
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.URLError("temporary")
        Path(filename).write_bytes(b"data" * 64)
        return filename, None

    monkeypatch.setattr(download_module.urllib.request, "urlretrieve", fake_urlretrieve)
    manifest = download_module.download_mmcif_ids(
        ["1ABC"],
        tmp_path,
        retries=2,
        workers=1,
        backoff_seconds=0.0,
    )
    row = manifest.iloc[0].to_dict()
    assert row["status"] == "downloaded"
    assert row["bytes"] == 256
    assert row["sha256"] == download_module.file_sha256(tmp_path / "1abc.cif.gz")
    assert len(calls) == 2


def test_download_mmcif_uses_valid_cache_without_network(tmp_path: Path, monkeypatch) -> None:
    """Valid existing files are not downloaded again unless forced."""
    cached = tmp_path / "1abc.cif.gz"
    cached.write_bytes(b"x" * 256)

    def fail_urlretrieve(url, filename):
        raise AssertionError("valid cache should not be downloaded")

    monkeypatch.setattr(download_module.urllib.request, "urlretrieve", fail_urlretrieve)
    manifest = download_module.download_mmcif_ids(["1ABC"], tmp_path, workers=1)
    row = manifest.iloc[0].to_dict()
    assert row["status"] == "cached"
    assert row["bytes"] == 256
    assert row["sha256"] == download_module.file_sha256(cached)
