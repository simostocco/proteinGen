"""Dataset summary regression tests."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_summary_module():
    """Load summarize_dataset.py as a test module."""
    script = Path(__file__).parents[1] / "scripts" / "summarize_dataset.py"
    spec = importlib.util.spec_from_file_location("summarize_dataset", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["summarize_dataset"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summary_auto_loads_adjacent_preprocessing_summary(synthetic_manifest: Path) -> None:
    """The summary script finds `preprocess_summary.json` next to the manifest by default."""
    summarize_manifest = _load_summary_module().summarize_manifest
    summary_path = synthetic_manifest.parent / "preprocess_summary.json"
    summary_path.write_text('{"accepted_samples": 3, "rejection_counts": {"SOLUTION NMR": 2}}\n')
    summary = summarize_manifest(synthetic_manifest)
    assert summary["preprocessing_summary"]["accepted_samples"] == 3
    assert summary["preprocessing_summary"]["rejection_counts"]["SOLUTION NMR"] == 2


def test_empty_resolution_statistics_are_explicit(tmp_path: Path, synthetic_manifest: Path) -> None:
    """Unavailable resolution summaries use count zero and null numeric fields."""
    summarize_manifest = _load_summary_module().summarize_manifest
    frame = pd.read_parquet(synthetic_manifest)
    frame["resolution_angstrom"] = None
    manifest = tmp_path / "manifest.parquet"
    frame.to_parquet(manifest, index=False)
    summary = summarize_manifest(manifest)
    assert summary["resolution_distribution"] == {
        "count": 0,
        "mean": None,
        "std": None,
        "min": None,
        "25%": None,
        "50%": None,
        "75%": None,
        "max": None,
    }


def _summary_for_compare(summary: dict) -> dict:
    trimmed = dict(summary)
    trimmed.pop("updated_utc", None)
    trimmed.pop("state_db_path", None)
    return trimmed


def test_summary_interruption_resume_matches_uninterrupted(synthetic_manifest: Path, tmp_path: Path) -> None:
    """A partial summary followed by resume matches a clean uninterrupted summary."""
    module = _load_summary_module()
    clean = module.summarize_manifest(
        synthetic_manifest,
        output_dir=tmp_path / "clean",
        workers=1,
        restart=True,
        checkpoint_every=1,
    )
    partial = module.summarize_manifest(
        synthetic_manifest,
        output_dir=tmp_path / "resumed",
        workers=1,
        restart=True,
        checkpoint_every=1,
        interrupt_after_completed=1,
    )
    assert partial["accepted_chains"] == 3
    assert (tmp_path / "resumed" / "dataset_summary.partial.json").exists()
    resumed = module.summarize_manifest(
        synthetic_manifest,
        output_dir=tmp_path / "resumed",
        workers=1,
        resume=True,
        checkpoint_every=1,
    )
    assert _summary_for_compare(resumed) == _summary_for_compare(clean)


def test_summary_resume_creates_no_duplicate_aggregates(synthetic_manifest: Path, tmp_path: Path) -> None:
    """Repeated resume does not duplicate per-sample aggregate rows."""
    module = _load_summary_module()
    output = tmp_path / "summary"
    module.summarize_manifest(synthetic_manifest, output_dir=output, workers=1, restart=True, checkpoint_every=1)
    module.summarize_manifest(synthetic_manifest, output_dir=output, workers=1, resume=True, checkpoint_every=1)
    conn = sqlite3.connect(output / "dataset_summary_state.sqlite")
    try:
        assert conn.execute("SELECT COUNT(*) FROM sample_aggregates").fetchone()[0] == 3
    finally:
        conn.close()


def test_summary_resume_rejects_manifest_hash_mismatch(synthetic_manifest: Path, tmp_path: Path) -> None:
    """Changing the manifest refuses resume unless restarted."""
    module = _load_summary_module()
    output = tmp_path / "summary"
    module.summarize_manifest(synthetic_manifest, output_dir=output, workers=1, restart=True, checkpoint_every=1)
    frame = pd.read_parquet(synthetic_manifest)
    frame.loc[0, "sequence"] = "AAAAAA"
    changed = tmp_path / "changed.parquet"
    frame.to_parquet(changed, index=False)
    with pytest.raises(ValueError, match="manifest hash differs"):
        module.summarize_manifest(changed, output_dir=output, workers=1, resume=True)


def test_summary_resume_rejects_settings_hash_mismatch(synthetic_manifest: Path, tmp_path: Path) -> None:
    """Changing contact settings refuses resume unless restarted."""
    module = _load_summary_module()
    output = tmp_path / "summary"
    module.summarize_manifest(
        synthetic_manifest,
        output_dir=output,
        workers=1,
        restart=True,
        contact_thresholds=[8.0],
    )
    with pytest.raises(ValueError, match="settings hash differs"):
        module.summarize_manifest(
            synthetic_manifest,
            output_dir=output,
            workers=1,
            resume=True,
            contact_thresholds=[6.0],
        )


def test_summary_counts_missing_or_corrupt_samples(synthetic_manifest: Path, tmp_path: Path) -> None:
    """Corrupt sample files are tracked without crashing the whole summary."""
    module = _load_summary_module()
    frame = pd.read_parquet(synthetic_manifest)
    Path(frame.loc[0, "path"]).write_text("not an npz")
    summary = module.summarize_manifest(synthetic_manifest, output_dir=tmp_path / "summary", workers=1, restart=True)
    assert summary["missing_or_corrupt_samples"] == 1
    data = json.loads((tmp_path / "summary" / "dataset_summary.json").read_text())
    assert data["missing_or_corrupt_samples"] == 1
