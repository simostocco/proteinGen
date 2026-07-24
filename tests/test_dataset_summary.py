"""Dataset summary regression tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_summarize_manifest():
    """Load summarize_manifest from the CLI script path."""
    script = Path(__file__).parents[1] / "scripts" / "summarize_dataset.py"
    spec = importlib.util.spec_from_file_location("summarize_dataset", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.summarize_manifest


def test_summary_auto_loads_adjacent_preprocessing_summary(synthetic_manifest: Path) -> None:
    """The summary script finds `preprocess_summary.json` next to the manifest by default."""
    summarize_manifest = _load_summarize_manifest()
    summary_path = synthetic_manifest.parent / "preprocess_summary.json"
    summary_path.write_text('{"accepted_samples": 3, "rejection_counts": {"SOLUTION NMR": 2}}\n')
    summary = summarize_manifest(synthetic_manifest)
    assert summary["preprocessing_summary"]["accepted_samples"] == 3
    assert summary["preprocessing_summary"]["rejection_counts"]["SOLUTION NMR"] == 2


def test_empty_resolution_statistics_are_explicit(tmp_path: Path, synthetic_manifest: Path) -> None:
    """Unavailable resolution summaries use count zero and null numeric fields."""
    summarize_manifest = _load_summarize_manifest()
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
