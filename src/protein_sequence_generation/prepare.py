"""Prepare and audit standalone sequence manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from protein_sequence_generation.dataset import (
    assert_no_split_overlap,
    load_sequence_manifest,
    validate_records,
)
from protein_sequence_generation.utils import atomic_write_json
from protein_sequence_generation.vocabulary import ProteinVocabulary


def prepare_sequence_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Validate existing leakage-safe splits and write sequence audit summaries.

    This function does not silently random-split unsplit manifests. If split files are
    absent, it raises with instructions to provide leakage-safe splits or explicit
    clustering preparation.
    """
    vocab = ProteinVocabulary()
    min_length = int(config.get("min_length", 20))
    max_length = int(config.get("max_length", 500))
    allow_unknown = bool(config.get("allow_unknown", False))
    output_dir = Path(config.get("output_dir", "data/sequence"))
    audit_dir = Path(config.get("audit_dir", output_dir / "audits"))
    split_paths = config.get("splits", {})
    required = ["train", "validation", "test"]
    missing = [split for split in required if not split_paths.get(split) or not Path(split_paths[split]).exists()]
    if missing:
        unsplit = config.get("unsplit_manifest")
        if unsplit:
            raise ValueError(
                "Unsplit manifests require explicit sequence-cluster/PDB grouping. "
                "Provide train/validation/test split files or add a standalone MMseqs2 preparation step."
            )
        raise FileNotFoundError(f"Missing leakage-safe split files for: {missing}")
    records_by_split = {}
    audits = {}
    for split in required:
        path = Path(split_paths[split])
        frame = load_sequence_manifest(path)
        records, audit = validate_records(
            frame,
            source_path=path,
            vocabulary=vocab,
            min_length=min_length,
            max_length=max_length,
            allow_unknown=allow_unknown,
        )
        records_by_split[split] = records
        audits[split] = audit
    assert_no_split_overlap(records_by_split)
    summary = {
        "min_length": min_length,
        "max_length": max_length,
        "allow_unknown": allow_unknown,
        "splits": audits,
    }
    atomic_write_json(audit_dir / "sequence_dataset_audit.json", summary)
    return summary
