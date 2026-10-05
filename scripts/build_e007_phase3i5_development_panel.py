#!/usr/bin/env python3
"""Build the hash-ranked Phase 3I.5 panel from clean identity-30 validation IDs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = Path(
        "reports/experiments/E007_matrix_sequence_cogeneration/"
        "split_homology_audit_v1_publication_correction_v1/mmseqs/identity_30/clean_validation_manifest.parquet"
    )
    calibration = Path(
        "reports/experiments/E007_matrix_sequence_cogeneration/phase3i4_calibration_v2/active_panel.json"
    )
    holdout_hash = Path(
        "reports/experiments/E007_matrix_sequence_cogeneration/phase3i4_calibration_v2/holdout_identity_hash.json"
    )
    records = pq.read_table(source, columns=["sample_id", "split", "coordinate_accepted", "length"]).to_pylist()
    calibration_rows = json.loads(calibration.read_text())
    excluded_calibration = sorted(str(row["identity"]) for row in calibration_rows)
    excluded_seeds = sorted(int(row["seed"]) for row in calibration_rows)
    holdout_fingerprint = json.loads(holdout_hash.read_text())["sha256"]
    selected = []
    for target in (64, 128, 256, 384, 500):
        candidates = [
            row
            for row in records
            if row["split"] == "validation"
            and row["coordinate_accepted"] is True
            and abs(int(row["length"]) - target) <= 4
            and str(row["sample_id"]) not in excluded_calibration
        ]
        candidates.sort(
            key=lambda row: hashlib.sha256(f"e007-phase3i5-dev:{target}:{row['sample_id']}".encode()).digest()
        )
        if len(candidates) < 2:
            raise ValueError(f"development panel underfilled near target length {target}")
        for replicate, row in enumerate(candidates[:2]):
            selected.append(
                {
                    "identity": str(row["sample_id"]),
                    "sample_id": str(row["sample_id"]),
                    "target_length": target,
                    "observed_length": int(row["length"]),
                    "split": str(row["split"]),
                    "seed": 24_000_000 + target * 100 + replicate,
                }
            )
    ids = [row["sample_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("development panel identities are not unique")
    if any(row["split"] != "validation" for row in selected):
        raise ValueError("development panel contains non-validation identity")
    panel = {
        "version": "e007_phase3i5_development_panel_v1",
        "selection": "sha256-rank clean identity-30 validation within +/-4 residues of each target length",
        "source_manifest": str(source),
        "source_manifest_sha256": sha(source),
        "calibration_exclusion": {
            "source": str(calibration),
            "sha256": sha(calibration),
            "identities": excluded_calibration,
            "seeds": excluded_seeds,
        },
        "holdout_exclusion": {
            "policy": (
                "never read holdout identities/outcomes; dev sample IDs come only from clean validation split; "
                "sampler seeds use disjoint 24-million namespace"
            ),
            "identity_hash_record": str(holdout_hash),
            "identity_hash_record_sha256": sha(holdout_hash),
            "canonical_identity_sha256": holdout_fingerprint,
            "excluded_namespace": "holdout-n",
        },
        "training_exclusion": {"source_split": "train", "development_split": "validation", "disjoint_by_split": True},
        "records": selected,
    }
    output = Path("configs/e007_phase3i5_development_panel_v1.json")
    output.write_text(json.dumps(panel, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"path": str(output), "sha256": sha(output), "panel_size": len(selected)}, sort_keys=True))


if __name__ == "__main__":
    main()
