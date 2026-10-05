"""Distance-AF dry-run benchmark preparation and metric helpers."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from protein_distance_diffusion.evaluation.repairability import (
    atomic_write_json,
    atomic_write_text,
    classical_mds_rank3_projection,
    contact_metrics,
    symmetrize_zero_diagonal,
    upper_pair_mask,
)

SEQUENCE_PROVENANCE = {
    "native_pdb_sequence",
    "model_generated_sequence",
    "external_inverse_folding_sequence",
    "sequence_only_baseline",
    "shuffled_negative_control",
    "mismatched_negative_control",
}
GENERATED_PRIMARY_COHORTS = {"E002_designed", "E004_designed", "generated_e002", "generated_e004"}
REAL_COHORTS = {"real_native", "real_corrupted", "mismatched_negative", "sequence_only"}


@dataclass(frozen=True)
class Restraint:
    """One CA-distance restraint row."""

    sample_id: str
    i_zero_based: int
    j_zero_based: int
    i_distance_af: int
    j_distance_af: int
    target_distance_angstrom: float
    selection_strategy: str
    matrix_source: str
    projection_residual_angstrom: float
    split: str
    sequence_separation: int
    target_distance_bin: str


def validate_sequence_provenance(value: str) -> str:
    """Validate sequence provenance."""
    if value not in SEQUENCE_PROVENANCE:
        raise ValueError(f"sequence_provenance must be one of {sorted(SEQUENCE_PROVENANCE)}")
    return value


def require_sequence_for_cohort(cohort: str, sequence: str | None, sequence_provenance: str | None) -> None:
    """Reject primary generated matrix jobs without explicit sequence source."""
    if sequence_provenance is None:
        if cohort in GENERATED_PRIMARY_COHORTS:
            raise ValueError("Primary generated Distance-AF jobs require explicit sequence_provenance")
        return
    validate_sequence_provenance(sequence_provenance)
    if sequence is None or not sequence:
        raise ValueError("Distance-AF jobs require an explicit residue sequence")
    if cohort in GENERATED_PRIMARY_COHORTS and sequence_provenance not in {
        "model_generated_sequence",
        "external_inverse_folding_sequence",
    }:
        raise ValueError("Primary generated E002/E004 jobs require generated or inverse-folded sequences")


def eligible_pairs(matrix: np.ndarray, *, min_sequence_separation: int = 12) -> pd.DataFrame:
    """Return deduplicated upper-triangular eligible CA distance pairs."""
    d = symmetrize_zero_diagonal(matrix)
    mask = upper_pair_mask(d.shape[0], min_separation=int(min_sequence_separation))
    i, j = np.where(mask)
    finite = np.isfinite(d[i, j])
    return pd.DataFrame(
        {
            "i_zero_based": i[finite].astype(int),
            "j_zero_based": j[finite].astype(int),
            "i_distance_af": i[finite].astype(int) + 1,
            "j_distance_af": j[finite].astype(int) + 1,
            "target_distance_angstrom": d[i[finite], j[finite]].astype(float),
            "sequence_separation": (j[finite] - i[finite]).astype(int),
        }
    )


def _distance_bin(value: float) -> str:
    if value <= 8.0:
        return "contact"
    if value <= 20.0:
        return "medium"
    return "long"


def _separation_bin(value: int) -> str:
    if value < 24:
        return "medium_range"
    return "long_range"


def select_restraints(
    matrix: np.ndarray,
    *,
    sample_id: str,
    matrix_source: str,
    strategy: str,
    count: int,
    seed: int,
    min_sequence_separation: int = 12,
    heldout_multiplier: int = 1,
) -> list[Restraint]:
    """Select deterministic guidance and held-out restraints."""
    if strategy not in {"uniform_long_range", "contact_enriched", "projection_consistent", "random_control"}:
        raise ValueError(f"Unsupported restraint selection strategy: {strategy}")
    pairs = eligible_pairs(matrix, min_sequence_separation=min_sequence_separation)
    if pairs.empty:
        return []
    projection = classical_mds_rank3_projection(matrix).projected_distances
    pairs["projection_residual_angstrom"] = np.abs(
        pairs.apply(lambda row: matrix[int(row.i_zero_based), int(row.j_zero_based)], axis=1)
        - pairs.apply(lambda row: projection[int(row.i_zero_based), int(row.j_zero_based)], axis=1)
    )
    pairs["distance_bin"] = pairs["target_distance_angstrom"].map(_distance_bin)
    pairs["separation_bin"] = pairs["sequence_separation"].map(_separation_bin)
    need = min(len(pairs), int(count) * (1 + int(heldout_multiplier)))
    rng = np.random.default_rng(int(seed))
    if strategy == "projection_consistent":
        selected = pairs.sort_values(["projection_residual_angstrom", "i_zero_based", "j_zero_based"]).head(need)
    elif strategy == "contact_enriched":
        selected_parts = []
        per_bin = max(1, need // max(1, pairs["distance_bin"].nunique()))
        for _, group in pairs.groupby("distance_bin"):
            take = min(len(group), per_bin)
            selected_parts.append(group.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
        selected = pd.concat(selected_parts).drop_duplicates(["i_zero_based", "j_zero_based"]).head(need)
        if len(selected) < need:
            rest = pairs.drop(selected.index, errors="ignore")
            selected = pd.concat(
                [
                    selected,
                    rest.sample(n=min(len(rest), need - len(selected)), random_state=int(seed)),
                ]
            )
    else:
        selected = pairs.sample(n=need, random_state=int(seed))
    selected = selected.sort_values(["i_zero_based", "j_zero_based"]).reset_index(drop=True)
    guide_count = min(int(count), len(selected))
    rows = []
    for idx, row in selected.iterrows():
        split = "guidance" if idx < guide_count else "heldout"
        rows.append(
            Restraint(
                sample_id=str(sample_id),
                i_zero_based=int(row.i_zero_based),
                j_zero_based=int(row.j_zero_based),
                i_distance_af=int(row.i_distance_af),
                j_distance_af=int(row.j_distance_af),
                target_distance_angstrom=float(row.target_distance_angstrom),
                selection_strategy=strategy,
                matrix_source=matrix_source,
                projection_residual_angstrom=float(row.projection_residual_angstrom),
                split=split,
                sequence_separation=int(row.sequence_separation),
                target_distance_bin=str(row.distance_bin),
            )
        )
    return rows


def restraints_to_frame(restraints: list[Restraint]) -> pd.DataFrame:
    """Convert restraints to a dataframe."""
    return pd.DataFrame([r.__dict__ for r in restraints])


def write_distance_af_restraints(path: Path, restraints: pd.DataFrame) -> None:
    """Write Distance-AF comma-separated 1-based restraint file."""
    guidance = restraints[restraints["split"] == "guidance"]
    lines = [
        f"{int(row.i_distance_af)},{int(row.j_distance_af)},{float(row.target_distance_angstrom):.6f}"
        for _, row in guidance.iterrows()
    ]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def write_fasta(path: Path, *, sample_id: str, sequence: str) -> None:
    """Write FASTA for Distance-AF."""
    atomic_write_text(path, f">{sample_id}\n{sequence}\n")


def build_distance_af_command(
    *,
    distance_af_python: str,
    distance_af_root: Path,
    target_file: Path,
    dist_info: Path,
    fasta_file: Path,
    output_dir: Path,
    external_command_template: str | None = None,
) -> list[str]:
    """Construct a subprocess-safe Distance-AF command list."""
    if external_command_template:
        return [
            part.format(
                distance_af_root=str(distance_af_root),
                target_file=str(target_file),
                dist_info=str(dist_info),
                fasta_file=str(fasta_file),
                output_dir=str(output_dir),
            )
            for part in shlex.split(external_command_template)
        ]
    return [
        str(distance_af_python),
        str(distance_af_root / "Distance_AF.py"),
        f"--target_file={target_file}",
        f"--dist_info={dist_info}",
        f"--fasta_file={fasta_file}",
        f"--output_dir={output_dir}",
    ]


def run_external_command(command: list[str], *, execute: bool) -> dict[str, Any]:
    """Run or dry-run an external command without shell=True."""
    if not execute:
        return {"status": "dry_run", "command": command, "return_code": None, "stdout": "", "stderr": ""}
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "status": "completed" if completed.returncode == 0 else "failed",
        "command": command,
        "return_code": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def evaluate_restraint_satisfaction(
    target_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    restraints: pd.DataFrame,
    *,
    split: str,
) -> dict[str, float | int]:
    """Evaluate prediction distances against guidance or held-out restraints."""
    subset = restraints[restraints["split"] == split]
    errors = []
    for _, row in subset.iterrows():
        i = int(row.i_zero_based)
        j = int(row.j_zero_based)
        errors.append(float(predicted_matrix[i, j] - target_matrix[i, j]))
    arr = np.asarray(errors, dtype=np.float64)
    out: dict[str, float | int] = {"count": int(arr.size)}
    for tolerance in (1.0, 2.0, 4.0):
        out[f"fraction_within_{int(tolerance)}A"] = (
            float(np.mean(np.abs(arr) <= tolerance)) if arr.size else float("nan")
        )
    out["mae"] = float(np.mean(np.abs(arr))) if arr.size else float("nan")
    out["rmse"] = float(np.sqrt(np.mean(arr * arr))) if arr.size else float("nan")
    return out


def heldout_contact_metrics(
    target_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    restraints: pd.DataFrame,
) -> dict[str, float]:
    """Compute held-out contact metrics over held-out restraint pairs."""
    heldout = restraints[restraints["split"] == "heldout"]
    mask = np.zeros_like(target_matrix, dtype=bool)
    for _, row in heldout.iterrows():
        i = int(row.i_zero_based)
        j = int(row.j_zero_based)
        mask[i, j] = True
        mask[j, i] = True
    return contact_metrics(target_matrix, predicted_matrix, threshold=8.0, mask=mask)


def reject_malformed_prediction_table(frame: pd.DataFrame) -> None:
    """Validate a minimal parsed Distance-AF result table."""
    required = {"job_id", "prediction_kind", "prediction_path", "status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Malformed Distance-AF output table missing columns: {sorted(missing)}")


def write_metric_definitions(path: Path) -> None:
    """Write benchmark metric definitions."""
    atomic_write_json(
        path,
        {
            "supplied_restraint_satisfaction": (
                "Agreement with guidance restraints supplied to Distance-AF; partly circular."
            ),
            "held_out_restraint_generalization": "Agreement with restraints never supplied to Distance-AF.",
            "unrestrained_fold_back": "Independent sequence-only folding toward the proposed target.",
            "confidence": "pLDDT/pTM/PAE summaries when available; confidence alone is not target validation.",
        },
    )
