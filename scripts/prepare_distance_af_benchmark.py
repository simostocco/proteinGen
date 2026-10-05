#!/usr/bin/env python
"""Prepare an analysis-only Distance-AF dry-run benchmark manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from protein_distance_diffusion.evaluation.distance_af import (
    GENERATED_PRIMARY_COHORTS,
    build_distance_af_command,
    require_sequence_for_cohort,
    restraints_to_frame,
    select_restraints,
    write_distance_af_restraints,
    write_fasta,
    write_metric_definitions,
)
from protein_distance_diffusion.evaluation.repairability import (
    atomic_write_json,
    atomic_write_text,
    corrupt_distance_matrix,
    pairwise_distances,
    sha256_file,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _load_matrix(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    for key in ("physical_distance_matrix_angstrom", "distance_matrix", "raw_physical_distance_matrix_angstrom"):
        if key in data:
            return np.asarray(data[key], dtype=np.float64)
    raise ValueError(f"No supported distance matrix key found in {path}")


def _synthetic_samples(root: Path) -> pd.DataFrame:
    root.mkdir(parents=True, exist_ok=True)
    coords = np.stack([np.arange(24) * 3.8, np.sin(np.arange(24) / 3.0), np.cos(np.arange(24) / 4.0)], axis=1)
    matrix = pairwise_distances(coords)
    npz = root / "synthetic_native.npz"
    tmp = root / ".synthetic_native.npz.tmp"
    with tmp.open("wb") as handle:
        np.savez(handle, distance_matrix=matrix)
    tmp.replace(npz)
    return pd.DataFrame(
        [
            {
                "sample_id": "synthetic_native",
                "cohort": "real_native",
                "matrix_path": str(npz),
                "sequence": "A" * matrix.shape[0],
                "sequence_provenance": "native_pdb_sequence",
            },
            {
                "sample_id": "synthetic_generated_blocked",
                "cohort": "generated_e004",
                "matrix_path": str(npz),
                "sequence": "",
                "sequence_provenance": "",
            },
        ]
    )


def _read_sample_table(config: dict[str, Any], output_root: Path) -> pd.DataFrame:
    if config.get("synthetic_smoke"):
        return _synthetic_samples(output_root / "synthetic_inputs")
    path = config.get("sample_manifest")
    if not path:
        return pd.DataFrame()
    sample_path = Path(path)
    return pd.read_parquet(sample_path) if sample_path.suffix == ".parquet" else pd.read_csv(sample_path)


def _directories(output_root: Path) -> dict[str, Path]:
    names = [
        "sequences",
        "guidance_restraints",
        "heldout_restraints",
        "dry_run_commands",
        "results",
        "logs",
    ]
    dirs = {name: output_root / name for name in names}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def prepare_distance_af_benchmark(config_path: Path) -> Path:
    """Prepare Distance-AF dry-run files without executing external inference."""
    config = _load_yaml(config_path)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    dirs = _directories(output_root)
    samples = _read_sample_table(config, output_root)
    strategies = list(config.get("restraint_strategies", ["uniform_long_range"]))
    budgets = [int(value) for value in config.get("restraint_budgets", [64])]
    min_sep = int(config.get("min_sequence_separation", 12))
    seed = int(config.get("seed", 0))
    execute = bool(config.get("execute_distance_af", False))
    if execute:
        raise ValueError("This preparation script is configured for dry-run planning only; execution is disabled here")

    job_rows = []
    raw_hashes = {str(config_path): sha256_file(config_path)}
    all_restraints = []
    for _, row in samples.iterrows():
        sample_id = str(row["sample_id"])
        cohort = str(row.get("cohort", "real_native"))
        matrix_path = Path(row["matrix_path"])
        raw_hashes[str(matrix_path)] = sha256_file(matrix_path)
        sequence = str(row.get("sequence", "") or "")
        provenance = str(row.get("sequence_provenance", "") or "") or None
        for strategy in strategies:
            for budget in budgets:
                job_id = f"{sample_id}_{cohort}_{strategy}_{budget}"
                try:
                    require_sequence_for_cohort(cohort, sequence, provenance)
                except ValueError as exc:
                    job_rows.append(
                        {
                            "job_id": job_id,
                            "sample_id": sample_id,
                            "cohort": cohort,
                            "status": "blocked_pending_sequence_model"
                            if cohort in GENERATED_PRIMARY_COHORTS
                            else "blocked_invalid_sequence",
                            "block_reason": str(exc),
                            "matrix_path": str(matrix_path),
                            "sequence_provenance": provenance,
                            "restraint_strategy": strategy,
                            "restraint_budget": budget,
                        }
                    )
                    continue
                matrix = _load_matrix(matrix_path)
                if len(sequence) != matrix.shape[0]:
                    raise ValueError(f"Sequence length mismatch for {sample_id}: {len(sequence)} != {matrix.shape[0]}")
                if cohort == "real_corrupted":
                    matrix = corrupt_distance_matrix(
                        matrix,
                        noise_angstrom=float(row.get("noise_angstrom", 2.0)),
                        seed=seed,
                    )
                restraints = restraints_to_frame(
                    select_restraints(
                        matrix,
                        sample_id=sample_id,
                        matrix_source=cohort,
                        strategy=strategy,
                        count=budget,
                        heldout_multiplier=int(config.get("heldout_multiplier", 1)),
                        min_sequence_separation=min_sep,
                        seed=seed,
                    )
                )
                fasta = dirs["sequences"] / f"{job_id}.fasta"
                guidance = dirs["guidance_restraints"] / f"{job_id}.txt"
                heldout = dirs["heldout_restraints"] / f"{job_id}.parquet"
                command_path = dirs["dry_run_commands"] / f"{job_id}.json"
                write_fasta(fasta, sample_id=sample_id, sequence=sequence)
                write_distance_af_restraints(guidance, restraints)
                restraints[restraints["split"] == "heldout"].to_parquet(heldout, index=False)
                command = build_distance_af_command(
                    distance_af_python=str(config.get("distance_af_python", "python")),
                    distance_af_root=Path(config.get("distance_af_root", "/path/to/Distance-AF")),
                    target_file=fasta,
                    dist_info=guidance,
                    fasta_file=fasta,
                    output_dir=dirs["results"] / job_id,
                    external_command_template=config.get("external_command_template"),
                )
                atomic_write_json(command_path, {"execute": False, "command": command})
                job_rows.append(
                    {
                        "job_id": job_id,
                        "sample_id": sample_id,
                        "cohort": cohort,
                        "status": "dry_run_ready",
                        "block_reason": None,
                        "matrix_path": str(matrix_path),
                        "matrix_sha256": raw_hashes[str(matrix_path)],
                        "sequence_path": str(fasta),
                        "sequence_provenance": provenance,
                        "guidance_restraints_path": str(guidance),
                        "heldout_restraints_path": str(heldout),
                        "dry_run_command_path": str(command_path),
                        "restraint_strategy": strategy,
                        "restraint_budget": budget,
                        "guidance_restraint_count": int((restraints["split"] == "guidance").sum()),
                        "heldout_restraint_count": int((restraints["split"] == "heldout").sum()),
                    }
                )
                all_restraints.append(restraints)
    manifest = pd.DataFrame(job_rows)
    manifest.to_parquet(output_root / "job_manifest.parquet", index=False)
    if all_restraints:
        pd.concat(all_restraints, ignore_index=True).to_parquet(output_root / "all_restraints.parquet", index=False)
    write_metric_definitions(output_root / "distance_af_metric_definitions.json")
    atomic_write_json(
        output_root / "benchmark_protocol.json",
        {
            "status": "dry_run_prepared",
            "execute_distance_af": False,
            "generated_primary_cohorts": sorted(GENERATED_PRIMARY_COHORTS),
            "primary_generated_jobs_require_explicit_sequence": True,
            "restraint_budgets": budgets,
            "restraint_strategies": strategies,
            "min_sequence_separation": min_sep,
            "raw_input_hashes_before": raw_hashes,
            "raw_input_hashes_after": raw_hashes,
        },
    )
    atomic_write_text(
        output_root / "README.md",
        "# Distance-AF Repairability Benchmark\n\n"
        "This directory is a dry-run scaffold. It prepares FASTA files, 1-based "
        "comma-separated CA-distance restraints, held-out restraint tables, and "
        "subprocess-safe command lists, but it does not execute Distance-AF or "
        "AlphaFold/OpenFold inference.\n\n"
        "Primary generated E002/E004 jobs are blocked until an explicit sequence "
        "source is supplied. Supplied-restraint satisfaction is not independent "
        "validation because Distance-AF optimizes against those restraints.\n",
    )
    atomic_write_text(
        output_root / "environment_requirements.md",
        "# External Environment Requirements\n\n"
        "Distance-AF is external GPLv3 software derived from AlphaFold/OpenFold. "
        "Keep it outside this repository unless licensing is reviewed. A real run "
        "requires a compatible Distance-AF checkout, its Python environment, model "
        "weights/databases as required by that project, GPU capacity, and explicit "
        "FASTA sequences for every generated-distance job.\n",
    )
    atomic_write_text(
        output_root / "distance_af_positive_control_plan.md",
        "# Positive-Control Pilot Plan\n\n"
        "Start with real-native sequences and native CA-distance restraints, then "
        "repeat with corrupted restraints at fixed noise levels. Compare supplied "
        "restraint satisfaction, held-out CA-distance recovery, fold-back to the "
        "native distance map, and Distance-AF confidence outputs. Only after those "
        "controls behave sensibly should generated E002/E004 matrices be tested.\n",
    )
    return output_root / "benchmark_protocol.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_distance_af_benchmark(args.config))


if __name__ == "__main__":
    main()
