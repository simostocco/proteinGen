#!/usr/bin/env python3
"""Build leakage-safe split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.splitting import run_split_workflow


def main() -> None:
    """Run split construction."""
    parser = argparse.ArgumentParser(description="Assign sequence clusters to train/validation/test splits.")
    parser.add_argument("--config", required=True, type=Path, help="YAML split config.")
    parser.add_argument("--force", action="store_true", help="Recompute cached MMseqs2 clustering outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Report planned full workflow without writing outputs.")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    result = run_split_workflow(
        manifest_path=cfg["manifest_path"],
        deduplicated_manifest_path=cfg["deduplicated_manifest_path"],
        deduplication_report_path=cfg["deduplication_report_path"],
        fasta_path=cfg["fasta_path"],
        mmseqs_output_prefix=cfg["mmseqs_output_prefix"],
        mmseqs_tmp_dir=cfg["mmseqs_tmp_dir"],
        cluster_assignments_path=cfg["cluster_assignments_path"],
        output_dir=cfg["output_dir"],
        mmseqs=cfg.get("mmseqs_executable", "mmseqs"),
        sequence_identity_threshold=float(cfg.get("sequence_identity_threshold", 0.30)),
        alignment_coverage_threshold=float(cfg.get("alignment_coverage_threshold", 0.80)),
        mmseqs_threads=int(cfg["mmseqs_threads"]) if cfg.get("mmseqs_threads") is not None else None,
        mmseqs_cov_mode=int(cfg.get("mmseqs_cov_mode", 0)),
        mmseqs_split_memory_limit=cfg.get("mmseqs_split_memory_limit"),
        mmseqs_remove_tmp_files=bool(cfg.get("mmseqs_remove_tmp_files", False)),
        mmseqs_log_path=Path(cfg["output_dir"]) / "mmseqs.log",
        state_db_path=cfg.get("state_db_path"),
        checkpoint_every=int(cfg["checkpoint_every"]) if cfg.get("checkpoint_every") is not None else None,
        minimum_sequence_length=(
            int(cfg["minimum_sequence_length"]) if cfg.get("minimum_sequence_length") is not None else None
        ),
        seed=int(cfg.get("split_seed", 42)),
        fractions=(
            float(cfg.get("train_fraction", 0.8)),
            float(cfg.get("validation_fraction", 0.1)),
            float(cfg.get("test_fraction", 0.1)),
        ),
        external_group_file=cfg.get("external_group_label_file"),
        retention_mode=str(cfg.get("retention_mode", "exact_sequence_representative")),
        collapse_within_pdb_exact_duplicates=bool(cfg.get("collapse_within_pdb_exact_duplicates", False)),
        exact_sequence_weight_exponent=float(cfg.get("exact_sequence_weight_exponent", 1.0)),
        force=args.force,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(result)
        return
    print(f"Wrote split files to {cfg['output_dir']}")
    print(
        "Retained {retained_count}/{original_count} samples; clusters={cluster_count}; splits={split_sizes}".format(
            **result
        )
    )


if __name__ == "__main__":
    main()
