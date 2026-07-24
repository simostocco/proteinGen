#!/usr/bin/env python3
"""Preprocess local structure files into distance-map samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.constants import DEFAULT_RESIDUE_MAPPINGS
from protein_distance_diffusion.data.preprocess import save_processed_sample, write_manifest
from protein_distance_diffusion.data.structure_parser import iter_supported_structure_files, parse_structure_file
from protein_distance_diffusion.utils.io import write_json


def _parse_optional_min_length(value: object) -> int | None:
    """Parse `min_length` from YAML, where null disables the lower bound."""
    if value is None:
        return None
    min_length = int(value)
    if min_length <= 0:
        raise ValueError("min_length must be a positive integer or null")
    return min_length


def _parse_optional_float(value: object) -> float | None:
    """Parse an optional float threshold from YAML."""
    return None if value is None else float(value)


def main() -> None:
    """Run preprocessing."""
    parser = argparse.ArgumentParser(description="Parse structure files and save C-alpha distance matrices.")
    parser.add_argument("--config", required=True, type=Path, help="YAML preprocessing config.")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    source = Path(cfg["source_dir"])
    output_samples = Path(cfg["samples_dir"])
    rows = []
    rejections: list[dict[str, str]] = []
    residue_mappings = {**DEFAULT_RESIDUE_MAPPINGS, **cfg.get("residue_mappings", {})}
    min_length = _parse_optional_min_length(cfg.get("min_length", 40))
    for path in iter_supported_structure_files(source):
        try:
            samples = parse_structure_file(
                path,
                backend=cfg.get("backend", "auto"),
                min_length=min_length,
                max_length=int(cfg.get("max_length", 128)),
                chain_id=cfg.get("chain_id"),
                residue_mappings=residue_mappings,
                allowed_methods=cfg.get("allowed_methods"),
                max_xray_resolution_angstrom=_parse_optional_float(cfg.get("max_xray_resolution_angstrom")),
                max_cryoem_resolution_angstrom=_parse_optional_float(cfg.get("max_cryoem_resolution_angstrom")),
            )
        except Exception as exc:
            rejections.append({"source_file": str(path), "reason": type(exc).__name__, "message": str(exc)})
            continue
        if not samples:
            rejections.append(
                {
                    "source_file": str(path),
                    "reason": "NoAcceptedSamples",
                    "message": "No chains passed structure metadata, length, or residue filters.",
                }
            )
            continue
        for sample in samples:
            rows.append(save_processed_sample(sample, output_samples))
    write_manifest(rows, cfg["manifest_path"])
    summary_path = Path(cfg.get("summary_path", Path(cfg["manifest_path"]).with_suffix(".preprocess_summary.json")))
    counts: dict[str, int] = {}
    for rejection in rejections:
        counts[rejection["reason"]] = counts.get(rejection["reason"], 0) + 1
    write_json(
        summary_path,
        {
            "accepted_samples": len(rows),
            "rejected_files": len(rejections),
            "rejection_counts": counts,
            "rejections": rejections,
            "backend": cfg.get("backend", "auto"),
            "source_dir": str(source),
        },
    )
    print(f"Wrote {len(rows)} samples to {cfg['manifest_path']}")
    print(f"Wrote preprocessing summary to {summary_path}")


if __name__ == "__main__":
    main()
