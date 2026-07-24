#!/usr/bin/env python3
"""Summarize processed distance-map datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from protein_distance_diffusion.data.clustering import add_sequence_hashes
from protein_distance_diffusion.data.preprocess import load_manifest
from protein_distance_diffusion.evaluation.contact_maps import binary_contact_map
from protein_distance_diffusion.utils.io import write_json


def _maybe_load_json(path: str | Path | None) -> dict[str, Any]:
    """Load optional JSON metadata."""
    if path is None:
        return {}
    src = Path(path)
    return json.loads(src.read_text()) if src.exists() else {}


def _default_preprocessing_summary_path(manifest: str | Path) -> Path:
    """Return the conventional preprocessing summary path next to a manifest."""
    return Path(manifest).parent / "preprocess_summary.json"


def _resolution_summary(frame) -> dict[str, float | int | None]:
    """Summarize resolution values with clear unavailable numeric fields."""
    empty = {
        "count": 0,
        "mean": None,
        "std": None,
        "min": None,
        "25%": None,
        "50%": None,
        "75%": None,
        "max": None,
    }
    if "resolution_angstrom" not in frame:
        return empty
    import pandas as pd

    values = pd.to_numeric(frame["resolution_angstrom"], errors="coerce").dropna()
    if values.empty:
        return empty
    desc = values.describe()
    return {
        "count": int(desc["count"]),
        "mean": float(desc["mean"]),
        "std": float(desc["std"]) if not pd.isna(desc["std"]) else None,
        "min": float(desc["min"]),
        "25%": float(desc["25%"]),
        "50%": float(desc["50%"]),
        "75%": float(desc["75%"]),
        "max": float(desc["max"]),
    }


def _contact_fraction(path: str | Path, thresholds: list[float], *, exclude_near_diagonal: int) -> dict[str, float]:
    """Calculate contact fractions for one processed sample."""
    data = np.load(path, allow_pickle=False)
    matrix = data["distance_matrix"].astype(np.float32)
    out = {}
    for threshold in thresholds:
        contacts = binary_contact_map(
            matrix,
            threshold_angstrom=threshold,
            exclude_diagonal=True,
            exclude_near_diagonal=exclude_near_diagonal,
        )
        idx = np.arange(matrix.shape[0])
        valid = np.triu(np.ones(matrix.shape, dtype=bool), k=1)
        if exclude_near_diagonal > 0:
            valid &= np.abs(idx[:, None] - idx[None, :]) > exclude_near_diagonal
        out[str(threshold)] = float(contacts[valid].mean()) if valid.any() else float("nan")
    return out


def summarize_manifest(
    manifest: str | Path,
    *,
    split_manifest: str | Path | None = None,
    cluster_assignments: str | Path | None = None,
    preprocessing_summary: str | Path | None = None,
    contact_thresholds: list[float] | None = None,
    exclude_near_diagonal: int = 0,
) -> dict[str, Any]:
    """Build machine-readable dataset summary statistics.

    Args:
        manifest: Complete processed manifest.
        split_manifest: Optional all-splits manifest with `split` column.
        cluster_assignments: Optional MMseqs2 cluster TSV.
        preprocessing_summary: Optional preprocessing summary JSON.
        contact_thresholds: Contact thresholds in angstrom.
        exclude_near_diagonal: Exclude contacts with `|i-j| <= value`.

    Returns:
        JSON-serializable summary.
    """
    thresholds = contact_thresholds or [6.0, 8.0, 10.0]
    frame = add_sequence_hashes(load_manifest(manifest))
    summary_path = preprocessing_summary or _default_preprocessing_summary_path(manifest)
    split_counts: dict[str, int] = {}
    if split_manifest:
        split_frame = load_manifest(split_manifest)
        if "split" in split_frame:
            split_counts = {str(k): int(v) for k, v in split_frame["split"].value_counts().to_dict().items()}
    clusters = None
    if cluster_assignments:
        import pandas as pd

        clusters = pd.read_csv(cluster_assignments, sep="\t")
    aa_counts: Counter[str] = Counter()
    for sequence in frame["sequence"].astype(str):
        aa_counts.update(sequence)
    contact_values = {str(threshold): [] for threshold in thresholds}
    for row in frame.itertuples(index=False):
        fractions = _contact_fraction(row.path, thresholds, exclude_near_diagonal=exclude_near_diagonal)
        for threshold, value in fractions.items():
            contact_values[threshold].append(value)
    missing_metadata = {
        column: int(frame[column].isna().sum())
        for column in ["experimental_method", "resolution_angstrom"]
        if column in frame
    }
    return {
        "accepted_chains": int(len(frame)),
        "unique_sequences": int(frame["sequence_hash"].nunique()),
        "mmseqs2_clusters": int(clusters.iloc[:, 0].nunique()) if clusters is not None else None,
        "split_counts": split_counts,
        "length_distribution": {
            "min": int(frame["length"].min()) if len(frame) else 0,
            "max": int(frame["length"].max()) if len(frame) else 0,
            "mean": float(frame["length"].mean()) if len(frame) else 0.0,
        },
        "experimental_method_distribution": frame.get("experimental_method", []).value_counts().to_dict()
        if "experimental_method" in frame
        else {},
        "resolution_distribution": _resolution_summary(frame),
        "amino_acid_frequency": dict(sorted(aa_counts.items())),
        "contact_fraction_mean": {threshold: float(np.nanmean(values)) for threshold, values in contact_values.items()},
        "missing_metadata_counts": missing_metadata,
        "preprocessing_summary": _maybe_load_json(summary_path),
    }


def write_plots(manifest: str | Path, output_dir: str | Path) -> None:
    """Write human-readable length and resolution plots."""
    frame = load_manifest(manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(frame["length"], bins=30, color="#1b9e77", edgecolor="white")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Accepted chains")
    fig.tight_layout()
    fig.savefig(out / "length_distribution.png", dpi=140)
    plt.close(fig)
    if "resolution_angstrom" in frame and frame["resolution_angstrom"].notna().any():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(frame["resolution_angstrom"].dropna(), bins=30, color="#7570b3", edgecolor="white")
        ax.set_xlabel("Resolution (A)")
        ax.set_ylabel("Accepted chains")
        fig.tight_layout()
        fig.savefig(out / "resolution_distribution.png", dpi=140)
        plt.close(fig)


def main() -> None:
    """Run dataset summary generation."""
    parser = argparse.ArgumentParser(description="Summarize processed protein distance-map samples.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--cluster-assignments", type=Path, default=None)
    parser.add_argument("--preprocessing-summary", type=Path, default=None)
    parser.add_argument("--contact-thresholds", type=float, nargs="+", default=[6.0, 8.0, 10.0])
    parser.add_argument("--exclude-near-diagonal", type=int, default=0)
    args = parser.parse_args()
    summary = summarize_manifest(
        args.manifest,
        split_manifest=args.split_manifest,
        cluster_assignments=args.cluster_assignments,
        preprocessing_summary=args.preprocessing_summary,
        contact_thresholds=args.contact_thresholds,
        exclude_near_diagonal=args.exclude_near_diagonal,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "dataset_summary.json", summary)
    write_plots(args.manifest, args.output_dir)
    print(f"Wrote dataset summary to {args.output_dir}")


if __name__ == "__main__":
    main()
