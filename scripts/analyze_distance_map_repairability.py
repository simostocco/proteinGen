#!/usr/bin/env python
"""Analysis-only rank-3 repairability and C-alpha trace audit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/proteingen_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from protein_distance_diffusion.evaluation.repairability import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    classical_mds_rank3_projection,
    contact_metrics,
    corrupt_distance_matrix,
    export_ca_pdb,
    kabsch_rmsd,
    pairwise_distances,
    repairability_metrics,
    replace_dataframe,
    sha256_file,
    symmetrize_zero_diagonal,
    trace_metrics,
    upper_pair_mask,
)

RAW_FILES = [
    "protocol.json",
    "calibrated_analysis_protocol.json",
    "metrics/calibrated_summary.json",
    "metrics/empirical_validity_by_length.csv",
    "metrics/empirical_validity_per_sample.parquet",
    "metrics/real_control_metrics.parquet",
    "metrics/real_control_selection.csv",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _hash_inputs(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in sorted(paths)}


def _evaluation_input_paths(root: Path) -> list[Path]:
    paths = [root / rel for rel in RAW_FILES if (root / rel).exists()]
    paths.extend(sorted((root / "generated").glob("N*/*.npz")))
    return paths


def _validate_protocol(root: Path, *, expected_checkpoint_sha256: str | None) -> dict[str, Any]:
    protocol = _read_json(root / "protocol.json")
    if protocol.get("status") != "completed":
        raise ValueError(f"{root}/protocol.json is not completed")
    checkpoint_sha256 = protocol.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or not checkpoint_sha256:
        raise ValueError(f"{root}/protocol.json is missing checkpoint_sha256")
    if expected_checkpoint_sha256 and checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            "candidate checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint_sha256}, protocol records {checkpoint_sha256}"
        )
    expected_counts = {int(k): int(v) for k, v in protocol.get("sample_counts_by_length", {}).items()}
    for length, expected_count in expected_counts.items():
        observed = len(list((root / "generated" / f"N{length:04d}").glob("*.npz")))
        if observed != expected_count:
            raise ValueError(f"Generated count mismatch for N={length}: expected {expected_count}, found {observed}")
    return protocol


def _load_generated_samples(root: Path, *, label: str, limit_per_length: int | None = None) -> list[dict[str, Any]]:
    samples = []
    for path in sorted((root / "generated").glob("N*/*.npz")):
        data = np.load(path, allow_pickle=False)
        length = int(data["requested_length"])
        length_seen = sum(1 for row in samples if row["requested_length"] == length)
        if limit_per_length is not None and length_seen >= int(limit_per_length):
            continue
        matrix = np.asarray(data["physical_distance_matrix_angstrom"], dtype=np.float64)
        samples.append(
            {
                "source_type": label,
                "sample_id": str(data["sample_id"]),
                "requested_length": length,
                "sample_index": int(data["sample_index"]),
                "seed": int(data["seed"]),
                "path": path,
                "matrix": matrix,
                "actual_length": int(matrix.shape[0]),
            }
        )
    return samples


def _load_npz_matrix(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    for key in ("distance_matrix", "physical_distance_matrix_angstrom", "raw_physical_distance_matrix_angstrom"):
        if key in data:
            return np.asarray(data[key], dtype=np.float64)
    raise ValueError(f"No supported distance matrix key found in {path}")


def _read_manifest_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _select_real_control_rows(
    frame: pd.DataFrame,
    *,
    manifest_path: Path,
    required_lengths: list[int],
    limit_per_length: int | None,
) -> pd.DataFrame:
    required_columns = {"path", "requested_length", "length"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Real-control manifest {manifest_path} is missing columns: {missing_columns}")
    if limit_per_length is not None and limit_per_length < 1:
        raise ValueError("--limit-per-length must be at least 1")

    controls = frame.copy()
    controls["requested_length"] = pd.to_numeric(controls["requested_length"], errors="raise").astype(int)
    controls["length"] = pd.to_numeric(controls["length"], errors="raise").astype(int)
    controls = controls[controls["requested_length"].isin(required_lengths)]

    for identity_column in ("sample_id", "path"):
        if identity_column not in controls.columns:
            continue
        cohort_counts = controls.groupby(identity_column, dropna=False)["requested_length"].nunique()
        reused = cohort_counts[cohort_counts > 1]
        if not reused.empty:
            raise ValueError(
                f"Real-control manifest reuses {identity_column} values across requested-length cohorts: "
                f"{reused.index[0]}"
            )

    selected = []
    required_count = limit_per_length if limit_per_length is not None else 1
    for requested_length in required_lengths:
        cohort = controls[controls["requested_length"] == requested_length]
        if len(cohort) < required_count:
            raise ValueError(
                "Insufficient real controls for requested_length="
                f"{requested_length}: require {required_count}, found {len(cohort)}"
            )
        selected.append(cohort.head(limit_per_length) if limit_per_length is not None else cohort)
    return pd.concat(selected, ignore_index=True)


def _load_manifest_samples(frame: pd.DataFrame, *, label: str) -> list[dict[str, Any]]:
    samples = []
    for _, row in frame.iterrows():
        matrix_path = Path(row["path"])
        matrix = _load_npz_matrix(matrix_path)
        actual_length = int(row["length"])
        if matrix.shape != (actual_length, actual_length):
            raise ValueError(
                f"Real-control matrix shape {matrix.shape} does not match actual length {actual_length}: {matrix_path}"
            )
        samples.append(
            {
                "source_type": label,
                "sample_id": str(row.get("sample_id", matrix_path.stem)),
                "requested_length": int(row["requested_length"]),
                "actual_length": actual_length,
                "sample_index": int(row.get("sample_index", len(samples))),
                "seed": int(row.get("seed", 0)),
                "path": matrix_path,
                "matrix": matrix,
            }
        )
    return samples


def _sample_rows(samples: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    repair_rows = []
    trace_rows = []
    for sample in samples:
        matrix = symmetrize_zero_diagonal(sample["matrix"])
        projection = classical_mds_rank3_projection(matrix)
        base = {
            "source_type": sample["source_type"],
            "sample_id": sample["sample_id"],
            "requested_length": sample["requested_length"],
            "sample_index": sample["sample_index"],
            "seed": sample["seed"],
            "path": str(sample["path"]),
            "path_sha256": sha256_file(sample["path"]) if Path(sample["path"]).exists() else "",
            "length": int(matrix.shape[0]),
            "actual_length": int(sample["actual_length"]),
        }
        repair_rows.append({**base, **repairability_metrics(matrix)})
        trace_rows.append({**base, **trace_metrics(projection.coordinates)})
    return pd.DataFrame(repair_rows), pd.DataFrame(trace_rows)


def _summary_by_length(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for (source_type, length), group in frame.groupby(["source_type", "requested_length"], dropna=False):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                continue
            rows.append(
                {
                    "source_type": source_type,
                    "requested_length": int(length),
                    "metric": metric,
                    "count": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p10": float(np.percentile(values, 10)),
                    "p90": float(np.percentile(values, 90)),
                    "bootstrap_ci_low": _bootstrap_mean_ci(values, 0)[0] if values.size > 1 else float("nan"),
                    "bootstrap_ci_high": _bootstrap_mean_ci(values, 0)[1] if values.size > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    means = [float(np.mean(rng.choice(values, size=values.size, replace=True))) for _ in range(200)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _corruption_rows(real_samples: list[dict[str, Any]], noise_levels: list[float]) -> pd.DataFrame:
    rows = []
    for sample in real_samples:
        original = symmetrize_zero_diagonal(sample["matrix"])
        original_projection = classical_mds_rank3_projection(original)
        for noise in noise_levels:
            corrupted = corrupt_distance_matrix(
                original,
                noise_angstrom=noise,
                seed=int(sample["seed"]) + int(noise * 1000),
            )
            projection = classical_mds_rank3_projection(corrupted)
            mask = upper_pair_mask(original.shape[0], min_separation=1)
            delta = projection.projected_distances[mask] - original[mask]
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "requested_length": sample["requested_length"],
                    "noise_angstrom": float(noise),
                    "distance_rmse_angstrom": float(np.sqrt(np.mean(delta * delta))),
                    "distance_mae_angstrom": float(np.mean(np.abs(delta))),
                    "coordinate_rmsd_reflection_allowed": kabsch_rmsd(
                        original_projection.coordinates, projection.coordinates, allow_reflection=True
                    ),
                    **{
                        f"contact_{int(threshold)}A_{name}": value
                        for threshold in (6.0, 8.0, 10.0)
                        for name, value in contact_metrics(
                            original, projection.projected_distances, threshold=threshold, mask=mask
                        ).items()
                    },
                }
            )
    return pd.DataFrame(rows)


def _pareto_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["offdiagonal_rmse_angstrom", "negative_eigenvalue_mass_fraction_before_projection"]
    rows = []
    e004 = frame[frame["source_type"] == "E004"]
    for _length, group in e004.groupby("requested_length"):
        values = group.sort_values(metrics).reset_index(drop=True)
        if values.empty:
            continue
        for label, index in [("best", 0), ("median", len(values) // 2), ("worst", len(values) - 1)]:
            row = values.iloc[index].to_dict()
            row["representative_label"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def _counts_by_source_and_requested_length(samples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for sample in samples:
        source_counts = counts.setdefault(str(sample["source_type"]), {})
        length = str(int(sample["requested_length"]))
        source_counts[length] = source_counts.get(length, 0) + 1
    return counts


def _validate_protocol_cohorts(
    protocol: dict[str, Any], *, label: str, required_lengths: list[int], limit_per_length: int | None
) -> None:
    counts = {int(k): int(v) for k, v in protocol.get("sample_counts_by_length", {}).items()}
    required_count = limit_per_length if limit_per_length is not None else 1
    for requested_length in required_lengths:
        observed = counts.get(requested_length, 0)
        if observed < required_count:
            raise ValueError(
                f"Insufficient {label} samples for requested_length={requested_length}: "
                f"require {required_count}, found {observed}"
            )


def _write_figures(frame: pd.DataFrame, corruption: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for metric, name in [
        ("offdiagonal_rmse_angstrom", "repair_distortion_by_length.png"),
        ("contact_8A_f1", "contact_preservation_by_length.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        if not frame.empty:
            for label, group in frame.groupby("source_type"):
                grouped = group.groupby("requested_length")[metric].mean()
                ax.plot(grouped.index, grouped.values, marker="o", label=str(label))
        ax.set_xlabel("Requested length")
        ax.set_ylabel(metric)
        ax.legend(loc="best")
        fig.tight_layout()
        tmp = figures / f".{name}.tmp"
        fig.savefig(tmp, dpi=160, format="png")
        plt.close(fig)
        tmp.replace(figures / name)
    fig, ax = plt.subplots(figsize=(7, 4))
    if not corruption.empty:
        grouped = corruption.groupby("noise_angstrom")["distance_rmse_angstrom"].mean()
        ax.plot(grouped.index, grouped.values, marker="o")
    ax.set_xlabel("Gaussian noise (A)")
    ax.set_ylabel("Recovered distance RMSE (A)")
    fig.tight_layout()
    tmp = figures / ".corruption_calibration.png.tmp"
    fig.savefig(tmp, dpi=160, format="png")
    plt.close(fig)
    tmp.replace(figures / "corruption_calibration.png")


def _write_representatives(frame: pd.DataFrame, samples: list[dict[str, Any]], output_dir: Path) -> None:
    by_id = {sample["sample_id"]: sample for sample in samples}
    rep_dir = output_dir / "representative_structures"
    rep_dir.mkdir(parents=True, exist_ok=True)
    for _, row in frame.iterrows():
        sample = by_id.get(str(row["sample_id"]))
        if sample is None:
            continue
        projection = classical_mds_rank3_projection(sample["matrix"])
        export_ca_pdb(
            rep_dir / f"{row['requested_length']}_{row['representative_label']}_{row['sample_id']}.pdb",
            projection.coordinates,
            sample_id=str(row["sample_id"]),
        )


def run_repairability_analysis(
    *,
    candidate_dir: Path,
    output_dir: Path,
    baseline_dir: Path | None = None,
    real_manifest: Path | None = None,
    expected_candidate_checkpoint_sha256: str | None = None,
    restart: bool = False,
    resume: bool = False,
    limit_per_length: int | None = None,
) -> Path:
    """Run the read-only repairability analysis."""
    started_at = time.perf_counter()
    del resume
    # Validate the completed candidate identity before touching derived output.
    protocol = _validate_protocol(candidate_dir, expected_checkpoint_sha256=expected_candidate_checkpoint_sha256)
    required_lengths = sorted(int(k) for k in protocol.get("sample_counts_by_length", {}))
    if not required_lengths:
        raise ValueError("Candidate protocol has no requested-length cohorts")
    _validate_protocol_cohorts(
        protocol, label="E004", required_lengths=required_lengths, limit_per_length=limit_per_length
    )
    if baseline_dir is not None:
        baseline_protocol = _validate_protocol(baseline_dir, expected_checkpoint_sha256=None)
        _validate_protocol_cohorts(
            baseline_protocol, label="E002", required_lengths=required_lengths, limit_per_length=limit_per_length
        )

    real_frame = None
    selected_real_frame = None
    if real_manifest is not None:
        real_frame = _read_manifest_table(real_manifest)
        selected_real_frame = _select_real_control_rows(
            real_frame,
            manifest_path=real_manifest,
            required_lengths=required_lengths,
            limit_per_length=limit_per_length,
        )

    raw_paths = _evaluation_input_paths(candidate_dir)
    if baseline_dir is not None:
        raw_paths.extend(_evaluation_input_paths(baseline_dir))
    if real_manifest is not None and real_frame is not None:
        raw_paths.append(real_manifest)
        raw_paths.extend(Path(path) for path in real_frame["path"].tolist())
    before = _hash_inputs(raw_paths)

    if output_dir.exists() and restart:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_generated_samples(candidate_dir, label="E004", limit_per_length=limit_per_length)
    if baseline_dir is not None:
        samples.extend(_load_generated_samples(baseline_dir, label="E002", limit_per_length=limit_per_length))
    real_samples = (
        _load_manifest_samples(selected_real_frame, label="real_control") if selected_real_frame is not None else []
    )
    samples.extend(real_samples)
    repair, trace = _sample_rows(samples)
    corruption = _corruption_rows(real_samples, [0.25, 0.5, 1.0, 2.0, 4.0]) if real_samples else pd.DataFrame()
    pareto = _pareto_candidates(repair)

    replace_dataframe(
        output_dir / "per_sample_repairability.parquet",
        lambda path: repair.to_parquet(path, index=False),
    )
    replace_dataframe(output_dir / "projected_trace_metrics.parquet", lambda path: trace.to_parquet(path, index=False))
    replace_dataframe(
        output_dir / "real_corruption_calibration.parquet",
        lambda path: corruption.to_parquet(path, index=False),
    )
    _summary_by_length(
        repair,
        [
            "relative_frobenius_stress",
            "offdiagonal_rmse_angstrom",
            "offdiagonal_mae_angstrom",
            "contact_8A_f1",
            "negative_eigenvalue_mass_fraction_before_projection",
            "rank3_residual_energy_fraction_before_projection",
        ],
    ).to_csv(output_dir / "repairability_by_length.csv", index=False)
    _summary_by_length(
        trace,
        [
            "ca_adjacent_distance_mean",
            "ca_adjacent_rmse_to_3p8A",
            "radius_of_gyration",
            "maximum_chain_discontinuity",
        ],
    ).to_csv(output_dir / "projected_trace_by_length.csv", index=False)
    if not corruption.empty:
        corruption.groupby(["requested_length", "noise_angstrom"], as_index=False).agg(
            distance_rmse_angstrom_mean=("distance_rmse_angstrom", "mean"),
            coordinate_rmsd_reflection_allowed_mean=("coordinate_rmsd_reflection_allowed", "mean"),
            contact_8A_f1_mean=("contact_8A_f1", "mean"),
            count=("sample_id", "count"),
        ).to_csv(output_dir / "corruption_recovery_by_length.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "corruption_recovery_by_length.csv", index=False)
    pareto.to_csv(output_dir / "pareto_candidates.csv", index=False)
    _write_representatives(pareto, samples, output_dir)
    _write_figures(repair, corruption, output_dir)

    after = _hash_inputs(raw_paths)
    if before != after:
        raise RuntimeError("Raw input hash changed during repairability analysis")
    atomic_write_json(
        output_dir / "repairability_protocol.json",
        {
            "status": "completed",
            "candidate_dir": str(candidate_dir),
            "baseline_dir": str(baseline_dir) if baseline_dir else None,
            "real_manifest": str(real_manifest) if real_manifest else None,
            "expected_candidate_checkpoint_sha256": expected_candidate_checkpoint_sha256,
            "candidate_checkpoint_sha256": protocol.get("checkpoint_sha256"),
            "candidate_status": protocol.get("status"),
            "sample_count": int(len(samples)),
            "sample_counts_by_source_and_requested_length": _counts_by_source_and_requested_length(samples),
            "raw_input_hashes_before": before,
            "raw_input_hashes_after": after,
            "raw_inputs_unchanged": True,
            "raw_input_hashes_preserved": True,
            "runtime_seconds": float(time.perf_counter() - started_at),
            "method": "rank-3 classical-MDS projection; not claimed nearest EDM",
            "representative_selection": (
                "Within each E004 requested-length cohort, rows are ordered lexicographically by "
                "off-diagonal projection RMSE and then negative-eigenvalue mass. Labels mark best, middle-index, "
                "and worst ranking positions; they are not minimum-stress labels or a formal Pareto frontier. "
                "For two samples, median and worst intentionally identify the same sample."
            ),
            "short_range_definition": "3 <= |i-j| < 24",
            "long_range_definition": "|i-j| >= 24",
            "chirality_note": (
                "Distance matrices do not determine absolute chirality; reflection is allowed for coordinate recovery."
            ),
        },
    )
    _write_readme(output_dir)
    return output_dir / "repairability_protocol.json"


def _write_readme(output_dir: Path) -> None:
    atomic_write_text(
        output_dir / "README.md",
        """# E004 Repairability Analysis

This is a derived, read-only analysis of rank-3 classical-MDS repairability and
projected C-alpha trace plausibility. It does not load models, generate samples,
train, preprocess, split, or mutate source NPZ/protocol/metric files.

The projection computes `G = -0.5 J D^2 J`, retains the three largest positive
eigenvalues, reconstructs coordinates, and compares the projected distance
matrix with the source matrix. This is called a rank-3 classical-MDS projection,
not a mathematically nearest distance matrix.

Short range is `3 <= |i-j| < 24`; long range is `|i-j| >= 24`. Virtual
dihedrals are reported with the caveat that distance matrices do not determine
absolute chirality.

Real controls are selected independently within the evaluator's
`requested_length` cohorts, while `actual_length` retains each matrix's true
size. Representative labels are deterministic repairability ranking positions
using off-diagonal projection RMSE followed by negative-eigenvalue mass. They
are not minimum projection-stress labels or a formal Pareto frontier. With two
samples, median and worst intentionally refer to the same sample.
""",
    )


def run_synthetic_smoke(output_dir: Path) -> Path:
    """Run a tiny synthetic smoke without real evaluation inputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    coords = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.0, 2.0, 0.0], [10.0, 3.5, 1.0]])
    matrix = pairwise_distances(coords)
    generated_dir = output_dir / "generated" / "N0004"
    generated_dir.mkdir(parents=True, exist_ok=True)
    path = generated_dir / "N0004_i00000_seed1.npz"
    tmp = generated_dir / ".N0004_i00000_seed1.npz.tmp"
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            sample_id="N0004_i00000_seed1",
            requested_length=4,
            sample_index=0,
            seed=1,
            physical_distance_matrix_angstrom=matrix,
        )
    tmp.replace(path)
    atomic_write_json(
        output_dir / "protocol.json",
        {
            "status": "completed",
            "checkpoint_sha256": "synthetic",
            "sample_counts_by_length": {"4": 1},
        },
    )
    manifest = output_dir / "synthetic_manifest.parquet"
    pd.DataFrame([{"sample_id": "synthetic", "path": str(path), "length": 4, "requested_length": 4}]).to_parquet(
        manifest, index=False
    )
    return run_repairability_analysis(
        candidate_dir=output_dir,
        output_dir=output_dir / "repairability",
        real_manifest=manifest,
        restart=True,
        limit_per_length=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--real-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-candidate-checkpoint-sha256", default=None)
    parser.add_argument("--limit-per-length", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        print(run_synthetic_smoke(args.output_dir))
        return
    if args.candidate_dir is None:
        print("--candidate-dir is required unless --synthetic-smoke is used", file=sys.stderr)
        raise SystemExit(2)
    print(
        run_repairability_analysis(
            candidate_dir=args.candidate_dir,
            baseline_dir=args.baseline_dir,
            real_manifest=args.real_manifest,
            output_dir=args.output_dir,
            expected_candidate_checkpoint_sha256=args.expected_candidate_checkpoint_sha256,
            restart=bool(args.restart),
            resume=bool(args.resume),
            limit_per_length=args.limit_per_length,
        )
    )


if __name__ == "__main__":
    main()
