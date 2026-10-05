#!/usr/bin/env python
"""Evaluate prepared Distance-AF benchmark result tables without launching jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from protein_distance_diffusion.evaluation.distance_af import (
    evaluate_restraint_satisfaction,
    heldout_contact_metrics,
    reject_malformed_prediction_table,
)


def _load_matrix(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    for key in ("prediction_distance_matrix_angstrom", "distance_matrix", "physical_distance_matrix_angstrom"):
        if key in data:
            return np.asarray(data[key], dtype=np.float64)
    raise ValueError(f"No supported prediction matrix key found in {path}")


def evaluate_distance_af_benchmark(job_manifest: Path, predictions: Path, output: Path) -> Path:
    jobs = pd.read_parquet(job_manifest)
    preds = pd.read_parquet(predictions) if predictions.suffix == ".parquet" else pd.read_csv(predictions)
    reject_malformed_prediction_table(preds)
    rows = []
    for _, pred in preds.iterrows():
        job = jobs[jobs["job_id"] == pred["job_id"]]
        if job.empty:
            raise ValueError(f"Prediction references unknown job_id: {pred['job_id']}")
        job_row = job.iloc[0]
        target = _load_matrix(Path(job_row["matrix_path"]))
        predicted = _load_matrix(Path(pred["prediction_path"]))
        restraints = pd.read_parquet(job_row["heldout_restraints_path"])
        guidance_path = Path(job_row["guidance_restraints_path"])
        guidance = []
        for line in guidance_path.read_text().splitlines():
            if not line.strip():
                continue
            i, j, value = line.split(",")
            guidance.append(
                {
                    "i_zero_based": int(i) - 1,
                    "j_zero_based": int(j) - 1,
                    "i_distance_af": int(i),
                    "j_distance_af": int(j),
                    "target_distance_angstrom": float(value),
                    "split": "guidance",
                }
            )
        all_restraints = pd.concat([pd.DataFrame(guidance), restraints], ignore_index=True)
        row = {
            "job_id": pred["job_id"],
            "prediction_kind": pred["prediction_kind"],
            "status": pred["status"],
        }
        guidance_metrics = evaluate_restraint_satisfaction(target, predicted, all_restraints, split="guidance")
        heldout_metrics = evaluate_restraint_satisfaction(target, predicted, all_restraints, split="heldout")
        contact = heldout_contact_metrics(target, predicted, all_restraints)
        row.update({f"guidance_{key}": value for key, value in guidance_metrics.items()})
        row.update({f"heldout_{key}": value for key, value in heldout_metrics.items()})
        row.update({f"heldout_contact_8A_{key}": value for key, value in contact.items()})
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(evaluate_distance_af_benchmark(args.job_manifest, args.predictions, args.output))


if __name__ == "__main__":
    main()
