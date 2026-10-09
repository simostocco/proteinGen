"""Certification of shared-reference preprocessing and historical replay only."""

# ruff: noqa: I001, E501
from e014_bootstrap import ROOT
import argparse
import json
import numpy as np
import pyarrow.parquet as pq
from diagnose_e014_high_noise import OUT, BASE_MARGINAL
from protein_distance_diffusion.evaluation.high_noise_resolution import LABELS
from protein_distance_diffusion.training.e014_runtime import atomic_json, sha

assert OUT.is_relative_to(ROOT)


def keyed(path):
    rows = pq.read_table(path, use_threads=False).to_pylist()
    return {(r["sample_id"], r["timestep"], r["population"]): np.array(r["descriptor"]) for r in rows}


def replay():
    label = LABELS[2]
    assert (OUT / f"{label}.INFERENCE_COMPLETE.json").exists()
    old = keyed(BASE_MARGINAL / "one_step_descriptors.parquet")
    new = keyed(OUT / f"{label}.descriptors.parquet")
    differences = [np.nanmax(np.abs(new[k] - v)) for k, v in old.items()]
    gaussold = keyed(BASE_MARGINAL / "gaussian_prefix_descriptors.parquet")
    gaussnew = keyed(OUT / f"{label}.gaussian_descriptors.parquet")
    gaussdifferences = [np.nanmax(np.abs(gaussnew[k] - v)) for k, v in gaussold.items() if k[1] in [490, 480, 440]]
    historical = json.loads((ROOT / "e014/evaluation/epoch_1_fraction_0.50_development.metrics.json").read_text())
    fresh = pq.read_table(OUT / f"{label}.paired.parquet", use_threads=False).to_pylist()
    rows = {(r["sample_id"], r["timestep"]): r for r in fresh}
    vd = max(abs(rows[r["sample_id"], r["timestep"]]["v_mse"] - r["v_mse"]) for r in historical)
    local = max(
        abs(rows[r["sample_id"], r["timestep"]]["local_mse_angstrom2"] - r["local_mse_angstrom2"]) for r in historical
    )
    rgs = max(abs(rows[r["sample_id"], r["timestep"]]["radius_ratio"] - r["radius_ratio"]) for r in historical)
    assert max(differences) <= 5e-6 and max(gaussdifferences) <= 5e-6 and vd <= 5e-6 and rgs <= 5e-6
    assert all(
        np.isclose(
            rows[r["sample_id"], r["timestep"]]["local_mse_angstrom2"], r["local_mse_angstrom2"], atol=1e-4, rtol=1e-5
        )
        for r in historical
    )
    atomic_json(
        OUT / "REPLAY_CERTIFICATION.json",
        dict(
            marginal_descriptor_conditions=len(differences),
            maximum_descriptor_absolute_error=float(max(differences)),
            gaussian_overlap_descriptor_conditions=len(gaussdifferences),
            maximum_gaussian_descriptor_absolute_error=float(max(gaussdifferences)),
            paired_conditions=len(historical),
            maximum_v_mse_absolute_error=vd,
            maximum_local_mse_absolute_error=local,
            maximum_rg_ratio_absolute_error=rgs,
            passed=True,
        ),
    )
    print("frozen50% previous protocols replay certified", flush=True)


def shared_references():
    data = json.loads((OUT / "metrics.json").read_text())
    for phase, key in [("one_step", "marginal"), ("gaussian_prefix", "gaussian")]:
        by = {}
        for g in data[key]:
            identity = (g["timestep"], g["length_stratum"])
            # Exact floats expected: identical stored references and deterministic code.
            current = (
                g["reference_preprocessing"],
                g["bandwidth"],
                g["descriptor_columns"],
                g["raw_descriptor_distributions"]["ref_a"],
                g["raw_descriptor_distributions"]["ref_b"],
            )
            if identity in by:
                assert current == by[identity], (phase, identity, g["checkpoint"])
            else:
                by[identity] = current
    identity = json.loads((OUT / "checkpoint_identity.json").read_text())
    assert all(sha(c["path"]) == c["sha256"] for c in identity["checkpoints"])
    atomic_json(
        OUT / "SHARED_REFERENCE_CERTIFICATION.json",
        dict(
            exact_shared_preprocessing=True,
            exact_shared_bandwidth=True,
            exact_shared_descriptor_columns=True,
            exact_shared_raw_reference_summaries=True,
            all_checkpoint_hashes_unchanged=True,
            passed=True,
        ),
    )
    print("shared references and no checkpoint-specific preprocessing certified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["replay", "shared"])
    args = parser.parse_args()
    replay() if args.phase == "replay" else shared_references()
