#!/usr/bin/env python3
"""Fixed CPU Cartesian oracle, paired analysis and complete result reproduction."""

import argparse
import hashlib
import json
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import yaml

from protein_distance_diffusion.training.e010_cartesian_oracle_v5 import (
    classify,
    contribution,
    convergence_pairs,
    metric_row,
    solve,
    summarize,
)
from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from protein_distance_diffusion.training.e010_recurrent_capacity import batch
from scripts.run_e010_phase4d_bounded_oracle_v4 import CONFIG as V4_CONFIG
from scripts.run_e010_phase4d_bounded_oracle_v4 import OUT as V4
from scripts.run_e010_phase4d_bounded_oracle_v4 import safe
from scripts.run_e010_phase4d_recurrent_capacity_v3 import OUT as V3
from scripts.run_e010_phase4d_recurrent_capacity_v3 import ROOT, load_cache, write

OUT = V4.parent / "cartesian_oracle_v5"
CONFIG = ROOT / "configs/e010_phase4d_cartesian_oracle_v5.yaml"
CACHE = None
CFG = None
MODE = None


def setup():
    cfg = yaml.safe_load(CONFIG.read_text())
    previous = yaml.safe_load(V4_CONFIG.read_text())
    for key in (
        "optimizer",
        "objective",
        "bound",
        "safety",
        "initialization",
        "randomness",
        "precision",
        "device",
        "threads",
    ):
        if cfg[key] != previous[key]:
            raise ValueError(f"v4 contract changed: {key}")
    if cfg["recurrence"]["steps"] != 4:
        raise ValueError("K changed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return cfg


def register():
    setup()
    _, manifest = load_cache()
    paths = subprocess.check_output(
        ["git", "ls-files", str(V4.parent.relative_to(ROOT))], cwd=ROOT, text=True
    ).splitlines()
    dependencies = [
        "models/e010_hybrid_local.py",
        "training/e010_phase4d.py",
        "training/local_geometry.py",
        "training/e010_phase4d_diagnostic.py",
        "training/e010_phase4d_objective_v2.py",
        "training/e010_recurrent_capacity.py",
        "training/e010_bounded_oracle_v4.py",
        "training/e010_cartesian_oracle_v5.py",
    ]
    pins = {str(ROOT / p): file_hash(ROOT / p) for p in paths}
    for rel in dependencies:
        p = ROOT / "src/protein_distance_diffusion" / rel
        pins[str(p)] = file_hash(p)
    for p in [
        CONFIG,
        V4_CONFIG,
        ROOT / "scripts/run_e010_phase4d_cartesian_oracle_v5.py",
        ROOT / "docs/e010_phase4d_cartesian_oracle_v5.md",
    ]:
        pins[str(p)] = file_hash(p)
    write(
        OUT / "execution_contract.json",
        {
            "source_result_commit": "f9b42c36459f17c9f7fb42c20496b7c7b58a4ea6",
            "protected_sha256": pins,
            "cache_sha256": manifest["cache_sha256"],
            "workers": 4,
            "threads_each": 1,
            "complete_reproduction_required_before_commit": True,
            "cuda_used": False,
            "neural_parameters": 0,
        },
    )


def canonical(record):
    value = json.loads(json.dumps(record))
    value["optimizer"].pop("runtime_seconds")
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def example(index):
    setup()
    record = CACHE["records"][index]
    b = {k: v.double() if v.is_floating_point() else v for k, v in batch(CACHE, [index], "cpu").items()}
    baseline = metric_row(b["pg"], b, record)
    tr, log = solve(b, examples_total=60, quartets_total=13029, settings=CFG["optimizer"])
    states = [
        metric_row(p, b, record, step_delta=tr["steps"][t - 1]["delta"] if t else None)
        for t, p in enumerate(tr["states"])
    ]
    assert all(r["finite"] and r["step_correction_max"] <= 0.04 and r["displacement_max"] <= 0.160001 for r in states)
    values = [contribution(baseline)] + [h["objective_contribution"] for h in log["history"]]
    log.update(
        initial_objective=contribution(baseline),
        final_objective=contribution(states[-1]),
        line_search_failures=None,
        line_search_failure_availability="not exposed by unchanged v4 solver",
        explicit_optimizer_restarts=0,
        objective_trajectory_summary={
            "initial": values[0],
            "final": contribution(states[-1]),
            "minimum_at_recorded_checks": min(values),
            "recorded_checks": len(log["history"]),
        },
    )
    result = {
        "record": record,
        "baseline": baseline,
        "oracle": states[-1],
        "states": states,
        "optimizer": log,
        "state_coordinate_sha256": [hashlib.sha256(p.contiguous().numpy().tobytes()).hexdigest() for p in tr["states"]],
    }
    if MODE == "reproduce":
        expected = json.loads((OUT / "examples" / f"example_{index:02d}.json").read_text())
        a, b = canonical(expected), canonical(result)
        if a != b:
            raise ValueError(f"reproduction mismatch: example {index}")
        write(
            OUT / "reproduction" / f"example_{index:02d}.json",
            {
                "index": index,
                "exact_non_timing_match": True,
                "scientific_record_sha256": hashlib.sha256(a.encode()).hexdigest(),
                "repeat_runtime_seconds": log["runtime_seconds"],
                "converged": log["converged"],
                "iterations": log["iterations"],
            },
        )
    else:
        write(OUT / "examples" / f"example_{index:02d}.json", result)
    return index, record, log["converged"], log["iterations"], log["projected_gradient_residual"]


def paired(old, new):
    rows = []
    for i, (a, b) in enumerate(zip(old, new, strict=True)):
        if a["record"] != b["record"]:
            raise ValueError("panel identities changed")
        x, y = a["oracle"], b["oracle"]
        oa, ob = contribution(x), contribution(y)
        gain_a = 100 * (1 - x["mean_local_rmse"] / a["baseline"]["mean_local_rmse"])
        gain_b = 100 * (1 - y["mean_local_rmse"] / b["baseline"]["mean_local_rmse"])
        tol = CFG["paired_equivalence"]
        equivalent = (
            abs(ob - oa) <= tol["objective_relative_tolerance"] * max(1, abs(oa))
            and abs(y["mean_local_rmse"] - x["mean_local_rmse"]) <= tol["local_rmse_absolute_angstrom"]
            and abs(y["raw_cartesian"] - x["raw_cartesian"]) <= tol["cartesian_absolute_angstrom_squared"]
            and abs(y["continuous_chiral_loss"] - x["continuous_chiral_loss"]) <= tol["chiral_loss_absolute"]
            and abs(y["displacement_rms"] - x["displacement_rms"]) <= tol["correction_rms_absolute_angstrom"]
            and y["chirality_inversions"] == x["chirality_inversions"]
        )
        rows.append(
            {
                "index": i,
                "record": b["record"],
                "v4_converged": a["optimizer"]["converged"],
                "v5_converged": b["optimizer"]["converged"],
                "objective_v4": oa,
                "objective_v5": ob,
                "objective_difference": ob - oa,
                "objective_relative_difference": (ob - oa) / max(1, abs(oa)),
                "local_gain_v4_pct": gain_a,
                "local_gain_v5_pct": gain_b,
                "local_gain_difference_percentage_points": gain_b - gain_a,
                "cartesian_difference": y["raw_cartesian"] - x["raw_cartesian"],
                "chiral_loss_difference": y["continuous_chiral_loss"] - x["continuous_chiral_loss"],
                "inversion_difference": y["chirality_inversions"] - x["chirality_inversions"],
                "correction_rms_v4": x["displacement_rms"],
                "correction_rms_v5": y["displacement_rms"],
                "correction_max_v4": x["displacement_max"],
                "correction_max_v5": y["displacement_max"],
                "effectively_equivalent": equivalent,
            }
        )
    groups = {}
    for name, predicate in [("converged_both", lambda a, b: a and b), ("newly_converged", lambda a, b: not a and b)]:
        ids = [i for i, r in enumerate(rows) if predicate(r["v4_converged"], r["v5_converged"])]
        groups[name] = {
            "indices": ids,
            "examples": len(ids),
            "equivalent_examples": sum(rows[i]["effectively_equivalent"] for i in ids),
            "lower_objective_v5_examples": sum(
                rows[i]["objective_difference"] < -1e-6 * max(1, abs(rows[i]["objective_v4"])) for i in ids
            ),
            "mean_objective_difference": sum(rows[i]["objective_difference"] for i in ids) / len(ids) if ids else None,
            "baseline": summarize([new[i]["baseline"] for i in ids]) if ids else None,
            "v4": summarize([old[i]["oracle"] for i in ids]) if ids else None,
            "v5": summarize([new[i]["oracle"] for i in ids]) if ids else None,
        }
    return {"per_example": rows, "groups": groups, "convergence_pairs": convergence_pairs(old, new)}


def aggregate():
    old = [json.loads((V4 / "examples" / f"example_{i:02d}.json").read_text()) for i in range(60)]
    new = [json.loads((OUT / "examples" / f"example_{i:02d}.json").read_text()) for i in range(60)]
    baseline, final = summarize([r["baseline"] for r in new]), summarize([r["oracle"] for r in new])
    groups = {"overall": list(range(60))}
    groups.update(
        {
            f"condition_{c}": [i for i, r in enumerate(new) if str(r["record"]["condition"]) == c]
            for c in ["50", "250", "450"]
        }
    )
    groups.update(
        {f"stratum_{s}": [i for i, r in enumerate(new) if r["record"]["stratum"] == s] for s in final["by_stratum"]}
    )
    conv = {
        k: {
            "examples": len(ids),
            "v4_converged": sum(old[i]["optimizer"]["converged"] for i in ids),
            "v5_converged": sum(new[i]["optimizer"]["converged"] for i in ids),
            "pairs": convergence_pairs([old[i] for i in ids], [new[i] for i in ids]),
        }
        for k, ids in groups.items()
    }
    gains = {
        c: 100 * (1 - final["by_condition"][c]["mean_local_rmse"] / baseline["by_condition"][c]["mean_local_rmse"])
        for c in ["50", "250", "450"]
    }
    safety = {c: safe(final["by_condition"][c], baseline["by_condition"][c], CFG) for c in gains}
    write(OUT / "paired_comparison.json", paired(old, new))
    write(
        OUT / "oracle_result.json",
        {
            "baseline": baseline,
            "oracle": final,
            "convergence": conv,
            "condition_local_gain_pct": gains,
            "condition_safety": safety,
            "classification": classify(
                conv["overall"]["v5_converged"], conv["condition_450"]["v5_converged"], gains, safety
            ),
            "iterations": [r["optimizer"]["iterations"] for r in new],
            "cuda_used": False,
            "neural_training_launched": False,
            "objective_initial": sum(r["optimizer"]["initial_objective"] for r in new),
            "objective_final": sum(r["optimizer"]["final_objective"] for r in new),
            "sum_example_runtime_seconds": sum(r["optimizer"]["runtime_seconds"] for r in new),
        },
    )
    print("Aggregate", conv["overall"], gains, safety, flush=True)


def execute(mode):
    global CACHE, CFG, MODE
    CFG = setup()
    MODE = mode
    contract = json.loads((OUT / "execution_contract.json").read_text())
    assert_file_pins(contract["protected_sha256"])
    CACHE, manifest = load_cache()
    directory = "examples" if mode == "run" else "reproduction"
    pending = [i for i in range(60) if not (OUT / directory / f"example_{i:02d}.json").exists()]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
        for future in as_completed([pool.submit(example, i) for i in pending]):
            print(mode, *future.result(), flush=True)
    assert_file_pins(contract["protected_sha256"])
    assert_file_pins(manifest["protected_input_sha256"])
    if file_hash(V3 / "frozen_panel.npz") != contract["cache_sha256"]:
        raise ValueError("frozen cache mutation")
    if mode == "run":
        aggregate()
    else:
        write(
            OUT / "reproduction_result.json",
            {
                "examples": 60,
                "all_exact_non_timing_matches": True,
                "states_coordinates_metrics_histories_convergence_reproduced": True,
                "fresh_zero_initialization": True,
                "same_workers_threads_optimizer": True,
                "protected_hashes_verified": True,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["register", "run", "reproduce"])
    args = parser.parse_args()
    register() if args.mode == "register" else execute(args.mode)
