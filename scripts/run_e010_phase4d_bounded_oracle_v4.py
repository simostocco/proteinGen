#!/usr/bin/env python3
"""CPU independent bounded correction oracle; immutable versioned results."""

import argparse
import json
import subprocess

import torch
import yaml

from protein_distance_diffusion.training.e010_bounded_oracle_v4 import metric_row, solve, summarize
from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from protein_distance_diffusion.training.e010_recurrent_capacity import batch
from scripts.run_e010_phase4d_recurrent_capacity_v3 import OUT as V3
from scripts.run_e010_phase4d_recurrent_capacity_v3 import ROOT, load_cache, write

OUT = V3.parent / "bounded_oracle_v4"
CONFIG = ROOT / "configs/e010_phase4d_bounded_oracle_v4.yaml"


def setup():
    cfg = yaml.safe_load(CONFIG.read_text())
    if (
        cfg["recurrence"]["steps"] != 4
        or cfg["bound"]["s_max_angstrom"] != 0.04
        or cfg["objective"]["beta"] != 16.8
        or cfg["objective"]["gamma"] != 2
    ):
        raise ValueError("oracle scientific contract mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return cfg


def preflight():
    cfg = setup()
    t = torch.arange(16, dtype=torch.float64)
    pg = torch.stack((2 * t, torch.sin(t), torch.cos(t)), -1)[None]
    target = pg.clone()
    target[:, 3:13] += 0.01 * torch.stack((torch.sin(t[3:13]), torch.cos(t[3:13]), torch.sin(2 * t[3:13])), -1)
    b = {"pg": pg, "source": pg.clone(), "target": target, "mask": torch.ones(1, 16, dtype=torch.bool)}
    tr, log = solve(b, examples_total=60, quartets_total=13029, settings=cfg["optimizer"])
    write(
        OUT / "synthetic_preflight.json",
        {
            "convergence": log,
            "max_displacement_angstrom": float((tr["prediction"] - pg).norm(dim=-1).max()),
            "settings_sha256": file_hash(CONFIG),
            "panel_used": False,
        },
    )
    print("Synthetic convergence", log["converged"], log["iterations"], log["projected_gradient_residual"], flush=True)


def register():
    setup()
    _, manifest = load_cache()
    tracked = subprocess.check_output(["git", "ls-files", str(V3.relative_to(ROOT))], cwd=ROOT, text=True).splitlines()
    paths = [ROOT / p for p in tracked] + [
        CONFIG,
        ROOT / "scripts/run_e010_phase4d_bounded_oracle_v4.py",
        ROOT / "src/protein_distance_diffusion/training/e010_bounded_oracle_v4.py",
    ]
    write(
        OUT / "execution_contract.json",
        {
            "source_result_commit": "43864b38001c2ee0118bc8b7530b547cba4ca5f1",
            "frozen_panel_sha256": manifest["cache_sha256"],
            "protected_v3_and_implementation_sha256": {str(p): file_hash(p) for p in paths},
            "neural_parameters": 0,
            "cuda_used": False,
        },
    )


def safe(m, b, cfg):
    s = cfg["safety"]
    return (
        m["raw_cartesian"] <= (1 + s["raw_cartesian_relative_tolerance"]) * b["raw_cartesian"]
        and m["aligned_rmsd"] <= 1.01 * b["aligned_rmsd"]
        and m["continuous_chiral_loss"] <= (1 + s["continuous_chiral_relative_tolerance"]) * b["continuous_chiral_loss"]
        and m["chirality_inversions"] <= b["chirality_inversions"]
        and m["chirality_assessable"] == b["chirality_assessable"]
        and m["chirality_assessability_lost"] == 0
        and m["all_finite"]
        and m["frame_assessability_preserved"]
        and all(m["local_rmse"][k] <= b["local_rmse"][k] for k in ["1", "2", "3"])
    )


def optimistic_local_bound(b):
    """Independent pair residual lower bound, relaxing all coupling/safety.

    Each interior residue can move at most .16 Å; endpoints cannot move.
    Reverse triangle inequality bounds reduction in each absolute distance
    residual by the sum of endpoint displacement radii. This is an optimistic
    upper bound on achievable local gain, not an oracle outcome or new loss.
    """
    pg, y, m = b["pg"], b["target"], b["mask"]
    radius = torch.zeros_like(m, dtype=pg.dtype)
    radius[:, 1:-1] = 0.16
    values = []
    for k in [1, 2, 3]:
        ok = m[:, k:] & m[:, :-k]
        error = ((pg[:, k:] - pg[:, :-k]).norm(dim=-1) - (y[:, k:] - y[:, :-k]).norm(dim=-1)).abs()
        lower = (error - radius[:, k:] - radius[:, :-k]).clamp_min(0)
        values.append(float(lower[ok].square().mean().sqrt()))
    return {
        "offset_rmse_lower_bounds": dict(zip(["1", "2", "3"], values, strict=True)),
        "mean_local_rmse_lower_bound": sum(values) / 3,
    }


def run():
    cfg = setup()
    contract = json.loads((OUT / "execution_contract.json").read_text())
    assert_file_pins(contract["protected_v3_and_implementation_sha256"])
    cache, manifest = load_cache()
    records = []
    for i, record in enumerate(cache["records"]):
        path = OUT / "examples" / f"example_{i:02d}.json"
        if path.exists():
            records.append(json.loads(path.read_text()))
            continue
        b = {k: v.double() if v.is_floating_point() else v for k, v in batch(cache, [i], "cpu").items()}
        baseline = metric_row(b["pg"], b, record)
        tr, log = solve(b, examples_total=60, quartets_total=13029, settings=cfg["optimizer"])
        states = [
            metric_row(p, b, record, step_delta=tr["steps"][t - 1]["delta"] if t else None)
            for t, p in enumerate(tr["states"])
        ]
        assert all(
            r["step_correction_max"] <= 0.04000000001 and r["displacement_max"] <= 0.160001 and r["finite"]
            for r in states
        )
        result = {
            "record": record,
            "baseline": baseline,
            "oracle": states[-1],
            "states": states,
            "optimizer": log,
            "optimistic_geometry_bound": optimistic_local_bound(b),
        }
        write(path, result)
        records.append(result)
        gain = 100 * (1 - states[-1]["mean_local_rmse"] / baseline["mean_local_rmse"])
        print(
            i,
            record["sample_id"],
            record["condition"],
            "gain",
            round(gain, 5),
            "iterations",
            log["iterations"],
            "converged",
            log["converged"],
            "residual",
            log["projected_gradient_residual"],
            flush=True,
        )
    baseline = summarize([r["baseline"] for r in records])
    final = summarize([r["oracle"] for r in records])
    gains = {
        c: 100 * (1 - final["by_condition"][c]["mean_local_rmse"] / baseline["by_condition"][c]["mean_local_rmse"])
        for c in ["50", "250", "450"]
    }
    safety = {c: safe(final["by_condition"][c], baseline["by_condition"][c], cfg) for c in gains}
    upper_gains = {
        c: 100
        * (
            1
            - sum(
                r["optimistic_geometry_bound"]["mean_local_rmse_lower_bound"]
                for r in records
                if str(r["record"]["condition"]) == c
            )
            / 20
            / baseline["by_condition"][c]["mean_local_rmse"]
        )
        for c in gains
    }
    converged = all(r["optimizer"]["converged"] for r in records)
    if not converged:
        classification = "O5"
    elif gains["450"] >= 5 and safety["450"]:
        classification = "O1"
    elif gains["450"] >= 5 and not safety["450"]:
        classification = "O3"
    elif all(gains[c] >= 5 and safety[c] for c in ["50", "250"]):
        classification = "O4"
    elif upper_gains["450"] < 5:
        classification = "O2"
    else:
        classification = "O5"
    assert_file_pins(contract["protected_v3_and_implementation_sha256"])
    assert_file_pins(manifest["protected_input_sha256"])
    if file_hash(V3 / "frozen_panel.npz") != contract["frozen_panel_sha256"]:
        raise ValueError("cache mutation")
    write(
        OUT / "oracle_result.json",
        {
            "baseline": baseline,
            "oracle": final,
            "condition_local_gain_pct": gains,
            "condition_safety": safety,
            "optimistic_geometry_gain_upper_bound_pct": upper_gains,
            "classification": classification,
            "all_examples_converged": converged,
            "converged_examples": sum(r["optimizer"]["converged"] for r in records),
            "examples": 60,
            "cuda_used": False,
            "neural_training_launched": False,
            "protected_hashes_verified_after": True,
            "runtime_seconds": sum(r["optimizer"]["runtime_seconds"] for r in records),
            "iterations": [r["optimizer"]["iterations"] for r in records],
        },
    )
    print(classification, gains, safety, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "register", "run"])
    args = parser.parse_args()
    {"preflight": preflight, "register": register, "run": run}[args.mode]()
