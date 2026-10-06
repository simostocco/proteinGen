#!/usr/bin/env python3
"""Operational CPU concurrency only; frozen oracle solver/settings unchanged."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from protein_distance_diffusion.training.e010_bounded_oracle_v4 import metric_row, solve
from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from protein_distance_diffusion.training.e010_recurrent_capacity import batch
from scripts.run_e010_phase4d_bounded_oracle_v4 import (
    OUT,
    load_cache,
    optimistic_local_bound,
    run,
    setup,
    write,
)

CACHE = None
CFG = None


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
    assert all(
        r["finite"] and r["step_correction_max"] <= 0.04000000001 and r["displacement_max"] <= 0.160001 for r in states
    )
    result = {
        "record": record,
        "baseline": baseline,
        "oracle": states[-1],
        "states": states,
        "optimizer": log,
        "optimistic_geometry_bound": optimistic_local_bound(b),
    }
    write(OUT / "examples" / f"example_{index:02d}.json", result)
    return index, record, log, 100 * (1 - states[-1]["mean_local_rmse"] / baseline["mean_local_rmse"])


if __name__ == "__main__":
    CFG = setup()
    contract = json.loads((OUT / "execution_contract.json").read_text())
    assert_file_pins(contract["protected_v3_and_implementation_sha256"])
    CACHE, _ = load_cache()
    pending = [i for i in range(60) if not (OUT / "examples" / f"example_{i:02d}.json").exists()]
    operational = OUT / "cpu_execution.json"
    if not operational.exists():
        write(
            operational,
            {
                "workers": 4,
                "threads_each": 1,
                "cuda_used": False,
                "solver_or_settings_changed": False,
                "wrapper_sha256": file_hash(__file__),
                "preparation_commit": "20c20dea94f258ea3032aadd2f31566f0ef6eebe",
                "completed_examples_before_parallel_launch": [i for i in range(60) if i not in pending],
            },
        )
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
        jobs = [pool.submit(example, i) for i in pending]
        for job in as_completed(jobs):
            i, rec, log, gain = job.result()
            print(
                i,
                rec["sample_id"],
                rec["condition"],
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
    run()
