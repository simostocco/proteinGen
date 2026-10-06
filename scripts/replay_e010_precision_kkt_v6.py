#!/usr/bin/env python3
"""Optional pure-float64 replay, only after the fixed-state audit is complete."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import FunctionType

from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins
from scripts import audit_e010_phase4d_precision_kkt_v6 as run
from scripts.run_e010_phase4d_cartesian_oracle_v5 import setup
from scripts.run_e010_phase4d_recurrent_capacity_v3 import load_cache, write


def pure_components(pred, b, *, examples_total, quartets_total):
    return run.audit.objective(pred, b, pure_float64=True, examples_total=examples_total, quartets_total=quartets_total)


def replay(i):
    solver = FunctionType(run.v5.solve.__code__, {**run.v5.solve.__globals__, "objective_components": pure_components})
    solver.__kwdefaults__ = run.v5.solve.__kwdefaults__
    tr, log = solver(run.load_example(i), examples_total=60, quartets_total=13029, settings=setup()["optimizer"])
    b = run.load_example(i)
    old = json.loads((run.V5 / "examples" / f"example_{i:02d}.json").read_text())
    terms = pure_components(tr["prediction"], b, examples_total=60, quartets_total=13029)
    row = run.v5.metric_row(tr["prediction"], b, old["record"], step_delta=tr["steps"][-1]["delta"])
    row["raw_cartesian"] = float(terms["cartesian"]) * 60
    write(
        run.OUT / "precision_replay" / f"example_{i:02d}.json",
        dict(
            index=i,
            record=old["record"],
            start="original_zero",
            optimizer=log,
            metrics=row,
            terms={k: float(v.detach()) for k, v in terms.items()},
            state_coordinate_sha256=[run.tensor_digest(p) for p in tr["states"]],
            step_correction_max=max(float(s["delta"].norm(dim=-1).max()) for s in tr["steps"]),
            objective_arithmetic="pure_float64_only_cartesian_cast_removed",
            optimizer_settings_exactly_historical=True,
        ),
    )
    return i, log["converged"], log["iterations"], log["projected_gradient_residual"]


if __name__ == "__main__":
    run.CFG = run.contract()
    completion = json.loads((run.OUT / "sections_1_to_7_complete.json").read_text())
    assert completion["examples"] == 22
    execution = json.loads((run.OUT / "execution_contract.json").read_text())
    assert_file_pins(execution["protected_sha256"])
    # Never proceed past a fixed-state gradient validation failure.
    for i in run.CFG["stalled_indices"] + run.CFG["control_indices"]:
        r = json.loads((run.OUT / "examples" / f"example_{i:02d}.json").read_text())
        assert not r["finite_differences"]["failed_directions"]
        assert r["chain_rule_relative_error"] <= run.CFG["finite_difference"]["relative_error_tolerance"]
    run.CACHE, manifest = load_cache()
    pending = [
        i for i in run.CFG["stalled_indices"] if not (run.OUT / "precision_replay" / f"example_{i:02d}.json").exists()
    ]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
        for future in as_completed([pool.submit(replay, i) for i in pending]):
            print("float64 replay", future.result(), flush=True)
    assert_file_pins(execution["protected_sha256"])
    assert_file_pins(manifest["protected_input_sha256"])
    records = [
        json.loads((run.OUT / "precision_replay" / f"example_{i:02d}.json").read_text())
        for i in run.CFG["stalled_indices"]
    ]
    write(
        run.OUT / "precision_replay_result.json",
        dict(
            examples=16,
            converged=sum(r["optimizer"]["converged"] for r in records),
            by_condition={
                str(c): dict(
                    examples=sum(r["record"]["condition"] == c for r in records),
                    converged=sum(r["optimizer"]["converged"] for r in records if r["record"]["condition"] == c),
                )
                for c in [50, 250, 450]
            },
            protected_hashes_checked=True,
            cuda_used=False,
            neural_training_launched=False,
        ),
    )
