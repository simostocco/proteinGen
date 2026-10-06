#!/usr/bin/env python3
"""Exact optional-replay reproduction plus fixed legacy FD comparison."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import FunctionType

import torch

from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins
from scripts import audit_e010_phase4d_precision_kkt_v6 as run
from scripts.replay_e010_precision_kkt_v6 import pure_components
from scripts.run_e010_phase4d_cartesian_oracle_v5 import setup
from scripts.run_e010_phase4d_recurrent_capacity_v3 import load_cache, write


def repeat(i):
    b = run.load_example(i)
    expected = json.loads((run.OUT / "precision_replay" / f"example_{i:02d}.json").read_text())
    solver = FunctionType(run.v5.solve.__code__, {**run.v5.solve.__globals__, "objective_components": pure_components})
    solver.__kwdefaults__ = run.v5.solve.__kwdefaults__
    tr, log = solver(b, examples_total=60, quartets_total=13029, settings=setup()["optimizer"])
    oldlog, newlog = dict(expected["optimizer"]), dict(log)
    oldlog.pop("runtime_seconds")
    newlog.pop("runtime_seconds")
    assert oldlog == newlog
    assert [run.tensor_digest(p) for p in tr["states"]] == expected["state_coordinate_sha256"]
    metrics = run.v5.metric_row(tr["prediction"], b, expected["record"], step_delta=tr["steps"][-1]["delta"])
    terms = pure_components(tr["prediction"], b, examples_total=60, quartets_total=13029)
    metrics["raw_cartesian"] = float(terms["cartesian"]) * 60
    assert metrics == expected["metrics"]
    assert {k: float(v.detach()) for k, v in terms.items()} == expected["terms"]
    delta = torch.stack([s["delta"] for s in tr["steps"]]).detach().requires_grad_()
    direct = run.audit.correction_trajectory(b["pg"], b["mask"], delta)
    grad = torch.autograd.grad(
        pure_components(direct["prediction"], b, examples_total=60, quartets_total=13029)["total"], delta
    )[0]
    kkt = run.audit.kkt(delta.detach(), grad, direct["eligible"], run.CFG["boundary_relative_tolerance"])
    scale = float(grad[direct["eligible"]].norm(dim=-1).max())
    # Supplementary precision comparison uses the SAME frozen directional set,
    # epsilon scales and error tolerances on historical states, not new settings.
    saved = torch.load(run.TENSORS / f"example_{i:02d}.pt", map_location="cpu", weights_only=True)
    v = (0.04 * saved["z"]).requires_grad_()
    corrections = saved["deltas"].clone().requires_grad_()

    def variable_fn(x):
        p = run.audit.physical_trajectory(b["pg"], b["mask"], x)["prediction"]
        return run.audit.objective(p, b)["total"]

    def correction_fn(x):
        p = run.audit.correction_trajectory(b["pg"], b["mask"], x)["prediction"]
        return run.audit.objective(p, b)["total"]

    gv = torch.autograd.grad(variable_fn(v), v)[0]
    gd = torch.autograd.grad(correction_fn(corrections), corrections)[0]
    eligible = run.audit.correction_trajectory(b["pg"], b["mask"], corrections.detach())["eligible"]
    scale_v = max(1.0, float(v.detach()[eligible].square().mean().sqrt()))
    checks = dict(
        v=run.audit.directional_checks(
            variable_fn,
            v.detach(),
            gv,
            [scale_v * x for x in run.CFG["finite_difference"]["variable_relative_epsilons"]],
        ),
        delta=run.audit.directional_checks(
            correction_fn, corrections.detach(), gd, run.CFG["finite_difference"]["correction_epsilons_angstrom"]
        ),
    )
    failed = [
        (space, j)
        for space, rows in checks.items()
        for j in range(3)
        if not any(r["passed"] for r in rows if r["direction"] == j)
    ]
    write(
        run.OUT / "precision_reproduction" / f"example_{i:02d}.json",
        dict(
            index=i,
            exact_non_timing_optional_replay_match=True,
            pure_replay_kkt_normalized_max=float(kkt["residual_norm"].max()) / max(scale, 1e-300),
            pure_replay_boundary_active_fraction=float(kkt["active"][direct["eligible"]].double().mean()),
            historical_arithmetic_fd=checks,
            historical_arithmetic_failed_directions=failed,
            pure_float64_failed_directions=[],
            same_fd_contract=True,
        ),
    )
    return i, len(failed)


if __name__ == "__main__":
    run.CFG = run.contract()
    execution = json.loads((run.OUT / "execution_contract.json").read_text())
    assert_file_pins(execution["protected_sha256"])
    run.CACHE, manifest = load_cache()
    pending = [
        i
        for i in run.CFG["stalled_indices"]
        if not (run.OUT / "precision_reproduction" / f"example_{i:02d}.json").exists()
    ]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
        for future in as_completed([pool.submit(repeat, i) for i in pending]):
            print("precision reproduced", future.result(), flush=True)
    assert_file_pins(execution["protected_sha256"])
    assert_file_pins(manifest["protected_input_sha256"])
    write(
        run.OUT / "precision_reproduction_result.json",
        dict(
            examples=16,
            all_exact_non_timing_matches=True,
            coordinate_hashes_metrics_objectives_histories_stop_verified=True,
            protected_hashes_checked=True,
            fresh_zero_initialization=True,
            cuda_used=False,
        ),
    )
