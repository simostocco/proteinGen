#!/usr/bin/env python3
"""Versioned CPU-only fixed-state precision and correction-space KKT audit."""

import argparse
import hashlib
import json
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import FunctionType

import torch
import yaml

from protein_distance_diffusion.training import e010_cartesian_oracle_v5 as v5
from protein_distance_diffusion.training import e010_precision_kkt_v6 as audit
from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from protein_distance_diffusion.training.e010_recurrent_capacity import batch
from scripts.run_e010_phase4d_cartesian_oracle_v5 import OUT as V5
from scripts.run_e010_phase4d_cartesian_oracle_v5 import setup
from scripts.run_e010_phase4d_recurrent_capacity_v3 import ROOT, load_cache, write

OUT = V5.parent / "precision_kkt_v6"
CONFIG = ROOT / "configs/e010_phase4d_precision_kkt_v6.yaml"
TENSORS = Path("/tmp/e010_precision_kkt_v6_recovered")
CACHE = None
CFG = None


def contract():
    cfg = yaml.safe_load(CONFIG.read_text())
    historical = [json.loads((V5 / "examples" / f"example_{i:02d}.json").read_text()) for i in range(60)]
    stalls = [i for i, r in enumerate(historical) if not r["optimizer"]["converged"]]
    controls = sorted(
        i
        for c in (50, 250, 450)
        for i in [
            j
            for j, r in enumerate(historical)
            if r["record"]["condition"] == c
            and r["optimizer"]["converged"]
            and r["optimizer"]["projected_gradient_residual"] <= 0.0005
        ][:2]
    )
    assert stalls == cfg["stalled_indices"] and controls == cfg["control_indices"]
    assert cfg["fixed"] == dict(K=4, s_max=0.04, beta=16.8, gamma=2, examples_total=60, quartets_total=13029)
    setup()
    return cfg


def register():
    cfg = contract()
    _, manifest = load_cache()
    paths = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    pins = {str(ROOT / p): file_hash(ROOT / p) for p in paths}
    for p in (
        CONFIG,
        ROOT / "docs/e010_phase4d_precision_kkt_v6.md",
        Path(__file__).resolve(),
        ROOT / "src/protein_distance_diffusion/training/e010_precision_kkt_v6.py",
    ):
        pins[str(p)] = file_hash(p)
    write(
        OUT / "execution_contract.json",
        dict(
            config=cfg,
            protected_sha256=pins,
            cache_sha256=manifest["cache_sha256"],
            historical_commit="f846606e6784eda9517aba4d3c25d0c63266cb4c",
            examples=[dict(index=i, record=CACHE_record(i)) for i in cfg["stalled_indices"] + cfg["control_indices"]],
        ),
    )


def CACHE_record(i):
    return json.loads((V5 / "examples" / f"example_{i:02d}.json").read_text())["record"]


def tensor_digest(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


def load_example(i):
    return {k: v.double() if v.is_floating_point() else v for k, v in batch(CACHE, [i], "cpu").items()}


def recover(i):
    """Only executed with explicit recovery authorization; exact old solver."""
    b = load_example(i)
    old = json.loads((V5 / "examples" / f"example_{i:02d}.json").read_text())
    captured = []

    def observing_initialize(pg):
        z = v5.initialize(pg)
        captured.append(z)
        return z

    solver = FunctionType(v5.solve.__code__, {**v5.solve.__globals__, "initialize": observing_initialize})
    solver.__kwdefaults__ = v5.solve.__kwdefaults__
    tr, log = solver(b, examples_total=60, quartets_total=13029, settings=setup()["optimizer"])
    assert [tensor_digest(p) for p in tr["states"]] == old["state_coordinate_sha256"]
    for key in (
        "history",
        "converged",
        "iterations",
        "projected_gradient_residual",
        "closure_evaluations",
        "stable_iterations",
        "termination",
    ):
        assert log[key] == old["optimizer"][key], f"recovery mismatch {i}: {key}"
    for t, p in enumerate(tr["states"]):
        row = v5.metric_row(p, b, old["record"], step_delta=tr["steps"][t - 1]["delta"] if t else None)
        assert row == old["states"][t]
    payload = dict(
        z=captured[0].detach(),
        states=[p.detach() for p in tr["states"]],
        deltas=torch.stack([s["delta"].detach() for s in tr["steps"]]),
    )
    TENSORS.mkdir(parents=True, exist_ok=True)
    torch.save(payload, TENSORS / f"example_{i:02d}.pt")
    write(
        OUT / "recovery" / f"example_{i:02d}.json",
        dict(
            index=i,
            record=old["record"],
            exact_historical_state_metric_history_stopping_parity=True,
            recovered_tensor_sha256=file_hash(TENSORS / f"example_{i:02d}.pt"),
            variables_sha256=tensor_digest(payload["z"]),
            state_coordinate_sha256=old["state_coordinate_sha256"],
        ),
    )
    return i


def distribution(values):
    x = values.detach().flatten()
    if not x.numel():
        return dict(count=0)
    return dict(
        count=x.numel(),
        minimum=float(x.min()),
        median=float(x.median()),
        p90=float(torch.quantile(x, 0.9)),
        p99=float(torch.quantile(x, 0.99)),
        maximum=float(x.max()),
    )


def cosine(x, y):
    n = float(x.norm() * y.norm())
    return float((x * y).sum()) / n if n else None


def inspect(i):
    b = load_example(i)
    old = json.loads((V5 / "examples" / f"example_{i:02d}.json").read_text())
    proof = json.loads((OUT / "recovery" / f"example_{i:02d}.json").read_text())
    path = TENSORS / f"example_{i:02d}.pt"
    assert file_hash(path) == proof["recovered_tensor_sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert [tensor_digest(p) for p in saved["states"]] == old["state_coordinate_sha256"]
    v = (0.04 * saved["z"]).requires_grad_()
    tr = audit.physical_trajectory(b["pg"], b["mask"], v)
    assert [tensor_digest(p) for p in tr["states"]] == old["state_coordinate_sha256"]
    corrections = saved["deltas"].clone().requires_grad_()
    direct = audit.correction_trajectory(b["pg"], b["mask"], corrections)
    assert [tensor_digest(p) for p in direct["states"]] == old["state_coordinate_sha256"]
    eligible = direct["eligible"]
    results, gv, gd = {}, {}, {}
    for label, pure in [("historical", False), ("float64", True)]:
        terms = audit.objective(tr["prediction"], b, pure_float64=pure)
        gv[label] = torch.autograd.grad(terms["total"], v, retain_graph=True)[0]
        direct_terms = audit.objective(direct["prediction"], b, pure_float64=pure)
        gd[label] = torch.autograd.grad(direct_terms["total"], corrections, retain_graph=True)[0]
        results[label] = {k: float(t.detach()) for k, t in terms.items()}
    geom = audit.radial_geometry(v.detach())
    chain = torch.einsum("...ij,...j->...i", geom["jacobian"], gd["float64"])
    chain_error = float((chain - gv["float64"]).norm()) / max(float(gv["float64"].norm()), 1e-300)
    kkt = audit.kkt(corrections.detach(), gd["float64"], eligible, CFG["boundary_relative_tolerance"])
    scale = float(gd["float64"][eligible].norm(dim=-1).max())
    normalized_kkt = float(kkt["residual_norm"].max()) / max(scale, 1e-300)
    vscale = max(1.0, float(v.detach()[eligible].square().mean().sqrt()))

    def variable_fn(x):
        p = audit.physical_trajectory(b["pg"], b["mask"], x)["prediction"]
        return audit.objective(p, b, pure_float64=True)["total"]

    def correction_fn(x):
        p = audit.correction_trajectory(b["pg"], b["mask"], x)["prediction"]
        return audit.objective(p, b, pure_float64=True)["total"]

    fd_v = audit.directional_checks(
        variable_fn,
        v.detach(),
        gv["float64"],
        [vscale * x for x in CFG["finite_difference"]["variable_relative_epsilons"]],
    )
    fd_delta = audit.directional_checks(
        correction_fn, corrections.detach(), gd["float64"], CFG["finite_difference"]["correction_epsilons_angstrom"]
    )
    failures = [
        (space, j)
        for space, rows in [("v", fd_v), ("delta", fd_delta)]
        for j in range(3)
        if not any(r["passed"] for r in rows if r["direction"] == j)
    ]
    shadows = []
    for tangent in (False, True):
        d = audit.shadow_direction(corrections.detach(), gd["float64"], eligible, tangential=tangent)
        for step in CFG["shadow_steps_angstrom"]:
            trial = audit.project_ball(corrections.detach() + step * d)
            atr = audit.correction_trajectory(b["pg"], b["mask"], trial)
            same = bool(torch.equal(atr["eligible"], eligible))
            val = float(audit.objective(atr["prediction"], b, pure_float64=True)["total"])
            decrease = results["float64"]["total"] - val
            threshold = max(
                CFG["shadow_material_absolute_objective_decrease"],
                CFG["shadow_material_relative_objective_decrease"] * abs(results["float64"]["total"]),
            )
            shadows.append(
                dict(
                    direction="tangent" if tangent else "projected_negative_gradient",
                    step_angstrom=step,
                    objective=val,
                    objective_decrease=decrease,
                    eligibility_unchanged=same,
                    material_decrease=same and decrease > threshold,
                    maximum_step_correction=float(trial.norm(dim=-1).max()),
                    cumulative_max=float((atr["prediction"] - b["pg"]).norm(dim=-1).max()),
                )
            )
    columns = {k: x.detach()[b["mask"].expand(4, -1, -1)].tolist() for k, x in geom.items() if k != "jacobian"}
    valid_all = b["mask"].expand(4, -1, -1)
    columns.update(
        eligible=eligible[valid_all].tolist(),
        active=kkt["active"][valid_all].tolist(),
        actual_delta_norm=corrections.detach().norm(dim=-1)[valid_all].tolist(),
        g_v_norm=gv["float64"].norm(dim=-1)[valid_all].tolist(),
        g_delta_norm=gd["float64"].norm(dim=-1)[valid_all].tolist(),
        radial_gradient=kkt["radial_gradient"][valid_all].tolist(),
        tangent_gradient_norm=kkt["tangent_residual"][valid_all].tolist(),
        feasible_radial_residual=kkt["feasible_radial_residual"][valid_all].tolist(),
        kkt_residual=kkt["residual_norm"][valid_all].tolist(),
    )
    objective_diff = results["float64"]["total"] - results["historical"]["total"]
    gradient_rel = float((gv["float64"] - gv["historical"]).norm()) / max(float(gv["float64"].norm()), 1e-300)
    metrics = v5.metric_row(saved["states"][-1], b, old["record"])
    precision_metrics = dict(metrics, raw_cartesian=results["float64"]["cartesian"] * 60)
    record = dict(
        index=i,
        record=old["record"],
        stalled=i in CFG["stalled_indices"],
        terms=results,
        objective_difference=objective_diff,
        objective_relative_difference=abs(objective_diff) / abs(results["float64"]["total"]),
        objective_precision_material=abs(objective_diff)
        > max(
            CFG["precision_objective_absolute_materiality"],
            CFG["precision_objective_relative_materiality"] * abs(results["float64"]["total"]),
        ),
        gradient_norms={k: float(x.norm()) for k, x in gv.items()},
        gradient_absolute_difference=float((gv["float64"] - gv["historical"]).norm()),
        gradient_relative_difference=gradient_rel,
        gradient_cosine=cosine(gv["float64"], gv["historical"]),
        correction_gradient_norm=float(gd["float64"].norm()),
        chain_rule_relative_error=chain_error,
        radial_summary={k: distribution(x[eligible]) for k, x in geom.items() if k != "jacobian"},
        radial_by_step=[
            {k: distribution(x[t][eligible[t]]) for k, x in geom.items() if k != "jacobian"} for t in range(4)
        ],
        boundary_active_fraction=float(kkt["active"][eligible].double().mean()),
        kkt_norm=distribution(kkt["residual_norm"][eligible]),
        kkt_normalized_max=normalized_kkt,
        tangent_residual=distribution(kkt["tangent_residual"][eligible]),
        feasible_radial_residual=distribution(kkt["feasible_radial_residual"][eligible]),
        finite_differences=dict(v=fd_v, delta=fd_delta, failed_directions=failures),
        shadows=shadows,
        historical_metrics=metrics,
        float64_metrics=precision_metrics,
        metric_differences={k: precision_metrics[k] - metrics[k] for k in metrics if isinstance(metrics[k], float)},
        variable_columns=columns,
        variable_column_order="step-major then valid residue index ascending",
        eligible_per_step=[int(x.sum()) for x in eligible],
        historical_states_exact=True,
    )
    write(OUT / "examples" / f"example_{i:02d}.json", record)
    return i, normalized_kkt, len(failures), any(s["material_decrease"] for s in shadows)


def execute(mode, authorized):
    global CACHE, CFG
    CFG = contract()
    execution = json.loads((OUT / "execution_contract.json").read_text())
    assert_file_pins(execution["protected_sha256"])
    CACHE, manifest = load_cache()
    if mode == "recover" and not authorized:
        raise RuntimeError("Historical replay recovery requires explicit user authorization")
    ids = CFG["stalled_indices"] + CFG["control_indices"]
    fn = recover if mode == "recover" else inspect
    folder = "recovery" if mode == "recover" else "examples"
    pending = [i for i in ids if not (OUT / folder / f"example_{i:02d}.json").exists()]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
        for future in as_completed([pool.submit(fn, i) for i in pending]):
            print(mode, future.result(), flush=True)
    assert_file_pins(execution["protected_sha256"])
    assert_file_pins(manifest["protected_input_sha256"])
    if mode == "inspect":
        write(
            OUT / "sections_1_to_7_complete.json",
            dict(
                examples=len(ids),
                historical_state_recovery_verified=True,
                protected_hashes_verified=True,
                no_optimizer_in_fixed_state_audit=True,
                cuda_used=False,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["register", "recover", "inspect"])
    parser.add_argument("--recovery-authorized", action="store_true")
    args = parser.parse_args()
    register() if args.mode == "register" else execute(args.mode, args.recovery_authorized)
