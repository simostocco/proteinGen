#!/usr/bin/env python3
"""Frozen CPU objective-v2 calibration. No optimizer or training."""

import copy
import json
from pathlib import Path

import torch
import yaml

from protein_distance_diffusion.models.e010_hybrid_local import SOURCE_SHA256, load_frozen_hybrid
from protein_distance_diffusion.training.e010_phase4d import example_metrics, hybrid_losses
from protein_distance_diffusion.training.e010_phase4d_diagnostic import (
    aggregate_rows,
    assert_file_pins,
    changes,
    displaced_copy,
    dot,
    file_hash,
    module_norms,
    signed_status,
    state_hash,
)
from protein_distance_diffusion.training.e010_phase4d_objective_v2 import (
    BETA,
    chiral_sum,
    combined_gradient,
    common_descent_interval,
    direction_rms_slope,
    freeze_chirality,
    gradient_audit,
    select_gamma,
)
from scripts.diagnose_e010_phase4d_frozen_local import PREP, ROOT, load_panel

OUT = PREP / "objective_v2_diagnostic"


def write_once(name, value):
    with (OUT / name).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def evaluate(branch, panel):
    params = tuple(branch.parameters())
    gs = {k: [torch.zeros_like(p) for p in params] for k in ("local", "cartesian", "chiral")}
    total_eligible = sum(int(r["frozen"]["eligible"].sum()) for r in panel)
    if not total_eligible:
        raise ValueError("no existing eligible chirality quartets")
    objective = {k: 0.0 for k in gs}
    rows = []
    for item in panel:
        pg, x, y, m = (item[k] for k in ("pg", "source", "target", "mask"))
        out = branch(pg, m)
        losses = hybrid_losses(out["prediction"], pg, x, y, m)
        losses = {
            "local": losses["local_mean"],
            "cartesian": losses["cartesian"],
            "chiral": chiral_sum(out["prediction"], m, item["frozen"]),
        }
        for name, loss in losses.items():
            denom = total_eligible if name == "chiral" else len(panel)
            objective[name] += float(loss.detach()) / denom
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            for a, g in zip(gs[name], grads, strict=True):
                if g is not None:
                    a.add_(g / denom)
        with torch.no_grad():
            raw = hybrid_losses(out["prediction"], pg, x, y, m)
            row = example_metrics(out["prediction"], pg, x, y, m)[0]
            row.update({k: item[k] for k in ("sample_id", "condition", "stratum")})
            row.update(
                local_mse={str(k): float(raw[f"local_{k}"]) for k in (1, 2, 3)},
                displacement_mean_square=float(raw["displacement"]),
            )
            assess, inv = signed_status(out["prediction"], y, m)
            ba, bi = signed_status(pg, y, m)
            common = assess & ba
            count = int(item["frozen"]["eligible"].sum())
            row.update(
                chirality_common_assessable=int(common.sum()),
                chirality_common_inversions=int((inv & common).sum()),
                chirality_common_baseline_inversions=int((bi & common).sum()),
                chirality_assessability_lost=int((ba & ~assess).sum()),
                chirality_assessability_gained=int((assess & ~ba).sum()),
                input_frame_eligible=int(out["eligible"].sum()),
                input_frame_degenerate=int(out["degenerate"].sum()),
                chiral_error_sum=float(losses["chiral"]),
                chiral_loss_eligible=count,
                chiral_loss=float(losses["chiral"]) / count if count else None,
            )
            rows.append(row)
    if not all(torch.isfinite(g).all() for v in gs.values() for g in v):
        raise ValueError("nonfinite component gradient")
    summary = aggregate_rows(rows)

    def add_chiral(s, group):
        count = sum(r["chiral_loss_eligible"] for r in group)
        s["chiral_loss_eligible"] = count
        s["continuous_chiral_loss"] = sum(r["chiral_error_sum"] for r in group) / count if count else None

    add_chiral(summary["overall"], rows)
    for c, s in summary["by_condition"].items():
        add_chiral(s, [r for r in rows if str(r["condition"]) == c])
    for st, s in summary["by_stratum"].items():
        add_chiral(s, [r for r in rows if r["stratum"] == st])
    return {"objective": objective, "metrics": summary, "per_example": rows}, gs


def main():
    if (OUT / "results.json").exists() or (OUT / "gradient_audit.json").exists():
        raise FileExistsError("refusing to replace objective-v2 record")
    protocol = json.loads((OUT / "protocol.json").read_text())
    if protocol["beta"] != BETA or protocol["correction_rms_targets_angstrom"] != [
        0.0,
        0.03,
        0.05,
        0.1,
        0.15,
        0.2,
        0.27,
    ]:
        raise ValueError("unregistered coefficients or correction scales")
    torch.set_num_threads(2)
    torch.manual_seed(protocol["seed"])
    torch.use_deterministic_algorithms(True)
    pins = {p: file_hash(p) for p in PREP.rglob("*") if p.is_file() and OUT not in p.parents}
    for directory in ("phase4b_real_denoiser_v1", "phase4c_local_auxiliary_v1"):
        pins.update({p: file_hash(p) for p in (PREP.parent / directory).rglob("*") if p.is_file()})
    pins[OUT / "protocol.json"] = file_hash(OUT / "protocol.json")
    config = ROOT / "configs/e010_phase4d_hybrid_local_global_v1.yaml"
    pins[config] = file_hash(config)
    cp = Path(yaml.safe_load(config.read_text())["global"]["checkpoint"])
    pins[cp] = SOURCE_SHA256
    assert_file_pins(pins)
    model = load_frozen_hybrid(cp)
    global_pin = state_hash(model.global_model)
    local_pin = state_hash(model.local)
    prep = json.loads((PREP / "preparation.json").read_text())
    if global_pin != prep["model_state_sha256"]:
        raise ValueError("global tensor state pin")
    panel, archives = load_panel(model, json.loads((PREP / "tiny_panel.json").read_text()))
    pins.update(archives)
    for item in panel:
        item["frozen"] = freeze_chirality(item["pg"], item["target"], item["mask"])
    initial, gs = evaluate(model.local, panel)
    audit = gradient_audit(gs)
    interval = common_descent_interval(gs)
    gamma = select_gamma(interval, gs)
    audit.update(
        beta=BETA,
        feasible_gamma_interval=interval,
        selected_gamma=gamma,
        per_module={name: module_norms(model.local, g) for name, g in gs.items()},
    )
    groups = list(audit["per_module"]["local"])
    for group in groups:
        indices = [
            i
            for i, (name, _) in enumerate(model.local.named_parameters())
            if (".".join(name.split(".")[:2]) if name.startswith("blocks.") else name.split(".")[0]) == group
        ]
        audit.setdefault("per_module_dot_products", {})[group] = gradient_audit(
            {k: [v[i] for i in indices] for k, v in gs.items()}
        )
    points = []
    if gamma is None:
        classification = "NO_COMMON_FIRST_ORDER_DESCENT_GAMMA; finite sweep stopped"
        print(classification, flush=True)
    else:
        total = combined_gradient(gs["local"], gs["cartesian"], gs["chiral"], gamma)
        direction = [-g for g in total]
        audit["predicted_derivatives"] = {k: float(dot(g, direction)) for k, g in gs.items()}
        if not all(v < 0 for v in audit["predicted_derivatives"].values()):
            raise AssertionError("common-descent contract")
        slope = direction_rms_slope(model.local, direction, panel)
        if slope <= 0:
            raise ValueError("no nonzero correction direction")
        audit["correction_rms_per_alpha_angstrom"] = slope
        print("Initial gradient audit", json.dumps(audit), flush=True)
        classification = "COMMON_FIRST_ORDER_DESCENT_GAMMA_EXISTS"
        for rms in protocol["correction_rms_targets_angstrom"]:
            alpha = rms / slope
            candidate = displaced_copy(model.local, direction, alpha)
            point, g = evaluate(candidate, panel) if rms else (initial, gs)
            point = copy.deepcopy(point)
            s = point["metrics"]["overall"]
            b = initial["metrics"]["overall"]
            checks = {
                "mean_local_improves": s["mean_local_rmse"] < b["mean_local_rmse"],
                "all_offsets_non_harmful": all(s["local_rmse"][k] <= b["local_rmse"][k] for k in ("1", "2", "3")),
                "raw_cartesian_non_harmful_with_tiny_tolerance": s["raw_cartesian"]
                <= (1 + protocol["raw_cartesian_tolerance_fraction"]) * b["raw_cartesian"],
                "aligned_degradation_le_1pct": s["aligned_rmsd"] <= 1.01 * b["aligned_rmsd"],
                "inversion_count_non_increasing": s["chirality_inversions"] <= b["chirality_inversions"],
                "assessability_preserved": s["chirality_assessable"] == b["chirality_assessable"]
                and s["chirality_assessability_lost"] == 0,
                "all_finite": s["all_finite"],
            }
            point.update(
                target_correction_rms=rms,
                alpha=alpha,
                changes=changes(s, b),
                primary_criteria={**checks, "simultaneous_pass": all(checks.values())},
                continuous_chiral_loss_change=s["continuous_chiral_loss"] - b["continuous_chiral_loss"],
                gradients=gradient_audit(g),
                condition_changes={
                    k: changes(v, initial["metrics"]["by_condition"][k])
                    for k, v in point["metrics"]["by_condition"].items()
                },
                stratum_changes={
                    k: changes(v, initial["metrics"]["by_stratum"][k])
                    for k, v in point["metrics"]["by_stratum"].items()
                },
            )
            points.append(point)
            if state_hash(model.local) != local_pin:
                raise AssertionError("original local branch mutated")
            print(
                f"RMS={rms} A, alpha={alpha:.8g}: local={s['mean_local_rmse']:.6f}, "
                f"chirality={s['chirality_inversions']}, pass={all(checks.values())}",
                flush=True,
            )
    audit["local_cartesian_without_chirality_derivatives"] = {k: -interval["constraints"][k]["intercept"] for k in gs}
    assert_file_pins(pins)
    isolated = all(not p.requires_grad and p.grad is None for p in model.global_model.parameters())
    if not isolated or state_hash(model.global_model) != global_pin:
        raise AssertionError("global freeze violated")
    write_once("gradient_audit.json", audit)
    write_once(
        "results.json",
        {
            "classification": classification,
            "audit": audit,
            "initial_metrics": initial,
            "finite_points": points,
            "global_gradient_isolation": isolated,
            "global_unchanged": True,
            "local_base_unchanged": True,
            "protected_inputs_sha256": {str(p): h for p, h in pins.items()},
            "protected_inputs_unchanged": True,
            "protocol_sha256": file_hash(OUT / "protocol.json"),
            "diagnostic_v1_commit": protocol["diagnostic_v1_commit"],
            "cuda_used": False,
            "training_launched": False,
            "optimizer_steps": 0,
        },
    )


if __name__ == "__main__":
    main()
