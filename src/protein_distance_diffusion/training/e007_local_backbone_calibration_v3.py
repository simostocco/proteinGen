"""Plan-only, v2-derived E007 Phase-3I.2 calibration v3 contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TERMS = ("adjacent", "i_plus_2", "i_plus_3", "bond_angle_cosine", "discontinuity", "clash")
TIMESTEPS = (25, 250, 425, 499)
LENGTHS = (64, 128, 256, 384, 500)
V2_REPORT_SHA256 = "8e95fc94bad1633ca2e5af7880f6a1b1aa7b49af670ae10795f5e50965078b70"
V2_PROTOCOL_SHA256 = "d27acf93b9378435113b82d95139a017b5222ddaacf90ffe517f8bcbfa3e164a"


def finite_counts(value: Any) -> dict[str, int | bool | float]:
    """Count exact finite elements; float64 coverage is descriptive only."""
    import torch

    finite = torch.isfinite(value)
    total = int(value.numel())
    count = int(finite.sum().item())
    nonfinite = total - count
    return {
        "finite_count": count,
        "total_count": total,
        "non_finite_count": nonfinite,
        "all_finite": bool(torch.isfinite(value).all().item()),
        "coverage_float64": float(count / total) if total else 1.0,
    }


def cosine_matrix(vectors: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Return symmetric pairwise cosine values for one cell's auxiliary gradients."""
    import torch

    names = list(vectors)
    result: dict[str, dict[str, float]] = {name: {} for name in names}
    for left in names:
        for right in names:
            a, b = vectors[left], vectors[right]
            den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
            value = torch.dot(a, b) / den.clamp_min(torch.finfo(a.dtype).eps)
            result[left][right] = float(value.detach())
    return result


def evaluate_candidate_gates(
    cells: list[dict[str, Any]], config: dict[str, Any], candidates: list[str]
) -> dict[str, Any]:
    """Evaluate declared upper and lower gates and preserve cell-specific reasons."""
    upper, lower = config["upper_gates"], config["lower_activity_gates"]
    result: dict[str, Any] = {}
    for name in candidates:
        failures = []
        for cell in cells:
            ident, item = cell["identity"], cell["candidate_coefficients"][name]
            if cell.get("non_finite_count", 0) or not cell.get("all_finite", False):
                failures.append({"identity": ident, "gate": "exact_finite"})
            if any(x > upper["individual_weighted_term_to_v_max"] for x in item["individual_ratios"].values()):
                failures.append({"identity": ident, "gate": "individual_term_upper"})
            if item["combined_auxiliary_to_v_ratio"] > upper["combined_auxiliary_to_v_max"]:
                failures.append({"identity": ident, "gate": "combined_auxiliary_upper"})
            if not upper["total_to_v_min"] <= item["total_to_v_ratio"] <= upper["total_to_v_max"]:
                failures.append({"identity": ident, "gate": "total_to_v_range"})
        for timestep in lower["applies_timesteps"]:
            subset = [c["candidate_coefficients"][name] for c in cells if c["identity"]["timestep"] == timestep]
            if not subset:
                failures.append({"timestep": timestep, "gate": "missing_activity_cells"})
                continue
            for term in lower["core_terms"]:
                values = [c["individual_ratios"][term] for c in subset]
                if (
                    max(float(np.median(values)), float(np.percentile(values, 90)))
                    < lower["core_median_or_p90_minimum"]
                ):
                    failures.append({"timestep": timestep, "term": term, "gate": "core_activity"})
            for term in lower["tail_terms"]:
                values = [c["individual_ratios"][term] for c in subset]
                if (
                    max(float(np.median(values)), float(np.percentile(values, 90)))
                    < lower["tail_median_or_p90_minimum"]
                ):
                    failures.append({"timestep": timestep, "term": term, "gate": "tail_activity"})
            combined = [c["combined_auxiliary_to_v_ratio"] for c in subset]
            if float(np.median(combined)) < lower["combined_auxiliary_to_v_median_minimum"]:
                failures.append({"timestep": timestep, "gate": "combined_activity"})
        result[name] = {"passes": not failures, "failures": failures}
    return result


def select_strongest_passing(gates: dict[str, Any], ordering: list[str]) -> dict[str, Any]:
    passing = [name for name in ordering if gates.get(name, {}).get("passes")]
    if len(set(ordering)) != len(ordering) or not ordering:
        return {"selected": None, "status": "fail_closed_ambiguous_ordering"}
    return {
        "selected": passing[-1] if passing else None,
        "status": "selected_strongest_passing" if passing else "fail_closed_no_passing_candidate",
        "passing_in_order": passing,
    }


def _report(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if hashlib.sha256(p.read_bytes()).hexdigest() != V2_REPORT_SHA256:
        raise ValueError("protected v2 report hash mismatch")
    return json.loads(p.read_text())


def correction_record(report_path: str | Path) -> dict[str, Any]:
    report = _report(report_path)
    if len(report["cells"]) != 80:
        raise ValueError("protected v2 cell count mismatch")
    coverage = [cell["gradient_profile"] for cell in report["cells"]]
    gate_results = {}
    for name in ("very_conservative", "schedule_tapered", "schedule_tapered_half"):
        individual_failures = []
        total_failures = []
        finite_failures = []
        for cell in report["cells"]:
            item = cell["candidate_coefficients"][name]
            if not cell.get("raw_and_gradient_finite", False):
                finite_failures.append({"identity": cell["identity"], "gate": "raw_and_gradient_finite"})
            for term, record in item["terms"].items():
                if record["weighted_gradient_ratio"] > 0.20:
                    individual_failures.append(
                        {"identity": cell["identity"], "term": term, "ratio": record["weighted_gradient_ratio"]}
                    )
                if not record["raw_gradient_finite"] or not record["weighted_gradient_finite"]:
                    finite_failures.append({"identity": cell["identity"], "term": term})
            if not 0.80 <= item["total_to_v_gradient_ratio"] <= 1.30:
                total_failures.append({"identity": cell["identity"], "ratio": item["total_to_v_gradient_ratio"]})
        gate_results[name] = {
            "passes_upper_stability_gates": not individual_failures and not total_failures and not finite_failures,
            "failed_individual_ratio_gates": len(
                {tuple(sorted(failure["identity"].items())) for failure in individual_failures}
            ),
            "individual_ratio_exceedance_count": len(individual_failures),
            "individual_ratio_failures": individual_failures,
            "total_ratio_failures": total_failures,
            "nonfinite_gradient_failures": finite_failures,
        }
    return {
        "record_version": "e007_phase3i2_v2_gate_correction_v1",
        "source_v2_report_sha256": V2_REPORT_SHA256,
        "source_v2_protocol_sha256": V2_PROTOCOL_SHA256,
        "cell_count": len(coverage),
        "reported_coverage_values": sorted({cell["v_finite_gradient_coverage"] for cell in coverage}),
        "terms_per_cell": len(coverage[0]["terms"]),
        "term_coverage_values": sorted(
            {term["finite_gradient_coverage"] for cell in coverage for term in cell["terms"].values()}
        ),
        "element_counts": {
            "available_in_protected_v2_report": False,
            "reason": (
                "immutable v2 retained float coverage and exact all-finite flags, but omitted gradient-vector lengths"
            ),
            "count_method_for_future_calibrations": [
                "finite_count = integer sum(isfinite(gradient_vector))",
                "total_count = integer gradient_vector.numel()",
                "non_finite_count = total_count - finite_count",
                "all_finite = boolean isfinite(gradient_vector).all()",
            ],
        },
        "evidence": (
            "all 80 raw_and_gradient_finite flags and all per-candidate raw/weighted term finite flags are true; "
            "the 560 coverage values are 80 v-gradient plus 480 local-term profile values"
        ),
        "fault": "v2 reduced isfinite booleans to float32 mean and compared mean == 1.0",
        "correct_rule": "non_finite_count == 0 and finite_count == total_count (or exact boolean all-finite)",
        "v2_counterfactual_gate": (
            "pass: all cell raw_and_gradient_finite flags and all term raw/weighted gradient finite flags are true"
        ),
        "candidate_gate_correction": gate_results,
        "selection": "no unique selection: both tapered candidates pass; pilot remains unauthorized",
        "tapered_combined_auxiliary_to_v": {
            "schedule_tapered": {"maximum": 0.0514492, "p95": 0.0312324, "median": 0.00207180, "mean": 0.00770567},
            "schedule_tapered_half": {"maximum": 0.0257246, "p95": 0.0156162, "median": 0.00103590, "mean": 0.00385283},
        },
        "timestep_25_schedule_tapered_max_individual_term_ratio": 0.00112,
        "stability_interpretation": (
            "stable but potentially ineffective; a larger coefficient alone is not a preference criterion"
        ),
        "pilot_authorized": False,
    }


def derive_coefficients(
    report_path: str | Path, budgets: dict[str, dict[str, float]], caps: dict[str, float]
) -> dict[str, Any]:
    report = _report(report_path)
    result: dict[str, Any] = {}
    for candidate, term_budgets in budgets.items():
        result[candidate] = {}
        for timestep in TIMESTEPS:
            result[candidate][str(timestep)] = {}
            for term in TERMS:
                ratios = [
                    cell["gradient_profile"]["terms"][term]["gradient_norm_to_v"]
                    for cell in report["cells"]
                    if cell["timestep"] == timestep
                ]
                # Use the observed p95 as denominator with maximum-derived hard cap.
                p95 = float(np.percentile(np.asarray(ratios, dtype=np.float64), 95))
                maximum = max(ratios)
                coefficient = min(term_budgets[term] / max(p95, 1e-15), caps[term] / max(maximum, 1e-15))
                result[candidate][str(timestep)][term] = coefficient
    return result


def validate_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if config.get("version") != "e007_local_backbone_repair_calibration_v3":
        raise ValueError("configuration version mismatch")
    if tuple(config["lengths"]) != LENGTHS or tuple(config["timesteps"]) != TIMESTEPS:
        raise ValueError("panel contract mismatch")
    ordering = config["selection_rule"]["ordering"]
    if ordering != ["budget_low", "budget_medium"] or len(set(ordering)) != len(ordering):
        raise ValueError("candidate ordering is ambiguous or differs from the declared budget tiers")
    if any(set(config["budgets"][candidate]) != set(TERMS) for candidate in ordering):
        raise ValueError("each budget tier must declare every local term exactly once")
    if any(config["budgets"]["budget_medium"][term] <= config["budgets"]["budget_low"][term] for term in TERMS):
        raise ValueError("budget_medium must strictly dominate budget_low term by term")
    if set(config["hard_maximum_caps"]) != set(TERMS):
        raise ValueError("hard maximum caps must declare every local term exactly once")
    for tier, terms in config["budgets"].items():
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in terms.values()):
            raise ValueError(f"{tier} target budgets must be finite and positive")
    if any(
        not math.isfinite(float(value)) or float(value) <= 0 or float(value) > 0.20
        for value in config["hard_maximum_caps"].values()
    ):
        raise ValueError("hard maximum caps must be finite, positive, and within the individual upper gate")
    if (
        config["protected_v2"]["report_sha256"] != V2_REPORT_SHA256
        or config["protected_v2"]["protocol_sha256"] != V2_PROTOCOL_SHA256
    ):
        raise ValueError("protected v2 hashes mismatch")
    protocol_path = Path(config["protected_v2"]["protocol_path"])
    if not protocol_path.is_file() or hashlib.sha256(protocol_path.read_bytes()).hexdigest() != V2_PROTOCOL_SHA256:
        raise ValueError("protected v2 protocol file is missing or changed")
    return config


def plan_only(config_path: str | Path, report_path: str | Path) -> dict[str, Any]:
    config = validate_config(config_path)
    coefficients = derive_coefficients(report_path, config["budgets"], config["hard_maximum_caps"])
    return {
        "mode": "plan_only",
        "version": config["version"],
        "configuration_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "source_v2_report_sha256": V2_REPORT_SHA256,
        "source_v2_protocol_sha256": V2_PROTOCOL_SHA256,
        "lengths": list(LENGTHS),
        "timesteps": list(TIMESTEPS),
        "structures_per_length": 4,
        "cell_count": 80,
        "estimated_model_forwards": 80,
        "derived_coefficients": coefficients,
        "target_budgets": config["budgets"],
        "upper_gates": config["upper_gates"],
        "lower_activity_gates": config["lower_activity_gates"],
        "selection_rule": config["selection_rule"],
        "pilot_authorized": False,
        "staging_created": False,
        "model_created": False,
        "checkpoint_loaded": False,
        "cuda_initialized": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "forward_pass": False,
        "backward_pass": False,
        "update_state_created": False,
        "output_created": False,
    }


def validate_panel(config_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read-only v3 panel validation using the established canonical v2 panel selector."""
    validate_config(config_path)
    v2_config_path = Path(config_path).with_name("e007_local_backbone_repair_calibration_v2.yaml")
    from protein_distance_diffusion.training.e007_local_backbone_calibration_v2 import (
        validate_panel as _validate_v2_panel,
    )

    report, panel = _validate_v2_panel(v2_config_path)
    report["validated_for_v3"] = True
    return report, panel


def run_calibration(config_path: str | Path) -> dict[str, Any]:
    """Execute the bounded 80-cell v3 panel; this entry point is never called by plan-only."""
    config = validate_config(config_path)
    # Complete real panel and protected-input validation before importing/initializing model or CUDA paths.
    panel_report, panel = validate_panel(config_path)
    protected_report = Path(config["protected_v2"]["report_path"])
    coefficients = derive_coefficients(protected_report, config["budgets"], config["hard_maximum_caps"])
    output = Path(config["calibration_output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"v3 calibration output exists: {output} or {staging}")
    import torch

    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization
    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
    from protein_distance_diffusion.training.e007_coordinate_real_pilot import uniform_coordinate_v_mse
    from protein_distance_diffusion.training.e007_local_backbone_repair import (
        _atomic_json,
        _fsync_directory,
        gradient_profile,
        local_backbone_losses,
        weighted_local_objective,
    )

    torch.backends.cuda.matmul.allow_tf32 = bool(config["numerics"]["allow_matmul_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["numerics"]["allow_cudnn_tf32"])
    torch.use_deterministic_algorithms(bool(config["numerics"]["deterministic_algorithms"]))

    source = localization.load_config(config["dataset_source_config"])
    prepared = [localization._prepared_reference(source, row["canonical_row"]) for row in panel]
    if config["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v3 calibration requires configured CUDA")
    device = torch.device("cuda")
    model = localization._load_model(source, device).requires_grad_(True).train()
    diffusion = CoordinateVPDiffusion(int(config["diffusion_steps"]))
    staging.mkdir(parents=True)
    cells: list[dict[str, Any]] = []
    try:
        for record in prepared:
            length = int(record["lengths"][0])
            for timestep in TIMESTEPS:
                sample_id = record["sample_ids"][0]
                identity_seed = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16)
                generator = torch.Generator(device=device).manual_seed(
                    3915000 + length * 1000 + timestep + identity_seed
                )
                clean = record["coordinates"].to(device)
                mask = record["residue_mask"].to(device)
                continuity = record["chain_continuity_mask"].to(device)
                timestep_tensor = torch.tensor([timestep], dtype=torch.long, device=device)
                batch = diffusion.make_training_batch(clean, mask, timesteps=timestep_tensor, generator=generator)
                prediction = model(
                    batch.noisy_coordinates, timestep_tensor, record["lengths"].to(device), mask, continuity
                )["v_prediction"]
                v_loss = uniform_coordinate_v_mse(prediction, batch.coordinate_v_target, mask)
                x0 = diffusion.reconstruct_x0(batch.noisy_coordinates, timestep_tensor, prediction, mask)
                settings = dict(config["local_objective"])
                settings["coordinate_scale_angstrom"] = float(config["coordinate_scale_angstrom"])
                losses = local_backbone_losses(x0, clean, mask, continuity, settings)
                profile = gradient_profile(v_loss, losses, tuple(model.parameters()))

                def grad_vector(loss: Any) -> Any:
                    grads = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True, allow_unused=True)
                    pieces = [grad.reshape(-1) for grad in grads if grad is not None]
                    return torch.cat(pieces) if pieces else loss.new_zeros((0,))

                v_vector = grad_vector(v_loss)
                v_norm = torch.linalg.vector_norm(v_vector)
                v_counts = finite_counts(v_vector)
                candidate_records: dict[str, Any] = {}
                all_counts = [
                    v_counts,
                    profile["v_finite_counts"],
                    *(profile["terms"][term] for term in TERMS),
                ]
                for candidate in config["selection_rule"]["ordering"]:
                    weights = coefficients[candidate][str(timestep)]
                    auxiliary, weighted = weighted_local_objective(losses, weights)
                    auxiliary_vector = grad_vector(auxiliary)
                    auxiliary_norm = torch.linalg.vector_norm(auxiliary_vector)
                    term_vectors = {term: grad_vector(weighted[term]) for term in TERMS}
                    pairwise = cosine_matrix(term_vectors)
                    terms: dict[str, Any] = {}
                    for term in TERMS:
                        vector = term_vectors[term]
                        counts = finite_counts(vector)
                        all_counts.append(counts)
                        norm = torch.linalg.vector_norm(vector)
                        terms[term] = {
                            **counts,
                            "coefficient": weights[term],
                            "weighted_gradient_norm": float(norm.detach()),
                            "weighted_gradient_ratio": float((norm / v_norm.clamp_min(1e-30)).detach()),
                            "gradient_contribution_cosine_with_v": float(
                                (torch.dot(vector, v_vector) / (norm * v_norm).clamp_min(1e-30)).detach()
                            ),
                        }
                    total_vector = grad_vector(v_loss + auxiliary)
                    candidate_records[candidate] = {
                        "terms": terms,
                        "individual_ratios": {term: terms[term]["weighted_gradient_ratio"] for term in TERMS},
                        "individual_term_v_cosines": {
                            term: terms[term]["gradient_contribution_cosine_with_v"] for term in TERMS
                        },
                        "pairwise_auxiliary_cosines": pairwise,
                        "combined_auxiliary_gradient_norm": float(auxiliary_norm.detach()),
                        "combined_auxiliary_to_v_ratio": float((auxiliary_norm / v_norm.clamp_min(1e-30)).detach()),
                        "combined_auxiliary_cosine_with_v": float(
                            (
                                torch.dot(auxiliary_vector, v_vector) / (auxiliary_norm * v_norm).clamp_min(1e-30)
                            ).detach()
                        ),
                        "total_to_v_ratio": float(
                            (torch.linalg.vector_norm(total_vector) / v_norm.clamp_min(1e-30)).detach()
                        ),
                    }
                exact_counts = {
                    "finite_count": sum(int(item["finite_count"]) for item in all_counts),
                    "total_count": sum(int(item["total_count"]) for item in all_counts),
                    "non_finite_count": sum(int(item["non_finite_count"]) for item in all_counts),
                }
                cell = {
                    "identity": {"length": length, "timestep": timestep, "sample_id": sample_id},
                    "v_gradient": {**v_counts, "norm": float(v_norm.detach())},
                    "raw_gradient_profile": profile,
                    "raw_losses": {term: float(losses[term].detach()) for term in TERMS},
                    "candidate_coefficients": candidate_records,
                    **exact_counts,
                    "all_finite": exact_counts["non_finite_count"] == 0
                    and exact_counts["finite_count"] == exact_counts["total_count"],
                }
                cells.append(cell)
        candidate_names = list(config["selection_rule"]["ordering"])
        gates = evaluate_candidate_gates(cells, config, candidate_names)
        selection = select_strongest_passing(gates, candidate_names)
        by_axis: dict[str, Any] = {}
        for axis in ("length", "timestep"):
            axis_values = config["lengths"] if axis == "length" else config["timesteps"]
            by_axis[axis] = {}
            for value in axis_values:
                subset = [cell for cell in cells if cell["identity"][axis] == value]
                by_axis[axis][str(value)] = {
                    name: {
                        "cell_count": len(subset),
                        "combined_auxiliary_to_v": {
                            "median": float(
                                np.median(
                                    [c["candidate_coefficients"][name]["combined_auxiliary_to_v_ratio"] for c in subset]
                                )
                            ),
                            "p90": float(
                                np.percentile(
                                    [
                                        c["candidate_coefficients"][name]["combined_auxiliary_to_v_ratio"]
                                        for c in subset
                                    ],
                                    90,
                                )
                            ),
                            "maximum": max(
                                c["candidate_coefficients"][name]["combined_auxiliary_to_v_ratio"] for c in subset
                            ),
                        },
                        "total_to_v": {
                            "median": float(
                                np.median([c["candidate_coefficients"][name]["total_to_v_ratio"] for c in subset])
                            ),
                            "p90": float(
                                np.percentile(
                                    [c["candidate_coefficients"][name]["total_to_v_ratio"] for c in subset], 90
                                )
                            ),
                            "maximum": max(c["candidate_coefficients"][name]["total_to_v_ratio"] for c in subset),
                        },
                        "individual_term_ratios": {
                            term: {
                                "median": float(
                                    np.median(
                                        [c["candidate_coefficients"][name]["individual_ratios"][term] for c in subset]
                                    )
                                ),
                                "p90": float(
                                    np.percentile(
                                        [c["candidate_coefficients"][name]["individual_ratios"][term] for c in subset],
                                        90,
                                    )
                                ),
                                "maximum": max(
                                    c["candidate_coefficients"][name]["individual_ratios"][term] for c in subset
                                ),
                            }
                            for term in TERMS
                        },
                    }
                    for name in candidate_names
                }
        report = {
            "version": config["version"],
            "mode": "gradient_calibration_v3",
            "status": "completed",
            "source_v2_report_sha256": V2_REPORT_SHA256,
            "source_v2_protocol_sha256": V2_PROTOCOL_SHA256,
            "configuration_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
            "panel_evidence": panel_report,
            "cells": cells,
            "candidate_gates": gates,
            "selection": selection,
            "calibration_authorizes_pilot": False,
            "selected_coefficient_set": selection["selected"],
            "review_required": True,
            "aggregation_axes": {axis: list(config[axis]) for axis in ("lengths", "timesteps")},
            "results_by_length_and_timestep": by_axis,
            "worst_cell_identities": {
                name: {
                    "maximum_combined_auxiliary_ratio": max(
                        cells, key=lambda cell: cell["candidate_coefficients"][name]["combined_auxiliary_to_v_ratio"]
                    )["identity"],
                    "maximum_total_ratio": max(
                        cells, key=lambda cell: cell["candidate_coefficients"][name]["total_to_v_ratio"]
                    )["identity"],
                    "maximum_term_ratios": {
                        term: max(
                            cells,
                            key=lambda cell: cell["candidate_coefficients"][name]["individual_ratios"][term],
                        )["identity"]
                        for term in TERMS
                    },
                }
                for name in candidate_names
            },
        }
        _atomic_json(staging / "report.json", report)
        _atomic_json(
            staging / "protocol.json",
            {
                "version": config["version"],
                "cell_count": len(cells),
                "model_forward_count": len(cells),
                "optimizer_updates": 0,
                "pairwise_auxiliary_cosine_matrix_per_cell": True,
                "pilot_authorized": False,
            },
        )
        staging.replace(output)
        _fsync_directory(output.parent)
        return report
    except BaseException:
        raise
