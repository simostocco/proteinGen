"""Strict multi-structure, schedule-aware E007 Phase-3I.2 calibration v2."""

from __future__ import annotations

import hashlib
import math
import numbers
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

TERMS = ("adjacent", "i_plus_2", "i_plus_3", "bond_angle_cosine", "discontinuity", "clash")
LENGTHS = (64, 128, 256, 384, 500)
TIMESTEPS = (25, 250, 425, 499)
V1_REPORT_SHA256 = "1ed084323d95ac6245aca3a17506a92dbabbd07a6993caab590304a576ca9144"


REQUIRED_PATHS = (
    "version output_dir calibration_output_dir device selected_checkpoint.path selected_checkpoint.sha256 "
    "selected_checkpoint.optimizer_update protected_evidence dataset_source_config coordinate_scale_angstrom "
    "diffusion_steps arms samplers lengths local_objective.convention "
    "local_objective.distance_normalizer_angstrom local_objective.discontinuity_threshold_angstrom "
    "local_objective.clash_threshold_angstrom local_objective.clash_minimum_sequence_separation "
    "local_objective.maximum_clash_pairs_per_structure local_objective.smooth_tail_beta "
    "local_objective.candidate_coefficient_sets local_objective.selected_coefficient_set "
    "calibration.report_path calibration.report_sha256 calibration.reviewed calibration.lengths "
    "calibration.timesteps calibration.structures_per_length calibration.selection_seed "
    "calibration.maximum_auxiliary_to_v_gradient_ratio calibration.minimum_total_to_v_gradient_ratio "
    "calibration.maximum_total_to_v_gradient_ratio calibration.maximum_samples_per_cell "
    "calibration.schedule_weighting calibration.candidate_selection_preference calibration.v1_report_sha256 "
    "pilot.maximum_optimizer_updates_per_arm pilot.evaluation_updates pilot.samples_per_length_per_sampler "
    "pilot.initialization pilot.optimizer_state pilot.scheduler_state pilot.paired_seed "
    "pilot.recovery_checkpoint_frequency guidance.strengths guidance.correction_steps "
    "guidance.maximum_displacement_angstrom guidance.adjacent_target_angstrom guidance.i_plus_2_target_angstrom "
    "guidance.i_plus_3_target_angstrom guidance.bond_angle_degrees guidance.discontinuity_threshold_angstrom "
    "guidance.clash_threshold_angstrom guidance.maximum_clash_pairs_per_structure numerics.allow_matmul_tf32 "
    "numerics.allow_cudnn_tf32 numerics.deterministic_algorithms numerics.autocast_enabled "
    "numerics.equivariance_policy memory.maximum_rss_mib memory.maximum_cuda_allocated_mib "
    "memory.maximum_cuda_reserved_mib runtime.estimated_calibration_model_forwards "
    "runtime.estimated_pilot_training_forwards runtime.estimated_pilot_denoising_forwards "
    "runtime.estimated_pilot_sampling_forwards runtime.maximum_calibration_wall_seconds "
    "runtime.maximum_pilot_wall_seconds publication.bootstrap_replicates publication.bootstrap_seed "
    "publication.maximum_failure_examples publication.minimum_improved_length_strata "
    "publication.maximum_global_metric_relative_degradation publication.maximum_diversity_relative_degradation "
    "publication.all_authorization_fields_false"
).split()


def validate_config(config: Any) -> dict[str, Any]:
    """Validate the complete execution contract before touching datasets or outputs."""
    if not isinstance(config, dict):
        raise ValueError("configuration: expected mapping")
    for dotted in REQUIRED_PATHS:
        value: Any = config
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"{dotted}: required field is missing")
            value = value[part]
        if value is None and dotted not in {
            "calibration.report_path",
            "calibration.report_sha256",
            "local_objective.selected_coefficient_set",
        }:
            raise ValueError(f"{dotted}: null is not allowed")
    if config["version"] != "e007_local_backbone_repair_calibration_v2":
        raise ValueError("version: expected e007_local_backbone_repair_calibration_v2")

    def integer(path: str, value: Any, low: int, high: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{path}: expected integer in [{low}, {high}]")

    def finite(path: str, value: Any, low: float, high: float, *, inclusive_low: bool = True) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
            or not (low <= value <= high if inclusive_low else low < value <= high)
        ):
            raise ValueError(f"{path}: expected finite number in {'[' if inclusive_low else '('}[{low}, {high}]")

    integer("calibration.selection_seed", config["calibration"]["selection_seed"], 0, 2**32 - 1)
    integer("diffusion_steps", config["diffusion_steps"], 1, 10000)
    integer("calibration.structures_per_length", config["calibration"]["structures_per_length"], 1, 100)
    integer("calibration.maximum_samples_per_cell", config["calibration"]["maximum_samples_per_cell"], 1, 10000)
    if type(config["calibration"]["lengths"]) is not list or tuple(config["calibration"]["lengths"]) != LENGTHS:
        raise ValueError("calibration.lengths: expected [64, 128, 256, 384, 500]")
    if type(config["calibration"]["timesteps"]) is not list or tuple(config["calibration"]["timesteps"]) != TIMESTEPS:
        raise ValueError("calibration.timesteps: expected [25, 250, 425, 499]")
    if type(config["protected_evidence"]) is not list or not config["protected_evidence"]:
        raise ValueError("protected_evidence: expected non-empty list")
    for i, row in enumerate(config["protected_evidence"]):
        for key in ("path", "sha256"):
            if not isinstance(row, dict) or not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"protected_evidence[{i}].{key}: expected non-empty string")
        if len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]):
            raise ValueError(f"protected_evidence[{i}].sha256: expected lowercase SHA-256")
    if not isinstance(config["calibration_output_dir"], str) or "calibration_v1" in config["calibration_output_dir"]:
        raise ValueError("calibration_output_dir: must be a v2 path, separate from protected v1")
    if not isinstance(config["output_dir"], str) or not config["output_dir"]:
        raise ValueError("output_dir: expected non-empty path")
    if config["local_objective"]["selected_coefficient_set"] is not None:
        raise ValueError("local_objective.selected_coefficient_set: must be null before calibration")
    coefficients = config["local_objective"]["candidate_coefficient_sets"]
    if not isinstance(coefficients, dict) or not coefficients:
        raise ValueError("local_objective.candidate_coefficient_sets: expected non-empty mapping")
    for name, terms in coefficients.items():
        if not isinstance(terms, dict) or set(terms) != set(TERMS):
            raise ValueError(f"local_objective.candidate_coefficient_sets.{name}: expected terms {list(TERMS)}")
        for term, value in terms.items():
            finite(f"local_objective.candidate_coefficient_sets.{name}.{term}", value, 0, 1, inclusive_low=False)
    for path in (
        "calibration.maximum_auxiliary_to_v_gradient_ratio",
        "calibration.minimum_total_to_v_gradient_ratio",
        "calibration.maximum_total_to_v_gradient_ratio",
    ):
        finite(path, config["calibration"][path.split(".")[-1]], 0, 100, inclusive_low=False)
    taper = config["calibration"].get("schedule_weighting")
    if taper != "alpha_over_alpha_plus_sigma":
        raise ValueError("calibration.schedule_weighting: unsupported taper policy")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        config = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"configuration: unable to read YAML: {exc}") from exc
    return validate_config(config)


def cosine_vp_schedule(steps: int = 500) -> list[float]:
    """Return alpha-bar values from the exact cosine_beta_schedule implementation."""
    import torch

    from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule

    beta = cosine_beta_schedule(steps).to(torch.float64)
    return torch.cumprod(1.0 - beta, dim=0).tolist()


def schedule_weight(alpha_bar: float) -> float:
    """Schedule-derived signal-fraction taper alpha/(alpha+sigma).

    It attenuates high-noise structural supervision; it is not the x0-from-v
    Jacobian, which is -sigma times the identity for this VP parameterization.
    """
    if not math.isfinite(alpha_bar) or not 0 < alpha_bar < 1:
        raise ValueError("alpha_bar must be finite and within (0, 1)")
    alpha = math.sqrt(alpha_bar)
    sigma = math.sqrt(1.0 - alpha_bar)
    return alpha / (alpha + sigma)


def candidate_weights(config: dict[str, Any], timestep: int) -> dict[str, dict[str, float]]:
    alpha_bar = cosine_vp_schedule(int(config["diffusion_steps"]))[timestep]
    taper = schedule_weight(alpha_bar)
    base = config["local_objective"]["candidate_coefficient_sets"]["very_conservative"]
    return {
        "very_conservative": dict(base),
        "schedule_tapered": {key: float(value) * taper for key, value in base.items()},
        "schedule_tapered_half": {key: float(value) * taper * 0.5 for key, value in base.items()},
    }


def summarize(values: list[float]) -> dict[str, float]:
    import numpy as np

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {key: math.nan for key in ("minimum", "median", "mean", "p90", "p95", "maximum")}
    return dict(
        zip(
            ("minimum", "median", "mean", "p90", "p95", "maximum"),
            map(
                float, (finite.min(), np.median(finite), finite.mean(), *np.percentile(finite, [90, 95]), finite.max())
            ),
            strict=True,
        )
    )


def evaluate_gates(rows: list[dict[str, Any]], candidates: list[str]) -> dict[str, Any]:
    def incomplete(record: dict[str, Any]) -> bool:
        if "all_finite" in record:
            return not record["all_finite"]
        if "finite_count" in record and "total_count" in record:
            return record["non_finite_count"] != 0 or record["finite_count"] != record["total_count"]
        # Legacy fractional coverage is telemetry only. It is not an exact finiteness test.
        return False

    outcomes = {}
    for candidate in candidates:
        reasons = []
        for row in rows:
            item = row["candidate_coefficients"][candidate]
            if not row.get("raw_and_gradient_finite", False):
                reasons.append("nonfinite_raw_or_gradient")
            if incomplete(row["gradient_profile"].get("v_finite_counts", {})) or any(
                incomplete(term) for term in row["gradient_profile"]["terms"].values()
            ):
                reasons.append("incomplete_finite_gradient_coverage")
            if item["total_to_v_gradient_ratio"] < 0.80 or item["total_to_v_gradient_ratio"] > 1.30:
                reasons.append("total_ratio_out_of_range")
            if any(term["weighted_gradient_ratio"] > 0.20 for term in item["terms"].values()):
                reasons.append("individual_auxiliary_ratio_exceeded")
            if any(
                not term["raw_gradient_finite"] or not term["weighted_gradient_finite"]
                for term in item["terms"].values()
            ):
                reasons.append("nonfinite_candidate_gradient")
            if any(
                term["eligible_count"] > 0 and term["weighted_gradient_norm"] == 0 for term in item["terms"].values()
            ):
                reasons.append("eligible_term_zero_gradient")
        outcomes[candidate] = {"passes": not reasons, "failure_counts": dict(Counter(reasons))}
    passing = [name for name, result in outcomes.items() if result["passes"]]
    selected = passing[0] if len(passing) == 1 else None
    return {
        "candidates": outcomes,
        "passing_sets": passing,
        "selected_set": selected,
        "selection_status": "selected_unique_pass" if selected else "fail_closed_no_unique_pass",
    }


def worst_cells(rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    ranked = sorted(
        rows, key=lambda row: row["candidate_coefficients"][candidate]["total_to_v_gradient_ratio"], reverse=True
    )
    return {"maximum_total_ratio": ranked[0]["identity"] if ranked else None}


def aggregate_report(rows: list[dict[str, Any]], candidates: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"by_timestep": {}, "by_length": {}, "candidate_summaries": {}}
    for axis in ("timestep", "length"):
        for value in sorted({row[axis] for row in rows}):
            subset = [row for row in rows if row[axis] == value]
            result[f"by_{axis}"][str(value)] = {
                "cell_count": len(subset),
                "raw_losses": {term: summarize([row["raw_losses"][term] for row in subset]) for term in TERMS},
                "v_loss": summarize([row["v_loss"] for row in subset]),
                "gradient_ratios": {
                    candidate: {
                        "total_to_v": summarize(
                            [row["candidate_coefficients"][candidate]["total_to_v_gradient_ratio"] for row in subset]
                        ),
                        "terms": {
                            term: summarize(
                                [
                                    row["candidate_coefficients"][candidate]["terms"][term]["weighted_gradient_ratio"]
                                    for row in subset
                                ]
                            )
                            for term in TERMS
                        },
                    }
                    for candidate in candidates
                },
            }
    for candidate in candidates:
        cosines = {
            term: [
                row["candidate_coefficients"][candidate]["terms"][term]["gradient_contribution_cosine_with_v"]
                for row in rows
            ]
            for term in TERMS
        }
        result["candidate_summaries"][candidate] = {
            "combined_auxiliary_to_v_gradient_ratio": summarize(
                [row["candidate_coefficients"][candidate]["combined_auxiliary_to_v_gradient_ratio"] for row in rows]
            ),
            "combined_auxiliary_cosine_with_v": summarize(
                [row["candidate_coefficients"][candidate]["combined_auxiliary_cosine_with_v"] for row in rows]
            ),
            "worst_cells": {
                "total_ratio": worst_cells(rows, candidate),
                "terms": {
                    term: max(
                        rows,
                        key=lambda row: row["candidate_coefficients"][candidate]["terms"][term][
                            "weighted_gradient_ratio"
                        ],
                    )["identity"]
                    for term in TERMS
                },
            },
            "cosine_distributions": {
                term: {
                    **summarize(values),
                    "count_below_0": sum(value < 0 for value in values),
                    "count_below_minus_0_25": sum(value < -0.25 for value in values),
                    "count_below_minus_0_5": sum(value < -0.5 for value in values),
                }
                for term, values in cosines.items()
            },
        }
    return result


def plan_only(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    alpha_bars = cosine_vp_schedule(int(config["diffusion_steps"]))
    taper_schedule = {}
    base = config["local_objective"]["candidate_coefficient_sets"]["very_conservative"]
    for timestep in TIMESTEPS:
        alpha_bar = alpha_bars[timestep]
        alpha = math.sqrt(alpha_bar)
        sigma = math.sqrt(1.0 - alpha_bar)
        taper = schedule_weight(alpha_bar)
        taper_schedule[str(timestep)] = {
            "alpha_bar": alpha_bar,
            "alpha": alpha,
            "sigma": sigma,
            "signal_fraction_taper": taper,
            "coefficient_multipliers": {
                "very_conservative": 1.0,
                "schedule_tapered": taper,
                "schedule_tapered_half": taper * 0.5,
            },
            "resulting_coefficients": {
                "very_conservative": {key: float(value) for key, value in base.items()},
                "schedule_tapered": {key: float(value) * taper for key, value in base.items()},
                "schedule_tapered_half": {key: float(value) * taper * 0.5 for key, value in base.items()},
            },
        }
    return {
        "mode": "plan_only",
        "version": config["version"],
        "selection_seed": config["calibration"]["selection_seed"],
        "configuration_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "lengths": list(LENGTHS),
        "timesteps": list(TIMESTEPS),
        "vp_parameterization": {
            "forward_equation": "x_t = alpha_t * x0 + sigma_t * epsilon",
            "velocity_equation": "v_t = alpha_t * epsilon - sigma_t * x0",
            "x0_from_velocity": "x0_hat = alpha_t * x_t - sigma_t * v_hat",
            "x0_from_velocity_jacobian": "d(x0_hat)/d(v_hat) = -sigma_t I",
        },
        "taper_definition": "alpha_t / (alpha_t + sigma_t)",
        "taper_interpretation": (
            "schedule-derived signal-fraction taper to reduce unreliable high-noise structural supervision; "
            "not the x0-from-v Jacobian"
        ),
        "v_mse_coefficient": 1.0,
        "taper_schedule": taper_schedule,
        "structures_per_cell": 4,
        "structure_timestep_cells": 80,
        "forward_count": 80,
        "staging_created": False,
        "checkpoint_loaded": False,
        "model_created": False,
        "cuda_initialized": False,
        "calibration_authorizes_pilot": False,
        "optimizer_created": False,
        "forward_pass": False,
        "backward_pass": False,
    }


def validate_panel(config_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization
    from protein_distance_diffusion.training.e007_local_backbone_repair import verify_protected_evidence

    config = load_config(config_path)
    verify_protected_evidence(config)
    source = localization.load_config(config["dataset_source_config"])
    source["panel"]["samples_per_length"] = 4
    source["panel"]["selection_seed"] = int(config["calibration"]["selection_seed"])
    localization.verify_prerequisites(source, full=True)
    compact, diagnostics = localization.select_validation_panel(source)
    panel, canonical = localization.reconstruct_authoritative_panel(source, compact)
    counts = Counter(int(row["selection"]["target_length"]) for row in panel)
    actual_counts = Counter(int(row["selection"]["actual_length"]) for row in panel)
    expected = {length: 4 for length in LENGTHS}
    if dict(counts) != expected or len({row["selection"]["sample_id"] for row in panel}) != 20:
        raise ValueError("v2 canonical panel must contain four distinct structures at each length")
    payload = {
        "selection_seed": config["calibration"]["selection_seed"],
        "panel_count": len(panel),
        "unique_selected_structures": len({row["selection"]["sample_id"] for row in panel}),
        "counts_by_length": {str(k): v for k, v in sorted(counts.items())},
        "counts_by_actual_length": {str(k): v for k, v in sorted(actual_counts.items())},
        "panel_identity_sha256": canonical["identity_sha256"],
        "work_cell_identity_sha256": hashlib.sha256(
            yaml.safe_dump(
                [
                    {
                        "sample_id": row["selection"]["sample_id"],
                        "length": int(row["selection"]["target_length"]),
                        "timestep": timestep,
                    }
                    for row in panel
                    for timestep in TIMESTEPS
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "structure_timestep_cells": len(panel) * len(TIMESTEPS),
        "selection_sha256": diagnostics["record_sha256"],
        "model_created": False,
        "checkpoint_loaded": False,
        "cuda_initialized": False,
        "staging_created": False,
        "optimizer_created": False,
        "forward_pass": False,
        "backward_pass": False,
    }
    return payload, panel


def run_calibration(config_path: str | Path) -> dict[str, Any]:
    """Execution entry point. Panel is fully validated before imports with runtime side effects."""
    config = load_config(config_path)
    panel_report, panel = validate_panel(config_path)
    # V2 execution is intentionally exposed only after panel validation. It remains a zero-update
    # diagnostic; this function is not invoked by plan-only.
    import resource

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

    output = Path(config["calibration_output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"v2 calibration output exists: {output} or {staging}")
    source = localization.load_config(config["dataset_source_config"])
    prepared = [localization._prepared_reference(source, row["canonical_row"]) for row in panel]
    if config["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v2 calibration requires configured CUDA")
    device = torch.device("cuda")
    model = localization._load_model(source, device).requires_grad_(True).train()
    diffusion = CoordinateVPDiffusion(int(config["diffusion_steps"]))
    staging.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    try:
        for record in prepared:
            length = int(record["lengths"][0])
            for timestep_value in TIMESTEPS:
                sample_seed = int(hashlib.sha256(record["sample_ids"][0].encode()).hexdigest()[:8], 16)
                seed = 3915000 + length * 1000 + timestep_value + sample_seed
                generator = torch.Generator(device=device).manual_seed(seed)
                clean = record["coordinates"].to(device)
                mask = record["residue_mask"].to(device)
                continuity = record["chain_continuity_mask"].to(device)
                timestep = torch.tensor([timestep_value], dtype=torch.long, device=device)
                batch = diffusion.make_training_batch(clean, mask, timesteps=timestep, generator=generator)
                prediction = model(batch.noisy_coordinates, timestep, record["lengths"].to(device), mask, continuity)[
                    "v_prediction"
                ]
                v_loss = uniform_coordinate_v_mse(prediction, batch.coordinate_v_target, mask)
                x0 = diffusion.reconstruct_x0(batch.noisy_coordinates, timestep, prediction, mask)
                settings = dict(config["local_objective"])
                settings["coordinate_scale_angstrom"] = float(config["coordinate_scale_angstrom"])
                losses = local_backbone_losses(x0, clean, mask, continuity, settings)
                profile = gradient_profile(v_loss, losses, tuple(model.parameters()))
                v_gradients = torch.autograd.grad(
                    v_loss, tuple(model.parameters()), retain_graph=True, allow_unused=True
                )
                v_vector = torch.cat([g.reshape(-1) for g in v_gradients if g is not None])
                candidates: dict[str, Any] = {}
                for name, weights in candidate_weights(config, timestep_value).items():
                    auxiliary, weighted = weighted_local_objective(losses, weights)
                    total = v_loss + auxiliary
                    total_grad = torch.autograd.grad(
                        total, tuple(model.parameters()), retain_graph=True, allow_unused=True
                    )
                    total_vec = torch.cat([g.reshape(-1) for g in total_grad if g is not None])
                    auxiliary_grad = torch.autograd.grad(
                        auxiliary, tuple(model.parameters()), retain_graph=True, allow_unused=True
                    )
                    auxiliary_vec = torch.cat([g.reshape(-1) for g in auxiliary_grad if g is not None])
                    v_norm = profile["v_gradient_norm"]
                    term_records = {}
                    for term in TERMS:
                        raw = torch.autograd.grad(
                            losses[term], tuple(model.parameters()), retain_graph=True, allow_unused=True
                        )
                        weighted_grad = torch.autograd.grad(
                            weighted[term], tuple(model.parameters()), retain_graph=True, allow_unused=True
                        )
                        raw_vec = torch.cat([g.reshape(-1) for g in raw if g is not None])
                        weighted_vec = torch.cat([g.reshape(-1) for g in weighted_grad if g is not None])
                        term_records[term] = {
                            "weighted_gradient_norm": float(torch.linalg.vector_norm(weighted_vec)),
                            "raw_gradient_norm": float(torch.linalg.vector_norm(raw_vec)),
                            "raw_gradient_finite": bool(torch.isfinite(raw_vec).all()),
                            "weighted_gradient_finite": bool(torch.isfinite(weighted_vec).all()),
                            "weighted_gradient_ratio": float(
                                torch.linalg.vector_norm(weighted_vec) / max(v_norm, 1e-30)
                            ),
                            "gradient_contribution_cosine_with_v": float(
                                torch.dot(
                                    weighted_vec,
                                    v_vector,
                                )
                                / (torch.linalg.vector_norm(weighted_vec) * max(v_norm, 1e-30))
                            ),
                            "eligible_count": int(losses["denominators"][term].sum().item()),
                            "weighted_loss": float(weighted[term].detach()),
                        }
                    candidates[name] = {
                        "weighted_losses": {k: float(v.detach()) for k, v in weighted.items()},
                        "weighted_auxiliary_loss": float(auxiliary.detach()),
                        "terms": term_records,
                        "combined_auxiliary_gradient_norm": float(torch.linalg.vector_norm(auxiliary_vec)),
                        "combined_auxiliary_to_v_gradient_ratio": float(
                            torch.linalg.vector_norm(auxiliary_vec) / max(v_norm, 1e-30)
                        ),
                        "combined_auxiliary_cosine_with_v": float(
                            torch.dot(auxiliary_vec, v_vector)
                            / (torch.linalg.vector_norm(auxiliary_vec) * max(v_norm, 1e-30))
                        ),
                        "total_gradient_norm": float(torch.linalg.vector_norm(total_vec)),
                        "total_to_v_gradient_ratio": float(torch.linalg.vector_norm(total_vec) / max(v_norm, 1e-30)),
                    }
                rows.append(
                    {
                        "identity": {
                            "length": length,
                            "timestep": timestep_value,
                            "sample_id": record["sample_ids"][0],
                        },
                        "length": length,
                        "timestep": timestep_value,
                        "sample_id": record["sample_ids"][0],
                        "raw_losses": {k: float(losses[k].detach()) for k in TERMS},
                        "mask_eligibility_counts": {k: int(losses["denominators"][k].sum()) for k in TERMS},
                        "v_loss": float(v_loss.detach()),
                        "gradient_profile": profile,
                        "candidate_coefficients": candidates,
                        "raw_and_gradient_finite": bool(
                            all(math.isfinite(float(losses[k].detach())) for k in TERMS)
                            and all(
                                math.isfinite(x)
                                for k in TERMS
                                for x in profile["terms"][k].values()
                                if isinstance(x, (int, float))
                            )
                            and all(
                                term["raw_gradient_finite"] and term["weighted_gradient_finite"]
                                for candidate in candidates.values()
                                for term in candidate["terms"].values()
                            )
                        ),
                    }
                )
        gates = evaluate_gates(rows, list(rows[0]["candidate_coefficients"]))
        report = {
            "version": config["version"],
            "mode": "gradient_calibration_v2",
            "status": "completed",
            "cells": rows,
            "panel_evidence": panel_report,
            "candidate_gates": gates,
            "aggregates": aggregate_report(rows, list(rows[0]["candidate_coefficients"])),
            "calibration_authorizes_pilot": False,
            "selected_coefficient_set": gates["selected_set"],
            "review_required": True,
            "signal_fraction_tapers_by_timestep": {
                str(t): schedule_weight(cosine_vp_schedule(int(config["diffusion_steps"]))[t]) for t in TIMESTEPS
            },
            "vp_parameterization": {
                "forward_equation": "x_t = alpha_t * x0 + sigma_t * epsilon",
                "velocity_equation": "v_t = alpha_t * epsilon - sigma_t * x0",
                "x0_from_velocity": "x0_hat = alpha_t * x_t - sigma_t * v_hat",
                "x0_from_velocity_jacobian": "d(x0_hat)/d(v_hat) = -sigma_t I",
                "taper_definition": "alpha_t / (alpha_t + sigma_t)",
                "taper_interpretation": (
                    "schedule-derived signal-fraction taper to reduce unreliable high-noise structural supervision; "
                    "not the x0-from-v Jacobian"
                ),
            },
            "protected_v1_report_sha256": V1_REPORT_SHA256,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        _atomic_json(staging / "report.json", report)
        _atomic_json(
            staging / "protocol.json",
            {
                "version": config["version"],
                "status": "completed",
                "cell_count": len(rows),
                "forward_count": len(rows),
                "zero_optimizer_updates": True,
                "vp_parameterization": {
                    "forward_equation": "x_t = alpha_t * x0 + sigma_t * epsilon",
                    "velocity_equation": "v_t = alpha_t * epsilon - sigma_t * x0",
                    "x0_from_velocity": "x0_hat = alpha_t * x_t - sigma_t * v_hat",
                    "x0_from_velocity_jacobian": "d(x0_hat)/d(v_hat) = -sigma_t I",
                    "taper_definition": "alpha_t / (alpha_t + sigma_t)",
                },
            },
        )
        staging.replace(output)
        _fsync_directory(output.parent)
        return report
    except BaseException:
        raise
