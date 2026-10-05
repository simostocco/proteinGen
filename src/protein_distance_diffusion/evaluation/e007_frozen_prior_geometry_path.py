"""Bounded E007 Phase 4C.2 geometry-conditioning-path diagnostics."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import random
import resource
from collections import defaultdict
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from protein_distance_diffusion.evaluation.e007_pretrained_loaders import CANONICAL
from protein_distance_diffusion.evaluation.e007_pretrained_loaders_v3 import (
    load_reviewed_candidate,
    parameter_identity_hash,
)
from protein_distance_diffusion.models.e007_frozen_prior_geometry_capacity import (
    trainable_parameter_count,
)
from protein_distance_diffusion.training import e007_frozen_prior_geometry_capacity as phase4c1
from protein_distance_diffusion.training.coordinate_diffusion import coordinates_to_distance_matrix

VERSION = "e007_frozen_prior_geometry_path_diagnostic_v1"
CAPACITIES = ("small", "medium")
ARMS = ("correct_geometry", "shuffled_geometry", "null_geometry")
POLICIES = ("existing_learned_gate", "gate_logit_zero")
NON_AUTHORIZING = {
    "authorizes_training": False,
    "authorizes_production_training": False,
    "authorizes_joint_training": False,
    "authorizes_additional_coordinate_training": False,
    "authorizes_sequence_conditioned_coordinate_generation": False,
    "authorizes_progen2_unfreezing": False,
    "authorizes_prior_selection": False,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or config.get("version") != VERSION:
        raise ValueError("E007 Phase 4C.2 configuration version contradiction")
    representation = config["representation"]
    if representation["updates"] != [0, 1000] or representation["arms"] != list(ARMS):
        raise ValueError("E007 Phase 4C.2 checkpoint/arm contract changed")
    sweep = [float(value) for value in config["gate_sweep"]["effective_gate_values"]]
    if sweep != [0.0, 0.018, 0.05, 0.25, 0.5, 1.0]:
        raise ValueError("E007 Phase 4C.2 gate sweep changed")
    optimization = config["optimization_diagnostic"]
    if optimization["capacities"] != list(CAPACITIES) or optimization["arms"] != list(ARMS):
        raise ValueError("E007 Phase 4C.2 optimization arm contract changed")
    if optimization["policies"] != list(POLICIES):
        raise ValueError("E007 Phase 4C.2 optimization policy contract changed")
    if not 1 <= int(optimization["maximum_updates"]) <= 200:
        raise ValueError("E007 Phase 4C.2 exceeds its 200-update bound")
    expected_evaluations = [value for value in [0, 25, 50, 100, 200] if value <= optimization["maximum_updates"]]
    if optimization["evaluation_updates"] != expected_evaluations:
        raise ValueError("E007 Phase 4C.2 evaluation schedule changed")
    if set(config["decision_categories"]) != {
        "conditioning_path_defect",
        "gate_suppression_supported",
        "optimization_insufficient",
        "representation_not_discriminative",
        "architecture_not_using_matching_geometry",
        "bounded_diagnostic_inconclusive",
    }:
        raise ValueError("E007 Phase 4C.2 decision categories changed")
    return config


def _iter_pins(config: dict[str, Any]):
    yield from config["prerequisites"].items()
    for capacity, arms in config["conditioner_checkpoints"].items():
        for arm, updates in arms.items():
            for update, record in updates.items():
                yield f"checkpoint:{capacity}:{arm}:{update}", record


def verify_prerequisites(config: dict[str, Any]) -> dict[str, str]:
    observed = {}
    for name, record in _iter_pins(config):
        path = Path(record["path"])
        digest = sha256_file(path)
        if digest != str(record["sha256"]):
            raise ValueError(f"E007 Phase 4C.2 prerequisite hash contradiction: {name}")
        observed[name] = digest
    protocol = json.loads(Path(config["prerequisites"]["phase4c1_protocol"]["path"]).read_text())
    if (
        protocol.get("status") != "completed_non_authorizing"
        or protocol.get("mode") != "pilot"
        or protocol.get("configuration_sha256") != config["prerequisites"]["phase4c1_config"]["sha256"]
        or protocol.get("report_sha256") != config["prerequisites"]["phase4c1_report"]["sha256"]
        or any(protocol.get(key) is not False for key in NON_AUTHORIZING if key in protocol)
    ):
        raise ValueError("E007 Phase 4C.2 Phase 4C.1 protocol contract contradiction")
    report = json.loads(Path(config["prerequisites"]["phase4c1_report"]["path"]).read_text())
    identities = {(row["capacity"], row["arm"]) for row in report.get("results", [])}
    if report.get("status") != "completed_non_authorizing" or identities != {
        (capacity, arm) for capacity in CAPACITIES for arm in ARMS
    }:
        raise ValueError("E007 Phase 4C.2 Phase 4C.1 report contract contradiction")
    return observed


def injection_contract() -> dict[str, Any]:
    """Publish the audited static dataflow without executing a model."""
    return {
        "small": {
            "entry": "token embeddings before ProGen transformer block 0",
            "operation": "inputs_embeds = frozen_token_embedding + gated_conditioning",
            "depths": ["input_embedding"],
            "residual_order": "conditioning is present before every frozen transformer block",
        },
        "medium": {
            "entry": "forward-pre-hook on frozen ProGen transformer blocks",
            "operation": "block_hidden_input = block_hidden_input + gated_conditioning",
            "depths": [0, 5, 11],
            "residual_order": (
                "addition precedes each selected frozen block; subsequent frozen transforms may attenuate it"
            ),
        },
        "alignment": {
            "framing": "[BOS, biological residues, EOS, padding]",
            "geometry_nonzero_positions": "biological residues only",
            "bos_eos_padding": "exact zero",
            "causal_targets": "logit positions 0..N-1 predict canonical residues 1..N",
            "target_exposure": False,
            "correct_shuffled_pairing": (
                "same recipient sequence, target tokens, masks, and objective; geometry donor alone changes"
            ),
        },
        "masking": {
            "small": "biological residue mask and pair mask",
            "medium": "biological residue mask plus same-chain continuity pair mask",
        },
    }


def plan(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    protected = verify_prerequisites(config)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase 4C.2 output already exists: {output}")
    optimization = config["optimization_diagnostic"]
    return {
        "status": "planned_non_authorizing",
        "version": VERSION,
        "configuration_sha256": sha256_file(config_path),
        "checkpoint_count": sum(1 for name in protected if name.startswith("checkpoint:")),
        "capacities": list(CAPACITIES),
        "arms": list(ARMS),
        "gate_values": config["gate_sweep"]["effective_gate_values"],
        "optimization_policies": list(POLICIES),
        "isolated_optimization_arm_count": len(CAPACITIES) * len(ARMS) * len(POLICIES),
        "maximum_updates_per_optimization_arm": optimization["maximum_updates"],
        "injection_contract": injection_contract(),
        "model_created": False,
        "cuda_initialized": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "dataset_scanned": False,
        "output_created": False,
        "protected_hashes": protected,
        **NON_AUTHORIZING,
    }


def tensor_statistics(value: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
    selected = value
    if mask is not None:
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        selected = value.masked_select(mask.expand_as(value))
    if selected.numel() == 0:
        return {"element_count": 0, "finite": True, "rms": 0.0, "variance": 0.0, "sha256": tensor_sha256(value)}
    floating = selected.detach().float()
    return {
        "element_count": int(selected.numel()),
        "finite": bool(torch.isfinite(floating).all()),
        "rms": float(floating.square().mean().sqrt()),
        "variance": float(floating.var(unbiased=False)),
        "sha256": tensor_sha256(value),
    }


def representation_difference(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if mask is None:
        left = first.detach().flatten().float()
        right = second.detach().flatten().float()
    else:
        expanded = mask
        while expanded.ndim < first.ndim:
            expanded = expanded.unsqueeze(-1)
        left = first.detach().masked_select(expanded.expand_as(first)).float()
        right = second.detach().masked_select(expanded.expand_as(second)).float()
    if left.numel() == 0 or left.shape != right.shape:
        raise ValueError("E007 Phase 4C.2 representation comparison mask is empty or inconsistent")
    return {
        "rms_difference": float((left - right).square().mean().sqrt()),
        "cosine_distance": float(1.0 - F.cosine_similarity(left, right, dim=0)),
    }


def o3_tensor_comparison(
    reference: torch.Tensor,
    transformed: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Compare invariant tensors with a scale-aware RMS criterion."""
    if reference.shape != transformed.shape:
        raise ValueError("E007 Phase 4C.2 O(3) tensor shape contradiction")
    left = reference.detach().to(torch.float64)
    right = transformed.detach().to(torch.float64)
    expanded = None
    if mask is not None:
        expanded = mask.detach().to(device=left.device, dtype=torch.bool)
        while expanded.ndim < left.ndim:
            expanded = expanded.unsqueeze(-1)
        try:
            expanded = expanded.expand_as(left)
        except RuntimeError as error:
            raise ValueError("E007 Phase 4C.2 O(3) mask shape contradiction") from error
        selected_left = left.masked_select(expanded)
        selected_right = right.masked_select(expanded)
    else:
        selected_left = left.reshape(-1)
        selected_right = right.reshape(-1)
    if selected_left.numel() == 0:
        raise ValueError("E007 Phase 4C.2 O(3) comparison has no valid values")
    residual = selected_right - selected_left
    rms_error = residual.square().mean().sqrt()
    reference_rms = selected_left.square().mean().sqrt()
    threshold = reference_rms * float(relative_tolerance) + float(absolute_tolerance)
    padded = (right - left).masked_select(~expanded) if expanded is not None else left.new_empty(0)
    finite = bool(torch.isfinite(selected_left).all() and torch.isfinite(selected_right).all())
    return {
        "maximum_absolute_error": float(residual.abs().max().cpu()),
        "rms_error": float(rms_error.cpu()),
        "relative_rms_error": float((rms_error / reference_rms.clamp_min(1e-12)).cpu()),
        "reference_tensor_norm": float(torch.linalg.vector_norm(selected_left).cpu()),
        "reference_rms_scale": float(reference_rms.cpu()),
        "reference_maximum_absolute_scale": float(selected_left.abs().max().cpu()),
        "valid_element_count": int(selected_left.numel()),
        "padded_element_count": int(padded.numel()),
        "padded_maximum_absolute_error": float(padded.abs().max().cpu()) if padded.numel() else 0.0,
        "padded_exact_zero": bool(not padded.numel() or torch.count_nonzero(padded) == 0),
        "finite": finite,
        "hashes_differ": tensor_sha256(reference) != tensor_sha256(transformed),
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "rms_threshold": float(threshold.cpu()),
        "criterion": "rms_error <= absolute_tolerance + relative_tolerance * reference_rms_scale",
        "passed": bool(finite and rms_error <= threshold),
    }


def _o3_transformed_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    coordinates = row["coordinates"]
    mask = row["residue_mask"].bool()
    if coordinates.shape != (len(mask), 3):
        raise ValueError("E007 Phase 4C.2 O(3) coordinate/mask shape contradiction")
    if bool(torch.count_nonzero(coordinates[~mask])):
        raise ValueError("E007 Phase 4C.2 O(3) input padding is not exactly zero")
    source = coordinates.new_tensor([[0.4, -0.3, 0.7], [0.2, 0.9, 0.1], [-0.8, 0.1, 0.5]])
    transformation, _ = torch.linalg.qr(source)
    identity = torch.eye(3, dtype=transformation.dtype, device=transformation.device)
    orthogonality_error = (transformation.T @ transformation - identity).abs().max()
    determinant = torch.linalg.det(transformation)
    structural_tolerance = 8 * torch.finfo(transformation.dtype).eps
    if orthogonality_error > structural_tolerance or abs(abs(float(determinant)) - 1.0) > structural_tolerance:
        raise ValueError("E007 Phase 4C.2 O(3) transformation is not orthogonal")
    transformed_coordinates = torch.zeros_like(coordinates)
    transformed_coordinates[mask] = coordinates[mask] @ transformation + 7.0
    if bool(torch.count_nonzero(transformed_coordinates[~mask])):
        raise ValueError("E007 Phase 4C.2 O(3) transformed padding is not exactly zero")
    transformed = dict(row)
    transformed["coordinates"] = transformed_coordinates
    metadata = {
        "matrix": transformation.detach().cpu().tolist(),
        "dtype": str(transformation.dtype),
        "determinant": float(determinant.detach().cpu()),
        "orthogonality_maximum_absolute_error": float(orthogonality_error.detach().cpu()),
        "structural_tolerance": float(structural_tolerance),
        "translation": [7.0, 7.0, 7.0],
        "valid_residue_count": int(mask.sum()),
        "padded_residue_count": int((~mask).sum()),
        "transformation_applied_only_to_valid_residues": True,
        "padding_exact_zero": True,
        "residue_order_unchanged": True,
        "residue_mask_unchanged": bool(torch.equal(row["residue_mask"], transformed["residue_mask"])),
        "continuity_mask_unchanged": bool(torch.equal(row["continuity_mask"], transformed["continuity_mask"])),
    }
    return transformed, metadata


def _validate_o3_structure(
    reference: dict[str, Any],
    transformed: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, bool]:
    matrix = torch.tensor(metadata["matrix"], dtype=torch.float64)
    identity = torch.eye(3, dtype=torch.float64)
    tolerance = float(metadata["structural_tolerance"])
    mask = reference["residue_mask"].bool()
    checks = {
        "orthogonal_transform": bool((matrix.T @ matrix - identity).abs().max() <= tolerance),
        "unit_determinant_magnitude": abs(abs(float(torch.linalg.det(matrix))) - 1.0) <= tolerance,
        "sample_identity_unchanged": reference["sample_id"] == transformed["sample_id"],
        "sequence_and_residue_order_unchanged": reference["sequence"] == transformed["sequence"],
        "residue_mask_unchanged": bool(torch.equal(reference["residue_mask"], transformed["residue_mask"])),
        "continuity_mask_unchanged": bool(torch.equal(reference["continuity_mask"], transformed["continuity_mask"])),
        "padding_exact_zero": bool(torch.count_nonzero(transformed["coordinates"][~mask]) == 0),
    }
    if not all(checks.values()):
        raise ValueError(f"E007 Phase 4C.2 O(3) structural contradiction: {checks}")
    return checks


def _canonical_inputs(
    tokenizer: Any,
    sequence: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    framed = phase4c1.progen_framed_sequence_contract(tokenizer, sequence, position_capacity=1024)
    values = framed["framed_model_input"]["ids"]
    ids = torch.tensor(values, dtype=torch.long, device=device)[None]
    canonical_ids = [tokenizer.token_to_id(residue) for residue in CANONICAL]
    remap = {value: index for index, value in enumerate(canonical_ids)}
    targets = torch.tensor([remap[value] for value in values[1:-1]], dtype=torch.long, device=device)
    return ids, targets, canonical_ids


def _conditioner_forward(
    conditioner: torch.nn.Module,
    capacity: str,
    row: dict[str, Any],
    *,
    arm: str,
    effective_gate: float | None,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    captures: dict[str, torch.Tensor] = {}
    hooks = ExitStack()

    def capture(name: str):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            captures[name] = output

        return hook

    hooks.callback(conditioner.pair_encoder.register_forward_hook(capture("pair_encoder_output")).remove)
    if capacity == "small":
        hooks.callback(conditioner.conditioning_adapter.register_forward_hook(capture("output_adapter_output")).remove)
    else:
        hooks.callback(conditioner.residue_input.register_forward_hook(capture("residue_input_output")).remove)
        hooks.callback(conditioner.message_blocks[-1].register_forward_hook(capture("residue_representation")).remove)
        hooks.callback(conditioner.output_adapter.register_forward_hook(capture("output_adapter_output")).remove)
    coordinates = row["coordinates"][None].to(device)
    residue_mask = row["residue_mask"][None].to(device)
    with hooks:
        if capacity == "small":
            returned = conditioner(coordinates, residue_mask, null_geometry=arm == "null_geometry")
            raw = captures["output_adapter_output"]
            gate = float(torch.sigmoid(conditioner.gate_logit.detach())) if effective_gate is None else effective_gate
            biological = raw * gate * residue_mask[..., None]
            captures.setdefault("residue_representation", captures["pair_encoder_output"])
        else:
            continuity = row["continuity_mask"][None].to(device)
            returned = conditioner(
                coordinates,
                residue_mask,
                continuity,
                null_geometry=arm == "null_geometry",
            )
            biological = returned
    captures["returned_output"] = returned
    captures["biological_conditioning"] = biological
    return biological, captures


def diagnostic_forward(
    prior: Any,
    tokenizer: Any,
    conditioner: torch.nn.Module,
    capacity: str,
    row: dict[str, Any],
    *,
    geometry_row: dict[str, Any],
    arm: str,
    effective_gate: float | None,
    device: torch.device,
) -> dict[str, Any]:
    ids, targets, canonical_ids = _canonical_inputs(tokenizer, str(row["sequence"]), device)
    biological, representations = _conditioner_forward(
        conditioner,
        capacity,
        geometry_row,
        arm=arm,
        effective_gate=effective_gate,
        device=device,
    )
    framed = phase4c1._framed_geometry(biological, len(row["sequence"]), framed_length=ids.shape[1])
    embeddings = prior.get_input_embeddings()(ids).detach()
    injection_records = []
    if capacity == "small":
        before = embeddings
        after = embeddings + framed
        injection_records.append(("input_embedding", before, after, framed))
        logits = prior(inputs_embeds=after, attention_mask=torch.ones_like(ids)).logits[0]
    else:
        gate_values = conditioner.gate_values() if effective_gate is None else [effective_gate] * 3
        stack = ExitStack()
        for injection_index, depth in enumerate(conditioner.injection_depths):
            addition = framed * float(gate_values[injection_index])

            def inject(_module: Any, arguments: tuple[Any, ...], *, depth_value: int = depth, value=addition):
                before = arguments[0]
                after = before + value.to(before.dtype)
                injection_records.append((depth_value, before, after, value))
                return (after, *arguments[1:])

            stack.callback(prior.transformer.h[depth].register_forward_pre_hook(inject).remove)
        with stack:
            logits = prior(inputs_embeds=embeddings, attention_mask=torch.ones_like(ids)).logits[0]
    canonical_logits = logits[: len(row["sequence"]), canonical_ids].float()
    losses = F.cross_entropy(canonical_logits, targets, reduction="none")
    probabilities = canonical_logits.softmax(dim=-1)
    prediction = canonical_logits.argmax(dim=-1)
    top5 = canonical_logits.topk(5, dim=-1).indices.eq(targets[:, None]).any(dim=-1)
    mask = geometry_row["residue_mask"][None].to(device)
    representation_stats = {
        name: tensor_statistics(value, mask if value.ndim == 3 and value.shape[1] == mask.shape[1] else None)
        for name, value in representations.items()
    }
    injections = []
    for depth, before, after, addition in injection_records:
        hidden_rms = before.detach().float().square().mean().sqrt()
        injection_rms = addition.detach().float().square().mean().sqrt()
        injections.append(
            {
                "depth": depth,
                "hidden_rms_before": float(hidden_rms),
                "hidden_rms_after": float(after.detach().float().square().mean().sqrt()),
                "injected_residual_rms": float(injection_rms),
                "injection_to_hidden_rms_ratio": float(injection_rms / hidden_rms.clamp_min(1e-12)),
            }
        )
    padded = phase4c1._framed_geometry(biological, len(row["sequence"]), framed_length=ids.shape[1] + 3)
    biological_nonzero = biological.float().abs().amax(dim=-1) > 0
    return {
        "sample_id": row["sample_id"],
        "cross_entropy": float(losses.detach().mean()),
        "top1_accuracy": float((prediction == targets).detach().float().mean()),
        "top5_accuracy": float(top5.detach().float().mean()),
        "logit_rms": float(canonical_logits.detach().square().mean().sqrt()),
        "canonical_logits": canonical_logits.detach(),
        "probabilities": probabilities.detach(),
        "targets": targets.detach(),
        "per_token_cross_entropy": losses.detach(),
        "representations": representations,
        "representation_statistics": representation_stats,
        "injections": injections,
        "fraction_biological_positions_nonzero": float(biological_nonzero.float().mean()),
        "bos_exact_zero": bool(torch.count_nonzero(framed[:, 0]) == 0),
        "eos_exact_zero": bool(torch.count_nonzero(framed[:, len(row["sequence"]) + 1]) == 0),
        "padding_exact_zero": bool(torch.count_nonzero(padded[:, len(row["sequence"]) + 2 :]) == 0),
    }


@torch.no_grad()
def o3_invariance_diagnostic(
    prior: Any,
    tokenizer: Any,
    conditioner: torch.nn.Module,
    capacity: str,
    row: dict[str, Any],
    *,
    reference_output: dict[str, Any] | None,
    device: torch.device,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Audit end-to-end O(3) invariance without requiring bit identity."""
    transformed_row, transformation = _o3_transformed_row(row)
    structural_checks = _validate_o3_structure(row, transformed_row, transformation)
    reference = reference_output or diagnostic_forward(
        prior,
        tokenizer,
        conditioner,
        capacity,
        row,
        geometry_row=row,
        arm="correct_geometry",
        effective_gate=1.0,
        device=device,
    )
    transformed = diagnostic_forward(
        prior,
        tokenizer,
        conditioner,
        capacity,
        row,
        geometry_row=transformed_row,
        arm="correct_geometry",
        effective_gate=1.0,
        device=device,
    )
    residue_mask = row["residue_mask"].bool()
    pair_mask = residue_mask[:, None] & residue_mask[None, :]
    comparisons = {}
    for name in sorted(set(reference["representations"]) & set(transformed["representations"])):
        left = reference["representations"][name]
        right = transformed["representations"][name]
        if left.ndim == 4 and left.shape[1:3] == pair_mask.shape:
            mask = pair_mask[None].to(device)
        elif left.ndim == 3 and left.shape[1] == len(residue_mask):
            mask = residue_mask[None].to(device)
        else:
            mask = None
        comparisons[name] = {
            **o3_tensor_comparison(
                left,
                right,
                mask=mask,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            ),
            "required_for_o3_gate": name == "biological_conditioning",
        }
    logit_comparison = o3_tensor_comparison(
        reference["canonical_logits"],
        transformed["canonical_logits"],
        mask=residue_mask.to(device),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    matrix = torch.tensor(transformation["matrix"], dtype=torch.float64)
    original_float64 = row["coordinates"].to(torch.float64)
    intended_float64 = torch.zeros_like(original_float64)
    intended_float64[residue_mask] = original_float64[residue_mask] @ matrix + 7.0
    reference_distances = coordinates_to_distance_matrix(
        original_float64,
        residue_mask,
        diagnostic_float64=True,
    )
    transformed_distances = coordinates_to_distance_matrix(
        intended_float64,
        residue_mask,
        diagnostic_float64=True,
    )
    structural_relative_tolerance = float(transformation["structural_tolerance"])
    distance_comparison = o3_tensor_comparison(
        reference_distances,
        transformed_distances,
        mask=pair_mask,
        absolute_tolerance=1e-12,
        relative_tolerance=structural_relative_tolerance,
    )
    positions = torch.arange(len(residue_mask), dtype=torch.int64)
    separation = (positions[:, None] - positions[None, :]).abs()
    sequence_separation = {
        "unchanged": bool(torch.equal(separation, separation.clone())),
        "reference_sha256": tensor_sha256(separation),
        "transformed_sha256": tensor_sha256(separation.clone()),
        "relative_tolerance": 0.0,
        "absolute_tolerance": 0.0,
    }
    required_representation = comparisons["biological_conditioning"]
    structural_checks.update(
        {
            "sequence_separation_exact": sequence_separation["unchanged"],
            "canonical_pair_distances": distance_comparison["passed"],
        }
    )
    all_finite = bool(
        distance_comparison["finite"]
        and logit_comparison["finite"]
        and all(value["finite"] for value in comparisons.values())
    )
    passed = bool(
        all(structural_checks.values())
        and all_finite
        and required_representation["passed"]
        and logit_comparison["passed"]
    )
    return {
        "status": "passed" if passed else "failed",
        "transformation": transformation,
        "structural_checks": structural_checks,
        "canonical_pair_distance_comparison": distance_comparison,
        "sequence_separation": sequence_separation,
        "representations": comparisons,
        "canonical_logits": logit_comparison,
        "required_representation_names": ["biological_conditioning", "canonical_logits"],
        "intermediate_representation_policy": (
            "recorded diagnostically; end-to-end gate uses final biological conditioning and canonical logits"
        ),
        "finite": all_finite,
    }


def paired_logit_effect(reference: dict[str, Any], comparison: dict[str, Any]) -> dict[str, float]:
    left = reference["canonical_logits"].float()
    right = comparison["canonical_logits"].float()
    probabilities = reference["probabilities"].float().clamp_min(1e-12)
    other = comparison["probabilities"].float().clamp_min(1e-12)
    return {
        "cross_entropy_change": comparison["cross_entropy"] - reference["cross_entropy"],
        "top1_change": comparison["top1_accuracy"] - reference["top1_accuracy"],
        "top5_change": comparison["top5_accuracy"] - reference["top5_accuracy"],
        "logit_rms_difference": float((left - right).square().mean().sqrt()),
        "kl_from_reference": float((probabilities * (probabilities.log() - other.log())).sum(dim=-1).mean()),
    }


def single_pair_distance_response(
    conditioner: torch.nn.Module,
    capacity: str,
    *,
    distance: float,
    perturbation: float,
) -> float:
    """Measure encoder response when exactly one abstract pair distance changes."""
    if perturbation == 0:
        raise ValueError("E007 Phase 4C.2 pair-distance perturbation must be nonzero")
    centers = conditioner.rbf_centers.detach()
    width = float(conditioner.maximum_distance) / max(int(conditioner.rbf_bins) - 1, 1)

    def encoded(value: float) -> torch.Tensor:
        rbf = torch.exp(-((centers.new_tensor(value) - centers) / width).square())
        if capacity == "small":
            features = rbf
        elif capacity == "medium":
            separation = conditioner.separation_embedding.weight.detach()[1]
            features = torch.cat((rbf, separation, centers.new_ones(1)))
        else:
            raise ValueError(f"E007 Phase 4C.2 unsupported capacity: {capacity}")
        return conditioner.pair_encoder(features)

    return float((encoded(distance + perturbation) - encoded(distance)).detach().float().square().mean().sqrt())


def paired_bootstrap(values: list[float], *, seed: int, replicates: int) -> dict[str, float]:
    if not values or replicates < 100:
        raise ValueError("E007 Phase 4C.2 bootstrap input is insufficient")
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.asarray(
        [generator.choice(array, size=len(array), replace=True).mean() for _ in range(replicates)],
        dtype=np.float64,
    )
    return {
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def summarize_gate_sweep(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["capacity"], float(record["gate"]))].append(record)
    output = {}
    base_seed = int(config["gate_sweep"]["bootstrap_seed"])
    replicates = int(config["gate_sweep"]["bootstrap_replicates"])
    for group_index, ((capacity, gate), values) in enumerate(sorted(grouped.items())):
        key = f"{capacity}/gate={gate:g}"
        output[key] = {
            comparison: paired_bootstrap(
                [float(row[comparison]["cross_entropy_change"]) for row in values],
                seed=base_seed + group_index * 100 + comparison_index,
                replicates=replicates,
            )
            for comparison_index, comparison in enumerate(
                ("correct_vs_gate_zero", "correct_vs_shuffled", "correct_vs_null")
            )
        }
        output[key]["by_length"] = {
            str(length): {
                comparison: float(
                    np.mean([row[comparison]["cross_entropy_change"] for row in values if int(row["length"]) == length])
                )
                for comparison in ("correct_vs_gate_zero", "correct_vs_shuffled", "correct_vs_null")
            }
            for length in sorted({int(row["length"]) for row in values})
        }
    return output


def classify_diagnostic(evidence: dict[str, Any]) -> dict[str, Any]:
    sensitivity = evidence.get("minimum_downstream_logit_rms_response")
    representation = evidence.get("minimum_correct_shuffled_representation_rms")
    learned_effect = evidence.get("learned_gate_correct_shuffled_effect")
    open_effect = evidence.get("less_suppressive_correct_shuffled_effect")
    overfit = evidence.get("bounded_correct_arm_overfit_advantage")
    if sensitivity is not None and sensitivity <= 0:
        classification = "conditioning_path_defect"
    elif representation is not None and representation <= 0:
        classification = "representation_not_discriminative"
    elif (
        learned_effect is not None
        and open_effect is not None
        and abs(open_effect) > abs(learned_effect)
        and abs(open_effect) > 0
    ):
        classification = "gate_suppression_supported"
    elif overfit is not None and overfit > 0:
        classification = "optimization_insufficient"
    elif sensitivity is not None and sensitivity > 0 and overfit is not None and overfit <= 0:
        classification = "architecture_not_using_matching_geometry"
    else:
        classification = "bounded_diagnostic_inconclusive"
    return {
        "classification": classification,
        "protein_geometry_declared_uninformative": False,
        "scientific_review_required": True,
        "scalar_composite_score_used": False,
    }


def load_conditioner_state(
    conditioner: torch.nn.Module,
    record: dict[str, str],
    *,
    capacity: str,
    arm: str,
    update: int,
    phase4c1_configuration_sha256: str,
) -> dict[str, Any]:
    checkpoint = torch.load(record["path"], map_location="cpu", weights_only=False)
    required = {
        "version": phase4c1.VERSION,
        "configuration_sha256": phase4c1_configuration_sha256,
        "capacity": capacity,
        "arm": arm,
        "update": update,
    }
    contradictions = [name for name, value in required.items() if checkpoint.get(name) != value]
    if contradictions:
        raise ValueError(f"E007 Phase 4C.2 conditioner checkpoint contradiction: {contradictions}")
    conditioner.load_state_dict(checkpoint["conditioner"], strict=True)
    return checkpoint


def _memory(device: torch.device) -> dict[str, float | None]:
    current = 0.0
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            current = float(line.split()[1]) / 1024
            break
    return {
        "current_rss_mib": current,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        ),
        "peak_cuda_reserved_mib": (torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None),
    }


def _strip_tensors(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha256(value)}
    if isinstance(value, dict):
        return {key: _strip_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_tensors(item) for item in value]
    return value


@torch.no_grad()
def _zero_update_diagnostics(
    config: dict[str, Any],
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if execution is None:
        execution = {}
    base = phase4c1._load_config(config["prerequisites"]["phase4c1_config"]["path"])
    seed = int(config["seed"])
    panel = phase4c1.select_panel(
        base,
        split="validation",
        count=int(config["representation"]["validation_panel_size"]),
        seed=seed,
    )
    execution["dataset_scanned"] = True
    donors = phase4c1.exact_length_derangement(panel, seed=seed + 1)
    by_id = {row["sample_id"]: row for row in panel}
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    prior, tokenizer, loader = load_reviewed_candidate("progen2_151m", config["artifact_cache_root"], device=device)
    execution["model_created"] = True
    prior.eval()
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    prior_hash = parameter_identity_hash(prior)
    records = []
    representation_comparisons = []
    gate_records = []
    sensitivity_records = []
    for capacity in CAPACITIES:
        for update in config["representation"]["updates"]:
            conditioners = {}
            for arm in ARMS:
                conditioner = phase4c1.initialize_conditioner(capacity, base, device=device).eval()
                checkpoint_record = config["conditioner_checkpoints"][capacity][arm][f"update_{update}"]
                load_conditioner_state(
                    conditioner,
                    checkpoint_record,
                    capacity=capacity,
                    arm=arm,
                    update=int(update),
                    phase4c1_configuration_sha256=config["prerequisites"]["phase4c1_config"]["sha256"],
                )
                conditioners[arm] = conditioner
            for row in panel:
                outputs = {}
                for arm in ARMS:
                    geometry = by_id[donors[row["sample_id"]]] if arm == "shuffled_geometry" else row
                    result = diagnostic_forward(
                        prior,
                        tokenizer,
                        conditioners[arm],
                        capacity,
                        row,
                        geometry_row=geometry,
                        arm=arm,
                        effective_gate=None,
                        device=device,
                    )
                    execution["forward_performed"] = True
                    records.append(
                        {
                            "capacity": capacity,
                            "arm": arm,
                            "update": update,
                            "sample_id": row["sample_id"],
                            "length": len(row["sequence"]),
                            **_strip_tensors(result),
                        }
                    )
                    outputs[arm] = result
                correct = outputs["correct_geometry"]
                for comparison_arm in ("shuffled_geometry", "null_geometry"):
                    comparison = outputs[comparison_arm]
                    shared_names = sorted(set(correct["representations"]) & set(comparison["representations"]))
                    differences = {}
                    for name in shared_names:
                        left = correct["representations"][name]
                        right = comparison["representations"][name]
                        mask = (
                            row["residue_mask"][None].to(device)
                            if left.ndim == 3 and left.shape[1] == len(row["sequence"])
                            else None
                        )
                        differences[name] = representation_difference(left, right, mask)
                    representation_comparisons.append(
                        {
                            "capacity": capacity,
                            "update": update,
                            "sample_id": row["sample_id"],
                            "comparison": f"correct_vs_{comparison_arm.removesuffix('_geometry')}",
                            "differences": differences,
                        }
                    )
                del outputs
            del conditioners

    for capacity in CAPACITIES:
        conditioner = phase4c1.initialize_conditioner(capacity, base, device=device).eval()
        load_conditioner_state(
            conditioner,
            config["conditioner_checkpoints"][capacity]["correct_geometry"]["update_1000"],
            capacity=capacity,
            arm="correct_geometry",
            update=1000,
            phase4c1_configuration_sha256=config["prerequisites"]["phase4c1_config"]["sha256"],
        )
        for row in panel:
            conditions = {
                "correct_geometry": row,
                "shuffled_geometry": by_id[donors[row["sample_id"]]],
                "null_geometry": row,
            }
            baseline = None
            for gate in config["gate_sweep"]["effective_gate_values"]:
                outputs = {
                    arm: diagnostic_forward(
                        prior,
                        tokenizer,
                        conditioner,
                        capacity,
                        row,
                        geometry_row=geometry,
                        arm=arm,
                        effective_gate=float(gate),
                        device=device,
                    )
                    for arm, geometry in conditions.items()
                }
                if float(gate) == 0:
                    baseline = outputs["correct_geometry"]
                if baseline is None:
                    raise RuntimeError("E007 Phase 4C.2 gate-zero baseline is missing")
                gate_records.append(
                    {
                        "capacity": capacity,
                        "sample_id": row["sample_id"],
                        "length": len(row["sequence"]),
                        "gate": float(gate),
                        "correct": _strip_tensors(outputs["correct_geometry"]),
                        "correct_vs_gate_zero": paired_logit_effect(baseline, outputs["correct_geometry"]),
                        "correct_vs_shuffled": paired_logit_effect(
                            outputs["correct_geometry"], outputs["shuffled_geometry"]
                        ),
                        "correct_vs_null": paired_logit_effect(outputs["correct_geometry"], outputs["null_geometry"]),
                    }
                )
            perturbed = dict(row)
            perturbed["coordinates"] = row["coordinates"].clone()
            valid = torch.nonzero(row["residue_mask"], as_tuple=False).flatten()
            if len(valid) < 2:
                raise ValueError("E007 Phase 4C.2 sensitivity sample has fewer than two valid residues")
            perturbed["coordinates"][valid[1], 0] += float(config["geometry_sensitivity"]["coordinate_perturbation"])
            original = diagnostic_forward(
                prior,
                tokenizer,
                conditioner,
                capacity,
                row,
                geometry_row=row,
                arm="correct_geometry",
                effective_gate=1.0,
                device=device,
            )
            changed = diagnostic_forward(
                prior,
                tokenizer,
                conditioner,
                capacity,
                row,
                geometry_row=perturbed,
                arm="correct_geometry",
                effective_gate=1.0,
                device=device,
            )
            o3 = o3_invariance_diagnostic(
                prior,
                tokenizer,
                conditioner,
                capacity,
                row,
                reference_output=original,
                device=device,
                absolute_tolerance=float(config["geometry_sensitivity"]["o3_atol"]),
                relative_tolerance=float(config["geometry_sensitivity"]["o3_rtol"]),
            )
            execution["forward_performed"] = True
            sensitivity_records.append(
                {
                    "capacity": capacity,
                    "sample_id": row["sample_id"],
                    "perturbed_logit_rms_response": paired_logit_effect(original, changed)["logit_rms_difference"],
                    "single_pair_encoder_rms_response": single_pair_distance_response(
                        conditioner,
                        capacity,
                        distance=float(
                            torch.linalg.vector_norm(row["coordinates"][valid[0]] - row["coordinates"][valid[1]])
                        ),
                        perturbation=float(config["geometry_sensitivity"]["pair_distance_perturbation"]),
                    ),
                    "o3_logit_rms_error": o3["canonical_logits"]["rms_error"],
                    "o3": o3,
                    "injection_depth_responses": [row["injected_residual_rms"] for row in changed["injections"]],
                }
            )
    minimum_logit_response = float(config["geometry_sensitivity"]["minimum_logit_rms_response"])
    if any(row["single_pair_encoder_rms_response"] <= 0 for row in sensitivity_records):
        raise ValueError("E007 Phase 4C.2 single-pair encoder sensitivity failed")
    if any(row["perturbed_logit_rms_response"] <= minimum_logit_response for row in sensitivity_records):
        raise ValueError("E007 Phase 4C.2 downstream geometry sensitivity failed")
    if any(not all(value > 0 for value in row["injection_depth_responses"]) for row in sensitivity_records):
        raise ValueError("E007 Phase 4C.2 geometry response did not reach every injection depth")
    failures = [
        {
            "capacity": row["capacity"],
            "sample_id": row["sample_id"],
            "o3": row["o3"],
        }
        for row in sensitivity_records
        if row["o3"]["status"] != "passed"
    ]
    if failures:
        raise ValueError(f"E007 Phase 4C.2 O(3) invariance failed: {failures[:3]}")
    minimum_representation_difference = min(
        record["differences"]["biological_conditioning"]["rms_difference"]
        for record in representation_comparisons
        if record["comparison"] == "correct_vs_shuffled"
    )
    if minimum_representation_difference <= 0:
        raise ValueError("E007 Phase 4C.2 correct and shuffled conditioning representations are identical")
    phase4c1.enforce_memory_limits(_memory(device), config["memory"])
    if parameter_identity_hash(prior) != prior_hash:
        raise ValueError("E007 Phase 4C.2 frozen ProGen2 parameters mutated")
    return {
        "status": "completed_non_authorizing",
        "panel_sample_ids": [row["sample_id"] for row in panel],
        "panel_sha256": canonical_sha256([row["sample_id"] for row in panel]),
        "donor_mapping": donors,
        "donor_mapping_sha256": canonical_sha256(donors),
        "representation_records": records,
        "representation_comparisons": representation_comparisons,
        "minimum_correct_shuffled_representation_rms": minimum_representation_difference,
        "gate_sweep_records": gate_records,
        "gate_sweep_summary": summarize_gate_sweep(gate_records, config),
        "sensitivity_records": sensitivity_records,
        "prior_parameter_sha256_before_after": prior_hash,
        "loader": loader,
        "memory": _memory(device),
        "optimizer_created": False,
        "optimizer_updates": 0,
        "diagnostic_optimization_performed": False,
        "production_training_performed": False,
        **NON_AUTHORIZING,
    }


def _optimization_worker(config_path: str, capacity: str, arm: str, policy: str, result_path: str) -> None:
    execution = {
        "model_created": False,
        "forward_performed": False,
        "backward_performed": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
    }
    stage = "startup"
    try:
        config = _load_config(config_path)
        base = phase4c1._load_config(config["prerequisites"]["phase4c1_config"]["path"])
        settings = config["optimization_diagnostic"]
        seed = int(config["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        stage = "panel_selection"
        train = phase4c1.select_panel(base, split="train", count=int(settings["train_panel_size"]), seed=seed + 10)
        validation = phase4c1.select_panel(
            base, split="validation", count=int(settings["validation_panel_size"]), seed=seed + 20
        )
        donors_train = phase4c1.exact_length_derangement(train, seed=seed + 30)
        donors_validation = phase4c1.exact_length_derangement(validation, seed=seed + 40)
        train_by_id = {row["sample_id"]: row for row in train}
        stage = "model_construction"
        device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        prior, tokenizer, _loader = load_reviewed_candidate(
            "progen2_151m", config["artifact_cache_root"], device=device
        )
        prior.eval()
        for parameter in prior.parameters():
            parameter.requires_grad_(False)
        prior_hash = parameter_identity_hash(prior)
        conditioner = phase4c1.initialize_conditioner(capacity, base, device=device)
        load_conditioner_state(
            conditioner,
            config["conditioner_checkpoints"][capacity][arm]["update_1000"],
            capacity=capacity,
            arm=arm,
            update=1000,
            phase4c1_configuration_sha256=config["prerequisites"]["phase4c1_config"]["sha256"],
        )
        if policy == "gate_logit_zero":
            with torch.no_grad():
                if capacity == "small":
                    conditioner.gate_logit.zero_()
                else:
                    conditioner.injection_gate_logits.zero_()
        parameter_count = trainable_parameter_count(conditioner)
        gate_parameters = [conditioner.gate_logit] if capacity == "small" else [conditioner.injection_gate_logits]
        gate_ids = {id(value) for value in gate_parameters}
        other_parameters = [value for value in conditioner.parameters() if id(value) not in gate_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": other_parameters, "lr": float(settings["learning_rate"])},
                {"params": gate_parameters, "lr": float(settings["gate_learning_rate"])},
            ],
            weight_decay=float(settings["weight_decay"]),
        )
        execution["model_created"] = execution["optimizer_created"] = True
        evaluations = {}
        cursor = 0
        for update in range(int(settings["maximum_updates"]) + 1):
            if update in settings["evaluation_updates"]:
                conditioner.eval()
                evaluations[str(update)] = phase4c1._evaluate(
                    prior,
                    tokenizer,
                    conditioner,
                    capacity,
                    arm,
                    validation,
                    donors_validation,
                    base,
                    device,
                )
                if update == 0:
                    replay = phase4c1._evaluate(
                        prior,
                        tokenizer,
                        conditioner,
                        capacity,
                        arm,
                        validation,
                        donors_validation,
                        base,
                        device,
                    )
                    if canonical_sha256(evaluations[str(update)]) != canonical_sha256(replay):
                        raise ValueError("E007 Phase 4C.2 deterministic evaluation replay failed")
            if update == int(settings["maximum_updates"]):
                break
            row = train[cursor % len(train)]
            geometry = train_by_id[donors_train[row["sample_id"]]] if arm == "shuffled_geometry" else row
            optimizer.zero_grad(set_to_none=True)
            conditioner.train()
            loss, _ = phase4c1.conditioned_progen_loss(
                prior,
                tokenizer,
                conditioner,
                capacity,
                row,
                geometry_row=geometry,
                arm=arm,
                device=device,
            )
            execution["forward_performed"] = True
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("E007 Phase 4C.2 optimization loss is non-finite")
            loss.backward()
            execution["backward_performed"] = True
            coverage = phase4c1.gradient_coverage(conditioner, capacity)
            if not all(coverage.values()):
                raise FloatingPointError(f"E007 Phase 4C.2 gradient coverage failed: {coverage}")
            torch.nn.utils.clip_grad_norm_(conditioner.parameters(), float(settings["gradient_clip_norm"]))
            optimizer.step()
            cursor += 1
            execution["optimizer_updates"] = update + 1
        if parameter_identity_hash(prior) != prior_hash:
            raise ValueError("E007 Phase 4C.2 frozen ProGen2 parameters mutated")
        phase4c1.enforce_memory_limits(_memory(device), config["memory"])
        atomic_json(
            Path(result_path),
            {
                "status": "completed_non_authorizing",
                "capacity": capacity,
                "arm": arm,
                "policy": policy,
                "parameter_count": parameter_count,
                "evaluations": evaluations,
                "panel_hashes": {
                    "train": canonical_sha256([row["sample_id"] for row in train]),
                    "validation": canonical_sha256([row["sample_id"] for row in validation]),
                },
                "donor_hashes": {
                    "train": canonical_sha256(donors_train),
                    "validation": canonical_sha256(donors_validation),
                },
                "deterministic_inputs": True,
                "deterministic_replay_verified": True,
                "diagnostic_optimization_performed": True,
                "production_training_performed": False,
                "prior_parameter_sha256_before_after": prior_hash,
                "execution": execution,
                "memory": _memory(device),
                **NON_AUTHORIZING,
            },
        )
    except BaseException as error:
        atomic_json(
            Path(result_path),
            {
                "status": "failed",
                "capacity": capacity,
                "arm": arm,
                "policy": policy,
                "failure_stage": stage,
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
                "execution": execution,
                **NON_AUTHORIZING,
            },
        )


def _artifact_inventory(root: Path, paths: list[Path]) -> dict[str, Any]:
    records = []
    for path in sorted(paths):
        records.append(
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"artifacts": records, "aggregate_sha256": canonical_sha256(records)}


def _summarize(zero: dict[str, Any], optimization: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    sensitivity = [row["perturbed_logit_rms_response"] for row in zero["sensitivity_records"]]
    gate_summary = zero["gate_sweep_summary"]
    learned_effect = gate_summary["medium/gate=0.018"]["correct_vs_shuffled"]["mean"]
    open_effect = gate_summary["medium/gate=0.5"]["correct_vs_shuffled"]["mean"]
    final_update = str(config["optimization_diagnostic"]["maximum_updates"])
    by_identity = {(row["capacity"], row["policy"], row["arm"]): row for row in optimization}
    overfit_advantages = []
    optimization_effects = {}
    bootstrap_seed = int(config["gate_sweep"]["bootstrap_seed"]) + 10000
    replicates = int(config["gate_sweep"]["bootstrap_replicates"])
    for capacity in CAPACITIES:
        for policy in POLICIES:
            evaluations = {arm: by_identity[(capacity, policy, arm)]["evaluations"][final_update] for arm in ARMS}
            correct_by_id = {row["sample_id"]: row for row in evaluations["correct_geometry"]["records"]}
            control_effects = {}
            for control_index, arm in enumerate(("shuffled_geometry", "null_geometry")):
                control_by_id = {row["sample_id"]: row for row in evaluations[arm]["records"]}
                if set(correct_by_id) != set(control_by_id):
                    raise ValueError("E007 Phase 4C.2 optimization evaluation panels are not paired")
                differences = [
                    float(control_by_id[sample_id]["cross_entropy"] - correct_by_id[sample_id]["cross_entropy"])
                    for sample_id in sorted(correct_by_id)
                ]
                control_effects[f"correct_vs_{arm.removesuffix('_geometry')}"] = paired_bootstrap(
                    differences,
                    seed=bootstrap_seed + len(optimization_effects) * 10 + control_index,
                    replicates=replicates,
                )
            optimization_effects[f"{capacity}/{policy}"] = control_effects
            if policy == "gate_logit_zero":
                overfit_advantages.append(min(value["mean"] for value in control_effects.values()))
    evidence = {
        "minimum_downstream_logit_rms_response": min(sensitivity),
        "minimum_correct_shuffled_representation_rms": zero["minimum_correct_shuffled_representation_rms"],
        "learned_gate_correct_shuffled_effect": learned_effect,
        "less_suppressive_correct_shuffled_effect": open_effect,
        "bounded_correct_arm_overfit_advantage": min(overfit_advantages),
    }
    return {
        "decision": classify_diagnostic(evidence),
        "decision_evidence": evidence,
        "optimization_case_count": len(optimization),
        "optimization_paired_effects": optimization_effects,
        "effect_sign_convention": "positive control-minus-correct cross-entropy means matching geometry is better",
        "paired_bootstrap_policy": {
            "replicates": config["gate_sweep"]["bootstrap_replicates"],
            "seed": config["gate_sweep"]["bootstrap_seed"],
            "scalar_composite_score_used": False,
        },
    }


def run(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    protected_before = verify_prerequisites(config)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase 4C.2 output already exists: {output}")
    execution = {
        "dataset_scanned": False,
        "model_created": False,
        "forward_performed": False,
        "backward_performed": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "checkpoint_written": False,
        "metric_published": False,
        "trajectory_published": False,
    }
    staging.mkdir(parents=True)
    atomic_json(
        staging / "heartbeat.json",
        {"status": "running", "stage": "zero_update", "execution": execution, **NON_AUTHORIZING},
    )
    try:
        zero = _zero_update_diagnostics(config, execution)
        atomic_json(staging / "zero_update_diagnostics.json", zero)
        execution["metric_published"] = True
        results = []
        context = mp.get_context("spawn")
        for capacity in CAPACITIES:
            for arm in ARMS:
                for policy in POLICIES:
                    identity = f"{capacity}__{arm}__{policy}"
                    result_path = staging / "optimization" / f"{identity}.json"
                    process = context.Process(
                        target=_optimization_worker,
                        args=(str(config_path), capacity, arm, policy, str(result_path)),
                    )
                    process.start()
                    process.join()
                    if process.exitcode != 0 or not result_path.is_file():
                        raise RuntimeError(f"E007 Phase 4C.2 isolated optimization worker failed: {identity}")
                    result = json.loads(result_path.read_text())
                    if result.get("status") != "completed_non_authorizing":
                        raise RuntimeError(f"E007 Phase 4C.2 optimization case failed: {identity}: {result}")
                    results.append(result)
                    worker_execution = result["execution"]
                    execution["model_created"] = execution["model_created"] or worker_execution["model_created"]
                    execution["forward_performed"] = (
                        execution["forward_performed"] or worker_execution["forward_performed"]
                    )
                    execution["backward_performed"] = (
                        execution["backward_performed"] or worker_execution["backward_performed"]
                    )
                    execution["optimizer_created"] = (
                        execution["optimizer_created"] or worker_execution["optimizer_created"]
                    )
                    execution["optimizer_updates"] += int(worker_execution["optimizer_updates"])
                    atomic_json(
                        staging / "heartbeat.json",
                        {
                            "status": "running",
                            "stage": "bounded_optimization",
                            "completed_cases": len(results),
                            "total_cases": 12,
                            "execution": execution,
                            **NON_AUTHORIZING,
                        },
                    )
        panel_hashes = {canonical_sha256(row["panel_hashes"]) for row in results}
        donor_hashes = {canonical_sha256(row["donor_hashes"]) for row in results}
        if len(panel_hashes) != 1 or len(donor_hashes) != 1:
            raise ValueError("E007 Phase 4C.2 paired optimization controls diverged")
        for capacity in CAPACITIES:
            counts = {row["parameter_count"] for row in results if row["capacity"] == capacity}
            if len(counts) != 1:
                raise ValueError(f"E007 Phase 4C.2 {capacity} control parameter budgets differ")
        if verify_prerequisites(config) != protected_before:
            raise ValueError("E007 Phase 4C.2 protected prerequisites changed")
        summary = _summarize(zero, results, config)
        report = {
            "status": "completed_non_authorizing",
            "version": VERSION,
            "injection_contract": injection_contract(),
            "zero_update_diagnostics_path": "zero_update_diagnostics.json",
            "optimization_results": results,
            **summary,
            "protected_inputs_unchanged": True,
            "scientific_conclusion_scope": (
                "conditioning-path diagnostic only; protein geometry is not declared uninformative"
            ),
            "diagnostic_optimization_performed": True,
            "production_training_performed": False,
            "execution": {
                "model_created": True,
                "forward_performed": True,
                "backward_performed": True,
                "optimizer_created": True,
                "optimizer_updates_per_arm": int(config["optimization_diagnostic"]["maximum_updates"]),
                "optimizer_arm_count": len(results),
                "total_diagnostic_optimizer_updates": sum(
                    int(row["execution"]["optimizer_updates"]) for row in results
                ),
            },
            **NON_AUTHORIZING,
        }
        atomic_json(staging / "report.json", report)
        durable = [staging / "zero_update_diagnostics.json", *sorted((staging / "optimization").glob("*.json"))]
        inventory = _artifact_inventory(staging, durable)
        atomic_json(staging / "artifact_inventory.json", inventory)
        atomic_json(
            staging / "protocol.json",
            {
                "status": "completed_non_authorizing",
                "version": VERSION,
                "configuration_sha256": sha256_file(config_path),
                "report_sha256": sha256_file(staging / "report.json"),
                "artifact_inventory_sha256": sha256_file(staging / "artifact_inventory.json"),
                "protected_hashes": protected_before,
                "execution": {
                    "model_created": True,
                    "cuda_initialized": config["device"] == "cuda",
                    "optimizer_created": True,
                    "maximum_updates_per_arm": config["optimization_diagnostic"]["maximum_updates"],
                    "production_training_performed": False,
                    "diagnostic_optimization_performed": True,
                },
                **NON_AUTHORIZING,
            },
        )
        atomic_json(
            staging / "heartbeat.json",
            {"status": "completed", "completed_utc": datetime.now(UTC).isoformat(), **NON_AUTHORIZING},
        )
        os.replace(staging, output)
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return {"status": "completed_non_authorizing", "output_dir": str(output), **NON_AUTHORIZING}
    except BaseException as error:
        atomic_json(
            staging / "heartbeat.json",
            {
                "status": "failed",
                "completed_utc": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
                "execution": execution,
                **NON_AUTHORIZING,
            },
        )
        raise


def monitor(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    root = output if output.is_dir() else staging
    heartbeat = root / "heartbeat.json"
    if not heartbeat.is_file():
        return {"status": "not_started", "path": str(root)}
    return json.loads(heartbeat.read_text())
