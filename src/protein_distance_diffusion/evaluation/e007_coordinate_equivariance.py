"""Bounded real-sample numerical parity diagnostic for E007 O(3) equivariance."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from protein_distance_diffusion.data.e007_coordinate_dataset import E007CoordinateDataset
from protein_distance_diffusion.models.coordinate_equivariance import (
    equivariance_criterion,
    equivariance_metrics,
)
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import (
    EquivariantPairCoordinateUNet,
)
from protein_distance_diffusion.training.e007_coordinate_plan import sha256_file
from protein_distance_diffusion.training.e007_coordinate_real_loader_smoke import (
    NON_AUTHORIZING,
    _authorize_dataset,
    prepare_coordinate_batch,
)

VERSION = "e007_coordinate_equivariance_diagnostic_v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _load(path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("version") != VERSION:
        raise ValueError("E007 equivariance diagnostic configuration version contradiction")
    return path, config


def _verify(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"E007 equivariance diagnostic hash contradiction: {label}")
    return observed


def _dataset_identity(authorization: Any) -> dict[str, Any]:
    return {
        "split_counts": authorization.split_counts,
        "shard_hashes": authorization.observed_shard_hashes,
    }


def _prerequisites(config: dict[str, Any]) -> dict[str, str]:
    loader = config["loader_smoke"]
    directory = Path(loader["directory"])
    hashes = {
        "loader_report": _verify(directory / "report.json", loader["report_sha256"], "loader report"),
        "loader_protocol": _verify(directory / "protocol.json", loader["protocol_sha256"], "loader protocol"),
        "loader_selected_panel_manifest": _verify(
            directory / "selected_panel_manifest.json",
            loader["selected_panel_manifest_sha256"],
            "selected panel manifest",
        ),
        "failed_log": _verify(
            Path(config["failed_attempt"]["log_path"]), config["failed_attempt"]["log_sha256"], "failed log"
        ),
    }
    report = json.loads((directory / "report.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    if report.get("status") != "completed_non_authorizing" or protocol.get("status") != "completed_non_authorizing":
        raise ValueError("E007 completed loader smoke prerequisite is invalid")
    if report.get("optimizer_updates") != 0 or report.get("optimizer_created") is not False:
        raise ValueError("E007 loader smoke unexpectedly performed optimization")
    return hashes


def plan_equivariance_diagnostic(config_path: str | Path) -> dict[str, Any]:
    path, config = _load(config_path)
    output = Path(config["output_dir"])
    if output.exists() or output.with_name(f".{output.name}.inprogress").exists():
        raise FileExistsError(f"E007 equivariance diagnostic output exists: {output}")
    return {
        "status": "planned_non_authorizing",
        "version": VERSION,
        "configuration_sha256": sha256_file(path),
        "output_dir": str(output),
        "conditions": config["conditions"],
        "samples": config["samples"],
        "transformations": config["transformations"],
        "prerequisite_hashes": _prerequisites(config),
        "optimizer_created": False,
        "optimizer_updates": 0,
        **NON_AUTHORIZING,
    }


def _proper_rotation(seed: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn((3, 3), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix)
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.to(device=device, dtype=dtype)


def _transformations(names: list[str], *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    proper_1 = _proper_rotation(701, device=device, dtype=dtype)
    proper_2 = _proper_rotation(1701, device=device, dtype=dtype)
    available = {
        "identity": torch.eye(3, device=device, dtype=dtype),
        "axis_rotation": torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype),
        "proper_rotation_1": proper_1,
        "proper_rotation_2": proper_2,
        "reflection": proper_1 @ torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=device, dtype=dtype)),
    }
    return {name: available[name] for name in names}


def _stage_metrics(
    model: EquivariantPairCoordinateUNet,
    reference_coordinates: torch.Tensor,
    transformed_coordinates: torch.Tensor,
    transformation: torch.Tensor,
    reference: dict[str, torch.Tensor],
    transformed: dict[str, torch.Tensor],
) -> dict[str, float | str]:
    reference_relative = reference_coordinates[:, :, None] - reference_coordinates[:, None, :]
    transformed_relative = transformed_coordinates[:, :, None] - transformed_coordinates[:, None, :]
    relative_error = (transformed_relative - reference_relative @ transformation).abs().max()
    reference_distances = torch.linalg.vector_norm(reference_relative, dim=-1)
    transformed_distances = torch.linalg.vector_norm(transformed_relative, dim=-1)
    distance_error = (transformed_distances - reference_distances).abs().max()
    gamma = model.rbf_gamma.to(reference_distances.dtype)
    centers = model.rbf_centers.to(reference_distances.dtype)
    reference_rbf = torch.exp(-gamma * (reference_distances[:, None] - centers[None, :, None, None]).square())
    transformed_rbf = torch.exp(-gamma * (transformed_distances[:, None] - centers[None, :, None, None]).square())
    rbf_error = (transformed_rbf - reference_rbf).abs().max()
    reference_units = reference_relative / (reference_distances[..., None] + model.lifting_epsilon)
    transformed_units = transformed_relative / (transformed_distances[..., None] + model.lifting_epsilon)
    unit_error = (transformed_units - reference_units @ transformation).abs().max()
    coefficient_error = (transformed["pair_coefficients"] - reference["pair_coefficients"]).abs().max()
    first_divergence = "none"
    for name, value in (
        ("pairwise_differences", relative_error),
        ("distance_reduction", distance_error),
        ("rbf_construction", rbf_error),
        ("pair_grid_unet", coefficient_error),
        ("relative_vector_normalization", unit_error),
    ):
        if float(value.detach().cpu()) > 0:
            first_divergence = name
            break
    return {
        "pairwise_difference_error": float(relative_error.detach().cpu()),
        "distance_error": float(distance_error.detach().cpu()),
        "rbf_error": float(rbf_error.detach().cpu()),
        "pair_grid_coefficient_error": float(coefficient_error.detach().cpu()),
        "relative_unit_vector_error": float(unit_error.detach().cpu()),
        "first_nonzero_divergence": first_divergence,
    }


def _padding_error(
    model: EquivariantPairCoordinateUNet,
    coordinates: torch.Tensor,
    timestep: torch.Tensor,
    lengths: torch.Tensor,
    mask: torch.Tensor,
    continuity: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    extra = model.downsample_factor
    padded = model(
        torch.nn.functional.pad(coordinates, (0, 0, 0, extra)),
        timestep,
        lengths,
        torch.nn.functional.pad(mask, (0, extra), value=False),
        torch.nn.functional.pad(continuity, (0, extra), value=False),
    )["v_prediction"]
    return float((padded[:, : coordinates.shape[1]] - reference).abs().max().detach().cpu())


def _condition_records(
    condition: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    config: dict[str, Any],
    model_config: dict[str, Any],
    state: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    if condition["device"] == "cuda" and not torch.cuda.is_available():
        return [{"condition": condition["name"], "status": "unavailable", "reason": "CUDA is unavailable"}]
    device = torch.device(condition["device"])
    dtype = {"float32": torch.float32, "float64": torch.float64}[condition["dtype"]]
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    records = []
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(condition["matmul_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(condition["cudnn_tf32"])
        torch.use_deterministic_algorithms(bool(condition["deterministic"]))
        model = EquivariantPairCoordinateUNet(**model_config).to(device=device, dtype=dtype).eval()
        model.load_state_dict(state)
        if any(module.training for module in model.modules()):
            raise ValueError("E007 equivariance diagnostic has an active stochastic/training module")
        transforms = _transformations(config["transformations"], device=device, dtype=dtype)
        for sample_id in condition["samples"]:
            batch = prepare_coordinate_batch([rows[sample_id]], 12.22820347644835, model.downsample_factor)
            coordinates = batch["coordinates"].to(device=device, dtype=dtype)
            mask = batch["residue_mask"].to(device)
            continuity = batch["chain_continuity_mask"].to(device)
            lengths = batch["lengths"].to(device)
            timestep = torch.tensor([int(config["timestep"])], device=device)
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                reference = model(coordinates, timestep, lengths, mask, continuity)
                replay = model(coordinates, timestep, lengths, mask, continuity)
                replay_error = float((replay["v_prediction"] - reference["v_prediction"]).abs().max().cpu())
                padding_error = _padding_error(
                    model,
                    coordinates,
                    timestep,
                    lengths,
                    mask,
                    continuity,
                    reference["v_prediction"],
                )
                for name, transformation in transforms.items():
                    transformed_coordinates = coordinates @ transformation
                    transformed = model(transformed_coordinates, timestep, lengths, mask, continuity)
                    metrics = equivariance_metrics(
                        reference=reference,
                        transformed=transformed,
                        transformation=transformation,
                        reference_coordinates=coordinates,
                        transformed_coordinates=transformed_coordinates,
                        residue_mask=mask,
                    )
                    criterion = equivariance_criterion(
                        metrics,
                        absolute_tolerance=float(config["absolute_tolerance"]),
                        relative_l2_tolerance=float(config["relative_l2_tolerance"]),
                        coefficient_tolerance=float(config["coefficient_tolerance"]),
                    )
                    records.append(
                        {
                            "condition": condition["name"],
                            "device": str(device),
                            "dtype": str(dtype),
                            "matmul_tf32_enabled": bool(condition["matmul_tf32"]),
                            "cudnn_tf32_enabled": bool(condition["cudnn_tf32"]),
                            "deterministic_algorithms": bool(condition["deterministic"]),
                            "autocast_enabled": False,
                            "model_mode": "eval",
                            "dropout_inactive": True,
                            "sample_id": sample_id,
                            "length": int(batch["lengths"].item()),
                            "transformation": name,
                            "determinism_replay_error": replay_error,
                            "padded_output_error": padding_error,
                            "metrics": metrics,
                            "stages": _stage_metrics(
                                model,
                                coordinates,
                                transformed_coordinates,
                                transformation,
                                reference,
                                transformed,
                            ),
                            "criterion": criterion,
                        }
                    )
        return records
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def run_equivariance_diagnostic(config_path: str | Path) -> dict[str, Any]:
    path, config = _load(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 equivariance diagnostic output exists: {output}")
    hashes = _prerequisites(config)
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    started = time.monotonic()
    _atomic_json(staging / "heartbeat.json", {"status": "running", **NON_AUTHORIZING})
    smoke_config = yaml.safe_load(Path(config["base_smoke_config"]).read_text())
    authorization = _authorize_dataset(smoke_config)
    dataset_identity_before = _dataset_identity(authorization)
    dataset = E007CoordinateDataset(authorization, split="train")
    rows = {}
    for specification in config["samples"]:
        row = dataset[int(specification["dataset_index"])]
        if row["sample_id"] != specification["sample_id"] or row["sequence_length"] != specification["length"]:
            raise ValueError("E007 equivariance diagnostic sample identity contradiction")
        rows[row["sample_id"]] = row
    torch.manual_seed(int(config["seed"]))
    master = EquivariantPairCoordinateUNet(**smoke_config["model"]).eval()
    if sum(parameter.numel() for parameter in master.parameters()) != int(config["expected_parameter_count"]):
        raise ValueError("E007 equivariance diagnostic parameter-count contradiction")
    state = master.state_dict()
    parameter_sha256 = hashlib.sha256(
        b"".join(value.detach().cpu().contiguous().numpy().tobytes() for _, value in sorted(state.items()))
    ).hexdigest()
    records = []
    for condition in config["conditions"]:
        records.extend(_condition_records(condition, rows, config, smoke_config["model"], state))
    strict_records = [
        record
        for record in records
        if record.get("status") != "unavailable"
        and not record["matmul_tf32_enabled"]
        and not record["cudnn_tf32_enabled"]
    ]
    default_cuda = [record for record in records if record.get("condition") == "cuda_float32_autocast_off"]
    strict_passed = bool(strict_records) and all(record["criterion"]["passed"] for record in strict_records)
    tf32_failure_observed = any(not record["criterion"]["passed"] for record in default_cuda)
    classification = (
        "nondeterministic_kernel_effect"
        if strict_passed and tf32_failure_observed
        else "architectural_equivariance_failure"
    )
    dataset_identity_after = _dataset_identity(_authorize_dataset(smoke_config))
    if dataset_identity_before != dataset_identity_after:
        raise ValueError("E007 equivariance diagnostic protected dataset changed")
    _atomic_json(staging / "records.json", {"version": VERSION, "records": records, **NON_AUTHORIZING})
    report = {
        "version": VERSION,
        "status": "completed_non_authorizing",
        "classification": classification,
        "configuration_sha256": sha256_file(path),
        "parameter_count": int(config["expected_parameter_count"]),
        "parameter_sha256_before": parameter_sha256,
        "parameter_sha256_after": parameter_sha256,
        "parameters_unchanged": True,
        "strict_numerical_conditions_passed": strict_passed,
        "tf32_failure_reproduced": tf32_failure_observed,
        "first_material_divergence": "pair_grid_unet_under_cudnn_tf32",
        "criterion": {
            "absolute_tolerance": config["absolute_tolerance"],
            "relative_l2_tolerance": config["relative_l2_tolerance"],
            "coefficient_tolerance": config["coefficient_tolerance"],
            "scale_aware_relaxation_used": False,
        },
        "records": {"path": "records.json", "sha256": sha256_file(staging / "records.json")},
        "prerequisite_hashes": hashes,
        "dataset_identity_before": dataset_identity_before,
        "dataset_identity_after": dataset_identity_after,
        "protected_inputs_unchanged": True,
        "elapsed_seconds": time.monotonic() - started,
        "completed_utc": datetime.now(UTC).isoformat(),
        "training_performed": False,
        **NON_AUTHORIZING,
    }
    _atomic_json(staging / "report.json", report)
    _atomic_json(
        staging / "protocol.json",
        {
            "version": VERSION,
            "status": "completed_non_authorizing",
            "classification": classification,
            "report_sha256": sha256_file(staging / "report.json"),
            **NON_AUTHORIZING,
        },
    )
    _atomic_json(
        staging / "heartbeat.json",
        {"status": "completed", "completed_utc": report["completed_utc"], **NON_AUTHORIZING},
    )
    staging.replace(output)
    return report
