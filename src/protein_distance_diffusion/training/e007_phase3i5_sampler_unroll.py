"""Bounded, non-authorizing Phase 3I.5 sampler-unrolled feasibility study."""

from __future__ import annotations

import contextlib
import copy
import gc
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from protein_distance_diffusion.models.coordinate_equivariance import coordinate_model_execution_context
from protein_distance_diffusion.training.coordinate_diffusion import (
    CoordinateVPDiffusion,
    coordinates_to_distance_matrix,
)

VERSION = "e007_phase3i5_sampler_unroll_v1"
NON_AUTHORIZING = {
    "authorizes_training": False,
    "authorizes_real_data_training": False,
    "authorizes_phase3j": False,
    "authorizes_production": False,
    "authorizes_additional_training": False,
    "authorizes_joint_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_downstream_generation": False,
    "authorizes_sampler_correction": False,
    "authorizes_larger_sampler_aware_experiment": False,
}


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def masked_local_geometry(
    predicted_x0: torch.Tensor,
    target_x0: torch.Tensor,
    residue_mask: torch.Tensor,
    continuity_mask: torch.Tensor,
    coordinate_scale_angstrom: float,
) -> torch.Tensor:
    """Protected local geometry objective, evaluated only on x0 reconstructions."""
    distances = coordinates_to_distance_matrix(predicted_x0, residue_mask) * coordinate_scale_angstrom
    target = coordinates_to_distance_matrix(target_x0, residue_mask) * coordinate_scale_angstrom
    valid = continuity_mask.bool() & residue_mask[:, :-1].bool() & residue_mask[:, 1:].bool()
    if not bool(valid.any()):
        return predicted_x0.sum() * 0.0
    predicted_adjacent = distances[:, :-1, 1:].diagonal(dim1=1, dim2=2)
    target_adjacent = target[:, :-1, 1:].diagonal(dim1=1, dim2=2)
    return F.smooth_l1_loss(predicted_adjacent[valid].float(), target_adjacent[valid].float(), beta=0.25)


def sampler_unroll_objective(
    model: torch.nn.Module,
    diffusion: CoordinateVPDiffusion,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    lengths: torch.Tensor,
    residue_mask: torch.Tensor,
    continuity_mask: torch.Tensor,
    clean: torch.Tensor,
    coordinate_scale_angstrom: float,
    *,
    unroll_transitions: int = 2,
    geometry_weight: float = 0.02,
    activation_checkpointing: bool = False,
    phase_telemetry: list[dict[str, Any]] | None = None,
) -> dict[str, torch.Tensor]:
    """Build v MSE and local x0 geometry at immediate and post-transition states.

    The second state is produced by the exact production deterministic reverse
    transition. No state is detached, and geometry never sees noisy coordinates.
    """
    if unroll_transitions != 2:
        raise ValueError("Phase 3I.5 contract requires exactly two reverse transitions")
    state = noisy
    t = timesteps
    v_losses: list[torch.Tensor] = []
    geometries: list[torch.Tensor] = []
    phase_telemetry = phase_telemetry if phase_telemetry is not None else []
    for _ in range(unroll_transitions):
        phase_device = state.device
        with _tracked_memory_phase(f"unrolled_denoiser_forward_{_}", phase_device, phase_telemetry):
            if activation_checkpointing and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                def denoise(x: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
                    return model(x, step, lengths, residue_mask, continuity_mask)["v_prediction"]

                v = checkpoint(denoise, state, t, use_reentrant=False, preserve_rng_state=True)
            else:
                v = model(state, t, lengths, residue_mask, continuity_mask)["v_prediction"]
        alpha, sigma = diffusion.alpha_sigma(t, state)
        x0_hat = diffusion.reconstruct_x0(state, t, v, residue_mask)
        # Target v is reconstructed from clean coordinates using the same noisy state.
        epsilon = (state - alpha * clean) / sigma.clamp_min(1e-8)
        v_target = alpha * epsilon - sigma * clean
        coordinate_mask = residue_mask[..., None].expand_as(v)
        v_losses.append(
            ((v.float() - v_target.float()).square() * coordinate_mask).sum() / coordinate_mask.sum().clamp_min(1)
        )
        with _tracked_memory_phase(f"local_loss_construction_{_}", phase_device, phase_telemetry):
            geometries.append(
                masked_local_geometry(x0_hat, clean, residue_mask, continuity_mask, coordinate_scale_angstrom)
            )
        # Production deterministic reverse step; preserve this graph in both arms.
        with _tracked_memory_phase(f"reverse_transition_{_}", phase_device, phase_telemetry):
            state, _, _ = diffusion.deterministic_reverse_step(state, t, v, residue_mask)
        t = (t - 1).clamp_min(0)
    v_loss = torch.stack(v_losses).mean()
    immediate = geometries[0]
    post_transition = geometries[1]
    total = v_loss + geometry_weight * (immediate + post_transition)
    return {
        "v_mse": v_loss,
        "immediate_x0_geometry": immediate,
        "post_transition_x0_geometry": post_transition,
        "total": total,
    }


def matched_timesteps(update: int, *, seed: int, timesteps: int = 500) -> int:
    """Deterministic round-robin low/middle/high noise strata."""
    if update < 1 or timesteps < 6:
        raise ValueError("invalid update or diffusion schedule")
    stratum = (update - 1) % 3
    bounds = ((0, timesteps // 3 - 1), (timesteps // 3, 2 * timesteps // 3 - 1), (2 * timesteps // 3, timesteps - 1))
    lo, hi = bounds[stratum]
    digest = hashlib.sha256(f"{seed}:{update}:phase3i5-timestep".encode()).digest()
    return lo + int.from_bytes(digest[:8], "big") % (hi - lo + 1)


def _memory_snapshot(device: torch.device) -> dict[str, float | int | None]:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        status = Path("/proc/self/status").read_text()
        current_kib = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:"))
    except (OSError, StopIteration, ValueError) as error:
        raise ValueError("current process RSS telemetry unavailable") from error

    def is_tensor_object(item: Any) -> bool:
        item_type = type(item)
        module = getattr(item_type, "__module__", None)
        name = getattr(item_type, "__name__", None)
        return isinstance(module, str) and module.startswith("torch") and name in {"Tensor", "Parameter"}

    result: dict[str, float | int | None] = {
        "rss_current_mib": float(current_kib / 1024),
        "rss_peak_mib": float(rss / 1024),
        "live_tensor_objects": sum(1 for item in gc.get_objects() if is_tensor_object(item)),
    }
    if device.type != "cuda":
        result.update(
            {
                "current_allocated_mib": None,
                "current_reserved_mib": None,
                "peak_allocated_mib": None,
                "peak_reserved_mib": None,
                "device_capacity_mib": None,
            }
        )
        return result
    torch.cuda.synchronize(device)
    result.update(
        {
            "current_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
            "current_reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "device_capacity_mib": torch.cuda.get_device_properties(device).total_memory / 2**20,
        }
    )
    return result


@contextlib.contextmanager
def _tracked_memory_phase(name: str, device: torch.device, records: list[dict[str, Any]]):
    """Capture a direct allocator peak for one phase; never aggregate phase peaks."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    try:
        yield
    finally:
        records.append({"phase": name, **_memory_snapshot(device)})


def aggregate_direct_phase_peaks(
    phase_snapshots: list[dict[str, Any]], final_snapshot: dict[str, Any]
) -> dict[str, Any]:
    """Report the maximum observed direct phase peak, never a sum of peaks."""
    result = dict(final_snapshot)
    for key in ("peak_allocated_mib", "peak_reserved_mib"):
        observed = [float(row[key]) for row in phase_snapshots if row.get(key) is not None]
        if observed:
            result[key] = max([float(result.get(key) or 0.0), *observed])
    return result


def validate_memory_telemetry(snapshot: dict[str, Any], limits: dict[str, float]) -> None:
    """Fail closed for missing, non-finite, or over-budget telemetry."""
    fields = {
        "rss_current_mib": "rss",
        "peak_allocated_mib": "allocated",
        "peak_reserved_mib": "reserved",
    }
    for field, limit in fields.items():
        value = snapshot.get(field)
        if value is None or not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))):
            raise ValueError(f"invalid memory telemetry: {field}")
        if float(value) > float(limits[limit]):
            raise MemoryError(f"memory limit exceeded: {field}={value} > {limits[limit]}")
    if snapshot.get("device_capacity_mib") is not None and snapshot["device_capacity_mib"] <= 0:
        raise ValueError("invalid CUDA device capacity")
    if snapshot.get("device_capacity_mib") is not None and snapshot.get("peak_allocated_mib") is not None:
        if float(snapshot["device_capacity_mib"]) - float(snapshot["peak_allocated_mib"]) < 512.0:
            raise MemoryError("insufficient physical device headroom: less than 512 MiB")
    for field in ("current_allocated_mib", "current_reserved_mib", "device_capacity_mib"):
        if field not in snapshot or not isinstance(snapshot[field], (int, float)) or not np.isfinite(snapshot[field]):
            raise ValueError(f"invalid memory telemetry: {field}")


def validate_boundary_rss_trend(snapshots: list[dict[str, Any]], *, tolerance_mib: float = 256.0) -> None:
    """Compare cleaned-up steady-state RSS at boundaries 10 and 25."""
    by_boundary = {int(row["boundary"]): row for row in snapshots if "boundary" in row}
    if 10 not in by_boundary or 25 not in by_boundary:
        return
    reference, final = by_boundary[10], by_boundary[25]
    reference_rss = float(reference["rss_current_mib"])
    final_rss = float(final["rss_current_mib"])
    delta = final_rss - reference_rss
    if delta > tolerance_mib:
        raise MemoryError(
            "post-warm-up current RSS growth exceeded tolerance: "
            f"boundary 10={reference_rss:.3f}, boundary 25={final_rss:.3f} MiB, "
            f"delta={delta:.3f} MiB (tolerance {tolerance_mib:.1f} MiB)"
        )
    for row in (reference, final):
        actual = row.get("trajectory_record_count")
        expected = row.get("expected_trajectory_record_count")
        if actual is not None and expected is not None and int(actual) > int(expected):
            raise MemoryError(
                "tracked resource accumulation: trajectory_record_count "
                f"boundary {row['boundary']}={actual} > expected {expected}"
            )
    resource_fields = ("live_tensor_objects", "open_file_handles")
    for field in resource_fields:
        if field in reference and field in final and int(final[field]) > int(reference[field]):
            raise MemoryError(
                f"tracked resource accumulation: {field} boundary 10={reference[field]}, boundary 25={final[field]}"
            )


def _open_file_handle_count() -> int | None:
    """Count descriptors when the platform exposes the process descriptor table."""
    descriptor_dir = Path("/proc/self/fd")
    return len(list(descriptor_dir.iterdir())) if descriptor_dir.is_dir() else None


def matched_update_identity(update: int, seed: int) -> dict[str, int | str]:
    """Shared immutable training identity assigned to both arms."""
    if update < 1:
        raise ValueError("update identity must be positive")
    identity = hashlib.sha256(f"{seed}:phase3i5-example:{update}".encode()).hexdigest()
    return {
        "update": update,
        "identity_sha256": identity,
        "corruption_seed": seed + update,
        "timestep": matched_timesteps(update, seed=seed),
    }


def updates_to_boundary(current_update: int, boundary: int) -> range:
    """Return only remaining declared update identities; this version never extends past 25."""
    if current_update not in range(26) or boundary not in {10, 25}:
        raise ValueError("invalid Phase 3I.5 update boundary transition")
    if boundary == 25 and current_update < 10:
        raise ValueError("update 10 evaluation must complete before proceeding toward update 25")
    return range(current_update + 1, boundary + 1)


def validate_resume_cursor(update: int, data_cursor: int, completed: list[int], arm: str) -> None:
    if arm != "paired_boundary" or data_cursor != update or update not in range(26):
        raise ValueError("resume checkpoint is not a complete paired data boundary")
    if not set(completed).issubset({0, 10, 25}):
        raise ValueError("resume checkpoint contains an undeclared evaluation boundary")
    if update > 0 and 0 not in completed:
        raise ValueError("resume checkpoint omitted the initial evaluation")
    if update > 10 and 10 not in completed:
        raise ValueError("resume checkpoint omitted update-10 evaluation")
    if update > 25 and 25 not in completed:
        raise ValueError("resume checkpoint exceeded update 25")


def state_hash(state: dict[str, Any]) -> str:
    """Stable hash for CPU tensors and nested optimizer/checkpoint state."""
    digest = hashlib.sha256()

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(prefix.encode() + str(tensor.dtype).encode() + repr(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(value[key], f"{prefix}/{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{prefix}/{index}")
        else:
            digest.update(prefix.encode() + json.dumps(value, sort_keys=True, default=str).encode())

    visit(state, "state")
    return digest.hexdigest()


def make_resume_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    update: int,
    data_cursor: int,
    update_identity: dict[str, Any],
    protected_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "update": int(update),
        "data_cursor": int(data_cursor),
        "update_identity": update_identity,
        "protected_hashes": dict(protected_hashes),
        "state_hash": None,
    }


def verify_resume_state(payload: dict[str, Any], protected_hashes: dict[str, str]) -> None:
    required = {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "cpu_rng",
        "cuda_rng",
        "update",
        "data_cursor",
        "update_identity",
        "protected_hashes",
    }
    if not required.issubset(payload) or payload["protected_hashes"] != protected_hashes:
        raise ValueError("resume checkpoint is incomplete or protected artifact hashes changed")
    if int(payload["update"]) < 0 or int(payload["data_cursor"]) != int(payload["update"]):
        raise ValueError("resume cursor/update identity contradiction")
    if int(payload["update"]) and payload["update_identity"].get("update") != int(payload["update"]):
        raise ValueError("resume update identity mismatch")
    if payload.get("state_hash") != state_hash({k: v for k, v in payload.items() if k != "state_hash"}):
        raise ValueError("resume state hash mismatch")


def validate_contract(config_path: str | Path) -> dict[str, Any]:
    from protein_distance_diffusion.training.e007_phase3i5_contract import validate_contract as check

    return check(config_path)


def atomic_publish(staging: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    if not staging.is_dir():
        raise FileNotFoundError(staging)
    staging.replace(output)


def plan(config_path: str | Path) -> dict[str, Any]:
    from protein_distance_diffusion.training.e007_phase3i5_contract import plan as build_plan

    return build_plan(config_path)


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _rng_record() -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
    }


def _restore_rng(record: dict[str, Any]) -> None:
    torch.set_rng_state(record["cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in record["cuda"]])
    np.random.set_state(record["numpy"])


def _resolve_data(config: dict[str, Any]):
    from protein_distance_diffusion.data.e007_coordinate_dataset import E007CoordinateDataset
    from protein_distance_diffusion.data.rich_geometry import authorize_rich_geometry_dataset

    source = yaml.safe_load(Path(config["source_config"]).read_text())
    dataset_cfg = dict(source["dataset"])
    dataset_cfg["protected_input_relocations"] = [config["data_relocation"]]
    auth = authorize_rich_geometry_dataset(
        dataset_cfg["root"],
        expected_protocol_sha256=dataset_cfg["protocol_sha256"],
        expected_schema_sha256=dataset_cfg["schema_sha256"],
        expected_vocabulary_sha256=dataset_cfg["vocabulary_sha256"],
        expected_normalization_sha256=dataset_cfg["normalization_sha256"],
        expected_shard_inventory_sha256=dataset_cfg["shard_inventory_sha256"],
        protected_input_relocations=dataset_cfg["protected_input_relocations"],
    )
    return auth, E007CoordinateDataset(auth, split="train"), E007CoordinateDataset(auth, split="validation")


def _load_rows_by_id(dataset: Any, wanted: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for index in range(len(dataset)):
        row = dataset[index]
        identity = str(row["sample_id"])
        if identity in wanted:
            if not row["accepted_contiguous_single_chain"]:
                raise ValueError(f"development panel row failed coordinate policy: {identity}")
            found[identity] = row
            if len(found) == len(wanted):
                break
    if set(found) != wanted:
        raise ValueError(f"development data rows missing: {sorted(wanted - set(found))[:10]}")
    return found


def _select_training_rows(dataset: Any, count: int, seed: int, excluded: set[str]) -> list[dict[str, Any]]:
    # Hash-ranked deterministic random-access scan. Identity selection is shared by both arms.
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    selected = []
    for index in indices:
        row = dataset[int(index)]
        if row["accepted_contiguous_single_chain"] and row["sample_id"] not in excluded:
            selected.append(row)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError("training stream underfilled")
    return selected


def _prepare(row: dict[str, Any], source_config: dict[str, Any]) -> dict[str, Any]:
    from protein_distance_diffusion.training.e007_coordinate_real_loader_smoke import prepare_coordinate_batch

    return prepare_coordinate_batch(
        [row], float(source_config["coordinate_scale_angstrom"]), int(source_config["expected_downsample_factor"])
    )


def _load_model_optimizer(config: dict[str, Any], source_config: dict[str, Any], device: torch.device):
    from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet

    model = EquivariantPairCoordinateUNet(**source_config["model"]).to(device)
    optimizer_cfg = source_config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["learning_rate"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
        betas=tuple(optimizer_cfg["betas"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("optimizer_update") != 500 or not checkpoint.get("successful_optimizer_boundary"):
        raise ValueError("pinned source checkpoint is not a successful update-500 boundary")
    model.load_state_dict(checkpoint["model"])
    if "optimizer" not in checkpoint or "scheduler" not in checkpoint:
        raise ValueError("pinned update-500 checkpoint lacks optimizer/scheduler state")
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return model, optimizer, scheduler, checkpoint


def _parameter_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    diffusion: CoordinateVPDiffusion,
    prepared: dict[str, Any],
    timestep: int,
    seed: int,
    source_config: dict[str, Any],
    arm: str,
    *,
    activation_checkpointing: bool = False,
    amp_dtype: torch.dtype | None = None,
    scaler: Any = None,
    phase_telemetry: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, float], dict[str, float], int, int]:
    phase_telemetry = phase_telemetry if phase_telemetry is not None else []
    clean = prepared["coordinates"].cuda()
    mask = prepared["residue_mask"].cuda()
    continuity = prepared["chain_continuity_mask"].cuda()
    lengths = prepared["lengths"].cuda()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    times = torch.tensor([timestep], dtype=torch.long, device="cuda")
    corruption = diffusion.make_training_batch(clean, mask, timesteps=times, generator=generator)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    use_amp = amp_dtype is not None
    autocast = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
    with autocast:
        if arm == "continued_v_only":
            with _tracked_memory_phase("control_forward", clean.device, phase_telemetry):
                prediction = model(corruption.noisy_coordinates, times, lengths, mask, continuity)["v_prediction"]
            valid = mask[..., None].expand_as(prediction)
            with _tracked_memory_phase("control_loss_construction", clean.device, phase_telemetry):
                v_loss = (prediction[valid].float() - corruption.coordinate_v_target[valid].float()).square().mean()
            terms = {
                "v_mse": v_loss,
                "immediate_x0_geometry": v_loss * 0,
                "post_transition_x0_geometry": v_loss * 0,
                "total": v_loss,
            }
            gradient_terms = ("v_mse", "total")
            forwards = 1
        else:
            terms = sampler_unroll_objective(
                model,
                diffusion,
                corruption.noisy_coordinates,
                times,
                lengths,
                mask,
                continuity,
                clean,
                float(source_config["coordinate_scale_angstrom"]),
                geometry_weight=0.02,
                activation_checkpointing=activation_checkpointing,
                phase_telemetry=phase_telemetry,
            )
            gradient_terms = ("v_mse", "immediate_x0_geometry", "post_transition_x0_geometry", "total")
            forwards = 2
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradient_norms: dict[str, float] = {}
    # The training graph is used only for the optimizer objective. Component
    # audits are recomputed after the update and never retain the full graph.
    objective_loss = terms["total"]
    backward_phase = "control_loss_backward" if arm == "continued_v_only" else "total_objective_backward"
    with _tracked_memory_phase(backward_phase, clean.device, phase_telemetry):
        if use_amp and scaler is not None:
            scaler.scale(objective_loss).backward()
            scaler.unscale_(optimizer)
        else:
            objective_loss.backward()
    del objective_loss
    if not all(bool(torch.isfinite(parameter.grad).all()) for parameter in parameters if parameter.grad is not None):
        raise FloatingPointError("non-finite Phase 3I.5 gradient")
    torch.nn.utils.clip_grad_norm_(
        parameters, float(source_config["optimizer"]["gradient_clip_norm"]), error_if_nonfinite=True
    )
    optimizer_phase = "control_optimizer_step" if arm == "continued_v_only" else "unrolled_optimizer_step"
    with _tracked_memory_phase(optimizer_phase, clean.device, phase_telemetry):
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    losses = {name: float(value.detach().float().cpu()) for name, value in terms.items()}
    del terms
    if "prediction" in locals():
        del prediction
    if "v_loss" in locals():
        del v_loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Audit-only recomputation. Graph is freed on the final component.
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        if arm == "continued_v_only":
            prediction = model(corruption.noisy_coordinates, times, lengths, mask, continuity)["v_prediction"]
            valid = mask[..., None].expand_as(prediction)
            audit_terms = {
                "v_mse": ((prediction[valid].float() - corruption.coordinate_v_target[valid].float()).square()).mean()
            }
            audit_terms["total"] = audit_terms["v_mse"]
            audit_forwards = 1
        else:
            audit_terms = sampler_unroll_objective(
                model,
                diffusion,
                corruption.noisy_coordinates,
                times,
                lengths,
                mask,
                continuity,
                clean,
                float(source_config["coordinate_scale_angstrom"]),
                geometry_weight=0.02,
                activation_checkpointing=activation_checkpointing,
                phase_telemetry=phase_telemetry,
            )
            audit_forwards = 2
    for index, name in enumerate(gradient_terms):
        with _tracked_memory_phase(f"component_gradient_audit_{name}", clean.device, phase_telemetry):
            gradients = torch.autograd.grad(
                audit_terms[name].float(), parameters, retain_graph=index < len(gradient_terms) - 1, allow_unused=True
            )
        sq = sum(
            float(gradient.detach().double().square().sum().cpu()) for gradient in gradients if gradient is not None
        )
        gradient_norms[name] = float(np.sqrt(sq))
        del gradients
    del audit_terms, corruption, clean, mask, continuity, lengths, times
    if "prediction" in locals():
        del prediction
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        losses,
        gradient_norms,
        forwards + audit_forwards,
        len(gradient_terms) + 1,
    )


def _geometry_metrics(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, continuity: torch.Tensor
) -> dict[str, float]:
    length = int(mask[0].sum())
    pred = predicted[0, :length].detach().double() * 12.22820347644835
    true = target[0, :length].detach().double() * 12.22820347644835
    valid = torch.ones(length, dtype=torch.bool, device=pred.device)
    continuity = continuity[0, : max(length - 1, 0)].bool()
    pred_dist, true_dist = torch.cdist(pred, pred), torch.cdist(true, true)
    n = int(valid.sum())
    result: dict[str, float] = {}
    for offset in (1, 2, 3):
        if n <= offset:
            result[f"i_plus_{offset}_distance_rmse_angstrom"] = float("nan")
            continue
        link = valid[:-offset] & valid[offset:]
        p, t = pred_dist.diagonal(offset=offset), true_dist.diagonal(offset=offset)
        result[f"i_plus_{offset}_distance_rmse_angstrom"] = float((p[link] - t[link]).square().mean().sqrt())
        if offset == 1:
            good = link & (p.sub(t).abs() <= 0.5)
            result["valid_bond_fraction"] = float(good.sum() / link.sum().clamp_min(1))
            result["valid_residue_fraction"] = float(good.sum() / valid.sum().clamp_min(1))
            result["discontinuity_fraction"] = float(((p > 4.5) & link).sum() / link.sum().clamp_min(1))
    centered_pred = pred[valid] - pred[valid].mean(0)
    centered_true = true[valid] - true[valid].mean(0)
    result["radius_of_gyration_error_angstrom"] = float(
        (centered_pred.square().sum(-1).mean().sqrt() - centered_true.square().sum(-1).mean().sqrt()).abs()
    )
    if n >= 4:
        pb = pred[1:] - pred[:-1]
        tb = true[1:] - true[:-1]
        pvol = torch.linalg.cross(pb[:-2], pb[1:-1], dim=-1).mul(pb[2:]).sum(-1)
        tvol = torch.linalg.cross(tb[:-2], tb[1:-1], dim=-1).mul(tb[2:]).sum(-1)
        link = continuity[: n - 3] & continuity[1 : n - 2] & continuity[2 : n - 1]
        sign = (pvol * tvol > 0) & (pvol.abs() > 1e-8) & (tvol.abs() > 1e-8)
        result["chirality_agreement_fraction"] = float((sign & link).sum() / link.sum().clamp_min(1))

        def torsions(points: torch.Tensor) -> torch.Tensor:
            bonds = points[1:] - points[:-1]
            axis = F.normalize(bonds[1:-1], dim=-1)
            left = -bonds[:-2]
            right = bonds[2:]
            left = left - (left * axis).sum(-1, keepdim=True) * axis
            right = right - (right * axis).sum(-1, keepdim=True) * axis
            return torch.atan2(torch.linalg.cross(axis, left, dim=-1).mul(right).sum(-1), (left * right).sum(-1))

        predicted_torsion, target_torsion = torsions(pred), torsions(true)
        torsion_delta = torch.atan2(
            torch.sin(predicted_torsion - target_torsion), torch.cos(predicted_torsion - target_torsion)
        )
        result["dihedral_mae_radians"] = float(torsion_delta.abs().mean())
    else:
        result["chirality_agreement_fraction"] = 1.0
        result["dihedral_mae_radians"] = 0.0
    idx = torch.arange(n, device=pred.device)
    long_pairs = (idx[:, None] < idx[None, :]) & ((idx[:, None] - idx[None, :]).abs() >= 24)
    result["contact_density_at_8a"] = (
        float((pred_dist[long_pairs] <= 8.0).double().mean()) if bool(long_pairs.any()) else 0.0
    )
    return result


def _evaluate_arm(
    model: torch.nn.Module,
    diffusion: CoordinateVPDiffusion,
    rows: dict[str, dict[str, Any]],
    source_config: dict[str, Any],
    device: torch.device,
    boundary: int,
) -> list[dict[str, Any]]:
    model.eval()
    output = []
    for identity, row in sorted(rows.items()):
        prepared = _prepare(row, source_config)
        clean = prepared["coordinates"].to(device)
        mask = prepared["residue_mask"].to(device)
        continuity = prepared["chain_continuity_mask"].to(device)
        lengths = prepared["lengths"].to(device)
        t = torch.tensor([250], device=device)
        generator = torch.Generator(device=device).manual_seed(
            24_500_000 + int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
        )
        corruption = diffusion.make_training_batch(clean, mask, timesteps=t, generator=generator)
        with torch.inference_mode():
            prediction = model(corruption.noisy_coordinates, t, lengths, mask, continuity)["v_prediction"]
            x0_hat = diffusion.reconstruct_x0(corruption.noisy_coordinates, t, prediction, mask)
            metrics = _geometry_metrics(x0_hat, clean, mask, continuity)
        output.append(
            {
                "identity": identity,
                "boundary": boundary,
                "observed_length": int(row["sequence_length"]),
                "v_mse": float(
                    ((prediction - corruption.coordinate_v_target).square() * mask[..., None]).sum()
                    / mask.sum().clamp_min(1)
                    / 3
                ),
                **metrics,
            }
        )
    return output


def _sample_arm(
    model: torch.nn.Module, diffusion: CoordinateVPDiffusion, device: torch.device, panel: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    model.eval()
    outputs = []
    samples: dict[int, list[tuple[str, torch.Tensor]]] = {}
    for record in panel:
        length, seed = int(record["target_length"]), int(record["seed"])
        sample = diffusion.sample(model, length=length, seed=seed, device=device)["coordinates"]
        coordinates = sample[0].detach()
        distances = torch.cdist(coordinates.double() * 12.22820347644835, coordinates.double() * 12.22820347644835)
        bonds = distances.diagonal(offset=1)
        angle2 = distances.diagonal(offset=2)
        angle3 = distances.diagonal(offset=3)
        metrics = {
            "adjacent_distance_rmse_angstrom": float((bonds - 3.8).square().mean().sqrt()),
            "i_plus_2_distance_rmse_angstrom": float((angle2 - 6.2).square().mean().sqrt()),
            "i_plus_3_distance_rmse_angstrom": float((angle3 - 8.0).square().mean().sqrt()),
            "valid_bond_fraction": float(((bonds - 3.8).abs() <= 0.5).double().mean()),
            "valid_residue_fraction": float(((bonds - 3.8).abs() <= 0.5).double().mean()),
            "discontinuity_fraction": float((bonds > 4.5).double().mean()),
            "radius_of_gyration_angstrom": float(
                (coordinates.double() * 12.22820347644835 - (coordinates.double() * 12.22820347644835).mean(0))
                .norm(dim=-1)
                .square()
                .mean()
                .sqrt()
            ),
        }
        separation = torch.arange(length, device=device)
        upper_long = (separation[:, None] < separation[None, :]) & (
            (separation[:, None] - separation[None, :]).abs() >= 24
        )
        metrics["contact_density_at_8a"] = (
            float((distances[upper_long] <= 8.0).double().mean()) if bool(upper_long.any()) else 0.0
        )
        if length >= 4:
            vectors = coordinates[1:] - coordinates[:-1]
            volumes = torch.linalg.cross(vectors[:-2], vectors[1:-1], dim=-1).mul(vectors[2:]).sum(-1)
            metrics["chirality_nonzero_fraction"] = float((volumes.abs() > 1e-8).double().mean())
            b0, b1, b2 = -vectors[:-2], vectors[1:-1], vectors[2:]
            axis = F.normalize(b1, dim=-1)
            v = b0 - (b0 * axis).sum(-1, keepdim=True) * axis
            w = b2 - (b2 * axis).sum(-1, keepdim=True) * axis
            dihedral = torch.atan2(torch.linalg.cross(axis, v, dim=-1).mul(w).sum(-1), (v * w).sum(-1))
            metrics["dihedral_abs_mean_radians"] = float(dihedral.abs().mean())
        else:
            metrics["chirality_nonzero_fraction"] = 1.0
            metrics["dihedral_abs_mean_radians"] = 0.0
        samples.setdefault(length, []).append((str(record["identity"]), coordinates))
        outputs.append({"identity": record["identity"], "seed": seed, "target_length": length, **metrics})
    for length_records in samples.values():
        if len(length_records) < 2:
            continue
        first, second = length_records[0][1], length_records[1][1]
        first, second = first.double() * 12.22820347644835, second.double() * 12.22820347644835
        a, b = torch.cdist(first, first), torch.cdist(second, second)
        diversity = float((a - b).square().mean().sqrt())
        for record in outputs:
            if record["identity"] in {length_records[0][0], length_records[1][0]}:
                record["within_length_diversity_descriptor_rmse"] = diversity
    return outputs


def _protected_hashes(config_path: Path, config: dict[str, Any], dataset_hashes: dict[str, str]) -> dict[str, str]:
    keys = (
        "checkpoint",
        "source_config",
        "phase3i4_protocol",
        "phase3i4_v2_config",
        "development_panel",
        "phase3i4_calibration_panel",
        "phase3i4_holdout_hash_record",
        "phase3i4_calibration_report",
        "phase3i4_performance_report",
        "phase3i3_report",
    )
    hashes = {key: file_sha256(config[key]) for key in keys}
    hashes["configuration"] = file_sha256(config_path)
    root = Path(__file__).resolve().parents[3]
    for label, relative in {
        "execution_script": "scripts/run_e007_phase3i5_sampler_unroll.py",
        "training_implementation": "src/protein_distance_diffusion/training/e007_phase3i5_sampler_unroll.py",
        "contract_implementation": "src/protein_distance_diffusion/training/e007_phase3i5_contract.py",
        "panel_builder": "scripts/build_e007_phase3i5_development_panel.py",
    }.items():
        hashes[f"execution_evidence_{label}"] = file_sha256(root / relative)
    return {**hashes, **{f"dataset_{key}": value for key, value in dataset_hashes.items()}}


def _commit(staging: Path, payload: dict[str, Any], update: int) -> None:
    payload = dict(payload)
    payload["state_hash"] = state_hash(payload)
    name = f"boundary-{update:03d}.pt"
    checkpoint = staging / "checkpoints" / name
    _atomic_torch(checkpoint, payload)
    pointer = {
        "checkpoint": str(checkpoint.relative_to(staging)),
        "sha256": file_sha256(checkpoint),
        "update": update,
        "state_hash": payload["state_hash"],
    }
    _atomic_json(staging / "CURRENT.json", pointer)


def _load_current(staging: Path, protected_hashes: dict[str, str]) -> dict[str, Any]:
    if (staging / "failure.json").exists():
        raise ValueError("failed Phase 3I.5 staging directories are not resumable")
    if not (staging / "run_manifest.json").is_file():
        raise ValueError("Phase 3I.5 staging directory lacks its immutable run manifest")
    pointer = json.loads((staging / "CURRENT.json").read_text())
    path = (staging / pointer["checkpoint"]).resolve()
    checkpoint_root = (staging / "checkpoints").resolve()
    try:
        path.relative_to(checkpoint_root)
    except ValueError as error:
        raise ValueError("current staging pointer escapes the checkpoints directory") from error
    if file_sha256(path) != pointer["sha256"]:
        raise ValueError("current staging checkpoint file hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    verify_resume_state(payload, protected_hashes)
    required = {"arms", "arm", "numpy_rng", "completed_evaluation_boundaries", "training_identities", "workload_counts"}
    if not required.issubset(payload):
        raise ValueError("Phase 3I.5 staging checkpoint lacks exact-resume fields")
    if payload["update"] != pointer["update"] or payload["state_hash"] != pointer["state_hash"]:
        raise ValueError("staging checkpoint pointer/state mismatch")
    return payload


def execute(config_path: str | Path, mode: str, resume: bool = False) -> dict[str, Any]:
    """Execute exact 25-update paired study, or the isolated disposable CUDA smoke."""
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    validation = validate_contract(config_path)
    if mode == "cuda-memory-smoke":
        return run_cuda_smoke(config_path, cfg, validation)
    if mode not in {"execute", "resume"}:
        raise ValueError(f"unsupported Phase 3I.5 lifecycle mode {mode}")
    staging, output = Path(cfg["staging_dir"]), Path(cfg["output_dir"])
    if output.exists():
        raise FileExistsError(output)
    if resume and not staging.is_dir():
        raise FileNotFoundError("exact-state resume requires a valid Phase 3I.5 staging directory")
    if not resume and (staging.exists() or output.exists()):
        raise FileExistsError("fresh execute refuses an existing Phase 3I.5 path")
    if not resume:
        staging.mkdir(parents=True)
        (staging / "checkpoints").mkdir()
    _atomic_json(staging / "heartbeat.json", {"status": "initializing", **NON_AUTHORIZING})
    try:
        started = time.perf_counter()
        if not torch.cuda.is_available():
            raise RuntimeError(f"Phase 3I.5 {mode} requires CUDA")
        source = yaml.safe_load(Path(cfg["source_config"]).read_text())
        auth, train_dataset, validation_dataset = _resolve_data(cfg)
        panel = json.loads(Path(cfg["development_panel"]).read_text())
        panel_records = panel["records"]
        panel_ids = {str(record["sample_id"]) for record in panel_records}
        panel_rows = _load_rows_by_id(validation_dataset, panel_ids)
        for record in panel_records:
            panel_rows[str(record["sample_id"])]["target_length"] = int(record["target_length"])
        train_rows = _select_training_rows(train_dataset, 25, int(cfg["seed"]), panel_ids)
        train_ids = [str(row["sample_id"]) for row in train_rows]
        calibration_ids = set(panel["calibration_exclusion"]["identities"])
        if len(set(train_ids)) != 25 or panel_ids & set(train_ids) or set(train_ids) & calibration_ids:
            raise ValueError("training/development identity disjointness failed")
        protected = _protected_hashes(config_path, cfg, auth.observed_shard_hashes)
        panel_bytes = json.dumps(panel, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        train_identity_doc = {"identities": train_ids, "split": "train"}
        train_bytes = json.dumps(train_identity_doc, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        run_manifest = {
            "version": cfg.get("version", VERSION),
            "configuration_sha256": file_sha256(config_path),
            "protected_hashes": protected,
            "development_panel_file_sha256": hashlib.sha256(panel_bytes).hexdigest(),
            "training_identities_file_sha256": hashlib.sha256(train_bytes).hexdigest(),
            "training_identities": train_ids,
            "panel_identities": sorted(panel_ids),
        }
        manifest_path = staging / "run_manifest.json"
        if resume:
            if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != run_manifest:
                raise ValueError("staging run manifest does not match current pinned inputs and selected identities")
            if (
                file_sha256(staging / "development_panel.json") != run_manifest["development_panel_file_sha256"]
                or file_sha256(staging / "training_identities.json") != run_manifest["training_identities_file_sha256"]
            ):
                raise ValueError("staging identity manifests changed")
        else:
            _atomic_json(staging / "development_panel.json", panel)
            _atomic_json(staging / "training_identities.json", train_identity_doc)
            _atomic_json(manifest_path, run_manifest)
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        model, optimizer, scheduler, source_checkpoint = _load_model_optimizer(cfg, source, device)
        is_v2 = cfg.get("version", "").endswith(("_v2", "_v3_reviewed", "_v4_reviewed", "_v5_reviewed"))
        amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if is_v2 else None
        scaler = torch.amp.GradScaler("cuda", enabled=is_v2 and amp_dtype == torch.float16)
        diffusion = CoordinateVPDiffusion(int(cfg["timesteps"]))
        evaluation_path = staging / "evaluations.json"
        evaluations = json.loads(evaluation_path.read_text()) if resume and evaluation_path.exists() else {}
        rss_boundaries_path = staging / "rss_boundaries.json"
        rss_boundaries = (
            json.loads(rss_boundaries_path.read_text()).get("snapshots", [])
            if resume and rss_boundaries_path.exists()
            else []
        )
        sampling_path = staging / "sampling.json"
        samplings = json.loads(sampling_path.read_text()) if resume and sampling_path.exists() else {}
        if resume:
            state = _load_current(staging, protected)
            states = state["arms"]
            update = int(state["update"])
            completed_boundaries = list(state["completed_evaluation_boundaries"])
            if state["training_identities"] != train_ids:
                raise ValueError("resume training identity stream changed")
            validate_resume_cursor(update, int(state["data_cursor"]), completed_boundaries, state["arm"])
            if update and state["update_identity"] != states["continued_v_only"]["last_identity"]:
                raise ValueError("resume top-level update identity differs from continued_v_only")
            _restore_rng({"cpu": state["cpu_rng"], "cuda": state["cuda_rng"], "numpy": state["numpy_rng"]})
            for arm in cfg["arms"]:
                if set(states[arm]) < {"model", "optimizer", "scheduler", "scaler", "rng", "last_identity"}:
                    raise ValueError(f"resume state is incomplete for {arm}")
                if update and (
                    states[arm]["last_identity"].get("sample_id") != train_ids[update - 1]
                    or states[arm]["last_identity"].get("update") != update
                    or states[arm]["last_identity"].get("timestep")
                    != matched_timesteps(update, seed=int(cfg["seed"]), timesteps=int(cfg["timesteps"]))
                ):
                    raise ValueError(f"resume update identity does not match the deterministic data cursor for {arm}")
                verify_resume_state(
                    {
                        "model": states[arm]["model"],
                        "optimizer": states[arm]["optimizer"],
                        "scheduler": states[arm]["scheduler"],
                        "scaler": states[arm]["scaler"],
                        "cpu_rng": states[arm]["rng"]["cpu"],
                        "cuda_rng": states[arm]["rng"]["cuda"],
                        "update": update,
                        "data_cursor": update,
                        "update_identity": states[arm].get("last_identity", {"update": update}),
                        "protected_hashes": protected,
                        "state_hash": state_hash(
                            {
                                "model": states[arm]["model"],
                                "optimizer": states[arm]["optimizer"],
                                "scheduler": states[arm]["scheduler"],
                                "scaler": states[arm]["scaler"],
                                "cpu_rng": states[arm]["rng"]["cpu"],
                                "cuda_rng": states[arm]["rng"]["cuda"],
                                "update": update,
                                "data_cursor": update,
                                "update_identity": states[arm].get("last_identity", {"update": update}),
                                "protected_hashes": protected,
                            }
                        ),
                    },
                    protected,
                )
            if states["continued_v_only"]["last_identity"] != states["v_plus_sampler_unroll"]["last_identity"]:
                raise ValueError("resume arm update identities are not matched")
        else:
            update = 0
            completed_boundaries = []
            base = {
                "model": copy.deepcopy(source_checkpoint["model"]),
                "optimizer": copy.deepcopy(source_checkpoint["optimizer"]),
                "scheduler": copy.deepcopy(source_checkpoint["scheduler"]),
                "scaler": scaler.state_dict(),
            }
            states = {arm: copy.deepcopy(base) for arm in cfg["arms"]}
            for arm in cfg["arms"]:
                states[arm].update({"rng": _rng_record(), "last_identity": {"update": 0}})
            initial_payload = {
                "model": states["continued_v_only"]["model"],
                "optimizer": states["continued_v_only"]["optimizer"],
                "scheduler": states["continued_v_only"]["scheduler"],
                "scaler": states["continued_v_only"]["scaler"],
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(),
                "arms": states,
                "update": 0,
                "data_cursor": 0,
                "arm": "paired_boundary",
                "update_identity": {"update": 0},
                "training_identities": train_ids,
                "completed_evaluation_boundaries": [],
                "protected_hashes": protected,
                "workload_counts": {
                    "forwards": {
                        "training": {arm: 0 for arm in cfg["arms"]},
                        "audit": 0,
                        "one_step_evaluation": 0,
                        "production_sampler": 0,
                    },
                    "backwards": {"component_gradient_audits": 0, "total_objective_backward": 0},
                },
            }
            _commit(staging, initial_payload, 0)
        if resume:
            forward_counts = state["workload_counts"]["forwards"]
            backward_counts = state["workload_counts"]["backwards"]
            history_path = staging / "training_history.json"
            training_history = (
                [row for row in json.loads(history_path.read_text()).get("records", []) if int(row["update"]) <= update]
                if history_path.exists()
                else []
            )
        else:
            forward_counts = {
                "training": {arm: 0 for arm in cfg["arms"]},
                "audit": 0,
                "one_step_evaluation": 0,
                "production_sampler": 0,
            }
            backward_counts = {"component_gradient_audits": 0, "total_objective_backward": 0}
            training_history = []
        if time.perf_counter() - started > float(cfg["maximum_wall_seconds"]):
            raise TimeoutError("Phase 3I.5 conservative hard wall bound exceeded during startup")
        prepared = None
        with coordinate_model_execution_context(source["numerics"], device):
            for boundary in (0, 10, 25):
                if boundary not in completed_boundaries and update == boundary:
                    boundary_rows = {}
                    boundary_sampling = {}
                    for arm in cfg["arms"]:
                        model.load_state_dict(states[arm]["model"])
                        boundary_rows[arm] = _evaluate_arm(model, diffusion, panel_rows, source, device, boundary)
                        forward_counts["one_step_evaluation"] += len(panel_rows)
                        if boundary in (0, 25):
                            boundary_sampling[arm] = _sample_arm(model, diffusion, device, panel_records)
                            forward_counts["production_sampler"] += len(panel_records) * int(cfg["timesteps"])
                    paired = []
                    control_by_id = {row["identity"]: row for row in boundary_rows["continued_v_only"]}
                    unroll_by_id = {row["identity"]: row for row in boundary_rows["v_plus_sampler_unroll"]}
                    for identity in sorted(panel_ids):
                        paired.append(
                            {
                                "identity": identity,
                                "target_length": panel_rows[identity]["target_length"],
                                "continued_v_only": control_by_id[identity],
                                "v_plus_sampler_unroll": unroll_by_id[identity],
                            }
                        )
                    evaluations[str(boundary)] = paired
                    expected_trajectory_records = len(panel_ids) * (len(evaluations))
                    actual_trajectory_records = sum(len(rows) for rows in evaluations.values())
                    if actual_trajectory_records != expected_trajectory_records:
                        raise RuntimeError("evaluation trajectory records accumulated or went missing")
                    if boundary in (0, 25):
                        samples_by_id = {
                            arm: {row["identity"]: row for row in boundary_sampling[arm]} for arm in cfg["arms"]
                        }
                        samplings[str(boundary)] = [
                            {
                                "identity": identity,
                                "continued_v_only": samples_by_id["continued_v_only"][identity],
                                "v_plus_sampler_unroll": samples_by_id["v_plus_sampler_unroll"][identity],
                            }
                            for identity in sorted(samples_by_id["continued_v_only"])
                        ]
                        if len(samplings[str(boundary)]) != len(panel_ids):
                            raise RuntimeError("production sampling trajectory records accumulated or went missing")
                        _atomic_json(sampling_path, samplings)
                    _atomic_json(evaluation_path, evaluations)
                    completed_boundaries.append(boundary)
                    state_payload = {
                        "model": states["continued_v_only"]["model"],
                        "optimizer": states["continued_v_only"]["optimizer"],
                        "scheduler": states["continued_v_only"]["scheduler"],
                        "scaler": states["continued_v_only"]["scaler"],
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                        "numpy_rng": np.random.get_state(),
                        "arms": states,
                        "update": update,
                        "data_cursor": update,
                        "arm": "paired_boundary",
                        "update_identity": states["continued_v_only"]["last_identity"],
                        "training_identities": train_ids,
                        "completed_evaluation_boundaries": completed_boundaries,
                        "protected_hashes": protected,
                        "workload_counts": {"forwards": forward_counts, "backwards": backward_counts},
                    }
                    _commit(staging, state_payload, update)
                    if len(training_history) != update:
                        raise RuntimeError("training trajectory records do not match the declared update cursor")
                    # Drop phase-local batches and evaluation/sampling result objects before
                    # measuring steady-state process RSS. The persisted records remain on disk.
                    prepared = None
                    del boundary_rows, paired, control_by_id, unroll_by_id
                    if "boundary_sampling" in locals():
                        del boundary_sampling
                    if "samples_by_id" in locals():
                        del samples_by_id
                    del state_payload
                    gc.collect()
                    memory = _memory_snapshot(device)
                    validate_memory_telemetry(memory, cfg["memory_limit_mib"])
                    actual_trajectory_records = sum(len(rows) for rows in evaluations.values()) + sum(
                        len(rows) for rows in samplings.values()
                    )
                    memory["trajectory_record_count"] = actual_trajectory_records
                    memory["expected_trajectory_record_count"] = expected_trajectory_records + len(panel_ids) * len(
                        samplings
                    )
                    open_handles = _open_file_handle_count()
                    if open_handles is not None:
                        memory["open_file_handles"] = open_handles
                    rss_boundaries = [row for row in rss_boundaries if int(row["boundary"]) != boundary]
                    rss_boundaries.append({"boundary": boundary, **memory})
                    rss_boundaries.sort(key=lambda row: int(row["boundary"]))
                    previous_rss = None
                    reference_rss = next(
                        (float(row["rss_current_mib"]) for row in rss_boundaries if int(row["boundary"]) == 10),
                        None,
                    )
                    for row in rss_boundaries:
                        current_rss = float(row["rss_current_mib"])
                        row["rss_delta_from_previous_boundary_mib"] = (
                            None if previous_rss is None else current_rss - previous_rss
                        )
                        row["rss_delta_from_boundary_10_mib"] = (
                            None if reference_rss is None else current_rss - reference_rss
                        )
                        previous_rss = current_rss
                    validate_boundary_rss_trend(
                        rss_boundaries,
                        tolerance_mib=float(
                            cfg.get("memory_policy", {}).get("rss_monotonic_growth_tolerance_mib", 256)
                        ),
                    )
                    _atomic_json(rss_boundaries_path, {"snapshots": rss_boundaries})
                    if time.perf_counter() - started > float(cfg["maximum_wall_seconds"]):
                        raise TimeoutError("Phase 3I.5 conservative hard wall bound exceeded")
                if boundary == 25:
                    break
                next_boundary = 10 if boundary == 0 else 25
                for next_update in updates_to_boundary(update, next_boundary):
                    row = train_rows[next_update - 1]
                    prepared = _prepare(row, source)
                    timestep = matched_timesteps(next_update, seed=int(cfg["seed"]), timesteps=int(cfg["timesteps"]))
                    identity = {
                        "update": next_update,
                        "sample_id": row["sample_id"],
                        "timestep": timestep,
                        "corruption_seed": int(cfg["seed"]) + next_update,
                    }
                    for arm in cfg["arms"]:
                        model.load_state_dict(states[arm]["model"])
                        optimizer.load_state_dict(states[arm]["optimizer"])
                        scheduler.load_state_dict(states[arm]["scheduler"])
                        scaler.load_state_dict(states[arm]["scaler"])
                        _restore_rng(states[arm]["rng"])
                        torch.manual_seed(int(cfg["seed"]) + next_update)
                        torch.cuda.manual_seed_all(int(cfg["seed"]) + next_update)
                        torch.cuda.synchronize(device)
                        begin = time.perf_counter()
                        loss_records, grad_records, forwards, backwards = _train_step(
                            model,
                            optimizer,
                            diffusion,
                            prepared,
                            timestep,
                            int(cfg["seed"]) + next_update,
                            source,
                            arm,
                            activation_checkpointing=is_v2,
                            amp_dtype=amp_dtype,
                            scaler=scaler,
                        )
                        scheduler.step()
                        torch.cuda.synchronize(device)
                        states[arm] = {
                            "model": copy.deepcopy(model.state_dict()),
                            "optimizer": copy.deepcopy(optimizer.state_dict()),
                            "scheduler": copy.deepcopy(scheduler.state_dict()),
                            "scaler": copy.deepcopy(scaler.state_dict()),
                            "rng": _rng_record(),
                            "last_identity": identity,
                            "last_losses": loss_records,
                            "last_gradients": grad_records,
                            "last_update_seconds": time.perf_counter() - begin,
                        }
                        forward_counts["training"][arm] += forwards // 2
                        forward_counts["audit"] += forwards // 2
                        backward_counts["component_gradient_audits"] += backwards - 1
                        backward_counts["total_objective_backward"] += 1
                    update = next_update
                    training_history.append(
                        {
                            "update": update,
                            "identity": identity,
                            "arms": {
                                arm: {
                                    "losses": states[arm]["last_losses"],
                                    "gradient_norms": states[arm]["last_gradients"],
                                }
                                for arm in cfg["arms"]
                            },
                        }
                    )
                    _atomic_json(staging / "training_history.json", {"records": training_history})
                    state_payload = {
                        "model": states["continued_v_only"]["model"],
                        "optimizer": states["continued_v_only"]["optimizer"],
                        "scheduler": states["continued_v_only"]["scheduler"],
                        "scaler": states["continued_v_only"]["scaler"],
                        "cpu_rng": states["continued_v_only"]["rng"]["cpu"],
                        "cuda_rng": states["continued_v_only"]["rng"]["cuda"],
                        "numpy_rng": np.random.get_state(),
                        "arms": states,
                        "update": update,
                        "data_cursor": update,
                        "arm": "paired_boundary",
                        "update_identity": identity,
                        "training_identities": train_ids,
                        "completed_evaluation_boundaries": completed_boundaries,
                        "protected_hashes": protected,
                        "workload_counts": {"forwards": forward_counts, "backwards": backward_counts},
                    }
                    _commit(staging, state_payload, update)
                    if time.perf_counter() - started > float(cfg["maximum_wall_seconds"]):
                        raise TimeoutError("Phase 3I.5 conservative hard wall bound exceeded")
                    memory = _memory_snapshot(device)
                    validate_memory_telemetry(memory, cfg["memory_limit_mib"])
            if update != 25 or completed_boundaries != [0, 10, 25]:
                raise RuntimeError("Phase 3I.5 did not complete all declared boundaries")
        report = {
            "status": "completed_non_authorizing_awaiting_checkpoint25_review",
            "version": cfg.get("version", VERSION),
            "evaluations": evaluations,
            "production_sampler_evaluations": samplings,
            "workload_counts": {
                "training_forwards": forward_counts["training"],
                **forward_counts,
                "backward_calls": backward_counts,
                "backward_calls_by_arm": {
                    "continued_v_only": {"component_gradient_audits": 50, "total_objective_backward": 25},
                    "v_plus_sampler_unroll": {"component_gradient_audits": 100, "total_objective_backward": 25},
                },
                "total_forwards": sum(forward_counts["training"].values())
                + int(forward_counts["audit"])
                + forward_counts["one_step_evaluation"]
                + forward_counts["production_sampler"],
            },
            "runtime_seconds": time.perf_counter() - started,
            "cuda_memory": _memory_snapshot(device),
            "protected_hashes": protected,
            "pre_smoke_conservative_runtime_bound_seconds": cfg["conservative_pre_smoke_runtime_bound_seconds"],
            "post_smoke_empirical_runtime_projection_seconds": None,
            "decision": _classify(evaluations, samplings, cfg),
            "checkpoint25_review": {"decision_required_before_any_extension": True, "extension_authorized": False},
            **NON_AUTHORIZING,
        }
        _atomic_json(
            staging / "checkpoint25_review.json",
            {
                "status": "awaiting_independent_review",
                "decision": None,
                "extension_authorized": False,
                "new_versioned_configuration_required_for_extension": True,
                "report_sha256": hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest(),
                **NON_AUTHORIZING,
            },
        )
        _atomic_json(staging / "report.json", report)
        atomic_publish(staging, output)
        return report
    except Exception as exc:
        _atomic_json(
            staging / "failure.json", {"status": "failed_closed", "mode": mode, "error": repr(exc), **NON_AUTHORIZING}
        )
        raise


def _classify(evaluations: dict[str, Any], samplings: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    rows = evaluations.get("25", [])
    if not rows:
        return {"label": "inconclusive", "paired_adjacent_rmse_improvement_angstrom": None}
    improvements = [
        row["continued_v_only"]["i_plus_1_distance_rmse_angstrom"]
        - row["v_plus_sampler_unroll"]["i_plus_1_distance_rmse_angstrom"]
        for row in rows
    ]
    mean = float(np.mean(improvements))
    decision = config["decision"]
    rng = np.random.default_rng(int(decision["paired_bootstrap_seed"]))
    draws = rng.choice(
        np.asarray(improvements), size=(int(decision["paired_bootstrap_replicates"]), len(improvements)), replace=True
    ).mean(axis=1)
    ci_low, ci_high = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
    statistically_positive = ci_low > 0
    threshold = float(decision["meaningful_adjacent_rmse_improvement_angstrom"])
    label = (
        "technically_positive_scientifically_negligible"
        if statistically_positive and 0 < mean < threshold
        else "meaningful"
        if statistically_positive and mean >= threshold
        else "no_statistically_positive_improvement"
    )
    bond_delta = float(
        np.mean(
            [
                row["v_plus_sampler_unroll"]["valid_bond_fraction"] - row["continued_v_only"]["valid_bond_fraction"]
                for row in rows
            ]
        )
    )
    residue_delta = float(
        np.mean(
            [
                row["v_plus_sampler_unroll"]["valid_residue_fraction"]
                - row["continued_v_only"]["valid_residue_fraction"]
                for row in rows
            ]
        )
    )
    v_mse_delta = float(
        np.mean([row["v_plus_sampler_unroll"]["v_mse"] - row["continued_v_only"]["v_mse"] for row in rows])
    )
    rg_delta = float(
        np.mean(
            [
                row["v_plus_sampler_unroll"]["radius_of_gyration_error_angstrom"]
                - row["continued_v_only"]["radius_of_gyration_error_angstrom"]
                for row in rows
            ]
        )
    )
    final_control_v_mse = float(np.mean([row["continued_v_only"]["v_mse"] for row in rows]))
    relative_v_mse_degradation = v_mse_delta / max(final_control_v_mse, 1e-12)
    sample_rows = samplings.get("25", [])
    sample_adjacent = (
        float(
            np.mean(
                [
                    row["continued_v_only"]["adjacent_distance_rmse_angstrom"]
                    - row["v_plus_sampler_unroll"]["adjacent_distance_rmse_angstrom"]
                    for row in sample_rows
                ]
            )
        )
        if sample_rows
        else float("nan")
    )
    sample_bond_delta = (
        float(
            np.mean(
                [
                    row["v_plus_sampler_unroll"]["valid_bond_fraction"] - row["continued_v_only"]["valid_bond_fraction"]
                    for row in sample_rows
                ]
            )
        )
        if sample_rows
        else float("nan")
    )
    rg_sample_delta = (
        float(
            np.mean(
                [
                    row["v_plus_sampler_unroll"]["radius_of_gyration_angstrom"]
                    - row["continued_v_only"]["radius_of_gyration_angstrom"]
                    for row in sample_rows
                ]
            )
        )
        if sample_rows
        else float("nan")
    )
    contact_delta = (
        float(
            np.mean(
                [
                    row["v_plus_sampler_unroll"]["contact_density_at_8a"]
                    - row["continued_v_only"]["contact_density_at_8a"]
                    for row in sample_rows
                ]
            )
        )
        if sample_rows
        else float("nan")
    )
    diversity_rows = [
        row
        for row in sample_rows
        if "within_length_diversity_descriptor_rmse" in row["continued_v_only"]
        and "within_length_diversity_descriptor_rmse" in row["v_plus_sampler_unroll"]
    ]
    diversity_delta = (
        float(
            np.mean(
                [
                    row["v_plus_sampler_unroll"]["within_length_diversity_descriptor_rmse"]
                    - row["continued_v_only"]["within_length_diversity_descriptor_rmse"]
                    for row in diversity_rows
                ]
            )
        )
        if diversity_rows
        else float("nan")
    )
    control_diversity = (
        float(np.mean([row["continued_v_only"]["within_length_diversity_descriptor_rmse"] for row in diversity_rows]))
        if diversity_rows
        else 0.0
    )
    diversity_relative_delta = diversity_delta / max(control_diversity, 1e-12) if diversity_rows else float("nan")
    tolerance = config["safeguard_tolerances"]
    i2_delta = float(
        np.mean(
            [
                row["v_plus_sampler_unroll"]["i_plus_2_distance_rmse_angstrom"]
                - row["continued_v_only"]["i_plus_2_distance_rmse_angstrom"]
                for row in rows
            ]
        )
    )
    i3_delta = float(
        np.mean(
            [
                row["v_plus_sampler_unroll"]["i_plus_3_distance_rmse_angstrom"]
                - row["continued_v_only"]["i_plus_3_distance_rmse_angstrom"]
                for row in rows
            ]
        )
    )
    discontinuity_delta = float(
        np.mean(
            [
                row["v_plus_sampler_unroll"]["discontinuity_fraction"]
                - row["continued_v_only"]["discontinuity_fraction"]
                for row in rows
            ]
        )
    )
    safeguards = {
        "v_mse": relative_v_mse_degradation <= tolerance["v_mse_relative_degradation"],
        "radius_of_gyration": rg_sample_delta >= -tolerance["radius_of_gyration_error_degradation_angstrom"],
        "contact_density": contact_delta >= -tolerance["contact_density_absolute_degradation"],
        "i_plus_2_global_geometry": i2_delta <= tolerance["i_plus_2_rmse_degradation_angstrom"],
        "i_plus_3_global_geometry": i3_delta <= tolerance["i_plus_3_rmse_degradation_angstrom"],
        "discontinuity": discontinuity_delta <= tolerance["discontinuity_fraction_degradation"],
        "validity": min(bond_delta, residue_delta) >= -tolerance["validity_fraction_material_regression"],
        "diversity": diversity_relative_delta >= -tolerance["diversity_relative_degradation"],
        "numerical_stability": all(np.isfinite(value) for value in [*improvements, *draws]),
    }
    by_length = {}
    for target_length in sorted({int(row["target_length"]) for row in rows}):
        length_rows = [row for row in rows if int(row["target_length"]) == target_length]
        length_delta = [
            row["continued_v_only"]["i_plus_1_distance_rmse_angstrom"]
            - row["v_plus_sampler_unroll"]["i_plus_1_distance_rmse_angstrom"]
            for row in length_rows
        ]
        by_length[str(target_length)] = {
            "paired_identity_count": len(length_rows),
            "mean_adjacent_rmse_improvement_angstrom": float(np.mean(length_delta)),
            "valid_bond_fraction_improvement": float(
                np.mean(
                    [
                        row["v_plus_sampler_unroll"]["valid_bond_fraction"]
                        - row["continued_v_only"]["valid_bond_fraction"]
                        for row in length_rows
                    ]
                )
            ),
            "valid_residue_fraction_improvement": float(
                np.mean(
                    [
                        row["v_plus_sampler_unroll"]["valid_residue_fraction"]
                        - row["continued_v_only"]["valid_residue_fraction"]
                        for row in length_rows
                    ]
                )
            ),
        }
    return {
        "label": label,
        "paired_adjacent_rmse_improvement_angstrom": mean,
        "paired_bootstrap_95pct_ci_angstrom": [ci_low, ci_high],
        "statistically_positive": statistically_positive,
        "paired_identity_count": len(rows),
        "paired_comparison_by_target_length": by_length,
        "paired_valid_bond_fraction_improvement": bond_delta,
        "paired_valid_residue_fraction_improvement": residue_delta,
        "paired_v_mse_change": v_mse_delta,
        "paired_radius_of_gyration_error_change_angstrom": rg_delta,
        "paired_i_plus_2_rmse_change_angstrom": i2_delta,
        "paired_i_plus_3_rmse_change_angstrom": i3_delta,
        "paired_discontinuity_fraction_change": discontinuity_delta,
        "production_sampler_paired_adjacent_rmse_improvement_angstrom": sample_adjacent,
        "production_sampler_valid_bond_fraction_improvement": sample_bond_delta,
        "production_sampler_radius_of_gyration_change_angstrom": rg_sample_delta,
        "production_sampler_contact_density_change": contact_delta,
        "production_sampler_diversity_change": diversity_delta,
        "safeguards": safeguards,
        "safeguards_all_pass": all(safeguards.values()),
        "validity_improvement_target_fraction": float(decision["validity_improvement_target_fraction"]),
        "validity_improvement_target_met": max(bond_delta, residue_delta)
        >= float(decision["validity_improvement_target_fraction"]),
        "meaningful_target_angstrom": 0.10,
        "desirable_target_angstrom": 0.20,
        "non_authorizing_recommendation": "review checkpoint 25; no extension or downstream authorization",
    }


def run_cuda_smoke(config_path: Path, cfg: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    """Disposable real-model update smoke; separate output and no scientific publication."""
    output = Path(cfg["smoke_output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError("Phase 3I.5 smoke path must be fresh")
    staging.mkdir(parents=True)
    smoke_started = time.perf_counter()
    if not torch.cuda.is_available():
        error = RuntimeError("Phase 3I.5 CUDA lifecycle smoke requires CUDA")
        _atomic_json(staging / "failure.json", {"status": "failed_closed", "error": repr(error), **NON_AUTHORIZING})
        atomic_publish(staging, output)
        raise error
    device = torch.device("cuda")
    report: dict[str, Any] = {
        "status": "running_non_authorizing",
        "source_hashes": validation["pinned_hashes"],
        **NON_AUTHORIZING,
    }
    before_rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}
    original_params = None
    phase_telemetry: list[dict[str, Any]] = []
    try:
        source = yaml.safe_load(Path(cfg["source_config"]).read_text())
        torch.cuda.reset_peak_memory_stats(device)
        with _tracked_memory_phase("model_checkpoint_load", device, phase_telemetry):
            model, optimizer, scheduler, checkpoint = _load_model_optimizer(cfg, source, device)
        original_params = _parameter_fingerprint(model)
        initial_model = copy.deepcopy(model.state_dict())
        initial_optimizer = copy.deepcopy(optimizer.state_dict())
        initial_scheduler = copy.deepcopy(scheduler.state_dict())
        initial_scaler = torch.amp.GradScaler("cuda", enabled=False)
        restoration_hashes = {
            "model": state_hash(initial_model),
            "optimizer": state_hash(initial_optimizer),
            "scheduler": state_hash(initial_scheduler),
            "scaler": state_hash(initial_scaler.state_dict()),
            "rng": state_hash(before_rng),
        }
        clean = (
            torch.randn((1, 500, 3), device=device, generator=torch.Generator(device=device).manual_seed(24_500_500))
            * 0.1
        )
        clean = clean - clean.mean(dim=1, keepdim=True)
        input_sha256 = hashlib.sha256(clean.detach().cpu().numpy().tobytes()).hexdigest()
        mask = torch.ones((1, 500), dtype=torch.bool, device=device)
        continuity = torch.ones((1, 499), dtype=torch.bool, device=device)
        prepared = {
            "coordinates": clean,
            "residue_mask": mask,
            "chain_continuity_mask": continuity,
            "lengths": torch.tensor([500], device=device),
        }
        diffusion = CoordinateVPDiffusion(int(cfg["timesteps"]))
        is_v2 = cfg.get("version", "").endswith(("_v2", "_v3_reviewed", "_v4_reviewed", "_v5_reviewed"))
        amp_dtype = None
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        if is_v2:
            amp_candidates = ([torch.bfloat16] if torch.cuda.is_bf16_supported() else []) + [torch.float16]
            model.eval()
            with torch.inference_mode():
                reference = model(clean, torch.tensor([250], device=device), prepared["lengths"], mask, continuity)[
                    "v_prediction"
                ].float()
            for candidate in amp_candidates:
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=candidate):
                    candidate_prediction = model(
                        clean, torch.tensor([250], device=device), prepared["lengths"], mask, continuity
                    )["v_prediction"].float()
                relative_rms = float(
                    (candidate_prediction - reference).square().mean().sqrt()
                    / reference.square().mean().sqrt().clamp_min(1e-8)
                )
                tolerance = 0.10 if candidate == torch.bfloat16 else 0.02
                passed = bool(torch.isfinite(candidate_prediction).all()) and relative_rms <= tolerance
                del candidate_prediction
                gc.collect()
                torch.cuda.empty_cache()
                if passed:
                    amp_dtype = candidate
                    break
            del reference
            if amp_dtype is None:
                raise FloatingPointError("neither BF16 nor FP16 passed AMP numerical validation")
            scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
        records = {}
        forward_count = backward_count = 0
        with coordinate_model_execution_context(source["numerics"], device):
            for arm in cfg["arms"]:
                model.load_state_dict(initial_model)
                optimizer.load_state_dict(checkpoint["optimizer"])
                torch.manual_seed(24_500_500)
                torch.cuda.manual_seed_all(24_500_500)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                losses, gradients, forwards, backwards = _train_step(
                    model,
                    optimizer,
                    diffusion,
                    prepared,
                    250,
                    24_500_500,
                    source,
                    arm,
                    activation_checkpointing=is_v2,
                    amp_dtype=amp_dtype,
                    scaler=scaler,
                    phase_telemetry=phase_telemetry,
                )
                scheduler.step()
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                after = _parameter_fingerprint(model)
                if after == original_params:
                    raise RuntimeError(f"smoke optimizer did not mutate disposable model in {arm}")
                optimizer_after_hash = state_hash(optimizer.state_dict())
                if optimizer_after_hash == restoration_hashes["optimizer"]:
                    raise RuntimeError(f"smoke optimizer state did not mutate in {arm}")
                if not all(np.isfinite(value) for value in [*losses.values(), *gradients.values()]):
                    raise FloatingPointError("non-finite smoke loss or gradient")
                records[arm] = {
                    "losses": losses,
                    "gradient_norms": gradients,
                    "forward_count": forwards,
                    "optimizer_training_forward_count": forwards // 2,
                    "audit_recomputation_forward_count": forwards // 2,
                    "backward_autograd_call_count": backwards,
                    "runtime_seconds": elapsed,
                    "parameter_mutation_expected": True,
                    "parameter_hash_after": after,
                    "optimizer_mutation_expected": True,
                    "optimizer_state_hash_after": optimizer_after_hash,
                    "amp_dtype": str(amp_dtype),
                    "scaler_state": scaler.state_dict(),
                }
                forward_count += forwards
                backward_count += backwards
                with _tracked_memory_phase(f"cleanup_between_arms_{arm}", device, phase_telemetry):
                    model.load_state_dict(initial_model)
                    optimizer.load_state_dict(initial_optimizer)
                    scheduler.load_state_dict(initial_scheduler)
                    scaler.load_state_dict(
                        torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16).state_dict()
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    if _parameter_fingerprint(model) != original_params:
                        raise RuntimeError("smoke disposable model restoration failed")
                    if state_hash(optimizer.state_dict()) != restoration_hashes["optimizer"]:
                        raise RuntimeError("smoke disposable optimizer restoration failed")
                    if state_hash(scheduler.state_dict()) != restoration_hashes["scheduler"]:
                        raise RuntimeError("smoke disposable scheduler restoration failed")
            model.eval()
            profile_t = torch.tensor([250], dtype=torch.long, device=device)
            profile_lengths = torch.tensor([500], dtype=torch.long, device=device)
            torch.cuda.synchronize(device)
            profile_started = time.perf_counter()
            with torch.inference_mode():
                profile_v = model(clean, profile_t, profile_lengths, mask, continuity)["v_prediction"]
                diffusion.deterministic_reverse_step(clean, profile_t, profile_v, mask)
            torch.cuda.synchronize(device)
            profile_step_seconds = time.perf_counter() - profile_started
        torch.set_rng_state(before_rng["cpu"])
        torch.cuda.set_rng_state_all(before_rng["cuda"])
        with _tracked_memory_phase("final_cleanup", device, phase_telemetry):
            del prepared, clean
            gc.collect()
            torch.cuda.empty_cache()
        first_plateau = _memory_snapshot(device)
        with _tracked_memory_phase("final_cleanup_stability_check", device, phase_telemetry):
            gc.collect()
            torch.cuda.empty_cache()
        second_plateau = _memory_snapshot(device)
        if abs(first_plateau["current_allocated_mib"] - second_plateau["current_allocated_mib"]) > 8.0:
            raise RuntimeError("post-phase CUDA allocated memory did not reach a stable plateau")
        if (
            state_hash({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
            != restoration_hashes["rng"]
        ):
            raise RuntimeError("smoke CPU/CUDA RNG restoration failed")
        if _parameter_fingerprint(model) != original_params:
            raise RuntimeError("smoke model state restoration failed")
        if file_sha256(cfg["checkpoint"]) != cfg["checkpoint_sha256"]:
            raise RuntimeError("smoke altered the protected source checkpoint")
        memory = aggregate_direct_phase_peaks(phase_telemetry, _memory_snapshot(device))
        validate_memory_telemetry(memory, cfg["memory_limit_mib"])
        projected_seconds = (
            25 * sum(record["runtime_seconds"] for record in records.values()) + 20_060 * profile_step_seconds
        )
        report.update(
            {
                "status": "completed_non_authorizing",
                "version": cfg.get("version", VERSION),
                "device": torch.cuda.get_device_name(device),
                "device_capacity_mib": memory["device_capacity_mib"],
                "memory": memory,
                "phase_telemetry": phase_telemetry,
                "post_phase_memory_plateau_mib": {
                    "first_current_allocated_mib": first_plateau["current_allocated_mib"],
                    "second_current_allocated_mib": second_plateau["current_allocated_mib"],
                    "stable_within_mib": 8.0,
                },
                "forward_count": forward_count,
                "total_forward_count": forward_count + 1,
                "backward_autograd_call_count": backward_count,
                "profiling_forward_count": 1,
                "profiled_longest_length_forward_transition_seconds": profile_step_seconds,
                "post_smoke_empirical_runtime_projection_seconds": projected_seconds,
                "smoke_runtime_seconds": time.perf_counter() - smoke_started,
                "projection_method": (
                    "25 paired updates at measured smoke update cost plus 20,060 measured length-500 "
                    "forward-transition units; sampler panel lengths make this conservative, I/O omitted"
                ),
                "arms": records,
                "paired_inputs": True,
                "checkpointing": is_v2,
                "rng_preservation": "torch.utils.checkpoint use_reentrant=False preserve_rng_state=True",
                "state_restoration_passed": True,
                "input_sha256": input_sha256,
                "restoration_hashes": restoration_hashes,
                "scientific_output_touched": False,
                **NON_AUTHORIZING,
            }
        )
        _atomic_json(staging / "report.json", report)
        atomic_publish(staging, output)
        return report
    except Exception as exc:
        torch.set_rng_state(before_rng["cpu"])
        torch.cuda.set_rng_state_all(before_rng["cuda"])
        _atomic_json(staging / "failure.json", {"status": "failed_closed", "error": repr(exc), **NON_AUTHORIZING})
        atomic_publish(staging, output)
        raise
