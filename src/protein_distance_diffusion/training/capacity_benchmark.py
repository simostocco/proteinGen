"""Isolated, read-only E005 forward/backward capacity benchmarking."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from torch.utils.checkpoint import checkpoint

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.collate import make_sequence_separation
from protein_distance_diffusion.data.sequence_geometry import (
    PAIRING_SCHEMA_VERSION,
    SequenceGeometryDataset,
    collate_sequence_geometry,
)
from protein_distance_diffusion.diffusion.gaussian import GaussianDiffusion
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.models.codesign import (
    CONDITIONING_MODES,
    CoDesignLossWeights,
    E005SequenceGeometryCoDesign,
    codesign_losses,
)
from protein_distance_diffusion.training.codesign import (
    _configuration_sha256,
    _gate_statistics,
    _peak_rss_mib,
    _rss_mib,
    _sha256_file,
    _synthetic_items,
    masked_sequence_inputs,
)

DEFAULT_LENGTHS = (64, 128, 256, 384, 500)
DEFAULT_MODES = (
    "sequence_only",
    "learned_geometry_gating",
    "forced_geometry_conditioning",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _memory_snapshot(device: torch.device) -> dict[str, float | None]:
    return {
        "current_rss_mib": _rss_mib(),
        "peak_rss_mib": _peak_rss_mib(),
        "cuda_allocated_mib": (torch.cuda.memory_allocated(device) / (1024**2) if device.type == "cuda" else None),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
        ),
        "cuda_reserved_mib": torch.cuda.memory_reserved(device) / (1024**2) if device.type == "cuda" else None,
        "peak_cuda_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else None
        ),
    }


def _guard_memory(device: torch.device, *, max_rss_mib: int, max_cuda_memory_mib: int) -> None:
    memory = _memory_snapshot(device)
    if float(memory["current_rss_mib"] or 0) > max_rss_mib:
        raise MemoryError(
            f"RSS limit exceeded: limit={max_rss_mib} MiB, "
            f"current={memory['current_rss_mib']:.1f} MiB, peak={memory['peak_rss_mib']:.1f} MiB"
        )
    if (
        device.type == "cuda"
        and max(
            float(memory["cuda_allocated_mib"] or 0),
            float(memory["cuda_reserved_mib"] or 0),
        )
        > max_cuda_memory_mib
    ):
        raise MemoryError(
            f"CUDA memory limit exceeded: limit={max_cuda_memory_mib} MiB, "
            f"allocated={memory['cuda_allocated_mib']:.1f} MiB, reserved={memory['cuda_reserved_mib']:.1f} MiB"
        )


def _select_nearest_real_row(config: dict[str, Any], target_length: int, seed: int) -> tuple[pa.Table, Path]:
    dataset_dir = Path(config["dataset"]["directory"])
    protocol_path = dataset_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "completed" or protocol.get("schema_version") != PAIRING_SCHEMA_VERSION:
        raise ValueError("The immutable pairing dataset protocol is incomplete or incompatible")
    manifest = dataset_dir / str(config["dataset"].get("train_dataset", "eligible_train.parquet"))
    dataset = ds.dataset(str(manifest), format="parquet")
    columns = [
        "sample_id",
        "schema_version",
        "sequence",
        "sequence_length",
        "matrix_length",
        "matrix_path",
        "practical_training_eligibility",
        "__filename",
    ]
    selected: tuple[tuple[int, str], dict[str, Any]] | None = None
    scanner = dataset.scanner(columns=columns, batch_size=4096, use_threads=False)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            length = int(row["sequence_length"])
            if not 1 <= length <= 500:
                continue
            rank = hashlib.sha256(f"{seed}:{target_length}:{row['sample_id']}".encode()).hexdigest()
            key = (abs(length - target_length), rank)
            if selected is None or key < selected[0]:
                selected = (key, row)
    if selected is None:
        raise ValueError("No eligible real sample is available")
    return pa.Table.from_pylist([selected[1]]), protocol_path


def _load_case_item(
    config: dict[str, Any],
    *,
    target_length: int,
    seed: int,
    synthetic: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    if synthetic:
        return _synthetic_items([target_length], seed)[0], {}
    selected, protocol_path = _select_nearest_real_row(config, target_length, seed)
    row = selected.to_pylist()[0]
    protected_paths = {
        protocol_path,
        Path(str(row["__filename"])),
        Path(str(row["matrix_path"])),
        Path(str(config["normalization_file"])),
    }
    hashes_before = {str(path): _sha256_file(path) for path in sorted(protected_paths, key=str)}
    item = SequenceGeometryDataset(selected, mode="geometry_conditioned", seed=seed)[0]
    if {str(path): _sha256_file(path) for path in sorted(protected_paths, key=str)} != hashes_before:
        raise RuntimeError("dataset_mutation_detected_during_capacity_load")
    return item, hashes_before


def run_capacity_case(
    config: dict[str, Any],
    *,
    target_length: int,
    mode: str,
    seed: int,
    max_rss_mib: int,
    max_cuda_memory_mib: int,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Run one isolated capacity case; callers are responsible for process isolation."""
    if mode not in CONDITIONING_MODES:
        raise ValueError(f"Unsupported conditioning mode: {mode}")
    if not 1 <= target_length <= 500:
        raise ValueError("target_length must be in [1, 500]")
    if max_rss_mib > 4096:
        raise ValueError("max_rss_mib must not exceed 4096")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(str(config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("E005 capacity config requests CUDA but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    stages = [{"stage": "startup", **_memory_snapshot(device)}]
    _guard_memory(device, max_rss_mib=max_rss_mib, max_cuda_memory_mib=max_cuda_memory_mib)

    item, hashes_before = _load_case_item(
        config,
        target_length=target_length,
        seed=seed,
        synthetic=synthetic,
    )
    stages.append({"stage": "sample_loading", **_memory_snapshot(device)})
    model_config = dict(config["model"])
    geometry_config = dict(model_config.pop("geometry_model"))
    model = E005SequenceGeometryCoDesign(geometry_model=geometry_config, **model_config).to(device)
    model.train()
    stages.append({"stage": "model_construction", **_memory_snapshot(device)})
    _guard_memory(device, max_rss_mib=max_rss_mib, max_cuda_memory_mib=max_cuda_memory_mib)

    batch = collate_sequence_geometry(
        [item],
        pad_id=model.pad_token_id,
        pad_to_multiple=model.downsample_factor,
    )
    actual_length = int(batch["lengths"][0])
    clean = batch["distance_matrices"][:, None].to(device)
    if synthetic:
        normalization_scale = float(config.get("normalization_scale_angstrom", 50.0))
    else:
        normalization = json.loads(Path(config["normalization_file"]).read_text())
        if normalization.get("mode") != "scale" or float(normalization.get("scale", 0)) <= 0:
            raise ValueError("E005 capacity benchmark requires positive scale normalization")
        normalization_scale = float(normalization["scale"])
    clean = clean / normalization_scale
    lengths = batch["lengths"].to(device)
    residue_mask = batch["sequence_mask"].to(device)
    pair_mask = (residue_mask[:, None, :, None] & residue_mask[:, None, None, :]).bool()
    separation = make_sequence_separation(lengths, clean.shape[-1]).to(device)
    token_targets = batch["sequence_token_ids"].to(device)
    token_inputs, masked_tokens = masked_sequence_inputs(
        token_targets,
        residue_mask,
        mask_token_id=model.mask_token_id,
        probability=float(config.get("masked_token_probability", 0.15)),
        seed=seed,
        step=0,
    )
    diffusion_steps = int(config.get("diffusion", {}).get("steps", 500))
    diffusion = GaussianDiffusion(cosine_beta_schedule(diffusion_steps)).to(device)
    generator = torch.Generator(device=device).manual_seed(seed + 7_000_061)
    timesteps = torch.randint(0, diffusion_steps, (1,), generator=generator, device=device)
    noisy, epsilon = diffusion.q_sample(clean, timesteps, pair_mask, generator=generator)
    target = diffusion.training_target(
        x_start=clean,
        t=timesteps,
        epsilon=epsilon,
        prediction_type=str(config.get("diffusion", {}).get("prediction_parameterization", "v")),
    )
    weights_config = config.get("loss", {})
    weights = CoDesignLossWeights(
        sequence=float(weights_config.get("sequence_weight", 1.0)),
        geometry=float(weights_config.get("geometry_weight", 1.0)),
        consistency=float(weights_config.get("consistency_weight", 0.0)),
    )
    mixed_precision = bool(config.get("mixed_precision", False)) and device.type == "cuda"
    amp_dtype_name = str(config.get("amp_dtype", "float16"))
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(amp_dtype_name)
    if amp_dtype is None:
        raise ValueError("amp_dtype must be float16 or bfloat16")
    activation_checkpointing = bool(config.get("dry_run", {}).get("activation_checkpointing", True))

    def model_tensors(noisy_input: torch.Tensor):
        result = model(
            sequence_token_ids=token_inputs,
            residue_mask=residue_mask,
            noisy_geometry=noisy_input,
            timesteps=timesteps,
            lengths=lengths,
            sequence_separation=separation,
            pair_mask=pair_mask,
            geometry_conditioning_mask=torch.ones(1, dtype=torch.bool, device=device),
            mode=mode,
        )
        return (
            result["sequence_logits"],
            result["geometry_prediction"],
            result["sequence_pair_prediction"],
            result["geometry_to_sequence_gate"],
            result["sequence_to_geometry_gate"],
            result["return_geometry_gate"],
        )

    step_started = time.monotonic()
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=mixed_precision):
        outputs_tuple = (
            checkpoint(model_tensors, noisy, use_reentrant=False) if activation_checkpointing else model_tensors(noisy)
        )
        sequence_logits, geometry_prediction, sequence_pair_prediction, geometry_gate, pair_gate, return_gate = (
            outputs_tuple
        )
        outputs = {
            "sequence_logits": sequence_logits,
            "geometry_prediction": geometry_prediction,
            "sequence_pair_prediction": sequence_pair_prediction,
            "residue_mask": residue_mask,
            "pair_mask": pair_mask,
        }
        losses = codesign_losses(
            outputs,
            sequence_targets=token_targets,
            masked_token_mask=masked_tokens,
            geometry_target=target,
            weights=weights,
        )
    stages.append({"stage": "forward", **_memory_snapshot(device)})
    _guard_memory(device, max_rss_mib=max_rss_mib, max_cuda_memory_mib=max_cuda_memory_mib)
    losses["total"].backward()
    stages.append({"stage": "backward", **_memory_snapshot(device)})
    _guard_memory(device, max_rss_mib=max_rss_mib, max_cuda_memory_mib=max_cuda_memory_mib)
    loss_values = {name: float(value.detach()) for name, value in losses.items()}
    if not all(np.isfinite(value) for value in loss_values.values()):
        raise RuntimeError(f"nonfinite_capacity_loss:{loss_values}")
    hashes_after = {path: _sha256_file(Path(path)) for path in hashes_before}
    if hashes_after != hashes_before:
        raise RuntimeError("dataset_mutation_detected_during_capacity_case")
    wall_time = time.monotonic() - step_started
    memory = _memory_snapshot(device)
    return {
        "status": "passed",
        "worker_pid": os.getpid(),
        "target_length": target_length,
        "actual_length": actual_length,
        "padded_length": int(clean.shape[-1]),
        "sample_id": str(item["sample_id"]),
        "mode": mode,
        "seed": seed,
        "device": str(device),
        "batch_size": 1,
        "wall_time_seconds": wall_time,
        "total_elapsed_seconds": time.monotonic() - started,
        "samples_per_second": 1.0 / max(wall_time, 1e-12),
        "losses": loss_values,
        "gate_statistics": _gate_statistics(geometry_gate, pair_gate, return_gate, residue_mask),
        "memory": memory,
        "memory_stages": stages,
        "mixed_precision": mixed_precision,
        "activation_checkpointing": activation_checkpointing,
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "dataset_inputs_unchanged": hashes_before == hashes_after,
    }


def _failed_case(
    *,
    target_length: int,
    mode: str,
    returncode: int | None,
    error: str,
    timed_out: bool = False,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "target_length": target_length,
        "mode": mode,
        "returncode": returncode,
        "timed_out": timed_out,
        "error": error,
    }


def _recommended_schedule(cases: list[dict[str, Any]], lengths: tuple[int, ...]) -> list[dict[str, Any]]:
    schedule = []
    for length in lengths:
        length_cases = [case for case in cases if int(case["target_length"]) == length]
        failed_modes = [str(case["mode"]) for case in length_cases if case["status"] != "passed"]
        schedule.append(
            {
                "target_length": length,
                "recommended_batch_size": 1 if not failed_modes else 0,
                "recommendation": "bounded_pilot_batch_1" if not failed_modes else "defer",
                "failed_modes": failed_modes,
                "interpretation": (
                    "All three isolated batch-1 cases passed; no evidence supports a larger batch."
                    if not failed_modes
                    else "At least one conditioning mode failed; exclude this length from the first pilot."
                ),
            }
        )
    return schedule


def run_capacity_benchmark(
    *,
    config_path: str | Path,
    report_path: str | Path,
    lengths: tuple[int, ...] = DEFAULT_LENGTHS,
    modes: tuple[str, ...] = DEFAULT_MODES,
    seed: int = 5005,
    max_rss_mib: int = 4096,
    max_cuda_memory_mib: int = 8192,
    timeout_seconds: int = 900,
    synthetic: bool = False,
    script_path: str | Path | None = None,
) -> dict[str, Any]:
    """Coordinate isolated cases and atomically publish one aggregate report."""
    config_path = Path(config_path)
    config = load_yaml(config_path)
    if max_rss_mib > 4096:
        raise ValueError("max_rss_mib must not exceed 4096")
    invalid_modes = sorted(set(modes) - CONDITIONING_MODES)
    if invalid_modes:
        raise ValueError(f"Unsupported conditioning modes: {invalid_modes}")
    script = (
        Path(script_path) if script_path else Path(__file__).resolve().parents[3] / "scripts/benchmark_e005_capacity.py"
    )
    cases = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="e005-capacity-") as temporary_directory:
        for target_length in lengths:
            for mode in modes:
                case_path = Path(temporary_directory) / f"length-{target_length}-{mode}.json"
                command = [
                    sys.executable,
                    str(script),
                    "--config",
                    str(config_path),
                    "--worker-case",
                    "--worker-report",
                    str(case_path),
                    "--target-length",
                    str(target_length),
                    "--mode",
                    mode,
                    "--seed",
                    str(seed),
                    "--max-rss-mib",
                    str(max_rss_mib),
                    "--max-cuda-memory-mib",
                    str(max_cuda_memory_mib),
                ]
                if synthetic:
                    command.append("--synthetic")
                environment = os.environ.copy()
                source_root = str(Path(__file__).resolve().parents[2])
                environment["PYTHONPATH"] = os.pathsep.join(
                    part for part in (source_root, environment.get("PYTHONPATH", "")) if part
                )
                try:
                    completed = subprocess.run(
                        command,
                        cwd=Path(__file__).resolve().parents[3],
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=timeout_seconds,
                        check=False,
                    )
                    if case_path.is_file():
                        case = json.loads(case_path.read_text())
                        case["returncode"] = completed.returncode
                    else:
                        case = _failed_case(
                            target_length=target_length,
                            mode=mode,
                            returncode=completed.returncode,
                            error=(completed.stderr or completed.stdout or "worker produced no report")[-4000:],
                        )
                except subprocess.TimeoutExpired as error:
                    case = _failed_case(
                        target_length=target_length,
                        mode=mode,
                        returncode=None,
                        error=f"case timed out after {timeout_seconds} seconds: {error}",
                        timed_out=True,
                    )
                cases.append(case)
    report = {
        "status": "completed",
        "benchmark": "e005_read_only_capacity_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "config_sha256": _configuration_sha256(config),
        "synthetic": synthetic,
        "seed": seed,
        "batch_size": 1,
        "lengths": list(lengths),
        "modes": list(modes),
        "limits": {
            "max_rss_mib": max_rss_mib,
            "max_cuda_memory_mib": max_cuda_memory_mib,
            "case_timeout_seconds": timeout_seconds,
        },
        "case_count": len(cases),
        "passed_case_count": sum(case["status"] == "passed" for case in cases),
        "failed_case_count": sum(case["status"] != "passed" for case in cases),
        "cases": cases,
        "recommended_pilot_schedule": _recommended_schedule(cases, lengths),
        "wall_time_seconds": time.monotonic() - started,
        "persistent_checkpoint_written": False,
    }
    _atomic_json(Path(report_path), report)
    return report


def write_failed_worker_report(
    path: str | Path,
    *,
    target_length: int,
    mode: str,
    error: BaseException,
) -> None:
    """Publish a bounded worker failure for parent-process collection."""
    _atomic_json(
        Path(path),
        {
            **_failed_case(
                target_length=target_length,
                mode=mode,
                returncode=1,
                error=f"{type(error).__name__}: {error}",
            ),
            "current_rss_mib": _rss_mib(),
            "peak_rss_mib": _peak_rss_mib(),
            "worker_pid": os.getpid(),
        },
    )
