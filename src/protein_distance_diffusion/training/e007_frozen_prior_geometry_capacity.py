"""Bounded E007 Phase 4C.1 frozen-ProGen2 conditioner-capacity experiment."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
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
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml

from protein_distance_diffusion.data.e007_coordinate_dataset import (
    coordinate_acceptance_reasons,
    stable_center_valid_coordinates,
)
from protein_distance_diffusion.data.rich_geometry import (
    RichGeometryDataset,
    authorize_rich_geometry_dataset,
)
from protein_distance_diffusion.evaluation.e007_pretrained_loaders import CANONICAL
from protein_distance_diffusion.evaluation.e007_pretrained_loaders_v3 import (
    load_reviewed_candidate,
    parameter_identity_hash,
    progen_framed_sequence_contract,
)
from protein_distance_diffusion.models.e007_frozen_prior_geometry import (
    InvariantGeometryConditioner,
)
from protein_distance_diffusion.models.e007_frozen_prior_geometry_capacity import (
    CAPACITY_CONDITIONER_VERSION,
    MediumInvariantGeometryConditioner,
    build_capacity_conditioner,
    trainable_parameter_count,
)

VERSION = "e007_frozen_prior_geometry_capacity_v1"
CANDIDATE = "progen2_151m"
CAPACITIES = ("small", "medium")
ARMS = ("correct_geometry", "shuffled_geometry", "null_geometry")
SMALL_PARAMETER_COUNT = 337_921
MEDIUM_PARAMETER_COUNT = 3_608_999
NON_AUTHORIZING = {
    "authorizes_training": False,
    "authorizes_production_training": False,
    "authorizes_joint_training": False,
    "authorizes_additional_coordinate_training": False,
    "authorizes_sequence_conditioned_coordinate_generation": False,
    "authorizes_progen2_unfreezing": False,
    "authorizes_prior_selection": False,
}


class WorkerFailure(RuntimeError):
    def __init__(self, capacity: str, arm: str, result: dict[str, Any]) -> None:
        self.capacity = capacity
        self.arm = arm
        self.result = result
        super().__init__(f"E007 Phase 4C.1 worker failed: {capacity}/{arm}: {result}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("E007 Phase 4C.1 configuration version contradiction")
    if payload.get("candidate") != CANDIDATE:
        raise ValueError("E007 Phase 4C.1 is ProGen2-only")
    if payload.get("capacities") != list(CAPACITIES) or payload.get("arms") != list(ARMS):
        raise ValueError("E007 Phase 4C.1 capacity/arm contract changed")
    pilot = payload["pilot"]
    updates = int(pilot["successful_optimizer_updates"])
    if not 1 <= updates <= 1000:
        raise ValueError("E007 Phase 4C.1 pilot exceeds its 1,000-update bound")
    required = [0, 100, 250, 500, 750, 1000]
    if pilot["evaluation_updates"] != [value for value in required if value <= updates]:
        raise ValueError("E007 Phase 4C.1 evaluation schedule changed")
    if payload["smoke"].get("optimizer_updates_performed_by_smoke") != 0:
        raise ValueError("E007 Phase 4C.1 smoke must perform zero optimizer updates")
    for mode in ("smoke", "pilot"):
        for key in ("train_panel_size", "validation_panel_size"):
            if int(payload[mode][key]) % (2 * len(payload["length_strata"])):
                raise ValueError("E007 Phase 4C.1 panels require an even allocation per length stratum")
    counts = analytical_parameter_counts(payload)
    if counts != {"small": SMALL_PARAMETER_COUNT, "medium": MEDIUM_PARAMETER_COUNT}:
        raise ValueError(f"E007 Phase 4C.1 parameter-count contract changed: {counts}")
    if not 3_000_000 <= counts["medium"] <= 4_000_000:
        raise ValueError("E007 Phase 4C.1 medium capacity is outside its declared range")
    return payload


def analytical_parameter_counts(config: dict[str, Any]) -> dict[str, int]:
    small = config["conditioners"]["small"]
    bins = int(small["rbf_bins"])
    hidden = int(small["hidden_width"])
    output = int(small["shared_output_width"])
    small_count = bins * hidden + hidden + 2 * hidden + hidden * hidden + hidden + hidden * output + output + 1

    medium = config["conditioners"]["medium"]
    rbf = int(medium["rbf_bins"])
    separation_bins = int(medium["separation_bins"])
    separation_width = int(medium["separation_width"])
    pair = int(medium["pair_width"])
    residue = int(medium["residue_width"])
    shared = int(medium["shared_output_width"])
    blocks = int(medium["message_blocks"])
    medium_count = (
        separation_bins * separation_width
        + (rbf + separation_width + 1) * pair
        + pair
        + 2 * pair
        + pair * residue
        + residue
        + 2 * residue
        + blocks * (pair * residue + 3 * residue * residue + 7 * residue + 1)
        + residue * shared
        + shared * shared
        + 2 * shared
        + 3
    )
    return {"small": small_count, "medium": medium_count}


def _verify_file(path: str | Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"E007 Phase 4C.1 prerequisite hash contradiction: {label}")
    return observed


def verify_prerequisites(config: dict[str, Any]) -> dict[str, str]:
    observed = {
        name: _verify_file(record["path"], record["sha256"], name) for name, record in config["prerequisites"].items()
    }
    phase4c = json.loads(Path(config["prerequisites"]["phase4c_smoke_report"]["path"]).read_text())
    if (
        phase4c.get("status") != "completed_non_authorizing"
        or phase4c.get("mode") != "smoke"
        or len(phase4c.get("results", [])) != 6
        or any(phase4c.get(key) is not False for key in NON_AUTHORIZING if key in phase4c)
    ):
        raise ValueError("E007 Phase 4C.1 requires the completed non-authorizing Phase 4C smoke")
    return observed


def plan(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    protected = verify_prerequisites(config)
    for key in ("smoke_output_dir", "pilot_output_dir"):
        output = Path(config[key])
        staging = output.with_name(f".{output.name}.inprogress")
        if output.exists() or staging.exists():
            raise FileExistsError(f"E007 Phase 4C.1 output already exists: {output}")
    return {
        "status": "planned_non_authorizing",
        "version": VERSION,
        "configuration_sha256": sha256_file(config_path),
        "candidate": CANDIDATE,
        "capacities": list(CAPACITIES),
        "arms": list(ARMS),
        "isolated_process_count": len(CAPACITIES) * len(ARMS),
        "parameter_counts": analytical_parameter_counts(config),
        "medium_injection_depths": config["conditioners"]["medium"]["injection_depths"],
        "maximum_updates_per_arm": int(config["pilot"]["successful_optimizer_updates"]),
        "evaluation_updates": config["pilot"]["evaluation_updates"],
        "exact_biological_length_derangement_required": True,
        "conditioner_version": CAPACITY_CONDITIONER_VERSION,
        "protected_hashes": protected,
        "dataset_scanned": False,
        "model_created": False,
        "cuda_initialized": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "output_created": False,
        **NON_AUTHORIZING,
    }


def length_stratum(length: int, strata: list[dict[str, Any]]) -> str:
    matches = [str(row["name"]) for row in strata if int(row["minimum"]) <= length <= int(row["maximum"])]
    if len(matches) != 1:
        raise ValueError(f"E007 Phase 4C.1 length has no unique stratum: {length}")
    return matches[0]


def exact_length_derangement(rows: list[dict[str, Any]], *, seed: int) -> dict[str, str]:
    groups: dict[int, list[str]] = defaultdict(list)
    rigid_hashes = {}
    coordinate_hashes = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        groups[len(str(row["sequence"]))].append(sample_id)
        rigid_hashes[sample_id] = row.get("coordinate_rigid_shape_sha256")
        coordinate_hashes[sample_id] = row.get("coordinate_sha256")
    result: dict[str, str] = {}
    for length, members in sorted(groups.items()):
        members.sort(key=lambda value: hashlib.sha256(f"{seed}|{length}|{value}".encode()).digest())
        if len(members) < 2:
            raise ValueError(f"E007 Phase 4C.1 exact-length donor unavailable: length={length} count={len(members)}")
        assignment = None
        for offset in range(1, len(members)):
            candidate = {member: members[(index + offset) % len(members)] for index, member in enumerate(members)}
            if all(
                (
                    rigid_hashes[source] is None
                    or rigid_hashes[donor] is None
                    or rigid_hashes[source] != rigid_hashes[donor]
                )
                and (
                    coordinate_hashes[source] is None
                    or coordinate_hashes[donor] is None
                    or coordinate_hashes[source] != coordinate_hashes[donor]
                )
                for source, donor in candidate.items()
            ):
                assignment = candidate
                break
        if assignment is None:
            raise ValueError(f"E007 Phase 4C.1 distinct-shape donor unavailable: length={length}")
        result.update(assignment)
    if set(result) != set(result.values()) or any(source == donor for source, donor in result.items()):
        raise ValueError("E007 Phase 4C.1 donor mapping is not a fixed-point-free bijection")
    return result


def deterministic_exact_length_selection(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    strata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select an equal, pairable allocation per stratum from projected candidates."""
    per_stratum = count // len(strata)
    if count % len(strata) or per_stratum % 2:
        raise ValueError("E007 Phase 4C.1 exact-length selection requires even per-stratum allocation")
    by_stratum_length: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        length = len(str(row["sequence"]))
        by_stratum_length[length_stratum(length, strata)][length].append(row)
    selected = []
    for stratum in strata:
        name = str(stratum["name"])
        groups = []
        for length, members in by_stratum_length[name].items():
            ordered = sorted(
                members,
                key=lambda row: hashlib.sha256(f"{seed}|sample|{row['sample_id']}".encode()).digest(),
            )
            if len(ordered) >= 2:
                rank = hashlib.sha256(f"{seed}|length|{name}|{length}".encode()).digest()
                groups.append((rank, length, ordered))
        groups.sort(key=lambda item: (item[0], item[1]))
        chosen = []
        while len(chosen) < per_stratum:
            progress = False
            for _rank, _length, members in groups:
                if len(members) >= 2 and len(chosen) + 2 <= per_stratum:
                    chosen.extend((members.pop(0), members.pop(0)))
                    progress = True
                if len(chosen) == per_stratum:
                    break
            if not progress:
                break
        if len(chosen) != per_stratum:
            available = sum(len(members) for _rank, _length, members in groups) + len(chosen)
            raise ValueError(
                "E007 Phase 4C.1 exact-length panel underfill: "
                f"stratum={name} required={per_stratum} pairable_candidates={available}"
            )
        selected.extend(chosen)
    if len({str(row["sample_id"]) for row in selected}) != count:
        raise ValueError("E007 Phase 4C.1 panel sample IDs are not unique")
    exact_length_derangement(selected, seed=seed + 101)
    return selected


def _project_row(row: dict[str, Any]) -> dict[str, Any]:
    sequence = str(row["sequence"])
    coordinates = torch.as_tensor(row["ca_coordinates"], dtype=torch.float32)
    mask = torch.as_tensor(row["ca_mask"], dtype=torch.bool)
    continuity = torch.as_tensor(row["chain_continuity_mask"], dtype=torch.bool)
    reasons = coordinate_acceptance_reasons(
        sequence_length=len(sequence),
        ca_mask=row["ca_mask"],
        chain_continuity_mask=row["chain_continuity_mask"],
        chain_break_mask=row["chain_break_mask"],
    )
    if reasons or coordinates.shape != (len(sequence), 3) or not bool(torch.isfinite(coordinates[mask]).all()):
        raise ValueError(f"E007 Phase 4C.1 row violates coordinate acceptance: {reasons}")
    coordinates = stable_center_valid_coordinates(coordinates, mask) / 12.22820347644835
    distances = torch.cdist(coordinates.double(), coordinates.double())
    rigid_hash = hashlib.sha256(torch.round(distances * 1e4).to(torch.int64).contiguous().numpy().tobytes()).hexdigest()
    coordinate_hash = hashlib.sha256(coordinates.contiguous().numpy().tobytes()).hexdigest()
    return {
        "sample_id": str(row["sample_id"]),
        "split": str(row["split"]),
        "sequence": sequence,
        "coordinates": coordinates,
        "residue_mask": mask,
        "continuity_mask": continuity,
        "coordinate_sha256": coordinate_hash,
        "coordinate_rigid_shape_sha256": rigid_hash,
    }


def _authorize(config: dict[str, Any]):
    dataset = config["dataset"]
    return authorize_rich_geometry_dataset(
        dataset["root"],
        expected_protocol_sha256=dataset["protocol_sha256"],
        expected_schema_sha256=dataset["schema_sha256"],
        expected_vocabulary_sha256=dataset["vocabulary_sha256"],
        expected_normalization_sha256=dataset["normalization_sha256"],
        expected_shard_inventory_sha256=dataset["shard_inventory_sha256"],
        protected_input_relocations=dataset.get("protected_input_relocations"),
    )


def _clean_validation_ids(config: dict[str, Any]) -> set[str]:
    row = config["clean_validation"]
    _verify_file(row["manifest_path"], row["manifest_sha256"], "clean_validation")
    table = pq.read_table(row["manifest_path"], columns=["sample_id", "split", "coordinate_accepted"])
    return {
        str(value["sample_id"])
        for value in table.to_pylist()
        if value["split"] == "validation" and value["coordinate_accepted"] is True
    }


def select_panel(config: dict[str, Any], *, split: str, count: int, seed: int) -> list[dict[str, Any]]:
    authorization = _authorize(config)
    dataset = RichGeometryDataset(authorization, split=split)
    permitted = _clean_validation_ids(config) if split == "validation" else None
    per_length_capacity = max(8, 2 * count // len(config["length_strata"]))
    locators: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id, length in dataset.iter_metadata():
        if permitted is not None and sample_id not in permitted:
            continue
        rank = int.from_bytes(hashlib.sha256(f"{seed}|{split}|{sample_id}".encode()).digest(), "big")
        item = (-rank, -index)
        heap = locators[int(length)]
        if len(heap) < per_length_capacity:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    projected = []
    for length in sorted(locators):
        for _rank, index in sorted([(-rank, -index) for rank, index in locators[length]]):
            try:
                projected.append(_project_row(dataset[index]))
            except ValueError:
                continue
    try:
        return deterministic_exact_length_selection(
            projected,
            count=count,
            seed=seed,
            strata=config["length_strata"],
        )
    except ValueError as error:
        raise ValueError(
            "E007 Phase 4C.1 panel construction failed before model construction: "
            f"candidate={CANDIDATE} arm=shuffled_geometry split={split} count={count}: {error}"
        ) from error


def _capacity_seed(config: dict[str, Any], capacity: str) -> int:
    namespace = str(config["capacity_initialization_namespaces"][capacity])
    return int.from_bytes(hashlib.sha256(f"{config['seed']}|{namespace}".encode()).digest()[:8], "big") % (2**31)


def initialize_conditioner(capacity: str, config: dict[str, Any], *, device: torch.device) -> torch.nn.Module:
    seed = _capacity_seed(config, capacity)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_capacity_conditioner(capacity, config["conditioners"][capacity]).to(device)
    expected = analytical_parameter_counts(config)[capacity]
    observed = trainable_parameter_count(model)
    if observed != expected:
        raise ValueError(f"E007 Phase 4C.1 instantiated parameter count contradiction: {observed} != {expected}")
    return model


def parameter_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _framed_geometry(
    conditioning: torch.Tensor,
    biological_length: int,
    *,
    framed_length: int | None = None,
) -> torch.Tensor:
    if conditioning.shape[1] != biological_length:
        raise ValueError("E007 Phase 4C.1 geometry/token length contradiction")
    minimum_length = biological_length + 2
    framed_length = minimum_length if framed_length is None else framed_length
    if framed_length < minimum_length:
        raise ValueError("E007 Phase 4C.1 framed token length is too short")
    framed = conditioning.new_zeros((conditioning.shape[0], framed_length, conditioning.shape[-1]))
    framed[:, 1 : biological_length + 1] = conditioning
    return framed


def _canonical_targets(tokenizer: Any, sequence: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    framed = progen_framed_sequence_contract(tokenizer, sequence, position_capacity=1024)
    ids = torch.tensor(framed["framed_model_input"]["ids"], dtype=torch.long, device=device)[None]
    canonical_ids = [tokenizer.token_to_id(residue) for residue in CANONICAL]
    remap = {value: index for index, value in enumerate(canonical_ids)}
    targets = torch.tensor([remap[value] for value in framed["framed_model_input"]["ids"][1:-1]], device=device)
    return ids, targets


def _medium_hooks(prior: Any, conditioner: MediumInvariantGeometryConditioner, framed: torch.Tensor) -> ExitStack:
    blocks = prior.transformer.h
    if max(conditioner.injection_depths) >= len(blocks):
        raise ValueError("E007 Phase 4C.1 ProGen injection depth exceeds transformer depth")
    stack = ExitStack()
    for injection_index, depth in enumerate(conditioner.injection_depths):
        addition = conditioner.gated_conditioning(framed, injection_index)

        def inject(_module: Any, arguments: tuple[Any, ...], value: torch.Tensor = addition):
            return (arguments[0] + value.to(arguments[0].dtype), *arguments[1:])

        stack.callback(blocks[depth].register_forward_pre_hook(inject).remove)
    return stack


def conditioned_progen_loss(
    prior: Any,
    tokenizer: Any,
    conditioner: torch.nn.Module,
    capacity: str,
    row: dict[str, Any],
    *,
    geometry_row: dict[str, Any],
    arm: str,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    ids, targets = _canonical_targets(tokenizer, str(row["sequence"]), device)
    coordinates = geometry_row["coordinates"][None].to(device)
    residue_mask = geometry_row["residue_mask"][None].to(device)
    null_geometry = arm == "null_geometry"
    embeddings = prior.get_input_embeddings()(ids).detach()
    if capacity == "small":
        if not isinstance(conditioner, InvariantGeometryConditioner):
            raise TypeError("E007 Phase 4C.1 small conditioner type contradiction")
        biological = conditioner(coordinates, residue_mask, null_geometry=null_geometry)
        framed = _framed_geometry(conditioner.for_prior_width(biological, prior.config.n_embd), len(row["sequence"]))
        logits = prior(inputs_embeds=embeddings + framed, attention_mask=torch.ones_like(ids)).logits[0]
        gates = [float(torch.sigmoid(conditioner.gate_logit.detach()))]
    else:
        if not isinstance(conditioner, MediumInvariantGeometryConditioner):
            raise TypeError("E007 Phase 4C.1 medium conditioner type contradiction")
        continuity = geometry_row["continuity_mask"][None].to(device)
        biological = conditioner(
            coordinates,
            residue_mask,
            continuity,
            null_geometry=null_geometry,
        )
        framed = _framed_geometry(biological, len(row["sequence"]))
        with _medium_hooks(prior, conditioner, framed):
            logits = prior(inputs_embeds=embeddings, attention_mask=torch.ones_like(ids)).logits[0]
        gates = conditioner.gate_values()
    canonical_ids = [tokenizer.token_to_id(residue) for residue in CANONICAL]
    canonical_logits = logits[: len(row["sequence"]), canonical_ids]
    loss = F.cross_entropy(canonical_logits.float(), targets)
    prediction = canonical_logits.argmax(dim=-1)
    top5 = canonical_logits.topk(5, dim=-1).indices.eq(targets[:, None]).any(dim=-1)
    predicted_composition = torch.bincount(prediction, minlength=len(CANONICAL)).float()
    target_composition = torch.bincount(targets, minlength=len(CANONICAL)).float()
    predicted_composition /= predicted_composition.sum().clamp_min(1)
    target_composition /= target_composition.sum().clamp_min(1)
    return loss, {
        "token_count": int(targets.numel()),
        "cross_entropy": float(loss.detach()),
        "perplexity": float(torch.exp(loss.detach())),
        "negative_log_likelihood_per_residue": float(loss.detach()),
        "top1_accuracy": float((prediction == targets).float().mean().detach()),
        "top5_accuracy": float(top5.float().mean().detach()),
        "canonical_validity": 1.0,
        "amino_acid_composition_total_variation": float(
            (predicted_composition - target_composition).abs().sum().mul(0.5).detach()
        ),
        "gate_values": gates,
        "bos_geometry_norm": float(framed[:, 0].norm().detach()),
        "eos_geometry_norm": float(framed[:, -1].norm().detach()),
    }


def gradient_coverage(conditioner: torch.nn.Module, capacity: str) -> dict[str, bool]:
    if capacity == "small":
        groups = {
            "pair_encoder": conditioner.pair_encoder,
            "conditioning_adapter": conditioner.conditioning_adapter,
        }
        result = {
            name: all(
                parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                for parameter in module.parameters()
            )
            for name, module in groups.items()
        }
        result["input_gate"] = conditioner.gate_logit.grad is not None and bool(
            torch.isfinite(conditioner.gate_logit.grad).all()
        )
        return result
    groups = {
        "separation_embedding": conditioner.separation_embedding,
        "pair_encoder": conditioner.pair_encoder,
        "residue_input": conditioner.residue_input,
        "output_adapter": conditioner.output_adapter,
        **{f"message_block_{index}": block for index, block in enumerate(conditioner.message_blocks)},
    }
    result = {
        name: all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in module.parameters()
        )
        for name, module in groups.items()
    }
    gate_gradient = conditioner.injection_gate_logits.grad
    for index in range(len(conditioner.injection_gate_logits)):
        result[f"injection_gate_{index}"] = gate_gradient is not None and bool(torch.isfinite(gate_gradient[index]))
    return result


def gradient_norms(conditioner: torch.nn.Module, capacity: str) -> dict[str, float]:
    if capacity == "small":
        groups = {
            "pair_encoder": list(conditioner.pair_encoder.parameters()),
            "conditioning_adapter": list(conditioner.conditioning_adapter.parameters()),
            "input_gate": [conditioner.gate_logit],
        }
    else:
        groups = {
            "separation_embedding": list(conditioner.separation_embedding.parameters()),
            "pair_encoder": list(conditioner.pair_encoder.parameters()),
            "residue_input": list(conditioner.residue_input.parameters()),
            "output_adapter": list(conditioner.output_adapter.parameters()),
            **{
                f"message_block_{index}": list(block.parameters())
                for index, block in enumerate(conditioner.message_blocks)
            },
            **{
                f"injection_gate_{index}": [conditioner.injection_gate_logits]
                for index in range(len(conditioner.injection_gate_logits))
            },
        }
    result = {}
    for name, parameters in groups.items():
        if name.startswith("injection_gate_"):
            index = int(name.rsplit("_", 1)[1])
            gradient = parameters[0].grad
            result[name] = float(abs(gradient[index])) if gradient is not None else float("nan")
            continue
        squared = sum(
            float(parameter.grad.detach().float().square().sum())
            for parameter in parameters
            if parameter.grad is not None
        )
        result[name] = math.sqrt(squared)
    return result


def _memory(device: torch.device) -> dict[str, float | None]:
    current_rss = 0.0
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            current_rss = float(line.split()[1]) / 1024
            break
    return {
        "current_rss_mib": current_rss,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "current_cuda_allocated_mib": torch.cuda.memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "current_cuda_reserved_mib": torch.cuda.memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
    }


def enforce_memory_limits(memory: dict[str, float | None], limits: dict[str, float]) -> None:
    for key, limit_key in (
        ("peak_rss_mib", "maximum_rss_mib"),
        ("peak_cuda_allocated_mib", "maximum_cuda_allocated_mib"),
        ("peak_cuda_reserved_mib", "maximum_cuda_reserved_mib"),
    ):
        value = memory.get(key)
        limit = float(limits[limit_key])
        if value is not None and (not math.isfinite(value) or value > limit):
            raise MemoryError(f"E007 Phase 4C.1 memory limit exceeded: {key}={value} limit={limit}")


def _evaluate(
    prior: Any,
    tokenizer: Any,
    conditioner: torch.nn.Module,
    capacity: str,
    arm: str,
    rows: list[dict[str, Any]],
    donors: dict[str, str],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    by_id = {row["sample_id"]: row for row in rows}
    records = []
    conditioner.eval()
    with torch.no_grad():
        for row in rows:
            geometry = by_id[donors[row["sample_id"]]] if arm == "shuffled_geometry" else row
            loss, metrics = conditioned_progen_loss(
                prior,
                tokenizer,
                conditioner,
                capacity,
                row,
                geometry_row=geometry,
                arm=arm,
                device=device,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("E007 Phase 4C.1 evaluation loss is non-finite")
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "length": len(row["sequence"]),
                    "length_stratum": length_stratum(len(row["sequence"]), config["length_strata"]),
                    **metrics,
                }
            )
    values = [row["cross_entropy"] for row in records]
    return {
        "records": records,
        "macro_cross_entropy": float(np.mean(values)),
        "macro_perplexity": float(math.exp(np.mean(values))),
        "token_weighted_cross_entropy": float(np.average(values, weights=[row["token_count"] for row in records])),
        "macro_top1_accuracy": float(np.mean([row["top1_accuracy"] for row in records])),
        "macro_top5_accuracy": float(np.mean([row["top5_accuracy"] for row in records])),
        "macro_composition_total_variation": float(
            np.mean([row["amino_acid_composition_total_variation"] for row in records])
        ),
        "by_length_stratum": {
            stratum["name"]: {
                "sample_count": sum(row["length_stratum"] == stratum["name"] for row in records),
                "macro_cross_entropy": float(
                    np.mean([row["cross_entropy"] for row in records if row["length_stratum"] == stratum["name"]])
                ),
            }
            for stratum in config["length_strata"]
        },
    }


def checkpoint_payload(
    conditioner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    capacity: str,
    arm: str,
    update: int,
    cursor: int,
    evaluations: dict[str, Any],
    config_sha256: str,
    panel_hashes: dict[str, str],
    donor_mapping_sha256: str | None,
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "configuration_sha256": config_sha256,
        "capacity": capacity,
        "arm": arm,
        "conditioner": conditioner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "update": update,
        "cursor": cursor,
        "evaluations": evaluations,
        "completed_evaluation_updates": sorted(int(value) for value in evaluations),
        "panel_hashes": panel_hashes,
        "donor_mapping_sha256": donor_mapping_sha256,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        **NON_AUTHORIZING,
    }


def restore_checkpoint(
    path: Path,
    conditioner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    capacity: str,
    arm: str,
    config_sha256: str,
    panel_hashes: dict[str, str],
    donor_mapping_sha256: str | None,
) -> tuple[int, int, dict[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "version": VERSION,
        "configuration_sha256": config_sha256,
        "capacity": capacity,
        "arm": arm,
        "panel_hashes": panel_hashes,
        "donor_mapping_sha256": donor_mapping_sha256,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("E007 Phase 4C.1 checkpoint contract contradiction")
    conditioner.load_state_dict(value["conditioner"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    scheduler.load_state_dict(value["scheduler"])
    random.setstate(value["python_rng_state"])
    np.random.set_state(value["numpy_rng_state"])
    torch.set_rng_state(value["torch_rng_state"].cpu())
    if value["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in value["cuda_rng_state"]])
    return int(value["update"]), int(value["cursor"]), value["evaluations"]


def _panel_hash(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(row["sample_id"] for row in rows).encode()).hexdigest()


def _worker(
    config_path: str,
    capacity: str,
    arm: str,
    mode: str,
    result_path_value: str,
    checkpoint_directory_value: str,
    resume_path_value: str | None,
) -> None:
    result_path = Path(result_path_value)
    checkpoint_directory = Path(checkpoint_directory_value)
    execution = {
        "model_created": False,
        "forward_performed": False,
        "backward_performed": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "checkpoint_written": False,
        "parameter_mutation_checked": False,
    }
    stage = "startup"
    try:
        config = _load_config(config_path)
        verify_prerequisites(config)
        seed = int(config["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        stage = "panel_selection"
        train_rows = select_panel(config, split="train", count=int(config[mode]["train_panel_size"]), seed=seed)
        validation_rows = select_panel(
            config,
            split="validation",
            count=int(config[mode]["validation_panel_size"]),
            seed=seed + 1,
        )
        train_ids = {row["sample_id"] for row in train_rows}
        validation_ids = {row["sample_id"] for row in validation_rows}
        if train_ids & validation_ids:
            raise ValueError("E007 Phase 4C.1 train/validation panel overlap")
        train_donors = exact_length_derangement(train_rows, seed=seed + 2)
        validation_donors = exact_length_derangement(validation_rows, seed=seed + 3)
        donor_hash = canonical_sha256({"train": train_donors, "validation": validation_donors})
        panel_hashes = {"train": _panel_hash(train_rows), "validation": _panel_hash(validation_rows)}

        stage = "model_construction"
        device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        prior, tokenizer, loader = load_reviewed_candidate(CANDIDATE, config["artifact_cache_root"], device=device)
        prior.eval()
        for parameter in prior.parameters():
            parameter.requires_grad_(False)
        prior_before = parameter_identity_hash(prior)
        conditioner = initialize_conditioner(capacity, config, device=device)
        initial_conditioner_hash = parameter_hash(conditioner)
        execution["model_created"] = True
        optimizer = torch.optim.AdamW(conditioner.parameters(), **config["optimizer"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["pilot"]["successful_optimizer_updates"]),
            eta_min=float(config["scheduler"]["minimum_lr"]),
        )
        execution["optimizer_created"] = True
        update = cursor = 0
        evaluations: dict[str, Any] = {}
        if resume_path_value:
            update, cursor, evaluations = restore_checkpoint(
                Path(resume_path_value),
                conditioner,
                optimizer,
                scheduler,
                capacity=capacity,
                arm=arm,
                config_sha256=sha256_file(config_path),
                panel_hashes=panel_hashes,
                donor_mapping_sha256=donor_hash,
            )
        stage = "initial_evaluation"
        if str(update) not in evaluations:
            evaluations[str(update)] = _evaluate(
                prior,
                tokenizer,
                conditioner,
                capacity,
                arm,
                validation_rows,
                validation_donors,
                config,
                device,
            )
        execution["forward_performed"] = True
        replay = _evaluate(
            prior,
            tokenizer,
            conditioner,
            capacity,
            arm,
            validation_rows,
            validation_donors,
            config,
            device,
        )
        if canonical_sha256(evaluations[str(update)]) != canonical_sha256(replay):
            raise ValueError("E007 Phase 4C.1 deterministic replay failed")

        maximum_updates = 0 if mode == "smoke" else int(config["pilot"]["successful_optimizer_updates"])
        by_id = {row["sample_id"]: row for row in train_rows}
        gradient_status = None
        latest_gradient_norms = None
        initial_gates = (
            [float(torch.sigmoid(conditioner.gate_logit.detach()))]
            if capacity == "small"
            else conditioner.gate_values()
        )
        if mode == "pilot" and update == 0:
            initial_checkpoint = checkpoint_directory / "step-0000.pt"
            if not initial_checkpoint.exists():
                payload = checkpoint_payload(
                    conditioner,
                    optimizer,
                    scheduler,
                    capacity=capacity,
                    arm=arm,
                    update=update,
                    cursor=cursor,
                    evaluations=evaluations,
                    config_sha256=sha256_file(config_path),
                    panel_hashes=panel_hashes,
                    donor_mapping_sha256=donor_hash,
                )
                atomic_checkpoint(initial_checkpoint, payload)
                atomic_checkpoint(checkpoint_directory / "latest.pt", payload)
                execution["checkpoint_written"] = True
        while update < maximum_updates:
            row = train_rows[cursor % len(train_rows)]
            geometry = by_id[train_donors[row["sample_id"]]] if arm == "shuffled_geometry" else row
            optimizer.zero_grad(set_to_none=True)
            conditioner.train()
            loss, _metrics = conditioned_progen_loss(
                prior,
                tokenizer,
                conditioner,
                capacity,
                row,
                geometry_row=geometry,
                arm=arm,
                device=device,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("E007 Phase 4C.1 training loss is non-finite")
            loss.backward()
            execution["backward_performed"] = True
            gradient_status = gradient_coverage(conditioner, capacity)
            if not all(gradient_status.values()):
                raise FloatingPointError(f"E007 Phase 4C.1 gradient coverage failed: {gradient_status}")
            latest_gradient_norms = gradient_norms(conditioner, capacity)
            torch.nn.utils.clip_grad_norm_(conditioner.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
            scheduler.step()
            update += 1
            cursor += 1
            execution["optimizer_updates"] = update
            if update in config["pilot"]["evaluation_updates"]:
                evaluations[str(update)] = _evaluate(
                    prior,
                    tokenizer,
                    conditioner,
                    capacity,
                    arm,
                    validation_rows,
                    validation_donors,
                    config,
                    device,
                )
                payload = checkpoint_payload(
                    conditioner,
                    optimizer,
                    scheduler,
                    capacity=capacity,
                    arm=arm,
                    update=update,
                    cursor=cursor,
                    evaluations=evaluations,
                    config_sha256=sha256_file(config_path),
                    panel_hashes=panel_hashes,
                    donor_mapping_sha256=donor_hash,
                )
                atomic_checkpoint(checkpoint_directory / f"step-{update:04d}.pt", payload)
                atomic_checkpoint(checkpoint_directory / "latest.pt", payload)
                execution["checkpoint_written"] = True
            elif update % int(config["pilot"]["recovery_checkpoint_frequency"]) == 0:
                atomic_checkpoint(
                    checkpoint_directory / "latest.pt",
                    checkpoint_payload(
                        conditioner,
                        optimizer,
                        scheduler,
                        capacity=capacity,
                        arm=arm,
                        update=update,
                        cursor=cursor,
                        evaluations=evaluations,
                        config_sha256=sha256_file(config_path),
                        panel_hashes=panel_hashes,
                        donor_mapping_sha256=donor_hash,
                    ),
                )
                execution["checkpoint_written"] = True
        if mode == "smoke":
            row = train_rows[0]
            geometry = by_id[train_donors[row["sample_id"]]] if arm == "shuffled_geometry" else row
            optimizer.zero_grad(set_to_none=True)
            conditioner.train()
            loss, _metrics = conditioned_progen_loss(
                prior,
                tokenizer,
                conditioner,
                capacity,
                row,
                geometry_row=geometry,
                arm=arm,
                device=device,
            )
            loss.backward()
            execution["backward_performed"] = True
            gradient_status = gradient_coverage(conditioner, capacity)
            latest_gradient_norms = gradient_norms(conditioner, capacity)
            optimizer.zero_grad(set_to_none=True)
        prior_after = parameter_identity_hash(prior)
        if prior_before != prior_after:
            raise ValueError("E007 Phase 4C.1 frozen ProGen2 parameters mutated")
        execution["parameter_mutation_checked"] = True
        memory = _memory(device)
        enforce_memory_limits(memory, config["memory"])
        final_gates = (
            [float(torch.sigmoid(conditioner.gate_logit.detach()))]
            if capacity == "small"
            else conditioner.gate_values()
        )
        atomic_json(
            result_path,
            {
                "status": "completed_non_authorizing",
                "version": VERSION,
                "candidate": CANDIDATE,
                "capacity": capacity,
                "arm": arm,
                "mode": mode,
                "optimizer_updates": update,
                "execution": execution,
                "parameter_count": trainable_parameter_count(conditioner),
                "initial_conditioner_parameter_sha256": initial_conditioner_hash,
                "final_conditioner_parameter_sha256": parameter_hash(conditioner),
                "prior_parameter_sha256_before": prior_before,
                "prior_parameter_sha256_after": prior_after,
                "pretrained_parameters_unchanged": True,
                "panel_hashes": panel_hashes,
                "donor_mapping_sha256": donor_hash,
                "gradient_coverage": gradient_status,
                "gradient_norms_by_component_and_injection_depth": latest_gradient_norms,
                "injection_depths": (
                    ["input_embedding"] if capacity == "small" else list(conditioner.injection_depths)
                ),
                "gate_values_initial": initial_gates,
                "gate_values_final": final_gates,
                "evaluations": evaluations,
                "deterministic_replay_verified": True,
                "loader": loader,
                "memory": memory,
                **NON_AUTHORIZING,
            },
        )
    except BaseException as error:
        atomic_json(
            result_path,
            {
                "status": "failed",
                "version": VERSION,
                "candidate": CANDIDATE,
                "capacity": capacity,
                "arm": arm,
                "mode": mode,
                "failure_stage": stage,
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
                "optimizer_updates": execution["optimizer_updates"],
                "execution": execution,
                **NON_AUTHORIZING,
            },
        )


def paired_bootstrap(values: list[float], *, seed: int, replicates: int) -> dict[str, float]:
    if not values or replicates < 100:
        raise ValueError("E007 Phase 4C.1 bootstrap input is insufficient")
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = [float(generator.choice(array, size=len(array), replace=True).mean()) for _ in range(replicates)]
    return {
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "fraction_improved": float(np.mean(array > 0)),
    }


def capacity_effects(results: list[dict[str, Any]], capacity: str, config: dict[str, Any]) -> dict[str, Any]:
    arms = {row["arm"]: row for row in results if row["capacity"] == capacity}
    if set(arms) != set(ARMS):
        return {"classification": "capacity_or_optimization_inconclusive", "effects": {}}
    update = str(config["pilot"]["successful_optimizer_updates"])
    correct = {row["sample_id"]: row for row in arms["correct_geometry"]["evaluations"][update]["records"]}
    effects = {}
    for control in ("null_geometry", "shuffled_geometry"):
        values = []
        by_stratum: dict[str, list[float]] = defaultdict(list)
        for row in arms[control]["evaluations"][update]["records"]:
            if row["sample_id"] not in correct:
                raise ValueError("E007 Phase 4C.1 paired evaluation identity contradiction")
            difference = row["cross_entropy"] - correct[row["sample_id"]]["cross_entropy"]
            values.append(difference)
            by_stratum[row["length_stratum"]].append(difference)
        effects[control] = {
            "sign_convention": "positive_means_correct_geometry_has_lower_cross_entropy",
            "bootstrap_95": paired_bootstrap(
                values,
                seed=int(config["evaluation"]["bootstrap_seed"]),
                replicates=int(config["evaluation"]["bootstrap_replicates"]),
            ),
            "by_length_stratum": {name: float(np.mean(rows)) for name, rows in by_stratum.items()},
        }
    supported = all(effect["bootstrap_95"]["lower_95"] > 0 for effect in effects.values())
    improved_strata = sum(
        all(effects[control]["by_length_stratum"].get(stratum["name"], 0.0) > 0 for control in effects)
        for stratum in config["length_strata"]
    )
    not_outlier = all(
        effect["bootstrap_95"]["fraction_improved"] >= float(config["decision"]["minimum_fraction_improved"])
        for effect in effects.values()
    )
    usable = supported and improved_strata >= 2 and not_outlier
    return {
        "classification": f"geometry_signal_detected_{capacity}_capacity" if usable else "no_supported_signal",
        "usable_geometry_conditioning": usable,
        "improved_length_strata": improved_strata,
        "effects": effects,
    }


def capacity_decision(comparisons: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    small = comparisons["small"]
    medium = comparisons["medium"]
    small_pass = bool(small["usable_geometry_conditioning"])
    medium_pass = bool(medium["usable_geometry_conditioning"])
    if medium_pass and not small_pass:
        classification = "medium_capacity_preferred"
    elif small_pass and not medium_pass:
        classification = "geometry_signal_detected_small_capacity"
    elif small_pass and medium_pass:
        threshold = float(config["decision"]["material_effect_difference"])
        gains = {
            control: medium["effects"][control]["bootstrap_95"]["mean"]
            - small["effects"][control]["bootstrap_95"]["mean"]
            for control in ("null_geometry", "shuffled_geometry")
        }
        classification = (
            "medium_capacity_preferred"
            if all(value >= threshold for value in gains.values())
            else "small_capacity_sufficient"
        )
    else:
        classification = "no_signal_at_tested_capacities"
        gains = {}
    return {
        "classification": classification,
        "capacity_effect_differences": gains if small_pass and medium_pass else {},
        "scalar_composite_score_used": False,
        "pareto_comparison": [
            capacity for capacity in CAPACITIES if comparisons[capacity]["usable_geometry_conditioning"]
        ],
        "scientific_review_required": True,
    }


def _run_impl(config_path: str | Path, *, mode: str, resume: bool = False) -> dict[str, Any]:
    if mode not in {"smoke", "pilot"}:
        raise ValueError("E007 Phase 4C.1 execution mode is invalid")
    config_path = Path(config_path)
    config = _load_config(config_path)
    protected_before = verify_prerequisites(config)
    output = Path(config[f"{mode}_output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or (staging.exists() and not resume):
        raise FileExistsError(f"E007 Phase 4C.1 output exists: {output}")
    staging.mkdir(parents=True, exist_ok=resume)
    journal_path = staging / "journal.json"
    journal = json.loads(journal_path.read_text()) if resume and journal_path.is_file() else {"committed_results": {}}
    atomic_json(staging / "heartbeat.json", {"status": "running", "mode": mode, **NON_AUTHORIZING})
    context = mp.get_context("spawn")
    results = []
    for capacity in CAPACITIES:
        for arm in ARMS:
            identity = f"{capacity}/{arm}"
            result_path = staging / f"{capacity}__{arm}.json"
            checkpoint_directory = staging / "checkpoints" / f"{capacity}__{arm}"
            latest = checkpoint_directory / "latest.pt"
            committed = journal["committed_results"].get(identity)
            if committed:
                if not result_path.is_file() or sha256_file(result_path) != committed:
                    raise ValueError(f"E007 Phase 4C.1 committed result hash contradiction: {identity}")
            else:
                process = context.Process(
                    target=_worker,
                    args=(
                        str(config_path),
                        capacity,
                        arm,
                        mode,
                        str(result_path),
                        str(checkpoint_directory),
                        str(latest) if resume and latest.is_file() else None,
                    ),
                )
                process.start()
                process.join()
                if process.exitcode != 0 or not result_path.is_file():
                    raise WorkerFailure(
                        capacity,
                        arm,
                        {
                            "status": "failed",
                            "error_type": "IsolatedWorkerFailure",
                            "error_message": f"exitcode={process.exitcode}, result_exists={result_path.is_file()}",
                            "optimizer_updates": 0,
                            "execution": {},
                            **NON_AUTHORIZING,
                        },
                    )
            result = json.loads(result_path.read_text())
            if result.get("status") != "completed_non_authorizing":
                raise WorkerFailure(capacity, arm, result)
            if not committed:
                journal["committed_results"][identity] = sha256_file(result_path)
                atomic_json(journal_path, journal)
            results.append(result)
            atomic_json(
                staging / "heartbeat.json",
                {
                    "status": "running",
                    "mode": mode,
                    "completed_cases": [f"{row['capacity']}/{row['arm']}" for row in results],
                    **NON_AUTHORIZING,
                },
            )
    if verify_prerequisites(config) != protected_before:
        raise ValueError("E007 Phase 4C.1 protected inputs changed")
    comparisons = (
        {capacity: capacity_effects(results, capacity, config) for capacity in CAPACITIES} if mode == "pilot" else {}
    )
    decision = capacity_decision(comparisons, config) if mode == "pilot" else None
    report = {
        "status": "completed_non_authorizing",
        "version": VERSION,
        "mode": mode,
        "candidate": CANDIDATE,
        "results": results,
        "capacity_comparisons": comparisons,
        "decision": decision,
        "comparison_policy": "within_capacity_paired_effects_then_cross_capacity_effect_comparison",
        "raw_training_loss_used_as_capacity_criterion": False,
        "scalar_composite_score_used": False,
        "protected_inputs_unchanged": True,
        **NON_AUTHORIZING,
    }
    atomic_json(staging / "report.json", report)
    atomic_json(
        staging / "protocol.json",
        {
            "status": report["status"],
            "version": VERSION,
            "mode": mode,
            "configuration_sha256": sha256_file(config_path),
            "report_sha256": sha256_file(staging / "report.json"),
            "protected_hashes": protected_before,
            "parameter_counts": analytical_parameter_counts(config),
            **NON_AUTHORIZING,
        },
    )
    atomic_json(
        staging / "heartbeat.json",
        {
            "status": "completed",
            "mode": mode,
            "optimizer_updates": sum(int(row["optimizer_updates"]) for row in results),
            "completed_utc": datetime.now(UTC).isoformat(),
            **NON_AUTHORIZING,
        },
    )
    os.replace(staging, output)
    descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"status": report["status"], "output_dir": str(output), **NON_AUTHORIZING}


def run(config_path: str | Path, *, mode: str, resume: bool = False) -> dict[str, Any]:
    try:
        return _run_impl(config_path, mode=mode, resume=resume)
    except BaseException as error:
        config = _load_config(config_path)
        output = Path(config[f"{mode}_output_dir"])
        staging = output.with_name(f".{output.name}.inprogress")
        if staging.is_dir():
            worker = error.result if isinstance(error, WorkerFailure) else {}
            execution = worker.get("execution", {})
            atomic_json(
                staging / "heartbeat.json",
                {
                    "status": "failed",
                    "mode": mode,
                    "updated_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:2000],
                    "failed_capacity": error.capacity if isinstance(error, WorkerFailure) else None,
                    "failed_arm": error.arm if isinstance(error, WorkerFailure) else None,
                    "optimizer_updates": int(worker.get("optimizer_updates", 0)),
                    "execution": execution,
                    "resumable": mode == "pilot",
                    **NON_AUTHORIZING,
                },
            )
        raise


def monitor(config_path: str | Path, *, mode: str) -> dict[str, Any]:
    config = _load_config(config_path)
    output = Path(config[f"{mode}_output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    root = output if output.is_dir() else staging
    heartbeat = root / "heartbeat.json"
    if not heartbeat.is_file():
        return {"status": "not_started", "mode": mode, "path": str(root)}
    return json.loads(heartbeat.read_text())
