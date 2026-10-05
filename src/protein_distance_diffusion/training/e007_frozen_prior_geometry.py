"""Bounded E007 Phase 4C frozen-prior geometry-conditioning pilot."""

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
    CONDITIONER_VERSION,
    InvariantGeometryConditioner,
    conditioner_parameter_count,
)

VERSION = "e007_frozen_prior_geometry_conditioning_v1"
ARMS = ("correct_geometry", "shuffled_geometry", "null_geometry")
CANDIDATES = ("esm2_150m", "progen2_151m")
NON_AUTHORIZING = {
    "authorizes_training": False,
    "authorizes_joint_training": False,
    "authorizes_production_training": False,
    "authorizes_additional_coordinate_training": False,
    "authorizes_sequence_conditioned_coordinate_generation": False,
}


class WorkerFailure(RuntimeError):
    def __init__(self, candidate: str, arm: str, result: dict[str, Any]) -> None:
        self.candidate = candidate
        self.arm = arm
        self.result = result
        super().__init__(f"E007 Phase 4C worker did not complete: {candidate}/{arm}: {result}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("E007 Phase 4C configuration version contradiction")
    if payload.get("candidates") != list(CANDIDATES) or payload.get("arms") != list(ARMS):
        raise ValueError("E007 Phase 4C candidate/arm contract changed")
    updates = int(payload["pilot"]["successful_optimizer_updates"])
    if not 1 <= updates <= 1000:
        raise ValueError("E007 Phase 4C pilot must use at most 1000 updates per arm")
    required = [0, 100, 250, 500, 750, 1000]
    if payload["pilot"]["evaluation_updates"] != [value for value in required if value <= updates]:
        raise ValueError("E007 Phase 4C evaluation schedule changed")
    if payload["smoke"].get("optimizer_updates_performed_by_smoke") != 0:
        raise ValueError("E007 Phase 4C loader smoke must perform zero optimizer updates")
    return payload


def _verify_file(path: str | Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"E007 Phase 4C prerequisite hash contradiction: {label}")
    return observed


def verify_prerequisites(config: dict[str, Any]) -> dict[str, str]:
    rows = config["prerequisites"]
    observed = {name: _verify_file(record["path"], record["sha256"], name) for name, record in rows.items()}
    report = json.loads(Path(rows["phase4b_v3_report"]["path"]).read_text())
    protocol = json.loads(Path(rows["phase4b_v3_protocol"]["path"]).read_text())
    predecessor_authorizations = (
        "authorizes_training",
        "authorizes_joint_training",
        "authorizes_production_training",
        "authorizes_additional_coordinate_training",
        "authorizes_sequence_conditioning",
    )
    if (
        report.get("status") != "completed_non_authorizing"
        or report.get("decision") != "esm2_and_progen2_advance"
        or protocol.get("status") != report["status"]
        or any(report.get(key) is not False for key in predecessor_authorizations)
    ):
        raise ValueError("E007 Phase 4C Phase 4B-v3 prerequisite is not eligible")
    return observed


def trainable_budget(config: dict[str, Any]) -> dict[str, int | bool]:
    settings = config["conditioner"]
    bins = int(settings["rbf_bins"])
    hidden = int(settings["hidden_width"])
    output = int(settings["shared_output_width"])
    count = bins * hidden + hidden + 2 * hidden + hidden * hidden + hidden + hidden * output + output + 1
    return {
        "esm2_150m": count,
        "progen2_151m": count,
        "difference": 0,
        "material_difference": False,
    }


def plan(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    protected = verify_prerequisites(config)
    for key in ("smoke_output_dir", "pilot_output_dir"):
        output = Path(config[key])
        staging = output.with_name(f".{output.name}.inprogress")
        if output.exists() or staging.exists():
            raise FileExistsError(f"E007 Phase 4C output already exists: {output}")
    budget = trainable_budget(config)
    if budget["material_difference"]:
        raise ValueError("E007 Phase 4C trainable budgets differ materially")
    return {
        "status": "planned_non_authorizing",
        "version": VERSION,
        "configuration_sha256": sha256_file(config_path),
        "candidates": list(CANDIDATES),
        "arms": list(ARMS),
        "isolated_process_count": len(CANDIDATES) * len(ARMS),
        "maximum_updates_per_arm": int(config["pilot"]["successful_optimizer_updates"]),
        "evaluation_updates": config["pilot"]["evaluation_updates"],
        "conditioner_version": CONDITIONER_VERSION,
        "geometry_representation": "masked_calpha_pair_distance_rbf_residue_mean",
        "geometry_invariance": "O(3)",
        "absolute_coordinates_exposed_to_prior": False,
        "backbone_atoms_fabricated": False,
        "trainable_parameter_budget": budget,
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
    matches = [row["name"] for row in strata if int(row["minimum"]) <= length <= int(row["maximum"])]
    if len(matches) != 1:
        raise ValueError(f"E007 Phase 4C length has no unique stratum: {length}")
    return str(matches[0])


def same_length_bin_derangement(
    rows: list[dict[str, Any]], *, seed: int, strata: list[dict[str, Any]]
) -> dict[str, str]:
    """Pair each sequence with another geometry from the same declared length bin."""
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[length_stratum(len(str(row["sequence"])), strata)].append(str(row["sample_id"]))
    result: dict[str, str] = {}
    hashes = {str(row["sample_id"]): row.get("coordinate_rigid_shape_sha256") for row in rows}
    for stratum, members in sorted(groups.items()):
        members.sort(key=lambda value: hashlib.sha256(f"{seed}|{stratum}|{value}".encode()).digest())
        if len(members) < 2:
            raise ValueError(f"E007 Phase 4C shuffled arm lacks a same-bin donor: {stratum}")
        assignments = None
        for offset in range(1, len(members)):
            candidate = {value: members[(index + offset) % len(members)] for index, value in enumerate(members)}
            if all(
                hashes[source] is None or hashes[donor] is None or hashes[source] != hashes[donor]
                for source, donor in candidate.items()
            ):
                assignments = candidate
                break
        if assignments is None:
            raise ValueError(f"E007 Phase 4C shuffled arm lacks a distinct rigid-shape derangement: {stratum}")
        result.update(assignments)
    if any(key == value for key, value in result.items()):
        raise ValueError("E007 Phase 4C shuffled geometry retained a matching sample")
    if set(result.values()) != set(result):
        raise ValueError("E007 Phase 4C shuffled geometry is not a donor bijection")
    return result


def build_arm_donor_maps(
    arm: str,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    seed: int,
    strata: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    if arm not in ARMS:
        raise ValueError(f"E007 Phase 4C unknown arm: {arm}")
    if arm != "shuffled_geometry":
        return {}, {}
    return (
        same_length_bin_derangement(train_rows, seed=seed + 2, strata=strata),
        same_length_bin_derangement(validation_rows, seed=seed + 3, strata=strata),
    )


def _project_row(row: dict[str, Any]) -> dict[str, Any]:
    sequence = str(row["sequence"])
    coordinates = torch.as_tensor(row["ca_coordinates"], dtype=torch.float32)
    mask = torch.as_tensor(row["ca_mask"], dtype=torch.bool)
    reasons = coordinate_acceptance_reasons(
        sequence_length=len(sequence),
        ca_mask=row["ca_mask"],
        chain_continuity_mask=row["chain_continuity_mask"],
        chain_break_mask=row["chain_break_mask"],
    )
    if reasons or coordinates.shape != (len(sequence), 3) or not bool(torch.isfinite(coordinates[mask]).all()):
        raise ValueError(f"E007 Phase 4C row violates coordinate acceptance: {reasons}")
    coordinates = stable_center_valid_coordinates(coordinates, mask) / 12.22820347644835
    distances = torch.cdist(coordinates.double(), coordinates.double())
    rigid_hash = hashlib.sha256(torch.round(distances * 1e4).to(torch.int64).contiguous().numpy().tobytes()).hexdigest()
    return {
        "sample_id": str(row["sample_id"]),
        "split": str(row["split"]),
        "sequence": sequence,
        "coordinates": coordinates,
        "residue_mask": mask,
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
    )


def _clean_validation_ids(config: dict[str, Any]) -> set[str]:
    path = Path(config["clean_validation"]["manifest_path"])
    _verify_file(path, config["clean_validation"]["manifest_sha256"], "clean_validation")
    table = pq.read_table(path, columns=["sample_id", "split", "coordinate_accepted"])
    return {
        str(row["sample_id"])
        for row in table.to_pylist()
        if row["split"] == "validation" and row["coordinate_accepted"] is True
    }


def select_panel(config: dict[str, Any], *, split: str, count: int, seed: int) -> list[dict[str, Any]]:
    authorization = _authorize(config)
    dataset = RichGeometryDataset(authorization, split=split)
    permitted = _clean_validation_ids(config) if split == "validation" else None
    strata = config["length_strata"]
    minimum = int(config["smoke"]["minimum_samples_per_bin"])
    if count % len(strata):
        raise ValueError("E007 Phase 4C panel size must divide evenly across length bins")
    per_bin = count // len(strata)
    if per_bin < minimum:
        raise ValueError("E007 Phase 4C panel allocation is below the per-bin minimum")
    candidates: dict[str, list[tuple[int, int]]] = defaultdict(list)
    capacity = per_bin * 4
    for index, sample_id, length in dataset.iter_metadata():
        if permitted is not None and sample_id not in permitted:
            continue
        name = length_stratum(length, strata)
        rank = int.from_bytes(hashlib.sha256(f"{seed}|{split}|{sample_id}".encode()).digest(), "big")
        item = (-rank, -index)
        heap = candidates[name]
        if len(heap) < capacity:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    rows = []
    counts = {}
    for stratum in strata:
        name = str(stratum["name"])
        selected = []
        ordered = sorted([(-rank, -index) for rank, index in candidates[name]])
        for _rank, index in ordered:
            try:
                selected.append(_project_row(dataset[index]))
            except ValueError:
                continue
            if len(selected) == per_bin:
                break
        counts[name] = len(selected)
        rows.extend(selected)
    shortages = {name: value for name, value in counts.items() if value < per_bin}
    if shortages:
        raise ValueError(
            f"E007 Phase 4C {split} panel lacks eligible per-bin structures: "
            f"candidate=shared arm=shuffled_geometry counts={counts} required={per_bin}"
        )
    if len(rows) != count:
        raise ValueError(f"E007 Phase 4C {split} panel underfill: {len(rows)} != {count}")
    if len({row["sample_id"] for row in rows}) != count:
        raise ValueError("E007 Phase 4C panel sample IDs are not unique")
    validate_panel_bin_minimum(rows, strata=strata, minimum=minimum, split=split)
    return rows


def validate_panel_bin_minimum(
    rows: list[dict[str, Any]],
    *,
    strata: list[dict[str, Any]],
    minimum: int,
    split: str,
) -> dict[str, int]:
    counts = {str(row["name"]): 0 for row in strata}
    for row in rows:
        counts[length_stratum(len(str(row["sequence"])), strata)] += 1
    shortages = {name: count for name, count in counts.items() if count < minimum}
    if shortages:
        raise ValueError(
            "E007 Phase 4C shuffled panel minimum contradiction: "
            f"candidate=shared arm=shuffled_geometry split={split} counts={counts} minimum={minimum}"
        )
    return counts


def deterministic_mask(length: int, *, fraction: float, seed: int) -> torch.Tensor:
    if not 0 < fraction < 1 or length < 1:
        raise ValueError("E007 Phase 4C mask contract is invalid")
    count = max(1, round(length * fraction))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(length, generator=generator)
    result = torch.zeros(length, dtype=torch.bool)
    result[order[:count]] = True
    return result


def paired_bootstrap_interval(values: list[float], *, seed: int, replicates: int) -> dict[str, float]:
    if not values or replicates < 100:
        raise ValueError("E007 Phase 4C bootstrap input is insufficient")
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        means[index] = generator.choice(array, size=len(array), replace=True).mean()
    return {
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def classify_candidate(effects: dict[str, Any], *, length_strata: list[str]) -> str:
    controls = ("null_geometry", "shuffled_geometry")
    if any(effects[name]["bootstrap_95"]["lower_95"] <= 0 for name in controls):
        return "no_supported_held_out_improvement"
    improved = {
        stratum
        for stratum in length_strata
        if all(effects[name]["by_length_stratum"].get(stratum, 0.0) > 0 for name in controls)
    }
    return "usable_geometry_conditioning" if len(improved) >= 2 else "improvement_confined_to_one_stratum"


def _parameter_groups_have_gradients(conditioner: InvariantGeometryConditioner) -> dict[str, bool]:
    groups = {
        "pair_encoder": conditioner.pair_encoder,
        "conditioning_adapter": conditioner.conditioning_adapter,
    }
    result = {
        name: any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in module.parameters()
        )
        for name, module in groups.items()
    }
    result["gate"] = conditioner.gate_logit.grad is not None and bool(torch.isfinite(conditioner.gate_logit.grad).all())
    return result


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_payload(
    conditioner: InvariantGeometryConditioner,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    cursor: int,
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "configuration_sha256": config_sha256,
        "conditioner": conditioner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "cursor": cursor,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "authorizes_training": False,
    }


def restore_checkpoint(
    path: Path,
    conditioner: InvariantGeometryConditioner,
    optimizer: torch.optim.Optimizer,
    *,
    config_sha256: str,
) -> tuple[int, int]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("version") != VERSION or value.get("configuration_sha256") != config_sha256:
        raise ValueError("E007 Phase 4C checkpoint contract contradiction")
    conditioner.load_state_dict(value["conditioner"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    random.setstate(value["python_rng_state"])
    np.random.set_state(value["numpy_rng_state"])
    torch.set_rng_state(value["torch_rng_state"].cpu())
    if value["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in value["cuda_rng_state"]])
    return int(value["update"]), int(value["cursor"])


def _prior_inputs(candidate: str, tokenizer: Any, sequence: str, mask_seed: int, mask_fraction: float):
    if any(residue not in CANONICAL for residue in sequence):
        raise ValueError("E007 Phase 4C sequence contains noncanonical residues")
    if candidate == "progen2_151m":
        framed = progen_framed_sequence_contract(tokenizer, sequence, position_capacity=1024)
        ids = torch.tensor(framed["framed_model_input"]["ids"], dtype=torch.long)
        return ids, ids[1:-1], torch.ones(len(sequence), dtype=torch.bool)
    encoded = tokenizer(sequence, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    selected = deterministic_mask(len(sequence), fraction=mask_fraction, seed=mask_seed)
    targets = encoded[1:-1].clone()
    encoded[1:-1][selected] = int(tokenizer.mask_token_id)
    return encoded, targets, selected


def frozen_prior_loss(
    candidate: str,
    prior: Any,
    tokenizer: Any,
    conditioner: InvariantGeometryConditioner,
    row: dict[str, Any],
    *,
    geometry_row: dict[str, Any],
    arm: str,
    mask_seed: int,
    mask_fraction: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    ids, targets, selected = _prior_inputs(candidate, tokenizer, row["sequence"], mask_seed, mask_fraction)
    ids = ids[None].to(device)
    targets = targets.to(device)
    selected = selected.to(device)
    coordinates = geometry_row["coordinates"][None].to(device)
    residue_mask = geometry_row["residue_mask"][None].to(device)
    conditioning = conditioner(coordinates, residue_mask, null_geometry=arm == "null_geometry")
    hidden_width = int(prior.config.hidden_size if candidate == "esm2_150m" else prior.config.n_embd)
    biological = conditioner.for_prior_width(conditioning, hidden_width)
    if biological.shape[1] != len(row["sequence"]):
        biological = F.interpolate(
            biological.transpose(1, 2),
            size=len(row["sequence"]),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    embeddings = prior.get_input_embeddings()(ids).detach()
    conditioned = torch.zeros_like(embeddings)
    if candidate == "esm2_150m":
        conditioned[:, 1:-1] = biological
        logits = prior(inputs_embeds=embeddings + conditioned, attention_mask=torch.ones_like(ids)).logits[0, 1:-1]
        token_logits = logits[selected]
        token_targets = targets[selected]
    else:
        conditioned[:, : biological.shape[1]] = biological
        logits = prior(inputs_embeds=embeddings + conditioned, attention_mask=torch.ones_like(ids)).logits[0]
        token_logits = logits[: len(row["sequence"])]
        token_targets = targets
    canonical_ids = [
        tokenizer.token_to_id(residue) if candidate == "progen2_151m" else tokenizer.convert_tokens_to_ids(residue)
        for residue in CANONICAL
    ]
    canonical_logits = token_logits[:, canonical_ids]
    remap = {value: index for index, value in enumerate(canonical_ids)}
    canonical_targets = torch.tensor([remap[int(value)] for value in token_targets], device=device)
    loss = F.cross_entropy(canonical_logits.float(), canonical_targets)
    predictions = canonical_logits.argmax(dim=-1)
    accuracy = (predictions == canonical_targets).float().mean()
    top_k = min(5, len(CANONICAL))
    topk_accuracy = (
        canonical_logits.topk(top_k, dim=-1).indices.eq(canonical_targets[:, None]).any(dim=-1).float().mean()
    )
    predicted_composition = torch.bincount(predictions, minlength=len(CANONICAL)).float()
    target_composition = torch.bincount(canonical_targets, minlength=len(CANONICAL)).float()
    predicted_composition /= predicted_composition.sum().clamp_min(1)
    target_composition /= target_composition.sum().clamp_min(1)
    return loss, {
        "token_count": len(canonical_targets),
        "cross_entropy": float(loss.detach()),
        "perplexity": float(torch.exp(loss.detach())),
        "top1_accuracy": float(accuracy.detach()),
        "top5_accuracy": float(topk_accuracy.detach()),
        "canonical_validity": 1.0,
        "amino_acid_composition_total_variation": float(
            (predicted_composition - target_composition).abs().sum().mul(0.5).detach()
        ),
    }


def _memory(torch_module: Any, device: torch.device) -> dict[str, float | None]:
    rss = 0.0
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = float(line.split()[1]) / 1024
                break
    return {
        "current_rss_mib": rss,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_cuda_allocated_mib": (
            torch_module.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "peak_cuda_reserved_mib": (
            torch_module.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None
        ),
    }


def _evaluate(
    candidate: str,
    arm: str,
    prior: Any,
    tokenizer: Any,
    conditioner: InvariantGeometryConditioner,
    rows: list[dict[str, Any]],
    donors: dict[str, str],
    *,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    by_id = {row["sample_id"]: row for row in rows}
    records = []
    conditioner.eval()
    for index, row in enumerate(rows):
        geometry = by_id[donors[row["sample_id"]]] if arm == "shuffled_geometry" else row
        fractions = config["evaluation"]["mask_fractions"] if candidate == "esm2_150m" else [1.0]
        for fraction in fractions:
            with torch.no_grad():
                loss, metrics = frozen_prior_loss(
                    candidate,
                    prior,
                    tokenizer,
                    conditioner,
                    row,
                    geometry_row=geometry,
                    arm=arm,
                    mask_seed=int(config["seed"]) + index,
                    mask_fraction=float(fraction if fraction < 1 else 0.15),
                    device=device,
                )
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "length_stratum": length_stratum(len(row["sequence"]), config["length_strata"]),
                    "mask_fraction": fraction if candidate == "esm2_150m" else None,
                    "negative_log_likelihood_per_residue": float(loss),
                    **metrics,
                }
            )
    values = [record["cross_entropy"] for record in records]
    return {
        "records": records,
        "macro_cross_entropy": float(np.mean(values)),
        "macro_perplexity": float(math.exp(np.mean(values))),
        "macro_top1_accuracy": float(np.mean([record["top1_accuracy"] for record in records])),
        "macro_top5_accuracy": float(np.mean([record["top5_accuracy"] for record in records])),
        "macro_composition_total_variation": float(
            np.mean([record["amino_acid_composition_total_variation"] for record in records])
        ),
        "token_weighted_cross_entropy": float(
            np.average(values, weights=[record["token_count"] for record in records])
        ),
        "by_length_stratum": {
            name: {
                "sample_records": sum(record["length_stratum"] == name for record in records),
                "macro_cross_entropy": float(
                    np.mean([record["cross_entropy"] for record in records if record["length_stratum"] == name])
                ),
            }
            for name in {record["length_stratum"] for record in records}
        },
    }


def _candidate_effects(results: list[dict[str, Any]], candidate: str, config: dict[str, Any]) -> dict[str, Any]:
    arms = {row["arm"]: row for row in results if row["candidate"] == candidate}
    if set(arms) != set(ARMS) or any(row["status"] != "completed_non_authorizing" for row in arms.values()):
        return {"classification": "inconclusive_requires_scientific_review", "effects": {}}
    update = str(config["pilot"]["successful_optimizer_updates"])
    correct = arms["correct_geometry"]["evaluations"][update]["records"]
    correct_by_key = {(row["sample_id"], row["mask_fraction"]): row for row in correct}
    effects = {}
    for control in ("null_geometry", "shuffled_geometry"):
        control_rows = arms[control]["evaluations"][update]["records"]
        paired = []
        by_stratum: dict[str, list[float]] = defaultdict(list)
        for row in control_rows:
            key = (row["sample_id"], row["mask_fraction"])
            if key not in correct_by_key:
                raise ValueError("E007 Phase 4C paired evaluation identity contradiction")
            value = row["cross_entropy"] - correct_by_key[key]["cross_entropy"]
            paired.append(value)
            by_stratum[row["length_stratum"]].append(value)
        effects[control] = {
            "sign_convention": "positive_means_correct_geometry_has_lower_cross_entropy",
            "bootstrap_95": paired_bootstrap_interval(
                paired,
                seed=int(config["evaluation"]["bootstrap_seed"]),
                replicates=int(config["evaluation"]["bootstrap_replicates"]),
            ),
            "per_residue_nll_improvement_mean": float(np.mean(paired)),
            "by_length_stratum": {name: float(np.mean(values)) for name, values in by_stratum.items()},
        }
    classification = classify_candidate(
        effects,
        length_strata=[row["name"] for row in config["length_strata"]],
    )
    return {"classification": classification, "effects": effects}


def _worker(
    config_path: str,
    candidate: str,
    arm: str,
    mode: str,
    output_path: str,
    resume_path: str | None,
) -> None:
    result_path = Path(output_path)
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
        train_count = int(config[mode]["train_panel_size"])
        validation_count = int(config[mode]["validation_panel_size"])
        train_rows = select_panel(config, split="train", count=train_count, seed=seed)
        validation_rows = select_panel(config, split="validation", count=validation_count, seed=seed + 1)
        train_ids = {row["sample_id"] for row in train_rows}
        validation_ids = {row["sample_id"] for row in validation_rows}
        if train_ids & validation_ids:
            raise ValueError("E007 Phase 4C train/validation overlap")
        if arm == "shuffled_geometry":
            stage = "shuffled_donor_validation"
        train_donors, validation_donors = build_arm_donor_maps(
            arm,
            train_rows,
            validation_rows,
            seed=seed,
            strata=config["length_strata"],
        )
        donor_mapping_sha256 = (
            _canonical_sha256({"train": train_donors, "validation": validation_donors})
            if arm == "shuffled_geometry"
            else None
        )
        stage = "model_construction"
        device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        prior, tokenizer, loader = load_reviewed_candidate(candidate, config["artifact_cache_root"], device=device)
        prior.eval()
        for parameter in prior.parameters():
            parameter.requires_grad_(False)
        prior_before = parameter_identity_hash(prior)
        conditioner = InvariantGeometryConditioner(**config["conditioner"]).to(device)
        execution["model_created"] = True
        optimizer = torch.optim.AdamW(conditioner.parameters(), **config["optimizer"])
        execution["optimizer_created"] = True
        by_id = {row["sample_id"]: row for row in train_rows}
        update = cursor = 0
        if resume_path:
            update, cursor = restore_checkpoint(
                Path(resume_path),
                conditioner,
                optimizer,
                config_sha256=sha256_file(config_path),
            )
        stage = "initial_evaluation"
        execution["forward_performed"] = True
        initial_evaluation = _evaluate(
            candidate,
            arm,
            prior,
            tokenizer,
            conditioner,
            validation_rows,
            validation_donors,
            config=config,
            device=device,
        )
        replay_evaluation = _evaluate(
            candidate,
            arm,
            prior,
            tokenizer,
            conditioner,
            validation_rows,
            validation_donors,
            config=config,
            device=device,
        )
        if _canonical_sha256(initial_evaluation) != _canonical_sha256(replay_evaluation):
            raise ValueError("E007 Phase 4C deterministic evaluation replay failed")
        evaluations = {str(update): initial_evaluation}
        maximum_updates = 0 if mode == "smoke" else int(config["pilot"]["successful_optimizer_updates"])
        gradient_coverage = None
        while update < maximum_updates:
            row = train_rows[cursor % len(train_rows)]
            geometry = by_id[train_donors[row["sample_id"]]] if arm == "shuffled_geometry" else row
            optimizer.zero_grad(set_to_none=True)
            loss, _metrics = frozen_prior_loss(
                candidate,
                prior,
                tokenizer,
                conditioner,
                row,
                geometry_row=geometry,
                arm=arm,
                mask_seed=seed + cursor,
                mask_fraction=float(config["evaluation"]["training_mask_fraction"]),
                device=device,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("E007 Phase 4C training loss is non-finite")
            loss.backward()
            execution["backward_performed"] = True
            gradient_coverage = _parameter_groups_have_gradients(conditioner)
            if not all(gradient_coverage.values()):
                raise FloatingPointError(f"E007 Phase 4C gradient coverage failed: {gradient_coverage}")
            torch.nn.utils.clip_grad_norm_(conditioner.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
            update += 1
            execution["optimizer_updates"] = update
            cursor += 1
            if update in config["pilot"]["evaluation_updates"]:
                evaluations[str(update)] = _evaluate(
                    candidate,
                    arm,
                    prior,
                    tokenizer,
                    conditioner,
                    validation_rows,
                    validation_donors,
                    config=config,
                    device=device,
                )
            if update % int(config["pilot"]["checkpoint_frequency"]) == 0:
                atomic_checkpoint(
                    result_path.with_suffix(".checkpoint.pt"),
                    checkpoint_payload(
                        conditioner,
                        optimizer,
                        update=update,
                        cursor=cursor,
                        config_sha256=sha256_file(config_path),
                    ),
                )
                execution["checkpoint_written"] = True
        if mode == "smoke":
            row = train_rows[0]
            geometry = by_id[train_donors[row["sample_id"]]] if arm == "shuffled_geometry" else row
            optimizer.zero_grad(set_to_none=True)
            loss, _ = frozen_prior_loss(
                candidate,
                prior,
                tokenizer,
                conditioner,
                row,
                geometry_row=geometry,
                arm=arm,
                mask_seed=seed,
                mask_fraction=float(config["evaluation"]["training_mask_fraction"]),
                device=device,
            )
            loss.backward()
            execution["backward_performed"] = True
            gradient_coverage = _parameter_groups_have_gradients(conditioner)
        prior_after = parameter_identity_hash(prior)
        if prior_after != prior_before:
            raise ValueError("E007 Phase 4C pretrained prior mutated")
        execution["parameter_mutation_checked"] = True
        memory = _memory(torch, device)
        if memory["peak_rss_mib"] > float(config["memory"]["maximum_rss_mib"]):
            raise MemoryError("E007 Phase 4C RSS limit exceeded")
        for key, limit in (
            ("peak_cuda_allocated_mib", "maximum_cuda_allocated_mib"),
            ("peak_cuda_reserved_mib", "maximum_cuda_reserved_mib"),
        ):
            if memory[key] is not None and memory[key] > float(config["memory"][limit]):
                raise MemoryError(f"E007 Phase 4C {key} limit exceeded")
        _atomic_json(
            result_path,
            {
                "status": "completed_non_authorizing",
                "candidate": candidate,
                "arm": arm,
                "mode": mode,
                "optimizer_updates": update,
                "execution": execution,
                "train_sample_id_sha256": hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),
                "validation_sample_id_sha256": hashlib.sha256("\n".join(sorted(validation_ids)).encode()).hexdigest(),
                "prior_parameter_sha256_before": prior_before,
                "prior_parameter_sha256_after": prior_after,
                "prior_parameters_unchanged": True,
                "conditioner_parameter_count": conditioner_parameter_count(conditioner),
                "gradient_coverage": gradient_coverage,
                "donor_mapping_sha256": donor_mapping_sha256,
                "evaluations": evaluations,
                "deterministic_replay_verified": True,
                "loader": loader,
                "memory": memory,
                "deterministic_replay_required": True,
                **NON_AUTHORIZING,
            },
        )
    except BaseException as error:
        _atomic_json(
            result_path,
            {
                "status": "failed",
                "candidate": candidate,
                "arm": arm,
                "mode": mode,
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
                "failure_stage": stage,
                "execution": execution,
                "optimizer_updates": execution["optimizer_updates"],
                **NON_AUTHORIZING,
            },
        )


def _run_impl(config_path: str | Path, *, mode: str, resume: bool = False) -> dict[str, Any]:
    if mode not in {"smoke", "pilot"}:
        raise ValueError("E007 Phase 4C execution mode is invalid")
    config_path = Path(config_path)
    config = _load_config(config_path)
    protected_before = verify_prerequisites(config)
    output = Path(config[f"{mode}_output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or (staging.exists() and not resume):
        raise FileExistsError(f"E007 Phase 4C output exists: {output}")
    staging.mkdir(parents=True, exist_ok=resume)
    journal_path = staging / "journal.json"
    journal = json.loads(journal_path.read_text()) if resume and journal_path.is_file() else {"committed_results": {}}
    _atomic_json(staging / "heartbeat.json", {"status": "running", "mode": mode, **NON_AUTHORIZING})
    context = mp.get_context("spawn")
    results = []
    for candidate in CANDIDATES:
        for arm in ARMS:
            result_path = staging / f"{candidate}__{arm}.json"
            checkpoint = result_path.with_suffix(".checkpoint.pt")
            identity = f"{candidate}/{arm}"
            committed = journal["committed_results"].get(identity)
            if committed:
                if not result_path.is_file() or sha256_file(result_path) != committed:
                    raise ValueError(f"E007 Phase 4C committed result hash contradiction: {identity}")
            else:
                process = context.Process(
                    target=_worker,
                    args=(
                        str(config_path),
                        candidate,
                        arm,
                        mode,
                        str(result_path),
                        str(checkpoint) if resume and checkpoint.is_file() else None,
                    ),
                )
                process.start()
                process.join()
                if process.exitcode != 0 or not result_path.is_file():
                    raise WorkerFailure(
                        candidate,
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
            if result["status"] != "completed_non_authorizing":
                raise WorkerFailure(candidate, arm, result)
            if not committed:
                journal["committed_results"][identity] = sha256_file(result_path)
                _atomic_json(journal_path, journal)
            results.append(result)
            _atomic_json(
                staging / "heartbeat.json",
                {
                    "status": "running",
                    "mode": mode,
                    "completed_cases": [f"{row['candidate']}/{row['arm']}" for row in results],
                    **NON_AUTHORIZING,
                },
            )
    protected_after = verify_prerequisites(config)
    if protected_after != protected_before:
        raise ValueError("E007 Phase 4C protected inputs changed")
    comparisons = (
        {candidate: _candidate_effects(results, candidate, config) for candidate in CANDIDATES}
        if mode == "pilot"
        else {}
    )
    pareto = (
        sorted(
            candidate
            for candidate, row in comparisons.items()
            if row["classification"] == "usable_geometry_conditioning"
        )
        if mode == "pilot"
        else []
    )
    report = {
        "status": "completed_non_authorizing",
        "version": VERSION,
        "mode": mode,
        "results": results,
        "candidate_comparisons": comparisons,
        "pareto_candidates": pareto,
        "candidate_selection": "scientific_review_required" if mode == "pilot" else None,
        "comparison_policy": "within_candidate_paired_effects_only_no_cross_objective_raw_loss",
        "scalar_score_used": False,
        "pareto_set_required": True,
        "protected_inputs_unchanged": True,
        **NON_AUTHORIZING,
    }
    _atomic_json(staging / "report.json", report)
    _atomic_json(
        staging / "protocol.json",
        {
            "status": report["status"],
            "version": VERSION,
            "mode": mode,
            "configuration_sha256": sha256_file(config_path),
            "report_sha256": sha256_file(staging / "report.json"),
            "protected_hashes": protected_before,
            **NON_AUTHORIZING,
        },
    )
    _atomic_json(
        staging / "heartbeat.json",
        {
            "status": "completed",
            "mode": mode,
            "optimizer_updates": sum(r["optimizer_updates"] for r in results),
            **NON_AUTHORIZING,
        },
    )
    staging.replace(output)
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
            _atomic_json(
                staging / "heartbeat.json",
                {
                    "status": "failed",
                    "mode": mode,
                    "updated_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:2000],
                    "failed_candidate": error.candidate if isinstance(error, WorkerFailure) else None,
                    "failed_arm": error.arm if isinstance(error, WorkerFailure) else None,
                    "optimizer_updates": int(worker.get("optimizer_updates", 0)),
                    "model_created": execution.get("model_created"),
                    "forward_performed": execution.get("forward_performed"),
                    "backward_performed": execution.get("backward_performed"),
                    "optimizer_created": execution.get("optimizer_created"),
                    "checkpoint_written": execution.get("checkpoint_written"),
                    **NON_AUTHORIZING,
                },
            )
        raise
