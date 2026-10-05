"""Bounded real-artifact CPU diagnostic for E007 Phase 4B.2."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from protein_distance_diffusion.evaluation import e007_pretrained_sequence_prior_smoke as smoke
from protein_distance_diffusion.evaluation.e007_pretrained_loaders import CANONICAL, offline_network_guard
from protein_distance_diffusion.evaluation.e007_pretrained_loaders_v3 import (
    load_reviewed_candidate,
    parameter_identity_hash,
    progen_tokenizer_diagnostic,
)

VERSION = "e007_pretrained_sequence_prior_cpu_diagnostic_v1"
NON_AUTHORIZING = {
    "authorizes_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_joint_training": False,
    "authorizes_production_training": False,
    "training_performed": False,
    "sampling_performed": False,
    "optimizer_created": False,
    "optimizer_updates": 0,
    "cuda_used": False,
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("E007 Phase 4B.2 diagnostic configuration version contradiction")
    return payload


def _protected(config: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for group in ("v1", "v2"):
        section = config["protected_outputs"][group]
        root = Path(section["path"])
        for name, expected in section["hashes"].items():
            key = f"{group}_{name.removesuffix('.json')}"
            observed[key] = _sha(root / name)
            if observed[key] != expected:
                raise ValueError(f"E007 Phase 4B.2 protected {group} hash contradiction: {name}")
    artifact = config["artifact_lock"]
    observed["artifact_lock"] = _sha(artifact["path"])
    if observed["artifact_lock"] != artifact["sha256"]:
        raise ValueError("E007 Phase 4B.2 artifact-lock hash contradiction")
    loader = config["loader_source"]
    observed["loader_source"] = _sha(loader["path"])
    if observed["loader_source"] != loader["sha256"]:
        raise ValueError("E007 Phase 4B.2 loader-source hash contradiction")
    verification = config["artifact_verification_config"]
    observed["artifact_verification_config"] = _sha(verification["path"])
    if observed["artifact_verification_config"] != verification["sha256"]:
        raise ValueError("E007 Phase 4B.2 artifact-verification configuration hash contradiction")
    smoke.verify_artifacts_offline(verification["path"])
    return observed


def plan(config_path: str | Path) -> dict[str, Any]:
    config = _config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError("E007 Phase 4B.2 diagnostic final or staging output exists")
    hashes = _protected(config)
    return {
        "status": "planned_non_authorizing_cpu_only",
        "version": VERSION,
        "configuration_sha256": _sha(config_path),
        "candidates": ["esm2_150m", "progen2_151m"],
        "real_artifacts_read": False,
        "model_created": False,
        "forward_performed": False,
        "output_created": False,
        "protected_hashes": hashes,
        **NON_AUTHORIZING,
    }


def _all_finite(model: Any) -> bool:
    import torch

    return all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())


def _candidate(candidate: str, cache: Path, sequence: str) -> dict[str, Any]:
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("E007 Phase 4B.2 CPU diagnostic found initialized CUDA")
    model, tokenizer, metadata = load_reviewed_candidate(candidate, cache, device=torch.device("cpu"))
    model.eval()
    before = parameter_identity_hash(model)
    if not _all_finite(model):
        raise FloatingPointError(f"E007 Phase 4B.2 non-finite state tensor: {candidate}")
    if candidate == "esm2_150m":
        encoded = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"]
        attention = encoded["attention_mask"]
        tokenizer_diagnostic = metadata["tokenizer_contract"]
    else:
        tokenizer_diagnostic = progen_tokenizer_diagnostic(tokenizer)
        framed = tokenizer_diagnostic["canonical_alphabet"]["framed_model_input"]
        input_ids = torch.tensor([framed["ids"]], dtype=torch.long)
        attention = torch.ones_like(input_ids)
    with torch.no_grad():
        first = model(input_ids=input_ids, attention_mask=attention).logits
        second = model(input_ids=input_ids, attention_mask=attention).logits
    if not torch.isfinite(first).all() or not torch.equal(first, second):
        raise FloatingPointError(f"E007 Phase 4B.2 deterministic finite CPU replay failed: {candidate}")
    first_hash = hashlib.sha256(first.contiguous().numpy().tobytes()).hexdigest()
    second_hash = hashlib.sha256(second.contiguous().numpy().tobytes()).hexdigest()
    after = parameter_identity_hash(model)
    if before != after:
        raise ValueError(f"E007 Phase 4B.2 parameter mutation detected: {candidate}")
    if torch.cuda.is_initialized():
        raise RuntimeError("E007 Phase 4B.2 CPU diagnostic initialized CUDA")
    return {
        "candidate": candidate,
        "status": "passed_cpu_real_artifact_diagnostic",
        "device": "cpu",
        "input_biological_sequence": sequence,
        "biological_residue_count": len(sequence),
        "total_input_token_count": int(input_ids.shape[1]),
        "logit_shape": list(first.shape),
        "first_logits_sha256": first_hash,
        "replay_logits_sha256": second_hash,
        "logits_exactly_replayed": first_hash == second_hash,
        "all_state_tensors_finite": True,
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "parameter_unchanged": before == after,
        "tokenizer_diagnostic": tokenizer_diagnostic,
        "loader_metadata": metadata,
        **NON_AUTHORIZING,
    }


def run(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError("E007 Phase 4B.2 diagnostic final or staging output exists")
    protected_before = _protected(config)
    staging.mkdir(parents=True)
    _atomic_json(staging / "heartbeat.json", {"status": "running", "started_utc": _utc_now(), **NON_AUTHORIZING})
    try:
        with offline_network_guard():
            esm = _candidate("esm2_150m", Path(config["artifact_cache_root"]), CANONICAL)
            _atomic_json(staging / "esm2_150m.json", esm)
            del esm
            gc.collect()
            progen = _candidate("progen2_151m", Path(config["artifact_cache_root"]), CANONICAL)
            _atomic_json(staging / "progen2_151m.json", progen)
            del progen
            gc.collect()
        protected_after = _protected(config)
        if protected_after != protected_before:
            raise ValueError("E007 Phase 4B.2 protected inputs changed")
        esm_summary = json.loads((staging / "esm2_150m.json").read_text())
        progen_summary = json.loads((staging / "progen2_151m.json").read_text())
        report = {
            "status": "completed_non_authorizing_cpu_diagnostic",
            "version": VERSION,
            "esm": {
                "status": esm_summary["status"],
                "accounting": esm_summary["loader_metadata"]["diagnostics"]["accounting"],
                "logits_sha256": esm_summary["first_logits_sha256"],
                "parameter_sha256": esm_summary["parameter_sha256_after"],
            },
            "progen2": {
                "status": progen_summary["status"],
                "tokenizer_diagnostic": progen_summary["tokenizer_diagnostic"],
                "state_validation": progen_summary["loader_metadata"]["diagnostics"]["state_validation"],
                "logits_sha256": progen_summary["first_logits_sha256"],
                "parameter_sha256": progen_summary["parameter_sha256_after"],
            },
            "protected_inputs_unchanged": True,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "report.json", report)
        protocol = {
            "status": report["status"],
            "version": VERSION,
            "configuration_sha256": _sha(config_path),
            "artifact_lock_sha256": protected_before["artifact_lock"],
            "esm_result_sha256": _sha(staging / "esm2_150m.json"),
            "progen2_result_sha256": _sha(staging / "progen2_151m.json"),
            "report_sha256": _sha(staging / "report.json"),
            "completed_utc": _utc_now(),
            "protected_hashes": protected_before,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "protocol.json", protocol)
        names = ["esm2_150m.json", "progen2_151m.json", "report.json", "protocol.json"]
        rows = [
            {"path": name, "size_bytes": (staging / name).stat().st_size, "sha256": _sha(staging / name)}
            for name in names
        ]
        _atomic_json(staging / "artifact_inventory.json", {"artifacts": rows, "aggregate_sha256": _canonical_sha(rows)})
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "completed",
                "completed_utc": _utc_now(),
                "report_sha256": protocol["report_sha256"],
                **NON_AUTHORIZING,
            },
        )
        staging.replace(output)
        return {"status": report["status"], "output_dir": str(output), **NON_AUTHORIZING}
    except BaseException as error:
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "failed",
                "completed_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
                **NON_AUTHORIZING,
            },
        )
        raise
