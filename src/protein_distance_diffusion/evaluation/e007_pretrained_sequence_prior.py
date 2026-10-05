"""Read-only E007 Phase 4A pretrained sequence-prior selection audit."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

VERSION = "e007_pretrained_sequence_prior_audit_v1"
CONTRACT_VERSION = "e007_pretrained_sequence_prior_contract_v1"
PHASE4B_VERSION = "e007_pretrained_sequence_prior_weight_smoke_plan_v1"
OUTCOME = "both_candidates_require_bounded_weight_smoke"
CANONICAL_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
NON_AUTHORIZING = {
    "pretrained_weights_downloaded": False,
    "pretrained_model_loaded": False,
    "model_created": False,
    "cuda_used": False,
    "optimizer_created": False,
    "backward_performed": False,
    "training_performed": False,
    "sampling_performed": False,
    "dataset_modified": False,
    "authorizes_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_joint_training": False,
    "authorizes_production_use": False,
    "authorizes_additional_coordinate_training": False,
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or config.get("version") != VERSION:
        raise ValueError("E007 Phase 4A configuration version contradiction")
    decision = config.get("decision", {})
    if decision.get("phase4a_outcome") != OUTCOME:
        raise ValueError("E007 Phase 4A decision outcome changed")
    if decision.get("shortlisted_candidates") != ["esm2_150m", "progen2_151m"]:
        raise ValueError("E007 Phase 4A primary shortlist changed")
    phase4b = config.get("phase4b", {})
    if phase4b.get("version") != PHASE4B_VERSION or phase4b.get("execution_authorized") is not False:
        raise ValueError("E007 Phase 4B must remain planned and non-authorizing")
    if int(phase4b.get("optimizer_updates", -1)) != 0:
        raise ValueError("E007 Phase 4B optimizer-update contract changed")
    if phase4b.get("lengths") != [64, 128, 256, 384, 500]:
        raise ValueError("E007 Phase 4B length panel changed")
    return config


def _verify_file(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"E007 Phase 4A prerequisite is absent: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"E007 Phase 4A prerequisite hash contradiction: {label}")
    return observed


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("version") != CONTRACT_VERSION:
        raise ValueError("E007 Phase 4A contract version contradiction")
    sections = {"verified_facts", "estimates", "unresolved_questions", "future_executable_tests"}
    if not sections.issubset(contract):
        raise ValueError("E007 Phase 4A contract section is absent")
    facts = contract["verified_facts"]
    if facts["dataset"]["canonical_residues"] != CANONICAL_RESIDUES:
        raise ValueError("E007 Phase 4A canonical residue contract changed")
    models = facts["models"]
    if set(models) != {"esm2_150m", "progen2_151m", "proteinmpnn_ca_only", "esm3_open_small"}:
        raise ValueError("E007 Phase 4A candidate set changed")
    for name, model in models.items():
        if model.get("continuous_latent_decoder") is not False:
            raise ValueError(f"E007 Phase 4A unsupported latent-decoder claim: {name}")
        if model.get("directly_sampleable_internal_representation") is not False:
            raise ValueError(f"E007 Phase 4A unsupported sampleable-representation claim: {name}")
    geometry = contract["scientific_contract"]["common_geometry"]
    if geometry.get("geometry_identity_inputs_forbidden") is not True:
        raise ValueError("E007 Phase 4A geometry identity-leakage guard changed")
    if contract["scientific_contract"]["evaluation"].get("aggregate_scalar_score_forbidden") is not True:
        raise ValueError("E007 Phase 4A scalar-score prohibition changed")


def _verify_selection(config: Mapping[str, Any]) -> dict[str, str]:
    section = config["coordinate_selection"]
    directory = Path(section["directory"])
    paths = {
        "selection_report": directory / "report.json",
        "selection_protocol": directory / "protocol.json",
        "selection_checkpoint_record": directory / "selected_checkpoint.json",
        "selection_inventory": directory / "artifact_inventory.json",
    }
    expected = {
        "selection_report": section["report_sha256"],
        "selection_protocol": section["protocol_sha256"],
        "selection_checkpoint_record": section["selected_checkpoint_record_sha256"],
        "selection_inventory": section["artifact_inventory_sha256"],
    }
    hashes = {name: _verify_file(path, expected[name], name) for name, path in paths.items()}
    report = json.loads(paths["selection_report"].read_text())
    protocol = json.loads(paths["selection_protocol"].read_text())
    selected = json.loads(paths["selection_checkpoint_record"].read_text())
    expected_status = "completed_publication_only_non_authorizing"
    if report.get("status") != expected_status or protocol.get("status") != expected_status:
        raise ValueError("E007 Phase 4A coordinate-selection completion contradiction")
    if report.get("authorizes_sequence_conditioning") or protocol.get("authorizes_sequence_conditioning"):
        raise ValueError("E007 Phase 4A source unexpectedly authorizes sequence conditioning")
    checkpoint = config["coordinate_checkpoint"]
    if (
        selected.get("checkpoint_path") != checkpoint["path"]
        or selected.get("checkpoint_sha256") != checkpoint["sha256"]
    ):
        raise ValueError("E007 Phase 4A selected coordinate-checkpoint identity contradiction")
    return hashes


def _verify_dataset(config: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    section = config["rich_geometry_dataset"]
    directory = Path(section["directory"])
    files = {
        "dataset_protocol": (directory / "protocol.json", section["protocol_sha256"]),
        "dataset_schema": (directory / "schema.json", section["schema_sha256"]),
        "dataset_vocabulary": (directory / "vocabulary.json", section["vocabulary_sha256"]),
        "dataset_normalization": (directory / "normalization.json", section["normalization_sha256"]),
        "dataset_shard_inventory": (directory / "shard_hashes.sha256", section["shard_inventory_sha256"]),
    }
    hashes = {name: _verify_file(path, expected, name) for name, (path, expected) in files.items()}
    vocabulary = json.loads((directory / "vocabulary.json").read_text())
    expected_tokens = ["<PAD>", "<MASK>", *CANONICAL_RESIDUES]
    if vocabulary != {"tokens": expected_tokens, "unknown_token": None, "version": "canonical_20_pad_mask_v1"}:
        raise ValueError("E007 Phase 4A dataset vocabulary contradiction")
    clean = config["clean_validation"]
    hashes["identity30_clean_validation"] = _verify_file(
        Path(clean["path"]), clean["sha256"], "identity30_clean_validation"
    )
    if int(clean["sample_count"]) != 21583 or float(clean["maximum_identity"]) != 0.3:
        raise ValueError("E007 Phase 4A clean-validation contract contradiction")
    return hashes, vocabulary


def verify_prerequisites(config: Mapping[str, Any]) -> dict[str, Any]:
    contract_path = Path(config["contract"]["path"])
    hashes = {"contract": _verify_file(contract_path, config["contract"]["sha256"], "contract")}
    contract = json.loads(contract_path.read_text())
    _validate_contract(contract)
    hashes.update(_verify_selection(config))
    checkpoint = config["coordinate_checkpoint"]
    hashes["coordinate_checkpoint"] = _verify_file(
        Path(checkpoint["path"]), checkpoint["sha256"], "coordinate_checkpoint"
    )
    dataset_hashes, vocabulary = _verify_dataset(config)
    hashes.update(dataset_hashes)
    return {"hashes": hashes, "contract": contract, "vocabulary": vocabulary, "protected_inputs_verified": True}


def _installed_packages(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _torch_build_metadata() -> dict[str, str | None]:
    try:
        distribution = importlib.metadata.distribution("torch")
    except importlib.metadata.PackageNotFoundError:
        return {"torch": None, "cuda_build": None}
    version_path = Path(distribution.locate_file("torch/version.py"))
    cuda = None
    full_version = distribution.version
    if version_path.is_file():
        text = version_path.read_text(errors="replace")
        version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", text)
        cuda_match = re.search(r"cuda\s*:\s*Optional\[str\]\s*=\s*['\"]([^'\"]+)", text)
        if version_match:
            full_version = version_match.group(1)
        if cuda_match:
            cuda = cuda_match.group(1)
    return {"torch": full_version, "cuda_build": cuda}


def _bounded_cache_inventory(roots: Sequence[str], *, maximum_examples: int = 25) -> dict[str, Any]:
    examples: list[str] = []
    candidate_examples: list[str] = []
    total_files = 0
    total_bytes = 0
    existing_roots: list[str] = []
    candidate_patterns = (
        "models--facebook--esm2",
        "models--evolutionaryscale--esm3",
        "progen2",
        "proteinmpnn",
        "esm2_t30_150m_ur50d",
    )
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.is_dir():
            continue
        existing_roots.append(str(root))
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            total_files += 1
            total_bytes += path.stat().st_size
            if len(examples) < maximum_examples:
                examples.append(str(path))
            if any(pattern in str(path).lower() for pattern in candidate_patterns):
                candidate_examples.append(str(path))
    return {
        "existing_roots": existing_roots,
        "file_count": total_files,
        "total_bytes": total_bytes,
        "bounded_examples": examples,
        "examples_truncated": total_files > len(examples),
        "candidate_weight_match_count": len(candidate_examples),
        "candidate_weight_matches": candidate_examples[:maximum_examples],
        "candidate_weight_matches_truncated": len(candidate_examples) > maximum_examples,
        "cache_modified": False,
    }


def repository_environment_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    package_names = config["repository_audit"]["candidate_packages"]
    packages = _installed_packages(package_names)
    build = _torch_build_metadata()
    return {
        "python_packages": {"torch": build["torch"], **packages},
        "torch_cuda_build": build["cuda_build"],
        "candidate_package_installed": {name: version is not None for name, version in packages.items()},
        "candidate_cache": _bounded_cache_inventory(config["repository_audit"]["bounded_cache_roots"]),
        "pyproject_candidate_dependencies": [],
        "existing_candidate_model_integration": False,
        "proteinmpnn_reference_scope": "pseudo_cb_convention_only",
        "network_accessed": False,
    }


def _phase4b_blockers(config: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for name, candidate in config["phase4b"]["candidates"].items():
        if candidate["permitted"] and not candidate.get("revision"):
            blockers.append(f"{name}:{candidate['revision_status']}")
    if not config["phase4b"]["execution_authorized"]:
        blockers.append("phase4b_execution_not_authorized")
    return blockers


def plan_phase4b(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    verify_prerequisites(config)
    output = Path(config["phase4b"]["output_dir"])
    staging = Path(config["phase4b"]["staging_dir"])
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase 4B output exists: {output} or {staging}")
    cases = [
        {"candidate": candidate, "length": length, "case": case}
        for candidate, details in config["phase4b"]["candidates"].items()
        if details["permitted"]
        for length in config["phase4b"]["lengths"]
        for case in config["phase4b"]["cases"]
    ]
    return {
        "status": "planned_non_authorizing_blocked_pending_revisions_and_authorization",
        "version": PHASE4B_VERSION,
        "configuration_sha256": sha256_file(config_path),
        "case_count": len(cases),
        "cases": cases,
        "blockers": _phase4b_blockers(config),
        "one_candidate_case_per_child_process": True,
        "weight_download_performed": False,
        "model_loaded": False,
        "network_accessed": False,
        "cache_written": False,
        "output_created": False,
        "output_dir": str(output),
        **NON_AUTHORIZING,
    }


def plan_audit(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase 4A output exists: {output} or {staging}")
    prerequisites = verify_prerequisites(config)
    environment = repository_environment_audit(config)
    phase4b = plan_phase4b(config_path)
    return {
        "status": "planned_read_only_non_authorizing",
        "version": VERSION,
        "configuration_sha256": sha256_file(config_path),
        "contract_sha256": prerequisites["hashes"]["contract"],
        "outcome": OUTCOME,
        "shortlist": config["decision"]["shortlisted_candidates"],
        "baseline": config["decision"]["baseline_candidate"],
        "deferred": config["decision"]["deferred_candidate"],
        "verified_prerequisite_hashes": prerequisites["hashes"],
        "dataset_vocabulary": prerequisites["vocabulary"],
        "environment": environment,
        "phase4b": phase4b,
        "network_accessed": False,
        "weight_cache_written": False,
        "output_created": False,
        "output_dir": str(output),
        **NON_AUTHORIZING,
    }


def publish_audit(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase 4A output exists: {output} or {staging}")
    before = verify_prerequisites(config)
    plan = plan_audit(config_path)
    staging.mkdir(parents=True)
    heartbeat = staging / "heartbeat.json"
    _atomic_json(heartbeat, {"status": "publishing", "updated_utc": _utc_now(), **NON_AUTHORIZING})
    try:
        report = {
            "status": "completed_read_only_non_authorizing",
            "version": VERSION,
            "outcome": OUTCOME,
            "candidate_comparison": before["contract"]["verified_facts"]["models"],
            "estimates": before["contract"]["estimates"],
            "unresolved_questions": before["contract"]["unresolved_questions"],
            "conditioning_contract": before["contract"]["scientific_contract"],
            "repository_environment": plan["environment"],
            "phase4b_plan": plan["phase4b"],
            "protected_inputs_unchanged": True,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "report.json", report)
        inventory_rows = [
            {
                "path": "report.json",
                "size_bytes": (staging / "report.json").stat().st_size,
                "sha256": sha256_file(staging / "report.json"),
            }
        ]
        inventory = {"artifacts": inventory_rows, "aggregate_sha256": _canonical_sha(inventory_rows)}
        _atomic_json(staging / "artifact_inventory.json", inventory)
        protocol = {
            "status": report["status"],
            "version": VERSION,
            "configuration_sha256": sha256_file(config_path),
            "contract_sha256": before["hashes"]["contract"],
            "report_sha256": inventory_rows[0]["sha256"],
            "artifact_inventory_sha256": sha256_file(staging / "artifact_inventory.json"),
            "source_hashes": before["hashes"],
            "completed_utc": _utc_now(),
            "network_accessed": False,
            "weight_cache_written": False,
            "protected_inputs_unchanged": True,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "protocol.json", protocol)
        if (staging / "report.json").read_bytes() == (staging / "protocol.json").read_bytes():
            raise ValueError("E007 Phase 4A report/protocol publication separation failure")
        after = verify_prerequisites(config)
        if after["hashes"] != before["hashes"]:
            raise ValueError("E007 Phase 4A protected inputs changed during publication")
        _atomic_json(
            heartbeat,
            {
                "status": "completed",
                "completed_utc": _utc_now(),
                "report_sha256": protocol["report_sha256"],
                **NON_AUTHORIZING,
            },
        )
        staging.replace(output)
        return {"status": report["status"], "output_dir": str(output), "outcome": OUTCOME, **NON_AUTHORIZING}
    except BaseException as error:
        _atomic_json(
            heartbeat,
            {
                "status": "failed",
                "failed_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
                **NON_AUTHORIZING,
            },
        )
        raise
