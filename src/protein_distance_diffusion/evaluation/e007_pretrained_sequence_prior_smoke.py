"""E007 Phase 4B immutable artifact acquisition and isolated feasibility smoke."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from protein_distance_diffusion.evaluation import e007_pretrained_sequence_prior as phase4a

VERSION = "e007_pretrained_sequence_prior_smoke_v1"
VERSION_V2 = "e007_pretrained_sequence_prior_smoke_v2"
VERSION_V3 = "e007_pretrained_sequence_prior_smoke_v3"
SUPPORTED_VERSIONS = {VERSION, VERSION_V2, VERSION_V3}
ARTIFACT_LOCK_VERSION = "e007_pretrained_sequence_prior_artifact_lock_v1"
ENVIRONMENT_LOCK_VERSION = "e007_phase4b_environment_lock_v1"
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
HEX40 = frozenset("0123456789abcdef")
NON_AUTHORIZING = {
    "authorizes_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_joint_training": False,
    "authorizes_production_training": False,
    "authorizes_additional_coordinate_training": False,
}
NO_WORK = {
    "model_created": False,
    "cuda_used": False,
    "optimizer_created": False,
    "optimizer_updates": 0,
    "training_performed": False,
    "sampling_performed": False,
    "dataset_scanned": False,
    "dataset_modified": False,
}

_SENSITIVE_REDIRECT_HEADERS = ("Authorization", "Cookie", "Proxy-Authorization")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _is_commit(value: Any) -> bool:
    text = str(value)
    return len(text) == 40 and all(character in HEX40 for character in text)


def _contained(root: Path, relative: str) -> Path:
    logical = PurePosixPath(relative)
    if logical.is_absolute() or ".." in logical.parts or not logical.parts:
        raise ValueError(f"E007 Phase 4B unsafe relative path: {relative}")
    target = (root / Path(*logical.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"E007 Phase 4B path escapes root: {relative}")
    return target


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") not in SUPPORTED_VERSIONS:
        raise ValueError("E007 Phase 4B configuration version contradiction")
    requested_version = payload["version"]
    if requested_version in {VERSION_V2, VERSION_V3}:
        base_path = Path(str(payload.get("base_configuration", "")))
        if not base_path.is_file():
            raise ValueError("E007 Phase 4B.1 base configuration is absent")
        if sha256_file(base_path) != payload.get("base_configuration_sha256"):
            raise ValueError("E007 Phase 4B.1 base configuration hash contradiction")
        base = _load_config(base_path)
        expected_base_version = VERSION if requested_version == VERSION_V2 else VERSION_V2
        if base.get("version") != expected_base_version:
            raise ValueError("E007 Phase 4B.1 base configuration version contradiction")
        payload = {**deepcopy(base), **payload}
    if payload.get("smoke", {}).get("lengths") != [64, 128, 256, 384, 500]:
        raise ValueError("E007 Phase 4B length panel changed")
    if int(payload["smoke"].get("optimizer_updates", -1)) != 0:
        raise ValueError("E007 Phase 4B must perform zero optimizer updates")
    expected_candidates = (
        ["esm2_150m", "progen2_151m"]
        if payload["version"] == VERSION_V3
        else ["esm2_150m", "progen2_151m", "proteinmpnn_ca_only"]
    )
    if payload["smoke"].get("candidates") != expected_candidates:
        raise ValueError("E007 Phase 4B candidate set changed")
    if payload["smoke"]["tokenizer"].get("canonical_residues") != CANONICAL:
        raise ValueError("E007 Phase 4B canonical tokenizer contract changed")
    for name, source in payload.get("sources", {}).items():
        if not _is_commit(source.get("source_commit")):
            raise ValueError(f"E007 Phase 4B source commit is not immutable: {name}")
        if name == "esm2_150m":
            if not _is_commit(source.get("model_commit")) or not _is_commit(source.get("phase4a_config_commit")):
                raise ValueError("E007 Phase 4B ESM model/config commit is not immutable")
        if source.get("trust_remote_code") is not False:
            raise ValueError(f"E007 Phase 4B remote-code execution enabled: {name}")
        for artifact in source.get("artifacts", []):
            _contained(Path("/tmp/e007-path-validation"), str(artifact["path"]))
            parsed = urllib.parse.urlparse(str(artifact["url"]))
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError(f"E007 Phase 4B non-HTTPS artifact URL: {name}")
            if parsed.hostname == "raw.githubusercontent.com" and source["source_commit"] not in parsed.path:
                raise ValueError(f"E007 Phase 4B mutable GitHub artifact URL: {name}")
            if parsed.hostname == "huggingface.co" and source["model_commit"] not in parsed.path:
                raise ValueError(f"E007 Phase 4B mutable Hugging Face artifact URL: {name}")
    if payload["sources"]["proteinmpnn_ca_only"].get("full_backbone_checkpoint_forbidden") is not True:
        raise ValueError("E007 Phase 4B ProteinMPNN full-backbone guard changed")
    acquisition = payload["acquisition"]
    if int(acquisition.get("maximum_redirects", 0)) < 1:
        raise ValueError("E007 Phase 4B maximum redirect count must be positive")
    for field in (
        "archive_maximum_members",
        "archive_maximum_file_bytes",
        "archive_maximum_total_bytes",
    ):
        if int(acquisition.get(field, 0)) < 1:
            raise ValueError(f"E007 Phase 4B archive limit must be positive: {field}")
    if int(acquisition["archive_maximum_file_bytes"]) > int(acquisition["archive_maximum_total_bytes"]):
        raise ValueError("E007 Phase 4B per-file archive limit exceeds total limit")
    policies = acquisition.get("redirect_policies")
    if not isinstance(policies, dict):
        raise ValueError("E007 Phase 4B source-specific redirect policies are required")
    expected_origins = {
        "huggingface": "huggingface.co",
        "github_raw": "raw.githubusercontent.com",
        "google_storage": "storage.googleapis.com",
    }
    for name, origin in expected_origins.items():
        policy = policies.get(name)
        if not isinstance(policy, dict) or policy.get("source_hosts") != [origin]:
            raise ValueError(f"E007 Phase 4B redirect policy contradiction: {name}")
    if policies["huggingface"].get("destination_host_suffixes") != ["hf.co"]:
        raise ValueError("E007 Phase 4B Hugging Face delivery suffix policy changed")
    return payload


def _verify_v1_predecessor(config: Mapping[str, Any]) -> dict[str, str]:
    if config["version"] not in {VERSION_V2, VERSION_V3}:
        return {}
    names = ["predecessor_v1"] + (["predecessor_v2"] if config["version"] == VERSION_V3 else [])
    results: dict[str, str] = {}
    for section_name in names:
        section = config.get(section_name)
        if not isinstance(section, Mapping):
            raise ValueError(f"E007 predecessor contract is absent: {section_name}")
        root = Path(str(section["path"]))
        expected = section.get("hashes")
        if not root.is_dir() or not isinstance(expected, Mapping):
            raise ValueError(f"E007 predecessor publication is absent: {section_name}")
        observed = {name: sha256_file(root / name) for name in expected}
        if observed != dict(expected):
            raise ValueError(f"E007 immutable predecessor hash contradiction: {section_name}")
        results.update({f"{section_name}_{name.removesuffix('.json')}": value for name, value in observed.items()})
    return results


def _verify_v3_cpu_diagnostic(config: Mapping[str, Any]) -> dict[str, str]:
    if config["version"] != VERSION_V3:
        return {}
    section = config.get("cpu_diagnostic")
    if not isinstance(section, Mapping):
        raise ValueError("E007 Phase 4B.2 CPU diagnostic contract is absent")
    root = Path(str(section["path"]))
    expected = section.get("hashes")
    if not root.is_dir() or not isinstance(expected, Mapping):
        raise ValueError("E007 Phase 4B.2 CPU diagnostic publication is absent")
    observed = {name: sha256_file(root / name) for name in expected}
    if observed != dict(expected):
        raise ValueError("E007 Phase 4B.2 CPU diagnostic hash contradiction")
    report = json.loads((root / "report.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if (
        report.get("status") != "completed_non_authorizing_cpu_diagnostic"
        or protocol.get("status") != report["status"]
        or report.get("cuda_used") is not False
        or report.get("optimizer_updates") != 0
        or report.get("authorizes_training") is not False
    ):
        raise ValueError("E007 Phase 4B.2 CPU diagnostic semantic contract contradiction")
    return {f"cpu_diagnostic_{name.removesuffix('.json')}": value for name, value in observed.items()}


def _verify_phase4a(config: Mapping[str, Any]) -> dict[str, str]:
    section = config["phase4a"]
    hashes = {
        "phase4a_config": phase4a.sha256_file(section["config_path"]),
        "phase4a_contract": phase4a.sha256_file(section["contract_path"]),
    }
    if hashes["phase4a_config"] != section["config_sha256"]:
        raise ValueError("E007 Phase 4B Phase 4A configuration hash contradiction")
    if hashes["phase4a_contract"] != section["contract_sha256"]:
        raise ValueError("E007 Phase 4B Phase 4A contract hash contradiction")
    phase4a_config = phase4a._load_config(section["config_path"])
    verified = phase4a.verify_prerequisites(phase4a_config)
    hashes.update({f"phase4a_{key}": value for key, value in verified["hashes"].items()})
    environment = config["environment_lock"]
    hashes["environment_lock"] = phase4a.sha256_file(environment["path"])
    if hashes["environment_lock"] != environment["sha256"]:
        raise ValueError("E007 Phase 4B environment-lock hash contradiction")
    lock = yaml.safe_load(Path(environment["path"]).read_text())
    if lock.get("version") != ENVIRONMENT_LOCK_VERSION or lock["constraints"].get("trust_remote_code") is not False:
        raise ValueError("E007 Phase 4B environment-lock contract contradiction")
    return hashes


def _verify_loader_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    from protein_distance_diffusion.evaluation.e007_pretrained_loaders import (
        declared_environment_fingerprint,
        environment_contract,
    )

    if config["version"] == VERSION_V3:
        from protein_distance_diffusion.evaluation.e007_pretrained_loaders_v3 import EXPECTED

        expected_readiness_version = "e007_phase4b_loader_readiness_v3"
    elif config["version"] == VERSION_V2:
        from protein_distance_diffusion.evaluation.e007_pretrained_loaders_v2 import EXPECTED

        expected_readiness_version = "e007_phase4b_loader_readiness_v2"
    else:
        from protein_distance_diffusion.evaluation.e007_pretrained_loaders import EXPECTED

        expected_readiness_version = "e007_phase4b_loader_readiness_v1"

    section = config.get("loader_readiness")
    if not isinstance(section, Mapping):
        raise ValueError("E007 Phase 4B loader-readiness contract is absent")
    path = Path(str(section["path"]))
    if sha256_file(path) != section["sha256"]:
        raise ValueError("E007 Phase 4B loader-readiness hash contradiction")
    readiness = yaml.safe_load(path.read_text())
    if readiness.get("version") != expected_readiness_version:
        raise ValueError("E007 Phase 4B loader-readiness version contradiction")
    environment = readiness["installable_environment"]
    if sha256_file(environment["path"]) != environment["sha256"]:
        raise ValueError("E007 Phase 4B installable environment hash contradiction")
    environment_contract(environment["path"])
    if declared_environment_fingerprint(environment["path"]) != environment["fingerprint"]:
        raise ValueError("E007 Phase 4B environment fingerprint contradiction")
    for name, loader in readiness["loaders"].items():
        if name not in EXPECTED or loader.get("status") != "reviewed_ready":
            raise ValueError(f"E007 Phase 4B candidate loader is not reviewed: {name}")
        for source in loader["project_owned_sources"]:
            if sha256_file(source["path"]) != source["sha256"]:
                raise ValueError(f"E007 Phase 4B reviewed loader source hash contradiction: {name}")
    return readiness


def _artifact_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    paths: set[str] = set()
    for candidate, source in config["sources"].items():
        for artifact in source["artifacts"]:
            logical = str(artifact["path"])
            if logical in paths:
                raise ValueError(f"E007 Phase 4B duplicate configured artifact path: {logical}")
            paths.add(logical)
            rows.append({"candidate": candidate, **artifact})
    return rows


def unresolved_artifact_hashes(config: Mapping[str, Any]) -> list[str]:
    return [str(row["path"]) for row in _artifact_specs(config) if not row.get("sha256")]


def plan(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    hashes = _verify_phase4a(config)
    hashes.update(_verify_v1_predecessor(config))
    hashes.update(_verify_v3_cpu_diagnostic(config))
    readiness = _verify_loader_readiness(config)
    paths = [
        Path(config["smoke_output"]),
        Path(config["smoke_output"]).with_name(f".{Path(config['smoke_output']).name}.inprogress"),
    ]
    if any(path.exists() for path in paths):
        raise FileExistsError("E007 Phase 4B final or staging output already exists")
    cases = [
        {"candidate": candidate, "length": length, "test": test}
        for candidate in config["smoke"]["candidates"]
        for length in config["smoke"]["lengths"]
        for test in config["smoke"]["tests"]
    ]
    lock_exists = Path(config["artifact_lock_output"]).is_dir()
    locked = verify_artifacts_offline(config_path) if lock_exists else None
    return {
        "status": "planned_non_authorizing",
        "version": config["version"],
        "configuration_sha256": sha256_file(config_path),
        "source_commits": {name: source["source_commit"] for name, source in config["sources"].items()},
        "esm_model_commit": config["sources"]["esm2_150m"]["model_commit"],
        "artifact_count": len(_artifact_specs(config)),
        "unresolved_artifact_hashes": [] if locked else unresolved_artifact_hashes(config),
        "artifact_lock_status": locked["status"] if locked else "not_acquired",
        "smoke_case_count": len(cases),
        "isolated_candidate_process_count": len(config["smoke"]["candidates"]),
        "protected_hashes": hashes,
        "environment_fingerprint": readiness["installable_environment"]["fingerprint"],
        "loader_statuses": {name: row["status"] for name, row in readiness["loaders"].items()},
        "network_accessed": False,
        "cache_written": False,
        "package_installed": False,
        "output_created": False,
        **NO_WORK,
        **NON_AUTHORIZING,
    }


def _normalized_host(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.hostname is None:
        raise ValueError(f"E007 Phase 4B redirect URL lacks a hostname: {url[:160]}")
    return parsed.hostname.rstrip(".").lower().encode("idna").decode("ascii")


def _destination_path_category(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    host = _normalized_host(url)
    if host == "huggingface.co" and "/resolve/" in parsed.path:
        return "huggingface_pinned_resolve"
    if host == "hf.co" or host.endswith(".hf.co"):
        return "huggingface_signed_delivery"
    if host == "raw.githubusercontent.com":
        return "github_raw_content"
    if host == "storage.googleapis.com":
        return "google_storage_object"
    return "other"


def _has_signed_delivery_parameters(url: str) -> bool:
    names = {name.lower() for name, _value in urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query)}
    signatures = {
        "expires",
        "key-pair-id",
        "policy",
        "signature",
        "x-amz-signature",
        "x-goog-signature",
        "x-xet-cas-uid",
    }
    return bool(names & signatures)


def _policy_for_origin(settings: Mapping[str, Any], origin_host: str) -> tuple[str, Mapping[str, Any]]:
    for name, policy in settings["redirect_policies"].items():
        if origin_host in policy.get("source_hosts", []):
            return str(name), policy
    raise ValueError(f"E007 Phase 4B redirect origin has no explicit policy: {origin_host}")


def _host_matches_policy(host: str, policy: Mapping[str, Any]) -> bool:
    if host in policy.get("destination_hosts", []):
        return True
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in policy.get("destination_host_suffixes", []))


def _validate_redirect(
    source_url: str,
    destination_url: str,
    *,
    origin_url: str,
    status_code: int,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    source = urllib.parse.urlsplit(source_url)
    destination = urllib.parse.urlsplit(destination_url)
    source_host = _normalized_host(source_url)
    destination_host = _normalized_host(destination_url)
    origin_host = _normalized_host(origin_url)
    policy_name, policy = _policy_for_origin(settings, origin_host)
    if source.scheme.lower() != "https" or destination.scheme.lower() != "https":
        raise ValueError("E007 Phase 4B HTTPS redirect downgrade refused")
    if destination.username is not None or destination.password is not None:
        raise ValueError("E007 Phase 4B redirect URL credentials refused")
    if source_host != destination_host and not _host_matches_policy(destination_host, policy):
        raise ValueError(f"E007 Phase 4B cross-host redirect refused: {source_host} -> {destination_host}")
    return {
        "status_code": int(status_code),
        "source_scheme": source.scheme.lower(),
        "source_host": source_host,
        "destination_scheme": destination.scheme.lower(),
        "destination_host": destination_host,
        "destination_path_category": _destination_path_category(destination_url),
        "signed_delivery_parameters": _has_signed_delivery_parameters(destination_url),
        "policy": policy_name,
        "cross_host": source_host != destination_host,
        "authorization_headers_forwarded": False,
    }


class _SecureRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, settings: Mapping[str, Any], origin_url: str) -> None:
        self.settings = settings
        self.origin_url = origin_url
        self.redirect_chain: list[dict[str, Any]] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        if len(self.redirect_chain) >= int(self.settings["maximum_redirects"]):
            raise ValueError("E007 Phase 4B maximum redirect count exceeded")
        hop = _validate_redirect(
            req.full_url,
            newurl,
            origin_url=self.origin_url,
            status_code=code,
            settings=self.settings,
        )
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        for header in _SENSITIVE_REDIRECT_HEADERS:
            redirected.remove_header(header)
            redirected.unredirected_hdrs.pop(header, None)
            redirected.unredirected_hdrs.pop(header.lower(), None)
        self.redirect_chain.append(hop)
        return redirected


def _download_file(spec: Mapping[str, Any], target: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    settings = config["acquisition"]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}{settings['atomic_temporary_suffix']}")
    if temporary.exists():
        temporary.unlink()
    source_url = str(spec["url"])
    request = urllib.request.Request(source_url, headers={"User-Agent": "proteinGen-E007-Phase4B/1"})
    redirect_handler = _SecureRedirectHandler(settings, source_url)
    opener = urllib.request.build_opener(redirect_handler)
    digest = hashlib.sha256()
    size = 0
    try:
        with opener.open(request, timeout=float(settings["connect_timeout_seconds"])) as response:  # noqa: S310
            final_url = response.geturl()
            final_host = _normalized_host(final_url)
            origin_host = _normalized_host(source_url)
            if final_host != origin_host:
                _policy_name, policy = _policy_for_origin(settings, origin_host)
                if not _host_matches_policy(final_host, policy):
                    raise ValueError(f"E007 Phase 4B final response host refused: {final_host}")
            redirect_chain = [*redirect_handler.redirect_chain]
            redirect_chain.append(
                {
                    "status_code": int(response.getcode()),
                    "source_scheme": urllib.parse.urlsplit(final_url).scheme.lower(),
                    "source_host": final_host,
                    "destination_scheme": None,
                    "destination_host": None,
                    "destination_path_category": _destination_path_category(final_url),
                    "signed_delivery_parameters": _has_signed_delivery_parameters(final_url),
                    "policy": _policy_for_origin(settings, origin_host)[0],
                    "cross_host": False,
                    "authorization_headers_forwarded": False,
                }
            )
            with temporary.open("wb") as handle:
                while chunk := response.read(int(settings["read_chunk_bytes"])):
                    size += len(chunk)
                    if size > int(settings["maximum_artifact_bytes"]):
                        raise ValueError(f"E007 Phase 4B artifact exceeds size limit: {spec['path']}")
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        observed = digest.hexdigest()
        expected = spec.get("sha256")
        if expected is not None and observed != expected:
            raise ValueError(f"E007 Phase 4B artifact SHA-256 contradiction: {spec['path']}")
        temporary.replace(target)
        return {
            "size_bytes": size,
            "sha256": observed,
            "resolved_url": source_url,
            "final_host": final_host,
            "redirect_chain": redirect_chain,
        }
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _normalized_member_name(name: str) -> str:
    if "\x00" in name:
        raise ValueError("E007 Phase 4B archive member contains NUL")
    if "\\" in name:
        raise ValueError(f"E007 Phase 4B archive backslash ambiguity refused: {name}")
    if name.startswith("/") or name.startswith("//") or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"E007 Phase 4B absolute archive member refused: {name}")
    value = name
    while value.startswith("./"):
        value = value[2:]
    if value in {"", "."}:
        raise ValueError(f"E007 Phase 4B empty archive member refused: {name}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"E007 Phase 4B unsafe archive member: {name}")
    logical = PurePosixPath(*parts)
    if logical.is_absolute() or ".." in logical.parts or not logical.parts:
        raise ValueError(f"E007 Phase 4B unsafe archive member: {name}")
    return logical.as_posix()


def _preflight_tar(
    archive_path: str | Path,
    *,
    expected_members: Sequence[str],
    maximum_members: int,
    maximum_file_bytes: int,
    maximum_total_bytes: int,
) -> list[dict[str, Any]]:
    expected = set(expected_members)
    if len(expected) != len(expected_members):
        raise ValueError("E007 Phase 4B duplicate expected archive member")
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > maximum_members:
            raise ValueError("E007 Phase 4B archive member count exceeds limit")
        if archive.pax_headers:
            raise ValueError("E007 Phase 4B global PAX headers are unsupported")
        regular: list[tuple[tarfile.TarInfo, str]] = []
        seen_source: set[str] = set()
        seen_source_casefold: dict[str, str] = {}
        total = 0
        for member in members:
            if member.name in {".", "./"}:
                if not member.isdir():
                    raise ValueError("E007 Phase 4B archive root marker is not a directory")
                continue
            name = _normalized_member_name(member.name)
            folded = name.casefold()
            if name in seen_source:
                raise ValueError(f"E007 Phase 4B duplicate archive member: {name}")
            if folded in seen_source_casefold and seen_source_casefold[folded] != name:
                previous = seen_source_casefold[folded]
                raise ValueError(f"E007 Phase 4B case-colliding archive members: {previous} and {name}")
            seen_source.add(name)
            seen_source_casefold[folded] = name
            if member.issym() or member.islnk():
                raise ValueError(f"E007 Phase 4B archive link refused: {name}")
            if getattr(member, "sparse", None):
                raise ValueError(f"E007 Phase 4B sparse archive member refused: {name}")
            if member.pax_headers:
                raise ValueError(f"E007 Phase 4B PAX archive member refused: {name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"E007 Phase 4B non-regular archive member refused: {name}")
            if int(member.size) > maximum_file_bytes:
                raise ValueError(f"E007 Phase 4B archive member exceeds size limit: {name}")
            total += int(member.size)
            if total > maximum_total_bytes:
                raise ValueError("E007 Phase 4B extracted archive exceeds size limit")
            regular.append((member, name))

        prefixes = {name.split("/", 1)[0] for _member, name in regular if "/" in name}
        has_common_prefix = len(prefixes) == 1 and all("/" in name for _member, name in regular)
        strip_prefix = next(iter(prefixes)) if has_common_prefix else None
        plan: list[dict[str, Any]] = []
        outputs: set[str] = set()
        output_casefold: dict[str, str] = {}
        for member, source_name in regular:
            logical = source_name.split("/", 1)[1] if strip_prefix else source_name
            logical = _normalized_member_name(logical)
            folded = logical.casefold()
            if logical in outputs:
                raise ValueError(f"E007 Phase 4B duplicate normalized archive member: {logical}")
            if folded in output_casefold and output_casefold[folded] != logical:
                raise ValueError(
                    f"E007 Phase 4B case-colliding normalized archive members: {output_casefold[folded]} and {logical}"
                )
            outputs.add(logical)
            output_casefold[folded] = logical
            plan.append({"source_name": member.name, "path": logical, "size_bytes": int(member.size)})
        extra = outputs - expected
        if extra:
            raise ValueError(f"E007 Phase 4B unexpected archive member: {sorted(extra)}")
        if outputs != expected:
            raise ValueError(f"E007 Phase 4B archive inventory contradiction: missing={sorted(expected - outputs)}")
        return sorted(plan, key=lambda row: row["path"])


def safe_extract_tar(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    expected_members: Sequence[str],
    maximum_members: int = 1024,
    maximum_file_bytes: int = 2 * 1024**3,
    maximum_total_bytes: int = 2 * 1024**3,
) -> list[dict[str, Any]]:
    output = Path(output_dir)
    plan = _preflight_tar(
        archive_path,
        expected_members=expected_members,
        maximum_members=maximum_members,
        maximum_file_bytes=maximum_file_bytes,
        maximum_total_bytes=maximum_total_bytes,
    )
    if output.exists():
        inventory = _artifact_inventory(output)
        if {row["path"] for row in inventory} != set(expected_members):
            raise ValueError("E007 Phase 4B committed extraction inventory contradiction")
        return inventory
    temporary = output.with_name(f".{output.name}.{os.getpid()}.extract.inprogress")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            by_name = {member.name: member for member in archive.getmembers()}
            for item in plan:
                member = by_name[item["source_name"]]
                target = _contained(temporary, str(item["path"]))
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"E007 Phase 4B archive member cannot be read: {item['path']}")
                digest = hashlib.sha256()
                with target.open("xb") as handle:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                if target.stat().st_size != item["size_bytes"]:
                    raise ValueError(f"E007 Phase 4B extracted member size contradiction: {item['path']}")
                rows.append({"path": item["path"], "size_bytes": target.stat().st_size, "sha256": digest.hexdigest()})
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        temporary.replace(output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return sorted(rows, key=lambda row: row["path"])
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != ".acquisition_journal.json"
        and not path.name.endswith(".download.inprogress")
    ]


def _acquisition_journal(root: Path) -> dict[str, Any]:
    path = root / ".acquisition_journal.json"
    if not path.is_file():
        return {"version": VERSION, "committed": {}}
    value = json.loads(path.read_text())
    if value.get("version") != VERSION or not isinstance(value.get("committed"), dict):
        raise ValueError("E007 Phase 4B acquisition journal contract contradiction")
    return value


def _committed_metadata(spec: Mapping[str, Any], target: Path, journal: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = journal.get("committed", {}).get(str(spec["path"]))
    if not isinstance(metadata, dict) or not target.is_file():
        return None
    observed = sha256_file(target)
    if observed != metadata.get("sha256") or target.stat().st_size != int(metadata.get("size_bytes", -1)):
        return None
    if spec.get("sha256") is not None and observed != spec["sha256"]:
        raise ValueError(f"E007 Phase 4B committed artifact hash contradiction: {spec['path']}")
    return dict(metadata)


def _locked_rows_from_cache(config: Mapping[str, Any], cache: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    journal = _acquisition_journal(cache)
    for spec in _artifact_specs(config):
        path = _contained(cache, str(spec["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"E007 Phase 4B resumed cache artifact is absent: {spec['path']}")
        observed = sha256_file(path)
        if spec.get("sha256") is not None and observed != spec["sha256"]:
            raise ValueError(f"E007 Phase 4B resumed cache hash contradiction: {spec['path']}")
        metadata = journal.get("committed", {}).get(str(spec["path"]), {})
        rows.append(
            {
                "candidate": spec["candidate"],
                "path": str(spec["path"]),
                "url": str(spec["url"]),
                "size_bytes": path.stat().st_size,
                "sha256": observed,
                "archive": spec.get("archive"),
                "resolved_url": metadata.get("resolved_url", spec["url"]),
                "final_host": metadata.get("final_host"),
                "redirect_chain": metadata.get("redirect_chain", []),
            }
        )
        if spec.get("archive") == "tar_gz":
            for member in spec["expected_members"]:
                logical = f"{spec['extract_to']}/{member}"
                extracted = _contained(cache, logical)
                if not extracted.is_file():
                    raise FileNotFoundError(f"E007 Phase 4B extracted cache artifact is absent: {logical}")
                rows.append(
                    {
                        "candidate": spec["candidate"],
                        "path": logical,
                        "url": f"archive:{spec['path']}#{member}",
                        "size_bytes": extracted.stat().st_size,
                        "sha256": sha256_file(extracted),
                        "archive": None,
                        "resolved_url": f"archive:{spec['path']}#{member}",
                        "final_host": None,
                        "redirect_chain": [],
                    }
                )
    inventory = _artifact_inventory(cache)
    if {(row["path"], row["sha256"]) for row in rows} != {(row["path"], row["sha256"]) for row in inventory}:
        raise ValueError("E007 Phase 4B resumed cache inventory contains missing or unexpected files")
    return sorted(rows, key=lambda row: row["path"])


def _publish_lock(
    config_path: Path,
    config: Mapping[str, Any],
    cache: Path,
    source_rows: Sequence[Mapping[str, Any]],
    protected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    output = Path(config["artifact_lock_output"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase 4B artifact-lock output exists: {output} or {staging}")
    staging.mkdir(parents=True)
    _atomic_json(staging / "heartbeat.json", {"status": "publishing", "updated_utc": _utc_now(), **NON_AUTHORIZING})
    try:
        source_inventory = {
            "version": ARTIFACT_LOCK_VERSION,
            "cache_root": str(cache),
            "sources": list(source_rows),
            "candidate_sources": {
                name: {
                    "role": source["role"],
                    "source_repository": source["source_repository"],
                    "source_commit": source["source_commit"],
                    "license": source["license"],
                    "loader": source["loader"],
                    "trust_remote_code": source["trust_remote_code"],
                }
                for name, source in config["sources"].items()
            },
            "aggregate_sha256": _canonical_sha(list(source_rows)),
        }
        _atomic_json(staging / "source_inventory.json", source_inventory)
        environment = yaml.safe_load(Path(config["environment_lock"]["path"]).read_text())
        _atomic_json(staging / "environment_lock.json", environment)
        lock = {
            "version": ARTIFACT_LOCK_VERSION,
            "status": "completed_immutable_artifact_acquisition",
            "configuration_sha256": sha256_file(config_path),
            "source_inventory_sha256": sha256_file(staging / "source_inventory.json"),
            "source_inventory_aggregate_sha256": source_inventory["aggregate_sha256"],
            "environment_lock_sha256": sha256_file(staging / "environment_lock.json"),
            "source_commits": {name: source["source_commit"] for name, source in config["sources"].items()},
            "protected_hashes": dict(protected_hashes),
            "downloaded_repository_code_executed": False,
            "cuda_used": False,
            "model_created": False,
            "artifact_count": len(source_rows),
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "artifact_lock.json", lock)
        report = {
            "status": lock["status"],
            "version": ARTIFACT_LOCK_VERSION,
            "artifacts_acquired": len(source_rows),
            "models_loaded": 0,
            "protected_inputs_unchanged": True,
            "offline_verification_required_before_smoke": True,
            **NO_WORK,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "report.json", report)
        protocol = {
            "status": lock["status"],
            "version": ARTIFACT_LOCK_VERSION,
            "configuration_sha256": sha256_file(config_path),
            "artifact_lock_sha256": sha256_file(staging / "artifact_lock.json"),
            "source_inventory_sha256": lock["source_inventory_sha256"],
            "environment_lock_sha256": lock["environment_lock_sha256"],
            "completed_utc": _utc_now(),
            **NO_WORK,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "protocol.json", protocol)
        durable = [
            "artifact_lock.json",
            "source_inventory.json",
            "environment_lock.json",
            "report.json",
            "protocol.json",
        ]
        inventory_rows = [
            {
                "path": name,
                "size_bytes": (staging / name).stat().st_size,
                "sha256": sha256_file(staging / name),
            }
            for name in durable
        ]
        _atomic_json(
            staging / "artifact_inventory.json",
            {"artifacts": inventory_rows, "aggregate_sha256": _canonical_sha(inventory_rows)},
        )
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "completed",
                "completed_utc": _utc_now(),
                "protocol_sha256": sha256_file(staging / "protocol.json"),
                **NON_AUTHORIZING,
            },
        )
        staging.replace(output)
        return lock
    except BaseException as error:
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "failed",
                "failed_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
                **NON_AUTHORIZING,
            },
        )
        raise


def acquire_artifacts(
    config_path: str | Path,
    *,
    download_file: Callable[[Mapping[str, Any], Path, Mapping[str, Any]], Mapping[str, Any]] = _download_file,
    resume: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    protected_before = _verify_phase4a(config)
    cache = Path(config["artifact_cache_root"])
    staging = cache.with_name(f".{cache.name}.acquisition.inprogress")
    if cache.exists() and not resume:
        raise FileExistsError(f"E007 Phase 4B cache already exists: {cache}")
    if Path(config["artifact_lock_output"]).exists():
        raise FileExistsError("E007 Phase 4B artifact lock already exists")
    if resume and cache.is_dir():
        rows = _locked_rows_from_cache(config, cache)
        protected_after = _verify_phase4a(config)
        if protected_after != protected_before:
            raise ValueError("E007 Phase 4B protected inputs changed before acquisition resume")
        lock = _publish_lock(config_path, config, cache, rows, protected_before)
        return {
            "status": lock["status"],
            "artifact_count": len(rows),
            "cache_root": str(cache),
            "artifact_lock_output": config["artifact_lock_output"],
            "resumed_after_cache_promotion": True,
            "model_created": False,
            "cuda_used": False,
            **NON_AUTHORIZING,
        }
    staging.mkdir(parents=True, exist_ok=True)
    journal = _acquisition_journal(staging)
    rows: list[dict[str, Any]] = []
    try:
        for spec in _artifact_specs(config):
            target = _contained(staging, str(spec["path"]))
            temporary = target.with_name(f".{target.name}{config['acquisition']['atomic_temporary_suffix']}")
            temporary.unlink(missing_ok=True)
            metadata = _committed_metadata(spec, target, journal)
            if metadata is None:
                target.unlink(missing_ok=True)
                metadata = dict(download_file(spec, target, config))
                journal["committed"][str(spec["path"])] = metadata
                _atomic_json(staging / ".acquisition_journal.json", journal)
            row = {
                "candidate": spec["candidate"],
                "path": str(spec["path"]),
                "url": str(spec["url"]),
                "size_bytes": int(metadata["size_bytes"]),
                "sha256": str(metadata["sha256"]),
                "archive": spec.get("archive"),
                "resolved_url": str(metadata.get("resolved_url", spec["url"])),
                "final_host": metadata.get("final_host"),
                "redirect_chain": list(metadata.get("redirect_chain", [])),
            }
            rows.append(row)
            if spec.get("archive") == "tar_gz":
                extract_dir = _contained(staging, str(spec["extract_to"]))
                extracted = safe_extract_tar(
                    target,
                    extract_dir,
                    expected_members=spec["expected_members"],
                    maximum_members=int(config["acquisition"]["archive_maximum_members"]),
                    maximum_file_bytes=int(config["acquisition"]["archive_maximum_file_bytes"]),
                    maximum_total_bytes=int(config["acquisition"]["archive_maximum_total_bytes"]),
                )
                rows.extend(
                    {
                        "candidate": spec["candidate"],
                        "path": f"{spec['extract_to']}/{item['path']}",
                        "url": f"archive:{spec['path']}#{item['path']}",
                        "size_bytes": item["size_bytes"],
                        "sha256": item["sha256"],
                        "archive": None,
                        "resolved_url": f"archive:{spec['path']}#{item['path']}",
                        "final_host": None,
                        "redirect_chain": [],
                    }
                    for item in extracted
                )
        observed_inventory = _artifact_inventory(staging)
        if {(row["path"], row["sha256"]) for row in rows} != {
            (row["path"], row["sha256"]) for row in observed_inventory
        }:
            raise ValueError("E007 Phase 4B acquired cache inventory contradiction")
        if cache.exists():
            raise FileExistsError(f"E007 Phase 4B refuses to overwrite cache: {cache}")
        staging.replace(cache)
        protected_after = _verify_phase4a(config)
        if protected_after != protected_before:
            raise ValueError("E007 Phase 4B protected inputs changed during acquisition")
        lock = _publish_lock(config_path, config, cache, sorted(rows, key=lambda row: row["path"]), protected_before)
        return {
            "status": lock["status"],
            "artifact_count": len(rows),
            "cache_root": str(cache),
            "artifact_lock_output": config["artifact_lock_output"],
            "model_created": False,
            "cuda_used": False,
            **NON_AUTHORIZING,
        }
    except BaseException:
        raise


def verify_artifacts_offline(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    protected_before = _verify_phase4a(config)
    protected_before.update(_verify_v1_predecessor(config))
    protected_before.update(_verify_v3_cpu_diagnostic(config))
    readiness = _verify_loader_readiness(config)
    lock_dir = Path(config["artifact_lock_output"])
    cache = Path(config["artifact_cache_root"])
    required = {
        "artifact_lock.json",
        "source_inventory.json",
        "environment_lock.json",
        "report.json",
        "protocol.json",
        "heartbeat.json",
        "artifact_inventory.json",
    }
    if not lock_dir.is_dir() or not required.issubset(path.name for path in lock_dir.iterdir()):
        raise FileNotFoundError("E007 Phase 4B artifact-lock publication is incomplete")
    lock = json.loads((lock_dir / "artifact_lock.json").read_text())
    protocol = json.loads((lock_dir / "protocol.json").read_text())
    inventory = json.loads((lock_dir / "source_inventory.json").read_text())
    publication_inventory = json.loads((lock_dir / "artifact_inventory.json").read_text())
    if lock.get("status") != "completed_immutable_artifact_acquisition" or protocol.get("status") != lock["status"]:
        raise ValueError("E007 Phase 4B artifact-lock status contradiction")
    authoritative_lock = Path(readiness["artifact_lock"]["path"]).resolve()
    is_authoritative_lock = lock_dir.resolve() == authoritative_lock
    expected_acquisition_config = (
        readiness["artifact_lock"]["acquisition_configuration_sha256"]
        if is_authoritative_lock
        else sha256_file(config_path)
    )
    if lock.get("configuration_sha256") != expected_acquisition_config:
        raise ValueError("E007 Phase 4B artifact-lock configuration contradiction")
    if is_authoritative_lock:
        if sha256_file(lock_dir / "artifact_lock.json") != readiness["artifact_lock"]["artifact_lock_sha256"]:
            raise ValueError("E007 Phase 4B pinned artifact-lock hash contradiction")
        if sha256_file(lock_dir / "source_inventory.json") != readiness["artifact_lock"]["source_inventory_sha256"]:
            raise ValueError("E007 Phase 4B pinned source-inventory hash contradiction")
        if inventory.get("aggregate_sha256") != readiness["artifact_lock"]["source_inventory_aggregate_sha256"]:
            raise ValueError("E007 Phase 4B pinned source-inventory aggregate contradiction")
    if sha256_file(lock_dir / "source_inventory.json") != lock["source_inventory_sha256"]:
        raise ValueError("E007 Phase 4B source-inventory hash contradiction")
    if sha256_file(lock_dir / "artifact_lock.json") != protocol["artifact_lock_sha256"]:
        raise ValueError("E007 Phase 4B artifact-lock publication hash contradiction")
    if sha256_file(lock_dir / "environment_lock.json") != protocol["environment_lock_sha256"]:
        raise ValueError("E007 Phase 4B environment-lock publication hash contradiction")
    publication_rows = publication_inventory.get("artifacts", [])
    if _canonical_sha(publication_rows) != publication_inventory.get("aggregate_sha256"):
        raise ValueError("E007 Phase 4B publication inventory aggregate contradiction")
    for row in publication_rows:
        path = _contained(lock_dir, str(row["path"]))
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"E007 Phase 4B publication inventory size contradiction: {row['path']}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"E007 Phase 4B publication inventory hash contradiction: {row['path']}")
    if _canonical_sha(inventory["sources"]) != inventory["aggregate_sha256"]:
        raise ValueError("E007 Phase 4B source-inventory aggregate contradiction")
    seen: set[str] = set()
    for row in inventory["sources"]:
        logical = str(row["path"])
        if logical in seen:
            raise ValueError(f"E007 Phase 4B duplicate locked artifact: {logical}")
        seen.add(logical)
        path = _contained(cache, logical)
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"E007 Phase 4B locked artifact size contradiction: {logical}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"E007 Phase 4B locked artifact hash contradiction: {logical}")
    protected_after = _verify_phase4a(config)
    protected_after.update(_verify_v1_predecessor(config))
    protected_after.update(_verify_v3_cpu_diagnostic(config))
    acquisition_protected = {
        key: value
        for key, value in protected_before.items()
        if not key.startswith("predecessor_") and not key.startswith("cpu_diagnostic_")
    }
    if protected_after != protected_before or lock.get("protected_hashes") != acquisition_protected:
        raise ValueError("E007 Phase 4B protected-input verification contradiction")
    return {
        "status": "verified_offline",
        "artifact_count": len(seen),
        "source_inventory_aggregate_sha256": inventory["aggregate_sha256"],
        "network_accessed": False,
        "model_created": False,
        "cuda_used": False,
        "protected_inputs_unchanged": True,
        "environment_fingerprint": readiness["installable_environment"]["fingerprint"],
        **NON_AUTHORIZING,
    }


def verify_environment_readiness(config_path: str | Path, *, require_environment_name: bool = True) -> dict[str, Any]:
    from protein_distance_diffusion.evaluation.e007_pretrained_loaders import verify_runtime_environment

    config = _load_config(config_path)
    readiness = _verify_loader_readiness(config)
    result = verify_runtime_environment(
        readiness["installable_environment"]["path"], require_name=require_environment_name
    )
    return {
        **result,
        "offline_environment": readiness["offline_contract"],
        "loader_statuses": {name: row["status"] for name, row in readiness["loaders"].items()},
        "model_created": False,
        "cuda_used": False,
        **NON_AUTHORIZING,
    }


def validate_loader_metadata(config_path: str | Path) -> dict[str, Any]:
    from protein_distance_diffusion.evaluation.e007_pretrained_loaders import (
        safe_torch_checkpoint_metadata,
        safetensors_metadata,
        validate_tensor_mapping,
    )

    config = _load_config(config_path)
    if config["version"] == VERSION_V3:
        from protein_distance_diffusion.evaluation.e007_pretrained_loaders_v3 import EXPECTED

        esm_tensor_count = EXPECTED["esm2_150m"]["raw_state_tensor_count"]
        progen_state_elements = EXPECTED["progen2_151m"]["state_element_count"]
    elif config["version"] == VERSION_V2:
        from protein_distance_diffusion.evaluation.e007_pretrained_loaders_v2 import EXPECTED

        esm_tensor_count = EXPECTED["esm2_150m"]["raw_state_tensor_count"]
        progen_state_elements = EXPECTED["progen2_151m"]["state_element_count"]
    else:
        from protein_distance_diffusion.evaluation.e007_pretrained_loaders import EXPECTED

        esm_tensor_count = EXPECTED["esm2_150m"]["state_tensor_count"]
        progen_state_elements = EXPECTED["progen2_151m"]["parameter_count"]
    readiness = _verify_loader_readiness(config)
    verify_artifacts_offline(config_path)
    cache = Path(config["artifact_cache_root"])
    esm = safetensors_metadata(cache / "esm2_150m" / "model.safetensors")
    if len(esm) != esm_tensor_count:
        raise ValueError("E007 Phase 4B ESM metadata tensor-count contradiction")
    progen, progen_metadata = safe_torch_checkpoint_metadata(
        cache / "progen2_151m" / "checkpoint" / "pytorch_model.bin"
    )
    progen_result = validate_tensor_mapping(
        progen,
        expected_tensor_count=EXPECTED["progen2_151m"]["state_tensor_count"],
        expected_parameter_count=progen_state_elements,
    )
    mpnn, mpnn_metadata = safe_torch_checkpoint_metadata(
        cache / "proteinmpnn_ca_only" / "ca_model_weights" / "v_48_020.pt",
        nested_key="model_state_dict",
    )
    mpnn_result = validate_tensor_mapping(
        mpnn,
        expected_tensor_count=EXPECTED["proteinmpnn_ca_only"]["state_tensor_count"],
        expected_parameter_count=EXPECTED["proteinmpnn_ca_only"]["parameter_count"],
    )
    if progen_metadata or mpnn_metadata != {"num_edges": 48, "noise_level": 0.2}:
        raise ValueError("E007 Phase 4B checkpoint metadata contradiction")
    return {
        "status": "metadata_validated_without_model_construction",
        "esm": {"state_tensor_count": len(esm)},
        "progen2": progen_result,
        "proteinmpnn_ca_only": mpnn_result,
        "loader_statuses": {name: row["status"] for name, row in readiness["loaders"].items()},
        **NO_WORK,
        **NON_AUTHORIZING,
    }


def _current_rss_mib() -> float:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def _peak_rss_mib() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _cuda_device_baseline(torch: Any, device: Any) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "device_total_mib": None,
            "device_free_before_mib": None,
            "device_wide_baseline_used_mib": None,
        }
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "device_total_mib": total_bytes / 2**20,
        "device_free_before_mib": free_bytes / 2**20,
        "device_wide_baseline_used_mib": (total_bytes - free_bytes) / 2**20,
    }


def _tensor_sha(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _parameter_sha(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _geometry_features(torch: Any, length: int, device: Any) -> Any:
    position = torch.linspace(0.0, 1.0, length, dtype=torch.float32, device=device)
    channels = (
        position,
        torch.sin(position * math.pi),
        torch.cos(position * math.pi),
        torch.ones_like(position),
    )
    return torch.stack(channels, -1)[None]


def _tokenizer_checks(candidate: str, tokenizer: Any, lengths: Sequence[int]) -> dict[str, Any]:
    sequence = CANONICAL
    if candidate == "progen2_151m":
        encoded = tokenizer.encode(f"1{sequence}2")
        ids = encoded.ids
        decoded = tokenizer.decode(ids).replace(" ", "")
    else:
        encoded = tokenizer(sequence, add_special_tokens=True)
        ids = encoded["input_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=True).replace(" ", "")
    if decoded != sequence:
        raise ValueError(f"E007 Phase 4B canonical tokenizer round-trip failed: {candidate}")
    long_sequence = (CANONICAL * 25)[:500]
    if candidate == "progen2_151m":
        long_ids = tokenizer.encode(f"1{long_sequence}2").ids
        long_decoded = tokenizer.decode(long_ids).replace(" ", "")
        canonical_ids = {residue: tokenizer.encode(residue).ids for residue in CANONICAL}
        unknown_id = tokenizer.token_to_id("<|unk|>")
        ambiguous = {}
        for residue in "BJOUXZ":
            residue_ids = tokenizer.encode(residue).ids
            residue_decoded = tokenizer.decode(residue_ids).replace(" ", "")
            status = (
                "exact"
                if residue_decoded == residue
                else "explicit_unknown"
                if unknown_id in residue_ids
                else "substituted"
            )
            ambiguous[residue] = {"token_ids": residue_ids, "decoded": residue_decoded, "status": status}
        special_tokens = {
            "bos": tokenizer.encode("1").ids,
            "eos": tokenizer.encode("2").ids,
            "pad": tokenizer.encode("<|pad|>").ids,
        }
    else:
        long_encoded = tokenizer(long_sequence, add_special_tokens=True)
        long_ids = long_encoded["input_ids"]
        long_decoded = tokenizer.decode(long_ids, skip_special_tokens=True).replace(" ", "")
        canonical_ids = {residue: tokenizer.convert_tokens_to_ids(residue) for residue in CANONICAL}
        ambiguous = {}
        for residue in "BJOUXZ":
            residue_id = tokenizer.convert_tokens_to_ids(residue)
            residue_decoded = tokenizer.decode([residue_id], skip_special_tokens=True).replace(" ", "")
            status = (
                "exact"
                if residue_decoded == residue
                else "explicit_unknown"
                if residue_id == tokenizer.unk_token_id
                else "substituted"
            )
            ambiguous[residue] = {"token_ids": [residue_id], "decoded": residue_decoded, "status": status}
        special_tokens = {
            "bos": tokenizer.cls_token_id,
            "eos": tokenizer.eos_token_id,
            "pad": tokenizer.pad_token_id,
            "mask": tokenizer.mask_token_id,
            "unknown": tokenizer.unk_token_id,
        }
    if long_decoded != long_sequence or len(long_ids) != 502:
        raise ValueError(f"E007 Phase 4B length-500 tokenizer accounting failed: {candidate}")
    if any(value == [] or value is None for value in canonical_ids.values()):
        raise ValueError(f"E007 Phase 4B canonical tokenizer mapping is incomplete: {candidate}")
    if any(value["status"] == "substituted" for value in ambiguous.values()):
        raise ValueError(f"E007 Phase 4B ambiguous residue substitution detected: {candidate}")
    return {
        "canonical_round_trip": True,
        "canonical_token_count": len(sequence),
        "encoded_token_count": len(ids),
        "canonical_mapping": canonical_ids,
        "length_500_requested": 500 in lengths,
        "length_500_encoded_token_count": len(long_ids),
        "length_500_round_trip": True,
        "ambiguous_residue_round_trips": ambiguous,
        "special_tokens": special_tokens,
        "silent_substitution_allowed": False,
    }


def _load_transformer_candidate(
    candidate: str, cache: Path, device: Any, *, configuration_version: str = VERSION
) -> tuple[Any, Any, dict[str, Any]]:
    if configuration_version == VERSION_V3:
        from protein_distance_diffusion.evaluation.e007_pretrained_loaders_v3 import load_reviewed_candidate
    elif configuration_version == VERSION_V2:
        from protein_distance_diffusion.evaluation.e007_pretrained_loaders_v2 import load_reviewed_candidate
    else:
        from protein_distance_diffusion.evaluation.e007_pretrained_loaders import load_reviewed_candidate

    return load_reviewed_candidate(candidate, cache, device=device)


def _worker_case(config_path: str, candidate: str, result_path: str) -> None:
    started = time.monotonic()
    stage = "startup"
    result: dict[str, Any] = {
        "candidate": candidate,
        "status": "failed",
        "optimizer_updates": 0,
        "smoke_completed": False,
    }
    try:
        import importlib.metadata
        import socket
        import sys

        from protein_distance_diffusion.evaluation.e007_pretrained_loaders import (
            OFFLINE_ENVIRONMENT,
            verify_runtime_environment,
        )

        os.environ.update(OFFLINE_ENVIRONMENT)

        def network_refused(*_args, **_kwargs):
            raise RuntimeError("E007 Phase 4B child-process network access is forbidden")

        socket.socket = network_refused  # type: ignore[assignment]
        socket.create_connection = network_refused  # type: ignore[assignment]
        import torch

        config = _load_config(config_path)
        stage = "environment_and_artifact_verification"
        readiness = _verify_loader_readiness(config)
        environment_verification = verify_runtime_environment(readiness["installable_environment"]["path"])
        verify_artifacts_offline(config_path)
        cache = Path(config["artifact_cache_root"])
        loader_status = readiness["loaders"].get(candidate, {}).get("status")
        if loader_status != "reviewed_ready":
            result.update({"status": "loader_review_required", "loader_status": loader_status})
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device_baseline = _cuda_device_baseline(torch, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(int(config["smoke"]["seed"]))
            stage = "candidate_loading"
            model, tokenizer, loader_metadata = _load_transformer_candidate(
                candidate, cache, device, configuration_version=config["version"]
            )
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            before = _parameter_sha(model)
            model_weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
            if config["version"] == VERSION_V3:
                from protein_distance_diffusion.evaluation.e007_pretrained_loaders_v3 import (
                    validate_tokenizer_contract_v3,
                )

                tokenizer_result = validate_tokenizer_contract_v3(candidate, tokenizer)
            elif candidate != "proteinmpnn_ca_only":
                tokenizer_result = _tokenizer_checks(candidate, tokenizer, config["smoke"]["lengths"])
            else:
                tokenizer_result = {
                    "canonical_mapping": {residue: index for index, residue in enumerate(CANONICAL)},
                    "canonical_round_trip": True,
                    "length_500_requested": True,
                    "length_500_encoded_token_count": 500,
                    "ca_only": True,
                }
            cases = []
            stage = "smoke_execution"
            for length in config["smoke"]["lengths"]:
                sequence = (CANONICAL * ((length + 19) // 20))[:length]
                if candidate == "esm2_150m":
                    tokens = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
                    input_ids = tokens["input_ids"].to(device)
                    attention = tokens["attention_mask"].to(device)
                elif candidate == "progen2_151m":
                    ids = tokenizer.encode(f"1{sequence}2").ids
                    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                    attention = torch.ones_like(input_ids)
                else:
                    input_ids = torch.zeros((1, length), dtype=torch.long, device=device)
                    attention = torch.ones_like(input_ids)
                    coordinates = torch.stack(
                        [
                            torch.arange(length, device=device) * 3.8,
                            torch.zeros(length, device=device),
                            torch.zeros(length, device=device),
                        ],
                        dim=-1,
                    ).unsqueeze(0)
                    residue_indices = torch.arange(length, device=device).unsqueeze(0)
                    chain_labels = torch.zeros_like(input_ids)
                    random_values = torch.linspace(0.1, 1.0, length, device=device).unsqueeze(0)
                with torch.no_grad():
                    if candidate == "proteinmpnn_ca_only":
                        first = model(
                            coordinates,
                            input_ids,
                            attention,
                            attention,
                            residue_indices,
                            chain_labels,
                            random_values,
                        )
                        second = model(
                            coordinates,
                            input_ids,
                            attention,
                            attention,
                            residue_indices,
                            chain_labels,
                            random_values,
                        )
                    else:
                        first = model(input_ids=input_ids, attention_mask=attention).logits
                        second = model(input_ids=input_ids, attention_mask=attention).logits
                if not torch.equal(first, second):
                    raise ValueError(f"E007 Phase 4B deterministic logits replay failed: {candidate}/{length}")
                if candidate == "proteinmpnn_ca_only":
                    adapter = torch.nn.Linear(3, 3, bias=False, device=device, dtype=torch.float32)
                    conditioned = coordinates + adapter(coordinates)
                    embeddings = coordinates
                    logits = model(
                        conditioned,
                        input_ids,
                        attention,
                        attention,
                        residue_indices,
                        chain_labels,
                        random_values,
                    )
                else:
                    hidden = int(model.config.hidden_size if candidate == "esm2_150m" else model.config.n_embd)
                    adapter = torch.nn.Linear(4, hidden, bias=False, device=device, dtype=torch.float32)
                    embeddings = model.get_input_embeddings()(input_ids).detach()
                    geometry = _geometry_features(torch, embeddings.shape[1], device)
                    conditioned = embeddings + adapter(geometry)
                    logits = model(inputs_embeds=conditioned, attention_mask=attention).logits
                loss = logits.float().square().mean()
                loss.backward()
                if adapter.weight.grad is None or not torch.isfinite(adapter.weight.grad).all():
                    raise FloatingPointError("E007 Phase 4B geometry-adapter gradient is absent or non-finite")
                cases.append(
                    {
                        "length": length,
                        "biological_length": length,
                        "input_token_count": int(input_ids.shape[1]),
                        "logit_shape": list(first.shape),
                        "logits_sha256": _tensor_sha(first),
                        "logits_bytes": first.numel() * first.element_size(),
                        "activation_bytes_measured": embeddings.numel() * embeddings.element_size()
                        + conditioned.numel() * conditioned.element_size(),
                        "adapter_gradient_bytes": adapter.weight.grad.numel() * adapter.weight.grad.element_size(),
                        "adapter_gradient_norm": float(adapter.weight.grad.float().norm().item()),
                        "loss": float(loss.item()),
                    }
                )
                del first, second, logits, loss, embeddings, conditioned, adapter
            after = _parameter_sha(model)
            if before != after:
                raise ValueError(f"E007 Phase 4B frozen parameter mutation detected: {candidate}")
            if device.type == "cuda":
                sequence = (CANONICAL * 25)[:500]
                if candidate == "esm2_150m":
                    descriptive_tokens = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
                    descriptive_ids = descriptive_tokens["input_ids"].to(device)
                    descriptive_attention = descriptive_tokens["attention_mask"].to(device)
                elif candidate == "progen2_151m":
                    descriptive_ids = torch.tensor(
                        [tokenizer.encode(f"1{sequence}2").ids], dtype=torch.long, device=device
                    )
                    descriptive_attention = torch.ones_like(descriptive_ids)
                else:
                    descriptive_ids = torch.zeros((1, 500), dtype=torch.long, device=device)
                    descriptive_attention = torch.ones_like(descriptive_ids)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    if candidate == "proteinmpnn_ca_only":
                        descriptive_coordinates = torch.stack(
                            [
                                torch.arange(500, device=device) * 3.8,
                                torch.zeros(500, device=device),
                                torch.zeros(500, device=device),
                            ],
                            dim=-1,
                        ).unsqueeze(0)
                        descriptive_positions = torch.arange(500, device=device).unsqueeze(0)
                        descriptive_chains = torch.zeros_like(descriptive_ids)
                        descriptive_random = torch.linspace(0.1, 1.0, 500, device=device).unsqueeze(0)
                        descriptive_logits = model(
                            descriptive_coordinates,
                            descriptive_ids,
                            descriptive_attention,
                            descriptive_attention,
                            descriptive_positions,
                            descriptive_chains,
                            descriptive_random,
                        )
                    else:
                        descriptive_logits = model(
                            input_ids=descriptive_ids, attention_mask=descriptive_attention
                        ).logits
                descriptive = {
                    "status": "completed_descriptive_only",
                    "dtype": str(descriptive_logits.dtype),
                    "finite": bool(torch.isfinite(descriptive_logits).all()),
                }
                torch.cuda.synchronize()
            else:
                descriptive = {"status": "not_run_cuda_unavailable", "dtype": None, "finite": None}
            if device.type == "cuda":
                torch.cuda.synchronize()
                device_free_after, _device_total_after = torch.cuda.mem_get_info()
            else:
                device_free_after = None
            result.update(
                {
                    "status": "passed",
                    "smoke_completed": True,
                    "evidence_mode": {
                        VERSION: "executed_v1",
                        VERSION_V2: "newly_executed_v2",
                        VERSION_V3: "newly_executed_v3_after_measured_cpu_diagnostic",
                    }[config["version"]],
                    "device": str(device),
                    "tokenizer": tokenizer_result,
                    "cases": cases,
                    "parameter_sha256_before": before,
                    "parameter_sha256_after": after,
                    "model_weight_bytes": model_weight_bytes,
                    "descriptive_lower_precision": descriptive,
                    "current_cuda_allocated_mib": (
                        torch.cuda.memory_allocated() / 2**20 if device.type == "cuda" else None
                    ),
                    "current_cuda_reserved_mib": (
                        torch.cuda.memory_reserved() / 2**20 if device.type == "cuda" else None
                    ),
                    "peak_cuda_allocated_mib": (
                        torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
                    ),
                    "peak_cuda_reserved_mib": (
                        torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else None
                    ),
                    "process_local_cuda_telemetry": True,
                    **device_baseline,
                    "device_free_after_mib": None if device_free_after is None else device_free_after / 2**20,
                    "external_device_usage_attribution": "reported_separately_not_attributed_to_candidate",
                    "downloaded_code_executed": False,
                    "loader_metadata": loader_metadata,
                    "environment_verification": environment_verification,
                    "offline_environment": dict(OFFLINE_ENVIRONMENT),
                }
            )
        result.update(
            {
                "elapsed_seconds": time.monotonic() - started,
                "current_rss_mib": _current_rss_mib(),
                "peak_rss_mib": _peak_rss_mib(),
                "runtime_versions": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "cuda_build": torch.version.cuda,
                    **{
                        package: importlib.metadata.version(package)
                        for package in ("transformers", "tokenizers", "safetensors", "accelerate")
                        if package
                        in {distribution.metadata["Name"] for distribution in importlib.metadata.distributions()}
                    },
                },
                **NON_AUTHORIZING,
            }
        )
    except BaseException as error:
        status = (
            "oom"
            if "outofmemory" in type(error).__name__.lower() or "out of memory" in str(error).lower()
            else "failed"
        )
        result.update(
            {
                "status": status,
                "failure_stage": stage,
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
                "elapsed_seconds": time.monotonic() - started,
                "current_rss_mib": _current_rss_mib(),
                "peak_rss_mib": _peak_rss_mib(),
                **NON_AUTHORIZING,
            }
        )
    _atomic_json(Path(result_path), result)


def _decision(results: Sequence[Mapping[str, Any]], *, version: str = VERSION) -> str:
    statuses = {row["candidate"]: row["status"] for row in results}
    esm = statuses.get("esm2_150m") == "passed"
    progen = statuses.get("progen2_151m") == "passed"
    if version in {VERSION_V2, VERSION_V3}:
        primary = [row for row in results if row["candidate"] in {"esm2_150m", "progen2_151m"}]
        if len(primary) != 2 or any(not row.get("smoke_completed", False) for row in primary):
            return "candidate_loader_review_required"
    if esm and progen:
        return "esm2_and_progen2_advance"
    if esm:
        return "esm2_only_advances_after_completed_smoke" if version == VERSION_V3 else "esm2_only_advances"
    if progen:
        return "progen2_only_advances_after_completed_smoke" if version == VERSION_V3 else "progen2_only_advances"
    if version in {VERSION_V2, VERSION_V3}:
        if all(row.get("smoke_completed", False) for row in primary):
            return "neither_primary_candidate_advances_after_completed_smoke"
        return "inconclusive_requires_scientific_review"
    if any("artifact" in str(status) or "environment" in str(status) for status in statuses.values()):
        return "environment_or_artifact_review_required"
    return "neither_primary_candidate_advances"


def run_smoke(
    config_path: str | Path,
    *,
    resume: bool = False,
    worker_target: Callable[[str, str, str], None] = _worker_case,
    environment_verifier: Callable[[str | Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    verify_artifacts_offline(config_path)
    if environment_verifier is None:
        environment_verifier = verify_environment_readiness
    environment_verification = dict(environment_verifier(config_path))
    output = Path(config["smoke_output"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists():
        raise FileExistsError(f"E007 Phase 4B completed smoke exists: {output}")
    if staging.exists() and not resume:
        raise FileExistsError(f"E007 Phase 4B smoke staging exists: {staging}")
    staging.mkdir(parents=True, exist_ok=resume)
    journal_path = staging / "journal.json"
    journal = json.loads(journal_path.read_text()) if resume and journal_path.is_file() else {"committed": {}}
    _atomic_json(staging / "heartbeat.json", {"status": "running", "updated_utc": _utc_now(), **NON_AUTHORIZING})
    results = []
    context = mp.get_context("spawn")
    for candidate in config["smoke"]["candidates"]:
        path = staging / f"{candidate}.json"
        expected_hash = journal["committed"].get(candidate)
        if resume and path.is_file() and expected_hash:
            if sha256_file(path) != expected_hash:
                raise ValueError(f"E007 Phase 4B resumed candidate result hash contradiction: {candidate}")
            result = json.loads(path.read_text())
        else:
            process = context.Process(target=worker_target, args=(str(config_path), candidate, str(path)))
            process.start()
            process.join()
            if process.exitcode != 0 or not path.is_file():
                raise RuntimeError(f"E007 Phase 4B isolated candidate process failed: {candidate}")
            result = json.loads(path.read_text())
            journal["committed"][candidate] = sha256_file(path)
            _atomic_json(journal_path, journal)
        results.append(result)
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "running",
                "updated_utc": _utc_now(),
                "completed_candidates": [row["candidate"] for row in results],
                **NON_AUTHORIZING,
            },
        )
    for result in results:
        if float(result["peak_rss_mib"]) > float(config["smoke"]["memory"]["maximum_rss_mib"]):
            result["status"] = "memory_limit_exceeded"
        for metric, limit in (
            ("peak_cuda_allocated_mib", "maximum_cuda_allocated_mib"),
            ("peak_cuda_reserved_mib", "maximum_cuda_reserved_mib"),
        ):
            if result.get(metric) is not None and float(result[metric]) > float(config["smoke"]["memory"][limit]):
                result["status"] = "memory_limit_exceeded"
    metrics_text = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in results)
    _atomic_text(staging / "detailed_metrics.jsonl", metrics_text)
    decision = _decision(results, version=config["version"])
    report = {
        "status": "completed_non_authorizing",
        "version": config["version"],
        "decision": decision,
        "candidate_results": results,
        "final_conditioning_architecture_selected": False,
        "optimizer_updates": 0,
        "training_performed": False,
        "sampling_performed": False,
        "environment_verification": environment_verification,
        "v1_primary_decision_interpretation": (
            "loader_implementation_failure_not_scientific_evidence" if config["version"] == VERSION_V2 else None
        ),
        "supersedes_v1_for_primary_candidate_feasibility_only": config["version"] == VERSION_V2,
        "supersedes_v2_for_primary_candidate_feasibility_only": config["version"] == VERSION_V3,
        "reused_evidence": config.get("reused_evidence", {}),
        **NON_AUTHORIZING,
    }
    _atomic_json(staging / "report.json", report)
    protocol = {
        "status": report["status"],
        "version": config["version"],
        "configuration_sha256": sha256_file(config_path),
        "artifact_lock_sha256": sha256_file(Path(config["artifact_lock_output"]) / "artifact_lock.json"),
        "report_sha256": sha256_file(staging / "report.json"),
        "detailed_metrics_sha256": sha256_file(staging / "detailed_metrics.jsonl"),
        "completed_utc": _utc_now(),
        **NON_AUTHORIZING,
    }
    _atomic_json(staging / "protocol.json", protocol)
    names = [
        "report.json",
        "protocol.json",
        "detailed_metrics.jsonl",
        "journal.json",
        *[f"{name}.json" for name in config["smoke"]["candidates"]],
    ]
    inventory_rows = [
        {"path": name, "size_bytes": (staging / name).stat().st_size, "sha256": sha256_file(staging / name)}
        for name in names
    ]
    _atomic_json(
        staging / "artifact_inventory.json",
        {"artifacts": inventory_rows, "aggregate_sha256": _canonical_sha(inventory_rows)},
    )
    _atomic_json(
        staging / "heartbeat.json",
        {
            "status": "completed",
            "completed_utc": _utc_now(),
            "decision": decision,
            "report_sha256": protocol["report_sha256"],
            **NON_AUTHORIZING,
        },
    )
    staging.replace(output)
    return {"status": report["status"], "decision": decision, "output_dir": str(output), **NON_AUTHORIZING}


def resume(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    lock = Path(config["artifact_lock_output"])
    if not lock.exists():
        return acquire_artifacts(config_path, resume=True)
    return run_smoke(config_path, resume=True)
