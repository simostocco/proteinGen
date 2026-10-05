"""Reviewed, local-only loaders for E007 Phase 4B pretrained candidates."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import socket
import struct
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
OFFLINE_ENVIRONMENT = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
LOADER_VERSION = "e007_phase4b_reviewed_local_loaders_v1"

EXPECTED = {
    "esm2_150m": {
        "loader": "transformers_native_esm_local_v1",
        "state_tensor_count": 522,
        "parameter_count": 148_796_794,
        "layers": 30,
        "width": 640,
        "heads": 20,
        "vocabulary_size": 33,
        "position_capacity": 1026,
        "special_tokens": {"cls": 0, "pad": 1, "eos": 2, "unk": 3, "mask": 32},
        "output_embedding_tied": True,
    },
    "progen2_151m": {
        "loader": "transformers_native_progen_local_v1",
        "state_tensor_count": 125,
        "parameter_count": 163_731_500,
        "layers": 12,
        "width": 1024,
        "heads": 16,
        "vocabulary_size": 32,
        "tokenizer_vocabulary_size": 30,
        "position_capacity": 1024,
        "special_tokens": {"pad": 0, "bos": 1, "eos": 2},
        "output_embedding_tied": False,
    },
    "proteinmpnn_ca_only": {
        "loader": "reviewed_project_owned_proteinmpnn_ca_v1",
        "state_tensor_count": 123,
        "parameter_count": 1_645_765,
        "encoder_layers": 3,
        "decoder_layers": 3,
        "width": 128,
        "vocabulary_size": 21,
        "neighbors": 48,
        "noise_level": 0.2,
        "coordinate_input": "ca_only_B_L_3",
        "full_backbone_allowed": False,
    },
}


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def environment_contract(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if payload.get("name") != "proteingen-e007-phase4b":
        raise ValueError("E007 Phase 4B isolated environment name contradiction")
    if payload.get("channels") != ["conda-forge", "nodefaults"]:
        raise ValueError("E007 Phase 4B isolated environment channels changed")
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list) or not any(isinstance(item, dict) and "pip" in item for item in dependencies):
        raise ValueError("E007 Phase 4B environment is not an installable Conda specification")
    return payload


def declared_environment_versions(path: str | Path) -> dict[str, str]:
    payload = environment_contract(path)
    rows: dict[str, str] = {}
    for item in payload["dependencies"]:
        if isinstance(item, str) and "=" in item:
            name, value = item.split("=", 1)
            rows[name.lower()] = value
        elif isinstance(item, dict):
            for requirement in item.get("pip", []):
                if requirement.startswith("--"):
                    continue
                name, value = requirement.split("==", 1)
                rows[name.lower()] = value
    return rows


def declared_environment_fingerprint(path: str | Path) -> str:
    payload = environment_contract(path)
    return canonical_sha({"name": payload["name"], "versions": declared_environment_versions(path)})


def verify_runtime_environment(path: str | Path, *, require_name: bool = True) -> dict[str, Any]:
    payload = environment_contract(path)
    expected = declared_environment_versions(path)
    if require_name and os.environ.get("CONDA_DEFAULT_ENV") != payload["name"]:
        raise ValueError("E007 Phase 4B smoke must run in proteingen-e007-phase4b")
    observed = {"python": ".".join(map(str, sys.version_info[:3]))}
    aliases = {"pyyaml": "PyYAML", "huggingface-hub": "huggingface-hub", "mdanalysis": "MDAnalysis"}
    modules = {
        "pyyaml": "yaml",
        "huggingface-hub": "huggingface_hub",
        "mdanalysis": "MDAnalysis",
    }
    for name in expected:
        if name in {"python", "pip"}:
            continue
        distribution = aliases.get(name, name)
        try:
            observed[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"E007 Phase 4B environment dependency is absent: {name}") from error
        if name != "ruff":  # Ruff publishes a CLI binary rather than an importable Python package.
            importlib.import_module(modules.get(name, name.replace("-", "_")))
    comparable = dict(observed)
    if "torch" in comparable:
        comparable["torch"] = comparable["torch"].split("+", 1)[0]
    mismatches = {
        name: {"expected": value, "observed": observed.get(name)}
        for name, value in expected.items()
        if name != "pip" and comparable.get(name) != value
    }
    if mismatches:
        raise ValueError(f"E007 Phase 4B environment version contradiction: {mismatches}")
    import torch

    if torch.version.cuda != "13.0":
        raise ValueError(f"E007 Phase 4B CUDA build contradiction: {torch.version.cuda}")
    if torch.cuda.is_initialized():
        raise ValueError("E007 Phase 4B environment verification unexpectedly initialized CUDA")
    observed["cuda_build"] = str(torch.version.cuda)
    runtime = {"name": payload["name"], "versions": observed}
    return {
        "status": "verified",
        "declared_fingerprint": declared_environment_fingerprint(path),
        "runtime_fingerprint": canonical_sha(runtime),
        "runtime": runtime,
        "cuda_initialized": False,
    }


@contextlib.contextmanager
def offline_network_guard() -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in OFFLINE_ENVIRONMENT}
    old_socket = socket.socket
    old_connection = socket.create_connection

    def refused(*_args, **_kwargs):
        raise RuntimeError("E007 Phase 4B network access is forbidden")

    try:
        os.environ.update(OFFLINE_ENVIRONMENT)
        socket.socket = refused  # type: ignore[assignment]
        socket.create_connection = refused  # type: ignore[assignment]
        yield
    finally:
        socket.socket = old_socket  # type: ignore[assignment]
        socket.create_connection = old_connection
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def require_local_path(path: str | Path, *, root: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = Path(root) / value
    resolved = value.resolve()
    if not resolved.is_relative_to(Path(root).resolve()) or not resolved.is_file():
        raise ValueError(f"E007 Phase 4B local artifact is absent or outside the cache: {path}")
    return resolved


def safetensors_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError("E007 Phase 4B invalid safetensors header")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > 16 * 1024**2:
            raise ValueError("E007 Phase 4B safetensors header exceeds limit")
        header = json.loads(handle.read(header_length))
    return {key: value for key, value in header.items() if key != "__metadata__"}


def validate_tensor_mapping(
    state: Mapping[str, Any],
    *,
    expected_keys: Sequence[str] | None = None,
    expected_shapes: Mapping[str, Sequence[int]] | None = None,
    expected_tensor_count: int | None = None,
    expected_parameter_count: int | None = None,
) -> dict[str, int]:
    import torch

    if not state or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError("E007 Phase 4B state dictionary contains unaudited values")
    keys = set(state)
    if expected_keys is not None and keys != set(expected_keys):
        raise ValueError(
            f"E007 Phase 4B state-dict key contradiction: missing={sorted(set(expected_keys) - keys)[:20]}, "
            f"unexpected={sorted(keys - set(expected_keys))[:20]}"
        )
    for key, shape in (expected_shapes or {}).items():
        if key not in state or tuple(state[key].shape) != tuple(shape):
            raise ValueError(f"E007 Phase 4B state-dict shape contradiction: {key}")
    tensor_count = len(state)
    parameter_count = sum(value.numel() for value in state.values())
    if expected_tensor_count is not None and tensor_count != expected_tensor_count:
        raise ValueError("E007 Phase 4B state tensor-count contradiction")
    if expected_parameter_count is not None and parameter_count != expected_parameter_count:
        raise ValueError("E007 Phase 4B state parameter-count contradiction")
    return {"state_tensor_count": tensor_count, "state_element_count": parameter_count}


def safe_torch_checkpoint_metadata(
    path: str | Path, *, nested_key: str | None = None, map_location: str = "meta"
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    import torch

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True, mmap=True)
    except Exception as error:
        raise ValueError("E007 Phase 4B safe weights-only checkpoint loading failed") from error
    if not isinstance(payload, Mapping):
        raise ValueError("E007 Phase 4B checkpoint payload is not a mapping")
    if nested_key is None:
        state = payload
        metadata: dict[str, Any] = {}
    else:
        state = payload.get(nested_key)
        metadata = {key: value for key, value in payload.items() if key != nested_key}
    if not isinstance(state, Mapping):
        raise ValueError("E007 Phase 4B checkpoint state dictionary is absent")
    validate_tensor_mapping(state)
    allowed_metadata = (str, int, float, bool, type(None))
    if any(not isinstance(value, allowed_metadata) for value in metadata.values()):
        raise ValueError("E007 Phase 4B checkpoint metadata contains unaudited globals")
    return state, metadata


def validate_tokenizer_contract(candidate: str, tokenizer: Any) -> dict[str, Any]:
    expected = EXPECTED[candidate]
    sequence = CANONICAL * 25
    if candidate == "esm2_150m":
        mapping = {residue: tokenizer.convert_tokens_to_ids(residue) for residue in CANONICAL}
        encoded = tokenizer(sequence, add_special_tokens=True)["input_ids"]
        decoded = tokenizer.decode(encoded, skip_special_tokens=True).replace(" ", "")
        specials = {
            "cls": tokenizer.cls_token_id,
            "pad": tokenizer.pad_token_id,
            "eos": tokenizer.eos_token_id,
            "unk": tokenizer.unk_token_id,
            "mask": tokenizer.mask_token_id,
        }
        vocabulary_size = len(tokenizer)
    elif candidate == "progen2_151m":
        mapping = {residue: tokenizer.token_to_id(residue) for residue in CANONICAL}
        encoded = tokenizer.encode(f"1{sequence}2").ids
        decoded = tokenizer.decode(encoded).replace(" ", "").removeprefix("1").removesuffix("2")
        specials = {
            "pad": tokenizer.token_to_id("<|pad|>"),
            "bos": tokenizer.token_to_id("<|bos|>"),
            "eos": tokenizer.token_to_id("<|eos|>"),
        }
        vocabulary_size = tokenizer.get_vocab_size()
    else:
        raise ValueError(f"E007 Phase 4B tokenizer is unsupported: {candidate}")
    if len(set(mapping.values())) != 20 or any(value is None for value in mapping.values()):
        raise ValueError("E007 Phase 4B canonical residue tokenizer mapping contradiction")
    if decoded != sequence or len(encoded) != 502:
        raise ValueError("E007 Phase 4B tokenizer N=500 accounting contradiction")
    expected_vocabulary = expected.get("tokenizer_vocabulary_size", expected["vocabulary_size"])
    if vocabulary_size != expected_vocabulary or specials != expected["special_tokens"]:
        raise ValueError("E007 Phase 4B tokenizer vocabulary or special-token contradiction")
    return {"canonical_mapping": mapping, "encoded_length_500": len(encoded), "vocabulary_size": vocabulary_size}


def load_reviewed_candidate(candidate: str, cache_root: str | Path, *, device: Any) -> tuple[Any, Any, dict[str, Any]]:
    """Construct a reviewed model from locked local files. Called only by a smoke child."""

    cache = Path(cache_root).resolve()
    if candidate == "esm2_150m":
        from safetensors.torch import load_file
        from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer

        directory = cache / "esm2_150m"
        config = EsmConfig.from_json_file(str(require_local_path(directory / "config.json", root=cache)))
        expected = EXPECTED[candidate]
        if (
            config.hidden_size != expected["width"]
            or config.num_hidden_layers != expected["layers"]
            or config.num_attention_heads != expected["heads"]
            or config.vocab_size != expected["vocabulary_size"]
            or config.max_position_embeddings != expected["position_capacity"]
        ):
            raise ValueError("E007 Phase 4B ESM architecture contract contradiction")
        tokenizer = EsmTokenizer.from_pretrained(directory, local_files_only=True)
        model = EsmForMaskedLM(config)
        weight_path = require_local_path(directory / "model.safetensors", root=cache)
        schema = safetensors_metadata(weight_path)
        if len(schema) != expected["state_tensor_count"]:
            raise ValueError("E007 Phase 4B ESM state tensor-count contradiction")
        for key, shape in {
            "esm.embeddings.word_embeddings.weight": [33, 640],
            "esm.embeddings.position_embeddings.weight": [1026, 640],
            "lm_head.bias": [33],
        }.items():
            if key not in schema or schema[key]["shape"] != shape:
                raise ValueError(f"E007 Phase 4B ESM state shape contradiction: {key}")
        state = load_file(str(weight_path), device="cpu")
        result = model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError("E007 Phase 4B ESM strict state loading contradiction")
        if sum(parameter.numel() for parameter in model.parameters()) != expected["parameter_count"]:
            raise ValueError("E007 Phase 4B ESM parameter-count contradiction")
        if model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
            raise ValueError("E007 Phase 4B ESM tied-weight contradiction")
    elif candidate == "progen2_151m":
        from tokenizers import Tokenizer
        from transformers import ProGenConfig, ProGenForCausalLM

        directory = cache / "progen2_151m" / "checkpoint"
        config = ProGenConfig.from_json_file(str(require_local_path(directory / "config.json", root=cache)))
        expected = EXPECTED[candidate]
        if (
            config.n_embd != expected["width"]
            or config.n_layer != expected["layers"]
            or config.n_head != expected["heads"]
            or config.vocab_size != expected["vocabulary_size"]
            or config.n_positions != expected["position_capacity"]
        ):
            raise ValueError("E007 Phase 4B ProGen architecture contract contradiction")
        tokenizer = Tokenizer.from_file(
            str(require_local_path(cache / "progen2_151m" / "source" / "tokenizer.json", root=cache))
        )
        metadata_state, _metadata = safe_torch_checkpoint_metadata(
            require_local_path(directory / "pytorch_model.bin", root=cache)
        )
        validate_tensor_mapping(
            metadata_state,
            expected_tensor_count=expected["state_tensor_count"],
            expected_parameter_count=expected["parameter_count"],
        )
        model = ProGenForCausalLM(config)
        state, _metadata = safe_torch_checkpoint_metadata(
            require_local_path(directory / "pytorch_model.bin", root=cache), map_location="cpu"
        )
        result = model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError("E007 Phase 4B ProGen strict state loading contradiction")
        if sum(parameter.numel() for parameter in model.parameters()) != expected["parameter_count"]:
            raise ValueError("E007 Phase 4B ProGen parameter-count contradiction")
        if model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr():
            raise ValueError("E007 Phase 4B ProGen unexpectedly tied output weights")
    elif candidate == "proteinmpnn_ca_only":
        from protein_distance_diffusion.models.e007_proteinmpnn_ca import ProteinMPNNCA

        expected = EXPECTED[candidate]
        checkpoint = require_local_path(cache / "proteinmpnn_ca_only" / "ca_model_weights" / "v_48_020.pt", root=cache)
        metadata_state, metadata = safe_torch_checkpoint_metadata(checkpoint, nested_key="model_state_dict")
        if metadata != {"num_edges": 48, "noise_level": 0.2}:
            raise ValueError("E007 Phase 4B ProteinMPNN CA checkpoint metadata contradiction")
        validate_tensor_mapping(
            metadata_state,
            expected_tensor_count=expected["state_tensor_count"],
            expected_parameter_count=expected["parameter_count"],
        )
        model = ProteinMPNNCA(k_neighbors=48)
        state, metadata = safe_torch_checkpoint_metadata(checkpoint, nested_key="model_state_dict", map_location="cpu")
        result = model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError("E007 Phase 4B ProteinMPNN CA strict state loading contradiction")
        if sum(parameter.numel() for parameter in model.parameters()) != expected["parameter_count"]:
            raise ValueError("E007 Phase 4B ProteinMPNN CA parameter-count contradiction")
        tokenizer = None
    else:
        raise ValueError(f"E007 Phase 4B unknown candidate: {candidate}")
    tokenizer_contract = validate_tokenizer_contract(candidate, tokenizer) if tokenizer is not None else None
    model = model.to(device)
    metadata = {
        "loader_version": LOADER_VERSION,
        "candidate": candidate,
        "expectation": EXPECTED[candidate],
        "offline": dict(OFFLINE_ENVIRONMENT),
        "trust_remote_code": False,
        "downloaded_source_executed": False,
        "tokenizer_contract": tokenizer_contract,
    }
    return model, tokenizer, metadata
