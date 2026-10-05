"""Version-specific local loaders for E007 Phase 4B.1 smoke v2."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from protein_distance_diffusion.evaluation.e007_pretrained_loaders import (
    CANONICAL,
    OFFLINE_ENVIRONMENT,
    require_local_path,
    safe_torch_checkpoint_metadata,
    validate_tensor_mapping,
    validate_tokenizer_contract,
)

LOADER_VERSION = "e007_phase4b_reviewed_local_loaders_v2"
ESM_NORMALIZATION_VERSION = "esm2_rotary_tied_safetensors_normalization_v1"
PROGEN_IMPLEMENTATION_VERSION = "e007_reviewed_progen2_causal_lm_v1"

EXPECTED = {
    "esm2_150m": {
        "loader": "transformers_native_esm_rotary_normalized_v2",
        "raw_state_tensor_count": 522,
        "parameter_count": 148_796_794,
        "layers": 30,
        "width": 640,
        "heads": 20,
        "vocabulary_size": 33,
        "position_capacity": 1026,
        "position_embedding_type": "rotary",
        "special_tokens": {"cls": 0, "pad": 1, "eos": 2, "unk": 3, "mask": 32},
        "output_embedding_tied": True,
    },
    "progen2_151m": {
        "loader": "reviewed_project_owned_progen2_v1",
        "state_tensor_count": 125,
        "state_element_count": 163_731_500,
        "parameter_count": 151_148_576,
        "layers": 12,
        "width": 1024,
        "heads": 16,
        "vocabulary_size": 32,
        "tokenizer_vocabulary_size": 30,
        "position_capacity": 1024,
        "rotary_dim": 32,
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

_ESM_TIED_SOURCE = "esm.embeddings.word_embeddings.weight"
_ESM_TIED_TARGET = "lm_head.decoder.weight"
_ESM_ROTARY_ARTIFACTS = {
    "esm.embeddings.position_ids": "legacy_persistent_buffer_unused_by_rotary_forward",
    "esm.embeddings.position_embeddings.weight": "legacy_learned_absolute_embedding_unused_by_rotary_forward",
}


def _tensor_sha256(value: Any) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tensor_inventory(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "sha256": _tensor_sha256(value),
        }
        for key, value in sorted(state.items())
    ]


def normalize_esm_rotary_state_dict(
    state: Mapping[str, Any],
    *,
    model_state_keys: Sequence[str],
    position_embedding_type: str,
    tie_word_embeddings: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the only accepted semantic normalization for the pinned ESM snapshot."""

    raw_keys = set(state)
    expected_keys = set(model_state_keys)
    missing = expected_keys - raw_keys
    unexpected = raw_keys - expected_keys
    if position_embedding_type != "rotary":
        raise ValueError("E007 ESM positional conversion artifacts require rotary configuration")
    if tie_word_embeddings is not True:
        raise ValueError("E007 ESM omitted decoder weight requires tied embeddings")
    if missing != {_ESM_TIED_TARGET} or unexpected != set(_ESM_ROTARY_ARTIFACTS):
        raise ValueError(
            "E007 ESM raw state mismatch exceeds approved normalization: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if _ESM_TIED_SOURCE not in state:
        raise ValueError("E007 ESM tied embedding source is absent")
    normalized = {key: value for key, value in state.items() if key not in _ESM_ROTARY_ARTIFACTS}
    normalized[_ESM_TIED_TARGET] = normalized[_ESM_TIED_SOURCE]
    normalized_missing = expected_keys - set(normalized)
    normalized_unexpected = set(normalized) - expected_keys
    if normalized_missing or normalized_unexpected:
        raise ValueError("E007 ESM normalized semantic state is not strict")
    if normalized[_ESM_TIED_TARGET].data_ptr() != normalized[_ESM_TIED_SOURCE].data_ptr():
        raise ValueError("E007 ESM normalized tied tensors do not share storage")
    return normalized, {
        "version": ESM_NORMALIZATION_VERSION,
        "raw_missing_keys": sorted(missing),
        "raw_unexpected_keys": sorted(unexpected),
        "removed_exact_keys": dict(_ESM_ROTARY_ARTIFACTS),
        "synthesized_exact_keys": {_ESM_TIED_TARGET: _ESM_TIED_SOURCE},
        "normalized_missing_keys": [],
        "normalized_unexpected_keys": [],
    }


def verify_esm_tied_output(model: Any) -> None:
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if input_weight.data_ptr() != output_weight.data_ptr():
        raise ValueError("E007 ESM decoder and input embedding do not share storage")
    if model.lm_head.decoder.weight.data_ptr() != input_weight.data_ptr():
        raise ValueError("E007 ESM logits do not use the tied embedding weight")


def inspect_transformers_progen_support() -> dict[str, Any]:
    try:
        configuration = importlib.util.find_spec("transformers.models.progen.configuration_progen")
        modeling = importlib.util.find_spec("transformers.models.progen.modeling_progen")
    except ModuleNotFoundError:
        configuration = None
        modeling = None
    return {
        "configuration_module": None if configuration is None else configuration.origin,
        "modeling_module": None if modeling is None else modeling.origin,
        "supported": configuration is not None and modeling is not None,
        "top_level_import_used": False,
    }


def _load_esm(cache: Path) -> tuple[Any, Any, dict[str, Any]]:
    from safetensors.torch import load_file
    from transformers.models.esm.configuration_esm import EsmConfig
    from transformers.models.esm.modeling_esm import EsmForMaskedLM
    from transformers.models.esm.tokenization_esm import EsmTokenizer

    directory = cache / "esm2_150m"
    config_path = require_local_path(directory / "config.json", root=cache)
    raw_config = json.loads(config_path.read_text())
    config = EsmConfig.from_json_file(str(config_path))
    expected = EXPECTED["esm2_150m"]
    if (
        config.hidden_size != expected["width"]
        or config.num_hidden_layers != expected["layers"]
        or config.num_attention_heads != expected["heads"]
        or config.vocab_size != expected["vocabulary_size"]
        or config.max_position_embeddings != expected["position_capacity"]
        or config.position_embedding_type != expected["position_embedding_type"]
        or config.tie_word_embeddings is not True
        or raw_config.get("model_type") != "esm"
        or raw_config.get("architectures") != ["EsmForMaskedLM"]
    ):
        raise ValueError("E007 Phase 4B.1 ESM architecture contract contradiction")
    tokenizer = EsmTokenizer.from_pretrained(directory, local_files_only=True)
    model = EsmForMaskedLM(config)
    state = load_file(str(require_local_path(directory / "model.safetensors", root=cache)), device="cpu")
    if len(state) != expected["raw_state_tensor_count"]:
        raise ValueError("E007 Phase 4B.1 ESM raw tensor-count contradiction")
    raw_inventory = tensor_inventory(state)
    normalized, normalization = normalize_esm_rotary_state_dict(
        state,
        model_state_keys=list(model.state_dict()),
        position_embedding_type=config.position_embedding_type,
        tie_word_embeddings=config.tie_word_embeddings,
    )
    normalized_inventory = tensor_inventory(normalized)
    result = model.load_state_dict(normalized, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("E007 Phase 4B.1 ESM normalized strict load contradiction")
    verify_esm_tied_output(model)
    if sum(parameter.numel() for parameter in model.parameters()) != expected["parameter_count"]:
        raise ValueError("E007 Phase 4B.1 ESM parameter-count contradiction")
    return (
        model,
        tokenizer,
        {
            "raw_config": {
                "position_embedding_type": config.position_embedding_type,
                "tie_word_embeddings_effective": config.tie_word_embeddings,
                "tie_word_embeddings_explicit_in_json": "tie_word_embeddings" in raw_config,
                "vocabulary_size": config.vocab_size,
                "position_capacity": config.max_position_embeddings,
                "model_type": raw_config["model_type"],
                "architectures": raw_config["architectures"],
            },
            "normalization": normalization,
            "raw_state_inventory": raw_inventory,
            "normalized_state_inventory": normalized_inventory,
        },
    )


def _load_progen(cache: Path) -> tuple[Any, Any, dict[str, Any]]:
    from tokenizers import Tokenizer

    from protein_distance_diffusion.models.e007_progen2 import ProGen2Config, ProGen2ForCausalLM

    directory = cache / "progen2_151m" / "checkpoint"
    config_path = require_local_path(directory / "config.json", root=cache)
    raw_config = json.loads(config_path.read_text())
    config = ProGen2Config.from_dict(raw_config)
    expected = EXPECTED["progen2_151m"]
    if (
        config.n_embd != expected["width"]
        or config.n_layer != expected["layers"]
        or config.n_head != expected["heads"]
        or config.vocab_size != expected["vocabulary_size"]
        or config.n_positions != expected["position_capacity"]
        or config.rotary_dim != expected["rotary_dim"]
        or raw_config.get("model_type") != "progen"
        or raw_config.get("architectures") != ["ProGenForCausalLM"]
    ):
        raise ValueError("E007 Phase 4B.1 ProGen2 architecture contract contradiction")
    support = inspect_transformers_progen_support()
    if support["supported"]:
        raise ValueError("E007 Phase 4B.1 expected reviewed project-owned ProGen2 path")
    tokenizer = Tokenizer.from_file(
        str(require_local_path(cache / "progen2_151m" / "source" / "tokenizer.json", root=cache))
    )
    state, metadata = safe_torch_checkpoint_metadata(
        require_local_path(directory / "pytorch_model.bin", root=cache), map_location="cpu"
    )
    if metadata:
        raise ValueError("E007 Phase 4B.1 ProGen2 checkpoint metadata is unexpected")
    state_result = validate_tensor_mapping(
        state,
        expected_tensor_count=expected["state_tensor_count"],
        expected_parameter_count=expected["state_element_count"],
    )
    model = ProGen2ForCausalLM(config)
    if set(model.state_dict()) != set(state):
        raise ValueError("E007 Phase 4B.1 ProGen2 semantic key contradiction")
    for key, value in model.state_dict().items():
        if value.shape != state[key].shape or value.dtype != state[key].dtype:
            raise ValueError(f"E007 Phase 4B.1 ProGen2 state shape/dtype contradiction: {key}")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("E007 Phase 4B.1 ProGen2 strict state load contradiction")
    if sum(parameter.numel() for parameter in model.parameters()) != expected["parameter_count"]:
        raise ValueError("E007 Phase 4B.1 ProGen2 parameter-count contradiction")
    if model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr():
        raise ValueError("E007 Phase 4B.1 ProGen2 unexpectedly tied output weights")
    return (
        model,
        tokenizer,
        {
            "transformers_4_56_2_support": support,
            "project_owned_implementation": PROGEN_IMPLEMENTATION_VERSION,
            "state_validation": state_result,
            "state_inventory": tensor_inventory(state),
        },
    )


def load_reviewed_candidate(candidate: str, cache_root: str | Path, *, device: Any) -> tuple[Any, Any, dict[str, Any]]:
    cache = Path(cache_root).resolve()
    if candidate == "esm2_150m":
        model, tokenizer, diagnostics = _load_esm(cache)
    elif candidate == "progen2_151m":
        model, tokenizer, diagnostics = _load_progen(cache)
    elif candidate == "proteinmpnn_ca_only":
        from protein_distance_diffusion.evaluation.e007_pretrained_loaders import load_reviewed_candidate as load_v1

        model, tokenizer, prior = load_v1(candidate, cache, device=device)
        prior["loader_version"] = LOADER_VERSION
        prior["evidence_mode"] = "reexecuted_identically_from_v1_reviewed_loader"
        return model, tokenizer, prior
    else:
        raise ValueError(f"E007 Phase 4B.1 unknown candidate: {candidate}")
    tokenizer_contract = validate_tokenizer_contract(candidate, tokenizer)
    model = model.to(device)
    return (
        model,
        tokenizer,
        {
            "loader_version": LOADER_VERSION,
            "candidate": candidate,
            "expectation": EXPECTED[candidate],
            "offline": dict(OFFLINE_ENVIRONMENT),
            "trust_remote_code": False,
            "downloaded_source_executed": False,
            "tokenizer_contract": tokenizer_contract,
            "diagnostics": diagnostics,
            "canonical_residues": CANONICAL,
        },
    )
