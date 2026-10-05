"""Measured E007 Phase 4B.2 loader contracts for CPU diagnostics and smoke v3."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from protein_distance_diffusion.evaluation.e007_pretrained_loaders import (
    CANONICAL,
    OFFLINE_ENVIRONMENT,
    require_local_path,
    validate_tokenizer_contract,
)
from protein_distance_diffusion.evaluation.e007_pretrained_loaders_v2 import (
    ESM_NORMALIZATION_VERSION,
    _load_progen,
    normalize_esm_rotary_state_dict,
    tensor_inventory,
    verify_esm_tied_output,
)
from protein_distance_diffusion.evaluation.e007_pretrained_loaders_v2 import (
    EXPECTED as V2_EXPECTED,
)

LOADER_VERSION = "e007_phase4b_measured_local_loaders_v3"
PARAMETER_DEFINITION = "unique_parameter_object_elements_remove_duplicate_true"
NAMED_PARAMETER_DEFINITION = "named_parameter_elements_remove_duplicate_false"
PROGEN_FRAMING_VERSION = "progen2_literal_1_prefix_2_suffix_v1"

EXPECTED = {name: dict(value) for name, value in V2_EXPECTED.items()}
EXPECTED["esm2_150m"].update(
    {
        "loader": "transformers_native_esm_measured_accounting_v3",
        "parameter_count": 148_140_154,
        "parameter_count_definition": PARAMETER_DEFINITION,
        "named_parameter_count_including_tied_alias": 148_161_274,
    }
)
EXPECTED["progen2_151m"].update(
    {
        "loader": "reviewed_project_owned_progen2_framed_v2",
        "maximum_biological_length": 1022,
        "framing_token_count": 2,
        "framing_version": PROGEN_FRAMING_VERSION,
    }
)


def parameter_identity_hash(model: Any) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _entries(values: Sequence[tuple[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": name, "shape": list(value.shape), "elements": value.numel()} for name, value in values]


def esm_parameter_accounting(
    model: Any,
    raw_state: Mapping[str, Any],
    normalized_state: Mapping[str, Any],
    *,
    ignored_names: Sequence[str],
    expected_unique_parameter_elements: int | None = None,
) -> dict[str, Any]:
    raw_buffers = [
        (name, value)
        for name, value in raw_state.items()
        if name.endswith(".inv_freq") or name == "esm.embeddings.position_ids"
    ]
    ignored = [(name, raw_state[name]) for name in ignored_names]
    retained_serialized_buffers = [(name, value) for name, value in raw_buffers if name not in ignored_names]
    named_parameters = list(model.named_parameters(remove_duplicate=False))
    unique_parameters = list(model.named_parameters(remove_duplicate=True))
    named_buffers = list(model.named_buffers(remove_duplicate=False))
    unique_parameter_elements = sum(value.numel() for _name, value in unique_parameters)
    named_parameter_elements = sum(value.numel() for _name, value in named_parameters)
    trainable_elements = sum(value.numel() for _name, value in unique_parameters if value.requires_grad)
    nontrainable_elements = unique_parameter_elements - trainable_elements
    ignored_elements = sum(value.numel() for _name, value in ignored)
    retained_buffer_elements = sum(value.numel() for _name, value in retained_serialized_buffers)
    tied_alias_elements = raw_state["esm.embeddings.word_embeddings.weight"].numel()
    raw_elements = sum(value.numel() for value in raw_state.values())
    normalized_elements = sum(value.numel() for value in normalized_state.values())

    categories = {
        "token_embedding_unique_elements": model.esm.embeddings.word_embeddings.weight.numel(),
        "lm_head_unique_excluding_tied_decoder_elements": sum(
            value.numel() for name, value in model.lm_head.named_parameters() if name != "decoder.weight"
        ),
        "lm_head_named_including_tied_decoder_elements": sum(
            value.numel() for _name, value in model.lm_head.named_parameters(remove_duplicate=False)
        ),
        "contact_head_elements": sum(value.numel() for value in model.esm.contact_head.parameters()),
    }
    categories["other_unique_parameter_elements"] = unique_parameter_elements - (
        categories["token_embedding_unique_elements"]
        + categories["lm_head_unique_excluding_tied_decoder_elements"]
        + categories["contact_head_elements"]
    )
    equations = {
        "raw_checkpoint": {
            "left": raw_elements,
            "right": unique_parameter_elements + ignored_elements + retained_buffer_elements,
            "identity": "raw = unique_parameters + ignored_compatibility + retained_serialized_buffers",
        },
        "normalized_state": {
            "left": normalized_elements,
            "right": unique_parameter_elements + tied_alias_elements + retained_buffer_elements,
            "identity": "normalized = unique_parameters + tied_alias + retained_serialized_buffers",
        },
        "named_parameters": {
            "left": named_parameter_elements,
            "right": unique_parameter_elements + tied_alias_elements,
            "identity": "named_parameters_with_alias = unique_parameters + tied_alias",
        },
        "categories": {
            "left": unique_parameter_elements,
            "right": sum(
                categories[name]
                for name in (
                    "token_embedding_unique_elements",
                    "lm_head_unique_excluding_tied_decoder_elements",
                    "contact_head_elements",
                    "other_unique_parameter_elements",
                )
            ),
            "identity": "unique_parameters = embedding + lm_head_excluding_alias + contact_head + other",
        },
    }
    if any(row["left"] != row["right"] for row in equations.values()):
        raise ValueError(f"E007 ESM parameter accounting does not conserve elements: {equations}")
    if (
        expected_unique_parameter_elements is not None
        and unique_parameter_elements != expected_unique_parameter_elements
    ):
        raise ValueError("E007 ESM measured unique parameter count contradiction")
    return {
        "raw_checkpoint_tensor_count": len(raw_state),
        "raw_checkpoint_total_tensor_elements": raw_elements,
        "raw_checkpoint_parameter_like_elements_excluding_buffers": raw_elements
        - sum(value.numel() for _name, value in raw_buffers),
        "raw_serialized_buffers": _entries(raw_buffers),
        "ignored_compatibility_tensors": _entries(ignored),
        "normalized_state_tensor_count": len(normalized_state),
        "normalized_state_elements": normalized_elements,
        "instantiated_named_parameter_count_including_alias": len(named_parameters),
        "instantiated_named_parameter_elements_including_alias": named_parameter_elements,
        "unique_parameter_object_count": len(unique_parameters),
        "unique_parameter_elements": unique_parameter_elements,
        "parameter_count_definition": PARAMETER_DEFINITION,
        "trainable_parameter_elements": trainable_elements,
        "nontrainable_parameter_elements": nontrainable_elements,
        "named_buffer_count": len(named_buffers),
        "named_buffer_elements": sum(value.numel() for _name, value in named_buffers),
        "named_buffers": _entries(named_buffers),
        "retained_serialized_buffer_elements": retained_buffer_elements,
        "tied_aliases": [
            {
                "alias": "lm_head.decoder.weight",
                "source": "esm.embeddings.word_embeddings.weight",
                "elements": tied_alias_elements,
                "shared_storage": model.lm_head.decoder.weight.data_ptr()
                == model.esm.embeddings.word_embeddings.weight.data_ptr(),
            }
        ],
        "categories": categories,
        "conservation_equations": equations,
    }


def progen_framed_sequence_contract(tokenizer: Any, sequence: str, *, position_capacity: int) -> dict[str, Any]:
    if not sequence or any(residue not in CANONICAL for residue in sequence):
        raise ValueError("E007 ProGen biological sequence contains noncanonical or ambiguous residues")
    prefix_id = tokenizer.token_to_id("1")
    suffix_id = tokenizer.token_to_id("2")
    if prefix_id is None or suffix_id is None:
        raise ValueError("E007 ProGen literal framing controls are absent")
    bare = tokenizer.encode(sequence)
    prefixed = tokenizer.encode(f"1{sequence}")
    framed = tokenizer.encode(f"1{sequence}2")
    if len(framed.ids) > position_capacity:
        raise ValueError("E007 ProGen framed sequence exceeds positional capacity")
    if framed.ids[0] != prefix_id or framed.ids[-1] != suffix_id:
        raise ValueError("E007 ProGen framing controls are misplaced")
    biological_ids = framed.ids[1:-1]
    recovered = tokenizer.decode(biological_ids, skip_special_tokens=False)
    if recovered != sequence or bare.ids != biological_ids:
        raise ValueError("E007 ProGen framed biological sequence recovery contradiction")
    if len(biological_ids) != len(sequence):
        raise ValueError("E007 ProGen silently substituted or dropped a biological residue")
    return {
        "framing_version": PROGEN_FRAMING_VERSION,
        "raw_tokenizer_round_trip": {
            "input": sequence,
            "ids": bare.ids,
            "tokens": bare.tokens,
            "decoded": tokenizer.decode(bare.ids, skip_special_tokens=False),
        },
        "prefixed_model_input": {
            "input": f"1{sequence}",
            "ids": prefixed.ids,
            "tokens": prefixed.tokens,
            "decoded": tokenizer.decode(prefixed.ids, skip_special_tokens=False),
        },
        "framed_model_input": {
            "input": f"1{sequence}2",
            "ids": framed.ids,
            "tokens": framed.tokens,
            "decoded": tokenizer.decode(framed.ids, skip_special_tokens=False),
        },
        "recovered_biological_sequence": recovered,
        "biological_residue_count": len(sequence),
        "total_framed_token_count": len(framed.ids),
        "framing_token_count": 2,
        "prefix_control": {"text": "1", "id": prefix_id},
        "suffix_control": {"text": "2", "id": suffix_id},
        "controls_are_amino_acids": False,
        "position_capacity": position_capacity,
    }


def progen_tokenizer_diagnostic(tokenizer: Any) -> dict[str, Any]:
    vocabulary = dict(sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]))
    single = {
        residue: {
            "ids": tokenizer.encode(residue).ids,
            "tokens": tokenizer.encode(residue).tokens,
            "decoded": tokenizer.decode(tokenizer.encode(residue).ids, skip_special_tokens=False),
        }
        for residue in CANONICAL
    }
    ambiguous = {
        residue: {
            "ids": tokenizer.encode(residue).ids,
            "tokens": tokenizer.encode(residue).tokens,
            "decoded": tokenizer.decode(tokenizer.encode(residue).ids, skip_special_tokens=False),
            "accepted_as_biological_input": False,
        }
        for residue in "BJOUXZ"
    }
    canonical = progen_framed_sequence_contract(tokenizer, CANONICAL, position_capacity=1024)
    length_500 = progen_framed_sequence_contract(tokenizer, CANONICAL * 25, position_capacity=1024)
    return {
        "vocabulary": vocabulary,
        "vocabulary_size": len(vocabulary),
        "canonical_single_residues": single,
        "canonical_alphabet": canonical,
        "ambiguous_and_unknown": ambiguous,
        "length_500": length_500,
        "whitespace_or_normalization_applied": False,
        "maximum_biological_length": 1022,
    }


def validate_tokenizer_contract_v3(candidate: str, tokenizer: Any) -> dict[str, Any]:
    if candidate == "esm2_150m":
        return validate_tokenizer_contract(candidate, tokenizer)
    if candidate != "progen2_151m":
        raise ValueError(f"E007 Phase 4B.2 tokenizer is unsupported: {candidate}")
    diagnostic = progen_tokenizer_diagnostic(tokenizer)
    return {
        "canonical_mapping": {
            residue: diagnostic["canonical_single_residues"][residue]["ids"][0] for residue in CANONICAL
        },
        "biological_length_500": diagnostic["length_500"]["biological_residue_count"],
        "framed_token_length_500": diagnostic["length_500"]["total_framed_token_count"],
        "vocabulary_size": diagnostic["vocabulary_size"],
        "framing_version": PROGEN_FRAMING_VERSION,
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
        or config.position_embedding_type != "rotary"
        or config.tie_word_embeddings is not True
        or raw_config.get("architectures") != ["EsmForMaskedLM"]
    ):
        raise ValueError("E007 Phase 4B.2 ESM architecture contract contradiction")
    tokenizer = EsmTokenizer.from_pretrained(directory, local_files_only=True)
    model = EsmForMaskedLM(config)
    state = load_file(str(require_local_path(directory / "model.safetensors", root=cache)), device="cpu")
    raw_inventory = tensor_inventory(state)
    normalized, normalization = normalize_esm_rotary_state_dict(
        state,
        model_state_keys=list(model.state_dict()),
        position_embedding_type=config.position_embedding_type,
        tie_word_embeddings=config.tie_word_embeddings,
    )
    result = model.load_state_dict(normalized, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("E007 Phase 4B.2 ESM strict state reconciliation failed")
    verify_esm_tied_output(model)
    accounting = esm_parameter_accounting(
        model,
        state,
        normalized,
        ignored_names=normalization["removed_exact_keys"],
        expected_unique_parameter_elements=expected["parameter_count"],
    )
    return (
        model,
        tokenizer,
        {
            "normalization_version": ESM_NORMALIZATION_VERSION,
            "normalization": normalization,
            "accounting": accounting,
            "raw_state_inventory": raw_inventory,
            "normalized_state_inventory": tensor_inventory(normalized),
        },
    )


def load_reviewed_candidate(candidate: str, cache_root: str | Path, *, device: Any) -> tuple[Any, Any, dict[str, Any]]:
    cache = Path(cache_root).resolve()
    if candidate == "esm2_150m":
        model, tokenizer, diagnostics = _load_esm(cache)
    elif candidate == "progen2_151m":
        model, tokenizer, diagnostics = _load_progen(cache)
    else:
        raise ValueError(f"E007 Phase 4B.2 loader supports primary candidates only: {candidate}")
    tokenizer_contract = validate_tokenizer_contract_v3(candidate, tokenizer)
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
        },
    )
