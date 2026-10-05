from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from protein_distance_diffusion.evaluation import e007_pretrained_loaders_v2 as loaders
from protein_distance_diffusion.evaluation import e007_pretrained_sequence_prior_smoke as smoke
from protein_distance_diffusion.models.e007_progen2 import ProGen2Config, ProGen2ForCausalLM

V1 = Path("reports/experiments/E007_matrix_sequence_cogeneration/pretrained_sequence_prior_smoke_v1")
V2_CONFIG = Path("configs/e007_pretrained_sequence_prior_smoke_v2.yaml")
V1_HASHES = {
    "report.json": "539fdf552bdae48250454f418aae0ff1b61f6989656ba75a757b2a29c0eb3bdf",
    "protocol.json": "c802df1aa97b2d76d9039fc145e48b194dd30f2dc77c140ade2b4684f91e7940",
    "esm2_150m.json": "1805ed9316d546c883291f1a85b97f7af08cb6a5402dd50ff4a712f1c4601246",
    "progen2_151m.json": "d350daff81b6ade3559971b94d0e6176d1665f43c5f416e388680af565c662db",
    "proteinmpnn_ca_only.json": "20d93c233c53b88c40849d1034b8c3257574dd4e355bac5f798c8627a6e9fbf2",
}


class _TinyTiedEsm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.esm = nn.Module()
        self.esm.embeddings = nn.Module()
        self.esm.embeddings.word_embeddings = nn.Embedding(7, 4)
        self.lm_head = nn.Module()
        self.lm_head.decoder = nn.Linear(4, 7, bias=False)
        self.lm_head.decoder.weight = self.esm.embeddings.word_embeddings.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.esm.embeddings.word_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head.decoder

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.lm_head.decoder(self.esm.embeddings.word_embeddings(values))


def _esm_raw_fixture() -> tuple[_TinyTiedEsm, dict[str, torch.Tensor]]:
    model = _TinyTiedEsm()
    state = {
        "esm.embeddings.word_embeddings.weight": torch.arange(28, dtype=torch.float32).reshape(7, 4),
        "esm.embeddings.position_ids": torch.arange(16)[None],
        "esm.embeddings.position_embeddings.weight": torch.ones(16, 4),
    }
    return model, state


def test_exact_esm_raw_mismatch_and_approved_normalization() -> None:
    model, raw = _esm_raw_fixture()
    expected = set(model.state_dict())
    assert expected - set(raw) == {"lm_head.decoder.weight"}
    assert set(raw) - expected == {
        "esm.embeddings.position_ids",
        "esm.embeddings.position_embeddings.weight",
    }
    normalized, evidence = loaders.normalize_esm_rotary_state_dict(
        raw,
        model_state_keys=list(model.state_dict()),
        position_embedding_type="rotary",
        tie_word_embeddings=True,
    )
    assert set(normalized) == expected
    assert (
        normalized["lm_head.decoder.weight"].data_ptr()
        == normalized["esm.embeddings.word_embeddings.weight"].data_ptr()
    )
    assert evidence["normalized_missing_keys"] == evidence["normalized_unexpected_keys"] == []


def test_esm_normalization_rejects_extra_difference_and_absolute_positions() -> None:
    model, raw = _esm_raw_fixture()
    with pytest.raises(ValueError, match="exceeds approved normalization"):
        loaders.normalize_esm_rotary_state_dict(
            {**raw, "unexpected.extra": torch.zeros(1)},
            model_state_keys=list(model.state_dict()),
            position_embedding_type="rotary",
            tie_word_embeddings=True,
        )
    with pytest.raises(ValueError, match="require rotary"):
        loaders.normalize_esm_rotary_state_dict(
            raw,
            model_state_keys=list(model.state_dict()),
            position_embedding_type="absolute",
            tie_word_embeddings=True,
        )


def test_esm_tied_storage_and_deterministic_fixture_logits() -> None:
    model, raw = _esm_raw_fixture()
    normalized, _ = loaders.normalize_esm_rotary_state_dict(
        raw,
        model_state_keys=list(model.state_dict()),
        position_embedding_type="rotary",
        tie_word_embeddings=True,
    )
    model.load_state_dict(normalized, strict=True)
    loaders.verify_esm_tied_output(model)
    tokens = torch.tensor([[0, 1, 2, 3]])
    assert torch.equal(model(tokens), model(tokens))
    model.lm_head.decoder = nn.Linear(4, 7, bias=False)
    with pytest.raises(ValueError, match="do not share storage"):
        loaders.verify_esm_tied_output(model)


def _tiny_progen() -> ProGen2ForCausalLM:
    return ProGen2ForCausalLM(
        ProGen2Config(n_embd=16, n_head=4, n_layer=2, n_positions=32, rotary_dim=4, vocab_size=11)
    )


def test_progen_strict_schema_and_deterministic_fixture_logits() -> None:
    torch.manual_seed(41)
    model = _tiny_progen().eval()
    state = model.state_dict()
    clone = _tiny_progen().eval()
    clone.load_state_dict(state, strict=True)
    tokens = torch.tensor([[1, 4, 7, 2]])
    mask = torch.ones_like(tokens)
    first = clone(input_ids=tokens, attention_mask=mask).logits
    second = clone(input_ids=tokens, attention_mask=mask).logits
    assert first.shape == (1, 4, 11)
    assert torch.equal(first, second)
    corrupted = dict(state)
    corrupted["unexpected"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        clone.load_state_dict(corrupted, strict=True)


def test_progen_preserves_four_way_query_value_key_checkpoint_packing() -> None:
    attention = _tiny_progen().transformer.h[0].attn

    class Packed(nn.Module):
        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return torch.arange(48, dtype=hidden.dtype).reshape(1, 1, 48)

    attention.qkv_proj = Packed()
    query, key, value = attention._project_qkv(torch.zeros(1, 1, 16))
    assert query.flatten().tolist() == [*range(0, 4), *range(12, 16), *range(24, 28), *range(36, 40)]
    assert value.flatten().tolist() == [*range(4, 8), *range(16, 20), *range(28, 32), *range(40, 44)]
    assert key.flatten().tolist() == [*range(8, 12), *range(20, 24), *range(32, 36), *range(44, 48)]


def test_progen_transformers_support_is_explicit_and_no_top_level_assumption() -> None:
    result = loaders.inspect_transformers_progen_support()
    assert result["top_level_import_used"] is False
    source = Path(loaders.__file__).read_text()
    assert "from transformers import ProGen" not in source


def test_v1_hashes_and_v2_output_are_separate() -> None:
    assert {name: smoke.sha256_file(V1 / name) for name in V1_HASHES} == V1_HASHES
    config = smoke._load_config(V2_CONFIG)
    assert Path(config["smoke_output"]).name == "pretrained_sequence_prior_smoke_v2"
    assert Path(config["smoke_output"]) != V1


def test_loader_failure_cannot_become_scientific_rejection() -> None:
    results = [
        {"candidate": "esm2_150m", "status": "passed", "smoke_completed": True},
        {
            "candidate": "progen2_151m",
            "status": "failed",
            "smoke_completed": False,
            "failure_stage": "candidate_loading",
        },
        {"candidate": "proteinmpnn_ca_only", "status": "passed", "smoke_completed": True},
    ]
    assert smoke._decision(results, version=smoke.VERSION_V2) == "candidate_loader_review_required"
    completed = [dict(row, smoke_completed=True) for row in results]
    assert smoke._decision(completed, version=smoke.VERSION_V2) == "esm2_only_advances"


def test_external_device_usage_is_separate_from_process_local_cuda_telemetry() -> None:
    fake = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None, mem_get_info=lambda: (6 * 2**20, 8 * 2**20)))
    result = smoke._cuda_device_baseline(fake, SimpleNamespace(type="cuda"))
    assert result == {
        "device_total_mib": 8.0,
        "device_free_before_mib": 6.0,
        "device_wide_baseline_used_mib": 2.0,
    }


def test_v2_configuration_preserves_v1_smoke_contract() -> None:
    config = smoke._load_config(V2_CONFIG)
    assert config["version"] == smoke.VERSION_V2
    assert config["smoke"]["lengths"] == [64, 128, 256, 384, 500]
    assert len(config["smoke"]["tests"]) == 4
    assert config["smoke"]["seed"] == 7401
    assert config["smoke"]["optimizer_updates"] == 0
    assert config["evidence_policy"]["supersedes_v1_scope"] == "primary_candidate_feasibility_only"


def test_v1_failure_is_recorded_as_loader_failure() -> None:
    esm = json.loads((V1 / "esm2_150m.json").read_text())
    progen = json.loads((V1 / "progen2_151m.json").read_text())
    assert "Missing key(s)" in esm["error_message"]
    assert "cannot import name 'ProGenConfig'" in progen["error_message"]
