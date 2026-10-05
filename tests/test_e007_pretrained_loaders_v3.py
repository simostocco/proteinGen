from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from protein_distance_diffusion.evaluation import e007_pretrained_loaders_v3 as loaders
from protein_distance_diffusion.evaluation import e007_pretrained_sequence_prior_smoke as smoke

V1 = Path("reports/experiments/E007_matrix_sequence_cogeneration/pretrained_sequence_prior_smoke_v1")
V2 = Path("reports/experiments/E007_matrix_sequence_cogeneration/pretrained_sequence_prior_smoke_v2")
V1_HASHES = {
    "report.json": "539fdf552bdae48250454f418aae0ff1b61f6989656ba75a757b2a29c0eb3bdf",
    "protocol.json": "c802df1aa97b2d76d9039fc145e48b194dd30f2dc77c140ade2b4684f91e7940",
    "esm2_150m.json": "1805ed9316d546c883291f1a85b97f7af08cb6a5402dd50ff4a712f1c4601246",
    "progen2_151m.json": "d350daff81b6ade3559971b94d0e6176d1665f43c5f416e388680af565c662db",
    "proteinmpnn_ca_only.json": "20d93c233c53b88c40849d1034b8c3257574dd4e355bac5f798c8627a6e9fbf2",
}
V2_HASHES = {
    "report.json": "abdb26840405203a84adcdebbb14a9a3dbd2c1bfaf4e33d5202a1bbb0d511367",
    "protocol.json": "f4d76200b840651dd58bc46d01fefa2eee1c013069bb9a420481384397613ae1",
    "esm2_150m.json": "e9684c088bd492dc0cef457b3c4247cee00f2b66a39e8df6263b718890683cd0",
    "progen2_151m.json": "faf612c6a03da11d9cf63820ccc7d212b43a62f78c9492ffc1582e879916fae3",
    "proteinmpnn_ca_only.json": "9e9730210686c9612469ddc2490c6190f0b910ddb2519b40a56ff889efc00b0c",
}
CPU_DIAGNOSTIC = Path(
    "reports/experiments/E007_matrix_sequence_cogeneration/pretrained_sequence_prior_cpu_diagnostic_v1"
)
CPU_DIAGNOSTIC_HASHES = {
    "report.json": "4dd71e39d85ae17e5fbe3a5b05c5484046c89cf9a7fdc91b6b68057fd3be6e4f",
    "protocol.json": "b19e540ff1c90571943fa027675e6ed2fe444ad4d82bb6bd5f1001c51f1c4e96",
    "esm2_150m.json": "6983c19cf0660a00f790663c9f4a484a75d64cf1ffba6bfa712fc823bdaf624f",
    "progen2_151m.json": "543c3b15e4734c8d4c1d5793ed0a578b19d56c596b9e0e54d4457a305cb5b944",
}


class _TinyEsm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.esm = nn.Module()
        self.esm.embeddings = nn.Module()
        self.esm.embeddings.word_embeddings = nn.Embedding(7, 4)
        self.esm.encoder = nn.Module()
        self.esm.encoder.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32))
        self.esm.contact_head = nn.Linear(4, 1)
        self.lm_head = nn.Module()
        self.lm_head.dense = nn.Linear(4, 4)
        self.lm_head.decoder = nn.Linear(4, 7, bias=False)
        self.lm_head.decoder.weight = self.esm.embeddings.word_embeddings.weight


def _accounting_fixture() -> tuple[_TinyEsm, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model = _TinyEsm()
    normalized = dict(model.state_dict())
    normalized["lm_head.decoder.weight"] = normalized["esm.embeddings.word_embeddings.weight"]
    raw = {name: value for name, value in normalized.items() if name != "lm_head.decoder.weight"}
    raw["esm.embeddings.position_ids"] = torch.arange(16)[None]
    raw["esm.embeddings.position_embeddings.weight"] = torch.ones(16, 4)
    return model, raw, normalized


def test_esm_accounting_conserves_unique_alias_buffer_and_ignored_elements() -> None:
    model, raw, normalized = _accounting_fixture()
    result = loaders.esm_parameter_accounting(
        model,
        raw,
        normalized,
        ignored_names=[
            "esm.embeddings.position_ids",
            "esm.embeddings.position_embeddings.weight",
        ],
    )
    assert all(row["left"] == row["right"] for row in result["conservation_equations"].values())
    assert result["instantiated_named_parameter_elements_including_alias"] > result["unique_parameter_elements"]
    assert result["retained_serialized_buffer_elements"] == 4
    assert result["ignored_compatibility_tensors"][0]["name"] == "esm.embeddings.position_ids"
    assert result["tied_aliases"][0]["shared_storage"] is True


def test_esm_accounting_rejects_unexplained_element_and_wrong_pinned_count() -> None:
    model, raw, normalized = _accounting_fixture()
    with pytest.raises(ValueError, match="does not conserve"):
        loaders.esm_parameter_accounting(
            model,
            {**raw, "unexplained": torch.zeros(1)},
            normalized,
            ignored_names=[
                "esm.embeddings.position_ids",
                "esm.embeddings.position_embeddings.weight",
            ],
        )
    with pytest.raises(ValueError, match="unique parameter count contradiction"):
        loaders.esm_parameter_accounting(
            model,
            raw,
            normalized,
            ignored_names=[
                "esm.embeddings.position_ids",
                "esm.embeddings.position_embeddings.weight",
            ],
            expected_unique_parameter_elements=1,
        )


class _Tokenizer:
    def __init__(self) -> None:
        tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "1", "2", *"ABCDEFGHIKLMNOPQRSTUVWXYZ"]
        self.vocabulary = {token: index for index, token in enumerate(tokens)}
        self.inverse = {index: token for token, index in self.vocabulary.items()}

    def token_to_id(self, token: str) -> int | None:
        return self.vocabulary.get(token)

    def get_vocab(self) -> dict[str, int]:
        return self.vocabulary

    def encode(self, text: str) -> SimpleNamespace:
        ids = [self.vocabulary[character] for character in text if character in self.vocabulary]
        return SimpleNamespace(ids=ids, tokens=[self.inverse[index] for index in ids])

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return "".join(self.inverse[index] for index in ids)


def test_progen_bare_framed_and_unframed_canonical_contract() -> None:
    tokenizer = _Tokenizer()
    result = loaders.progen_framed_sequence_contract(tokenizer, loaders.CANONICAL, position_capacity=1024)
    assert result["raw_tokenizer_round_trip"]["decoded"] == loaders.CANONICAL
    assert result["prefixed_model_input"]["decoded"] == f"1{loaders.CANONICAL}"
    assert result["framed_model_input"]["decoded"] == f"1{loaders.CANONICAL}2"
    assert result["recovered_biological_sequence"] == loaders.CANONICAL
    assert result["biological_residue_count"] == 20
    assert result["total_framed_token_count"] == 22
    assert result["controls_are_amino_acids"] is False


def test_progen_length_500_capacity_and_ambiguous_refusal() -> None:
    tokenizer = _Tokenizer()
    result = loaders.progen_framed_sequence_contract(tokenizer, loaders.CANONICAL * 25, position_capacity=1024)
    assert result["biological_residue_count"] == 500
    assert result["total_framed_token_count"] == 502
    with pytest.raises(ValueError, match="noncanonical or ambiguous"):
        loaders.progen_framed_sequence_contract(tokenizer, "ACDX", position_capacity=1024)
    with pytest.raises(ValueError, match="exceeds positional capacity"):
        loaders.progen_framed_sequence_contract(tokenizer, "A" * 1023, position_capacity=1024)


def test_parameter_hash_and_fixture_forward_are_deterministic() -> None:
    torch.manual_seed(7302)
    model = nn.Sequential(nn.Embedding(30, 8), nn.Linear(8, 30)).eval()
    before = loaders.parameter_identity_hash(model)
    values = torch.tensor([[3, 5, 7, 4]])
    first = model(values)
    second = model(values)
    assert torch.equal(first, second)
    assert loaders.parameter_identity_hash(model) == before
    assert (
        hashlib.sha256(first.detach().numpy().tobytes()).hexdigest()
        == hashlib.sha256(second.detach().numpy().tobytes()).hexdigest()
    )


def test_v1_v2_are_immutable_and_v3_is_separate() -> None:
    assert {name: smoke.sha256_file(V1 / name) for name in V1_HASHES} == V1_HASHES
    assert {name: smoke.sha256_file(V2 / name) for name in V2_HASHES} == V2_HASHES
    config = smoke._load_config("configs/e007_pretrained_sequence_prior_smoke_v3.yaml")
    assert config["version"] == smoke.VERSION_V3
    assert Path(config["smoke_output"]).name == "pretrained_sequence_prior_smoke_v3"
    assert config["smoke"]["candidates"] == ["esm2_150m", "progen2_151m"]


def test_real_cpu_diagnostic_is_pinned_non_authorizing_and_exactly_replayed() -> None:
    assert {name: smoke.sha256_file(CPU_DIAGNOSTIC / name) for name in CPU_DIAGNOSTIC_HASHES} == CPU_DIAGNOSTIC_HASHES
    esm = json.loads((CPU_DIAGNOSTIC / "esm2_150m.json").read_text())
    progen = json.loads((CPU_DIAGNOSTIC / "progen2_151m.json").read_text())
    accounting = esm["loader_metadata"]["diagnostics"]["accounting"]
    assert accounting["unique_parameter_elements"] == 148_140_154
    assert accounting["raw_checkpoint_total_tensor_elements"] == 148_798_300
    assert esm["first_logits_sha256"] == esm["replay_logits_sha256"]
    assert progen["first_logits_sha256"] == progen["replay_logits_sha256"]
    assert esm["parameter_sha256_before"] == esm["parameter_sha256_after"]
    assert progen["parameter_sha256_before"] == progen["parameter_sha256_after"]
    assert progen["tokenizer_diagnostic"]["length_500"]["total_framed_token_count"] == 502
    assert esm["authorizes_training"] is progen["authorizes_training"] is False


def test_v3_loader_failure_never_becomes_scientific_rejection() -> None:
    incomplete = [
        {"candidate": "esm2_150m", "status": "failed", "smoke_completed": False},
        {"candidate": "progen2_151m", "status": "passed", "smoke_completed": True},
    ]
    assert smoke._decision(incomplete, version=smoke.VERSION_V3) == "candidate_loader_review_required"
    complete = [dict(row, smoke_completed=True) for row in incomplete]
    assert smoke._decision(complete, version=smoke.VERSION_V3) == "progen2_only_advances_after_completed_smoke"


def test_v3_plan_is_side_effect_free(tmp_path: Path) -> None:
    source = yaml.safe_load(Path("configs/e007_pretrained_sequence_prior_smoke_v3.yaml").read_text())
    output = tmp_path / "never-created"
    source["smoke_output"] = str(output)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False))
    result = smoke.plan(path)
    assert result["model_created"] is False
    assert result["cuda_used"] is False
    assert not output.exists()
    assert not output.with_name(f".{output.name}.inprogress").exists()
