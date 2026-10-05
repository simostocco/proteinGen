from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
import torch
import yaml

from protein_distance_diffusion.evaluation import e007_pretrained_loaders as loaders
from protein_distance_diffusion.evaluation import e007_pretrained_sequence_prior_smoke as smoke
from protein_distance_diffusion.models.e007_proteinmpnn_ca import ProteinMPNNCA

ENVIRONMENT_PATH = Path("environment/e007_phase4b_environment.yaml")
CONFIG_PATH = Path("configs/e007_pretrained_sequence_prior_smoke_v1.yaml")


class _EsmTokenizerFixture:
    vocabulary = {token: index for index, token in enumerate(["<cls>", "<pad>", "<eos>", "<unk>", *loaders.CANONICAL])}
    vocabulary.update({"<mask>": 32})
    cls_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    mask_token_id = 32

    def __len__(self) -> int:
        return 33

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocabulary[token]

    def __call__(self, sequence: str, *, add_special_tokens: bool) -> dict:
        values = [self.vocabulary[residue] for residue in sequence]
        return {"input_ids": [self.cls_token_id, *values, self.eos_token_id] if add_special_tokens else values}

    def decode(self, values: list[int], *, skip_special_tokens: bool) -> str:
        reverse = {value: key for key, value in self.vocabulary.items()}
        specials = {self.cls_token_id, self.pad_token_id, self.eos_token_id, self.mask_token_id}
        return "".join(reverse[value] for value in values if not skip_special_tokens or value not in specials)


class _Encoding:
    def __init__(self, values: list[int]) -> None:
        self.ids = values


class _ProGenTokenizerFixture:
    vocabulary = {"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2, "1": 3, "2": 4}
    vocabulary.update({residue: index + 5 for index, residue in enumerate(loaders.CANONICAL)})
    vocabulary.update({"B": 25, "O": 26, "U": 27, "X": 28, "Z": 29})

    def token_to_id(self, token: str) -> int | None:
        return self.vocabulary.get(token)

    def get_vocab_size(self) -> int:
        return 30

    def encode(self, sequence: str) -> _Encoding:
        return _Encoding([self.vocabulary[token] for token in sequence])

    def decode(self, values: list[int]) -> str:
        reverse = {value: key for key, value in self.vocabulary.items()}
        return "".join(reverse[value] for value in values)


def test_installable_environment_contract_and_dependencies() -> None:
    payload = loaders.environment_contract(ENVIRONMENT_PATH)
    versions = loaders.declared_environment_versions(ENVIRONMENT_PATH)
    assert payload["name"] == "proteingen-e007-phase4b"
    assert {"python", "torch", "transformers", "tokenizers", "safetensors", "numpy", "pyyaml"} <= set(versions)
    assert loaders.declared_environment_fingerprint(ENVIRONMENT_PATH) == (
        "4c3c8496b4798a59ded9299279fd5417db57f6cd50e1f85750086c3a3fea2ba8"
    )


def test_source_environment_yaml_is_metadata_not_installable() -> None:
    source = yaml.safe_load(Path("environment/e007_phase4b_environment_lock.yaml").read_text())
    assert "channels" not in source
    assert isinstance(source["packages"], dict)


def test_offline_guard_sets_variables_refuses_network_and_restores() -> None:
    previous = {name: os.environ.get(name) for name in loaders.OFFLINE_ENVIRONMENT}
    original_socket = socket.socket
    with loaders.offline_network_guard():
        assert all(os.environ[name] == "1" for name in loaders.OFFLINE_ENVIRONMENT)
        with pytest.raises(RuntimeError, match="network access is forbidden"):
            socket.socket()
    assert socket.socket is original_socket
    assert {name: os.environ.get(name) for name in loaders.OFFLINE_ENVIRONMENT} == previous


@pytest.mark.parametrize("candidate", ["esm2_150m", "progen2_151m"])
def test_strict_state_key_and_shape_validation_with_tiny_fixtures(candidate: str) -> None:
    state = {"embedding.weight": torch.zeros(5, 3), "head.bias": torch.zeros(5)}
    result = loaders.validate_tensor_mapping(
        state,
        expected_keys=["embedding.weight", "head.bias"],
        expected_shapes={"embedding.weight": [5, 3], "head.bias": [5]},
        expected_tensor_count=2,
        expected_parameter_count=20,
    )
    assert result == {"state_tensor_count": 2, "state_element_count": 20}
    with pytest.raises(ValueError, match="key contradiction"):
        loaders.validate_tensor_mapping(state, expected_keys=["embedding.weight"])
    with pytest.raises(ValueError, match="shape contradiction"):
        loaders.validate_tensor_mapping(state, expected_shapes={"embedding.weight": [3, 5]})
    assert loaders.EXPECTED[candidate]["loader"].startswith("transformers_native")


def test_weights_only_loader_refuses_unsafe_pickle_global(tmp_path: Path) -> None:
    class Unsafe:
        def __reduce__(self):
            return (eval, ("1 + 1",))

    path = tmp_path / "unsafe.pt"
    torch.save({"payload": Unsafe()}, path)
    with pytest.raises(ValueError, match="safe weights-only checkpoint loading failed"):
        loaders.safe_torch_checkpoint_metadata(path)


def test_proteinmpnn_ca_contract_and_full_backbone_refusal() -> None:
    model = ProteinMPNNCA(k_neighbors=48)
    assert len(model.state_dict()) == loaders.EXPECTED["proteinmpnn_ca_only"]["state_tensor_count"]
    assert (
        sum(value.numel() for value in model.parameters()) == loaders.EXPECTED["proteinmpnn_ca_only"]["parameter_count"]
    )
    with pytest.raises(ValueError, match=r"shape \[batch, length, 3\]"):
        model.features(torch.zeros(1, 8, 4, 3), torch.ones(1, 8), torch.arange(8)[None], torch.zeros(1, 8))


def test_tokenizer_contracts_and_length_500_accounting() -> None:
    esm = loaders.validate_tokenizer_contract("esm2_150m", _EsmTokenizerFixture())
    progen = loaders.validate_tokenizer_contract("progen2_151m", _ProGenTokenizerFixture())
    assert esm["encoded_length_500"] == progen["encoded_length_500"] == 502
    assert len(set(esm["canonical_mapping"].values())) == 20
    assert len(set(progen["canonical_mapping"].values())) == 20


def test_local_path_confinement_and_missing_artifact_refusal(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    local = root / "artifact.bin"
    local.write_bytes(b"fixture")
    assert loaders.require_local_path(local, root=root) == local.resolve()
    with pytest.raises(ValueError, match="absent or outside"):
        loaders.require_local_path(tmp_path / "outside.bin", root=root)
    with pytest.raises(ValueError, match="absent or outside"):
        loaders.require_local_path(root / "missing.bin", root=root)


def test_reviewed_readiness_and_locked_artifact_hashes_verify() -> None:
    config = smoke._load_config(CONFIG_PATH)
    readiness = smoke._verify_loader_readiness(config)
    assert all(row["status"] == "reviewed_ready" for row in readiness["loaders"].values())
    result = smoke.verify_artifacts_offline(CONFIG_PATH)
    assert result["status"] == "verified_offline"


def test_environment_gate_precedes_output_creation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = smoke._load_config(CONFIG_PATH)
    output = tmp_path / "smoke"
    config["smoke_output"] = str(output)
    monkeypatch.setattr(smoke, "_load_config", lambda _path: config)
    monkeypatch.setattr(smoke, "verify_artifacts_offline", lambda _path: {"status": "verified"})

    def refused(_path):
        raise ValueError("environment fingerprint mismatch")

    with pytest.raises(ValueError, match="environment fingerprint mismatch"):
        smoke.run_smoke(CONFIG_PATH, environment_verifier=refused)
    assert not output.exists()
    assert not output.with_name(f".{output.name}.inprogress").exists()


def test_candidate_specific_failure_is_not_silently_skipped() -> None:
    result = smoke._decision(
        [
            {"candidate": "esm2_150m", "status": "passed"},
            {"candidate": "progen2_151m", "status": "loader_review_required"},
            {"candidate": "proteinmpnn_ca_only", "status": "passed"},
        ]
    )
    assert result == "esm2_only_advances"
