from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import yaml

import protein_distance_diffusion.evaluation.e007_pretrained_sequence_prior as prior

CONFIG_PATH = Path("configs/e007_pretrained_sequence_prior_audit_v1.yaml")


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _temporary_config(tmp_path: Path) -> Path:
    config = _config()
    config["output_dir"] = str(tmp_path / "audit")
    config["phase4b"]["output_dir"] = str(tmp_path / "phase4b")
    config["phase4b"]["staging_dir"] = str(tmp_path / ".phase4b.inprogress")
    config["repository_audit"]["bounded_cache_roots"] = [str(tmp_path / "cache")]
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_contract_separates_facts_estimates_questions_and_tests() -> None:
    config = prior._load_config(CONFIG_PATH)
    verified = prior.verify_prerequisites(config)
    contract = verified["contract"]
    assert contract["version"] == prior.CONTRACT_VERSION
    assert {"verified_facts", "estimates", "unresolved_questions", "future_executable_tests"} <= set(contract)
    models = contract["verified_facts"]["models"]
    assert models["esm2_150m"]["architecture"] == "bidirectional masked protein language model"
    assert models["progen2_151m"]["architecture"] == "causal autoregressive protein language model"
    assert "CA-only checkpoint required" in models["proteinmpnn_ca_only"]["atomic_coordinate_requirements"]
    assert models["esm3_open_small"]["parameter_count"] == 1_400_000_000
    assert all(model["continuous_latent_decoder"] is False for model in models.values())
    assert all(model["directly_sampleable_internal_representation"] is False for model in models.values())


def test_dataset_vocabulary_and_tokenizer_boundaries() -> None:
    verified = prior.verify_prerequisites(prior._load_config(CONFIG_PATH))
    assert verified["vocabulary"] == {
        "tokens": ["<PAD>", "<MASK>", *prior.CANONICAL_RESIDUES],
        "unknown_token": None,
        "version": "canonical_20_pad_mask_v1",
    }
    models = verified["contract"]["verified_facts"]["models"]
    assert models["esm2_150m"]["native_token_vocabulary_size"] == 33
    assert models["progen2_151m"]["native_token_vocabulary_size"] == 32
    assert models["proteinmpnn_ca_only"]["native_token_vocabulary_size"] == 21
    assert "X must never be emitted" in models["proteinmpnn_ca_only"]["tokenizer_notes"]


def test_conditioning_contract_prevents_identity_leakage_and_false_latent_claims() -> None:
    contract = prior.verify_prerequisites(prior._load_config(CONFIG_PATH))["contract"]
    scientific = contract["scientific_contract"]
    assert scientific["common_geometry"]["geometry_identity_inputs_forbidden"] is True
    assert scientific["common_geometry"]["native_backbone_reconstruction_forbidden"] is True
    assert scientific["evaluation"]["aggregate_scalar_score_forbidden"] is True
    assert scientific["esm2_conditioning"]["objective"].startswith("mean canonical 20-way")
    assert "every valid residue" in scientific["progen2_conditioning"]["causal_geometry_rule"]
    assert set(scientific["controls"]) >= {
        "sequence_prior_only",
        "zero_geometry_conditioning",
        "geometry_shuffle",
        "proper_rotation",
        "spatial_reflection",
        "padding_extension",
    }


def test_phase4b_plan_is_exact_bounded_and_blocked(tmp_path: Path) -> None:
    config_path = _temporary_config(tmp_path)
    result = prior.plan_phase4b(config_path)
    assert result["case_count"] == 3 * 5 * 4
    assert result["weight_download_performed"] is False
    assert result["model_loaded"] is False
    assert result["network_accessed"] is False
    assert result["cache_written"] is False
    assert result["output_created"] is False
    assert "phase4b_execution_not_authorized" in result["blockers"]
    assert any(item.startswith("esm2_150m:") for item in result["blockers"])
    assert not (tmp_path / "phase4b").exists()
    assert not (tmp_path / ".phase4b.inprogress").exists()


def test_plan_only_has_no_output_network_model_or_cache_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _temporary_config(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    marker = cache / "marker"
    marker.write_bytes(b"unchanged")
    before = prior.sha256_file(marker)

    def forbidden_network(*_args, **_kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr("socket.socket.connect", forbidden_network)
    result = prior.plan_audit(config_path)
    assert result["outcome"] == prior.OUTCOME
    assert result["shortlist"] == ["esm2_150m", "progen2_151m"]
    assert result["output_created"] is False
    assert result["network_accessed"] is False
    assert result["weight_cache_written"] is False
    assert result["environment"]["candidate_cache"]["candidate_weight_match_count"] == 0
    assert prior.sha256_file(marker) == before
    assert not (tmp_path / "audit").exists()
    assert not (tmp_path / ".audit.inprogress").exists()
    source = inspect.getsource(prior)
    assert "import torch" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source


def test_publication_is_atomic_separate_and_non_authorizing(tmp_path: Path) -> None:
    config_path = _temporary_config(tmp_path)
    result = prior.publish_audit(config_path)
    output = tmp_path / "audit"
    assert result["status"] == "completed_read_only_non_authorizing"
    assert output.is_dir()
    assert not (tmp_path / ".audit.inprogress").exists()
    assert {item.name for item in output.iterdir()} == {
        "report.json",
        "protocol.json",
        "artifact_inventory.json",
        "heartbeat.json",
    }
    assert (output / "report.json").read_bytes() != (output / "protocol.json").read_bytes()
    report = json.loads((output / "report.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    assert report["outcome"] == prior.OUTCOME
    assert {field: protocol[field] for field in prior.NON_AUTHORIZING} == prior.NON_AUTHORIZING
    assert protocol["network_accessed"] is False
    assert protocol["weight_cache_written"] is False
    with pytest.raises(FileExistsError):
        prior.publish_audit(config_path)


def test_hash_mismatch_refuses_before_output_creation(tmp_path: Path) -> None:
    config = _config()
    config["output_dir"] = str(tmp_path / "audit")
    config["contract"]["sha256"] = "0" * 64
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="contract"):
        prior.publish_audit(path)
    assert not (tmp_path / "audit").exists()
    assert not (tmp_path / ".audit.inprogress").exists()


def test_protected_prerequisites_remain_unchanged_after_temporary_publication(tmp_path: Path) -> None:
    config = prior._load_config(CONFIG_PATH)
    before = prior.verify_prerequisites(config)["hashes"]
    prior.publish_audit(_temporary_config(tmp_path))
    after = prior.verify_prerequisites(config)["hashes"]
    assert after == before


def test_phase4b_rejects_mutable_or_unresolved_revisions() -> None:
    config = prior._load_config(CONFIG_PATH)
    blockers = prior._phase4b_blockers(config)
    unresolved = [
        name
        for name, candidate in config["phase4b"]["candidates"].items()
        if candidate["permitted"] and candidate["revision"] is None
    ]
    assert unresolved == ["esm2_150m", "progen2_151m", "proteinmpnn_ca_only"]
    assert all(any(item.startswith(f"{name}:") for item in blockers) for name in unresolved)
    assert blockers[-1] == "phase4b_execution_not_authorized"


def test_primary_models_support_length_500_by_declared_position_contract() -> None:
    models = prior.verify_prerequisites(prior._load_config(CONFIG_PATH))["contract"]["verified_facts"]["models"]
    assert models["esm2_150m"]["maximum_positions"] >= 502
    assert models["progen2_151m"]["maximum_positions"] >= 502
    assert models["esm3_open_small"]["maximum_positions"] is None
