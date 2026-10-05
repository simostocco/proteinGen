from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

import protein_distance_diffusion.evaluation.e006_stage_a_objective_parity as parity
import protein_distance_diffusion.training.rich_codesign_production as production
from protein_distance_diffusion.evaluation.e006_stage_a_objective_parity import (
    classify_root_cause,
    evaluate_frozen_batch,
    published_artifacts,
    reduction_diagnostics,
    token_cross_entropy,
    validate_paired_context,
)
from protein_distance_diffusion.training.stage_a_context import context_corruption
from protein_distance_diffusion.training.stage_a_context_smoke import _tiny_model


def _batch() -> dict:
    return {
        "sample_ids": ["a", "b"],
        "tokens": torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 0, 0]]),
        "residue_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool),
    }


def test_visible_shuffle_preserves_masks_targets_padding_and_multisets() -> None:
    batch = _batch()
    corruption = context_corruption(
        batch["tokens"], batch["residue_mask"], mask_token_id=1, probability=0.4, seed=6, step=2
    )
    evidence = validate_paired_context(
        batch["tokens"],
        corruption.inputs,
        corruption.shuffled_inputs,
        corruption.corrupted_mask,
        batch["residue_mask"],
    )
    assert evidence["mask_positions_fixed"] is True
    assert evidence["padding_positions_fixed"] is True
    assert evidence["target_leakage_absent"] is True


def test_target_exposure_and_moved_mask_are_rejected() -> None:
    batch = _batch()
    corruption = context_corruption(
        batch["tokens"], batch["residue_mask"], mask_token_id=1, probability=0.4, seed=6, step=2
    )
    broken = corruption.shuffled_inputs.clone()
    broken[corruption.corrupted_mask] = batch["tokens"][corruption.corrupted_mask]
    with pytest.raises(ValueError, match="MASK positions"):
        validate_paired_context(
            batch["tokens"], corruption.inputs, broken, corruption.corrupted_mask, batch["residue_mask"]
        )


def test_cross_entropy_uses_only_corrupted_canonical_positions() -> None:
    batch = _batch()
    corrupted = torch.zeros_like(batch["residue_mask"])
    corrupted[0, 1] = True
    corrupted[1, 2] = True
    logits = torch.randn(2, 6, 22)
    losses, selected = token_cross_entropy(logits, batch["tokens"], corrupted, batch["residue_mask"])
    assert losses.shape == (2,)
    assert selected.sum() == 2
    changed = logits.clone()
    changed[~selected] = 1e6
    assert torch.equal(losses, token_cross_entropy(changed, batch["tokens"], corrupted, batch["residue_mask"])[0])


def test_reduction_diagnostics_exposes_easy_hard_cancellation() -> None:
    normal = torch.tensor([0.0, 3.0])
    shuffled = torch.tensor([3.0, 0.0])
    result = reduction_diagnostics(normal, shuffled, torch.tensor([0, 1]), sample_count=2, margin_nats=0.05)
    assert result["current_hinge_after_batch_token_mean"] == pytest.approx(0.05)
    assert result["mean_of_per_token_hinges"] > result["current_hinge_after_batch_token_mean"]
    assert result["easy_examples_cancel_hard_examples"] is True
    assert result["active_hinge_token_fraction"] == 0.5


def test_eval_parity_disables_dropout_and_train_mode_is_measured() -> None:
    model = _tiny_model(19)
    evidence, token_rows, sample_rows = evaluate_frozen_batch(
        model,
        _batch(),
        seed=12,
        corruption_step=3,
        mask_fraction=0.4,
        margin_nats=0.05,
        device=torch.device("cpu"),
        use_amp=False,
    )
    assert evidence["dropout_disabled_parity"]["identical_computation_confirmed"] is True
    assert evidence["train_eval_max_abs_normal_logit_difference"] > 0
    assert {row["path"] for row in token_rows} == {
        "monitor_eval_fp32",
        "production_eval_fp32",
        "production_eval_amp",
        "production_train_sequential_dropout",
        "train_paired_dropout",
    }
    assert sample_rows


def test_root_cause_stops_at_first_stochastic_divergence() -> None:
    record = {
        "invariants": {"all": True},
        "train_eval_max_abs_normal_logit_difference": 0.2,
        "dropout_disabled_parity": {"production_amp_max_abs_normal_logit_difference": 0.01},
        "paths": {"x": {"easy_examples_cancel_hard_examples": True}},
    }
    result = classify_root_cause([record])
    assert result["classification"] == "train_eval_stochastic_mode_divergence"
    assert result["objective_change_implemented"] is False


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_startup_uses_production_shaped_one_argument_authorization_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paused = tmp_path / "paused"
    paused.mkdir()
    (paused / "protocol.json").write_text(json.dumps({"status": "paused_for_scientific_review"}))
    checkpoint = paused / "step.pt"
    checkpoint.write_bytes(b"immutable-checkpoint")
    paused_before = parity._tree_hashes(paused)
    dataset = {
        "directory": str(tmp_path / "sidecars"),
        "protocol_sha256": "1" * 64,
        "schema_sha256": "2" * 64,
        "vocabulary_sha256": "3" * 64,
        "normalization_sha256": "4" * 64,
        "shard_inventory_sha256": "5" * 64,
    }
    base_path = tmp_path / "production.yaml"
    base_path.write_text(yaml.safe_dump({"dataset": dataset, "training": {"output_dir": str(paused)}}))
    output = tmp_path / "must-remain-absent"
    audit_path = tmp_path / "audit.yaml"
    audit_path.write_text(
        yaml.safe_dump(
            {
                "base_training_config": str(base_path),
                "base_training_config_sha256": _digest(base_path),
                "output_dir": str(output),
                "checkpoints": [{"name": "step_2500", "path": str(checkpoint), "sha256": _digest(checkpoint)}],
            }
        )
    )
    monkeypatch.setattr(parity, "load_checkpoint", lambda *_args, **_kwargs: {"optimizer_step": 2500})

    class ExpectedAuthorizationStop(RuntimeError):
        pass

    def one_argument_authorization(config):
        assert config["dataset"] == dataset
        raise ExpectedAuthorizationStop

    monkeypatch.setattr(parity, "_authorization", one_argument_authorization)
    with pytest.raises(ExpectedAuthorizationStop):
        parity.run_objective_parity_audit(audit_path)
    assert not output.exists()
    assert published_artifacts(output) == []
    assert parity._tree_hashes(paused) == paused_before
    assert _digest(checkpoint) == hashlib.sha256(b"immutable-checkpoint").hexdigest()


def test_canonical_authorization_forwards_every_pinned_dataset_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = {
        "directory": "/immutable/sidecars",
        "protocol_sha256": "1" * 64,
        "schema_sha256": "2" * 64,
        "vocabulary_sha256": "3" * 64,
        "normalization_sha256": "4" * 64,
        "shard_inventory_sha256": "5" * 64,
    }
    observed = {}

    def fake_authorize(root, **expected):
        observed.update(root=root, **expected)
        return "authorized"

    monkeypatch.setattr(production, "authorize_rich_geometry_dataset", fake_authorize)
    assert parity._authorization({"dataset": dataset}) == "authorized"
    assert observed == {
        "root": dataset["directory"],
        "expected_protocol_sha256": dataset["protocol_sha256"],
        "expected_schema_sha256": dataset["schema_sha256"],
        "expected_vocabulary_sha256": dataset["vocabulary_sha256"],
        "expected_normalization_sha256": dataset["normalization_sha256"],
        "expected_shard_inventory_sha256": dataset["shard_inventory_sha256"],
    }
