from __future__ import annotations

import hashlib
import json

import torch
import torch.nn.functional as F
import yaml

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.rich_codesign_production import validate_phase3_config
from protein_distance_diffusion.training.stage_a_context import (
    PairedDropoutEvidence,
    contextual_stage_a_loss_v6,
    paired_dropout_forwards,
)
from protein_distance_diffusion.training.stage_a_context_production import classify_context_health_v6
from protein_distance_diffusion.training.stage_a_context_v6_smoke import (
    OBJECTIVE_ARMS,
    plan_synthetic_smoke_v6,
    run_synthetic_smoke_v6,
)


class _DropoutRecorder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(4, 22, bias=False)
        self.observed_rng_states: list[torch.Tensor] = []

    def forward_sequence_pretraining(self, inputs: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        self.observed_rng_states.append(torch.get_rng_state().clone())
        encoded = F.one_hot(inputs.remainder(4), num_classes=4).float()
        return self.projection(F.dropout(encoded, p=0.5, training=True)) * residue_mask.unsqueeze(-1)


def test_paired_dropout_is_reproducible_and_advances_global_rng_once() -> None:
    model = _DropoutRecorder()
    normal = torch.tensor([[1, 2, 3, 2]])
    shuffled = torch.tensor([[1, 3, 2, 2]])
    mask = torch.ones_like(normal, dtype=torch.bool)
    torch.manual_seed(71)
    initial = torch.get_rng_state().clone()
    first = paired_dropout_forwards(model, normal, shuffled, mask)
    final = torch.get_rng_state().clone()
    assert torch.equal(model.observed_rng_states[0], model.observed_rng_states[1])
    assert first[2].cpu_rng_paired is True
    assert first[2].global_rng_advanced_once is True

    reference = _DropoutRecorder()
    reference.load_state_dict(model.state_dict())
    torch.set_rng_state(initial)
    reference.forward_sequence_pretraining(normal, mask)
    assert torch.equal(torch.get_rng_state(), final)

    model.observed_rng_states.clear()
    torch.set_rng_state(initial)
    second = paired_dropout_forwards(model, normal, shuffled, mask)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def _target_logits(targets: torch.Tensor, selected: torch.Tensor, losses: list[float]) -> torch.Tensor:
    logits = torch.zeros((*targets.shape, 22), dtype=torch.float32)
    for (row, column), loss in zip(torch.nonzero(selected).tolist(), losses, strict=True):
        target = int(targets[row, column])
        logits[row, column, target] = -loss
    return logits.requires_grad_()


def test_per_sample_hinge_survives_token_weighted_cancellation_and_equalizes_proteins() -> None:
    targets = torch.tensor([[2, 0, 0, 0], [3, 3, 3, 3]])
    residue_mask = targets.ne(0)
    corrupted = residue_mask.clone()
    normal = _target_logits(targets, corrupted, [2.0, 0.0, 0.0, 0.0, 0.0])
    shuffled = _target_logits(targets, corrupted, [0.0, 2.0, 2.0, 2.0, 2.0])
    result = contextual_stage_a_loss_v6(
        normal,
        shuffled,
        targets,
        corrupted,
        residue_mask,
        contrast_weight=1.0,
        contrast_margin_nats=0.05,
        paired_dropout_evidence=PairedDropoutEvidence(True, True, True),
    )
    old_token_weighted_hinge = F.relu(result["normal_minus_shuffled_batch_margin"] + 0.05)
    assert result["context_contrast"] > old_token_weighted_hinge
    assert result["context_active_hinge_sample_fraction"] == 0.5
    assert result["paired_dropout_verified"] == 1
    result["total"].backward()
    assert normal.grad is not None and normal.grad.abs().sum() > 0
    assert shuffled.grad is not None and shuffled.grad.abs().sum() > 0


def test_equal_protein_hinge_is_invariant_to_easy_sample_token_duplication() -> None:
    def loss_for(easy_length: int) -> torch.Tensor:
        targets = torch.zeros((2, easy_length), dtype=torch.long)
        targets[0, 0] = 2
        targets[1] = 3
        mask = targets.ne(0)
        normal = _target_logits(targets, mask, [2.0] + [0.0] * easy_length)
        shuffled = _target_logits(targets, mask, [0.0] + [2.0] * easy_length)
        return contextual_stage_a_loss_v6(
            normal,
            shuffled,
            targets,
            mask,
            mask,
            contrast_weight=1.0,
            contrast_margin_nats=0.05,
        )["context_contrast"]

    torch.testing.assert_close(loss_for(2), loss_for(8))


def test_v6_health_is_only_provisional_below_threshold_with_improving_ce() -> None:
    assert classify_context_health_v6(finite=True, shuffle_margin=-0.0049, ce_change_from_initial=-0.1) == "warning"
    assert (
        classify_context_health_v6(finite=True, shuffle_margin=-0.005, ce_change_from_initial=-0.1)
        == "provisional_healthy"
    )
    assert classify_context_health_v6(finite=True, shuffle_margin=-0.01, ce_change_from_initial=0.0) == "warning"
    assert classify_context_health_v6(finite=False, shuffle_margin=-1.0, ce_change_from_initial=-1.0) == "failing"


def test_v6_production_config_preserves_pause_warm_start_and_stage_b_block() -> None:
    config = load_yaml("configs/e006_rich_geometry_sequence_pretrain_v6.yaml")
    validate_phase3_config(config, mode="sequence-pretrain", synthetic=True)
    assert config["objective"]["version"] == "e006_stage_a_context_objective_v6"
    assert config["training"]["review_pause_steps"] == [2500]
    assert "v4_continuation" in config["initialization"]["checkpoint_path"]
    assert "v5" not in config["initialization"]["checkpoint_path"]
    assert config["post_training_context_diagnostic"]["report_sha256"] is None
    assert config["post_training_context_diagnostic"]["required_for_stage_b"] is True


def test_v6_smoke_plan_pins_completed_parity_report_and_both_objectives(tmp_path) -> None:
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({"status": "completed", "authorizes_training": False}))
    config = {
        "version": "e006_stage_a_context_v6_synthetic_smoke_v1",
        "output_dir": str(tmp_path / "output"),
        "seeds": [1],
        "optimizer_updates_per_seed": 1,
        "parity_audit": {
            "path": str(parity),
            "sha256": hashlib.sha256(parity.read_bytes()).hexdigest(),
        },
    }
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(config))
    plan = plan_synthetic_smoke_v6(path)
    assert plan["objective_arms"] == list(OBJECTIVE_ARMS)
    assert plan["authorizes_training"] is False


def test_v6_synthetic_smoke_compares_matched_non_authorizing_objectives(tmp_path) -> None:
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({"status": "completed", "authorizes_training": False}))
    config = {
        "version": "e006_stage_a_context_v6_synthetic_smoke_v1",
        "output_dir": str(tmp_path / "output"),
        "seeds": [5],
        "optimizer_updates_per_seed": 1,
        "learning_rate": 0.001,
        "mask_fraction": 0.3,
        "context_contrast_weight": 0.25,
        "context_contrast_margin_nats": 0.05,
        "minimum_unigram_improvement_nats": -100.0,
        "minimum_counterfactual_margin_nats": -100.0,
        "parity_audit": {
            "path": str(parity),
            "sha256": hashlib.sha256(parity.read_bytes()).hexdigest(),
        },
    }
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(config))
    report = run_synthetic_smoke_v6(path)
    assert report["status"] == "completed"
    assert {item["objective"] for item in report["runs"]} == set(OBJECTIVE_ARMS)
    assert report["gates"]["v6_paired_dropout_verified"] is True
    assert report["authorizes_training"] is False
    assert (tmp_path / "output" / "protocol.json").is_file()
