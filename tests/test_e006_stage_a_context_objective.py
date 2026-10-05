from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.sequence_geometry import SequenceGeometryVocabulary
from protein_distance_diffusion.models.rich_codesign import E006RichGeometryCoDesign
from protein_distance_diffusion.training.rich_codesign_production import (
    validate_phase3_config,
    verify_stage_a_v5_launch_gates,
)
from protein_distance_diffusion.training.stage_a_context import (
    canonical_corrupted_cross_entropy,
    context_corruption,
    contextual_stage_a_loss,
)


def _model() -> E006RichGeometryCoDesign:
    torch.manual_seed(17)
    return E006RichGeometryCoDesign(
        sequence_hidden_dim=32,
        sequence_layers=2,
        sequence_heads=4,
        sequence_feedforward_dim=64,
        sequence_dropout=0.0,
        max_length=16,
        rich_hidden_dim=16,
        fusion_layers=(0,),
        minimum_fusion_capacity_ratio=0.0,
        minimum_fusion_parameters=0,
        geometry_model={
            "base_channels": 8,
            "channel_multipliers": [1, 2],
            "residual_blocks_per_level": 1,
            "group_norm_groups": 4,
            "attention_heads": 1,
            "use_bottleneck_attention": False,
            "time_embedding_dim": 32,
            "length_embedding_dim": 32,
            "max_length": 16,
        },
    )


def test_v4_full_vocabulary_objective_does_not_match_canonical_diagnostic() -> None:
    logits = torch.zeros((1, 1, 22))
    logits[..., :2] = 10.0
    targets = torch.tensor([[2]])
    selected = torch.tensor([[True]])
    canonical = canonical_corrupted_cross_entropy(logits, targets, selected, selected)
    old_full_vocabulary = F.cross_entropy(logits[selected], targets[selected])
    assert canonical == pytest.approx(torch.tensor(math.log(20.0)))
    assert old_full_vocabulary > canonical + 5.0


def test_v5_production_config_pins_corrected_objective_and_new_output() -> None:
    config = load_yaml("configs/e006_rich_geometry_sequence_pretrain_v5.yaml")
    validate_phase3_config(config, mode="sequence-pretrain")
    assert config["objective"]["version"] == "e006_stage_a_context_objective_v5"
    assert config["objective"]["loss_positions"] == "corrupted_canonical_only"
    assert config["objective"]["corruption_state"] == "explicit_mask_token"
    assert config["training"]["output_dir"] == "outputs/e006_phase3_sequence_pretrain_v5"
    assert "continuation" not in config
    with pytest.raises(ValueError, match="launch gate is not pinned"):
        verify_stage_a_v5_launch_gates(config)


def test_v5_launch_gates_require_non_authorizing_contextual_evidence(tmp_path) -> None:
    config = load_yaml("configs/e006_rich_geometry_sequence_pretrain_v5.yaml")
    for name, record in config["gate_artifacts"].items():
        payload = {"status": "completed", "authorizes_training": False}
        if name == "context_diagnostic":
            payload["classification"] = "contextual_learning_verified"
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload))
        record["path"] = str(path)
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = verify_stage_a_v5_launch_gates(config)
    assert result["passed"] is True
    assert set(result["artifacts"]) == set(config["gate_artifacts"])
    context = config["gate_artifacts"]["context_diagnostic"]
    path = Path(context["path"])
    payload = json.loads(path.read_text())
    payload["classification"] = "marginal_frequency_only"
    path.write_text(json.dumps(payload))
    context["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="has not verified"):
        verify_stage_a_v5_launch_gates(config)


def test_zero_mask_identity_and_single_mask_contract() -> None:
    targets = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])
    residue_mask = targets.ne(0)
    identity = context_corruption(
        targets,
        residue_mask,
        mask_token_id=1,
        probability=0.0,
        seed=7,
        step=0,
        allow_zero_probability=True,
    )
    assert torch.equal(identity.inputs, targets)
    assert not identity.corrupted_mask.any()
    masked = context_corruption(
        targets,
        residue_mask,
        mask_token_id=1,
        probability=1e-9,
        seed=7,
        step=0,
    )
    assert masked.corrupted_mask.sum(dim=1).tolist() == [1, 1]
    assert torch.equal(masked.inputs[masked.corrupted_mask], torch.ones(2, dtype=torch.long))
    assert not masked.corrupted_mask[:, -1].any()


def test_token_round_trip_and_corrupted_only_target_alignment() -> None:
    vocabulary = SequenceGeometryVocabulary()
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    encoded = vocabulary.encode(sequence)
    assert vocabulary.decode(encoded) == sequence
    targets = torch.tensor([encoded])
    residue_mask = torch.ones_like(targets, dtype=torch.bool)
    selected = torch.zeros_like(residue_mask)
    selected[:, 3] = True
    logits = torch.zeros((1, len(encoded), 22), requires_grad=True)
    loss = canonical_corrupted_cross_entropy(logits, targets, selected, residue_mask)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 3].abs().sum() > 0
    assert logits.grad[0, torch.arange(len(encoded)) != 3].abs().sum() == 0


def test_context_contrast_penalizes_composition_only_logits() -> None:
    targets = torch.tensor([[2, 3, 4]])
    mask = torch.tensor([[False, True, False]])
    logits = torch.zeros((1, 3, 22), requires_grad=True)
    result = contextual_stage_a_loss(
        logits,
        logits.detach().clone(),
        targets,
        mask,
        torch.ones_like(mask),
        contrast_weight=0.5,
        contrast_margin_nats=0.1,
    )
    assert float(result["context_gap"].detach()) == pytest.approx(0.0)
    assert float(result["context_contrast"].detach()) == pytest.approx(0.1)
    assert result["total"] > result["sequence"]


def test_order_sensitivity_cross_position_influence_and_padding_invariance() -> None:
    model = _model().eval()
    tokens = torch.tensor([[2, 3, 1, 5]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    normal = model.forward_sequence_pretraining(tokens, mask)
    changed = tokens.clone()
    changed[:, 1] = 9
    perturbed = model.forward_sequence_pretraining(changed, mask)
    assert not torch.equal(normal[:, 2], perturbed[:, 2])
    reversed_logits = model.forward_sequence_pretraining(tokens.flip(1), mask)
    assert not torch.equal(normal, reversed_logits.flip(1))

    padded_tokens = F.pad(tokens, (0, 3))
    padded_mask = F.pad(mask, (0, 3))
    padded = model.forward_sequence_pretraining(padded_tokens, padded_mask)
    torch.testing.assert_close(normal, padded[:, : tokens.shape[1]], atol=1e-5, rtol=1e-5)


def test_context_toy_language_overfits_below_unigram() -> None:
    model = _model().train()
    targets = torch.tensor([[2, 2, 6, 3], [3, 3, 7, 2], [4, 4, 8, 5], [5, 5, 9, 4]])
    residue_mask = torch.ones_like(targets, dtype=torch.bool)
    corrupted = torch.zeros_like(residue_mask)
    corrupted[:, 2] = True
    inputs = targets.clone()
    inputs[corrupted] = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial = None
    final = None
    for _ in range(120):
        logits = model.forward_sequence_pretraining(inputs, residue_mask)
        loss = canonical_corrupted_cross_entropy(logits, targets, corrupted, residue_mask)
        initial = float(loss.detach()) if initial is None else initial
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = float(loss.detach())
    assert final is not None and initial is not None
    assert final < 0.2
    assert final < math.log(4.0) - 1.0
    assert final < initial * 0.1
