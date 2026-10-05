from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.rich_checkpoint_policy import storage_preflight
from protein_distance_diffusion.training.rich_codesign_production import run_training_stage, validate_phase3_config
from protein_distance_diffusion.training.stage_a_context_production import (
    MARGIN_SIGN_CONVENTION,
    append_rolling_training_metric,
    classify_context_health,
    contextual_monitoring_steps,
    contextual_selection_key,
    evaluate_context_monitor,
    format_trajectory,
    load_warm_start_weights_only,
    select_monitoring_panel,
    verify_review_decision,
    verify_v5_pretraining_gates,
)
from protein_distance_diffusion.training.stage_a_context_smoke import _tiny_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate_config(tmp_path: Path, *, recommendation: str = "v4_warm_start") -> dict:
    records = {}
    for name in ("synthetic_context_smoke", "real_loader_smoke", "comparison_pilot"):
        payload = {
            "status": "completed",
            "authorizes_training": False,
            "authorizes_joint_training": False,
        }
        if name == "comparison_pilot":
            payload["recommendation"] = {"recommendation": recommendation}
        else:
            payload["gates"] = {"passed": True}
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload))
        records[name] = {"path": str(path), "sha256": _sha256(path), "acceptable_statuses": ["completed"]}
    return {"gate_artifacts": records}


def test_v5_gate_hash_and_pilot_recommendation_are_strict(tmp_path) -> None:
    config = _gate_config(tmp_path)
    result = verify_v5_pretraining_gates(config)
    assert result["passed"] is True
    assert result["post_training_context_diagnostic_required"] is True
    config["gate_artifacts"]["synthetic_context_smoke"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash contradiction"):
        verify_v5_pretraining_gates(config)


def test_failed_pretraining_gate_does_not_create_output(tmp_path) -> None:
    config = load_yaml("configs/e006_rich_geometry_sequence_pretrain_v5_warm_start.yaml")
    output = tmp_path / "must-not-exist"
    config["training"]["output_dir"] = str(output)
    config["gate_artifacts"]["comparison_pilot"]["sha256"] = "0" * 64
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="hash contradiction"):
        run_training_stage(path, mode="sequence-pretrain")
    assert not output.exists()
    config = _gate_config(tmp_path, recommendation="scratch")
    with pytest.raises(ValueError, match="did not recommend"):
        verify_v5_pretraining_gates(config)


def test_v5_weights_only_initialization_restores_no_training_state(tmp_path) -> None:
    source = _tiny_model(5)
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "optimizer": {"forbidden": True},
            "scheduler": {"forbidden": True},
            "scaler": {"forbidden": True},
            "rng_state": {"forbidden": True},
            "optimizer_step": 35_000,
        },
        checkpoint,
    )
    target = _tiny_model(7)
    evidence = load_warm_start_weights_only(
        target,
        {
            "mode": "checkpoint_weights_only",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
        },
    )
    assert all(torch.equal(target.state_dict()[name], value) for name, value in source.state_dict().items())
    assert evidence["source_optimizer_step_ignored"] == 35_000
    assert all(
        evidence[name] is False
        for name in (
            "optimizer_restored",
            "scheduler_restored",
            "scaler_restored",
            "rng_restored",
            "cursor_restored",
            "training_progress_restored",
        )
    )


class _SequenceDataset:
    split = "validation"

    def __init__(self, count: int = 40) -> None:
        self.rows = [
            {
                "sample_id": f"sample-{index:03d}",
                "split": "validation",
                "sequence": "ACDEFGHIK",
                "token_ids": list(range(2, 11)),
            }
            for index in range(count)
        ]

    def iter_metadata(self):
        for index, row in enumerate(self.rows):
            yield index, row["sample_id"], len(row["sequence"])

    def __getitem__(self, index):
        return dict(self.rows[index])


def test_monitor_panel_is_deterministic_disjoint_and_bounded() -> None:
    dataset = _SequenceDataset()
    excluded = {"sample-000", "sample-001", "sample-002"}
    first, first_identity = select_monitoring_panel(
        dataset,
        count=16,
        seed=6206,
        maximum_length=128,
        excluded_sample_ids=excluded,
    )
    second, second_identity = select_monitoring_panel(
        dataset,
        count=16,
        seed=6206,
        maximum_length=128,
        excluded_sample_ids=excluded,
    )
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert first_identity == second_identity
    assert excluded.isdisjoint(first_identity["sample_ids"])
    assert first_identity["constructs_rich_pair_features"] is False


def test_context_monitor_is_deterministic_and_does_not_advance_optimizer() -> None:
    model = _tiny_model(13)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rows = _SequenceDataset(8).rows
    unigram = {"probabilities": [0.05] * 20}
    state_before = optimizer.state_dict()
    first = evaluate_context_monitor(
        model,
        rows,
        step=250,
        seed=6206,
        mask_fraction=0.3,
        unigram=unigram,
        device=torch.device("cpu"),
    )
    second = evaluate_context_monitor(
        model,
        rows,
        step=250,
        seed=6206,
        mask_fraction=0.3,
        unigram=unigram,
        device=torch.device("cpu"),
    )
    assert first == second
    assert optimizer.state_dict() == state_before
    assert first["normal_minus_shuffled_ce"] == pytest.approx(
        first["conditions"]["normal"]["canonical_cross_entropy"]
        - first["conditions"]["shuffled"]["canonical_cross_entropy"]
    )
    assert first["margin_sign_convention"] == MARGIN_SIGN_CONVENTION
    assert first["constructs_rich_pair_features"] is False


def test_health_uses_context_unigram_and_grace_period() -> None:
    assert (
        classify_context_health(
            step=500,
            finite=True,
            unigram_improvement=-0.1,
            shuffle_margin=0.1,
            null_margin=0.1,
        )
        == "warning"
    )
    assert (
        classify_context_health(
            step=2500,
            finite=True,
            unigram_improvement=-0.1,
            shuffle_margin=0.1,
            null_margin=0.1,
        )
        == "failing"
    )
    healthy = {
        "health": "healthy",
        "normal_minus_shuffled_ce": -0.2,
        "normal_minus_null_ce": -0.3,
        "conditions": {"normal": {"canonical_cross_entropy": 2.0}},
    }
    warning = {
        "health": "warning",
        "normal_minus_shuffled_ce": -1.0,
        "normal_minus_null_ce": -1.0,
        "conditions": {"normal": {"canonical_cross_entropy": 1.0}},
    }
    assert contextual_selection_key(healthy) < contextual_selection_key(warning)


def test_review_pause_schedule_storage_dedup_and_decision(tmp_path) -> None:
    schedule = contextual_monitoring_steps(6000, [2500, 5000, 6000])
    assert schedule[:4] == [0, 250, 500, 1000]
    assert 2500 in schedule and schedule[-1] == 6000
    storage = storage_preflight(
        lengths=[128] * 64,
        regimes=[
            {"maximum_length": maximum_length, "physical_batch_size": 32} for maximum_length in (128, 256, 384, 500)
        ],
        dataset_passes=1,
        validation_frequency=2,
        recovery_frequency=1,
        estimated_checkpoint_bytes=100,
        output_directory=tmp_path,
        minimum_free_disk_gib=0,
        review_pause_steps=[2],
        free_disk_bytes=10**9,
    )
    assert storage["scientific_review_pause_steps"] == [2]
    assert storage["immutable_checkpoint_steps"].count(2) == 1
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"recovery")
    decision = {
        "status": "completed",
        "decision": "approve_continue",
        "config_sha256": "c" * 64,
        "review_checkpoint_sha256": _sha256(checkpoint),
        "optimizer_step": 2500,
        "authorizes_training": False,
        "authorizes_joint_training": False,
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(decision))
    assert (
        verify_review_decision(
            path,
            _sha256(path),
            config_sha256="c" * 64,
            checkpoint_sha256=_sha256(checkpoint),
            optimizer_step=2500,
        )
        == decision
    )


def test_trajectory_table_and_production_config_contract(tmp_path) -> None:
    record = {
        "record_type": "contextual_monitoring",
        "optimizer_step": 250,
        "conditions": {
            "normal": {"canonical_cross_entropy": 2.0, "top1_accuracy": 0.2},
            "shuffled": {"canonical_cross_entropy": 2.2},
            "null": {"canonical_cross_entropy": 2.3},
        },
        "normal_minus_shuffled_ce": -0.2,
        "normal_minus_null_ce": -0.3,
        "improvement_over_training_unigram": 0.1,
        "health": "healthy",
        "learning_rate": 1e-4,
        "amp_overflows_total": 2,
    }
    path = tmp_path / "context.jsonl"
    path.write_text(json.dumps(record) + "\n")
    table = format_trajectory(path)
    assert "shuffle_margin" in table and "250 2.000000" in table

    config = load_yaml("configs/e006_rich_geometry_sequence_pretrain_v5_warm_start.yaml")
    comparison_base = load_yaml("configs/e006_rich_geometry_sequence_pretrain_v5.yaml")
    validate_phase3_config(config, mode="sequence-pretrain", synthetic=False)
    for section in (
        "dataset",
        "model",
        "objective",
        "optimizer",
        "mixed_precision",
        "memory",
        "calibration",
        "batching",
    ):
        assert config[section] == comparison_base[section]
    for name in (
        "dataset_passes",
        "maximum_length",
        "bounded_panel_size",
        "maximum_optimizer_updates",
        "recovery_checkpoint_frequency",
        "validation_frequency",
        "immutable_checkpoint_on_validation",
        "immutable_checkpoint_on_pass_end",
        "maintain_best_checkpoint",
        "estimated_checkpoint_bytes",
        "minimum_free_disk_gib",
        "maximum_total_amp_overflows",
        "maximum_consecutive_amp_overflows",
        "maximum_overflow_diagnostic_examples",
        "progress_units",
    ):
        assert config["training"][name] == comparison_base["training"][name]
    assert set(config["gate_artifacts"]) == {
        "synthetic_context_smoke",
        "real_loader_smoke",
        "comparison_pilot",
    }
    assert config["training"]["review_pause_steps"] == [2500]
    assert config["post_training_context_diagnostic"]["required_for_stage_b"] is True
    assert config["post_training_context_diagnostic"]["required_for_definitive_evaluation"] is True
    assert _sha256(Path("configs/e006_rich_geometry_sequence_pretrain_v5.yaml")) == (
        "9b37c4a0278630ae1aece35ae3995f7a5b8f8d6df38ca1d2732f5e7a0da80abe"
    )


def test_rolling_metric_publication_is_compact_and_durable(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    record = {
        "record_type": "training_update",
        "optimizer_step": 50,
        "canonical_cross_entropy": 2.5,
        "context_contrast": 0.1,
        "weighted_context_contrast": 0.025,
        "total_loss": 2.525,
        "normal_minus_shuffled_batch_margin": -0.05,
        "gradient_norm": 1.2,
        "learning_rate": 1e-4,
        "amp_scale_after": 16384,
        "amp_overflows_total": 0,
    }
    append_rolling_training_metric(path, record)
    assert json.loads(path.read_text()) == record
    with pytest.raises(ValueError, match="forbidden"):
        append_rolling_training_metric(path, {**record, "logits": [1.0]})
