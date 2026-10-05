from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from protein_distance_diffusion.data.rich_geometry import RichDatasetAuthorization
from protein_distance_diffusion.evaluation.e006_stage_a_context import REQUIRED_SEQUENCE_COLUMNS
from protein_distance_diffusion.training.stage_a_context import (
    canonical_corrupted_cross_entropy,
    contextual_stage_a_loss,
    visible_shuffle_inputs,
)
from protein_distance_diffusion.training.stage_a_context_smoke import (
    SEQUENCE_PARAMETER_PREFIXES,
    _bounded_rows,
    _comparison_arm_worker,
    _condition_metrics,
    _evaluate_comparison_panel,
    _padding_error,
    _paired_control_checks,
    _recommend_arm,
    _tiny_model,
    comparison_protocol,
    contextual_toy,
    initialize_comparison_arm,
    plan_loader_smoke,
    run_comparison_pilot,
    toy_contract,
    verify_smoke_prerequisites,
)


def test_contextual_toy_has_equal_marginals_disjoint_contexts_and_ambiguous_bags() -> None:
    contract = toy_contract()
    assert contract["marginals_equal"] is True
    assert contract["train_held_out_contexts_disjoint"] is True
    assert contract["bag_ambiguous_context_count"] == 6
    assert contract["target_leakage_absent"] is True
    assert contract["train_target_counts"] == contract["held_out_target_counts"]


def test_contextual_toy_generalizes_and_shuffle_destroys_ordered_signal() -> None:
    train = contextual_toy("train")
    held_out = contextual_toy("held_out")
    model = _tiny_model(41).train()
    sequence_parameters = [
        parameter for name, parameter in model.named_parameters() if name.startswith(SEQUENCE_PARAMETER_PREFIXES)
    ]
    optimizer = torch.optim.Adam(sequence_parameters, lr=0.02)
    for step in range(100):
        shuffled = visible_shuffle_inputs(
            train["inputs"], train["corrupted_mask"], train["residue_mask"], seed=41, step=step
        )
        normal_logits = model.forward_sequence_pretraining(train["inputs"], train["residue_mask"])
        with torch.no_grad():
            shuffled_logits = model.forward_sequence_pretraining(shuffled, train["residue_mask"])
        losses = contextual_stage_a_loss(
            normal_logits,
            shuffled_logits,
            train["targets"],
            train["corrupted_mask"],
            train["residue_mask"],
            contrast_weight=0.25,
            contrast_margin_nats=0.1,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
    metrics = _condition_metrics(model.eval(), held_out, shuffle_seed=99)
    assert metrics["normal"]["cross_entropy"] < metrics["unigram"]["cross_entropy"] - 0.5
    assert metrics["normal_to_shuffled_margin"] > 0.1
    assert metrics["normal_to_null_margin"] > 0.1
    assert metrics["normal"]["top1_accuracy"] > 0.9


def test_toy_leakage_rejection_margin_sign_and_corrupted_only_loss() -> None:
    batch = contextual_toy("train")
    assert torch.equal(
        batch["inputs"][batch["corrupted_mask"]],
        torch.ones(int(batch["corrupted_mask"].sum()), dtype=torch.long),
    )
    logits = torch.zeros((*batch["targets"].shape, 22), requires_grad=True)
    normal = canonical_corrupted_cross_entropy(logits, batch["targets"], batch["corrupted_mask"], batch["residue_mask"])
    result = contextual_stage_a_loss(
        logits,
        logits.detach(),
        batch["targets"],
        batch["corrupted_mask"],
        batch["residue_mask"],
        contrast_weight=0.25,
        contrast_margin_nats=0.1,
    )
    assert float(result["context_gap"].detach()) == pytest.approx(0.0)
    assert result["context_contrast"] > 0
    normal.backward()
    assert logits.grad is not None
    assert logits.grad[~batch["corrupted_mask"]].abs().sum() == 0
    leaked = batch["inputs"].clone()
    leaked[batch["corrupted_mask"]] = batch["targets"][batch["corrupted_mask"]]
    assert not torch.equal(
        leaked[batch["corrupted_mask"]],
        torch.ones_like(leaked[batch["corrupted_mask"]]),
    )


def test_contextual_toy_padding_invariance() -> None:
    model = _tiny_model(7).eval()
    assert _padding_error(model, contextual_toy("held_out")) <= 1e-5


def test_bounded_loader_selection_is_deterministic_and_sequence_only() -> None:
    class Dataset:
        split = "validation"

        def __init__(self) -> None:
            self.rows = [
                {
                    "sample_id": f"sample-{index}",
                    "sequence": "ACDE" * (index + 1),
                    "token_ids": [2, 3, 4, 5] * (index + 1),
                    "split": "validation",
                }
                for index in range(8)
            ]

        def iter_metadata(self):
            for index, row in enumerate(self.rows):
                yield index, row["sample_id"], len(row["sequence"])

        def __getitem__(self, index):
            return dict(self.rows[index])

    dataset = Dataset()
    first = _bounded_rows(dataset, count=4, seed=9, maximum_length=24)
    second = _bounded_rows(dataset, count=4, seed=9, maximum_length=24)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert all(set(row) == {"sample_id", "sequence", "token_ids", "split"} for row in first)
    assert all(len(row["sequence"]) == len(row["token_ids"]) <= 24 for row in first)


def test_loader_plan_projects_sequence_schema_and_preserves_protected_inputs(tmp_path, monkeypatch) -> None:
    import protein_distance_diffusion.training.stage_a_context_smoke as smoke

    class Dataset:
        columns = REQUIRED_SEQUENCE_COLUMNS

        def __init__(self, _authorization, *, split):
            self.split = split
            self.rows = [
                {
                    "sample_id": f"{split}-{index}",
                    "sequence": "ACDE",
                    "token_ids": [2, 3, 4, 5],
                    "split": split,
                }
                for index in range(4)
            ]

        def iter_metadata(self):
            for index, row in enumerate(self.rows):
                yield index, row["sample_id"], 4

        def __getitem__(self, index):
            return dict(self.rows[index])

    authorization = RichDatasetAuthorization(
        root=tmp_path,
        protocol_sha256="1" * 64,
        schema_sha256="2" * 64,
        vocabulary_sha256="3" * 64,
        normalization_sha256="4" * 64,
        shard_inventory_sha256="5" * 64,
        split_counts={"train": 4, "validation": 4},
        observed_shard_hashes={"train/part.parquet": "6" * 64},
    )
    monkeypatch.setattr(smoke, "_authorization", lambda _config: authorization)
    monkeypatch.setattr(smoke, "SequenceOnlyRichDataset", Dataset)
    base = Path("configs/e006_rich_geometry_sequence_pretrain_v5.yaml")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"immutable-checkpoint")
    config = {
        "version": "e006_stage_a_context_v5_loader_smoke_v1",
        "output_dir": str(tmp_path / "output"),
        "seed": 7,
        "base_training_config": str(base),
        "base_training_config_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "train_sample_count": 2,
        "validation_sample_count": 2,
        "maximum_length": 128,
    }
    config_path = tmp_path / "loader.yaml"
    config_path.write_text(yaml.safe_dump(config))
    before = (hashlib.sha256(base.read_bytes()).hexdigest(), hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    plan = plan_loader_smoke(config_path)
    after = (hashlib.sha256(base.read_bytes()).hexdigest(), hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    assert before == after
    assert plan["projected_columns"] == list(REQUIRED_SEQUENCE_COLUMNS)
    assert plan["constructs_rich_pair_features"] is False
    assert set(plan["train_sample_ids"]).isdisjoint(plan["validation_sample_ids"])


def test_comparison_prerequisites_require_hashes_and_passed_non_authorizing_reports(tmp_path) -> None:
    specification = {"prerequisites": {"synthetic_smoke": {}, "real_loader_smoke": {}}}
    with pytest.raises(ValueError, match="not pinned"):
        verify_smoke_prerequisites(specification)
    for name in specification["prerequisites"]:
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "gates": {"passed": True},
                    "authorizes_training": False,
                    "authorizes_joint_training": False,
                }
            )
        )
        specification["prerequisites"][name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    assert set(verify_smoke_prerequisites(specification)) == {"synthetic_smoke", "real_loader_smoke"}
    failed_path = tmp_path / "real_loader_smoke.json"
    failed_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "gates": {"passed": False},
                "authorizes_training": False,
                "authorizes_joint_training": False,
            }
        )
    )
    specification["prerequisites"]["real_loader_smoke"]["sha256"] = hashlib.sha256(failed_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="did not pass"):
        verify_smoke_prerequisites(specification)


def test_comparison_warm_start_loads_model_weights_only(tmp_path) -> None:
    source = _tiny_model(11)
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "optimizer": {"must": "not load"},
            "scheduler": {"must": "not load"},
            "scaler": {"must": "not load"},
            "rng_state": {"must": "not load"},
            "optimizer_step": 35_000,
        },
        checkpoint,
    )
    target = _tiny_model(19)
    result = initialize_comparison_arm(
        target,
        {
            "initialization": "checkpoint_weights_only",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
    )
    assert all(torch.equal(target.state_dict()[name], value) for name, value in source.state_dict().items())
    assert result["model_weights_loaded"] is True
    assert result["source_optimizer_step_ignored"] == 35_000
    assert all(
        result[name] is False
        for name in (
            "optimizer_restored",
            "scheduler_restored",
            "scaler_restored",
            "rng_restored",
            "cursor_restored",
            "training_progress_restored",
        )
    )


def _comparison_rows() -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"validation-{index}",
            "split": "validation",
            "sequence": "ACDEFGHIK",
            "token_ids": list(range(2, 11)),
        }
        for index in range(4)
    ]


def test_comparison_evaluation_is_deterministic_paired_and_sequence_only(monkeypatch) -> None:
    model = _tiny_model(23).eval()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("comparison evaluation constructed geometry features")

    monkeypatch.setattr(model, "forward", forbidden)
    kwargs = {
        "batch_size": 2,
        "seed": 91,
        "mask_probability": 0.3,
        "contrast_weight": 0.25,
        "contrast_margin_nats": 0.05,
        "device": torch.device("cpu"),
    }
    first = _evaluate_comparison_panel(model, _comparison_rows(), **kwargs)
    second = _evaluate_comparison_panel(model, list(reversed(list(reversed(_comparison_rows())))), **kwargs)
    assert first == second
    conditions = first["conditions"]
    assert conditions["normal_minus_shuffled_ce"] == pytest.approx(
        conditions["normal"]["canonical_cross_entropy"] - conditions["shuffled"]["canonical_cross_entropy"]
    )
    assert "negative favors ordered context" in conditions["margin_sign_convention"]
    assert first["objective_components"]["canonical_corrupted_position_ce"] == pytest.approx(
        conditions["normal"]["canonical_cross_entropy"]
    )


def _arm_result(name: str, process_id: int, normal_ce: float, margin: float, accuracy: float) -> dict:
    evaluation = {
        "conditions": {
            "normal": {"canonical_cross_entropy": normal_ce, "top1_accuracy": accuracy},
            "normal_minus_shuffled_ce": margin,
            "normal_minus_null_ce": margin - 0.1,
        },
        "target_sha256": "targets",
        "corruption_mask_sha256": "masks",
    }
    return {
        "arm": name,
        "process_id": process_id,
        "panel_identities": {"train": "same", "validation": "same"},
        "successful_optimizer_updates": 250,
        "training_target_sha256": "train-targets",
        "training_corruption_mask_sha256": "train-masks",
        "final_evaluation": evaluation,
        "constructs_rich_pair_features": False,
        "frozen_geometry_and_fusion_parameters_unchanged": True,
        "process_isolation": "spawned_dedicated_child",
        "finite_losses_and_gradients": True,
    }


def test_comparison_pairing_arm_isolation_recommendation_and_non_authorization(tmp_path) -> None:
    scratch = _arm_result("scratch", 101, 2.8, -0.05, 0.2)
    warm = _arm_result("v4_warm_start", 202, 2.7, -0.10, 0.3)
    checks = _paired_control_checks([scratch, warm], 250)
    assert all(checks.values())
    recommendation = _recommend_arm([scratch, warm])
    assert recommendation["recommendation"] == "v4_warm_start"
    assert recommendation["raw_ce_alone_used"] is False
    protocol = comparison_protocol(tmp_path / "report.json", "a" * 64, "v4_warm_start")
    assert protocol["authorizes_training"] is False
    assert protocol["authorizes_joint_training"] is False


def test_comparison_pilot_refuses_existing_output_before_workers(tmp_path, monkeypatch) -> None:
    import protein_distance_diffusion.training.stage_a_context_smoke as smoke

    output = tmp_path / "existing"
    output.mkdir()
    config = tmp_path / "comparison.yaml"
    config.write_text(yaml.safe_dump({"output_dir": str(output)}))
    monkeypatch.setattr(smoke, "comparison_preflight", lambda _path: pytest.fail("preflight must not run"))
    monkeypatch.setattr(smoke, "_comparison_arm_worker", lambda *_args: pytest.fail("worker must not run"))
    with pytest.raises(FileExistsError, match="already exists"):
        run_comparison_pilot(config)


def test_comparison_worker_is_module_level_for_spawn_isolation() -> None:
    assert _comparison_arm_worker.__module__ == "protein_distance_diffusion.training.stage_a_context_smoke"
