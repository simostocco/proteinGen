from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.models.rich_codesign import E006LossWeights
from protein_distance_diffusion.training.checkpointing import load_checkpoint, save_checkpoint
from protein_distance_diffusion.training.rich_codesign_production import (
    CALIBRATION_VERSION,
    CONTINUATION_PROTOCOL_VERSION,
    STAGE_A_CHECKPOINT_VERSION,
    BatchRecommendation,
    E006GradientError,
    TrainingInterrupted,
    _amp_context,
    _assert_finite_model,
    _expected_active_parameter_names,
    _gradient_evidence,
    _LazySelectedRows,
    _model,
    _optimizer_boundary_update,
    _require_finite_training_loss,
    _scheduler_multiplier,
    _stage_trainable,
    _validation_panel_preflight,
    build_validation_panel,
    execute_calibration_regime,
    plan_phase3,
    recommendation_for_length,
    run_training_stage,
    select_best_checkpoint,
    select_calibration_recommendations,
    validate_batch_budget,
    validate_phase3_config,
    validate_resume_checkpoint,
    verify_calibration_report,
    verify_continuation_source,
    verify_stage_a_checkpoint,
)


class _StrictScalarDataset:
    def __init__(self, sample_ids: list[str]) -> None:
        self.rows = [{"sample_id": sample_id} for sample_id in sample_ids]
        self.requested_indices: list[int] = []

    def __getitem__(self, index: int) -> dict:
        assert isinstance(index, int) and not isinstance(index, bool)
        self.requested_indices.append(index)
        return self.rows[index]


class _FakeGradScaler:
    def __init__(self, *, scale: float = 8.0, overflow: bool = False) -> None:
        self.scale_value = scale
        self.overflow = overflow
        self.optimizer_steps = 0
        self.gradients_at_step: list[torch.Tensor] = []

    def get_scale(self) -> float:
        return self.scale_value

    def unscale_(self, _optimizer) -> None:
        return None

    def step(self, optimizer) -> None:
        if not self.overflow:
            self.gradients_at_step = [
                parameter.grad.detach().clone()
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            optimizer.step()
            self.optimizer_steps += 1

    def update(self) -> None:
        if self.overflow:
            self.scale_value /= 2


def _stage_model(tmp_path: Path, stage: str = "sequence-pretrain"):
    config, _ = _config(tmp_path)
    model = _model(config)
    _stage_trainable(model, stage)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    return model, optimizer


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _continuation_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    source, source_config_path = _config(tmp_path, "migration-source")
    source["synthetic"] = {"lengths": [8, 10, 12]}
    source["training"].update(
        bounded_panel_size=3,
        maximum_optimizer_updates=3,
        interrupt_after_optimizer_steps=1,
    )
    source_config_path.write_text(json.dumps(source))
    with pytest.raises(TrainingInterrupted):
        run_training_stage(source_config_path, mode="sequence-pretrain", synthetic=True)
    source["training"].pop("interrupt_after_optimizer_steps")
    source_config_path.write_text(json.dumps(source))
    source_output = Path(source["training"]["output_dir"])
    checkpoint = source_output / "checkpoints" / "latest.pt"
    latest_metadata = source_output / "checkpoints" / "latest.json"
    manifest = source_output / "checkpoint_manifest.json"
    validations = source_output / "validation_journal.jsonl"
    best = source_output / "checkpoints" / "best.pt"
    best_metadata = source_output / "checkpoints" / "best.json"
    payload = load_checkpoint(checkpoint)
    best_record = json.loads(best_metadata.read_text())
    validation_steps = [
        json.loads(line)["optimizer_step"] for line in validations.read_text().splitlines() if line.strip()
    ]
    destination = copy.deepcopy(source)
    destination["experiment"] = "synthetic-audited-continuation"
    destination["training"]["output_dir"] = str(tmp_path / "migration-destination")
    destination["continuation"] = {
        "mode": CONTINUATION_PROTOCOL_VERSION,
        "source_output_dir": str(source_output),
        "source_checkpoint_path": str(checkpoint),
        "source_checkpoint_sha256": _file_sha256(checkpoint),
        "source_checkpoint_metadata_path": str(latest_metadata),
        "source_checkpoint_metadata_sha256": _file_sha256(latest_metadata),
        "source_checkpoint_manifest_path": str(manifest),
        "source_checkpoint_manifest_sha256": _file_sha256(manifest),
        "source_validation_journal_path": str(validations),
        "source_validation_journal_sha256": _file_sha256(validations),
        "source_best_checkpoint_path": str(best),
        "source_best_checkpoint_sha256": _file_sha256(best),
        "source_best_metadata_path": str(best_metadata),
        "source_best_metadata_sha256": _file_sha256(best_metadata),
        "source_config_path": str(source_config_path),
        "source_config_file_sha256": _file_sha256(source_config_path),
        "source_optimizer_step": payload["optimizer_step"],
        "source_microstep": payload["microstep"],
        "source_data_cursor": payload["data_cursor"],
        "source_dataset_pass": payload["dataset_pass"],
        "source_processed_valid_tokens": payload["processed_valid_tokens"],
        "source_best_sequence_cross_entropy": best_record["validation_sequence_cross_entropy"],
        "source_best_optimizer_step": best_record["optimizer_step"],
        "source_validation_steps": validation_steps,
        "next_scheduled_validation_step": 2,
        "dataset_pass_end_steps": [3],
        "total_target_optimizer_steps": 3,
        "discarded_uncheckpointed_updates": 0,
    }
    destination_path = tmp_path / "continuation.json"
    destination_path.write_text(json.dumps(destination))
    return destination, destination_path, source_output


def _populate_active_gradients(model, stage: str, *, nonfinite: str | None = None) -> None:
    active = _expected_active_parameter_names(model, stage)
    for name, parameter in model.named_parameters():
        if name in active:
            parameter.grad = torch.ones_like(parameter)
    if nonfinite is not None:
        dict(model.named_parameters())[nonfinite].grad.flatten()[0] = float("inf")


def _assert_nested_equal(first, second) -> None:
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            _assert_nested_equal(left, right)
    else:
        assert first == second


def _config(tmp_path: Path, name: str = "stage-a") -> tuple[dict, Path]:
    config = load_yaml("configs/e006_rich_geometry_sequence_pretrain.yaml")
    config["device"] = "cpu"
    config["mixed_precision"] = {"enabled": False, "dtype": "float16"}
    config["model"].update(
        sequence_hidden_dim=24,
        sequence_layers=2,
        sequence_heads=4,
        sequence_feedforward_dim=48,
        sequence_dropout=0.0,
        max_length=16,
        rich_hidden_dim=24,
        fusion_layers=[0, 1],
        minimum_fusion_capacity_ratio=0.01,
        minimum_fusion_parameters=100,
        geometry_model={
            "base_channels": 4,
            "channel_multipliers": [1, 2],
            "residual_blocks_per_level": 1,
            "group_norm_groups": 2,
            "attention_heads": 1,
            "use_bottleneck_attention": False,
            "time_embedding_dim": 16,
            "length_embedding_dim": 16,
            "max_length": 16,
        },
    )
    config["optimizer"].update(warmup_updates=0, learning_rate=0.01)
    config["batching"] = {"maximum_residues": 32, "maximum_pair_elements": 512 * 512}
    config["training"].update(
        output_dir=str(tmp_path / name),
        dataset_passes=1,
        bounded_panel_size=2,
        maximum_optimizer_updates=2,
        recovery_checkpoint_frequency=1,
        validation_frequency=1,
        immutable_checkpoint_on_validation=True,
        immutable_checkpoint_on_pass_end=True,
        maintain_best_checkpoint=True,
        estimated_checkpoint_bytes=1024,
        minimum_free_disk_gib=25,
    )
    config["validation"]["panel_size"] = 2
    config["synthetic"] = {"lengths": [8, 12]}
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(config))
    return config, path


def test_smoke_checkpoint_is_not_stage_a_authorization(tmp_path: Path) -> None:
    path = tmp_path / "smoke.pt"
    save_checkpoint(path, {"version": "e006_phase2_smoke_v1", "status": "completed"})
    with pytest.raises(ValueError, match="requires checkpoints/best.pt"):
        verify_stage_a_checkpoint(
            path,
            __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            dataset_identity="dataset",
            calibration_sha256="calibration",
        )


def test_validation_panel_uses_scalar_access_through_tuple_backed_wrapper() -> None:
    dataset = _StrictScalarDataset(["zero", "one", "two"])
    rows = _LazySelectedRows(dataset, [2, 0, 1], [8, 9, 10], ["two", "zero", "one"])
    assert [row["sample_id"] for row in build_validation_panel(rows, 2)] == ["two", "zero"]
    assert dataset.requested_indices == [2, 0]
    assert [row["sample_id"] for row in build_validation_panel(rows, 10)] == ["two", "zero", "one"]
    assert [row["sample_id"] for row in build_validation_panel(rows, 2)] == ["two", "zero"]


@pytest.mark.parametrize("index", [(0,), [0], slice(0, 1), True])
def test_selected_rows_reject_collection_and_boolean_indices(index: object) -> None:
    rows = _LazySelectedRows(_StrictScalarDataset(["zero"]), [0], [8], ["zero"])
    with pytest.raises(TypeError, match="scalar integers"):
        rows[index]  # type: ignore[index]


def test_selected_rows_accept_python_and_numpy_scalar_indices() -> None:
    rows = _LazySelectedRows(_StrictScalarDataset(["zero"]), [0], [8], ["zero"])
    assert rows[0]["sample_id"] == "zero"
    assert rows[np.int64(0)]["sample_id"] == "zero"


def test_stage_a_plan_reports_scalar_unique_disjoint_o_n_panel(tmp_path: Path) -> None:
    config, _ = _config(tmp_path, "planned")
    report = plan_phase3(config, mode="sequence-pretrain", synthetic=True)
    panel = report["validation_panel_preflight"]
    assert report["status"] == "planned"
    assert report["output_directory_exists"] is False
    assert panel == {
        "requested_sample_count": 2,
        "sample_count": 2,
        "unique_sample_count": 2,
        "sample_id_sha256": panel["sample_id_sha256"],
        "scalar_indexed": True,
        "train_validation_disjoint": True,
        "constructs_rich_pair_features": False,
        "feature_complexity": "O(N)",
    }


def test_production_sized_panel_is_exact_unique_and_deterministic() -> None:
    train = _LazySelectedRows(_StrictScalarDataset(["train"]), [0], [8], ["train"])
    sample_ids = [f"validation-{index:04d}" for index in range(300)]
    validation = _LazySelectedRows(
        _StrictScalarDataset(sample_ids),
        list(range(300)),
        [8] * 300,
        sample_ids,
    )
    first, diagnostics = _validation_panel_preflight(train, validation, 256)
    second, repeated = _validation_panel_preflight(train, validation, 256)
    assert len(first) == len({row["sample_id"] for row in first}) == 256
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert diagnostics["sample_id_sha256"] == repeated["sample_id_sha256"]


def test_validation_panel_failure_precedes_optimizer_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, path = _config(tmp_path, "panel-failure")
    optimizer_step_called = False

    def forbidden_step(*_args, **_kwargs):
        nonlocal optimizer_step_called
        optimizer_step_called = True
        raise AssertionError("optimizer step ran before validation-panel initialization")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_step)
    monkeypatch.setattr(
        "protein_distance_diffusion.training.rich_codesign_production._validation_panel_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("synthetic validation-panel failure")),
    )
    with pytest.raises(ValueError, match="synthetic validation-panel failure"):
        run_training_stage(path, mode="sequence-pretrain", synthetic=True)
    assert optimizer_step_called is False


def test_gradient_failure_protocol_retains_live_training_position(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, path = _config(tmp_path, "gradient-failure")
    calls = 0

    def fail_loss(total, evidence):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert torch.isfinite(total)
            return
        raise E006GradientError("synthetic gradient failure", {**evidence, "offender": "weight"})

    monkeypatch.setattr(
        "protein_distance_diffusion.training.rich_codesign_production._require_finite_training_loss",
        fail_loss,
    )
    with pytest.raises(E006GradientError, match="synthetic gradient failure"):
        run_training_stage(path, mode="sequence-pretrain", synthetic=True)
    report = json.loads((tmp_path / "gradient-failure" / "protocol.json").read_text())
    assert report["optimizer_step"] == report["microstep"] == report["data_cursor"] == 1
    assert report["dataset_pass"] == 0
    assert report["processed_valid_tokens"] == 8
    assert report["amp_overflows_total"] == report["amp_overflows_consecutive"] == 0
    assert report["latest_batch_evidence"]["sample_ids"]
    assert report["failure_diagnostics"]["offender"] == "weight"
    assert report["timestamp_utc"]
    assert report["memory"]["current_rss_mib"] > 0


def test_joint_config_rejects_missing_authorized_stage_a(tmp_path: Path) -> None:
    config, _ = _config(tmp_path)
    config["stage_a"] = {"checkpoint_path": None, "checkpoint_sha256": None}
    with pytest.raises(ValueError, match="authorized Stage-A"):
        validate_phase3_config(config, mode="joint-train", synthetic=True)

    path = tmp_path / "missing-stage-a.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="authorized Stage-A"):
        run_training_stage(path, mode="joint-train", synthetic=True)
    failure = json.loads(Path(config["training"]["output_dir"], "protocol.json").read_text())
    assert failure["status"] == "failed"
    assert failure["authorizes_joint_training"] is False


def test_length_batch_and_pair_budgets() -> None:
    recommendation = BatchRecommendation(128, 2, 4, 2 * 128 * 128)
    metrics = validate_batch_budget(
        [100, 120],
        recommendation=recommendation,
        maximum_residues=256,
        maximum_pair_elements=2 * 128 * 128,
    )
    assert metrics["physical_batch_size"] == 2
    assert metrics["accumulation_count"] == 4
    assert recommendation_for_length(120, [recommendation.__dict__]) == recommendation
    with pytest.raises(MemoryError, match="length regime"):
        validate_batch_budget([129], recommendation=recommendation, maximum_residues=256, maximum_pair_elements=100_000)
    with pytest.raises(MemoryError, match="pair-element"):
        validate_batch_budget([120, 120], recommendation=recommendation, maximum_residues=256, maximum_pair_elements=1)


def test_gradient_accumulation_matches_combined_mean() -> None:
    first = torch.nn.Linear(3, 1, bias=False)
    second = copy.deepcopy(first)
    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]])
    y = torch.tensor([[1.0], [0.5]])
    ((first(x[:1]) - y[:1]).square().mean() / 2).backward()
    ((first(x[1:]) - y[1:]).square().mean() / 2).backward()
    (second(x) - y).square().mean().backward()
    assert torch.allclose(first.weight.grad, second.weight.grad)


def test_stage_a_inactive_parameters_may_have_no_gradient(tmp_path: Path) -> None:
    model, optimizer = _stage_model(tmp_path)
    _populate_active_gradients(model, "sequence-pretrain")
    evidence = _gradient_evidence(model, stage="sequence-pretrain")
    assert evidence["missing_active_gradient_count"] == 0
    assert any(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
    result = _optimizer_boundary_update(
        model=model,
        optimizer=optimizer,
        scaler=_FakeGradScaler(),
        amp_enabled=True,
        stage="sequence-pretrain",
        gradient_clip_norm=1.0,
        amp_overflows_total=0,
        amp_overflows_consecutive=0,
        maximum_total_amp_overflows=10,
        maximum_consecutive_amp_overflows=3,
    )
    assert result["update_skipped"] is False


def test_missing_active_gradient_is_named_and_fatal(tmp_path: Path) -> None:
    model, optimizer = _stage_model(tmp_path)
    _populate_active_gradients(model, "sequence-pretrain")
    model.sequence_output.weight.grad = None
    with pytest.raises(E006GradientError, match="missing gradients") as captured:
        _optimizer_boundary_update(
            model=model,
            optimizer=optimizer,
            scaler=_FakeGradScaler(),
            amp_enabled=True,
            stage="sequence-pretrain",
            gradient_clip_norm=1.0,
            amp_overflows_total=0,
            amp_overflows_consecutive=0,
            maximum_total_amp_overflows=10,
            maximum_consecutive_amp_overflows=3,
        )
    assert "sequence_output.weight" in captured.value.diagnostics["missing_active_gradient_names"]


def test_finite_amp_update_advances_optimizer(tmp_path: Path) -> None:
    model, optimizer = _stage_model(tmp_path)
    _populate_active_gradients(model, "sequence-pretrain")
    before = model.sequence_output.weight.detach().clone()
    scaler = _FakeGradScaler()
    result = _optimizer_boundary_update(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        amp_enabled=True,
        stage="sequence-pretrain",
        gradient_clip_norm=1.0,
        amp_overflows_total=0,
        amp_overflows_consecutive=0,
        maximum_total_amp_overflows=10,
        maximum_consecutive_amp_overflows=3,
    )
    assert result["update_skipped"] is False
    assert scaler.optimizer_steps == 1
    assert not torch.equal(before, model.sequence_output.weight)


def test_corrected_finite_update_is_identical_to_legacy_finite_behavior(tmp_path: Path) -> None:
    corrected, corrected_optimizer = _stage_model(tmp_path)
    legacy = copy.deepcopy(corrected)
    legacy_optimizer = torch.optim.AdamW(
        [parameter for parameter in legacy.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    corrected_scheduler = torch.optim.lr_scheduler.LambdaLR(corrected_optimizer, lambda _step: 1.0)
    legacy_scheduler = torch.optim.lr_scheduler.LambdaLR(legacy_optimizer, lambda _step: 1.0)
    corrected_loss = sum(parameter.sum() for parameter in corrected.parameters() if parameter.requires_grad)
    legacy_loss = sum(parameter.sum() for parameter in legacy.parameters() if parameter.requires_grad)
    assert torch.equal(corrected_loss, legacy_loss)
    corrected_loss.backward()
    legacy_loss.backward()
    corrected_scaler = _FakeGradScaler()
    legacy_scaler = _FakeGradScaler()

    legacy_scaler.unscale_(legacy_optimizer)
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in legacy.parameters() if parameter.requires_grad],
        1.0,
    )
    legacy_scaler.step(legacy_optimizer)
    legacy_scaler.update()
    legacy_scheduler.step()

    result = _optimizer_boundary_update(
        model=corrected,
        optimizer=corrected_optimizer,
        scaler=corrected_scaler,
        amp_enabled=True,
        stage="sequence-pretrain",
        gradient_clip_norm=1.0,
        amp_overflows_total=0,
        amp_overflows_consecutive=0,
        maximum_total_amp_overflows=10,
        maximum_consecutive_amp_overflows=3,
    )
    corrected_scheduler.step()
    assert result["update_skipped"] is False
    for first, second in zip(
        corrected_scaler.gradients_at_step,
        legacy_scaler.gradients_at_step,
        strict=True,
    ):
        assert torch.equal(first, second)
    _assert_nested_equal(corrected.state_dict(), legacy.state_dict())
    _assert_nested_equal(corrected_optimizer.state_dict(), legacy_optimizer.state_dict())
    _assert_nested_equal(corrected_scheduler.state_dict(), legacy_scheduler.state_dict())
    assert corrected_scaler.get_scale() == legacy_scaler.get_scale()


def test_isolated_amp_overflow_is_skipped_and_recovers(tmp_path: Path) -> None:
    model, optimizer = _stage_model(tmp_path)
    name = sorted(_expected_active_parameter_names(model, "sequence-pretrain"))[0]
    _populate_active_gradients(model, "sequence-pretrain", nonfinite=name)
    scaler = _FakeGradScaler(overflow=True)
    skipped = _optimizer_boundary_update(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        amp_enabled=True,
        stage="sequence-pretrain",
        gradient_clip_norm=1.0,
        amp_overflows_total=0,
        amp_overflows_consecutive=0,
        maximum_total_amp_overflows=10,
        maximum_consecutive_amp_overflows=3,
    )
    assert skipped["update_skipped"] is True
    assert skipped["amp_scale_after"] == 4.0
    assert skipped["amp_overflows_total"] == skipped["amp_overflows_consecutive"] == 1
    _populate_active_gradients(model, "sequence-pretrain")
    scaler.overflow = False
    recovered = _optimizer_boundary_update(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        amp_enabled=True,
        stage="sequence-pretrain",
        gradient_clip_norm=1.0,
        amp_overflows_total=1,
        amp_overflows_consecutive=1,
        maximum_total_amp_overflows=10,
        maximum_consecutive_amp_overflows=3,
    )
    assert recovered["update_skipped"] is False
    assert recovered["amp_overflows_consecutive"] == 0


def test_consecutive_amp_overflow_limit_is_terminal(tmp_path: Path) -> None:
    model, optimizer = _stage_model(tmp_path)
    name = sorted(_expected_active_parameter_names(model, "sequence-pretrain"))[0]
    _populate_active_gradients(model, "sequence-pretrain", nonfinite=name)
    result = _optimizer_boundary_update(
        model=model,
        optimizer=optimizer,
        scaler=_FakeGradScaler(overflow=True),
        amp_enabled=True,
        stage="sequence-pretrain",
        gradient_clip_norm=1.0,
        amp_overflows_total=1,
        amp_overflows_consecutive=1,
        maximum_total_amp_overflows=10,
        maximum_consecutive_amp_overflows=1,
    )
    assert result["overflow_limit_exceeded"] is True


def test_nonfinite_loss_and_amp_disabled_gradient_are_fatal(tmp_path: Path) -> None:
    with pytest.raises(E006GradientError, match="non-finite training loss") as captured:
        _require_finite_training_loss(torch.tensor(float("nan")), {"sample_ids": ["sample"]})
    assert captured.value.diagnostics["sample_ids"] == ["sample"]
    model, optimizer = _stage_model(tmp_path)
    name = sorted(_expected_active_parameter_names(model, "sequence-pretrain"))[0]
    _populate_active_gradients(model, "sequence-pretrain", nonfinite=name)
    with pytest.raises(E006GradientError, match="AMP disabled") as gradient_error:
        _optimizer_boundary_update(
            model=model,
            optimizer=optimizer,
            scaler=_FakeGradScaler(),
            amp_enabled=False,
            stage="sequence-pretrain",
            gradient_clip_norm=1.0,
            amp_overflows_total=0,
            amp_overflows_consecutive=0,
            maximum_total_amp_overflows=10,
            maximum_consecutive_amp_overflows=3,
        )
    assert name in gradient_error.value.diagnostics["nonfinite_gradient_names"]


def test_stage_b_expects_every_parameter_gradient(tmp_path: Path) -> None:
    model, _ = _stage_model(tmp_path, "joint-train")
    evidence = _gradient_evidence(model, stage="joint-train")
    assert evidence["expected_active_parameter_count"] == len(tuple(model.parameters()))
    assert evidence["missing_active_gradient_count"] == evidence["expected_active_parameter_count"]


def test_calibration_oom_is_bounded_and_stops_larger_cases() -> None:
    calls = []

    def execute(case):
        calls.append(case["physical_batch_size"])
        if case["physical_batch_size"] == 2:
            raise torch.OutOfMemoryError("synthetic oom")
        return {"status": "passed", "finite": True}

    results = execute_calibration_regime(
        [
            {"target_length": 128, "physical_batch_size": 1, "accumulation_steps": 1},
            {"target_length": 128, "physical_batch_size": 2, "accumulation_steps": 1},
            {"target_length": 128, "physical_batch_size": 4, "accumulation_steps": 1},
        ],
        execute,
    )
    assert calls == [1, 2]
    assert results[-1]["status"] == "cuda_oom"


def test_calibration_selection_enforces_fifteen_percent_headroom() -> None:
    cases = []
    for length in (128, 256, 384, 500):
        cases.extend(
            [
                {
                    "target_length": length,
                    "physical_batch_size": 1,
                    "accumulation_steps": 1,
                    "pair_elements": length**2,
                    "peak_cuda_reserved_mib": 800,
                    "status": "passed",
                    "finite": True,
                },
                {
                    "target_length": length,
                    "physical_batch_size": 2,
                    "accumulation_steps": 1,
                    "pair_elements": 2 * length**2,
                    "peak_cuda_reserved_mib": 860,
                    "status": "passed",
                    "finite": True,
                },
            ]
        )
    selected = select_calibration_recommendations(cases, total_vram_mib=1000)
    assert all(item["physical_batch_size"] == 1 for item in selected)
    assert all(item["safety_headroom_fraction"] == 0.15 for item in selected)
    assert [item["effective_token_budget"] for item in selected] == [128, 256, 384, 500]


def test_calibration_and_configuration_hash_tampering(tmp_path: Path) -> None:
    report = {
        "status": "completed",
        "version": CALIBRATION_VERSION,
        "architecture_version": "e006_rich_geometry_codesign_v1",
        "dataset_identity": "dataset",
        "dataset_identity_after": "dataset",
        "protected_inputs_unchanged": True,
        "authorizes_training": False,
        "recommendations": [
            {"maximum_length": length, "safety_headroom_fraction": 0.15} for length in (128, 256, 384, 500)
        ],
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(report))
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    assert verify_calibration_report(path, digest, dataset_identity="dataset") == report
    with pytest.raises(ValueError, match="SHA-256"):
        verify_calibration_report(path, "0" * 64, dataset_identity="dataset")
    payload = {
        "version": STAGE_A_CHECKPOINT_VERSION,
        "stage": "sequence-pretrain",
        "architecture_version": "e006_rich_geometry_codesign_v1",
        "config_sha256": "config",
        "dataset_identity": "dataset",
        "calibration_sha256": "calibration",
    }
    validate_resume_checkpoint(
        payload,
        stage="sequence-pretrain",
        config_hash="config",
        dataset_identity="dataset",
        calibration_sha256="calibration",
    )
    with pytest.raises(ValueError, match="config_sha256"):
        validate_resume_checkpoint(
            payload,
            stage="sequence-pretrain",
            config_hash="changed",
            dataset_identity="dataset",
            calibration_sha256="calibration",
        )


def test_best_checkpoint_uses_sequence_ce_not_total() -> None:
    selected = select_best_checkpoint(
        [
            {"optimizer_step": 1, "sequence_cross_entropy": 1.0, "weighted_total": 0.1},
            {"optimizer_step": 2, "sequence_cross_entropy": 0.8, "weighted_total": 10.0},
        ]
    )
    assert selected["optimizer_step"] == 2


def test_loss_and_learning_rate_warmups() -> None:
    assert E006LossWeights(1.0, 0.1, 0.01).at_step(5, 10).geometry == pytest.approx(0.05)
    assert _scheduler_multiplier(0, 10, 100) == pytest.approx(0.1)
    assert _scheduler_multiplier(100, 10, 100) == pytest.approx(0.0)


def test_mixed_precision_is_explicit_and_cpu_safe() -> None:
    tensor = torch.ones(2)
    context, enabled = _amp_context(
        torch.device("cpu"),
        {"mixed_precision": {"enabled": True, "dtype": "float16"}},
    )
    with context:
        result = tensor + 1
    assert enabled is False
    assert result.dtype == torch.float32


def test_nonfinite_model_is_rejected() -> None:
    model = torch.nn.Linear(2, 2)
    model.weight.data[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite parameters"):
        _assert_finite_model(model)


def test_stage_a_atomic_authorization_and_deterministic_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "protein_distance_diffusion.training.rich_codesign_production.collate_rich_geometry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Stage A constructed pair features")),
    )
    config, path = _config(tmp_path, "resumed")
    config["training"]["interrupt_after_optimizer_steps"] = 1
    path.write_text(json.dumps(config))
    with pytest.raises(TrainingInterrupted):
        run_training_stage(path, mode="sequence-pretrain", synthetic=True)
    assert json.loads((tmp_path / "resumed" / "protocol.json").read_text())["authorizes_joint_training"] is False

    config["training"].pop("interrupt_after_optimizer_steps")
    path.write_text(json.dumps(config))
    resumed = run_training_stage(path, mode="sequence-pretrain", synthetic=True, resume=True)
    assert resumed["authorizes_joint_training"] is True
    manifest = json.loads((tmp_path / "resumed" / "checkpoint_manifest.json").read_text())
    assert [item["optimizer_step"] for item in manifest["checkpoints"]] == [0, 1, 2]
    assert manifest["rolling_latest"]["role"] == "rolling_recovery"
    assert manifest["best"]["role"] == "validation_selected_best"
    assert all(Path(item["path"]).is_file() for item in manifest["checkpoints"])
    assert resumed["checkpoint_manifest_sha256"]
    assert resumed["validation_journal_sha256"]
    checkpoint = Path(resumed["checkpoint_path"])
    verified = verify_stage_a_checkpoint(
        checkpoint,
        resumed["checkpoint_sha256"],
        dataset_identity="synthetic-e006-rich-v1",
        calibration_sha256="synthetic-calibration-v1",
    )
    assert verified["selected_validation_sequence_cross_entropy"] == resumed["best_sequence_cross_entropy"]

    _, uninterrupted_path = _config(tmp_path, "uninterrupted")
    uninterrupted = run_training_stage(uninterrupted_path, mode="sequence-pretrain", synthetic=True)
    uninterrupted_payload = load_checkpoint(uninterrupted["checkpoint_path"])
    for name, value in verified["model"].items():
        assert torch.equal(value, uninterrupted_payload["model"][name]), name


def test_resume_after_skipped_overflow_preserves_exact_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from protein_distance_diffusion.training import rich_codesign_production as production

    original_update = production._optimizer_boundary_update
    skip_next = [True]

    def skip_once(**kwargs):
        if skip_next[0]:
            skip_next[0] = False
            kwargs["optimizer"].zero_grad(set_to_none=True)
            evidence = production._gradient_evidence(kwargs["model"], stage=kwargs["stage"])
            return {
                "update_skipped": True,
                "amp_scale_before": 8.0,
                "amp_scale_after": 4.0,
                "amp_overflows_total": kwargs["amp_overflows_total"] + 1,
                "amp_overflows_consecutive": kwargs["amp_overflows_consecutive"] + 1,
                "overflow_limit_exceeded": False,
                "gradient_norm_preclip": float("inf"),
                "gradient_norms": {"sequence": None, "geometry": 0.0, "fusion": 0.0},
                "gradient_evidence": evidence,
            }
        return original_update(**kwargs)

    monkeypatch.setattr(production, "_optimizer_boundary_update", skip_once)
    config, path = _config(tmp_path, "overflow-resume")
    config["training"]["interrupt_after_optimizer_steps"] = 1
    path.write_text(json.dumps(config))
    with pytest.raises(TrainingInterrupted):
        run_training_stage(path, mode="sequence-pretrain", synthetic=True)
    checkpoint = load_checkpoint(tmp_path / "overflow-resume" / "checkpoints" / "latest.pt")
    assert checkpoint["amp_overflows_total"] == 1
    assert checkpoint["amp_overflows_consecutive"] == 0
    assert checkpoint["sampler_state"]["data_cursor"] == 2

    config["training"].pop("interrupt_after_optimizer_steps")
    path.write_text(json.dumps(config))
    resumed = run_training_stage(path, mode="sequence-pretrain", synthetic=True, resume=True)

    skip_next[0] = True
    _, uninterrupted_path = _config(tmp_path, "overflow-uninterrupted")
    uninterrupted = run_training_stage(uninterrupted_path, mode="sequence-pretrain", synthetic=True)
    resumed_payload = load_checkpoint(resumed["checkpoint_path"])
    uninterrupted_payload = load_checkpoint(uninterrupted["checkpoint_path"])
    assert resumed["optimizer_steps"] == uninterrupted["optimizer_steps"] == 1
    assert resumed["processed_valid_tokens"] == uninterrupted["processed_valid_tokens"]
    for name, value in resumed_payload["model"].items():
        assert torch.equal(value, uninterrupted_payload["model"][name]), name


def test_joint_stage_initializes_only_from_authorized_stage_a(tmp_path: Path) -> None:
    _, stage_a_path = _config(tmp_path, "stage-a-for-joint")
    stage_a = run_training_stage(stage_a_path, mode="sequence-pretrain", synthetic=True)
    joint, joint_path = _config(tmp_path, "joint")
    joint["objective"].update(geometry_weight=0.05, consistency_weight=0.01)
    joint["stage_a"] = {
        "checkpoint_path": stage_a["checkpoint_path"],
        "checkpoint_sha256": stage_a["checkpoint_sha256"],
    }
    joint["training"]["maximum_optimizer_updates"] = 1
    joint_path.write_text(json.dumps(joint))
    result = run_training_stage(joint_path, mode="joint-train", synthetic=True)
    assert result["authorizes_joint_training"] is False
    assert result["authorizes_definitive_evaluation"] is True
    assert result["best_checkpoint_criterion"] == "validation_sequence_cross_entropy"


def test_production_configs_pin_calibration_and_selection_without_smoke_checkpoint() -> None:
    for path in (
        "configs/e006_rich_geometry_sequence_pretrain.yaml",
        "configs/e006_rich_geometry_sequence_pretrain_v2.yaml",
        "configs/e006_rich_geometry_sequence_pretrain_v3.yaml",
        "configs/e006_rich_geometry_sequence_pretrain_v4_continuation.yaml",
        "configs/e006_rich_geometry_sequence_pretrain_v5.yaml",
        "configs/e006_rich_geometry_joint_train.yaml",
    ):
        config = load_yaml(path)
        assert config["calibration"]["report_sha256"] == (
            "47c9d9548ccdc037df15e5e12879b09e1a45ec9ed9dcf987192ccee5d6cf2c2a"
        )
        assert config["calibration"]["production_selection_sha256"] == (
            "eee314b655cb0614635dd8f152964bf2732876b921483c51e6a1000a8f5a6e31"
        )
        assert len(config["batching"]["regimes"]) == 4
        assert "smoke_checkpoint" not in json.dumps(config)


def test_stage_a_v3_preserves_v2_science_and_stage_b_waits_for_verified_v5() -> None:
    failed = load_yaml("configs/e006_rich_geometry_sequence_pretrain_v2.yaml")
    versioned = load_yaml("configs/e006_rich_geometry_sequence_pretrain_v3.yaml")
    failed["experiment"] = versioned["experiment"]
    failed["training"]["output_dir"] = versioned["training"]["output_dir"]
    for name in (
        "maximum_total_amp_overflows",
        "maximum_consecutive_amp_overflows",
        "maximum_overflow_diagnostic_examples",
    ):
        failed["training"][name] = versioned["training"][name]
    assert failed == versioned
    assert versioned["training"]["output_dir"] == "outputs/e006_phase3_sequence_pretrain_v3"
    joint = load_yaml("configs/e006_rich_geometry_joint_train.yaml")
    assert joint["stage_a"]["checkpoint_path"] == (
        "outputs/e006_phase3_sequence_pretrain_v5_warm_start/checkpoints/best_context_verified.pt"
    )
    assert joint["stage_a"]["requires_contextual_checkpoint"] is True
    assert joint["stage_a"]["checkpoint_sha256"] is None
    assert joint["stage_a"]["context_diagnostic_sha256"] is None
    assert "sequence_pretrain_v2" not in json.dumps(joint)
    assert "sequence_pretrain_v3" not in json.dumps(joint)


def test_audited_continuation_restores_exact_source_state(tmp_path: Path) -> None:
    config, _, _ = _continuation_fixture(tmp_path)
    payload, protocol = verify_continuation_source(config)
    expected = config["continuation"]
    assert payload["optimizer_step"] == expected["source_optimizer_step"] == 1
    assert payload["microstep"] == expected["source_microstep"]
    assert payload["sampler_state"]["data_cursor"] == expected["source_data_cursor"]
    assert payload["processed_valid_tokens"] == expected["source_processed_valid_tokens"]
    assert payload["accumulation_state"] == {
        "microstep": 1,
        "microbatches_accumulated": 0,
        "at_optimizer_boundary": True,
    }
    assert payload["scheduler"]["last_epoch"] == payload["optimizer_step"]
    assert payload["scaler"] == {}
    assert set(payload["rng_state"]) == {"python", "numpy", "torch", "cuda"}
    assert protocol["scientific_settings_equal"] is True
    assert protocol["next_scheduled_validation_step"] == 2


def test_audited_continuation_rejects_hash_and_scientific_mismatch(tmp_path: Path) -> None:
    config, _, _ = _continuation_fixture(tmp_path)
    wrong_hash = copy.deepcopy(config)
    wrong_hash["continuation"]["source_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source hash contradiction"):
        verify_continuation_source(wrong_hash)
    changed_science = copy.deepcopy(config)
    changed_science["objective"]["mask_fraction"] = 0.5
    with pytest.raises(ValueError, match="scientific configuration mismatch"):
        verify_continuation_source(changed_science)


def test_audited_continuation_rejects_nonboundary_checkpoint(tmp_path: Path) -> None:
    config, _, _ = _continuation_fixture(tmp_path)
    checkpoint = Path(config["continuation"]["source_checkpoint_path"])
    payload = load_checkpoint(checkpoint)
    payload["accumulation_state"]["at_optimizer_boundary"] = False
    save_checkpoint(checkpoint, payload)
    digest = _file_sha256(checkpoint)
    metadata_path = Path(config["continuation"]["source_checkpoint_metadata_path"])
    metadata = json.loads(metadata_path.read_text())
    metadata["sha256"] = digest
    metadata_path.write_text(json.dumps(metadata))
    manifest_path = Path(config["continuation"]["source_checkpoint_manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    manifest["rolling_latest"]["sha256"] = digest
    manifest_path.write_text(json.dumps(manifest))
    config["continuation"].update(
        source_checkpoint_sha256=digest,
        source_checkpoint_metadata_sha256=_file_sha256(metadata_path),
        source_checkpoint_manifest_sha256=_file_sha256(manifest_path),
    )
    with pytest.raises(ValueError, match="optimizer-boundary"):
        verify_continuation_source(config)


def test_continuation_imports_incumbent_and_resumes_in_same_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from protein_distance_diffusion.training import rich_codesign_production as production

    config, path, source_output = _continuation_fixture(tmp_path)
    source_fingerprint = {
        str(item.relative_to(source_output)): _file_sha256(item) for item in source_output.rglob("*") if item.is_file()
    }
    config["training"]["interrupt_after_optimizer_steps"] = 2
    path.write_text(json.dumps(config))
    monkeypatch.setattr(
        production,
        "_validation",
        lambda *_args, **_kwargs: {"sequence_cross_entropy": 999.0, "optimizer_step": 2},
    )
    with pytest.raises(TrainingInterrupted):
        run_training_stage(path, mode="sequence-pretrain", synthetic=True, continuation=True)
    output = Path(config["training"]["output_dir"])
    migration = json.loads((output / "migration_protocol.json").read_text())
    assert migration["status"] == "startup_verified"
    assert migration["authorizes_training"] is False
    assert migration["imported_state"]["optimizer_step"] == 1
    assert json.loads((output / "checkpoints" / "best.json").read_text())["optimizer_step"] >= 1
    resumed_checkpoint = load_checkpoint(output / "checkpoints" / "latest.pt")
    assert resumed_checkpoint["optimizer_step"] == 2
    assert resumed_checkpoint["sampler_state"]["data_cursor"] == 2
    assert resumed_checkpoint["migration_source"] == {
        "role": "audited_migration_source",
        "checkpoint_path": config["continuation"]["source_checkpoint_path"],
        "checkpoint_sha256": config["continuation"]["source_checkpoint_sha256"],
        "optimizer_step": 1,
        "authorizes_training": False,
    }
    assert json.loads((output / "checkpoints" / "best.json").read_text())["role"] == ("imported_validation_incumbent")

    config["training"].pop("interrupt_after_optimizer_steps")
    path.write_text(json.dumps(config))
    completed = run_training_stage(path, mode="sequence-pretrain", synthetic=True, resume=True)
    assert completed["optimizer_steps"] == 3
    assert completed["continuation_lineage"]["imported_best_incumbent"]["optimizer_step"] >= 1
    assert completed["continuation_source_unchanged"] is True
    assert {
        str(item.relative_to(source_output)): _file_sha256(item) for item in source_output.rglob("*") if item.is_file()
    } == source_fingerprint


def test_v4_continuation_config_is_not_scratch_or_ordinary_initialization() -> None:
    config = load_yaml("configs/e006_rich_geometry_sequence_pretrain_v4_continuation.yaml")
    assert config["training"]["output_dir"] == "outputs/e006_phase3_sequence_pretrain_v4_continuation"
    assert config["continuation"]["source_optimizer_step"] == 6000
    assert config["continuation"]["discarded_uncheckpointed_updates"] == 13
    assert config["continuation"]["next_scheduled_validation_step"] == 7500
    assert config["continuation"]["dataset_pass_end_steps"] == [15012, 30024, 45036]
    assert config["continuation"]["total_target_optimizer_steps"] == 45036
