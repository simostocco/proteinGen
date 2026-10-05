from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.collate import make_sequence_separation
from protein_distance_diffusion.data.sequence_geometry import (
    PAIRING_SCHEMA_VERSION,
    SequenceGeometryDataset,
    collate_sequence_geometry,
)
from protein_distance_diffusion.models.codesign import (
    CoDesignLossWeights,
    E005SequenceGeometryCoDesign,
    codesign_losses,
)
from protein_distance_diffusion.training.codesign import (
    MemoryStageReporter,
    _guard_memory,
    conditioning_mask,
    run_codesign_dry_run,
)


def _model() -> E005SequenceGeometryCoDesign:
    return E005SequenceGeometryCoDesign(
        sequence_hidden_dim=16,
        sequence_layers=1,
        sequence_heads=2,
        sequence_feedforward_dim=32,
        sequence_dropout=0.0,
        max_length=16,
        geometry_model={
            "base_channels": 4,
            "channel_multipliers": (1, 2),
            "residual_blocks_per_level": 1,
            "group_norm_groups": 2,
            "attention_heads": 1,
            "use_bottleneck_attention": False,
            "time_embedding_dim": 16,
            "length_embedding_dim": 16,
            "max_length": 16,
        },
    )


def _batch() -> dict[str, torch.Tensor]:
    items = []
    for index, length in enumerate((7, 11)):
        positions = torch.arange(length, dtype=torch.float32)[:, None] * 3.8
        matrix = torch.cdist(positions, positions)
        items.append(
            {
                "sample_id": f"sample-{index}",
                "sequence_token_ids": torch.arange(2, 2 + length) % 20 + 2,
                "sequence_mask": torch.ones(length, dtype=torch.bool),
                "distance_matrix": matrix,
                "pair_mask": torch.ones((length, length), dtype=torch.bool),
                "length": length,
                "geometry_availability_flag": True,
                "geometry_conditioning_flag": True,
            }
        )
    return collate_sequence_geometry(items, pad_to_multiple=2)


def _forward(model: E005SequenceGeometryCoDesign, batch: dict, mode: str, geometry: torch.Tensor | None = None):
    matrices = batch["distance_matrices"][:, None] if geometry is None else geometry
    lengths = batch["lengths"]
    residue_mask = batch["sequence_mask"]
    pair_mask = residue_mask[:, None, :, None] & residue_mask[:, None, None, :]
    return model(
        sequence_token_ids=batch["sequence_token_ids"],
        residue_mask=residue_mask,
        noisy_geometry=matrices,
        timesteps=torch.tensor([2, 3]),
        lengths=lengths,
        sequence_separation=make_sequence_separation(lengths, matrices.shape[-1]),
        pair_mask=pair_mask,
        geometry_conditioning_mask=torch.ones(2, dtype=torch.bool),
        mode=mode,
    )


@pytest.mark.parametrize(
    "mode",
    ["sequence_only", "learned_geometry_gating", "forced_geometry_conditioning"],
)
def test_e005_forward_backward_is_finite_and_masked(mode: str) -> None:
    torch.manual_seed(7)
    model = _model()
    batch = _batch()
    outputs = _forward(model, batch, mode)
    assert outputs["sequence_logits"].shape == (2, 12, 22)
    assert outputs["geometry_prediction"].shape == (2, 1, 12, 12)
    assert torch.isfinite(outputs["sequence_logits"]).all()
    assert torch.isfinite(outputs["geometry_prediction"]).all()
    assert not outputs["sequence_logits"][1, 11:].any()
    assert not outputs["geometry_prediction"][0, :, 7:, :].any()
    masked = torch.zeros_like(batch["sequence_mask"])
    masked[:, 0] = True
    losses = codesign_losses(
        outputs,
        sequence_targets=batch["sequence_token_ids"],
        masked_token_mask=masked,
        geometry_target=torch.zeros_like(outputs["geometry_prediction"]),
        weights=CoDesignLossWeights(consistency=0.1),
    )
    losses["total"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_sequence_only_logits_cannot_observe_ground_truth_geometry() -> None:
    torch.manual_seed(11)
    model = _model().eval()
    batch = _batch()
    first = _forward(model, batch, "sequence_only", torch.randn(2, 1, 12, 12))
    second = _forward(model, batch, "sequence_only", torch.randn(2, 1, 12, 12) * 100)
    assert torch.equal(first["sequence_logits"], second["sequence_logits"])
    assert not first["geometry_to_sequence_gate"].any()
    assert not first["sequence_to_geometry_gate"].any()
    assert not first["return_geometry_gate"].any()


def test_conditioning_dropout_is_deterministic_and_seeded() -> None:
    first = conditioning_mask(64, probability=0.5, seed=5, step=2, device=torch.device("cpu"))
    repeated = conditioning_mask(64, probability=0.5, seed=5, step=2, device=torch.device("cpu"))
    changed = conditioning_mask(64, probability=0.5, seed=6, step=2, device=torch.device("cpu"))
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)


def test_collate_can_pad_to_unet_factor_without_exposing_padding() -> None:
    batch = _batch()
    assert batch["sequence_token_ids"].shape == (2, 12)
    assert not batch["sequence_mask"][0, 7:].any()
    assert not batch["pair_mask"][0, 7:, :].any()
    with pytest.raises(ValueError, match="positive"):
        collate_sequence_geometry(
            [
                {
                    "sample_id": "x",
                    "length": 1,
                    "sequence_token_ids": torch.tensor([2]),
                    "sequence_mask": torch.ones(1, dtype=torch.bool),
                    "distance_matrix": torch.zeros((1, 1)),
                    "pair_mask": torch.ones((1, 1), dtype=torch.bool),
                    "geometry_availability_flag": True,
                    "geometry_conditioning_flag": True,
                }
            ],
            pad_to_multiple=0,
        )


def test_synthetic_dry_run_is_deterministic_and_bounded(tmp_path: Path) -> None:
    config_path = Path("configs/e005_sequence_geometry_codesign_synthetic_dry_run.yaml")
    config = load_yaml(config_path)
    first = run_codesign_dry_run(
        config,
        report_path=tmp_path / "first.json",
        checkpoint_path=tmp_path / "dry.pt",
        steps=1,
    )
    second = run_codesign_dry_run(config, report_path=tmp_path / "second.json", steps=1)
    assert first["input_kind"] == "synthetic"
    assert first["configuration"] == config
    assert first["losses_by_step"] == second["losses_by_step"]
    assert first["tensor_shapes"]["geometry_prediction"] == [2, 1, 16, 16]
    assert first["parameter_count"] == first["trainable_parameter_count"]
    assert (tmp_path / "dry.pt").is_file()
    assert json.loads((tmp_path / "first.json").read_text())["status"] == "completed"
    stages = [item["stage"] for item in first["stages"]]
    assert stages == [
        "imports_startup",
        "configuration_load",
        "dataset_construction",
        "sample_selection",
        "npz_loading",
        "model_construction",
        "forward",
        "backward",
        "optimizer_step",
        "dataset_integrity_verification",
        "checkpoint_writing",
    ]
    with pytest.raises(ValueError, match="steps"):
        run_codesign_dry_run(config, report_path=tmp_path / "invalid.json", steps=11)


def test_full_e005_config_preserves_e004_geometry_architecture_and_count() -> None:
    e005 = load_yaml("configs/e005_sequence_geometry_codesign_full.yaml")
    e004 = load_yaml("configs/train_recovered_full_v_axial_edm_triangle_e004_full.yaml")
    assert e005["model"]["geometry_model"] == e004["model"]
    model_config = dict(e005["model"])
    geometry_config = model_config.pop("geometry_model")
    model = E005SequenceGeometryCoDesign(geometry_model=geometry_config, **model_config)
    assert sum(parameter.numel() for parameter in model.geometry_model.parameters()) == 7_582_833
    assert sum(parameter.numel() for parameter in model.parameters()) == 8_512_649


def test_large_partitioned_loader_projects_rows_without_pandas_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "large.parquet"
    manifest.mkdir()
    rows_per_partition = 4_000
    partition_count = 25
    nested_payload = [[{"unused": index}] for index in range(rows_per_partition)]
    for partition in range(partition_count):
        offset = partition * rows_per_partition
        table = pa.table(
            {
                "sample_id": [f"sample-{offset + index:06d}" for index in range(rows_per_partition)],
                "schema_version": [PAIRING_SCHEMA_VERSION] * rows_per_partition,
                "sequence": ["ACDE"] * rows_per_partition,
                "sequence_length": [4] * rows_per_partition,
                "matrix_length": [4] * rows_per_partition,
                "matrix_path": [""] * rows_per_partition,
                "practical_training_eligibility": ["strict_verified_pair"] * rows_per_partition,
                "nested_candidate_evidence": nested_payload,
            }
        )
        pq.write_table(table, manifest / f"part-{partition:03d}.parquet", row_group_size=rows_per_partition)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"whole-dataset pandas materialization invoked: {args}, {kwargs}")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    allocated_before = pa.total_allocated_bytes()
    dataset = SequenceGeometryDataset(manifest, mode="sequence_only")
    assert len(dataset) == 100_000
    assert dataset[0]["sample_id"] == "sample-000000"
    assert dataset[-1]["sample_id"] == "sample-099999"
    assert dataset._rows.cached_row_count <= rows_per_partition
    assert pa.total_allocated_bytes() - allocated_before < 16 * 1024 * 1024


def test_failure_writes_incomplete_report_and_no_checkpoint(tmp_path: Path) -> None:
    config = load_yaml("configs/e005_sequence_geometry_codesign_synthetic_dry_run.yaml")
    config["conditioning"]["mode"] = "not-a-mode"
    report = tmp_path / "incomplete.json"
    checkpoint_path = tmp_path / "must-not-exist.pt"
    with pytest.raises(ValueError, match="Unsupported"):
        run_codesign_dry_run(config, report_path=report, checkpoint_path=checkpoint_path)
    payload = json.loads(report.read_text())
    assert payload["status"] == "incomplete"
    assert payload["error_type"] == "ValueError"
    assert not checkpoint_path.exists()


def test_memory_error_reports_current_and_peak_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("protein_distance_diffusion.training.codesign._rss_mib", lambda: 300.0)
    monkeypatch.setattr("protein_distance_diffusion.training.codesign._peak_rss_mib", lambda: 350.0)
    with pytest.raises(MemoryError, match=r"current_rss_mib=300.0, peak_rss_mib=350.0"):
        _guard_memory(256)


def test_reporter_checkpoints_partial_stage_atomically(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    reporter = MemoryStageReporter(path, max_memory_mib=2048)
    reporter.record("imports_startup")
    payload = json.loads(path.read_text())
    assert payload["status"] == "running"
    assert payload["current_stage"] == "imports_startup"
    assert payload["stages"][0]["current_rss_mib"] > 0


def test_synthetic_real_path_tiny_overfit_is_fixed_bounded_and_complete(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "pairing"
    manifest_dir = dataset_dir / "eligible_train.parquet"
    manifest_dir.mkdir(parents=True)
    sequence = "ACDEFGHIKLMN"
    positions = np.arange(len(sequence), dtype=np.float32)[:, None] * np.array([[3.8, 0.0, 0.0]])
    matrix = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1).astype(np.float32)
    matrix_path = tmp_path / "sample.npz"
    np.savez_compressed(matrix_path, sequence=np.asarray(sequence), distance_matrix=matrix)
    partition = manifest_dir / "part-000.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": ["synthetic-real-001"],
                "schema_version": [PAIRING_SCHEMA_VERSION],
                "sequence": [sequence],
                "sequence_length": [len(sequence)],
                "matrix_length": [len(sequence)],
                "matrix_path": [str(matrix_path)],
                "practical_training_eligibility": ["strict_verified_pair"],
            }
        ),
        partition,
    )
    protocol = dataset_dir / "protocol.json"
    protocol.write_text(json.dumps({"status": "completed", "schema_version": PAIRING_SCHEMA_VERSION}))
    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps({"mode": "scale", "scale": 50.0}))
    protected = (matrix_path, partition, protocol, normalization)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}

    config = load_yaml("configs/e005_sequence_geometry_codesign_synthetic_dry_run.yaml")
    config["dataset"] = {
        "directory": str(dataset_dir),
        "schema_version": PAIRING_SCHEMA_VERSION,
        "train_dataset": "eligible_train.parquet",
        "immutable": True,
    }
    config["normalization_file"] = str(normalization)
    config.pop("normalization_scale_angstrom")
    report_path = tmp_path / "tiny-overfit.json"
    checkpoint_path = tmp_path / "tiny-overfit.pt"
    report = run_codesign_dry_run(
        config,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        real_data=True,
        tiny_overfit=True,
    )

    assert report["status"] == "completed"
    assert report["tiny_overfit"] is True
    assert report["conditioning_mode"] == "learned_geometry_gating"
    assert report["conditioning_dropout_probability"] == 0.0
    assert report["sample_count"] == 1
    assert report["lengths"] == [len(sequence)]
    assert len(report["losses_by_step"]) == 8
    assert all(item["passed"] for item in report["loss_reduction_summary"].values())
    assert set(report["gradient_norms_by_step"][0]) == {
        "sequence_branch",
        "geometry_branch",
        "sequence_to_geometry_feedback",
        "geometry_to_sequence_feedback",
        "gates",
    }
    assert all(value > 0 for value in report["parameter_change_norms"].values())
    assert all(
        gate["saturated_fraction"] <= 0.95 for step in report["gate_statistics_by_step"] for gate in step.values()
    )
    fingerprints = report["condition_fingerprints_by_step"]
    assert fingerprints and all(item == fingerprints[0] for item in fingerprints)
    assert report["dataset_inputs_unchanged"] is True
    assert len(report["input_hashes_before"]) == 4
    assert report["input_hashes_before"] == report["input_hashes_after"]
    assert report["peak_rss_mib"] < 2048
    assert checkpoint_path.is_file()
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected} == before


def test_tiny_overfit_enforces_step_sample_length_and_memory_bounds(tmp_path: Path) -> None:
    config = load_yaml("configs/e005_sequence_geometry_codesign_synthetic_dry_run.yaml")
    cases = (
        {"steps": 51, "sample_count": 1, "maximum_length": 16, "max_memory_mib": 2048},
        {"steps": 8, "sample_count": 2, "maximum_length": 16, "max_memory_mib": 2048},
        {"steps": 8, "sample_count": 1, "maximum_length": 129, "max_memory_mib": 2048},
        {"steps": 8, "sample_count": 1, "maximum_length": 16, "max_memory_mib": 4097},
    )
    for index, arguments in enumerate(cases):
        with pytest.raises(ValueError):
            run_codesign_dry_run(
                config,
                report_path=tmp_path / f"invalid-{index}.json",
                tiny_overfit=True,
                **arguments,
            )
