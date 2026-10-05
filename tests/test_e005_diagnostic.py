from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.collate import make_sequence_separation
from protein_distance_diffusion.evaluation.codesign_diagnostic import (
    PanelSelectionError,
    _AtomicParquetRows,
    calibration_metrics,
    clustered_paired_bootstrap,
    geometry_diagnostics,
    holm_adjust,
    noise_level_to_timestep,
    reconstruct_validation_trajectory,
    select_diagnostic_panel,
    sequence_baseline_metrics,
    streaming_amino_acid_frequencies,
    validate_diagnostic_config,
)
from protein_distance_diffusion.models.codesign import (
    GATE_ABLATION_DISABLED_PATHS,
    GATE_ABLATIONS,
    E005SequenceGeometryCoDesign,
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
    ).eval()


def _gate_outputs(model: E005SequenceGeometryCoDesign, gate_ablation: str) -> dict[str, torch.Tensor]:
    length = 8
    tokens = torch.arange(2, 2 + length)[None]
    residue_mask = torch.ones((1, length), dtype=torch.bool)
    pair_mask = residue_mask[:, None, :, None] & residue_mask[:, None, None, :]
    geometry = torch.arange(length * length, dtype=torch.float32).reshape(1, 1, length, length)
    geometry = 0.5 * (geometry + geometry.transpose(-1, -2))
    geometry[:, :, torch.arange(length), torch.arange(length)] = 0
    return model(
        sequence_token_ids=tokens,
        residue_mask=residue_mask,
        noisy_geometry=geometry,
        timesteps=torch.tensor([10]),
        lengths=torch.tensor([length]),
        sequence_separation=make_sequence_separation(torch.tensor([length]), length),
        pair_mask=pair_mask,
        geometry_conditioning_mask=torch.ones(1, dtype=torch.bool),
        mode="learned_geometry_gating",
        gate_ablation=gate_ablation,
    )


def test_gate_ablation_domain_and_effective_values_are_isolated() -> None:
    assert set(GATE_ABLATION_DISABLED_PATHS) == GATE_ABLATIONS
    assert GATE_ABLATION_DISABLED_PATHS["both_geometry_to_sequence_disabled"] == {
        "incoming_geometry_to_sequence",
        "returned_geometry_to_sequence",
    }
    model = _model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    outputs = {condition: _gate_outputs(model, condition) for condition in GATE_ABLATIONS}
    keys = {
        "incoming_geometry_to_sequence": "geometry_to_sequence_gate",
        "sequence_to_geometry": "sequence_to_geometry_gate",
        "returned_geometry_to_sequence": "return_geometry_gate",
    }
    for condition, disabled in GATE_ABLATION_DISABLED_PATHS.items():
        for path in disabled:
            assert not outputs[condition][keys[path]].any()
    for value in keys.values():
        assert torch.all(outputs["all_forced_one"][value] == 1)
        assert not outputs["all_disabled"][value].any()
    assert torch.equal(
        outputs["returned_geometry_to_sequence_disabled"]["geometry_to_sequence_gate"],
        outputs["all_learned"]["geometry_to_sequence_gate"],
    )
    assert torch.equal(
        outputs["returned_geometry_to_sequence_disabled"]["sequence_to_geometry_gate"],
        outputs["all_learned"]["sequence_to_geometry_gate"],
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count


def test_streaming_training_frequencies_and_sequence_baselines(tmp_path: Path) -> None:
    path = tmp_path / "eligible_train.parquet"
    pq.write_table(
        pa.table(
            {
                "sequence": ["AC", "AA", "CD"],
                "practical_training_eligibility": ["strict_verified_pair"] * 3,
                "unused_nested": [[{"value": index}] for index in range(3)],
            }
        ),
        path,
    )
    frequencies = streaming_amino_acid_frequencies(path, batch_size=2)
    assert frequencies["sample_count"] == 3
    assert frequencies["counts"]["A"] == 3
    assert frequencies["counts"]["C"] == 2
    training = {residue: 1 for residue in "ACDEFGHIKLMNPQRSTVWY"}
    training["A"] = 21
    metrics = sequence_baseline_metrics({"A": 3, "C": 1}, training)
    assert metrics["uniform_20_class"]["perplexity"] == 20
    assert metrics["most_frequent_residue"] == "A"
    assert metrics["most_frequent_residue_accuracy"] == 0.75


def test_frequency_scan_does_not_materialize_full_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "eligible_train.parquet"
    pq.write_table(
        pa.table(
            {
                "sequence": ["ACDE"] * 10_000,
                "practical_training_eligibility": ["strict_verified_pair"] * 10_000,
                "nested_unused": [[{"candidate": index}] for index in range(10_000)],
            }
        ),
        path,
        row_group_size=1000,
    )
    monkeypatch.setattr(pq, "read_table", lambda *args, **kwargs: pytest.fail("full table materialized"))
    result = streaming_amino_acid_frequencies(path, batch_size=4096)
    assert result["sample_count"] == 10_000
    assert result["token_count"] == 40_000


def test_dedicated_configuration_preserves_preregistered_bounds() -> None:
    config = load_yaml("configs/evaluate_e005_large_diagnostic.yaml")
    validate_diagnostic_config(config)
    config["panel_size"] = 1023
    with pytest.raises(ValueError, match="1,024"):
        validate_diagnostic_config(config)


def _panel_row(index: int, *, method: str = "xray", classification: str = "unique") -> dict[str, object]:
    return {
        "sample_id": f"sample-{index:05d}",
        "schema_version": "sequence_geometry_pairing_v1",
        "sequence": "A" * 32,
        "sequence_length": 32,
        "matrix_length": 32,
        "matrix_path": f"/unused/sample-{index:05d}.npz",
        "practical_training_eligibility": "strict_verified_pair",
        "experimental_method": method,
        "pairing_classification": classification,
    }


def _write_panel_rows(path: Path, rows: list[dict[str, object]], *, fragments: int = 1) -> None:
    path.mkdir()
    for fragment_index in range(fragments):
        fragment_rows = rows[fragment_index::fragments]
        if fragment_rows:
            pq.write_table(pa.Table.from_pylist(fragment_rows), path / f"part-{fragment_index:03d}.parquet")


def test_sparse_strata_redistribute_quota_and_use_global_backfill(tmp_path: Path) -> None:
    rows = [
        *[_panel_row(0, method="sparse-a")],
        *[_panel_row(1, method="sparse-b")],
        *[_panel_row(index, method="large") for index in range(2, 22)],
    ]
    path = tmp_path / "validation"
    _write_panel_rows(path, rows)
    panel, diagnostics = select_diagnostic_panel(path, panel_size=12, seed=17)
    assert len(panel) == len({row["sample_id"] for row in panel}) == 12
    assert diagnostics["filtered_candidate_count"] == 22
    assert diagnostics["initially_stratified_count"] == 4
    assert diagnostics["backfilled_count"] == 8
    selected = {row["sample_id"] for row in panel}
    assert {"sample-00000", "sample-00001"} <= selected
    assert diagnostics["panel_sample_id_sha256"]


def test_observed_913_deficit_pattern_backfills_to_1024(tmp_path: Path) -> None:
    rows = []
    index = 0
    for method_index, count in enumerate([219, *([32] * 27), 17]):
        for _ in range(count):
            rows.append(_panel_row(index, method=f"method-{method_index:02d}"))
            index += 1
    path = tmp_path / "validation"
    _write_panel_rows(path, rows, fragments=3)
    panel, diagnostics = select_diagnostic_panel(path, panel_size=1024, seed=55, batch_size=7)
    assert diagnostics["filtered_candidate_count"] == 1100
    assert diagnostics["initially_stratified_count"] == 913
    assert diagnostics["backfilled_count"] == 111
    assert diagnostics["final_panel_size"] == diagnostics["unique_sample_count"] == 1024
    assert len(panel) == 1024


def test_panel_is_independent_of_rows_fragments_and_arrow_batch_size(tmp_path: Path) -> None:
    rows = [_panel_row(index, method=f"method-{index % 4}", classification=f"class-{index % 3}") for index in range(80)]
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_panel_rows(first, rows, fragments=1)
    _write_panel_rows(second, list(reversed(rows)), fragments=5)
    panel_a, diagnostics_a = select_diagnostic_panel(first, panel_size=50, seed=31, batch_size=1)
    panel_b, diagnostics_b = select_diagnostic_panel(second, panel_size=50, seed=31, batch_size=17)
    assert [row["sample_id"] for row in panel_a] == [row["sample_id"] for row in panel_b]
    assert diagnostics_a["panel_sample_id_sha256"] == diagnostics_b["panel_sample_id_sha256"]
    assert diagnostics_a["per_stratum"] == diagnostics_b["per_stratum"]


def test_duplicate_sample_ids_are_deduplicated_strictly(tmp_path: Path) -> None:
    rows = [_panel_row(index) for index in range(12)]
    rows.append(dict(rows[3]))
    path = tmp_path / "validation"
    _write_panel_rows(path, rows, fragments=2)
    panel, diagnostics = select_diagnostic_panel(path, panel_size=10, seed=2)
    assert len({row["sample_id"] for row in panel}) == 10
    assert diagnostics["filtered_candidate_count"] == 12
    assert diagnostics["filtered_candidate_row_count"] == 13
    assert diagnostics["rejected_counts"]["duplicate_sample_id"] == 1

    conflicting = [*rows[:-1], {**rows[3], "sequence": "C" * 32}]
    conflicting_path = tmp_path / "conflicting"
    _write_panel_rows(conflicting_path, conflicting)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        select_diagnostic_panel(conflicting_path, panel_size=10, seed=2)


def test_insufficient_population_reports_filters_and_strata(tmp_path: Path) -> None:
    rows = [_panel_row(index, method=f"method-{index}") for index in range(3)]
    rows.extend(
        [
            {**_panel_row(10), "sample_id": "", "sequence_length": 32},
            {**_panel_row(11), "sequence_length": 0},
            {**_panel_row(12), "sequence_length": 501},
            {**_panel_row(13), "practical_training_eligibility": "excluded_or_unresolved"},
        ]
    )
    path = tmp_path / "validation"
    _write_panel_rows(path, rows)
    with pytest.raises(PanelSelectionError) as captured:
        select_diagnostic_panel(path, panel_size=4, seed=9)
    diagnostics = captured.value.diagnostics
    assert diagnostics["requested_panel_size"] == 4
    assert diagnostics["filtered_candidate_count"] == 3
    assert diagnostics["rejected_counts"] == {
        "missing_sample_id": 1,
        "invalid_sequence_length": 1,
        "length_above_maximum": 1,
        "ineligible": 1,
        "duplicate_sample_id": 0,
        "conflicting_duplicate_sample_id": 0,
    }
    assert diagnostics["available_counts_by_dimension"]["experimental_method"] == {
        "method-0": 1,
        "method-1": 1,
        "method-2": 1,
    }


def _journal(path: Path, losses: list[float], *, mismatch: bool = False) -> None:
    entries = []
    for step_index, loss in enumerate(losses):
        records = []
        for sample in ("a", "b"):
            for mode in ("sequence_only", "learned_geometry_gating", "forced_geometry_conditioning"):
                for seed_index in (0, 1):
                    if mismatch and step_index == 1 and sample == "b" and mode == "sequence_only":
                        continue
                    records.append(
                        {
                            "sample_id": sample,
                            "mode": mode,
                            "seed_index": seed_index,
                            "token_count": 2,
                            "sequence_loss": loss if mode == "learned_geometry_gating" else loss + 0.1,
                        }
                    )
        entries.append({"optimizer_step": step_index, "records": records})
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))


def test_trajectory_matching_and_deterministic_classification(tmp_path: Path) -> None:
    path = tmp_path / "validation.jsonl"
    _journal(path, [1.0, 0.9, 0.8, 0.7])
    result = reconstruct_validation_trajectory(
        path,
        expected_steps=range(4),
        seed=5,
        mask_probability=0.15,
        diffusion_steps=500,
        plateau_relative_threshold=0.01,
        degradation_relative_threshold=0.02,
    )
    assert result["classification"] == "still_improving"
    assert result["record_membership_comparable"] is True
    assert result["historical_tensor_hashes_available"] is False
    assert result["fingerprint_count"] == 4

    _journal(path, [1.0, 0.8, 0.8, 0.8])
    assert (
        reconstruct_validation_trajectory(
            path,
            expected_steps=range(4),
            seed=5,
            mask_probability=0.15,
            diffusion_steps=500,
            plateau_relative_threshold=0.01,
            degradation_relative_threshold=0.02,
        )["classification"]
        == "plateaued"
    )
    _journal(path, [1.0, 0.7, 0.75, 0.8])
    assert (
        reconstruct_validation_trajectory(
            path,
            expected_steps=range(4),
            seed=5,
            mask_probability=0.15,
            diffusion_steps=500,
            plateau_relative_threshold=0.01,
            degradation_relative_threshold=0.02,
        )["classification"]
        == "degraded_after_best"
    )

    _journal(path, [1.0, 0.9], mismatch=True)
    with pytest.raises(ValueError, match="membership differs"):
        reconstruct_validation_trajectory(
            path,
            expected_steps=range(2),
            seed=5,
            mask_probability=0.15,
            diffusion_steps=500,
            plateau_relative_threshold=0.01,
            degradation_relative_threshold=0.02,
        )


def test_clustered_bootstrap_resamples_proteins_and_holm_correction() -> None:
    rows = [
        {"sample_id": "a", "difference": 1.0},
        {"sample_id": "a", "difference": 3.0},
        {"sample_id": "b", "difference": 4.0},
        {"sample_id": "b", "difference": 6.0},
    ]
    result = clustered_paired_bootstrap(rows, value_key="difference", iterations=10_000, seed=7)
    assert result["protein_count"] == 2
    assert result["mean_difference"] == 3.5
    assert result["median_difference"] == 3.5
    assert result == clustered_paired_bootstrap(rows, value_key="difference", iterations=10_000, seed=7)
    adjusted = holm_adjust({"first": 0.01, "second": 0.04, "third": 0.03})
    assert adjusted == {"first": 0.03, "third": 0.06, "second": 0.06}
    with pytest.raises(ValueError, match="10,000"):
        clustered_paired_bootstrap(rows, value_key="difference", iterations=9999)


def test_noise_mapping_geometry_and_calibration_diagnostics() -> None:
    assert [noise_level_to_timestep(value, 500) for value in (0, 0.25, 0.5, 0.75, 1)] == [
        0,
        125,
        250,
        374,
        499,
    ]
    coordinates = np.arange(8, dtype=np.float64)[:, None] * 3.8
    matrix = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    diagnostics = geometry_diagnostics(matrix / 50.0, scale=50.0, triangle_samples=1000, seed=3)
    assert diagnostics["finite"] is True
    assert diagnostics["symmetry_error_angstrom"] == pytest.approx(0)
    assert diagnostics["diagonal_error_angstrom"] == pytest.approx(0)
    assert diagnostics["negative_distance_count"] == 0
    assert diagnostics["triangle_violation_fraction"] == 0
    assert diagnostics["coordinate_reconstruction_stress"] == pytest.approx(0, abs=1e-7)
    assert diagnostics["adjacent_distance_mean_angstrom"] == pytest.approx(3.8)
    calibration = calibration_metrics(np.asarray([[0.8, 0.2], [0.4, 0.6]]), np.asarray([0, 1]), bins=5)
    assert 0 <= calibration["top_label_ece"] <= 1
    assert calibration["multiclass_brier_score"] == pytest.approx(0.2)


def test_sample_level_parquet_publication_is_atomic_and_bounded(tmp_path: Path) -> None:
    destination = tmp_path / "sample_level.parquet"
    writer = _AtomicParquetRows(destination, buffer_size=2)
    for index in range(5):
        writer.append({"sample_id": f"sample-{index}", "value": float(index)})
        assert len(writer.rows) <= 1
    assert not destination.exists()
    writer.publish()
    assert pq.ParquetFile(destination).metadata.num_rows == 5
    assert not list(tmp_path.glob("*.tmp"))
