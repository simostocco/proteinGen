from __future__ import annotations

import inspect
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
import yaml

import protein_distance_diffusion.evaluation.e007_coordinate_sampling_replication as replication


def _config() -> dict:
    return yaml.safe_load(Path("configs/e007_coordinate_sampling_replication_v1.yaml").read_text())


def _row(checkpoint: int, length: int, index: int, *, adjacent_error: float = 0.5) -> dict:
    seed = 3_700_001 + _config()["lengths"].index(length) * 100_000 + index
    return {
        "checkpoint_update": checkpoint,
        "length": length,
        "sample_index": index,
        "seed": seed,
        "finite_coordinates": True,
        "biological_mask_exact": True,
        "padded_zeros_exact": True,
        "deterministic_replay": True,
        "o3_compatible_derived_geometry": True,
        "centroid_max_abs_angstrom": 0.0,
        "distance_diagonal_error_angstrom": 0.0,
        "distance_symmetry_error_angstrom": 0.0,
        "triangle_valid_by_euclidean_construction": True,
        "sampled_maximum_triangle_violation_angstrom": 0.0,
        "centered_gram_negative_eigenmass_fraction": 0.0,
        "adjacent_original_gate_pass": adjacent_error <= 1.0,
        "adjacent_reference_error_angstrom": adjacent_error,
        "radius_of_gyration_relative_reference_error": 0.1,
        "non_neighbor_clash_fraction": 0.01,
        "non_neighbor_clash_reference_excess": 0.0,
        "contact_density_6a_reference_error": 0.01,
        "contact_density_8a_reference_error": 0.01,
        "contact_density_10a_reference_error": 0.01,
        "coordinate_sha256": f"coordinate-{checkpoint}-{length}-{index}",
        "rigid_distance_sha256": f"rigid-{checkpoint}-{length}-{index}",
        "noise_identity_sha256": f"noise-{length}-{index}",
        "initial_noise_tensor_sha256": f"tensor-{length}-{index}",
        "reverse_draw_provenance_sha256": "deterministic-ddim-500",
    }


def _rows() -> list[dict]:
    return [
        _row(int(checkpoint["optimizer_update"]), int(length), index)
        for checkpoint in _config()["checkpoints"]
        for length in _config()["lengths"]
        for index in range(32)
    ]


def test_checkpoint_hash_verification_refuses_a_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    report = source / "report.json"
    protocol = source / "protocol.json"
    report.write_text(json.dumps({"optimizer_updates": 1000}))
    protocol.write_text(json.dumps({"optimizer_updates": 1000}))
    pilot_config = tmp_path / "pilot.yaml"
    pilot_config.write_text("version: fixture\n")
    correction_report = tmp_path / "correction-report.json"
    correction_report.write_text(
        json.dumps({"decision_audit": {"corrected_classification": "denoising_learned_but_sampling_not_learned"}})
    )
    correction_protocol = tmp_path / "correction-protocol.json"
    correction_protocol.write_text("{}")
    checkpoint = source / "step.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = {
        "phase3f": {
            "source_dir": str(source),
            "report_sha256": replication.sha256_file(report),
            "protocol_sha256": replication.sha256_file(protocol),
            "aggregate_fingerprint": replication.directory_fingerprint(source),
            "config_path": str(pilot_config),
            "config_sha256": replication.sha256_file(pilot_config),
            "decision_correction_report_path": str(correction_report),
            "decision_correction_report_sha256": replication.sha256_file(correction_report),
            "decision_correction_protocol_path": str(correction_protocol),
            "decision_correction_protocol_sha256": replication.sha256_file(correction_protocol),
        },
        "checkpoints": [{"optimizer_update": 250, "path": str(checkpoint), "sha256": "wrong"}],
    }
    before = replication.directory_fingerprint(source)
    with pytest.raises(ValueError, match="checkpoint_250"):
        replication.verify_prerequisites(config)
    assert replication.directory_fingerprint(source) == before


def test_seed_provenance_is_independent_unique_and_checkpoint_paired() -> None:
    config = _config()
    records = replication.paired_seed_records(config)
    assert len(records) == 160
    assert len({row["seed"] for row in records}) == 160
    assert not ({row["seed"] for row in records} & set(config["known_prior_sampling_seeds"]))
    first = replication.paired_seed_records(config)
    assert first == records
    for checkpoint in config["checkpoints"]:
        identities = [(row["length"], row["sample_index"], row["noise_identity_sha256"]) for row in records]
        assert identities == [
            (row["length"], row["sample_index"], row["noise_identity_sha256"])
            for row in replication.paired_seed_records(config)
        ]
        assert checkpoint["optimizer_update"] in {250, 750, 1000}


def test_initial_noise_provenance_is_deterministic() -> None:
    first = replication.initial_noise_provenance(64, 3700001, torch.device("cpu"))
    second = replication.initial_noise_provenance(64, 3700001, torch.device("cpu"))
    changed = replication.initial_noise_provenance(64, 3700002, torch.device("cpu"))
    assert first == second
    assert first["tensor_sha256"] != changed["tensor_sha256"]
    assert first["centroid_max_abs"] < 1e-6


def test_coordinate_metrics_have_exact_canonical_diagonal() -> None:
    coordinates = torch.tensor([[[0.0, 0.0, 0.0], [0.31, 0.07, 0.02], [0.63, -0.03, 0.05]]], dtype=torch.float32)
    config = _config()
    reference = {
        "adjacent_distance_mean_angstrom": 3.8,
        "radius_of_gyration_angstrom": 2.0,
        "non_neighbor_clash_fraction": 0.0,
        "contact_density_6a": 0.0,
        "contact_density_8a": 0.0,
        "contact_density_10a": 0.0,
    }
    row, _ = replication.coordinate_metrics(
        coordinates,
        checkpoint=250,
        length=3,
        sample_index=0,
        seed=5,
        reference=reference,
        config=config,
    )
    assert row["distance_diagonal_error_angstrom"] == 0.0
    assert row["distance_symmetry_error_angstrom"] == 0.0
    assert row["triangle_valid_by_euclidean_construction"] is True


def test_exact_480_count_conservation_and_block_rejection() -> None:
    rows = _rows()
    assert replication.validate_count_conservation(rows, _config()) == {
        "block_count": 15,
        "sample_count": 480,
        "samples_per_block": 32,
    }
    with pytest.raises(ValueError, match="count conservation contradiction"):
        replication.validate_count_conservation(rows[:-1], _config())
    paired = replication.validate_paired_provenance(rows, _config())
    assert paired["paired_identity_count"] == 160
    assert paired["checkpoint_count_per_identity"] == 3


def test_paired_provenance_rejects_checkpoint_specific_noise() -> None:
    rows = _rows()
    rows[-1]["initial_noise_tensor_sha256"] = "changed"
    with pytest.raises(ValueError, match="paired provenance contradiction"):
        replication.validate_paired_provenance(rows, _config())


def test_block_summary_preserves_hard_gate_and_bootstrap_fraction() -> None:
    config = _config()
    config["aggregation"]["bootstrap_replicates"] = 50
    rows = [_row(1000, 500, index, adjacent_error=1.1 if index == 31 else 0.5) for index in range(32)]
    for row in rows:
        row.update(
            {
                "centroid_max_abs_angstrom": 0.0,
                "centroid_rms_angstrom": 0.0,
                "distance_symmetry_error_angstrom": 0.0,
                "distance_diagonal_error_angstrom": 0.0,
                "sampled_maximum_triangle_violation_angstrom": 0.0,
                "centered_gram_negative_eigenmass_fraction": 0.0,
                "radius_of_gyration_absolute_reference_error_angstrom": 0.1,
                "contact_density_6a_reference_error": 0.01,
                "contact_density_8a_reference_error": 0.01,
                "contact_density_10a_reference_error": 0.01,
                "long_range_contact_density_8a": 0.01,
                "neighborhood_count_mean_8a": 2.0,
                "neighborhood_count_std_8a": 0.5,
                "artifact_path": f"sample-{row['sample_index']}.npz",
            }
        )
    summary = replication.summarize_block(rows, config)
    assert summary["original_all_record_gate_pass"] is False
    assert summary["adjacent_failure_count"] == 1
    assert summary["adjacent_pass_fraction"] == pytest.approx(31 / 32)
    assert len(summary["adjacent_pass_fraction_bootstrap_ci_95"]) == 2
    assert summary["one_record_removal_allows_hard_gate_pass"] is True
    assert set(summary["representative_samples"]) == {"median", "p95", "worst"}


def test_duplicate_detection_and_pareto_are_scalar_free() -> None:
    config = _config()
    rows = _rows()
    rows[1]["coordinate_sha256"] = rows[0]["coordinate_sha256"]
    rows[1]["rigid_distance_sha256"] = rows[0]["rigid_distance_sha256"]
    duplicate = replication.duplicate_summary(rows, config)
    assert duplicate["coordinate_duplicate_pair_count"] == 1
    assert duplicate["rigid_distance_duplicate_pair_count"] == 1
    pareto = replication.checkpoint_pareto(rows, config)
    assert "score" not in pareto
    assert pareto["latest_checkpoint_selected_automatically"] is False
    assert set(pareto["nondominated_checkpoints"]).issubset({250, 750, 1000})


@pytest.mark.parametrize(
    ("failures", "expected"),
    [
        ({}, "sampling_replication_verified_all_records"),
        ({"250:500": 1}, "sampling_replication_verified_except_isolated_tail"),
        ({"1000:500": 4}, "systematic_n500_sampling_deficiency"),
        ({"1000:500": 4, "1000:384": 4}, "broader_length_dependent_sampling_deficiency"),
    ],
)
def test_classification_preserves_original_gate(failures: dict[str, int], expected: str) -> None:
    config = _config()
    rows = _rows()
    summaries = {}
    for checkpoint in (250, 750, 1000):
        for length in config["lengths"]:
            key = f"{checkpoint}:{length}"
            count = 32
            failure_count = failures.get(key, 0)
            summaries[key] = {"count": count, "adjacent_failure_count": failure_count}
            selected = [row for row in rows if row["checkpoint_update"] == checkpoint and row["length"] == length]
            for row in selected[:failure_count]:
                row["adjacent_original_gate_pass"] = False
    pareto = {"nondominated_checkpoints": [250, 750]}
    assert replication.classify_replication(summaries, rows, pareto, config) == expected


def test_numerical_failure_precedes_scientific_classification() -> None:
    config = _config()
    rows = _rows()
    rows[0]["deterministic_replay"] = False
    summaries = {
        f"{checkpoint}:{length}": {"count": 32, "adjacent_failure_count": 0}
        for checkpoint in (250, 750, 1000)
        for length in config["lengths"]
    }
    assert replication.classify_replication(summaries, rows, {"nondominated_checkpoints": [1000]}, config) == (
        "numerical_or_replay_failure"
    )


def test_block_journal_restart_is_idempotent(tmp_path: Path) -> None:
    journal = {"250:64": {"row_count": 32, "path": "blocks/a.parquet", "sha256": "hash"}}
    replication._atomic_json(tmp_path / "block_journal.json", journal)
    assert replication._completed_blocks(tmp_path) == journal
    assert replication._completed_blocks(tmp_path) == journal


def test_uncommitted_partial_block_is_replayed_without_touching_committed_blocks(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "samples" / "step-0250"
    partial = checkpoint_dir / ".length-64.inprogress"
    orphan = checkpoint_dir / "length-64"
    committed = checkpoint_dir / "length-128"
    for directory in (partial, orphan, committed):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "evidence.txt").write_text(directory.name)
    final, temporary = replication._prepare_uncommitted_sample_block(tmp_path, checkpoint=250, length=64)
    assert not final.exists()
    assert temporary.is_dir()
    assert not list(temporary.iterdir())
    assert (committed / "evidence.txt").read_text() == "length-128"


def test_journaled_block_is_hash_and_row_count_verified(tmp_path: Path) -> None:
    config = _config()
    block = tmp_path / "blocks" / "step-0250-length-64.parquet"
    block.parent.mkdir()
    replication._atomic_parquet(block, [{"row": index} for index in range(32)], "zstd")
    journal = {
        "250:64": {
            "row_count": 32,
            "path": block.relative_to(tmp_path).as_posix(),
            "sha256": replication.sha256_file(block),
        }
    }
    replication.verify_completed_blocks(tmp_path, journal, config)
    block.write_bytes(block.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="block hash contradiction"):
        replication.verify_completed_blocks(tmp_path, journal, config)


def test_atomic_parquet_publication(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    replication._atomic_parquet(path, [{"sample": 1}, {"sample": 2}], "zstd")
    assert pq.read_table(path).num_rows == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_plan_is_no_model_no_optimizer_no_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    config["output_dir"] = str(tmp_path / "future")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(replication, "verify_prerequisites", lambda _: {"hashes": {"source": "verified"}})
    monkeypatch.setattr(
        replication,
        "EquivariantPairCoordinateUNet",
        lambda **_: pytest.fail("plan-only constructed a model"),
    )
    monkeypatch.setattr(replication.torch.cuda, "is_available", lambda: pytest.fail("plan-only touched CUDA"))
    monkeypatch.setattr(
        replication.CoordinateVPDiffusion,
        "sample",
        lambda *_args, **_kwargs: pytest.fail("plan-only sampled coordinates"),
    )
    result = replication.plan_sampling_replication(path)
    assert result["total_samples"] == 480
    assert result["model_created"] is False
    assert result["optimizer_created"] is False
    assert not Path(config["output_dir"]).exists()


def test_execution_module_contains_no_optimizer_or_backward_path() -> None:
    source = inspect.getsource(replication)
    assert "torch.optim" not in source
    assert ".backward(" not in source


def test_memory_enforcement_uses_peak_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        replication,
        "_memory",
        lambda _device: {
            "cuda_allocated_mib": 10.0,
            "cuda_reserved_mib": 20.0,
            "peak_cuda_allocated_mib": 6144.1,
            "peak_cuda_reserved_mib": 20.0,
        },
    )
    with pytest.raises(MemoryError, match="memory envelope exceeded"):
        replication._enforce_memory(torch.device("cuda"), _config())
