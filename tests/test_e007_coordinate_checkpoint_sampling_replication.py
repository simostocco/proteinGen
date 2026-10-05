from __future__ import annotations

import inspect
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
import yaml

import protein_distance_diffusion.evaluation.e007_coordinate_checkpoint_sampling_replication as replication


def _config() -> dict:
    return yaml.safe_load(Path("configs/e007_coordinate_checkpoint_sampling_replication_v1.yaml").read_text())


def _row(checkpoint: int, length: int, index: int, *, shift: float = 0.0) -> dict:
    config = _config()
    seed = int(config["sampling_seed_base"]) + config["lengths"].index(length) * 100_000 + index
    row = {
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
        "centroid_rms_angstrom": 0.0,
        "distance_symmetry_error_angstrom": 0.0,
        "distance_diagonal_error_angstrom": 0.0,
        "sampled_maximum_triangle_violation_angstrom": 0.0,
        "centered_gram_negative_eigenmass_fraction": 0.0,
        "rank3_residual_energy_fraction": shift,
        "adjacent_reference_error_angstrom": 0.5 + shift,
        "radius_of_gyration_absolute_reference_error_angstrom": 0.2 + shift,
        "radius_of_gyration_relative_reference_error": 0.1 + shift,
        "non_neighbor_clash_fraction": 0.01 + shift,
        "non_neighbor_clash_reference_excess": 0.0 + shift,
        "contact_density_6a_reference_error": 0.01 + shift,
        "contact_density_8a_reference_error": 0.01 + shift,
        "contact_density_10a_reference_error": 0.01 + shift,
        "long_range_contact_density_8a": 0.02 + shift,
        "neighborhood_count_mean_8a": 3.0 + shift,
        "neighborhood_count_std_8a": 0.5 + shift,
        "adjacent_original_gate_pass": 0.5 + shift <= 1.0,
        "coordinate_sha256": f"coordinate-{checkpoint}-{length}-{index}",
        "rigid_distance_sha256": f"rigid-{checkpoint}-{length}-{index}",
        "noise_identity_sha256": f"noise-{length}-{index}",
        "initial_noise_tensor_sha256": f"tensor-{length}-{index}",
        "reverse_draw_provenance_sha256": "deterministic-ddim-500",
    }
    return row


def _rows(shifts: dict[int, float] | None = None) -> list[dict]:
    shifts = shifts or {}
    return [
        _row(
            int(checkpoint["optimizer_update"]),
            int(length),
            index,
            shift=shifts.get(int(checkpoint["optimizer_update"]), 0.0),
        )
        for checkpoint in _config()["checkpoints"]
        for length in _config()["lengths"]
        for index in range(32)
    ]


def test_seed_namespace_is_unique_paired_and_disjoint() -> None:
    config = _config()
    records = replication.paired_seed_records(config)
    assert len(records) == 160
    assert len({row["seed"] for row in records}) == 160
    assert records == replication.paired_seed_records(config)
    for prior in config["known_prior_sampling_seed_ranges"]:
        assert not any(int(prior["minimum"]) <= row["seed"] <= int(prior["maximum"]) for row in records)


def test_phase3g_fingerprint_methods_are_reconciled_and_inventory_verifies() -> None:
    config = _config()
    source = Path(config["phase3g"]["source_dir"])
    assert replication.phase3g.directory_fingerprint(source) == (
        "b7a77cd477e8125317a6ff2b55cb5411a075efde8675a2bfdd49f99ce7fe7f9b"
    )
    assert replication.canonical_directory_fingerprint(source) == (
        "73b4a3cbb8615bad4d453700308260bf565d1d858810c388aded400c4b508d48"
    )
    verified = replication.verify_phase3g_artifact_inventory(source, config["phase3g"])
    assert verified == {
        "inventory_sha256": "d3892748ea9dcbddbb974694a84370bda894e0a0df4cdd1dcaf96e65c012727c",
        "inventory_aggregate_sha256": "33e2ff88b7587e745830a83cf0d0ec84bf76d3fa0541db8222c5e9c5a7f9f8fe",
        "entry_count": 499,
        "all_entries_verified": True,
    }


def test_exact_480_rows_and_160_paired_identities() -> None:
    rows = _rows()
    assert replication.validate_count_conservation(rows, _config()) == {
        "block_count": 15,
        "sample_count": 480,
        "samples_per_block": 32,
    }
    paired = replication.validate_paired_provenance(rows, _config())
    assert paired["paired_identity_count"] == 160
    assert paired["checkpoint_count_per_identity"] == 3
    with pytest.raises(ValueError, match="count conservation"):
        replication.validate_count_conservation(rows[:-1], _config())


def test_paired_identity_rejects_checkpoint_specific_noise() -> None:
    rows = _rows()
    rows[-1]["initial_noise_tensor_sha256"] = "different"
    with pytest.raises(ValueError, match="paired provenance contradiction"):
        replication.validate_paired_provenance(rows, _config())


def test_canonical_distance_and_rank3_metrics() -> None:
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.31, 0.07, 0.02], [0.63, -0.03, 0.05], [0.91, 0.04, -0.01]]],
        dtype=torch.float32,
    )
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
        checkpoint=7500,
        length=4,
        sample_index=0,
        seed=1,
        reference=reference,
        config=_config(),
    )
    assert row["distance_diagonal_error_angstrom"] == 0.0
    assert row["distance_symmetry_error_angstrom"] == 0.0
    assert row["rank3_residual_energy_fraction"] < 1e-20


def test_paired_comparison_sign_and_bootstrap_are_deterministic() -> None:
    config = _config()
    config["aggregation"]["bootstrap_replicates"] = 50
    rows = _rows({7500: 0.2, 9000: 0.1, 10000: 0.0})
    first = replication.paired_comparisons(rows, config)
    second = replication.paired_comparisons(rows, config)
    assert first == second
    comparison = first["comparisons"]["10000_minus_9000"]["global"]
    assert comparison["adjacent_reference_error_angstrom"]["mean"] == pytest.approx(-0.1)
    assert comparison["adjacent_reference_error_angstrom"]["fraction_candidate_better"] == 1.0


def test_all_record_fraction_and_one_record_removal() -> None:
    config = _config()
    config["aggregation"]["bootstrap_replicates"] = 25
    rows = [_row(10000, 500, index, shift=0.6 if index == 31 else 0.0) for index in range(32)]
    summary = replication.descriptive_summary(rows, config)
    assert summary["adjacent_all_record_gate_pass"] is False
    assert summary["adjacent_pass_fraction"] == pytest.approx(31 / 32)
    assert summary["one_record_removal"]["would_pass_after_removing_one_record"] is True
    assert summary["failure_examples"][0]["sample_index"] == 31


def test_pareto_dominance_and_non_dominance_have_no_scalar_score() -> None:
    dominated = replication.checkpoint_pareto(_rows({7500: 0.2, 9000: 0.0, 10000: 0.1}))
    assert 9000 in dominated["dominated_by"]["10000"]
    assert "score" not in dominated
    rows = _rows()
    for row in rows:
        if row["checkpoint_update"] == 9000:
            row["adjacent_reference_error_angstrom"] = 0.1
            row["radius_of_gyration_relative_reference_error"] = 0.3
        elif row["checkpoint_update"] == 10000:
            row["adjacent_reference_error_angstrom"] = 0.3
            row["radius_of_gyration_relative_reference_error"] = 0.1
    nondominated = replication.checkpoint_pareto(rows)
    assert {9000, 10000}.issubset(nondominated["nondominated_checkpoints"])


def test_decision_uses_per_sample_paired_fields_without_scalar_score() -> None:
    config = _config()
    config["aggregation"]["bootstrap_replicates"] = 20
    rows = _rows({7500: 0.2, 9000: 0.1, 10000: 0.0})
    summaries = {}
    for checkpoint in (7500, 9000, 10000):
        selected = [row for row in rows if row["checkpoint_update"] == checkpoint]
        summaries[str(checkpoint)] = replication.descriptive_summary(selected, config)
        for length in config["lengths"]:
            summaries[f"{checkpoint}:{length}"] = replication.descriptive_summary(
                [row for row in selected if row["length"] == length], config
            )
    comparisons = replication.paired_comparisons(rows, config)
    pareto = replication.checkpoint_pareto(rows)
    decision = replication.decision_summary(summaries, comparisons, pareto)
    assert decision["strict_dominance"]["10000_dominates_9000"] is True
    assert decision["paired_bootstrap_support"]["10000_over_9000"] is True
    assert decision["recommended_checkpoint"] == 10000
    assert "score" not in decision


def test_uncommitted_block_replay_preserves_other_block(tmp_path: Path) -> None:
    checkpoint = tmp_path / "samples" / "step-07500"
    partial = checkpoint / ".length-64.inprogress"
    orphan = checkpoint / "length-64"
    committed = checkpoint / "length-128"
    for directory in (partial, orphan, committed):
        directory.mkdir(parents=True)
        (directory / "evidence").write_text(directory.name)
    final, temporary = replication._prepare_uncommitted_sample_block(tmp_path, checkpoint=7500, length=64)
    assert not final.exists()
    assert temporary.is_dir() and not list(temporary.iterdir())
    assert (committed / "evidence").read_text() == "length-128"


def test_completed_block_is_immutable_hash_verified_and_contained(tmp_path: Path) -> None:
    config = _config()
    block = tmp_path / "blocks" / "step-07500-length-64.parquet"
    block.parent.mkdir()
    artifact_rows = []
    for value in range(32):
        artifact = tmp_path / "samples" / f"sample-{value}.npz"
        artifact.parent.mkdir(exist_ok=True)
        artifact.write_bytes(f"sample-{value}".encode())
        artifact_rows.append(
            {
                "artifact_path": artifact.relative_to(tmp_path).as_posix(),
                "artifact_sha256": replication.sha256_file(artifact),
            }
        )
    replication.phase3g._atomic_parquet(block, artifact_rows, "zstd")
    journal = {
        "7500:64": {
            "path": "blocks/step-07500-length-64.parquet",
            "row_count": 32,
            "sha256": replication.sha256_file(block),
        }
    }
    replication.verify_completed_blocks(tmp_path, journal, config)
    block.write_bytes(block.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash contradiction"):
        replication.verify_completed_blocks(tmp_path, journal, config)
    with pytest.raises(ValueError, match="escapes staging"):
        replication.verify_completed_blocks(
            tmp_path,
            {"7500:64": {"path": "../outside.parquet", "row_count": 32, "sha256": "x"}},
            config,
        )


def test_atomic_report_protocol_separation_contract(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    protocol = tmp_path / "protocol.json"
    replication.phase3g._atomic_json(report, {"scientific_results": [1, 2, 3]})
    replication.phase3g._atomic_json(protocol, {"report_sha256": replication.sha256_file(report)})
    assert report.read_bytes() != protocol.read_bytes()
    assert replication.sha256_file(report) != replication.sha256_file(protocol)


def test_plan_has_no_model_cuda_payload_scan_or_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    config["output_dir"] = str(tmp_path / "future")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(replication, "verify_prerequisites", lambda _: {"hashes": {"source": "verified"}})
    monkeypatch.setattr(replication, "EquivariantPairCoordinateUNet", lambda **_: pytest.fail("model constructed"))
    monkeypatch.setattr(replication.torch.cuda, "is_available", lambda: pytest.fail("CUDA touched"))
    monkeypatch.setattr(replication.phase3g, "_reference_distributions", lambda _: pytest.fail("payload scanned"))
    result = replication.plan_checkpoint_sampling_replication(path)
    assert result["total_samples"] == 480
    assert result["paired_noise_identity_count"] == 160
    assert result["model_created"] is False
    assert result["cuda_touched"] is False
    assert not Path(config["output_dir"]).exists()


def test_execution_module_has_no_optimizer_backward_or_training_path() -> None:
    source = inspect.getsource(replication)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "optimizer.step" not in source


def test_memory_gate_uses_process_peaks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        replication.phase3g,
        "_memory",
        lambda _device: {
            "cuda_allocated_mib": 1.0,
            "cuda_reserved_mib": 2.0,
            "peak_cuda_allocated_mib": 6144.1,
            "peak_cuda_reserved_mib": 2.0,
        },
    )
    with pytest.raises(MemoryError, match="memory envelope exceeded"):
        replication.phase3g._enforce_memory(torch.device("cuda"), _config())


def test_checkpoint_protected_hash_refusal(tmp_path: Path) -> None:
    source = tmp_path / "continuation"
    source.mkdir()
    report = source / "report.json"
    protocol = source / "protocol.json"
    report.write_text(json.dumps({"global_optimizer_update": 10000, "parameter_count": 7586505}))
    protocol.write_text(json.dumps({"global_optimizer_update": 10000}))
    config = tmp_path / "source.yaml"
    config.write_text("version: fixture\n")
    checkpoint = source / "step.pt"
    checkpoint.write_bytes(b"checkpoint")
    candidate = {
        "continuation": {
            "source_dir": str(source),
            "report_sha256": replication.sha256_file(report),
            "protocol_sha256": replication.sha256_file(protocol),
            "aggregate_fingerprint": replication.phase3g.directory_fingerprint(source),
            "config_path": str(config),
            "config_sha256": replication.sha256_file(config),
        },
        "phase3f": {
            "source_dir": str(source),
            "report_sha256": replication.sha256_file(report),
            "protocol_sha256": replication.sha256_file(protocol),
            "aggregate_fingerprint": replication.phase3g.directory_fingerprint(source),
            "config_path": str(config),
            "config_sha256": replication.sha256_file(config),
        },
        "phase3g": {
            "source_dir": str(source),
            "report_sha256": replication.sha256_file(report),
            "protocol_sha256": replication.sha256_file(protocol),
            "aggregate_fingerprint": replication.phase3g.directory_fingerprint(source),
            "config_path": str(config),
            "config_sha256": replication.sha256_file(config),
        },
        "checkpoints": [
            {
                "optimizer_update": 7500,
                "path": str(checkpoint),
                "sha256": "wrong",
                "metadata_path": str(protocol),
                "metadata_sha256": replication.sha256_file(protocol),
            }
        ],
    }
    before = replication.phase3g.directory_fingerprint(source)
    with pytest.raises(ValueError, match="checkpoint_7500"):
        replication.verify_prerequisites(candidate)
    assert replication.phase3g.directory_fingerprint(source) == before


def test_atomic_parquet_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    replication.phase3g._atomic_parquet(path, [{"value": 1}, {"value": 2}], "zstd")
    assert pq.read_table(path).num_rows == 2
    assert not list(tmp_path.glob("*.tmp"))
