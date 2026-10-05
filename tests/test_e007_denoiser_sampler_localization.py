from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as diagnostic
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion

CONFIG = Path("configs/e007_denoiser_sampler_localization_v1.yaml")
REAL_SIDECAR_ROOT = Path("/mnt/d/Users/Simone Stocco/proteinGen_audits/e006_rich_geometry_sidecars_v2")


def test_configuration_declares_bounded_contract() -> None:
    config = diagnostic.load_config(CONFIG)
    assert config["panel"]["samples_per_length"] == 8
    assert config["panel"]["reverse_seeds_per_length"] == 4
    assert len(config["one_step"]["timesteps"]) == 4
    assert config["reverse_trajectory"]["save_full_coordinate_tensors"] is False
    assert config["selected_checkpoint"]["sha256"] == (
        "eb445f0b39067b8a00a47db27f966a6db97dc0e30bf81078b34ea54f111c7d82"
    )
    assert config["recovery"]["expected_total_units"] == 340
    assert config["recovery"]["expected_total_forwards"] == 10320


def test_plan_only_is_output_scan_model_and_cuda_free(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text())
    config["output_dir"] = str(tmp_path / "must-not-exist")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("plan-only reached a payload path")

    monkeypatch.setattr(diagnostic, "select_validation_panel", forbidden)
    monkeypatch.setattr(diagnostic, "native_reference_envelopes", forbidden)
    result = diagnostic.plan_denoiser_sampler_localization(path)
    assert result["dataset_scan_performed"] is False
    assert result["model_created"] is False
    assert result["cuda_initialized"] is False
    assert result["reference_panel_records"] == 40
    assert result["reverse_seed_records"] == 20
    assert not Path(config["output_dir"]).exists()


def test_plan_refuses_mutated_phase3i_hash(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text())
    config["output_dir"] = str(tmp_path / "output")
    config["phase3i"]["report_sha256"] = "0" * 64
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="phase3i_report"):
        diagnostic.plan_denoiser_sampler_localization(path)
    assert not Path(config["output_dir"]).exists()


def test_panel_adapter_retains_exact_first_reviewed_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    config = diagnostic.load_config(CONFIG)
    observed: dict[str, object] = {}

    def fake_select(adapted: dict[str, object]):
        panel = adapted["panel"]
        observed.update(panel)
        rows = [
            {
                "sample_id": f"s{index}",
                "target_length": length,
                "actual_length": length if length != 500 or local < 4 else 499,
                "dataset_shard_path": f"validation/part-{index}.parquet",
                "shard_row_index": index,
            }
            for index, (length, local) in enumerate(
                (length, local) for length in panel["lengths"] for local in range(panel["samples_per_length"])
            )
        ]
        return rows, {"policy": panel["selection_version"]}

    monkeypatch.setattr(diagnostic.phase3i, "select_reference_panel", fake_select)
    rows, details = diagnostic.select_validation_panel(config)
    assert len(rows) == len({row["sample_id"] for row in rows}) == 40
    assert observed["maximum_reference_length_mismatch"] == 4
    assert details["policy"] == "e007_phase3i1_exact_then_same_stratum_nearest_length_v1"


def test_reverse_seed_panel_is_unique_deterministic_and_length_balanced() -> None:
    config = diagnostic.load_config(CONFIG)
    first = diagnostic.reverse_seed_records(config)
    second = diagnostic.reverse_seed_records(config)
    assert first == second
    assert len(first) == len({row["seed"] for row in first}) == 20
    assert {
        length: sum(row["requested_length"] == length for row in first) for length in config["panel"]["lengths"]
    } == {length: 4 for length in config["panel"]["lengths"]}


def _authoritative_fixture(
    tmp_path: Path,
    *,
    raw_updates: dict[str, object] | None = None,
    selection_updates: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    root = tmp_path / "sidecars"
    shard = root / "validation" / "part-000000.parquet"
    shard.parent.mkdir(parents=True)
    source = tmp_path / "protected" / "source.cif.gz"
    npz = tmp_path / "protected" / "sample.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    npz.write_bytes(b"npz")
    sequence = "ACDE"
    coordinates = [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.1, 0.0], [11.4, 0.1, 0.2]]
    raw = {
        "sample_id": "fixture_A",
        "split": "validation",
        "sequence": sequence,
        "ca_coordinates": coordinates,
        "ca_mask": [True] * 4,
        "chain_continuity_mask": [True] * 3,
        "chain_break_mask": [False] * 3,
        "source_path": str(source),
        "source_sha256": diagnostic.sha256_file(source),
        "npz_path": str(npz),
        "npz_sha256": diagnostic.sha256_file(npz),
        "schema_version": "e006_rich_geometry_sidecar_v2",
    }
    raw.update(raw_updates or {})
    pq.write_table(pa.Table.from_pylist([raw]), shard)
    selection = {
        "sample_id": "fixture_A",
        "split": "validation",
        "coordinate_accepted": True,
        "length": 4,
        "target_length": 4,
        "actual_length": 4,
        "signed_length_mismatch": 0,
        "absolute_length_mismatch": 0,
        "match_type": "exact_length",
        "length_stratum": "1-64",
        "clean_validation_manifest_member": True,
        "source_path": str(source),
        "dataset_shard_path": "validation/part-000000.parquet",
        "shard_row_index": 0,
        "selection_rank": "rank",
        "selection_version": "fixture_v1",
    }
    selection.update(selection_updates or {})
    config = {
        "dataset": {"root": str(root)},
        "clean_validation": {"manifest_sha256": "c" * 64},
        "panel": {"lengths": [4], "samples_per_length": 1},
    }
    return config, selection


def test_authoritative_reconstruction_supplies_complete_canonical_row(tmp_path: Path) -> None:
    config, selection = _authoritative_fixture(tmp_path)
    rows, diagnostics = diagnostic.reconstruct_authoritative_panel(config, [selection], source_config=config)
    canonical = rows[0]["canonical_row"]
    assert canonical["accepted_contiguous_single_chain"] is True
    assert canonical["coordinate_acceptance_reasons"] == ()
    assert canonical["sequence_length"] == 4
    assert canonical["coordinates"].shape == (4, 3)
    assert diagnostics["accepted_row_count"] == 1


def test_incomplete_panel_record_cannot_reach_collator(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    module = importlib.import_module("protein_distance_diffusion.training.e007_coordinate_real_loader_smoke")
    monkeypatch.setattr(module, "prepare_coordinate_batch", forbidden)
    with pytest.raises(ValueError, match="incomplete canonical collator row"):
        diagnostic._prepared_reference(diagnostic.load_config(CONFIG), {"sample_id": "incomplete"})
    assert called is False


def test_missing_or_false_authoritative_acceptance_fails_before_collation() -> None:
    config = diagnostic.load_config(CONFIG)
    base = {
        "sample_id": "x",
        "sequence_length": 1,
        "coordinates": torch.zeros((1, 3)),
        "residue_mask": torch.ones(1, dtype=torch.bool),
        "chain_continuity_mask": torch.ones(0, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="incomplete canonical collator row"):
        diagnostic._prepared_reference(config, base)
    with pytest.raises(ValueError, match="is not accepted"):
        diagnostic._prepared_reference(config, {**base, "accepted_contiguous_single_chain": False})


@pytest.mark.parametrize(
    ("raw_updates", "selection_updates", "message"),
    [
        ({"sample_id": "other"}, {}, "sample-ID contradiction"),
        ({"npz_sha256": "0" * 64}, {}, "protected input hash contradiction"),
        ({}, {"actual_length": 3}, "length contradiction"),
        ({"ca_mask": [True, False, True, True]}, {}, "authoritative coordinate acceptance is false"),
    ],
)
def test_authoritative_reconstruction_rejects_contradictions(
    tmp_path: Path,
    raw_updates: dict[str, object],
    selection_updates: dict[str, object],
    message: str,
) -> None:
    config, selection = _authoritative_fixture(
        tmp_path,
        raw_updates=raw_updates,
        selection_updates=selection_updates,
    )
    with pytest.raises(ValueError, match=message):
        diagnostic.reconstruct_authoritative_panel(config, [selection], source_config=config)


def test_relocation_case_collision_uses_exact_authoritative_hash(tmp_path: Path) -> None:
    config, selection = _authoritative_fixture(tmp_path)
    root = Path(config["dataset"]["root"])
    raw = pq.read_table(root / selection["dataset_shard_path"]).to_pylist()[0]
    recorded_root = tmp_path / "recorded"
    relocated_root = tmp_path / "relocated"
    exact = relocated_root / "Data" / "sample.npz"
    collision = relocated_root / "Data" / "Sample.npz"
    exact.parent.mkdir(parents=True)
    exact.write_bytes(b"authoritative")
    collision.write_bytes(b"different")
    raw["npz_path"] = str(recorded_root / "Data" / "sample.npz")
    raw["npz_sha256"] = diagnostic.sha256_file(exact)
    pq.write_table(pa.Table.from_pylist([raw]), root / selection["dataset_shard_path"])
    config["dataset"]["protected_input_relocations"] = [
        {"recorded_root": str(recorded_root), "verification_root": str(relocated_root)}
    ]
    rows, _diagnostics = diagnostic.reconstruct_authoritative_panel(config, [selection], source_config=config)
    assert rows[0]["provenance"]["npz_resolved_path"] == str(exact.resolve())
    assert rows[0]["provenance"]["npz_sha256"] != diagnostic.sha256_file(collision)


def test_panel_schema_failure_precedes_model_cuda_and_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = yaml.safe_load(CONFIG.read_text())
    output = tmp_path / "result"
    config["output_dir"] = str(output)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(diagnostic, "verify_prerequisites", lambda *_args, **_kwargs: {"hashes": {}})
    monkeypatch.setattr(diagnostic, "select_validation_panel", lambda *_args: ([{"sample_id": "x"}], {}))
    monkeypatch.setattr(
        diagnostic,
        "reconstruct_authoritative_panel",
        lambda *_args: (_ for _ in ()).throw(ValueError("schema failure")),
    )
    monkeypatch.setattr(
        diagnostic,
        "_load_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("model constructed")),
    )
    with pytest.raises(ValueError, match="schema failure"):
        diagnostic.diagnose_denoiser_sampler_localization(config_path)
    assert not output.exists()
    assert not output.with_name(f".{output.name}.inprogress").exists()


@pytest.mark.skipif(not REAL_SIDECAR_ROOT.is_dir(), reason="definitive E006 sidecar dataset is unavailable")
def test_real_selected_40_row_panel_has_complete_canonical_schema() -> None:
    config = diagnostic.load_config(CONFIG)
    panel, _selection = diagnostic.select_validation_panel(config)
    rows, evidence = diagnostic.reconstruct_authoritative_panel(config, panel)
    assert len(rows) == evidence["validated_row_count"] == evidence["accepted_row_count"] == 40
    assert all(item["canonical_row"]["accepted_contiguous_single_chain"] is True for item in rows)


def _synthetic_unit(order: int, *, trajectory: bool = False) -> dict[str, object]:
    unit = {
        "order": order,
        "unit_id": f"{'trajectory' if trajectory else 'one'}:{order}",
        "unit_kind": "reverse_trajectory" if trajectory else "one_step",
        "orientation": "trajectory" if trajectory else "native",
        "paired_noise_identity_sha256": f"noise-{order}",
        "expected_row_count": 16 if trajectory else 3,
        "forward_count": 500 if trajectory else 1,
    }
    if trajectory:
        unit.update({"requested_length": 64, "sample_index": order, "seed": 1000 + order})
    return unit


def _synthetic_config() -> dict[str, object]:
    return {
        "numerics": {
            "allow_matmul_tf32": False,
            "allow_cudnn_tf32": False,
            "deterministic_algorithms": True,
            "autocast_enabled": False,
            "equivariance_policy": "strict_o3_v1",
        },
        "publication": {"parquet_compression": "zstd"},
        "recovery": {"journal_version": "e007_phase3i1_block_journal_v1"},
    }


def _synthetic_rows(count: int, value: int) -> list[dict[str, object]]:
    return [
        {
            "value": value,
            "row_index": index,
            "representation": "predicted_x0",
            "requested_length": 64,
            "timestep": 25,
            "sample_id": "sample",
            "timestep_name": "low",
            "noise_seed": 1,
            "coordinate_v_mse": 0.1,
            "sample_index": 0,
            "seed": 1,
            "noise_identity_sha256": "noise",
        }
        for index in range(count)
    ]


def test_resume_after_complete_one_step_unit(tmp_path: Path) -> None:
    unit = _synthetic_unit(0)
    record = diagnostic.commit_unit_artifact(tmp_path, unit, _synthetic_rows(3, 7), _synthetic_config())
    verified = diagnostic.verify_journaled_units(tmp_path, [unit], _synthetic_config())
    assert verified == [record]
    assert diagnostic.progress_from_records(verified, [unit])["completed_forwards"] == 1
    assert diagnostic.cleanup_uncommitted_units(tmp_path, verified) == []


def test_unit_commit_fsyncs_artifact_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fsynced: list[Path] = []
    original = diagnostic._fsync_directory

    def record_fsync(path: Path) -> None:
        fsynced.append(path)
        original(path)

    monkeypatch.setattr(diagnostic, "_fsync_directory", record_fsync)
    diagnostic.commit_unit_artifact(
        tmp_path,
        _synthetic_unit(0),
        _synthetic_rows(3, 7),
        _synthetic_config(),
    )
    assert tmp_path / "units" in fsynced
    assert tmp_path in fsynced


def test_resume_after_complete_trajectory_is_one_boundary(tmp_path: Path) -> None:
    unit = _synthetic_unit(0, trajectory=True)
    diagnostic.commit_unit_artifact(tmp_path, unit, _synthetic_rows(16, 9), _synthetic_config())
    verified = diagnostic.verify_journaled_units(tmp_path, [unit], _synthetic_config())
    progress = diagnostic.progress_from_records(verified, [unit])
    assert progress["completed_trajectories"] == 1
    assert progress["completed_forwards"] == 500


def test_incomplete_trajectory_is_removed_and_replayed(tmp_path: Path) -> None:
    unit = _synthetic_unit(0, trajectory=True)
    orphan = tmp_path / diagnostic._unit_artifact_relative(unit)
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"partial trajectory")
    temporary = orphan.with_name(f".{orphan.name}.123.tmp")
    temporary.write_bytes(b"partial parquet")
    removed = diagnostic.cleanup_uncommitted_units(tmp_path, [])
    assert sorted(removed) == sorted(
        [orphan.relative_to(tmp_path).as_posix(), temporary.relative_to(tmp_path).as_posix()]
    )
    assert not orphan.exists()
    assert not temporary.exists()


def test_corrupt_journaled_artifact_is_rejected(tmp_path: Path) -> None:
    unit = _synthetic_unit(0)
    record = diagnostic.commit_unit_artifact(tmp_path, unit, _synthetic_rows(3, 1), _synthetic_config())
    artifact = tmp_path / record["path"]
    with artifact.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="size contradiction"):
        diagnostic.verify_journaled_units(tmp_path, [unit], _synthetic_config())


def test_journal_count_and_order_conflicts_are_rejected(tmp_path: Path) -> None:
    units = [_synthetic_unit(0), _synthetic_unit(1)]
    diagnostic.commit_unit_artifact(tmp_path, units[0], _synthetic_rows(3, 1), _synthetic_config())
    journal = diagnostic._journal_path(tmp_path)
    record = json.loads(journal.read_text().splitlines()[0])
    record["unit_order"] = 1
    journal.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="out of order"):
        diagnostic.verify_journaled_units(tmp_path, units, _synthetic_config())


def test_journal_row_count_conflict_is_rejected(tmp_path: Path) -> None:
    unit = _synthetic_unit(0)
    diagnostic.commit_unit_artifact(tmp_path, unit, _synthetic_rows(3, 1), _synthetic_config())
    journal = diagnostic._journal_path(tmp_path)
    record = json.loads(journal.read_text().splitlines()[0])
    record["row_count"] = 2
    journal.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="row-count contradiction"):
        diagnostic.verify_journaled_units(tmp_path, [unit], _synthetic_config())


def test_duplicate_journal_entry_is_rejected(tmp_path: Path) -> None:
    units = [_synthetic_unit(0), _synthetic_unit(1)]
    diagnostic.commit_unit_artifact(tmp_path, units[0], _synthetic_rows(3, 1), _synthetic_config())
    journal = diagnostic._journal_path(tmp_path)
    first = json.loads(journal.read_text().splitlines()[0])
    duplicate = {**first, "sequence": 1, "unit_order": 1}
    with journal.open("a") as stream:
        stream.write(json.dumps(duplicate) + "\n")
    with pytest.raises(ValueError, match="unit conflicts"):
        diagnostic.verify_journaled_units(tmp_path, units, _synthetic_config())


def test_journal_path_escape_is_rejected(tmp_path: Path) -> None:
    unit = _synthetic_unit(0)
    diagnostic.commit_unit_artifact(tmp_path, unit, _synthetic_rows(3, 1), _synthetic_config())
    journal = diagnostic._journal_path(tmp_path)
    record = json.loads(journal.read_text().splitlines()[0])
    record["path"] = "../escape.parquet"
    journal.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="escapes staging"):
        diagnostic.verify_journaled_units(tmp_path, [unit], _synthetic_config())


def test_resumed_unit_artifacts_equal_uninterrupted_artifacts(tmp_path: Path) -> None:
    units = [_synthetic_unit(0), _synthetic_unit(1)]
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    for unit in units:
        diagnostic.commit_unit_artifact(
            uninterrupted, unit, _synthetic_rows(3, int(unit["order"])), _synthetic_config()
        )
    diagnostic.commit_unit_artifact(resumed, units[0], _synthetic_rows(3, 0), _synthetic_config())
    prefix = diagnostic.verify_journaled_units(resumed, units, _synthetic_config())
    assert len(prefix) == 1
    diagnostic.commit_unit_artifact(resumed, units[1], _synthetic_rows(3, 1), _synthetic_config())
    resumed_records = diagnostic.verify_journaled_units(resumed, units, _synthetic_config())
    uninterrupted_records = diagnostic.verify_journaled_units(uninterrupted, units, _synthetic_config())
    assert [record["sha256"] for record in resumed_records] == [record["sha256"] for record in uninterrupted_records]


def test_work_unit_and_forward_count_conservation() -> None:
    config = diagnostic.load_config(CONFIG)
    panel = [
        {
            "sample_id": f"sample-{index}",
            "target_length": length,
            "actual_length": length,
        }
        for index, length in enumerate(length for length in config["panel"]["lengths"] for _ in range(8))
    ]
    units = diagnostic.build_work_units(panel, config)
    assert len(units) == 340
    assert sum(unit["unit_kind"] == "one_step" for unit in units) == 320
    assert sum(unit["unit_kind"] == "reverse_trajectory" for unit in units) == 20
    assert sum(unit["forward_count"] for unit in units) == 10320
    assert len({unit["unit_id"] for unit in units}) == 340


def test_monitor_is_read_only_for_staging_and_final(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text())
    output = tmp_path / "result"
    config["output_dir"] = str(output)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    staging = output.with_name(f".{output.name}.inprogress")
    staging.mkdir()
    heartbeat = staging / "heartbeat.json"
    heartbeat.write_text('{"status":"running","completed_units":7}\n')
    before = diagnostic.sha256_file(heartbeat)
    result = diagnostic.monitor_denoiser_sampler_localization(config_path)
    assert result["source"] == "staging"
    assert result["completed_units"] == 7
    assert diagnostic.sha256_file(heartbeat) == before
    assert list(staging.iterdir()) == [heartbeat]


def test_heartbeat_reports_only_verified_boundary_and_execution_flags() -> None:
    units = [_synthetic_unit(0), _synthetic_unit(1, trajectory=True)]
    records = [
        {
            "unit_kind": "one_step",
            "forward_count": 1,
            "path": "units/0000.parquet",
        }
    ]
    payload = diagnostic._heartbeat_payload(
        status="running",
        records=records,
        units=units,
        current_unit=units[1],
        execution={
            "model_created": True,
            "cuda_initialized": True,
            "forward_performed": True,
            "sampling_performed": False,
        },
    )
    assert payload["completed_units"] == 1
    assert payload["completed_forwards"] == 1
    assert payload["latest_verified_artifact"] == "units/0000.parquet"
    assert payload["current_trajectory"] == {
        "requested_length": units[1].get("requested_length"),
        "seed": units[1].get("seed"),
    }
    assert payload["model_created"] is True
    assert payload["optimizer_updates"] == 0
    assert payload["authorizes_training"] is False


def test_paired_reflection_uses_reflected_noise_and_preserves_global_pairing() -> None:
    diffusion = CoordinateVPDiffusion(500)
    clean = torch.tensor([[[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.0, 0.0, 0.0]]])
    mask = torch.ones((1, 3), dtype=torch.bool)
    timestep = torch.tensor([250])
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    generator = torch.Generator().manual_seed(7)
    native, reflected = diagnostic.paired_reflection_batch(diffusion, clean, mask, timestep, generator, reflection)
    reflected_clean, reflected_noisy, reflected_target = reflected
    assert torch.equal(reflected_clean, clean @ reflection.T)
    assert torch.allclose(reflected_noisy, native.noisy_coordinates @ reflection.T, atol=1e-7, rtol=0)
    assert torch.allclose(reflected_target, native.coordinate_v_target @ reflection.T, atol=1e-7, rtol=0)


def test_reflection_is_orthogonal_and_reverses_pseudo_volume() -> None:
    reflection = np.diag([-1.0, 1.0, 1.0])
    assert np.allclose(reflection.T @ reflection, np.eye(3))
    assert np.linalg.det(reflection) == -1
    points = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=float)
    vectors = np.diff(points, axis=0)
    volume = np.dot(np.cross(vectors[0], vectors[1]), vectors[2])
    reflected_vectors = np.diff(points @ reflection.T, axis=0)
    reflected = np.dot(np.cross(reflected_vectors[0], reflected_vectors[1]), reflected_vectors[2])
    assert reflected == -volume


def test_envelope_status_and_trajectory_entry_exit_intervals() -> None:
    envelope = {field: {"lower": 0.0, "upper": 1.0} for field in diagnostic.ENVELOPE_FIELDS}
    records = []
    for timestep, value in ((499, 2.0), (250, 0.5), (0, 1.5)):
        row = {field: value for field in diagnostic.ENVELOPE_FIELDS}
        row["timestep"] = timestep
        records.append(row)
    transitions = diagnostic.trajectory_transitions(records, envelope)
    item = transitions["discontinuity_fraction"]
    assert item["first_entry_interval"] == [499, 250]
    assert item["first_exit_interval"] == [250, 0]
    assert item["final_inside"] is False


def test_controlled_length_slopes_are_separate_by_timestep() -> None:
    rows = [
        {
            "timestep": timestep,
            "representation": "predicted_x0",
            "requested_length": length,
            "error": length * slope,
        }
        for timestep, slope in ((25, 0.01), (250, -0.02))
        for length in (64, 128, 256, 384, 500)
    ]
    slopes = diagnostic.controlled_length_slopes(rows, "error")
    assert slopes["25"]["slope_per_residue"] == pytest.approx(0.01)
    assert slopes["250"]["slope_per_residue"] == pytest.approx(-0.02)


def test_v_mse_correlation_uses_no_scalar_decision_score() -> None:
    assert diagnostic.pearson_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert diagnostic.pearson_correlation([1, 1], [2, 3]) is None
    categories = diagnostic.classify_localization(
        denoiser_local_failure=True,
        sampler_accumulation=True,
        chirality_limitation=True,
        long_length_failure=True,
    )
    assert categories == [
        "chirality_symmetry_limitation",
        "combined_denoiser_and_sampler_failure",
        "long_length_scaling_failure",
    ]


@pytest.mark.parametrize(
    ("denoiser", "sampler", "expected"),
    [
        (True, False, "denoiser_local_geometry_failure"),
        (False, True, "reverse_sampler_error_accumulation"),
        (False, False, "bounded_diagnostic_inconclusive"),
    ],
)
def test_decision_branches(denoiser: bool, sampler: bool, expected: str) -> None:
    result = diagnostic.classify_localization(
        denoiser_local_failure=denoiser,
        sampler_accumulation=sampler,
        chirality_limitation=False,
        long_length_failure=False,
    )
    assert expected in result


def test_summary_localizes_all_three_structural_domains() -> None:
    envelope = {
        "by_requested_length": {
            "64": {
                **{field: {"lower": -1.0, "upper": 1.0} for field in diagnostic.ENVELOPE_FIELDS},
                "reference_count": 32,
            }
        }
    }
    base = {field: 0.0 for field in diagnostic.ENVELOPE_FIELDS}
    base.update(
        {
            "sample_id": "sample",
            "requested_length": 64,
            "timestep": 25,
            "timestep_name": "low",
            "coordinate_v_mse": 0.2,
            "coordinate_rmse_to_source_angstrom": 0.3,
            "adjacent_distance_rmse_to_native_angstrom": 0.4,
        }
    )
    source = {**base, "representation": "source_x0", "signed_pseudo_dihedral_mean": 0.5}
    predicted = {**base, "representation": "predicted_x0", "contact_order_8a": 5.0}
    reflected = {
        **base,
        "representation": "reflected_predicted_x0",
        "reflection_equivariance_relative_error": 1e-6,
    }
    trajectory = [{**base, "representation": "reverse_state", "timestep": 0, "contact_order_8a": 5.0}]
    result = diagnostic._summarize([source, predicted, reflected], trajectory, envelope)
    assert result["denoiser_outside_native_envelope_fraction_by_domain"]["global_topology"] == 1.0
    assert result["final_sampler_outside_native_envelope_fraction_by_domain"]["global_topology"] == 1.0
    assert "combined_denoiser_and_sampler_failure" in result["decision_categories"]


def test_plan_module_has_no_optimizer_or_backward_path() -> None:
    source = Path(diagnostic.__file__).read_text()
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "optimizer.step" not in source


def test_lazy_import_boundary_keeps_torch_out_of_plan_module(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "protein_distance_diffusion.evaluation.e007_denoiser_sampler_localization"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    original = sys.modules.pop("torch", None)
    try:
        importlib.import_module(module_name)
        assert "torch" not in sys.modules
    finally:
        if original is not None:
            sys.modules["torch"] = original


def test_atomic_json_publishes_distinct_payload(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    protocol = tmp_path / "protocol.json"
    diagnostic._atomic_json(report, {"scientific": [1, 2, 3]})
    diagnostic._atomic_json(protocol, {"contract": "non_authorizing"})
    assert json.loads(report.read_text()) != json.loads(protocol.read_text())
    assert diagnostic.sha256_file(report) != diagnostic.sha256_file(protocol)


def test_non_authorization_contract_is_complete() -> None:
    assert diagnostic.NON_AUTHORIZING["optimizer_updates"] == 0
    assert diagnostic.NON_AUTHORIZING["checkpoint_modified"] is False
    assert diagnostic.NON_AUTHORIZING["authorizes_phase3j"] is False
    assert not any(value for key, value in diagnostic.NON_AUTHORIZING.items() if key.startswith("authorizes_"))
