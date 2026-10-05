from __future__ import annotations

import copy
import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import protein_distance_diffusion.data.rich_geometry_sidecars as sidecar_module
from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.rich_geometry_sidecars import (
    JOURNAL_SCHEMA_VERSION,
    PERFORMANCE_STAGE_NAMES,
    SIDECAR_SCHEMA_VERSION,
    TORSION_CONVENTION_VERSION,
    TORSION_NEUTRAL_SIN_COS,
    ConstructionJournal,
    EvaluationOutcome,
    InputUnit,
    StageObservation,
    StageProfiler,
    _assign_physical_split,
    _input_units,
    _ordered_evaluations,
    _physical_split,
    _pilot_ids,
    _sidecar_row,
    _validate_cuda_features_from_shards,
    _verify_observed_inputs,
    _verify_reopened_sidecar,
    _write_part,
    attest_phase0,
    construct_sidecars,
    derive_backbone_torsions,
    eligibility_schema,
    evaluate_sample,
    exclusion_schema,
    rich_geometry_schema,
    run_performance_benchmark,
    validate_sidecar_config,
    verify_protected_inputs,
    verify_sidecar_dataset,
)
from protein_distance_diffusion.evaluation.e006_geometry_source_audit import (
    COORDINATE_ANCHOR_POLICY_VERSION,
    BackboneResidue,
    load_npz_geometry_evidence,
    sha256_file,
)


def _phase0(tmp_path: Path, *, status: str = "completed", authorized: bool = True) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    protected = tmp_path / "protected.txt"
    protected.write_text("immutable")
    eligibility = tmp_path / "phase0_eligibility.parquet"
    pq.write_table(pa.table({"sample_id": ["sample"]}), eligibility)
    hashes = {str(protected): sha256_file(protected), str(eligibility): sha256_file(eligibility)}
    protocol = {
        "status": status,
        "schema_version": "e006_rich_geometry_source_audit_v6",
        "phase1_authorization": {"authorized": authorized},
        "authorizes_phase1": authorized,
        "dataset_inputs_unchanged": True,
        "input_hashes_before": hashes,
        "input_hashes_after": hashes,
        "eligibility_manifest": {
            "path": str(eligibility),
            "sha256": sha256_file(eligibility),
            "row_count": 1,
        },
    }
    path = tmp_path / "phase0_protocol.json"
    path.write_text(json.dumps(protocol))
    return path


def _config(tmp_path: Path, phase0: Path) -> dict[str, object]:
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "phase0_protocol": str(phase0),
        "phase0_protocol_sha256": sha256_file(phase0),
        "pairing_dataset": str(tmp_path / "pairing"),
        "processed_manifest": str(tmp_path / "processed.parquet"),
        "audit_provenance": str(tmp_path / "audit"),
        "normalization_source": str(tmp_path / "normalization.json"),
        "output_dir": str(tmp_path / "full"),
        "pilot_output_dir": str(tmp_path / "pilot"),
        "pilot_sample_count": 2,
        "seed": 6006,
        "shard_size": 4,
        "row_group_size": 2,
        "maximum_failure_examples": 2,
        "maximum_rss_mib": 1024,
    }


def test_phase0_attestation_checks_authorization_artifacts_and_protected_inputs(tmp_path: Path) -> None:
    protocol_path = _phase0(tmp_path)
    protocol = attest_phase0(protocol_path, expected_sha256=sha256_file(protocol_path))
    assert verify_protected_inputs(protocol) == protocol["input_hashes_before"]
    Path(next(iter(protocol["input_hashes_before"]))).write_text("changed")
    with pytest.raises(ValueError, match="Protected Phase-0 input changed"):
        verify_protected_inputs(protocol)


def test_phase0_attestation_recursively_verifies_referenced_artifacts(tmp_path: Path) -> None:
    protocol_path = _phase0(tmp_path)
    protocol = json.loads(protocol_path.read_text())
    nested = tmp_path / "nested.json"
    nested.write_text("{}")
    protocol["derived_outputs"] = {"reports": [{"path": str(nested), "sha256": sha256_file(nested)}]}
    protocol_path.write_text(json.dumps(protocol))
    attest_phase0(protocol_path, expected_sha256=sha256_file(protocol_path))
    nested.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="referenced_artifact_hash_mismatch"):
        attest_phase0(protocol_path, expected_sha256=sha256_file(protocol_path))


@pytest.mark.parametrize(
    "status,authorized,error",
    [("running", True, "protocol_not_completed"), ("completed", False, "phase1_not_authorized")],
)
def test_phase0_attestation_refuses_incomplete_or_unauthorized(
    tmp_path: Path, status: str, authorized: bool, error: str
) -> None:
    path = _phase0(tmp_path, status=status, authorized=authorized)
    with pytest.raises(ValueError, match=error):
        attest_phase0(path)


def test_sidecar_schema_is_linear_nested_numeric_and_has_no_dense_pairs() -> None:
    schema = rich_geometry_schema()
    assert schema.metadata[b"schema_version"].decode() == SIDECAR_SCHEMA_VERSION
    assert schema.field("ca_coordinates").type.value_type.list_size == 3
    assert pa.types.is_boolean(schema.field("ca_mask").type.value_type)
    assert not any("distance_matrix" in field.name or field.name.startswith("pair_") for field in schema)
    assert schema.field("phi_sin_cos").type.value_type.list_size == 2
    assert pa.types.is_struct(schema.field("selected_atom_conformers").type.value_type)


def test_config_bounds_and_plan_only_create_no_dataset(tmp_path: Path) -> None:
    phase0 = _phase0(tmp_path)
    config = _config(tmp_path, phase0)
    validate_sidecar_config(config)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    result = construct_sidecars(config_path, mode="plan-only")
    assert result["status"] == "planned"
    assert result["dense_pair_features_stored"] is False
    assert not Path(config["output_dir"]).exists()
    invalid = {**config, "shard_size": 4097}
    with pytest.raises(ValueError, match="4096"):
        validate_sidecar_config(invalid)
    with pytest.raises(ValueError, match="worker_count"):
        validate_sidecar_config({**config, "worker_count": 5})
    with pytest.raises(ValueError, match="worker_queue_size"):
        validate_sidecar_config({**config, "worker_count": 2, "worker_queue_size": 1})
    with pytest.raises(ValueError, match="maximum_cuda_allocated_mib"):
        validate_sidecar_config({**config, "maximum_cuda_allocated_mib": 6145})


@pytest.mark.parametrize("override", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_runtime_rss_override_rejects_nonpositive_or_nonfinite_values(
    tmp_path: Path,
    override: float,
) -> None:
    config = _config(tmp_path, _phase0(tmp_path))
    config["maximum_rss_mib"] = 2048
    with pytest.raises(ValueError, match="finite and positive"):
        sidecar_module._runtime_rss_limits(config, override)


def test_runtime_rss_override_cannot_lower_configured_limit(tmp_path: Path) -> None:
    config = _config(tmp_path, _phase0(tmp_path))
    config["maximum_rss_mib"] = 2048
    with pytest.raises(ValueError, match="at least the configured"):
        sidecar_module._runtime_rss_limits(config, 1024)
    assert sidecar_module._runtime_rss_limits(config, None) == (2048.0, 2048.0, False)
    assert sidecar_module._runtime_rss_limits(config, 4096) == (2048.0, 4096.0, True)


def test_production_config_is_phase1_v2_and_separates_pilot_output() -> None:
    config = load_yaml("configs/e006_rich_geometry_sidecars.yaml")
    validate_sidecar_config(config)
    assert config["schema_version"] == SIDECAR_SCHEMA_VERSION
    assert config["phase0_protocol"].endswith("source_audit_v6/protocol.json")
    assert config["pilot_output_dir"].endswith("sidecar_pilot_v5")
    assert config["pilot_sample_count"] == 512
    assert config["output_dir"] == ("/mnt/d/Users/Simone Stocco/proteinGen_audits/e006_rich_geometry_sidecars_v2")
    assert Path(config["pilot_output_dir"]) != Path(config["output_dir"])
    assert config["shard_size"] <= 4096
    assert config["worker_count"] == 1
    assert config["worker_queue_size"] == 2
    assert config["feature_backend"] == "cpu"
    assert config["maximum_cuda_allocated_mib"] == 6144
    benchmark = config["performance_benchmark"]
    assert benchmark["version"] == 2
    assert benchmark["output_root"].endswith("sidecar_performance_benchmark_v2")
    assert benchmark["panel_sample_count"] == 512
    assert benchmark["maximum_rss_mib"] == 4096
    assert benchmark["minimum_end_to_end_speedup_fraction"] == 0.20


def test_journal_resume_contract_and_completed_units(tmp_path: Path) -> None:
    path = tmp_path / "construction_journal.sqlite"
    journal = ConstructionJournal(path, config_hash="config", resume=False)
    journal.connection.execute(
        "INSERT INTO completed_units VALUES (?,?,?,?,?)",
        ("unit", "eligible_train", "train", 0, 2),
    )
    journal.connection.commit()
    assert journal.get("schema_version") == JOURNAL_SCHEMA_VERSION
    journal.connection.close()
    resumed = ConstructionJournal(path, config_hash="config", resume=True)
    assert resumed.completed("unit") is True
    resumed.connection.close()
    with pytest.raises(ValueError, match="configuration hash"):
        ConstructionJournal(path, config_hash="different", resume=True)


def test_pilot_selection_is_deterministic_bounded_and_split_balanced(tmp_path: Path) -> None:
    root = tmp_path / "pairing"
    for split, sample_ids in {
        "train": [f"train-{index}" for index in range(20)],
        "validation": [f"validation-{index}" for index in range(10)],
    }.items():
        path = root / f"eligible_{split}.parquet"
        path.mkdir(parents=True)
        pq.write_table(pa.table({"sample_id": list(reversed(sample_ids))}), path / "part-000000.parquet")
    first = _pilot_ids(root, 6, 6006)
    second = _pilot_ids(root, 6, 6006)
    assert first == second
    assert len(first) == 6
    assert sum(split == "train" for split, _ in first) == 3
    assert sum(split == "validation" for split, _ in first) == 3
    assert all(sample_id.startswith(f"{split}-") for split, sample_id in first)


def test_representative_pilot_selects_exactly_256_samples_per_split(tmp_path: Path) -> None:
    root = tmp_path / "pairing"
    for split in ("train", "validation"):
        path = root / f"eligible_{split}.parquet"
        path.mkdir(parents=True)
        sample_ids = [f"{split}-{index:04d}" for index in range(300)]
        for partition_index, start in enumerate(range(0, len(sample_ids), 75)):
            pq.write_table(
                pa.table({"sample_id": list(reversed(sample_ids[start : start + 75]))}),
                path / f"part-{partition_index:06d}.parquet",
            )
    selected = _pilot_ids(root, 512, 6006)
    assert len(selected) == 512
    assert sum(split == "train" for split, _ in selected) == 256
    assert sum(split == "validation" for split, _ in selected) == 256
    assert selected == _pilot_ids(root, 512, 6006)


def test_stage_profiler_reports_bounded_latency_and_byte_statistics() -> None:
    profiler = StageProfiler(maximum_latency_samples=2)
    profiler.add(StageObservation("npz_loading", 0.2, bytes_read=100))
    profiler.add(StageObservation("npz_loading", 0.4, bytes_read=200))
    profiler.add(StageObservation("npz_loading", 0.8, bytes_read=300))
    report = profiler.report(2.0)["npz_loading"]
    assert report["total_seconds"] == pytest.approx(1.4)
    assert report["percentage_of_runtime"] == pytest.approx(70.0)
    assert report["sample_count"] == 3
    assert report["mean_latency_seconds"] == pytest.approx(1.4 / 3)
    assert report["median_latency_seconds"] == pytest.approx(0.3)
    assert report["p95_latency_seconds"] == pytest.approx(0.39)
    assert report["bytes_read"] == 600
    assert report["latency_samples_truncated"] is True
    restored = StageProfiler.from_snapshot(profiler.snapshot())
    assert restored.report(2.0) == profiler.report(2.0)


def test_ordered_thread_pipeline_is_deterministic_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def delayed(row: dict[str, object], _config: dict[str, object], split: str) -> EvaluationOutcome:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.002 * (6 - int(row["index"])))
        with lock:
            active -= 1
        decision = {"sample_id": row["sample_id"], "split": split}
        return EvaluationOutcome(row, decision, None, (), None)

    monkeypatch.setattr(sidecar_module, "_evaluate_with_timings", delayed)
    rows = [{"sample_id": f"sample-{index}", "index": index} for index in range(6)]
    outcomes = list(
        _ordered_evaluations(
            rows,
            {"worker_count": 2, "worker_queue_size": 3},
            "train",
        )
    )
    assert [item.row["sample_id"] for item in outcomes] == [row["sample_id"] for row in rows]
    assert maximum_active == 2


@pytest.mark.parametrize(
    "physical_dataset_name,explicit_value,expected",
    [
        ("eligible_train", None, "train"),
        ("eligible_validation", None, "validation"),
        ("eligible_train", " TRAINING ", "train"),
        ("eligible_validation", "val", "validation"),
    ],
)
def test_physical_dataset_assigns_and_validates_split(
    tmp_path: Path,
    physical_dataset_name: str,
    explicit_value: str | None,
    expected: str,
) -> None:
    row = {"sample_id": "sample"}
    if explicit_value is not None:
        row["split"] = explicit_value
    input_unit = InputUnit(
        physical_dataset_name=physical_dataset_name,
        canonical_split=expected,
        dataset_path=tmp_path / f"{physical_dataset_name}.parquet",
        fragment_identity="part-000000.parquet",
        input_unit_identity=f"{physical_dataset_name}:part-000000.parquet:0",
        rows=(),
    )
    assigned = _assign_physical_split(row, input_unit=input_unit)
    assert assigned["split"] == expected
    assert assigned["physical_dataset_name"] == physical_dataset_name
    assert "split" not in row or row["split"] == explicit_value


def test_split_contradiction_and_unknown_dataset_are_rejected(tmp_path: Path) -> None:
    train_unit = InputUnit(
        physical_dataset_name="eligible_train",
        canonical_split="train",
        dataset_path=tmp_path / "eligible_train.parquet",
        fragment_identity="part-000000.parquet",
        input_unit_identity="eligible_train:part-000000.parquet:0",
        rows=(),
    )
    with pytest.raises(ValueError, match="split contradiction"):
        _assign_physical_split(
            {"sample_id": "sample", "split": "validation"},
            input_unit=train_unit,
        )
    with pytest.raises(ValueError, match="Unknown physical"):
        _physical_split("all_pairs.parquet")
    with pytest.raises(ValueError, match="Unknown physical"):
        _physical_split("train")
    with pytest.raises(ValueError, match="Invalid canonical split"):
        InputUnit(
            physical_dataset_name="eligible_train",
            canonical_split="eligible_train",
            dataset_path=tmp_path / "eligible_train.parquet",
            fragment_identity="part-000000.parquet",
            input_unit_identity="eligible_train:part-000000.parquet:0",
            rows=(),
        )
    with pytest.raises(ValueError, match="Physical dataset and canonical split disagree"):
        InputUnit(
            physical_dataset_name="eligible_train",
            canonical_split="validation",
            dataset_path=tmp_path / "eligible_train.parquet",
            fragment_identity="part-000000.parquet",
            input_unit_identity="eligible_train:part-000000.parquet:0",
            rows=(),
        )
    with pytest.raises(ValueError, match="canonical physical split"):
        evaluate_sample({"split": "validation"}, {}, canonical_split="train")
    with pytest.raises(ValueError, match="Unknown canonical split"):
        evaluate_sample({"split": "eligible_train"}, {}, canonical_split="eligible_train")


def test_input_units_stamp_split_without_writing_source_parquet(tmp_path: Path) -> None:
    root = tmp_path / "pairing"
    directory = root / "eligible_validation.parquet"
    directory.mkdir(parents=True)
    source = directory / "part-000000.parquet"
    pq.write_table(pa.table({"sample_id": ["validation-sample"]}), source)
    before = sha256_file(source)
    units = list(_input_units(root, "eligible_validation", 4))
    assert units[0].physical_dataset_name == "eligible_validation"
    assert units[0].canonical_split == "validation"
    assert units[0].dataset_path == directory
    assert units[0].fragment_identity
    assert units[0].input_unit_identity.startswith("eligible_validation:")
    assert units[0].rows[0]["split"] == "validation"
    assert units[0].rows[0]["physical_dataset_name"] == "eligible_validation"
    assert sha256_file(source) == before
    with pytest.raises(ValueError, match="Unknown physical"):
        list(_input_units(root, "unowned", 4))


def test_enrichment_reassertion_does_not_translate_ownership_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_unit = InputUnit(
        physical_dataset_name="eligible_train",
        canonical_split="train",
        dataset_path=tmp_path / "eligible_train.parquet",
        fragment_identity="part-000000.parquet",
        input_unit_identity="eligible_train:part-000000.parquet:0",
        rows=(),
    )

    def forbidden_translation(_physical_dataset_name: str) -> str:
        raise AssertionError("physical ownership was translated twice")

    monkeypatch.setattr(sidecar_module, "_physical_split", forbidden_translation)
    enriched = _assign_physical_split(
        {
            "sample_id": "sample",
            "physical_dataset_name": "eligible_train",
            "split": "train",
        },
        input_unit=input_unit,
    )
    assert enriched["physical_dataset_name"] == "eligible_train"
    assert enriched["split"] == "train"


def _sample_sidecar(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    source = tmp_path / "source.cif"
    source.write_text("synthetic immutable source")
    atoms = {
        "N": np.asarray([-0.525, 1.363, 0.0]),
        "CA": np.zeros(3),
        "C": np.asarray([1.526, 0.0, 0.0]),
        "O": np.asarray([2.153, -1.062, 0.0]),
        "CB": np.asarray([-0.529, -0.774, -1.205]),
    }
    residue = BackboneResidue("ALA", "A", "1", "", atoms)
    npz_path = tmp_path / "sample.npz"
    np.savez_compressed(
        npz_path,
        residue_ids=np.asarray(["1"]),
        ca_coordinates=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        residue_mask=np.asarray([True]),
        distance_matrix=np.zeros((1, 1), dtype=np.float32),
        metadata=np.asarray(json.dumps({"residue_ids": ["1"]})),
    )
    npz = load_npz_geometry_evidence(npz_path)
    mapping = {
        "resolved_target_residue_ids": ["1"],
        "target_residue_classifications": ["canonical_target_residue"],
        "mapping_version": "mapping-v1",
        "canonicalization_version": "canonical-v1",
    }
    anchor = {
        "selected_calpha_conformers_json": "[null]",
        "coordinate_anchor_failure_examples_json": "{}",
        "npz_internal_matrix_rmse_angstrom": 0.0,
        "npz_internal_matrix_maximum_error_angstrom": 0.0,
        "source_to_npz_calpha_coordinate_rmse_angstrom": 0.0,
        "source_to_npz_calpha_coordinate_maximum_error_angstrom": 0.0,
        "ca_candidate_counts_json": "[1]",
        "blank_altloc_fallback_count": 0,
        "conflicting_nonblank_conformer_count": 0,
    }
    row = {
        "sample_id": "sample",
        "split": "train",
        "sequence": "A",
        "model_number": 1,
        "resolved_source_path": str(source),
        "resolved_npz_path": str(npz_path),
        "actual_source_sha256": sha256_file(source),
        "actual_npz_sha256": sha256_file(npz_path),
    }
    return (
        _sidecar_row(
            row,
            [residue],
            mapping,
            anchor,
            npz,
            canonical_split="train",
        ),
        source,
        npz_path,
    )


def _nontrivial_backbone(length: int = 5) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    coordinates = {atom: [] for atom in ("N", "CA", "C")}
    for index in range(length):
        ca = np.asarray(
            [3.8 * index, 0.4 * np.sin(1.1 * index), 0.5 * np.cos(0.7 * index)],
            dtype=np.float64,
        )
        coordinates["CA"].append(ca)
        coordinates["N"].append(ca + np.asarray([-1.25, 0.45 * np.cos(index + 0.2), 0.3 * np.sin(0.8 * index + 0.1)]))
        coordinates["C"].append(
            ca + np.asarray([1.3, 0.4 * np.sin(0.9 * index + 0.3), 0.35 * np.cos(0.6 * index + 0.2)])
        )
    arrays = {atom: np.asarray(values) for atom, values in coordinates.items()}
    masks = {atom: np.ones(length, dtype=np.bool_) for atom in arrays}
    return arrays, masks


def _multi_residue_sidecar(tmp_path: Path, *, sample_id: str, split: str) -> dict[str, object]:
    coordinates, _ = _nontrivial_backbone()
    source = tmp_path / f"{sample_id}.cif"
    source.write_text("synthetic immutable source")
    residues = []
    for index in range(len(coordinates["CA"])):
        atoms = {atom: coordinates[atom][index] for atom in ("N", "CA", "C")}
        atoms["O"] = coordinates["C"][index] + np.asarray([0.3, -0.8, 0.2])
        atoms["CB"] = coordinates["CA"][index] + np.asarray([-0.5, -0.8, -1.2])
        residues.append(BackboneResidue("ALA", "A", str(index + 1), "", atoms))
    residue_ids = [str(index + 1) for index in range(len(residues))]
    ca = coordinates["CA"].astype(np.float32)
    distance_matrix = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1).astype(np.float32)
    npz_path = tmp_path / f"{sample_id}.npz"
    np.savez_compressed(
        npz_path,
        residue_ids=np.asarray(residue_ids),
        ca_coordinates=ca,
        residue_mask=np.ones(len(residues), dtype=np.bool_),
        distance_matrix=distance_matrix,
        metadata=np.asarray(json.dumps({"residue_ids": residue_ids})),
    )
    npz = load_npz_geometry_evidence(npz_path)
    mapping = {
        "resolved_target_residue_ids": residue_ids,
        "target_residue_classifications": ["canonical_target_residue"] * len(residues),
        "mapping_version": "mapping-v1",
        "canonicalization_version": "canonical-v1",
    }
    anchor = {
        "selected_calpha_conformers_json": json.dumps([None] * len(residues)),
        "coordinate_anchor_failure_examples_json": "{}",
        "npz_internal_matrix_rmse_angstrom": 0.0,
        "npz_internal_matrix_maximum_error_angstrom": 0.0,
        "source_to_npz_calpha_coordinate_rmse_angstrom": 0.0,
        "source_to_npz_calpha_coordinate_maximum_error_angstrom": 0.0,
        "ca_candidate_counts_json": json.dumps([1] * len(residues)),
        "blank_altloc_fallback_count": 0,
        "conflicting_nonblank_conformer_count": 0,
    }
    row = {
        "sample_id": sample_id,
        "split": split,
        "sequence": "A" * len(residues),
        "model_number": 1,
        "resolved_source_path": str(source),
        "resolved_npz_path": str(npz_path),
        "actual_source_sha256": sha256_file(source),
        "actual_npz_sha256": sha256_file(npz_path),
    }
    return _sidecar_row(row, residues, mapping, anchor, npz, canonical_split=split)


def test_canonical_torsions_quantize_before_derivation_and_mask_boundaries() -> None:
    coordinates, masks = _nontrivial_backbone()
    derived = derive_backbone_torsions(coordinates, masks, peptide_bond_threshold_angstrom=2.0)
    quantized = derive_backbone_torsions(
        {atom: values.astype(np.float32) for atom, values in coordinates.items()},
        masks,
        peptide_bond_threshold_angstrom=2.0,
    )
    assert derived == quantized
    assert derived["phi_mask"] == [False, True, True, True, True]
    assert derived["omega_mask"] == [False, True, True, True, True]
    assert derived["psi_mask"] == [True, True, True, True, False]
    assert derived["phi_sin_cos"][0] == list(TORSION_NEUTRAL_SIN_COS)
    assert derived["psi_sin_cos"][-1] == list(TORSION_NEUTRAL_SIN_COS)
    for name in ("phi", "psi", "omega"):
        values = np.asarray(derived[f"{name}_sin_cos"])[derived[f"{name}_mask"]]
        assert np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=2e-6, rtol=0)


def test_canonical_torsions_mask_chain_breaks_missing_atoms_and_degeneracy() -> None:
    coordinates, masks = _nontrivial_backbone()
    broken = {atom: values.copy() for atom, values in coordinates.items()}
    broken["N"][3] += np.asarray([10.0, 0.0, 0.0])
    derived = derive_backbone_torsions(broken, masks, peptide_bond_threshold_angstrom=2.0)
    assert derived["chain_continuity_mask"][2] is False
    assert derived["psi_mask"][2] is False
    assert derived["phi_mask"][3] is False
    assert derived["omega_mask"][3] is False

    for missing_atom in ("N", "CA", "C"):
        missing_masks = {atom: values.copy() for atom, values in masks.items()}
        missing_masks[missing_atom][2] = False
        missing = derive_backbone_torsions(coordinates, missing_masks, peptide_bond_threshold_angstrom=2.0)
        for name, definition in sidecar_module.TORSION_DEFINITIONS.items():
            for residue_index in range(len(coordinates["CA"])):
                requires_missing_atom = any(
                    atom == missing_atom and residue_index + offset == 2 for atom, offset in definition
                )
                if requires_missing_atom:
                    assert missing[f"{name}_mask"][residue_index] is False

    length = 4
    collinear = {
        "N": np.asarray([[3.8 * i - 1.0, 0.0, 0.0] for i in range(length)]),
        "CA": np.asarray([[3.8 * i, 0.0, 0.0] for i in range(length)]),
        "C": np.asarray([[3.8 * i + 1.0, 0.0, 0.0] for i in range(length)]),
    }
    all_present = {atom: np.ones(length, dtype=np.bool_) for atom in collinear}
    degenerate = derive_backbone_torsions(collinear, all_present, peptide_bond_threshold_angstrom=2.0)
    assert not any(degenerate["phi_mask"] + degenerate["psi_mask"] + degenerate["omega_mask"])


@pytest.mark.parametrize("z,expected_sign", [(1e-5, 1), (-1e-5, -1)])
def test_canonical_torsions_are_stable_near_periodic_boundary(z: float, expected_sign: int) -> None:
    coordinates = {
        "N": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        "CA": np.asarray([[0.0, 0.0, 2.0], [0.0, 1.0, 0.0]]),
        "C": np.asarray([[1.0, 0.0, 0.0], [1.0, 1.0, z]]),
    }
    masks = {atom: np.ones(2, dtype=np.bool_) for atom in coordinates}
    derived = derive_backbone_torsions(coordinates, masks, peptide_bond_threshold_angstrom=2.0)
    sine, cosine = derived["phi_sin_cos"][1]
    assert np.sign(sine) == expected_sign
    assert cosine < -0.999999


def test_torsion_parquet_round_trip_verifies_multiple_proteins_and_splits(tmp_path: Path) -> None:
    rows = [
        _multi_residue_sidecar(tmp_path, sample_id="train-protein", split="train"),
        _multi_residue_sidecar(tmp_path, sample_id="validation-protein", split="validation"),
    ]
    path = tmp_path / "torsions.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=rich_geometry_schema()), path, row_group_size=1)
    reopened = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
        record = batch.to_pylist()[0]
        _verify_reopened_sidecar(record)
        reopened.append((record["sample_id"], record["split"]))
    assert reopened == [("train-protein", "train"), ("validation-protein", "validation")]


def test_cuda_feature_validation_batches_multiple_proteins_by_token_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_multi_residue_sidecar(tmp_path, sample_id=f"protein-{index}", split="train") for index in range(3)]
    path = tmp_path / "train" / "part-000000.parquet"
    path.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(rows, schema=rich_geometry_schema()), path)
    observed_batch_sizes = []

    def synthetic_cuda(
        sidecars: list[dict[str, object]],
        _config: dict[str, object],
        _profiler: StageProfiler,
    ) -> tuple[str, None, float]:
        observed_batch_sizes.append(len(sidecars))
        return "cuda", None, 12.0

    monkeypatch.setattr(sidecar_module, "_apply_cuda_feature_batching", synthetic_cuda)
    backend, reason, peak = _validate_cuda_features_from_shards(
        tmp_path,
        [{"dataset": "train", "path": "train/part-000000.parquet"}],
        {"feature_backend": "cuda", "cuda_residue_token_budget": 10},
        StageProfiler(),
    )
    assert (backend, reason, peak) == ("cuda", None, 12.0)
    assert observed_batch_sizes == [2, 1]


def test_cuda_feature_runtime_failure_falls_back_without_mutating_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _multi_residue_sidecar(tmp_path, sample_id="protein", split="validation")
    path = tmp_path / "validation" / "part-000000.parquet"
    path.parent.mkdir()
    pq.write_table(pa.Table.from_pylist([row], schema=rich_geometry_schema()), path)
    digest_before = sha256_file(path)

    def unavailable(*_args: object, **_kwargs: object) -> tuple[str, None, float]:
        raise RuntimeError("synthetic CUDA failure")

    monkeypatch.setattr(sidecar_module, "_apply_cuda_feature_batching", unavailable)
    backend, reason, peak = _validate_cuda_features_from_shards(
        tmp_path,
        [{"dataset": "validation", "path": "validation/part-000000.parquet"}],
        {"feature_backend": "cuda", "cuda_residue_token_budget": 100},
        StageProfiler(),
    )
    assert backend == "cpu"
    assert "cuda_runtime_fallback:RuntimeError" in str(reason)
    assert peak == 0.0
    assert sha256_file(path) == digest_before


def test_cuda_feature_backend_matches_canonical_cpu_when_available(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rows = [_multi_residue_sidecar(tmp_path, sample_id=f"cuda-{index}", split="train") for index in range(2)]
    backend, reason, peak = sidecar_module._apply_cuda_feature_batching(
        rows,
        {
            "feature_backend": "cuda",
            "cuda_residue_token_budget": 128,
            "maximum_cuda_allocated_mib": 6144,
        },
        StageProfiler(),
    )
    assert backend == "cuda"
    assert reason is None
    assert peak is not None and peak < 6144


@pytest.mark.parametrize("name", ["phi", "psi", "omega"])
def test_torsion_verification_rejects_corrupted_features_with_context(tmp_path: Path, name: str) -> None:
    row = _multi_residue_sidecar(tmp_path, sample_id=f"corrupt-{name}", split="validation")
    index = row[f"{name}_mask"].index(True)
    row[f"{name}_sin_cos"][index] = [1.0, 0.0]
    with pytest.raises(ValueError, match="Stored torsion features") as caught:
        _verify_reopened_sidecar(row)
    message = str(caught.value)
    for expected in (
        '"sample_id"',
        '"split"',
        '"torsion_name"',
        '"residue_index"',
        '"residue_id"',
        '"required_atom_masks"',
        '"continuity_mask_values"',
        '"wrapped_angular_error_radians"',
    ):
        assert expected in message


@pytest.mark.parametrize("name", ["phi", "psi", "omega"])
def test_torsion_verification_rejects_corrupted_masks(tmp_path: Path, name: str) -> None:
    row = _multi_residue_sidecar(tmp_path, sample_id=f"mask-{name}", split="train")
    index = row[f"{name}_mask"].index(True)
    row[f"{name}_mask"][index] = False
    row[f"{name}_sin_cos"][index] = list(TORSION_NEUTRAL_SIN_COS)
    with pytest.raises(ValueError, match="Stored torsion features") as caught:
        _verify_reopened_sidecar(row)
    assert f'"{name}": 1' in str(caught.value)


def test_torsion_verification_reports_phi_psi_and_omega_together(tmp_path: Path) -> None:
    row = _multi_residue_sidecar(tmp_path, sample_id="all-corrupt", split="train")
    for name in ("phi", "psi", "omega"):
        index = row[f"{name}_mask"].index(True)
        row[f"{name}_sin_cos"][index] = [1.0, 0.0]
    with pytest.raises(ValueError, match="Stored torsion features") as caught:
        _verify_reopened_sidecar(row)
    message = str(caught.value)
    assert all(f'"{name}": 1' in message for name in ("phi", "psi", "omega"))


def test_missing_native_cb_stores_a_masked_pseudo_cb_with_explicit_source(tmp_path: Path) -> None:
    row, _, _ = _sample_sidecar(tmp_path)
    assert row["cb_source"] == [1]

    source = tmp_path / "source.cif"
    npz = load_npz_geometry_evidence(tmp_path / "sample.npz")
    residue = BackboneResidue(
        "GLY",
        "G",
        "1",
        "",
        {
            "N": np.asarray([-0.525, 1.363, 0.0]),
            "CA": np.zeros(3),
            "C": np.asarray([1.526, 0.0, 0.0]),
            "O": np.asarray([2.153, -1.062, 0.0]),
        },
    )
    mapping = {
        "resolved_target_residue_ids": ["1"],
        "target_residue_classifications": ["canonical_target_residue"],
        "mapping_version": "mapping-v1",
        "canonicalization_version": "canonical-v1",
    }
    anchor = {
        "selected_calpha_conformers_json": "[null]",
        "coordinate_anchor_failure_examples_json": "{}",
        "npz_internal_matrix_rmse_angstrom": 0.0,
        "npz_internal_matrix_maximum_error_angstrom": 0.0,
        "source_to_npz_calpha_coordinate_rmse_angstrom": 0.0,
        "source_to_npz_calpha_coordinate_maximum_error_angstrom": 0.0,
        "ca_candidate_counts_json": "[1]",
        "blank_altloc_fallback_count": 0,
        "conflicting_nonblank_conformer_count": 0,
    }
    input_row = {
        "sample_id": "sample",
        "split": "train",
        "sequence": "G",
        "model_number": 1,
        "resolved_source_path": str(source),
        "resolved_npz_path": str(tmp_path / "sample.npz"),
        "actual_source_sha256": sha256_file(source),
        "actual_npz_sha256": sha256_file(tmp_path / "sample.npz"),
    }
    pseudo = _sidecar_row(
        input_row,
        [residue],
        mapping,
        anchor,
        npz,
        canonical_split="train",
    )
    assert pseudo["cb_source"] == [0]
    assert pseudo["cb_mask"] == [True]
    assert np.isfinite(np.asarray(pseudo["cb_coordinates"])).all()
    assert not np.allclose(pseudo["cb_coordinates"][0], [0.0, 0.0, 0.0])


def test_atomic_shard_resume_is_deterministic_and_does_not_rewrite(tmp_path: Path) -> None:
    row, _, _ = _sample_sidecar(tmp_path)
    path = tmp_path / "part.parquet"
    count, first_hash = _write_part(path, [row], rich_geometry_schema(), 1)
    first_mtime = path.stat().st_mtime_ns
    count_again, second_hash = _write_part(path, [row], rich_geometry_schema(), 1)
    assert (count, count_again) == (1, 1)
    assert first_hash == second_hash
    assert path.stat().st_mtime_ns == first_mtime


def test_verify_only_checks_shards_membership_nested_values_and_npz_anchor(tmp_path: Path) -> None:
    row, source, npz_path = _sample_sidecar(tmp_path)
    output = tmp_path / "sidecars"
    train_path = output / "train" / "part-000000.parquet"
    eligibility_path = output / "eligibility_manifest.parquet" / "part-000000.parquet"
    train_count, train_hash = _write_part(train_path, [row], rich_geometry_schema(), 1)
    decision = {
        "sample_id": "sample",
        "split": "train",
        "eligible": True,
        "exclusion_reason": None,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "sequence_length": 1,
        "coordinate_anchor_policy_version": COORDINATE_ANCHOR_POLICY_VERSION,
        "diagnostics": "{}",
    }
    eligibility_count, eligibility_hash = _write_part(eligibility_path, [decision], eligibility_schema(), 1)
    phase0 = _phase0(tmp_path / "authorization")
    shards = [
        {
            "path": "eligibility_manifest.parquet/part-000000.parquet",
            "dataset": "eligibility_manifest.parquet",
            "row_count": eligibility_count,
            "sha256": eligibility_hash,
        },
        {
            "path": "train/part-000000.parquet",
            "dataset": "train",
            "row_count": train_count,
            "sha256": train_hash,
        },
    ]
    protocol = {
        "status": "completed",
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "torsion_convention_version": TORSION_CONVENTION_VERSION,
        "mode": "full",
        "authorizes_definitive_dataset": True,
        "authorizes_training": True,
        "authorizes_full_dataset": True,
        "phase0_protocol": str(phase0),
        "phase0_protocol_sha256": sha256_file(phase0),
        "protected_inputs_unchanged": True,
        "observed_phase1_inputs_unchanged": True,
        "unexplained_failure_count": 0,
        "definitive_observed_counts": {"total": 1, "eligible": 1, "excluded": 0},
        "observed_split_counts": {
            "train": {"total": 1, "eligible": 1, "excluded": 0},
            "validation": {"total": 0, "eligible": 0, "excluded": 0},
        },
        "shards": shards,
        "shard_summaries_by_dataset": {
            "train": {"file_count": 1, "row_count": 1, "compressed_bytes": train_path.stat().st_size},
            "validation": {"file_count": 0, "row_count": 0, "compressed_bytes": 0},
            "eligibility_manifest.parquet": {
                "file_count": 1,
                "row_count": 1,
                "compressed_bytes": eligibility_path.stat().st_size,
            },
            "exclusions.parquet": {"file_count": 0, "row_count": 0, "compressed_bytes": 0},
        },
        "started_utc": "2026-01-01T00:00:00+00:00",
        "completed_utc": "2026-01-01T00:00:01+00:00",
        "elapsed_seconds": 1.0,
        "processed_samples": 1,
        "samples_per_second": 1.0,
        "eligible_samples_per_second": 1.0,
        "stage_timing": StageProfiler().report(1.0),
        "cpu_utilization_percent": 100.0,
    }
    journal = ConstructionJournal(output / "construction_journal.sqlite", config_hash="synthetic", resume=False)
    journal.set("construction_mode", "full")
    journal.set("status", "completed")
    journal.record_observed_input(str(source), sha256_file(source))
    journal.record_observed_input(str(npz_path), sha256_file(npz_path))
    journal.connection.commit()
    protocol["observed_phase1_input_verification"] = _verify_observed_inputs(journal.connection)
    journal.connection.close()
    (output / "protocol.json").write_text(json.dumps(protocol))
    (output / "shard_hashes.sha256").write_text("".join(f"{item['sha256']}  {item['path']}\n" for item in shards))
    verified = verify_sidecar_dataset(output, require_training_authorization=True)
    assert verified == {"status": "verified", "shard_count": 2, "reopened_shard_count": 1}

    pilot_protocol = {
        **protocol,
        "mode": "pilot",
        "authorizes_definitive_dataset": False,
        "authorizes_training": False,
        "authorizes_full_dataset": False,
    }
    connection = sqlite3.connect(output / "construction_journal.sqlite")
    connection.execute("UPDATE metadata SET value='pilot' WHERE key='construction_mode'")
    connection.commit()
    connection.close()
    (output / "protocol.json").write_text(json.dumps(pilot_protocol))
    verify_sidecar_dataset(output)
    with pytest.raises(ValueError, match="not an authorized definitive training dataset"):
        verify_sidecar_dataset(output, require_training_authorization=True)

    incorrect_summary = copy.deepcopy(pilot_protocol)
    incorrect_summary["shard_summaries_by_dataset"]["train"]["compressed_bytes"] += 1
    (output / "protocol.json").write_text(json.dumps(incorrect_summary))
    with pytest.raises(ValueError, match="shard summaries"):
        verify_sidecar_dataset(output)

    incorrect_timing = {**pilot_protocol, "elapsed_seconds": 2.0}
    (output / "protocol.json").write_text(json.dumps(incorrect_timing))
    with pytest.raises(ValueError, match="elapsed time"):
        verify_sidecar_dataset(output)

    missing_authorization = dict(pilot_protocol)
    missing_authorization.pop("authorizes_training")
    (output / "protocol.json").write_text(json.dumps(missing_authorization))
    with pytest.raises(ValueError, match="authorization fields are missing"):
        verify_sidecar_dataset(output)

    altered_pilot = {
        **pilot_protocol,
        "authorizes_definitive_dataset": True,
        "authorizes_training": True,
        "authorizes_full_dataset": True,
    }
    (output / "protocol.json").write_text(json.dumps(altered_pilot))
    with pytest.raises(ValueError, match="Pilot sidecars cannot authorize"):
        verify_sidecar_dataset(output)

    tampered_mode = {**altered_pilot, "mode": "full"}
    (output / "protocol.json").write_text(json.dumps(tampered_mode))
    with pytest.raises(ValueError, match="construction-journal provenance"):
        verify_sidecar_dataset(output)

    connection = sqlite3.connect(output / "construction_journal.sqlite")
    connection.execute("UPDATE metadata SET value='full' WHERE key='construction_mode'")
    connection.commit()
    connection.close()
    (output / "protocol.json").write_text(json.dumps(protocol))
    source.write_text("mutated")
    with pytest.raises(ValueError, match="Observed Phase-1 input changed"):
        verify_sidecar_dataset(output)


def test_exclusion_schema_keeps_evidence_compact_and_structured() -> None:
    schema = exclusion_schema()
    assert schema.names == [
        "sample_id",
        "split",
        "exclusion_reason",
        "source_path",
        "npz_path",
        "bounded_evidence",
    ]


def test_actual_pilot_pipeline_preserves_typed_ownership_across_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase0 = _phase0(tmp_path / "authorization")
    config = _config(tmp_path, phase0)
    config["maximum_rss_mib"] = 2048
    Path(config["normalization_source"]).write_text("{}")
    pairing = Path(config["pairing_dataset"])
    for split in ("train", "validation"):
        directory = pairing / f"eligible_{split}.parquet"
        directory.mkdir(parents=True)
        pq.write_table(
            pa.table({"sample_id": [f"{split}-sample"]}),
            directory / "part-000000.parquet",
        )
    source_partition_hashes = {str(path): sha256_file(path) for path in pairing.rglob("*.parquet") if path.is_file()}
    template, source, npz_path = _sample_sidecar(tmp_path)
    interrupt_validation = {"enabled": True}

    monkeypatch.setattr(sidecar_module, "enrich_source_locators", lambda rows, **_: rows)

    def synthetic_evaluation(
        row: dict[str, object],
        _config: dict[str, object],
        *,
        canonical_split: str,
        stage_observer: object = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        sample_id = str(row["sample_id"])
        assert row["split"] == canonical_split
        assert row["physical_dataset_name"] == f"eligible_{canonical_split}"
        if canonical_split == "validation" and interrupt_validation["enabled"]:
            raise KeyboardInterrupt
        rich = {**template, "sample_id": sample_id, "split": canonical_split}
        decision = {
            "sample_id": sample_id,
            "split": canonical_split,
            "eligible": True,
            "exclusion_reason": None,
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "npz_path": str(npz_path),
            "npz_sha256": sha256_file(npz_path),
            "sequence_length": 1,
            "coordinate_anchor_policy_version": COORDINATE_ANCHOR_POLICY_VERSION,
            "diagnostics": "{}",
        }
        return decision, rich

    monkeypatch.setattr(sidecar_module, "evaluate_sample", synthetic_evaluation)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = Path(config["pilot_output_dir"])
    with pytest.raises(KeyboardInterrupt):
        construct_sidecars(config_path, mode="pilot")
    interrupted = json.loads((output / "heartbeat.json").read_text())
    assert interrupted["status"] == "interrupted"
    assert interrupted["latest_committed_journal_position"]["physical_dataset_name"] == "eligible_train"
    assert interrupted["latest_committed_journal_position"]["split"] == "train"
    assert interrupted["resumable"] is True
    assert interrupted["configured_maximum_rss_mib"] == 2048
    assert interrupted["effective_maximum_rss_mib"] == 2048
    assert interrupted["runtime_rss_override_applied"] is False
    connection = sqlite3.connect(output / "construction_journal.sqlite")
    stored_config_hash_before = connection.execute("SELECT value FROM metadata WHERE key='config_hash'").fetchone()[0]
    connection.close()
    interrupt_validation["enabled"] = False
    protocol = construct_sidecars(config_path, mode="pilot", resume=True, maximum_rss_mib=4096)
    assert protocol["definitive_observed_counts"] == {"total": 2, "eligible": 2, "excluded": 0}
    assert protocol["observed_split_counts"] == {
        "train": {"total": 1, "eligible": 1, "excluded": 0},
        "validation": {"total": 1, "eligible": 1, "excluded": 0},
    }
    assert protocol["authorizes_full_dataset"] is False
    assert protocol["authorizes_definitive_dataset"] is False
    assert protocol["authorizes_training"] is False
    assert protocol["processed_samples"] == 2
    assert protocol["elapsed_seconds"] >= 0
    assert protocol["samples_per_second"] > 0
    assert protocol["eligible_samples_per_second"] > 0
    assert protocol["started_utc"] <= protocol["completed_utc"]
    assert protocol["configured_maximum_rss_mib"] == 2048
    assert protocol["effective_maximum_rss_mib"] == 4096
    assert protocol["runtime_rss_override_applied"] is True
    assert set(protocol["stage_timing"]) == set(PERFORMANCE_STAGE_NAMES)
    assert protocol["stage_timing"]["arrow_row_retrieval"]["sample_count"] > 0
    assert protocol["stage_timing"]["parquet_encoding_writing"]["bytes_written"] > 0
    assert protocol["stage_timing"]["final_verification"]["sample_count"] == 2
    assert sum(item["file_count"] for item in protocol["shard_summaries_by_dataset"].values()) == len(
        protocol["shards"]
    )
    assert sum(item["row_count"] for item in protocol["shard_summaries_by_dataset"].values()) == 4
    assert (output / "schema.json").is_file()
    assert (output / "vocabulary.json").is_file()
    assert (output / "normalization.json").is_file()
    assert (output / "construction_journal.sqlite").is_file()
    completed_heartbeat = json.loads((output / "heartbeat.json").read_text())
    assert completed_heartbeat["status"] == "completed"
    assert completed_heartbeat["configured_maximum_rss_mib"] == 2048
    assert completed_heartbeat["effective_maximum_rss_mib"] == 4096
    assert completed_heartbeat["runtime_rss_override_applied"] is True
    connection = sqlite3.connect(output / "construction_journal.sqlite")
    stored_config_hash_after = connection.execute("SELECT value FROM metadata WHERE key='config_hash'").fetchone()[0]
    connection.close()
    assert stored_config_hash_after == stored_config_hash_before
    assert protocol["configuration_hash"] == stored_config_hash_before

    changed_config = {**config, "seed": int(config["seed"]) + 1}
    config_path.write_text(json.dumps(changed_config))
    with pytest.raises(ValueError, match="configuration hash"):
        construct_sidecars(config_path, mode="pilot", resume=True, maximum_rss_mib=4096)
    config_path.write_text(json.dumps(config))
    protocol_mtime = (output / "protocol.json").stat().st_mtime_ns
    resumed = construct_sidecars(config_path, mode="pilot", resume=True)
    assert resumed == protocol
    assert (output / "protocol.json").stat().st_mtime_ns == protocol_mtime

    config["worker_count"] = 2
    config["worker_queue_size"] = 2
    config_path.write_text(json.dumps(config))
    full_protocol = construct_sidecars(config_path, mode="full")
    assert full_protocol["mode"] == "full"
    assert full_protocol["authorizes_definitive_dataset"] is True
    assert full_protocol["authorizes_training"] is True
    assert full_protocol["authorizes_full_dataset"] is True
    assert {item["path"]: item["sha256"] for item in full_protocol["shards"]} == {
        item["path"]: item["sha256"] for item in protocol["shards"]
    }
    assert verify_sidecar_dataset(config["output_dir"], require_training_authorization=True)["status"] == "verified"
    assert source_partition_hashes == {
        str(path): sha256_file(path) for path in pairing.rglob("*.parquet") if path.is_file()
    }
    train_ids = {row["sample_id"] for row in pq.read_table(output / "train").to_pylist()}
    validation_ids = {row["sample_id"] for row in pq.read_table(output / "validation").to_pylist()}
    assert train_ids == {"train-sample"}
    assert validation_ids == {"validation-sample"}
    assert train_ids.isdisjoint(validation_ids)
    connection = sqlite3.connect(output / "construction_journal.sqlite")
    try:
        assert connection.execute("SELECT split,sample_id FROM decisions ORDER BY split").fetchall() == [
            ("train", "train-sample"),
            ("validation", "validation-sample"),
        ]
        assert connection.execute("SELECT value FROM metadata WHERE key='last_committed_split'").fetchone() == (
            "validation",
        )
        assert connection.execute(
            "SELECT physical_dataset_name,canonical_split FROM completed_units ORDER BY unit_index"
        ).fetchall() == [
            ("eligible_train", "train"),
            ("eligible_validation", "validation"),
        ]
    finally:
        connection.close()


def test_split_contradiction_publishes_bounded_nonresumable_failure(
    tmp_path: Path,
) -> None:
    phase0 = _phase0(tmp_path / "authorization")
    config = _config(tmp_path, phase0)
    Path(config["normalization_source"]).write_text("{}")
    pairing = Path(config["pairing_dataset"])
    train = pairing / "eligible_train.parquet"
    validation = pairing / "eligible_validation.parquet"
    train.mkdir(parents=True)
    validation.mkdir(parents=True)
    train_part = train / "part-000000.parquet"
    validation_part = validation / "part-000000.parquet"
    pq.write_table(
        pa.table({"sample_id": ["train-sample"], "split": ["validation"]}),
        train_part,
    )
    pq.write_table(pa.table({"sample_id": ["validation-sample"]}), validation_part)
    source_hashes = {str(path): sha256_file(path) for path in (train_part, validation_part)}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="split contradiction"):
        construct_sidecars(config_path, mode="pilot")
    heartbeat = json.loads((Path(config["pilot_output_dir"]) / "heartbeat.json").read_text())
    assert heartbeat["status"] == "failed"
    assert heartbeat["stage"] == "pilot_selection"
    assert heartbeat["error_type"] == "ValueError"
    assert heartbeat["latest_committed_journal_position"] == {
        "processed_samples": 0,
        "unit_id": None,
        "unit_index": None,
        "physical_dataset_name": None,
        "split": None,
    }
    assert heartbeat["resumable"] is False
    assert heartbeat["resumability_status"] == "not_resumable"
    assert source_hashes == {str(path): sha256_file(path) for path in (train_part, validation_part)}


def test_observed_pre_shard_ownership_failure_publishes_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase0 = _phase0(tmp_path / "authorization")
    config = _config(tmp_path, phase0)
    Path(config["normalization_source"]).write_text("{}")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    def observed_failure(*_args: object, **_kwargs: object) -> set[tuple[str, str]]:
        raise ValueError("Unknown physical pairing dataset ownership: train")

    monkeypatch.setattr(sidecar_module, "_pilot_ids", observed_failure)
    with pytest.raises(ValueError, match="ownership: train"):
        construct_sidecars(config_path, mode="pilot")
    heartbeat = json.loads((Path(config["pilot_output_dir"]) / "heartbeat.json").read_text())
    assert heartbeat["status"] == "failed"
    assert heartbeat["stage"] == "pilot_selection"
    assert heartbeat["error_type"] == "ValueError"
    assert heartbeat["latest_committed_journal_position"]["unit_id"] is None
    assert heartbeat["resumability_status"] == "not_resumable"


def test_non_authorizing_performance_sweep_selects_only_equivalent_twenty_percent_gain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase0 = _phase0(tmp_path / "authorization")
    config = _config(tmp_path, phase0)
    config.update(
        worker_count=1,
        worker_queue_size=2,
        maximum_stage_latency_samples=8192,
        feature_backend="cpu",
        cuda_residue_token_budget=128,
        maximum_cuda_allocated_mib=6144,
        performance_benchmark={
            "version": 2,
            "output_root": str(tmp_path / "performance"),
            "panel_sample_count": 512,
            "projected_full_sample_count": 1000,
            "maximum_rss_mib": 4096,
            "minimum_end_to_end_speedup_fraction": 0.20,
        },
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    def synthetic_construct(path: Path, *, mode: str, resume: bool) -> dict[str, object]:
        case = load_yaml(path)
        assert mode == "pilot"
        assert resume is False
        workers = int(case["worker_count"])
        requested_backend = str(case["feature_backend"])
        parse_mode = str(case["mmcif_parse_mode"])
        throughput = 1.0 if parse_mode == "legacy_double_parse" else 1.3 if workers == 1 else 1.1
        return {
            "status": "completed",
            "worker_count": workers,
            "requested_feature_backend": requested_backend,
            "effective_feature_backend": requested_backend,
            "mmcif_parse_mode": str(case["mmcif_parse_mode"]),
            "feature_backend_fallback_reason": None,
            "cuda_feature_equivalence_passed": requested_backend == "cuda",
            "cuda_feature_float32_tolerance": sidecar_module.CUDA_FEATURE_FLOAT32_ATOL,
            "samples_per_second": throughput,
            "eligible_samples_per_second": throughput * 0.9,
            "peak_rss_mib": 1000.0,
            "peak_cuda_allocated_mib": 100.0 if requested_backend == "cuda" else None,
            "cpu_utilization_percent": 180.0 if workers > 1 else 95.0,
            "stage_timing": StageProfiler().report(1.0),
            "sample_order_sha256": "same-order",
            "definitive_observed_counts": {"total": 512, "eligible": 500, "excluded": 12},
            "observed_split_counts": {"train": {"total": 256}, "validation": {"total": 256}},
            "exclusion_reason_counts": {"resolved": 12},
            "shards": [{"path": "train/part.parquet", "sha256": "same-science"}],
            "authorizes_definitive_dataset": False,
            "authorizes_training": False,
        }

    monkeypatch.setattr(sidecar_module, "construct_sidecars", synthetic_construct)
    report = run_performance_benchmark(config_path)
    assert report["mode"] == "non_authorizing_performance_benchmark_v2"
    assert report["authorizes_definitive_dataset"] is False
    assert report["authorizes_training"] is False
    assert len(report["cases"]) == 4
    selected = report["recommended_production_configuration"]
    assert selected["case"] == "cpu_1_single_parse"
    assert selected["worker_count"] == 1
    assert selected["feature_backend"] == "cpu"
    assert selected["recommendation_type"] == "measured_optimization_winner"
    assert selected["measured_speedup_fraction"] == pytest.approx(0.3)
    assert [item["name"] for item in report["cases"]] == [
        "baseline_cpu_1",
        "cpu_1_single_parse",
        "cpu_2_workers",
        "cuda_production_features",
    ]
    assert report["cases"][-1]["status"] == "not_scheduled"
    assert all("case_peak_rss_mib" in item for item in report["cases"][:3])
    assert json.loads((tmp_path / "performance" / "benchmark_report.json").read_text()) == report


def test_performance_sweep_retains_safe_fallback_without_calling_it_a_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase0 = _phase0(tmp_path / "authorization")
    config = _config(tmp_path, phase0)
    config.update(
        worker_count=1,
        worker_queue_size=2,
        maximum_stage_latency_samples=8192,
        feature_backend="cpu",
        performance_benchmark={
            "version": 2,
            "output_root": str(tmp_path / "performance"),
            "panel_sample_count": 512,
            "projected_full_sample_count": 1000,
            "maximum_rss_mib": 4096,
            "minimum_end_to_end_speedup_fraction": 0.20,
        },
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    def equal_construct(path: Path, *, mode: str, resume: bool) -> dict[str, object]:
        case = load_yaml(path)
        return {
            "status": "completed",
            "worker_count": case["worker_count"],
            "requested_feature_backend": "cpu",
            "effective_feature_backend": "cpu",
            "mmcif_parse_mode": case["mmcif_parse_mode"],
            "feature_backend_fallback_reason": None,
            "cuda_feature_equivalence_passed": None,
            "cuda_feature_float32_tolerance": sidecar_module.CUDA_FEATURE_FLOAT32_ATOL,
            "samples_per_second": 1.0,
            "eligible_samples_per_second": 0.9,
            "peak_rss_mib": 900.0,
            "peak_cuda_allocated_mib": None,
            "cpu_utilization_percent": 100.0,
            "stage_timing": StageProfiler().report(1.0),
            "sample_order_sha256": "same-order",
            "definitive_observed_counts": {"total": 512, "eligible": 500, "excluded": 12},
            "observed_split_counts": {"train": {"total": 256}, "validation": {"total": 256}},
            "exclusion_reason_counts": {"resolved": 12},
            "shards": [{"path": "train/part.parquet", "sha256": "same-science"}],
            "authorizes_definitive_dataset": False,
            "authorizes_training": False,
        }

    monkeypatch.setattr(sidecar_module, "construct_sidecars", equal_construct)
    report = run_performance_benchmark(config_path)
    assert report["optimization_winner"] is None
    selected = report["recommended_production_configuration"]
    assert selected["case"] == "baseline_cpu_1"
    assert selected["recommendation_type"] == "verified_safe_fallback"
    assert selected["optimization_winner"] is False


def test_benchmark_failure_publishes_terminal_heartbeat_and_partial_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase0 = _phase0(tmp_path / "authorization")
    config = _config(tmp_path, phase0)
    config.update(
        worker_count=1,
        worker_queue_size=2,
        maximum_stage_latency_samples=8192,
        feature_backend="cpu",
        performance_benchmark={
            "version": 2,
            "output_root": str(tmp_path / "performance"),
            "panel_sample_count": 512,
            "projected_full_sample_count": 1000,
            "maximum_rss_mib": 4096,
            "minimum_end_to_end_speedup_fraction": 0.20,
        },
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    def fail(_path: Path, *, mode: str, resume: bool) -> dict[str, object]:
        raise ValueError("synthetic scientific contradiction")

    monkeypatch.setattr(sidecar_module, "construct_sidecars", fail)
    with pytest.raises(ValueError, match="scientific contradiction"):
        run_performance_benchmark(config_path)
    heartbeat = json.loads((tmp_path / "performance" / "heartbeat.json").read_text())
    assert heartbeat["status"] == "failed"
    assert heartbeat["failed_case"] == "baseline_cpu_1"
    assert heartbeat["completed_cases"] == []
    assert heartbeat["completed_utc"]
    assert heartbeat["partial_report_sha256"] == sha256_file(heartbeat["partial_report_path"])
    assert heartbeat["resumability_status"] == "new_output_required"
    assert heartbeat["fallback_production_configuration"]["case"] == "baseline_cpu_1"


def test_cuda_pseudo_cb_diagnostics_ignore_native_cb_and_bound_scientific_mismatch() -> None:
    row = {
        "sample_id": "sample",
        "residue_ids": ["42"],
        "cb_coordinates": [[1.0, 2.0, 3.0]],
        "n_mask": [True],
        "ca_mask": [True],
        "c_mask": [True],
    }
    observed = np.asarray([1.01, 2.0, 3.0], dtype=np.float32)
    assert (
        sidecar_module._cuda_pseudo_cb_contradiction(
            row,
            residue_index=0,
            cb_source=1,
            observed=observed,
            input_dtype=np.dtype(np.float32),
        )
        is None
    )
    diagnostic = sidecar_module._cuda_pseudo_cb_contradiction(
        row,
        residue_index=0,
        cb_source=0,
        observed=observed,
        input_dtype=np.dtype(np.float32),
    )
    assert diagnostic is not None
    assert diagnostic["native_cb_compared"] is False
    assert diagnostic["cb_source_meaning"] == "pseudo_cb"
    assert diagnostic["maximum_coordinate_error"] == pytest.approx(0.01)
    assert diagnostic["comparison_absolute_tolerance"] == sidecar_module.CUDA_FEATURE_FLOAT32_ATOL


def test_benchmark_equivalence_rejects_real_scientific_difference() -> None:
    reference = {
        "sample_order_sha256": "order",
        "definitive_observed_counts": {"eligible": 2},
        "observed_split_counts": {"train": 1, "validation": 1},
        "exclusion_reason_counts": {},
        "shard_scientific_hashes": {"train/part.parquet": "a"},
        "authorizes_definitive_dataset": False,
        "authorizes_training": False,
    }
    candidate = {**reference, "shard_scientific_hashes": {"train/part.parquet": "different"}}
    result = sidecar_module._benchmark_equivalence(reference, candidate)
    assert result["equivalent"] is False
    assert result["identical_tensor_shards"] is False
    assert result["mismatching_fields"] == ["shard_scientific_hashes"]
