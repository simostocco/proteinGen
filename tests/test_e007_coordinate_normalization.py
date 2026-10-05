from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from protein_distance_diffusion.data import e007_coordinate_normalization as normalization
from protein_distance_diffusion.data.rich_geometry import RichDatasetAuthorization


def _projected_row(
    sample_id: str,
    coordinates: np.ndarray,
    *,
    mask: list[bool] | None = None,
    continuity: list[bool] | None = None,
    split: str = "train",
) -> dict:
    length = len(coordinates)
    mask = [True] * length if mask is None else mask
    continuity = [True] * max(length - 1, 0) if continuity is None else continuity
    return {
        "sample_id": sample_id,
        "split": split,
        "sequence_length": length,
        "coordinates": torch.tensor(coordinates, dtype=torch.float32),
        "residue_mask": torch.tensor(mask, dtype=torch.bool),
        "chain_continuity_mask": torch.tensor(continuity, dtype=torch.bool),
        "accepted_contiguous_single_chain": all(mask) and all(continuity),
        "source_sha256": "a" * 64,
        "npz_sha256": "b" * 64,
        "sidecar_schema_version": "e006_rich_geometry_sidecar_v2",
    }


def _accumulate(rows: list[dict]) -> dict:
    accumulator = normalization.CalibrationAccumulator(
        maximum_values=1000,
        percentiles=[0.25, 0.5, 0.75],
        clash_distance_angstrom=2.0,
    )
    strata = [{"name": "20-64", "minimum": 1, "maximum": 64}]
    for row in rows:
        stratum = normalization._length_stratum(row["sequence_length"], strata)
        reasons = accumulator.observe_candidate(row, stratum)
        if reasons:
            accumulator.reject(row, stratum, reasons)
        else:
            accumulator.accept(row, stratum)
    return accumulator.result()


def _reference_scale(rows: list[dict]) -> float:
    total = 0.0
    count = 0
    for row in rows:
        values = row["coordinates"].double().numpy()[row["residue_mask"].numpy()]
        centered = values - values.mean(axis=0, keepdims=True)
        total += float(np.square(centered).sum())
        count += len(values)
    return np.sqrt(total / (3 * count))


def _minimal_config(output: Path) -> dict:
    return {
        "version": normalization.NORMALIZATION_VERSION,
        "output_dir": str(output),
        "split": "train",
        "selection_policy": "contiguous_single_chain_complete_calpha_v1",
        "coordinate_units": "angstrom",
        "formula": normalization.ESTIMATOR_FORMULA,
        "length_strata": [{"name": "20-64", "minimum": 1, "maximum": 64}],
        "robust_percentiles": [0.25, 0.5, 0.75],
        "maximum_streaming_values": 1000,
        "clash_distance_angstrom": 2.0,
        "write_per_sample_audit": True,
        "dataset": {},
    }


def _authorization(root: Path) -> RichDatasetAuthorization:
    return RichDatasetAuthorization(
        root=root,
        protocol_sha256="1" * 64,
        schema_sha256="2" * 64,
        vocabulary_sha256="3" * 64,
        normalization_sha256="4" * 64,
        shard_inventory_sha256="5" * 64,
        split_counts={"train": 2, "validation": 0},
        observed_shard_hashes={"train/part-0.parquet": "6" * 64},
    )


def test_exact_estimator_matches_float64_in_memory_reference() -> None:
    rows = [
        _projected_row("a", np.asarray([[0, 0, 0], [3, 1, 0], [7, -1, 2]], dtype=np.float64)),
        _projected_row("b", np.asarray([[10, 2, 1], [13, 4, 2]], dtype=np.float64)),
    ]
    result = _accumulate(rows)
    assert result["coordinate_scale_angstrom"] == pytest.approx(_reference_scale(rows), rel=1e-14)
    assert result["token_weighted_rms_radius_angstrom"] == pytest.approx(
        np.sqrt(3) * result["coordinate_scale_angstrom"]
    )


def test_scale_is_translation_and_o3_invariant() -> None:
    coordinates = np.asarray([[0.2, 1.0, -2.0], [2.1, -1.0, 0.5], [4.3, 2.0, 1.2]])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    base = _accumulate([_projected_row("base", coordinates)])["coordinate_scale_angstrom"]
    translated = _accumulate([_projected_row("translated", coordinates + 1000)])["coordinate_scale_angstrom"]
    transformed = _accumulate([_projected_row("rotated", coordinates @ rotation)])["coordinate_scale_angstrom"]
    assert translated == pytest.approx(base, rel=3e-6)
    assert transformed == pytest.approx(base, rel=1e-6)


def test_masked_and_broken_rows_are_reported_without_silent_repair() -> None:
    coordinates = np.asarray([[0, 0, 0], [0, 0, 0], [7.6, 0, 0]], dtype=np.float64)
    missing = _projected_row("missing", coordinates, mask=[True, False, True])
    broken = _projected_row("broken", coordinates, continuity=[False, True])
    complete = _projected_row("complete", coordinates)
    result = _accumulate([missing, broken, complete])
    assert result["accepted_sample_count"] == 1
    assert result["rejected_sample_count"] == 2
    assert result["rejection_reason_counts"] == {"chain_break": 1, "missing_calpha": 1}


def test_no_per_protein_rescaling_and_protein_equal_sensitivity_are_distinct() -> None:
    small = _projected_row("small", np.asarray([[0, 0, 0], [2, 0, 0]], dtype=np.float64))
    large = _projected_row("large", np.asarray([[0, 0, 0], [20, 0, 0], [40, 0, 0]], dtype=np.float64))
    result = _accumulate([small, large])
    assert result["coordinate_scale_angstrom"] == pytest.approx(_reference_scale([small, large]))
    assert result["alternative_protein_equal_coordinate_scale_angstrom"] != pytest.approx(
        result["coordinate_scale_angstrom"]
    )


def test_duplicate_ids_hashes_and_length_strata_are_deterministic() -> None:
    row = _projected_row("duplicate", np.asarray([[0, 0, 0], [3.8, 0, 0]], dtype=np.float64))
    accumulator = normalization.CalibrationAccumulator(100, [0.5], 2.0)
    for candidate in (row, copy.deepcopy(row)):
        reasons = accumulator.observe_candidate(candidate, "20-64")
        accumulator.accept(candidate, "20-64") if not reasons else accumulator.reject(candidate, "20-64", reasons)
    result = accumulator.result()
    assert result["duplicate_sample_id_count"] == 1
    assert result["duplicate_coordinate_hash_count"] == 1
    assert result["length_strata"]["accepted"] == {"20-64": 2}
    assert result["fitted_sample_id_sha256"] == normalization._canonical_sha256(["duplicate", "duplicate"])


def test_config_rejects_validation_fit_and_wrong_units(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path / "out")
    path = tmp_path / "config.yaml"
    config["split"] = "validation"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="training split"):
        normalization._load_config(path)
    config["split"] = "train"
    config["coordinate_units"] = "nanometer"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Angstrom"):
        normalization._load_config(path)


def test_length_strata_have_exact_boundaries() -> None:
    strata = [
        {"name": "20-64", "minimum": 20, "maximum": 64},
        {"name": "65-128", "minimum": 65, "maximum": 128},
    ]
    assert normalization._length_stratum(64, strata) == "20-64"
    assert normalization._length_stratum(65, strata) == "65-128"
    with pytest.raises(ValueError):
        normalization._length_stratum(19, strata)


def test_plan_only_does_not_scan_payload_or_create_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out"
    config = _minimal_config(output)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        normalization,
        "verify_normalization_prerequisites",
        lambda unused: {
            "phase3d_hashes": {"phase3d_report": "a" * 64},
            "dataset_hashes": {"protocol": "b" * 64},
            "dataset": {"eligible_split_counts": {"train": 12}},
        },
    )
    monkeypatch.setattr(
        normalization,
        "E007CoordinateDataset",
        lambda *args, **kwargs: pytest.fail("plan-only scanned coordinate payloads"),
    )
    plan = normalization.plan_coordinate_normalization(path)
    assert plan["candidate_sample_count"] == 12
    assert plan["coordinate_payloads_scanned"] is False
    assert plan["optimizer_created"] is False
    assert not output.exists()


def test_plan_refuses_existing_final_or_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out"
    config = _minimal_config(output)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(normalization, "verify_normalization_prerequisites", lambda unused: {})
    output.mkdir()
    with pytest.raises(FileExistsError):
        normalization.plan_coordinate_normalization(path)


def test_atomic_mock_calibration_is_train_only_non_authorizing_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    config = _minimal_config(output)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    rows = [
        _projected_row("a", np.asarray([[0, 0, 0], [3.8, 0, 0]], dtype=np.float64)),
        _projected_row("b", np.asarray([[0, 0, 0], [3.8, 1, 0], [7.6, 0, 0]], dtype=np.float64)),
    ]
    original = [row["coordinates"].clone() for row in rows]
    authorization = _authorization(tmp_path)
    plan = {
        "status": "planned_phase3e_a_non_authorizing",
        "output_dir": str(output),
        "prerequisite_hashes": {"dataset_protocol": "1" * 64},
        "configuration_sha256": "2" * 64,
        **normalization.NON_AUTHORIZING,
    }
    monkeypatch.setattr(normalization, "plan_coordinate_normalization", lambda unused: plan)
    monkeypatch.setattr(normalization, "_authorize_dataset", lambda unused: authorization)
    monkeypatch.setattr(normalization, "E007CoordinateDataset", lambda unused, split: rows)
    report = normalization.calibrate_coordinate_normalization(path)
    assert report["status"] == "completed_non_authorizing"
    assert report["protected_inputs_unchanged"]
    assert report["validation_coordinates_inspected"] is False
    assert {key: report[key] for key in normalization.NON_AUTHORIZING} == normalization.NON_AUTHORIZING
    assert output.is_dir()
    assert not output.with_name(f".{output.name}.inprogress").exists()
    assert all(torch.equal(before, row["coordinates"]) for before, row in zip(original, rows, strict=True))
    artifact = json.loads((output / "normalization.json").read_text())
    assert artifact["estimator_formula"] == normalization.ESTIMATOR_FORMULA
    assert artifact["fitting_split"] == "train"
    assert artifact["coordinate_units"] == "angstrom"
    assert json.loads((output / "protocol.json").read_text())["authorizes_training"] is False


def test_calibration_refuses_validation_row_before_fitting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out"
    config = _minimal_config(output)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    authorization = _authorization(tmp_path)
    row = _projected_row("validation", np.asarray([[0, 0, 0], [3.8, 0, 0]]), split="validation")
    monkeypatch.setattr(
        normalization,
        "plan_coordinate_normalization",
        lambda unused: {
            "output_dir": str(output),
            "prerequisite_hashes": {},
            "configuration_sha256": "2" * 64,
            **normalization.NON_AUTHORIZING,
        },
    )
    monkeypatch.setattr(normalization, "_authorize_dataset", lambda unused: authorization)
    monkeypatch.setattr(normalization, "E007CoordinateDataset", lambda unused, split: [row])
    with pytest.raises(ValueError, match="validation leakage"):
        normalization.calibrate_coordinate_normalization(path)
    assert not output.exists()
    heartbeat = json.loads((output.with_name(f".{output.name}.inprogress") / "heartbeat.json").read_text())
    assert heartbeat["status"] == "failed"
