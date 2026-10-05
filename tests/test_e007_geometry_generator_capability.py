from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from protein_distance_diffusion.evaluation import e007_geometry_generator_capability as audit


def _polymer(length: int, *, reflected: bool = False, phase: float = 0.0) -> np.ndarray:
    index = np.arange(length, dtype=np.float64)
    coordinates = np.column_stack((1.45 * index, 2.0 * np.sin(index * 1.1 + phase), 2.0 * np.cos(index * 1.1 + phase)))
    if reflected:
        coordinates[:, 2] *= -1
    return coordinates


def _config() -> dict:
    return {
        "metrics": {
            "local_distance_offsets": [1, 2, 3],
            "contact_thresholds_angstrom": [6.0, 8.0, 10.0, 12.0],
            "long_range_minimum_separation": 24,
            "clash_distance_angstrom": 3.0,
            "discontinuity_distance_angstrom": 4.5,
            "near_duplicate_rmsd_angstrom": 0.25,
            "descriptor_cluster_distance": 1.0,
        },
        "panel": {"lengths": [64, 128, 256, 384, 500], "samples_per_length": 32},
        "aggregation": {
            "bootstrap_replicates": 20,
            "bootstrap_seed": 17,
            "confidence_level": 0.95,
            "maximum_failure_examples": 20,
        },
    }


def _strata() -> list[dict[str, int | str]]:
    return [
        {"name": "1-64", "minimum": 1, "maximum": 64},
        {"name": "65-128", "minimum": 65, "maximum": 128},
        {"name": "129-256", "minimum": 129, "maximum": 256},
        {"name": "257-384", "minimum": 257, "maximum": 384},
        {"name": "385-500", "minimum": 385, "maximum": 500},
    ]


def _selection_config(path: Path, rows: list[dict], *, lengths: list[int], count: int, maximum: int = 4) -> dict:
    counts = Counter(int(row["length"]) for row in rows)
    return {
        "clean_validation": {"manifest_path": str(path), "expected_rows": len(rows)},
        "panel": {
            "lengths": lengths,
            "samples_per_length": count,
            "selection_seed": 31,
            "selection_version": "test_nearest_v2",
            "maximum_reference_length_mismatch": maximum,
            "length_strata": _strata(),
            "observed_exact_available_counts": {length: counts[length] for length in lengths},
            "observed_nearby_counts_within_maximum_mismatch": {
                length: {
                    candidate: counts[candidate]
                    for candidate in range(max(1, length - maximum), min(500, length + maximum) + 1)
                    if candidate != length and counts[candidate]
                }
                for length in lengths
            },
        },
    }


def _manifest_row(sample_id: str, length: int, row_index: int = 0) -> dict:
    return {
        "sample_id": sample_id,
        "split": "validation",
        "coordinate_accepted": True,
        "length": length,
        "length_stratum": "synthetic",
        "source_path": f"raw/{sample_id}.cif.gz",
        "dataset_shard_path": f"validation/{length}.parquet",
        "shard_row_index": row_index,
    }


@pytest.mark.parametrize("length", [64, 128, 500])
def test_geometry_metrics_are_finite_and_canonical(length: int) -> None:
    row, arrays = audit.geometry_metrics(
        _polymer(length), source="reference", sample_id=f"p{length}", length=length, config=_config()
    )
    assert row["finite_coordinates"]
    assert row["distance_diagonal_error_angstrom"] == 0
    assert row["distance_symmetry_error_angstrom"] == 0
    assert arrays["adjacent"].shape == (length - 1,)
    assert np.isfinite(np.asarray(list(audit._descriptor(row)))).all()


def test_reflection_reverses_pseudo_chirality_but_preserves_distances() -> None:
    native = _polymer(64)
    reflected = _polymer(64, reflected=True)
    native_row, native_arrays = audit.geometry_metrics(
        native, source="reference", sample_id="native", length=64, config=_config()
    )
    reflected_row, reflected_arrays = audit.geometry_metrics(
        reflected, source="generated", sample_id="reflected", length=64, config=_config()
    )
    assert np.allclose(native_arrays["adjacent"], reflected_arrays["adjacent"])
    assert np.allclose(
        native_arrays["signed_pseudo_dihedral_radians"],
        -reflected_arrays["signed_pseudo_dihedral_radians"],
    )
    assert native_row["radius_of_gyration_angstrom"] == pytest.approx(reflected_row["radius_of_gyration_angstrom"])


def test_chirality_summary_uses_independent_reflected_control() -> None:
    native = audit.geometry_metrics(_polymer(64), source="reference", sample_id="r", length=64, config=_config())[1]
    reflected = audit.geometry_metrics(
        _polymer(64, reflected=True), source="generated", sample_id="g", length=64, config=_config()
    )[1]
    result = audit.chirality_summary({"g": reflected}, {"r": native})
    assert result["mirror_likeness"] == "closer_to_reflected_control"
    assert result["not_native_residue_chirality"]


def test_stable_reference_selection_is_exact_length_and_order_independent(tmp_path: Path) -> None:
    rows = []
    for length in (64, 128, 256, 384, 500):
        for index in range(40):
            rows.append(_manifest_row(f"p{length}_{index}", length, index))
    path = tmp_path / "clean.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    config = _selection_config(path, rows, lengths=[64, 128, 256, 384, 500], count=32)
    first = audit.select_reference_manifest(config)
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), path)
    second = audit.select_reference_manifest(config)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    observed = [row["length"] for row in first]
    assert {value: observed.count(value) for value in set(observed)} == {
        64: 32,
        128: 32,
        256: 32,
        384: 32,
        500: 32,
    }


def test_reference_selection_rejects_underfill(tmp_path: Path) -> None:
    path = tmp_path / "clean.parquet"
    pq.write_table(
        pa.Table.from_pylist([_manifest_row("only", 64)]),
        path,
    )
    rows = [_manifest_row("only", 64)]
    config = _selection_config(path, rows, lengths=[64], count=2)
    with pytest.raises(ValueError, match="underfill"):
        audit.select_reference_manifest(config)


def test_measured_384_underfill_uses_three_distinct_383_controls(tmp_path: Path) -> None:
    rows = [_manifest_row(f"exact-{index}", 384, index) for index in range(29)]
    rows.extend(_manifest_row(f"near-{index}", 383, 29 + index) for index in range(10))
    path = tmp_path / "clean.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    config = _selection_config(path, rows, lengths=[384], count=32)
    selected, diagnostics = audit.select_reference_panel(config)
    assert len(selected) == len({row["sample_id"] for row in selected}) == 32
    assert Counter(row["actual_length"] for row in selected) == {384: 29, 383: 3}
    assert diagnostics["exact_match_count"] == 29
    assert diagnostics["nearest_match_count"] == 3
    assert all(row["length_stratum"] == "257-384" for row in selected)


def test_length_500_uses_minimum_sufficient_four_residue_bound(tmp_path: Path) -> None:
    counts = {500: 4, 499: 1, 498: 5, 497: 15, 496: 11}
    rows = []
    for length, count in counts.items():
        rows.extend(_manifest_row(f"p{length}-{index}", length, len(rows)) for index in range(count))
    path = tmp_path / "clean.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    config = _selection_config(path, rows, lengths=[500], count=32)
    selected, diagnostics = audit.select_reference_panel(config)
    assert Counter(row["actual_length"] for row in selected) == {500: 4, 499: 1, 498: 5, 497: 15, 496: 7}
    assert diagnostics["maximum_observed_absolute_mismatch"] == 4
    config["panel"]["maximum_reference_length_mismatch"] = 3
    config["panel"]["observed_nearby_counts_within_maximum_mismatch"][500].pop(496)
    with pytest.raises(ValueError, match="bounded same-stratum reference underfill"):
        audit.select_reference_panel(config)


def test_nearest_selection_is_stable_under_row_reordering(tmp_path: Path) -> None:
    rows = [_manifest_row(f"e{index}", 384, index) for index in range(29)]
    rows.extend(_manifest_row(f"n{index}", 383, 29 + index) for index in range(12))
    path = tmp_path / "clean.parquet"
    config = _selection_config(path, rows, lengths=[384], count=32)
    pq.write_table(pa.Table.from_pylist(rows), path)
    first = audit.select_reference_manifest(config)
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), path)
    second = audit.select_reference_manifest(config)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]


def test_distribution_bootstrap_is_deterministic() -> None:
    left = np.asarray([1.0, 2.0, 3.0])
    right = np.asarray([0.0, 1.0, 2.0])
    first = audit._bootstrap_difference(left, right, replicates=100, seed=4, confidence=0.95)
    second = audit._bootstrap_difference(left, right, replicates=100, seed=4, confidence=0.95)
    assert first == second
    assert first["mean_difference_generated_minus_reference"] == pytest.approx(1.0)


def test_near_duplicate_analysis_is_rigid_transform_invariant() -> None:
    coordinates = _polymer(64)
    angle = 0.4
    rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    result = audit._near_duplicate_summary(
        {"a": coordinates, "b": coordinates @ rotation.T + np.asarray([3.0, -2.0, 1.0])}, 1e-6
    )
    assert result["near_duplicate_count"] == 1


def test_plan_only_is_output_free_and_does_not_scan_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "version": audit.VERSION,
        "output_dir": str(tmp_path / "output"),
        "selected_checkpoint": {"optimizer_update": 9000},
        "phase3h": {"expected_step9000_samples": 160},
        "panel": {
            "lengths": [64, 128, 256, 384, 500],
            "samples_per_length": 32,
            "maximum_reference_length_mismatch": 4,
            "length_strata": _strata(),
            "selection_version": "test",
            "observed_exact_available_counts": {64: 32, 128: 32, 256: 32, 384: 29, 500: 4},
            "observed_nearby_counts_within_maximum_mismatch": {},
        },
        "prospective_phase3j": {"samples_per_length": 100},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    called = {}

    def fake_verify(_config, *, full):
        called["full"] = full
        return {"hashes": {}, "dataset_authorization": {"dataset_payload_scanned": False}}

    monkeypatch.setattr(audit, "verify_prerequisites", fake_verify)
    result = audit.plan_geometry_generator_capability(path)
    assert called == {"full": False}
    assert result["model_created"] is False
    assert result["sampling_performed"] is False
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / ".output.inprogress").exists()


def test_reference_provenance_uses_hash_verified_relocation(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    shard = dataset / "validation" / "part.parquet"
    shard.parent.mkdir(parents=True)
    recorded = tmp_path / "recorded"
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    source = relocated / "source.cif.gz"
    npz = relocated / "sample.npz"
    source.write_bytes(b"source")
    npz.write_bytes(b"npz")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    npz_sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    coordinates = _polymer(64).astype(np.float32)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": "sample",
                    "split": "validation",
                    "sequence": "A" * 64,
                    "ca_coordinates": coordinates.tolist(),
                    "ca_mask": [True] * 64,
                    "chain_continuity_mask": [True] * 63,
                    "chain_break_mask": [False] * 63,
                    "source_path": str(recorded / "source.cif.gz"),
                    "source_sha256": source_sha,
                    "npz_path": str(recorded / "sample.npz"),
                    "npz_sha256": npz_sha,
                }
            ]
        ),
        shard,
    )
    config = {
        "dataset": {
            "root": str(dataset),
            "protected_input_relocations": [{"recorded_root": str(recorded), "verification_root": str(relocated)}],
        },
        "clean_validation": {"manifest_sha256": "manifest-sha"},
    }
    record = {
        "sample_id": "sample",
        "length": 64,
        "target_length": 64,
        "actual_length": 64,
        "signed_length_mismatch": 0,
        "absolute_length_mismatch": 0,
        "match_type": "exact",
        "length_stratum": "1-64",
        "clean_validation_manifest_member": True,
        "dataset_shard_path": "validation/part.parquet",
        "shard_row_index": 0,
        "selection_rank": "rank",
        "selection_version": "v2",
    }
    loaded, provenance = audit.load_reference_coordinates(config, record)
    assert loaded.shape == (64, 3)
    assert provenance["source_relocation_resolution"] == "relocated_verification_path"
    assert provenance["npz_relocation_resolution"] == "relocated_verification_path"
    assert provenance["source_sha256"] == source_sha
    assert provenance["clean_validation_manifest_member"] is True


def test_terminal_selection_failure_heartbeat_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "phase3i"
    config = {
        "version": audit.VERSION,
        "output_dir": str(output),
        "selected_checkpoint": {"optimizer_update": 9000},
        "phase3h": {"expected_step9000_samples": 160},
        "panel": {
            "lengths": [64, 128, 256, 384, 500],
            "samples_per_length": 32,
            "maximum_reference_length_mismatch": 4,
            "length_strata": _strata(),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(audit, "verify_prerequisites", lambda *_args, **_kwargs: {"verified": True})

    def fail_selection(_config):
        raise ValueError("bounded panel unavailable")

    monkeypatch.setattr(audit, "select_reference_panel", fail_selection)
    with pytest.raises(ValueError, match="bounded panel unavailable"):
        audit.audit_geometry_generator_capability(config_path)
    heartbeat = json.loads((tmp_path / ".phase3i.inprogress" / "heartbeat.json").read_text())
    assert heartbeat["status"] == "failed"
    assert heartbeat["stage"] == "reference_selection"
    assert heartbeat["generated_coordinates_loaded"] == 0
    assert heartbeat["reference_coordinates_loaded"] == 0
    assert heartbeat["model_created"] is False
    assert heartbeat["optimizer_updates"] == 0


def test_inventory_refuses_hash_change(tmp_path: Path) -> None:
    root = tmp_path / "phase3h"
    root.mkdir()
    required = {
        "report": "report.json",
        "protocol": "protocol.json",
        "sample_metrics": "sample_metrics.parquet",
        "artifact_inventory": "artifact_inventory.json",
        "block_inventory": "block_inventory.json",
        "checkpoint_pareto": "checkpoint_pareto.json",
        "paired_comparisons": "paired_comparisons.json",
    }
    for filename in required.values():
        (root / filename).write_text("x")
    config = {"phase3h": {"root": str(root), "expected_inventory_entries": 0}}
    for key, filename in required.items():
        config["phase3h"][f"{key}_sha256"] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
    config["phase3h"]["artifact_inventory_aggregate_sha256"] = "bad"
    with pytest.raises((ValueError, json.JSONDecodeError)):
        audit._verify_phase3h_inventory(config, verify_entries=True)


def test_non_authorizing_contract_is_complete() -> None:
    assert all(value is False or value == 0 for value in audit.NON_AUTHORIZING.values())
    assert audit.NON_AUTHORIZING["authorizes_training"] is False
    assert audit.NON_AUTHORIZING["sampling_performed"] is False
