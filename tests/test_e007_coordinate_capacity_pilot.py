from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from protein_distance_diffusion.training import e007_coordinate_capacity_pilot as capacity

CONFIG_PATH = Path("configs/e007_coordinate_capacity_pilot_v1.yaml")


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _endpoint_row(name: str, seed: int, value: float) -> dict:
    return {
        "capacity": name,
        "seed": seed,
        "evaluations": {
            "1500": {
                "heldout": {
                    "means": {
                        "coordinate_v_mse": value,
                        "pair_distance_rmse_angstrom": value,
                    }
                },
                "unconditional_sampling": {"summary": {"global": {"near_duplicate_fraction": value / 10}}},
            }
        },
        "joint_polymer_quality": {"joint_error": value},
    }


def test_parameter_counts_and_architecture_contract() -> None:
    config = _config()
    assert capacity._parameter_counts(config) == {
        "small": 224_697,
        "medium": 1_902_581,
        "production": 7_586_505,
    }
    contract = capacity._architecture_contract(config)
    assert contract["implementation"] == "EquivariantPairCoordinateUNet"
    assert set(contract["observed_differing_fields"]) <= set(config["capacity_parameters"])
    assert contract["sequence_inputs"] is False
    assert contract["clean_coordinate_feature_inputs"] is False


def test_uniform_coordinate_v_objective_and_no_phase3c_terms() -> None:
    prediction = torch.tensor([[[1.0, 2.0, 3.0], [99.0, 99.0, 99.0]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False]])
    assert capacity.uniform_coordinate_v_loss(prediction, target, mask) == pytest.approx(14 / 3)
    config = _config()
    assert not set(capacity.FORBIDDEN_OBJECTIVE_FIELDS) & set(config)
    assert config["mixed_precision"] is False


def test_expanded_panels_are_balanced_frozen_and_disjoint() -> None:
    train, heldout, metadata = capacity._build_frozen_panels(_config())
    assert (len(train), len(heldout)) == (128, 64)
    assert set(metadata["family_counts"]["train"].values()) == {32}
    assert set(metadata["length_counts"]["heldout"].values()) == {16}
    assert metadata["parameter_seed_disjoint"]
    assert metadata["coordinate_hash_disjoint"]
    assert metadata["rigid_shape_hash_disjoint"]


def test_sampling_identity_is_deterministic_and_has_32_draws() -> None:
    config = _config()
    first = capacity.sampling_identity(config, 7301, 100)
    second = capacity.sampling_identity(config, 7301, 100)
    assert first == second
    assert first["count"] == 32
    assert len({(row[0], row[1]) for row in first["records"]}) == 32


def test_per_length_aggregation_and_bootstrap_are_deterministic() -> None:
    config = _config()
    rows = []
    fingerprints = []
    for length in config["sampling_lengths"]:
        for replicate in range(8):
            rows.append(
                {
                    "length": length,
                    "adjacent_distance_mean": 3.8 + replicate / 100,
                    "radius_of_gyration": length / 10 + replicate / 100,
                    "clash_fraction": 0.01,
                    "pair_distance_quantiles": [1.0, 2.0, 3.0],
                    "contact_density_8a": 0.2,
                    "contact_density_6a": 0.1,
                    "contact_density_10a": 0.3,
                    "neighborhood_count_mean_8a": 4.0,
                    "neighborhood_count_std_8a": 1.0,
                    "adjacent_distance_std": 0.1,
                    "finite_coordinates": True,
                    "symmetry_error": 0.0,
                    "diagonal_error": 0.0,
                    "maximum_triangle_violation": 0.0,
                    "centred_gram_negative_eigenmass_fraction": 0.0,
                    "rank3_residual_fraction": 0.0,
                    "rank3_reconstruction_error": 0.0,
                }
            )
            fingerprints.append(torch.arange(21, dtype=torch.float64) + replicate)
    targets = {
        str(length): {
            "adjacent_distance_mean": 3.8,
            "radius_of_gyration_mean": length / 10,
            "clash_fraction_mean": 0.01,
        }
        for length in config["sampling_lengths"]
    }
    targets["global"] = {
        "adjacent_distance_mean": 3.8,
        "radius_of_gyration_mean": np.mean(config["sampling_lengths"]) / 10,
        "clash_fraction_mean": 0.01,
    }
    first = capacity._summarize_sampling(rows, fingerprints, targets, config)
    second = capacity._summarize_sampling(rows, fingerprints, targets, config)
    assert first == second
    assert {row["count"] for row in first["by_length"].values()} == {8}
    assert first["global"]["count"] == 32


def test_duplicate_detection_is_bounded() -> None:
    rows = [{"length": 16} for _ in range(30)]
    fingerprints = [torch.zeros(21) for _ in rows]
    result = capacity._duplicate_summary(rows, fingerprints)
    assert result["exact_duplicate_count"] == 435
    assert result["near_duplicate_fraction"] == 1.0
    assert len(result["bounded_near_duplicate_examples"]) == 20


def test_paired_bootstrap_requires_supported_material_gain() -> None:
    baseline = [_endpoint_row("small", seed, 1.0) for seed in (7301, 7302, 7303)]
    candidate = [_endpoint_row("medium", seed, 0.75) for seed in (7301, 7302, 7303)]
    evidence = capacity._paired_pareto_evidence(
        baseline,
        candidate,
        _config()["decision_thresholds"],
        replicates=100,
    )
    assert evidence["passed"]
    assert evidence["material_supported_metrics"]


def test_classification_uses_smallest_eligible_without_pareto(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [_endpoint_row(name, seed, 1.0) for name in capacity.CAPACITIES for seed in (7301, 7302, 7303)]
    for row in results:
        row["finite_losses_and_gradients"] = True
    monkeypatch.setattr(capacity, "_capacity_eligible", lambda rows, thresholds: True)
    monkeypatch.setattr(capacity, "_per_length_regression", lambda *args: {"passed": True, "failures": []})
    monkeypatch.setattr(capacity, "_capacity_vector", lambda rows: {"value": 1.0})
    monkeypatch.setattr(
        capacity,
        "_paired_pareto_evidence",
        lambda *args, **kwargs: {"passed": False},
    )
    classification, evidence = capacity.classify_capacity(results, _config()["decision_thresholds"])
    assert classification == "small_capacity_sufficient"
    assert evidence["selected_capacity"] == "small"


def test_plan_is_read_only_and_refuses_existing_output(tmp_path: Path) -> None:
    config = _config()
    config["output_dir"] = str(tmp_path / "capacity")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    plan = capacity.plan_capacity_pilot(path)
    assert plan["total_planned_successful_updates"] == 13_500
    assert plan["unconditional_samples_per_capacity_seed_checkpoint"] == 32
    assert plan["optimizer_created"] is False
    assert not Path(config["output_dir"]).exists()
    Path(config["output_dir"]).mkdir()
    with pytest.raises(FileExistsError):
        capacity.plan_capacity_pilot(path)


def test_mocked_publication_is_atomic_non_authorizing_and_preserves_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    output = tmp_path / "capacity"
    config["output_dir"] = str(output)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    plan = {
        "status": "planned_non_authorizing",
        "output_dir": str(output),
        **capacity.NON_AUTHORIZING,
    }
    monkeypatch.setattr(capacity, "plan_capacity_pilot", lambda unused: plan)
    monkeypatch.setattr(capacity, "_verify_prerequisites", lambda unused: {"hashes": {"source": "same"}})
    monkeypatch.setattr(
        capacity,
        "_run_isolated",
        lambda unused_path, name, seed, unused_dir: {
            "capacity": name,
            "seed": seed,
            "successful_updates": 1500,
        },
    )
    monkeypatch.setattr(capacity, "verify_pairing", lambda rows: {"passed": True})
    monkeypatch.setattr(
        capacity,
        "classify_capacity",
        lambda rows, thresholds: ("small_capacity_sufficient", {"selected_capacity": "small"}),
    )
    report = capacity.run_capacity_pilot(path)
    assert report["successful_optimizer_updates"] == 13_500
    assert report["protected_inputs_unchanged"]
    assert all(report[field] is False for field in capacity.NON_AUTHORIZING)
    assert output.is_dir()
    assert not output.with_name(f".{output.name}.inprogress").exists()
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["status"] == "completed"
    assert protocol["authorizes_training"] is False
