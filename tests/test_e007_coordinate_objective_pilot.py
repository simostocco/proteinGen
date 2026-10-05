from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

import protein_distance_diffusion.training.e007_coordinate_objective_pilot as pilot
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e007_coordinate_losses import (
    ObjectiveCorrectionWeights,
    coordinate_objective_correction_losses,
)
from protein_distance_diffusion.training.e007_coordinate_smoke import _paired_diffusion_batch, build_polymer_panels

CONFIG_PATH = Path("configs/e007_coordinate_objective_correction_pilot_v1.yaml")


def _loss_inputs(lengths: tuple[int, ...] = (5, 3)) -> dict[str, torch.Tensor]:
    torch.manual_seed(19)
    side = max(lengths)
    clean = torch.randn((len(lengths), side, 3))
    predicted = clean + 0.1 * torch.randn_like(clean)
    mask = torch.zeros((len(lengths), side), dtype=torch.bool)
    continuity = torch.zeros((len(lengths), side - 1), dtype=torch.bool)
    for index, length in enumerate(lengths):
        mask[index, :length] = True
        continuity[index, : length - 1] = True
    clean = clean * mask[..., None]
    predicted = (predicted * mask[..., None]).detach().requires_grad_()
    return {
        "v_prediction": predicted,
        "v_target": torch.zeros_like(predicted),
        "predicted_clean_coordinates": predicted,
        "clean_coordinates": clean,
        "timesteps": torch.tensor([1] * len(lengths)),
        "timestep_weights": torch.ones(4),
        "residue_mask": mask,
        "chain_continuity_mask": continuity,
        "auxiliary_weights": ObjectiveCorrectionWeights(0.02, 0.02, 0.01, 0.005),
        "clash_distance_normalized": 0.3,
    }


def _write_config(tmp_path: Path, **updates: object) -> Path:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config.update(updates)
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _fake_result(arm: str, seed: int) -> dict:
    hashes = {
        field: f"{seed}:{field}"
        for field in (
            "initialization",
            "sample_order",
            "timesteps",
            "coordinate_noise",
            "targets",
            "evaluation_panel",
            "evaluation_corruptions",
            "sampling_initial_states",
            "sampling_stochastic_draws",
        )
    }
    return {
        "arm": arm,
        "seed": seed,
        "pairing_hashes": hashes,
        "finite_losses_and_gradients": True,
        "authorizes_training": False,
    }


def test_timestep_weights_are_exactly_normalized_and_monotonic() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    weights, bins, digest = pilot.normalized_timestep_weights(32, config["timestep_weight_bins"])
    assert len(weights) == 32
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-15)
    assert bool(torch.all(weights[1:] >= weights[:-1]))
    assert [record["name"] for record in bins] == [
        "very_low_noise",
        "low_noise",
        "intermediate_noise",
        "high_noise",
        "very_high_noise",
    ]
    assert digest == pilot._canonical_hash(
        [
            {
                "timestep": index,
                "weight": float(weight),
                "unnormalized_weight": float(
                    next(record["unnormalized_weight"] for record in bins if record["start"] <= index <= record["end"])
                ),
            }
            for index, weight in enumerate(weights)
        ]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quantiles_match_input_dtype_and_are_deterministic(dtype: torch.dtype) -> None:
    values = torch.arange(5, dtype=dtype)
    original = values.clone()
    first = pilot._matched_quantiles(values, [0.1, 0.5, 0.9])
    second = pilot._matched_quantiles(values, [0.1, 0.5, 0.9])
    assert first.dtype == dtype
    assert first.device == values.device
    assert torch.equal(first, second)
    assert torch.allclose(first, values.new_tensor([0.4, 2.0, 3.6]))
    assert values.dtype == dtype
    assert torch.equal(values, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quantiles_match_cuda_input_dtype_and_device(dtype: torch.dtype) -> None:
    values = torch.arange(5, dtype=dtype, device="cuda")
    result = pilot._matched_quantiles(values, [0.1, 0.5, 0.9])
    assert result.dtype == dtype
    assert result.device == values.device
    assert torch.allclose(result, values.new_tensor([0.4, 2.0, 3.6]))


def test_quantiles_reject_empty_or_fully_masked_values() -> None:
    with pytest.raises(ValueError, match="at least one valid value"):
        pilot._matched_quantiles(torch.empty(0, dtype=torch.float64), [0.5])
    with pytest.raises(ValueError, match="probabilities are empty"):
        pilot._matched_quantiles(torch.ones(1, dtype=torch.float32), [])


def test_sample_panel_quantiles_accept_float64_distance_diagnostics() -> None:
    class Model:
        def eval(self) -> Model:
            return self

    class Diffusion:
        def sample(self, _model: Model, *, length: int, seed: int, device: torch.device) -> dict:
            del seed
            coordinates = torch.arange(length * 3, dtype=torch.float32, device=device).reshape(1, length, 3)
            return {"coordinates": coordinates}

    config = {
        "sampling_lengths": [5],
        "sampling_seed": 7,
        "coordinate_scale_angstrom": 1.0,
    }
    first = pilot._sample_panel(Model(), Diffusion(), config, seed_offset=0, device=torch.device("cpu"))
    second = pilot._sample_panel(Model(), Diffusion(), config, seed_offset=0, device=torch.device("cpu"))
    assert first == second
    assert len(first["rows"][0]["adjacent_distance_quantiles"]) == 3
    assert len(first["rows"][0]["pair_distance_quantiles"]) == 3


def test_paired_draws_are_identical_for_all_arms() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    train, _, _ = build_polymer_panels(config)
    diffusion = CoordinateVPDiffusion(32)
    identities = []
    for _arm in pilot.ARMS:
        _, batch = _paired_diffusion_batch(
            train[0], diffusion, scale=10.0, draw_seed=7301000001, rotate=True, device=torch.device("cpu")
        )
        identities.append(
            (
                int(batch.timesteps.item()),
                pilot._tensor_sha256(batch.coordinate_noise),
                pilot._tensor_sha256(batch.coordinate_v_target),
            )
        )
    assert len(set(identities)) == 1


def test_objective_is_differentiable_per_sample_and_padding_invariant() -> None:
    inputs = _loss_inputs()
    losses = coordinate_objective_correction_losses(**inputs)
    losses["total"].backward()
    gradient = inputs["v_prediction"].grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[1, 3:]) == 0

    padded = _loss_inputs()
    padded["predicted_clean_coordinates"].retain_grad()
    padded["predicted_clean_coordinates"].data[1, 3:] = 1e6
    repeated = coordinate_objective_correction_losses(**padded)
    for name in ("raw_pair_distance", "raw_adjacent_distance", "raw_radius_of_gyration", "total"):
        assert torch.allclose(losses[name], repeated[name])

    one = _loss_inputs((5,))
    two = dict(one)
    for name in (
        "v_prediction",
        "v_target",
        "predicted_clean_coordinates",
        "clean_coordinates",
        "timesteps",
        "residue_mask",
        "chain_continuity_mask",
    ):
        value = one[name]
        two[name] = value.repeat((2,) + (1,) * (value.ndim - 1))
    doubled = coordinate_objective_correction_losses(**two)
    assert torch.allclose(one_loss := coordinate_objective_correction_losses(**one)["total"], doubled["total"])
    assert torch.isfinite(one_loss)


def test_geometry_auxiliaries_are_o3_invariant_and_finite_for_coincident_coordinates() -> None:
    inputs = _loss_inputs((6,))
    base = coordinate_objective_correction_losses(**inputs)
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    transformed = dict(inputs)
    transformed["clean_coordinates"] = inputs["clean_coordinates"] @ reflection + 7.0
    transformed["predicted_clean_coordinates"] = inputs["predicted_clean_coordinates"] @ reflection + 7.0
    rotated = coordinate_objective_correction_losses(**transformed)
    for name in ("raw_pair_distance", "raw_adjacent_distance", "raw_radius_of_gyration", "raw_steric_clash"):
        assert torch.allclose(base[name], rotated[name], atol=1e-6)

    coincident = _loss_inputs((4,))
    value = torch.zeros_like(coincident["clean_coordinates"], requires_grad=True)
    coincident["clean_coordinates"] = torch.zeros_like(value)
    coincident["predicted_clean_coordinates"] = value
    coincident["v_prediction"] = value
    coincident["v_target"] = torch.zeros_like(value)
    result = coordinate_objective_correction_losses(**coincident)
    result["total"].backward()
    assert all(torch.isfinite(item).all() for item in result.values())
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_auxiliary_scale_bound_accepts_equality_and_rejects_domination() -> None:
    losses = {
        "weighted_coordinate_v": torch.tensor(2.0),
        "raw_pair_distance": torch.tensor(10.0),
        "weighted_pair_distance": torch.tensor(0.2),
        "raw_adjacent_distance": torch.tensor(1.0),
        "weighted_adjacent_distance": torch.tensor(0.1),
        "raw_radius_of_gyration": torch.tensor(1.0),
        "weighted_radius_of_gyration": torch.tensor(0.1),
        "raw_steric_clash": torch.tensor(1.0),
        "weighted_steric_clash": torch.tensor(0.1),
        "weighted_auxiliary_total": torch.tensor(0.5),
    }
    bounds = {"maximum_each_to_primary_ratio": 0.1, "maximum_total_to_primary_ratio": 0.25}
    assert pilot.verify_auxiliary_scale_bound(losses, bounds)["passed"]
    losses["weighted_pair_distance"] = torch.tensor(0.21)
    with pytest.raises(ValueError, match="auxiliary scale bound failed"):
        pilot.verify_auxiliary_scale_bound(losses, bounds)


def test_pairing_evidence_detects_any_arm_divergence() -> None:
    rows = [_fake_result(arm, seed) for arm in pilot.ARMS for seed in (7301, 7302, 7303)]
    assert pilot.verify_pairing(rows)["passed"]
    rows[-1]["pairing_hashes"]["targets"] = "changed"
    with pytest.raises(ValueError, match="pairing contradiction"):
        pilot.verify_pairing(rows)


def test_gradient_norms_report_full_trunk_and_head_groups() -> None:
    gradients = {
        "pair_trunk.block.weight": torch.tensor([3.0]),
        "coefficient_head.weight": torch.tensor([4.0]),
    }
    assert pilot._gradient_norms(gradients) == {
        "full_model": 5.0,
        "coefficient_head": 4.0,
        "pair_grid_unet_trunk": 3.0,
    }


def test_evaluation_rows_are_externalized_to_deterministic_compressed_jsonl(tmp_path: Path) -> None:
    record = {
        "0": {
            "train": {"per_sample": [{"sample_id": "train:1", "metric": 1.0}]},
            "heldout": {"per_sample": [{"sample_id": "heldout:1", "metric": 2.0}]},
            "heldout_timestep_bins": {"high_noise": {"per_sample": [{"sample_id": "heldout:1", "metric": 3.0}]}},
        }
    }
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    first = pilot._externalize_evaluation_rows(left, deepcopy(record))
    second = pilot._externalize_evaluation_rows(right, deepcopy(record))
    assert first["row_count"] == 3
    assert first["sha256"] == second["sha256"]


@pytest.mark.parametrize("classification", pilot.CLASSIFICATIONS)
def test_classification_logic(classification: str, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_fake_result(arm, seed) for arm in pilot.ARMS for seed in (7301, 7302, 7303)]
    if classification == "invalid_execution":
        rows[0]["finite_losses_and_gradients"] = False
        assert pilot.classify_objective_correction(rows, {}, {})[0] == classification
        return
    passes = {
        "uniform_v_control": True,
        "high_noise_balanced_v": classification in {"objective_correction_verified", "high_noise_weighting_sufficient"},
        "high_noise_balanced_v_plus_x0_geometry": classification
        in {
            "objective_correction_verified",
            "x0_geometry_auxiliaries_required",
        },
    }
    monkeypatch.setattr(pilot, "_arm_passes_v2_thresholds", lambda values, _: passes[values[0]["arm"]])
    monkeypatch.setattr(
        pilot,
        "_pareto_vector",
        lambda _: {"metric": 1.0},
    )
    if classification == "objective_correction_regresses_quality":
        monkeypatch.setattr(pilot, "_regression_check", lambda *_: {"passed": False, "failures": ["metric"]})
    else:
        monkeypatch.setattr(pilot, "_regression_check", lambda *_: {"passed": True, "failures": []})
    if classification == "correction_improves_but_not_all_seeds":
        for row in rows:
            row["joint_polymer_quality"] = {"passed": row["arm"] != "uniform_v_control"}
    elif classification == "no_objective_correction_benefit":
        for row in rows:
            row["joint_polymer_quality"] = {"passed": False}
    observed, _ = pilot.classify_objective_correction(rows, {}, {})
    assert observed == classification


def test_plan_pins_v2_thresholds_is_read_only_and_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "phase3c"
    path = _write_config(tmp_path, output_dir=str(output))
    protected = [
        Path("reports/experiments/E007_matrix_sequence_cogeneration/coordinate_smoke_v1/report.json"),
        Path("reports/experiments/E007_matrix_sequence_cogeneration/coordinate_smoke_v2/report.json"),
        Path("reports/experiments/E007_matrix_sequence_cogeneration/coordinate_smoke_v2/protocol.json"),
    ]
    before = {str(item): hashlib.sha256(item.read_bytes()).hexdigest() for item in protected}
    plan = pilot.plan_objective_correction_pilot(path)
    assert plan["model_parameter_count"] == 224_697
    assert plan["total_planned_successful_optimizer_updates"] == 9_000
    assert plan["decision_thresholds"] == json.loads(protected[1].read_text())["decision_thresholds"]
    assert plan["optimizer_created"] is False
    assert plan["forward_executed"] is False
    assert plan["sampling_executed"] is False
    assert all(plan[name] is False for name in pilot.NON_AUTHORIZING)
    assert not output.exists()
    after = {str(item): hashlib.sha256(item.read_bytes()).hexdigest() for item in protected}
    assert before == after
    output.mkdir()
    with pytest.raises(FileExistsError):
        pilot.plan_objective_correction_pilot(path)


def test_atomic_publication_is_non_authorizing_and_stdout_payload_can_be_compact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "published"
    path = _write_config(tmp_path, output_dir=str(output))

    monkeypatch.setattr(pilot, "_protected_hashes", lambda _: {"protected": "same"})
    monkeypatch.setattr(
        pilot,
        "_run_isolated",
        lambda _path, arm, seed, _directory: _fake_result(arm, seed),
    )
    monkeypatch.setattr(pilot, "classify_objective_correction", lambda *_: ("no_objective_correction_benefit", {}))
    monkeypatch.setattr(pilot, "_bootstrap_comparisons", lambda *_: {})
    report = pilot.run_objective_correction_pilot(path)
    protocol = json.loads((output / "protocol.json").read_text())
    assert report["status"] == "completed"
    assert protocol["synthetic_only"] is True
    assert all(protocol[name] is False for name in pilot.NON_AUTHORIZING)
    assert not output.with_name(f".{output.name}.inprogress").exists()
    compact = {key: report[key] for key in ("status", "classification", "output_dir")}
    assert len(json.dumps(compact)) < 500
