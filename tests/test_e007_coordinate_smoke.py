from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

import protein_distance_diffusion.training.e007_coordinate_smoke as smoke
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion


def _base_config(tmp_path: Path) -> dict:
    production = yaml.safe_load(Path("configs/e007_coordinate_generator_v1.yaml").read_text())
    contract = Path("reports/experiments/E007_matrix_sequence_cogeneration/COORDINATE_MODEL_CONTRACT_V1.json")
    checkpoint = Path("outputs/recovered_full_b2_v_axial_edm_triangle_e004/checkpoints/final_validation_selected.pt")
    tiny_model = {
        "rbf_bins": 2,
        "rbf_min_distance": 0.0,
        "rbf_max_distance": 5.0,
        "lifting_epsilon": 1e-6,
        "base_channels": 2,
        "channel_multipliers": [1],
        "residual_blocks_per_level": 1,
        "dropout": 0.0,
        "group_norm_groups": 1,
        "attention_heads": 1,
        "use_bottleneck_attention": False,
        "use_pre_bottleneck_axial_attention": False,
        "axial_attention_heads": 1,
        "axial_attention_dropout": 0.0,
        "axial_attention_chunk_size": 8,
        "use_pre_bottleneck_triangle_multiplication": False,
        "triangle_hidden_channels": 2,
        "triangle_dropout": 0.0,
        "triangle_chunk_size": 1,
        "time_embedding_dim": 8,
        "length_embedding_dim": 8,
        "max_length": 500,
    }
    return {
        "version": smoke.SMOKE_VERSION,
        "output_dir": str(tmp_path / "smoke"),
        "device": "cpu",
        "seeds": [3, 4, 5],
        "lengths": [9],
        "families": list(smoke.FAMILIES),
        "train_replicates_per_family_length": 1,
        "heldout_replicates_per_family_length": 1,
        "train_parameter_seed_offset": 1000,
        "heldout_parameter_seed_offset": 9000,
        "bond_length_angstrom": 3.8,
        "coordinate_scale_angstrom": 10.0,
        "optimizer_updates_per_seed": 1,
        "evaluation_updates": [0, 1],
        "learning_rate": 1e-3,
        "gradient_clip_norm": 1.0,
        "diffusion_steps": 4,
        "batch_size": 1,
        "sampling_lengths": [9],
        "sampling_seed": 4,
        "metrics_frequency": 1,
        "phase3a": {
            "generator_config_path": "configs/e007_coordinate_generator_v1.yaml",
            "generator_config_sha256": hashlib.sha256(
                Path("configs/e007_coordinate_generator_v1.yaml").read_bytes()
            ).hexdigest(),
            "architecture_contract_path": str(contract),
            "architecture_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
            "e004_checkpoint_path": str(checkpoint),
            "e004_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "smoke_model": tiny_model,
        "loss": {
            "adjacent_weight": 0.02,
            "stratified_pair_weight": 0.02,
            "soft_contact_weight": 0.01,
            "steric_clash_weight": 0.005,
            "adjacent_huber_beta_angstrom": 0.25,
            "steric_clash_distance_angstrom": 3.0,
        },
        "decision_thresholds": {
            "minimum_train_v_mse_relative_improvement": 0.0,
            "minimum_heldout_v_mse_relative_improvement_per_seed": 0.0,
            "minimum_heldout_pair_rmse_relative_improvement_per_seed": 0.0,
            "minimum_sampling_adjacent_distribution_improvement": -1e9,
            "minimum_sampling_radius_distribution_improvement": -1e9,
            "equivariance_atol": 5e-5,
            "euclidean_scaled_tolerance": 5e-5,
            "maximum_optimizer_updates_per_seed": 2,
        },
        "_production_model": production["model"],
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({key: value for key, value in config.items() if not key.startswith("_")}))
    return path


def test_polymer_generation_is_deterministic_centered_and_has_expected_bonds() -> None:
    for family in smoke.FAMILIES:
        first = smoke.generate_polymer(family, 24, 91)
        second = smoke.generate_polymer(family, 24, 91)
        assert torch.equal(first, second)
        assert torch.allclose(first.mean(0), torch.zeros(3), atol=1e-6)
        bonds = torch.linalg.vector_norm(first[1:] - first[:-1], dim=-1)
        assert torch.allclose(bonds, torch.full_like(bonds, 3.8), atol=1e-5)


def test_panels_are_balanced_disjoint_and_hashed(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config["lengths"] = [16, 24, 32, 48]
    train, heldout, metadata = smoke.build_polymer_panels(config)
    assert len(train) == 16
    assert len(heldout) == 16
    assert set(metadata["family_counts"]["train"].values()) == {4}
    assert set(metadata["length_counts"]["heldout"].values()) == {4}
    assert metadata["parameter_seed_disjoint"]
    assert metadata["coordinate_hash_disjoint"]
    assert not ({sample.parameter_seed for sample in train} & {sample.parameter_seed for sample in heldout})


def test_rotation_augmentation_is_proper_and_preserves_distances() -> None:
    rotation = smoke.proper_rotation(7)
    coordinates = smoke.generate_polymer("helix", 17, 4)
    assert torch.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(
        torch.cdist(coordinates, coordinates), torch.cdist(coordinates @ rotation, coordinates @ rotation), atol=1e-5
    )


def test_model_and_training_features_have_no_clean_or_sequence_input(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    train, _, _ = smoke.build_polymer_panels(config)
    diffusion = CoordinateVPDiffusion(4)
    batch, diffused = smoke._paired_diffusion_batch(
        train[0], diffusion, scale=10.0, draw_seed=9, rotate=True, device=torch.device("cpu")
    )
    assert smoke.model_has_no_sequence_inputs()
    assert set(batch) == {"coordinates", "residue_mask", "continuity", "lengths"}
    assert diffused.noisy_coordinates.data_ptr() != batch["coordinates"].data_ptr()


def test_finite_guard_rejects_nonfinite_loss_and_missing_gradient() -> None:
    model = torch.nn.Linear(2, 1)
    with pytest.raises(FloatingPointError, match="non-finite synthetic loss"):
        smoke.require_finite_training_state(torch.tensor(float("nan")), model, seed=1, update=1)
    loss = model(torch.ones((1, 2))).sum()
    loss.backward()
    model.bias.grad = None
    with pytest.raises(FloatingPointError, match="missing/non-finite"):
        smoke.require_finite_training_state(loss, model, seed=1, update=1)


def test_gradient_flow_reaches_head_and_trunk(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    model = EquivariantPairCoordinateUNet(**config["smoke_model"])
    train, _, _ = smoke.build_polymer_panels(config)
    diffusion = CoordinateVPDiffusion(4)
    batch, diffused = smoke._paired_diffusion_batch(
        train[0], diffusion, scale=10.0, draw_seed=11, rotate=True, device=torch.device("cpu")
    )
    output = model(
        diffused.noisy_coordinates,
        diffused.timesteps,
        batch["lengths"],
        batch["residue_mask"],
        batch["continuity"],
    )
    loss = (output["v_prediction"] - diffused.coordinate_v_target).square().mean()
    loss.backward()
    gradients = smoke.require_finite_training_state(loss, model, seed=1, update=1)
    assert any("coefficient_head" in name and torch.count_nonzero(value) for name, value in gradients.items())
    assert any("pair_trunk" in name and "coefficient_head" not in name for name in gradients)


def test_paired_evaluation_and_euclidean_sampling_are_deterministic(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    model = EquivariantPairCoordinateUNet(**config["smoke_model"]).eval()
    _, heldout, _ = smoke.build_polymer_panels(config)
    diffusion = CoordinateVPDiffusion(4)
    left = smoke._evaluation_metrics(
        model, heldout, diffusion, scale=10.0, corruption_seed=31, device=torch.device("cpu")
    )
    right = smoke._evaluation_metrics(
        model, heldout, diffusion, scale=10.0, corruption_seed=31, device=torch.device("cpu")
    )
    assert left == right
    first = smoke._sampling_panel(model, diffusion, config, seed_offset=0, device=torch.device("cpu"))
    second = smoke._sampling_panel(model, diffusion, config, seed_offset=0, device=torch.device("cpu"))
    assert first == second
    assert first[0]["symmetry_error"] == 0
    assert first[0]["diagonal_error"] == 0
    assert first[0]["rank3_residual_fraction"] < 5e-5
    assert smoke._sampling_geometry_valid(first, 5e-5)


def test_oracle_sampler_and_timestep_stratification(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config["timestep_evaluation_bins"] = [
        {"name": "very_low_noise", "fraction": 0.0},
        {"name": "very_high_noise", "fraction": 1.0},
    ]
    diffusion = CoordinateVPDiffusion(8)
    oracle = smoke.verify_oracle_sampler_contract(diffusion, length=9, tolerance=2e-5)
    assert oracle["passed"]
    assert oracle["trajectory_state_count"] == 9
    model = EquivariantPairCoordinateUNet(**config["smoke_model"]).eval()
    _, heldout, _ = smoke.build_polymer_panels(config)
    metrics = smoke._timestep_stratified_metrics(
        model,
        heldout,
        diffusion,
        config,
        scale=10.0,
        corruption_seed=9,
        device=torch.device("cpu"),
    )
    assert set(metrics) == {"very_low_noise", "very_high_noise"}
    assert metrics["very_low_noise"]["timestep"] == 0
    assert metrics["very_high_noise"]["timestep"] == 7
    for record in metrics.values():
        assert set(record) >= {
            "coordinate_v_mse",
            "pair_distance_rmse_angstrom",
            "adjacent_distance_rmse_angstrom",
            "radius_of_gyration_error_angstrom",
        }


def test_v2_joint_quality_and_sampler_contract_classification() -> None:
    thresholds = {
        "minimum_train_v_mse_relative_improvement": 0.2,
        "minimum_heldout_v_mse_relative_improvement_per_seed": 0.1,
        "minimum_heldout_pair_rmse_relative_improvement_per_seed": 0.1,
        "maximum_joint_polymer_quality_error": 0.3,
    }
    row = {
        "finite_losses_and_gradients": True,
        "successful_updates": 1,
        "relative_improvements": {
            "train_coordinate_v_mse": 0.5,
            "heldout_coordinate_v_mse": 0.3,
            "heldout_pair_distance_rmse": 0.3,
        },
        "coefficient_head_gradient_observed": True,
        "unet_trunk_gradient_observed": True,
        "parameter_l2_change": 1.0,
        "trained_contract_checks": {"contract": True},
        "sampling_geometry_valid": True,
        "joint_polymer_quality": {"passed": True},
    }
    assert smoke.classify_smoke([row], thresholds) == "synthetic_learning_verified"
    assert smoke.classify_smoke([row], thresholds, sampler_contract_passed=False) == "sampler_contract_failed"


@pytest.mark.parametrize("classification", smoke.CLASSIFICATIONS)
def test_decision_classifications(classification: str) -> None:
    thresholds = {
        "minimum_train_v_mse_relative_improvement": 0.2,
        "minimum_heldout_v_mse_relative_improvement_per_seed": 0.1,
        "minimum_heldout_pair_rmse_relative_improvement_per_seed": 0.1,
        "minimum_sampling_adjacent_distribution_improvement": 0.1,
        "minimum_sampling_radius_distribution_improvement": 0.1,
    }
    base = {
        "finite_losses_and_gradients": True,
        "successful_updates": 1,
        "relative_improvements": {
            "train_coordinate_v_mse": 0.5,
            "heldout_coordinate_v_mse": 0.3,
            "heldout_pair_distance_rmse": 0.3,
        },
        "coefficient_head_gradient_observed": True,
        "unet_trunk_gradient_observed": True,
        "parameter_l2_change": 1.0,
        "trained_contract_checks": {"contract": True},
        "sampling_geometry_valid": True,
        "sampling_improvements": {
            "adjacent_vs_untrained": 0.2,
            "adjacent_vs_gaussian": 0.2,
            "radius_vs_untrained": 0.2,
            "radius_vs_gaussian": 0.2,
        },
    }
    row = json.loads(json.dumps(base))
    sampler_contract_passed = True
    if classification == "numerically_unstable":
        row["finite_losses_and_gradients"] = False
    elif classification == "memorization_without_heldout_learning":
        row["relative_improvements"]["heldout_coordinate_v_mse"] = 0.0
    elif classification == "denoising_learned_but_sampling_not_learned":
        row["sampling_improvements"]["adjacent_vs_gaussian"] = 0.0
    elif classification == "equivariant_lifting_underexpressive":
        row["coefficient_head_gradient_observed"] = False
    elif classification == "sampler_contract_failed":
        sampler_contract_passed = False
    elif classification == "inconclusive_requires_review":
        row["relative_improvements"]["train_coordinate_v_mse"] = 0.0
    assert smoke.classify_smoke([row], thresholds, sampler_contract_passed=sampler_contract_passed) == classification


def test_plan_is_non_authorizing_bounded_and_refuses_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config(tmp_path)
    path = _write_config(tmp_path, config)
    tiny_count = sum(
        parameter.numel() for parameter in EquivariantPairCoordinateUNet(**config["smoke_model"]).parameters()
    )
    monkeypatch.setattr(smoke, "_model_counts", lambda _: (tiny_count, 7_586_505, config["smoke_model"]))
    plan = smoke.plan_coordinate_smoke(path)
    assert plan["optimizer_created"] is False
    assert plan["forward_executed"] is False
    assert plan["optimizer_updates_per_seed"] == 1
    assert plan["authorizes_training"] is False
    assert not Path(config["output_dir"]).exists()
    Path(config["output_dir"]).mkdir()
    with pytest.raises(FileExistsError):
        smoke.plan_coordinate_smoke(path)


def test_atomic_tiny_publication_is_non_authorizing_and_preserves_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _base_config(tmp_path)
    path = _write_config(tmp_path, config)
    source_paths = [Path(value) for key, value in config["phase3a"].items() if key.endswith("_path")]
    before = {str(source): hashlib.sha256(source.read_bytes()).hexdigest() for source in source_paths}
    tiny_count = sum(
        parameter.numel() for parameter in EquivariantPairCoordinateUNet(**config["smoke_model"]).parameters()
    )
    monkeypatch.setattr(smoke, "_model_counts", lambda _: (tiny_count, 7_586_505, config["smoke_model"]))
    monkeypatch.setattr(
        smoke,
        "_production_compatibility_check",
        lambda *args, **kwargs: {
            "finite": True,
            "forward_completed": True,
            "backward_completed": True,
            "optimizer_created": False,
            "optimizer_updates": 0,
        },
    )
    report = smoke.run_coordinate_smoke(path)
    output = Path(config["output_dir"])
    protocol = json.loads((output / "protocol.json").read_text())
    checkpoint = torch.load(output / "synthetic_seed_3.pt", map_location="cpu", weights_only=False)
    assert report["successful_optimizer_updates"] == 3
    assert report["attempted_optimizer_updates"] == 3
    assert report["optimizer_created"] is True
    assert report["forward_executed"] is True
    assert report["backward_executed"] is True
    assert protocol["authorizes_training"] is False
    assert protocol["real_data_scanned"] is False
    assert checkpoint["synthetic_only"] is True
    assert checkpoint["authorizes_sequence_conditioning"] is False
    assert (output / "report.json").is_file()
    assert (output / "heartbeat.json").is_file()
    assert not output.with_name(f".{output.name}.inprogress").exists()
    after = {str(source): hashlib.sha256(source.read_bytes()).hexdigest() for source in source_paths}
    assert after == before


def test_v2_plan_pins_v1_panels_and_preserves_completed_evidence(tmp_path: Path) -> None:
    output = Path("reports/experiments/E007_matrix_sequence_cogeneration/coordinate_smoke_v2")
    before = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    config = yaml.safe_load(Path("configs/e007_coordinate_smoke_v2.yaml").read_text())
    planned_output = tmp_path / "coordinate_smoke_v2_plan_target"
    config["output_dir"] = str(planned_output)
    config_path = tmp_path / "coordinate_smoke_v2.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    plan = smoke.plan_coordinate_smoke(config_path)
    assert plan["version"] == smoke.SMOKE_VERSION_V2
    assert plan["optimizer_updates_per_seed"] == 1000
    assert plan["phase3b_v1_evidence"]["panel_hashes"]["sample_id_sha256"]["train"] == (
        "7dea3ad3c54aeddf0d4d1b79b1d26e72a7dc25eb8e93668628f720237a04b143"
    )
    assert plan["oracle_sampler_contract_check_planned"] is True
    assert plan["optimizer_created"] is False
    assert not planned_output.exists()
    after = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    assert after == before
