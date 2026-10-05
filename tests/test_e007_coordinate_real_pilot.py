from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

import protein_distance_diffusion.training.e007_coordinate_real_pilot as pilot
from protein_distance_diffusion.models.coordinate_equivariance import coordinate_model_execution_context
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet


def _config() -> dict:
    return yaml.safe_load(Path("configs/e007_coordinate_real_pilot_v1.yaml").read_text())


def _row(sample_id: str, length: int, *, accepted: bool = True) -> dict:
    coordinates = torch.arange(length * 3, dtype=torch.float32).reshape(length, 3) / 10
    return {
        "sample_id": sample_id,
        "sequence_length": length,
        "accepted_contiguous_single_chain": accepted,
        "coordinates": coordinates,
        "residue_mask": torch.ones(length, dtype=torch.bool),
        "chain_continuity_mask": torch.ones(length - 1, dtype=torch.bool),
    }


def _evaluation(value: float, *, high: float | None = None, very_high: float | None = None) -> dict:
    high = value if high is None else high
    very_high = value if very_high is None else very_high
    lengths = {item["name"]: {"coordinate_v_mse": value} for item in _config()["length_strata"]}
    return {
        "global": {"coordinate_v_mse": value},
        "by_length_stratum": lengths,
        "by_timestep_bin": {
            "very_low_noise": {"coordinate_v_mse": value},
            "low_noise": {"coordinate_v_mse": value},
            "intermediate_noise": {"coordinate_v_mse": value},
            "high_noise": {"coordinate_v_mse": high},
            "very_high_noise": {"coordinate_v_mse": very_high},
        },
    }


def _samples(adjacent: float, clash: float = 0.01) -> dict:
    return {
        "records": [
            {"adjacent_distance_mean_angstrom": adjacent, "non_neighbor_clash_fraction": clash} for _ in range(10)
        ]
    }


def test_exact_model_objective_and_production_schedule_contract() -> None:
    config = pilot._load_config("configs/e007_coordinate_real_pilot_v1.yaml")
    generator = yaml.safe_load(Path(config["production_generator"]["config_path"]).read_text())
    assert config["model"] == generator["model"]
    assert config["diffusion_steps"] == generator["diffusion_steps"] == 500
    assert config["objective"] == {
        "name": "uniform_valid_coordinate_v_mse",
        "timestep_sampling": "uniform_discrete",
        "timestep_weighting": "none",
        "auxiliary_losses": [],
    }
    model = EquivariantPairCoordinateUNet(**config["model"])
    assert sum(parameter.numel() for parameter in model.parameters()) == 7_586_505


def test_batching_schedule_and_accumulation_accounting() -> None:
    config = _config()
    accounting = pilot.planned_batch_accounting(config)
    assert accounting["optimizer_updates_by_stratum"] == {
        "20-64": 200,
        "65-128": 200,
        "129-256": 200,
        "257-384": 200,
        "385-500": 200,
    }
    assert accounting["planned_samples_by_stratum"] == {
        "20-64": 800,
        "65-128": 400,
        "129-256": 200,
        "257-384": 200,
        "385-500": 200,
    }
    assert all(item["accumulation_steps"] == 1 for item in config["batch_regimes"])


def test_identity30_membership_is_enforced_and_homologs_are_excluded() -> None:
    config = _config()
    rows = [_row("clean-a", 40), _row("homolog", 40), _row("clean-b", 80)]
    result = pilot._bounded_ranked_rows(
        rows,
        strata=config["length_strata"],
        capacities={"20-64": 1, "65-128": 1},
        seed=7,
        purpose="validation",
        permitted_ids={"clean-a", "clean-b"},
    )
    selected = {row["sample_id"] for values in result.values() for row in values}
    assert selected == {"clean-a", "clean-b"}
    assert "homolog" not in selected


def test_deterministic_selection_is_order_independent() -> None:
    config = _config()
    rows = [_row(f"sample-{index}", 40) for index in range(20)]
    kwargs = dict(strata=config["length_strata"], capacities={"20-64": 5}, seed=99, purpose="training")
    first = pilot._bounded_ranked_rows(rows, **kwargs)
    second = pilot._bounded_ranked_rows(list(reversed(rows)), **kwargs)
    assert [row["sample_id"] for row in first["20-64"]] == [row["sample_id"] for row in second["20-64"]]


def test_accepted_population_count_is_enforced() -> None:
    config = _config()
    with pytest.raises(ValueError, match="accepted population contradiction"):
        pilot._bounded_ranked_rows(
            [_row("one", 40), _row("rejected", 40, accepted=False)],
            strata=config["length_strata"],
            capacities={"20-64": 1},
            seed=4,
            purpose="population",
            expected_eligible_count=2,
        )


def test_uniform_loss_uses_only_valid_coordinates_and_padding_gradient_is_zero() -> None:
    prediction = torch.tensor([[[1.0, 2.0, 3.0], [100.0, 100.0, 100.0]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False]])
    loss = pilot.uniform_coordinate_v_mse(prediction, target, mask)
    assert float(loss.detach()) == pytest.approx(torch.tensor([1.0, 4.0, 9.0]).mean().item())
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:, 1]) == 0


def test_uniform_training_corruption_is_deterministic_and_not_batch_index_schedule() -> None:
    prepared = {
        "coordinates": torch.zeros((1, 8, 3)),
        "residue_mask": torch.ones((1, 8), dtype=torch.bool),
    }
    diffusion = pilot.CoordinateVPDiffusion(500)
    first = pilot.make_uniform_training_corruption(
        prepared,
        diffusion,
        seed=12345,
        device=torch.device("cpu"),
    )
    second = pilot.make_uniform_training_corruption(
        prepared,
        diffusion,
        seed=12345,
        device=torch.device("cpu"),
    )
    assert torch.equal(first["timesteps"], second["timesteps"])
    assert torch.equal(first["batch"].noisy_coordinates, second["batch"].noisy_coordinates)
    assert 0 <= int(first["timesteps"][0]) < 500
    assert int(first["timesteps"][0]) != 0


def test_checkpoint_resume_restores_exact_state(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 - step / 10)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    payload = pilot.checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        update=1,
        samples_processed=2,
        valid_residues_processed=12,
        config_sha256="config",
        protected_hashes_sha256="protected",
    )
    path = tmp_path / "checkpoint.pt"
    pilot._atomic_torch(path, payload)
    restored = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: 1.0 - step / 10)
    state = pilot._load_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        config_sha256="config",
        protected_hashes_sha256="protected",
    )
    assert state["optimizer_update"] == state["sampler_cursor"] == 1
    assert all(torch.equal(left, right) for left, right in zip(model.parameters(), restored.parameters(), strict=True))
    original_state = optimizer.state_dict()
    restored_state = restored_optimizer.state_dict()
    assert original_state["param_groups"] == restored_state["param_groups"]
    for parameter_id, values in original_state["state"].items():
        for key, value in values.items():
            restored_value = restored_state["state"][parameter_id][key]
            assert torch.equal(value, restored_value) if isinstance(value, torch.Tensor) else value == restored_value
    assert scheduler.state_dict() == restored_scheduler.state_dict()


def test_resume_rejects_checkpoint_identity_contradiction(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "checkpoint.pt"
    pilot._atomic_torch(
        path,
        pilot.checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            update=0,
            samples_processed=0,
            valid_residues_processed=0,
            config_sha256="a",
            protected_hashes_sha256="b",
        ),
    )
    with pytest.raises(ValueError, match="resume identity contradiction"):
        pilot._load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config_sha256="changed",
            protected_hashes_sha256="b",
        )


def test_classification_requires_denoising_and_sampling() -> None:
    config = _config()
    evaluations = {
        0: {"validation": _evaluation(1.0)},
        1000: {"validation": _evaluation(0.8, high=0.8, very_high=0.8)},
    }
    assert pilot.classify_pilot(evaluations, {0: _samples(6.0), 1000: _samples(3.8)}, config) == (
        "real_data_learning_and_sampling_verified"
    )
    assert pilot.classify_pilot(evaluations, {0: _samples(3.8), 1000: _samples(5.0)}, config) == (
        "denoising_learned_but_sampling_not_learned"
    )


def test_length_regression_prevents_success() -> None:
    config = _config()
    initial = _evaluation(1.0)
    final = _evaluation(0.8, high=0.8, very_high=0.8)
    final["by_length_stratum"]["385-500"]["coordinate_v_mse"] = 1.2
    result = pilot.classify_pilot(
        {0: {"validation": initial}, 1000: {"validation": final}},
        {0: _samples(6.0), 1000: _samples(3.8)},
        config,
    )
    assert result == "length_limited_learning"


def test_numerical_sampling_failure_has_distinct_classification() -> None:
    config = _config()
    final_sampling = _samples(3.8)
    final_sampling["records"][0]["strict_euclidean_valid"] = False
    result = pilot.classify_pilot(
        {
            0: {"validation": _evaluation(1.0)},
            1000: {"validation": _evaluation(0.8, high=0.8, very_high=0.8)},
        },
        {0: _samples(6.0), 1000: final_sampling},
        config,
    )
    assert result == "numerical_or_memory_failure"


def test_strict_backend_restores_state_after_exception() -> None:
    config = _config()
    device = torch.device("cpu")
    before = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError, match="intentional"):
        with coordinate_model_execution_context(config["numerics"], device) as telemetry:
            assert telemetry["during"]["deterministic_algorithms"] is True
            raise RuntimeError("intentional")
    assert torch.are_deterministic_algorithms_enabled() is before


def test_plan_is_no_scan_no_model_and_refuses_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    config["output_dir"] = str(tmp_path / "future")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(pilot, "verify_pilot_prerequisites", lambda _: {"hashes": {"pinned": "hash"}})
    monkeypatch.setattr(pilot, "EquivariantPairCoordinateUNet", lambda **_: pytest.fail("model constructed"))
    monkeypatch.setattr(pilot, "E007CoordinateDataset", lambda *_args, **_kwargs: pytest.fail("shard scanned"))
    result = pilot.plan_real_coordinate_pilot(path)
    assert result["coordinate_payloads_scanned"] is False
    assert result["model_created"] is result["optimizer_created"] is False
    assert not Path(config["output_dir"]).exists()
    Path(config["output_dir"]).mkdir()
    with pytest.raises(FileExistsError):
        pilot.plan_real_coordinate_pilot(path)


def test_atomic_json_publication_leaves_no_temporary_file(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    pilot._atomic_json(destination, {"status": "completed", "authorizes_training": False})
    assert yaml.safe_load(destination.read_text())["status"] == "completed"
    assert not list(tmp_path.glob(".*.tmp"))


def test_prerequisite_hash_failure_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    monkeypatch.setattr(pilot, "sha256_file", lambda _path: "wrong")
    with pytest.raises(ValueError, match="prerequisite hash contradiction"):
        pilot.verify_pilot_prerequisites(copy.deepcopy(config))


def test_fixed_sampling_noise_identity_is_checkpoint_independent() -> None:
    config = _config()
    identities = [
        (int(stratum["maximum"]), int(config["sampling_seed"]) + index * 100 + sample)
        for index, stratum in enumerate(config["length_strata"])
        for sample in range(int(config["sampling_samples_per_stratum"]))
    ]
    assert pilot._canonical_sha(identities) == pilot._canonical_sha(list(identities))
    assert len(set(identities)) == 10
