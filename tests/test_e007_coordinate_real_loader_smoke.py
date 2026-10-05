from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from protein_distance_diffusion.data.rich_geometry import RichDatasetAuthorization
from protein_distance_diffusion.models.coordinate_equivariance import coordinate_backend_state
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import (
    EquivariantPairCoordinateUNet,
)
from protein_distance_diffusion.training import e007_coordinate_real_loader_smoke as smoke
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion

CONFIG_PATH = Path("configs/e007_coordinate_real_loader_smoke_v1.yaml")


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _row(sample_id: str, split: str, length: int, seed: int, *, accepted: bool = True) -> dict:
    generator = np.random.default_rng(seed)
    steps = generator.normal(size=(length, 3))
    steps /= np.linalg.norm(steps, axis=1, keepdims=True)
    coordinates = np.cumsum(steps * 3.8, axis=0).astype(np.float32)
    mask = torch.ones(length, dtype=torch.bool)
    continuity = torch.ones(max(length - 1, 0), dtype=torch.bool)
    if not accepted:
        continuity[length // 2] = False
    return {
        "sample_id": sample_id,
        "split": split,
        "sequence_length": length,
        "coordinates": torch.from_numpy(coordinates),
        "residue_mask": mask,
        "chain_continuity_mask": continuity,
        "accepted_contiguous_single_chain": accepted,
        "source_sha256": hashlib_for(f"source:{sample_id}"),
        "npz_sha256": hashlib_for(f"npz:{sample_id}"),
        "sidecar_schema_version": "e006_rich_geometry_sidecar_v2",
    }


def hashlib_for(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _panel_rows(split: str, count: int = 4) -> list[dict]:
    lengths = {
        "20-64": [57, 61, 63, 64],
        "65-128": [113, 119, 127, 128],
        "129-256": [241, 247, 255, 256],
        "257-384": [369, 375, 383, 384],
        "385-500": [487, 493, 499, 500],
    }
    rows = []
    offset = 10_000 if split == "validation" else 0
    for stratum_index, values in enumerate(lengths.values()):
        for replicate, length in enumerate(values[:count]):
            rows.append(
                _row(
                    f"{split}:{stratum_index}:{replicate}:N{length}",
                    split,
                    length,
                    offset + stratum_index * 100 + replicate,
                )
            )
    return rows


def _authorization(root: Path) -> RichDatasetAuthorization:
    return RichDatasetAuthorization(
        root=root,
        protocol_sha256="1" * 64,
        schema_sha256="2" * 64,
        vocabulary_sha256="3" * 64,
        normalization_sha256="4" * 64,
        shard_inventory_sha256="5" * 64,
        split_counts={"train": 20, "validation": 20},
        observed_shard_hashes={"train/a": "6" * 64, "validation/a": "7" * 64},
    )


def _selected(rows: dict[str, list[dict]], config: dict) -> dict[str, list[dict]]:
    selected, _ = smoke._scan_panels_and_leakage(rows, config)
    return selected


def test_pinned_normalization_hash_scale_and_production_contract() -> None:
    config = _config()
    evidence = smoke.verify_smoke_prerequisites(config)
    assert evidence["hashes"]["normalization"] == ("1be9ef05c82110861482eb46e61ef433045fa9678b9cdfffd921f0defebb03cd")
    assert evidence["normalization"]["coordinate_scale_angstrom"] == smoke.NORMALIZATION_SCALE_ANGSTROM
    model = EquivariantPairCoordinateUNet(**config["model"])
    assert sum(parameter.numel() for parameter in model.parameters()) == 7_586_505
    assert evidence["available_group_identifier_columns"] == []


def test_deterministic_stratified_selection_constraints() -> None:
    config = _config()
    datasets = {"train": _panel_rows("train"), "validation": _panel_rows("validation")}
    first, evidence = smoke._scan_panels_and_leakage(datasets, config)
    second, second_evidence = smoke._scan_panels_and_leakage(datasets, config)
    for split in ("train", "validation"):
        assert len(first[split]) == 20
        assert [row["sample_id"] for row in first[split]] == [row["sample_id"] for row in second[split]]
        by_stratum = {}
        for record in first[split]:
            by_stratum.setdefault(record["length_stratum"], []).append(record)
        assert set(map(len, by_stratum.values())) == {4}
        assert all(any(row["sequence_length"] % 8 for row in records) for records in by_stratum.values())
        assert max(row["sequence_length"] for row in by_stratum["385-500"]) == 500
    assert evidence["selected_manifest_sha256"] == second_evidence["selected_manifest_sha256"]


def test_within_and_cross_split_duplicate_detection() -> None:
    config = _config()
    train = _panel_rows("train")
    validation = _panel_rows("validation")
    train.append(copy.deepcopy(train[0]))
    duplicate = copy.deepcopy(train[1])
    duplicate["sample_id"] = "validation:coordinate-duplicate"
    duplicate["split"] = "validation"
    validation.append(duplicate)
    _, evidence = smoke._scan_panels_and_leakage({"train": train, "validation": validation}, config)
    leakage = evidence["leakage"]
    assert leakage["within_split"]["train"]["duplicate_sample_id_rows"] == 1
    assert leakage["within_split"]["train"]["duplicate_coordinate_hash_rows"] == 1
    assert leakage["cross_split"]["coordinate_hashes"]["overlapping_unique_key_count"] == 1
    assert leakage["cross_split"]["rigid_shape_hashes"]["overlapping_unique_key_count"] == 1
    assert leakage["classification"] == "dataset_review_required"


def test_mixed_length_batch_masks_normalization_padding_and_roundtrip() -> None:
    rows = [_row("a", "train", 57, 1), _row("b", "train", 128, 2)]
    batch = smoke.prepare_coordinate_batch(rows, smoke.NORMALIZATION_SCALE_ANGSTROM, 8)
    assert batch["coordinates"].shape == (2, 128, 3)
    assert batch["lengths"].tolist() == [57, 128]
    assert batch["square_padded_length"] == 128
    assert torch.count_nonzero(batch["coordinates"][0, 57:]) == 0
    assert torch.equal(batch["pair_mask"], batch["residue_mask"][:, :, None] & batch["residue_mask"][:, None, :])
    physical = batch["coordinates"] * smoke.NORMALIZATION_SCALE_ANGSTROM
    assert torch.allclose(physical, batch["physical_coordinates_angstrom"], atol=4e-6, rtol=0)
    assert batch["sequence_inputs"] is False
    assert batch["clean_coordinate_features"] is False


def test_large_absolute_offset_reproduces_old_rounding_failure_and_is_corrected() -> None:
    generator = torch.Generator().manual_seed(2)
    coordinates = torch.randn((55, 3), generator=generator) * 15
    coordinates += torch.tensor([333.5, -200.2, 100.7])
    row = _row("rounding-fixture", "train", 55, 1)
    row["coordinates"] = coordinates
    old = (coordinates - coordinates.mean(dim=0)).mean(dim=0) / smoke.NORMALIZATION_SCALE_ANGSTROM
    assert float(old.abs().max()) > smoke.CENTERING_ATOL_NORMALIZED
    batch = smoke.prepare_coordinate_batch([row], smoke.NORMALIZATION_SCALE_ANGSTROM, 8)
    assert float(batch["centering_maximum_absolute_component"].max()) < smoke.CENTERING_ATOL_NORMALIZED
    assert batch["square_padded_length"] == 56
    padded = torch.nn.functional.pad(batch["coordinates"], (0, 0, 0, 1))
    assert torch.count_nonzero(padded[:, 55:]) == 0
    diagnostics = smoke.coordinate_preparation_diagnostics(row, smoke.NORMALIZATION_SCALE_ANGSTROM, 8)
    assert diagnostics["maximum_padded_coordinate_magnitude"] == 0.0
    assert diagnostics["criterion"]["absolute_tolerance"] == 2e-6
    assert diagnostics["criterion"]["includes_padding"] is False
    assert diagnostics["criterion"]["denominator"] == "valid_residue_count"


def test_phase3e_b_and_phase3f_share_canonical_preparation() -> None:
    from protein_distance_diffusion.training import e007_coordinate_real_pilot as phase3f

    assert phase3f.prepare_coordinate_batch is smoke.prepare_coordinate_batch


def test_genuinely_uncentered_collation_still_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _row("uncentered", "train", 31, 2)
    original = smoke.collate_e007_coordinates

    def uncentered(rows, **kwargs):
        result = original(rows, **kwargs)
        result["coordinates"][0, :31] += 1.0
        return result

    monkeypatch.setattr(smoke, "collate_e007_coordinates", uncentered)
    with pytest.raises(ValueError, match="criterion=max_abs<=2e-06"):
        smoke.prepare_coordinate_batch([row], smoke.NORMALIZATION_SCALE_ANGSTROM, 8)


def test_prepared_batch_has_finite_gradients() -> None:
    row = _row("gradient", "train", 55, 3)
    batch = smoke.prepare_coordinate_batch([row], smoke.NORMALIZATION_SCALE_ANGSTROM, 8)
    coordinates = batch["coordinates"].detach().requires_grad_(True)
    loss = coordinates[batch["residue_mask"]].square().mean()
    loss.backward()
    assert torch.isfinite(coordinates.grad).all()


def test_coordinate_v_corruption_is_deterministic_and_exact() -> None:
    batch = smoke.prepare_coordinate_batch([_row("a", "train", 31, 1)], smoke.NORMALIZATION_SCALE_ANGSTROM, 8)
    diffusion = CoordinateVPDiffusion(32)
    first = smoke.make_deterministic_corruption(batch, diffusion, seed=22, device=torch.device("cpu"))
    second = smoke.make_deterministic_corruption(batch, diffusion, seed=22, device=torch.device("cpu"))
    assert torch.equal(first["batch"].noisy_coordinates, second["batch"].noisy_coordinates)
    assert torch.equal(first["batch"].coordinate_v_target, second["batch"].coordinate_v_target)
    assert torch.count_nonzero(first["batch"].noisy_coordinates[~first["mask"]]) == 0


def test_forward_backward_without_optimizer_preserves_parameters_and_gradients() -> None:
    model_config = {
        "rbf_bins": 4,
        "base_channels": 4,
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
        "triangle_hidden_channels": 4,
        "triangle_dropout": 0.0,
        "triangle_chunk_size": 4,
        "time_embedding_dim": 16,
        "length_embedding_dim": 16,
        "max_length": 500,
    }
    model = EquivariantPairCoordinateUNet(**model_config)
    before = smoke._parameter_sha256(model)
    batch = smoke.prepare_coordinate_batch([_row("a", "train", 31, 1)], smoke.NORMALIZATION_SCALE_ANGSTROM, 1)
    corruption = smoke.make_deterministic_corruption(
        batch, CoordinateVPDiffusion(8), seed=44, device=torch.device("cpu")
    )
    output = model(
        corruption["batch"].noisy_coordinates,
        corruption["timesteps"],
        batch["lengths"],
        batch["residue_mask"],
        batch["chain_continuity_mask"],
    )
    loss = (output["v_prediction"] - corruption["batch"].coordinate_v_target).square().mean()
    loss.backward()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None}
    norms = smoke._gradient_norms(gradients)
    assert bool(torch.isfinite(loss))
    assert all(value > 0 for value in norms.values())
    assert smoke._parameter_sha256(model) == before
    model.zero_grad(set_to_none=True)


def test_near_500_tiny_model_execution_is_bounded_and_masked() -> None:
    model = EquivariantPairCoordinateUNet(
        rbf_bins=2,
        base_channels=2,
        channel_multipliers=[1],
        residual_blocks_per_level=1,
        group_norm_groups=1,
        attention_heads=1,
        use_bottleneck_attention=False,
        use_pre_bottleneck_axial_attention=False,
        use_pre_bottleneck_triangle_multiplication=False,
        time_embedding_dim=8,
        length_embedding_dim=8,
        max_length=500,
    )
    batch = smoke.prepare_coordinate_batch([_row("long", "train", 497, 8)], smoke.NORMALIZATION_SCALE_ANGSTROM, 1)
    corruption = smoke.make_deterministic_corruption(
        batch, CoordinateVPDiffusion(4), seed=8, device=torch.device("cpu")
    )
    output = model(
        corruption["batch"].noisy_coordinates,
        corruption["timesteps"],
        batch["lengths"],
        batch["residue_mask"],
        batch["chain_continuity_mask"],
    )["v_prediction"]
    output.square().mean().backward()
    assert output.shape == (1, 497, 3)
    assert torch.isfinite(output).all()


def test_distance_matrix_and_equivariance_checks() -> None:
    row = _row("a", "train", 31, 4)
    batch = smoke.prepare_coordinate_batch([row], smoke.NORMALIZATION_SCALE_ANGSTROM, 1)
    distance = smoke._distance_checks(batch["coordinates"][0], batch["residue_mask"][0], 5e-5)
    assert distance["passed"]
    quality = distance["quality"]
    assert quality["diagonal_error_max_angstrom"] == 0.0
    assert quality["symmetry_error_max_angstrom"] == 0.0
    assert quality["negative_distance_count"] == 0


def test_plan_only_is_metadata_only_and_refuses_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    config["loader_output_dir"] = str(tmp_path / "loader")
    config["forward_backward_output_dir"] = str(tmp_path / "forward")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(
        smoke,
        "verify_smoke_prerequisites",
        lambda unused: {"hashes": {}, "available_group_identifier_columns": []},
    )
    monkeypatch.setattr(smoke, "E007CoordinateDataset", lambda *args, **kwargs: pytest.fail("coordinate scan"))
    plan = smoke.plan_real_loader_smoke(path)
    assert plan["coordinate_payloads_scanned"] is False
    assert plan["model_created"] is False
    assert not Path(config["loader_output_dir"]).exists()
    Path(config["loader_output_dir"]).mkdir()
    with pytest.raises(FileExistsError):
        smoke.plan_real_loader_smoke(path)


def test_smoke_configuration_refuses_non_strict_backend_policy(tmp_path: Path) -> None:
    config = _config()
    config["numerics"]["allow_cudnn_tf32"] = True
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="policy mismatch"):
        smoke._load_config(path)


def test_leakage_refuses_forward_before_model_and_publishes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    output = tmp_path / "forward"
    config["loader_output_dir"] = str(tmp_path / "loader")
    config["forward_backward_output_dir"] = str(output)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    train = _panel_rows("train")
    validation = _panel_rows("validation")
    validation[0]["coordinates"] = train[0]["coordinates"].clone()
    authorization = _authorization(tmp_path)
    monkeypatch.setattr(
        smoke,
        "verify_smoke_prerequisites",
        lambda unused: {"hashes": {"normalization": "a" * 64}, "available_group_identifier_columns": []},
    )
    monkeypatch.setattr(smoke, "_authorize_dataset", lambda unused: authorization)
    monkeypatch.setattr(
        smoke,
        "E007CoordinateDataset",
        lambda unused, split: train if split == "train" else validation,
    )
    monkeypatch.setattr(
        smoke,
        "_forward_backward_checks",
        lambda *args, **kwargs: pytest.fail("model was constructed despite leakage"),
    )
    report = smoke.run_real_loader_smoke(path, mode="forward-backward-smoke")
    assert report["classification"] == "dataset_review_required"
    assert report["model_created"] is False
    assert report["forward_executed"] is False
    assert report["optimizer_updates"] == 0
    assert {key: report[key] for key in smoke.NON_AUTHORIZING} == smoke.NON_AUTHORIZING
    assert output.is_dir()
    assert not output.with_name(f".{output.name}.inprogress").exists()
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["authorizes_real_data_training"] is False
    assert not output.with_name(f".{output.name}.inprogress").exists()
    with pytest.raises(FileExistsError, match="output already exists"):
        smoke.run_real_loader_smoke(path, mode="forward-backward-smoke")


def test_whole_forward_backward_path_uses_and_restores_strict_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forward_states = []
    backward_states = []

    class ObservedModel(EquivariantPairCoordinateUNet):
        def forward(self, *args, **kwargs):
            forward_states.append(coordinate_backend_state(torch.device("cpu")))
            result = super().forward(*args, **kwargs)
            if result["v_prediction"].requires_grad:
                result["v_prediction"].register_hook(
                    lambda gradient: backward_states.append(coordinate_backend_state(torch.device("cpu"))) or gradient
                )
            return result

    model_config = {
        "rbf_bins": 4,
        "base_channels": 4,
        "channel_multipliers": [1],
        "residual_blocks_per_level": 1,
        "dropout": 0.0,
        "group_norm_groups": 1,
        "attention_heads": 1,
        "use_bottleneck_attention": False,
        "use_pre_bottleneck_axial_attention": False,
        "use_pre_bottleneck_triangle_multiplication": False,
        "time_embedding_dim": 16,
        "length_embedding_dim": 16,
        "max_length": 500,
    }
    expected_parameters = sum(parameter.numel() for parameter in ObservedModel(**model_config).parameters())
    config = {
        **_config(),
        "seed": 9,
        "diffusion_steps": 8,
        "model": model_config,
        "expected_parameter_count": expected_parameters,
        "expected_downsample_factor": 1,
        "length_strata": [{"name": "20-64", "minimum": 20, "maximum": 64}],
        "batch_regimes": [{"name": "short", "maximum_length": 64, "physical_batch_size": 1}],
    }
    selected = {
        split: [{"row": _row(f"{split}:31", split, 31, seed), "length_stratum": "20-64"}]
        for seed, split in enumerate(("train", "validation"), start=1)
    }
    monkeypatch.setattr(smoke, "EquivariantPairCoordinateUNet", ObservedModel)
    before = coordinate_backend_state(torch.device("cpu"))
    result = smoke._forward_backward_checks(selected, config, torch.device("cpu"))
    assert forward_states
    assert backward_states
    for state in [*forward_states, *backward_states]:
        assert state["allow_matmul_tf32"] is False
        assert state["allow_cudnn_tf32"] is False
        assert state["deterministic_algorithms"] is True
        assert state["autocast_enabled"] is False
    assert result["parameter_sha256_before"] == result["parameter_sha256_after"]
    assert result["numerical_backend"]["before"] == before
    assert result["numerical_backend"]["after"] == before
    assert result["numerical_backend"]["restored"] is True
