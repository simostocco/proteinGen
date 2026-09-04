"""Tests for analysis-only E003 gradient-interaction diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from protein_distance_diffusion.data.preprocess import ProteinSample, save_processed_sample
from protein_distance_diffusion.data.statistics import write_normalization
from protein_distance_diffusion.models.unet import DistanceUNet


def _load_profiler_module():
    path = Path("scripts/profile_edm_auxiliary_gradients.py")
    spec = importlib.util.spec_from_file_location("profile_edm_auxiliary_gradients", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_split(tmp_path: Path, *, lengths: list[int]) -> Path:
    rows = []
    for index, length in enumerate(lengths):
        coords = np.stack(
            [np.arange(length), np.full(length, float(index)), np.zeros(length)],
            axis=1,
        ).astype(np.float32)
        sample = ProteinSample(
            sample_id=f"s{index:02d}",
            pdb_id=f"P{index:03d}",
            chain_id="A",
            sequence="A" * length,
            residue_ids=[str(i) for i in range(length)],
            ca_coordinates=coords,
            metadata={"experimental_method": "synthetic", "resolution_angstrom": 1.0},
        )
        rows.append(save_processed_sample(sample, tmp_path / "samples"))
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    return manifest


def _config(tmp_path: Path) -> dict:
    manifest = _write_split(tmp_path, lengths=[8, 12])
    normalization = tmp_path / "normalization.json"
    write_normalization(manifest, normalization)
    return {
        "train_manifest": str(manifest),
        "normalization_file": str(normalization),
        "seed": 123,
        "device": "cpu",
        "batch_size": 1,
        "mixed_precision": False,
        "amp_dtype": "float16",
        "diffusion_steps": 8,
        "prediction_parameterization": "v",
        "physical_auxiliary_loss_enabled": True,
        "physical_auxiliary_loss_weight": 0.01,
        "physical_auxiliary_loss_warmup_steps": 0,
        "edm_subset_size": 4,
        "edm_subsets_per_sample": 1,
        "edm_negative_weight": 1.0,
        "edm_rank3_weight": 1.0,
        "physical_auxiliary_seed": 2002,
        "adjacent_auxiliary_loss_enabled": True,
        "adjacent_auxiliary_loss_weight": 0.001,
        "adjacent_auxiliary_loss_warmup_steps": 0,
        "adjacent_auxiliary_huber_beta_angstrom": 0.25,
        "model": {
            "base_channels": 4,
            "channel_multipliers": [1, 2],
            "residual_blocks_per_level": 1,
            "attention_heads": 1,
            "time_embedding_dim": 16,
            "length_embedding_dim": 16,
            "max_length": 16,
        },
    }


def test_gradient_interaction_diagnostic_outputs_fixed_timesteps_and_empty_groups(tmp_path: Path) -> None:
    profiler = _load_profiler_module()
    output_dir = tmp_path / "diagnostic"
    summary = profiler._run_gradient_interaction_diagnostic(
        config=_config(tmp_path),
        checkpoint=None,
        output_dir=output_dir,
        fixed_timesteps=[0, 3],
        length_bins=[(1, 8), (9, 12), (13, 16)],
        max_batches_per_group=1,
        subset_size=4,
        subsets_per_sample=1,
        adjacent_beta_angstrom=0.25,
        bootstrap_iterations=8,
        seed=11,
        device=torch.device("cpu"),
    )

    observations = pd.read_parquet(output_dir / "observations.parquet")
    assert set(observations["fixed_timestep"]) == {0, 3}
    assert set(observations["length_bin"]) == {"1-8", "9-12"}
    assert summary["observation_count"] == 4
    assert {"length_bin": "13-16", "fixed_timestep": 0} in summary["empty_groups"]
    assert (output_dir / "summary_by_timestep.csv").exists()
    assert (output_dir / "summary_by_length.csv").exists()
    assert (output_dir / "summary_by_length_timestep.csv").exists()
    assert (output_dir / "diagnostic_summary.json").exists()
    assert (output_dir / "README.md").exists()
    assert (output_dir / "heatmap_cosine_edm_adjacent.png").exists()
    assert (output_dir / "heatmap_cosine_diffusion_adjacent.png").exists()
    assert (output_dir / "heatmap_adjacent_diffusion_norm_ratio.png").exists()


def test_gradient_interaction_diagnostic_is_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    profiler = _load_profiler_module()
    config = _config(tmp_path)
    kwargs = {
        "config": config,
        "checkpoint": None,
        "fixed_timesteps": [0, 3],
        "length_bins": [(1, 12)],
        "max_batches_per_group": 1,
        "subset_size": 4,
        "subsets_per_sample": 1,
        "adjacent_beta_angstrom": 0.25,
        "bootstrap_iterations": 4,
        "seed": 19,
        "device": torch.device("cpu"),
    }
    profiler._run_gradient_interaction_diagnostic(output_dir=tmp_path / "a", **kwargs)
    profiler._run_gradient_interaction_diagnostic(output_dir=tmp_path / "b", **kwargs)

    a = pd.read_parquet(tmp_path / "a" / "observations.parquet")
    b = pd.read_parquet(tmp_path / "b" / "observations.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_gradient_interaction_diagnostic_has_finite_norms_and_cosines(tmp_path: Path) -> None:
    profiler = _load_profiler_module()
    output_dir = tmp_path / "diagnostic"
    profiler._run_gradient_interaction_diagnostic(
        config=_config(tmp_path),
        checkpoint=None,
        output_dir=output_dir,
        fixed_timesteps=[1],
        length_bins=[(1, 12)],
        max_batches_per_group=1,
        subset_size=4,
        subsets_per_sample=1,
        adjacent_beta_angstrom=0.25,
        bootstrap_iterations=0,
        seed=23,
        device=torch.device("cpu"),
    )
    observations = pd.read_parquet(output_dir / "observations.parquet")
    columns = [
        "diffusion_grad_norm",
        "edm_grad_norm",
        "adjacent_grad_norm",
        "edm_to_diffusion_grad_ratio",
        "adjacent_to_diffusion_grad_ratio",
        "cosine_diffusion_edm",
        "cosine_diffusion_adjacent",
        "cosine_edm_adjacent",
    ]
    assert np.isfinite(observations[columns].to_numpy(dtype=np.float64)).all()


def test_gradient_interaction_diagnostic_does_not_update_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler = _load_profiler_module()
    captured: dict[str, DistanceUNet] = {}
    before: list[torch.Tensor] = []
    original = profiler.build_model_from_config

    def build_and_capture(config: dict) -> DistanceUNet:
        model = original(config)
        captured["model"] = model
        before.extend(tensor.clone() for tensor in model.state_dict().values())
        return model

    monkeypatch.setattr(profiler, "build_model_from_config", build_and_capture)
    profiler._run_gradient_interaction_diagnostic(
        config=_config(tmp_path),
        checkpoint=None,
        output_dir=tmp_path / "diagnostic",
        fixed_timesteps=[2],
        length_bins=[(1, 12)],
        max_batches_per_group=1,
        subset_size=4,
        subsets_per_sample=1,
        adjacent_beta_angstrom=0.25,
        bootstrap_iterations=0,
        seed=29,
        device=torch.device("cpu"),
    )
    model = captured["model"]
    after = list(model.state_dict().values())
    assert all(torch.equal(left, right) for left, right in zip(before, after, strict=True))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_grad_vector_zero_fills_none_gradients() -> None:
    profiler = _load_profiler_module()
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))
    first.grad = torch.tensor([0.5, -0.5])
    second.grad = None
    vector = profiler._grad_vector([first, second])
    assert torch.equal(vector, torch.tensor([0.5, -0.5, 0.0]))


def test_existing_profiler_behavior_remains_compatible(tmp_path: Path) -> None:
    profiler = _load_profiler_module()
    result = profiler._profile_one(
        config=_config(tmp_path),
        checkpoint=None,
        candidate_weights=[0.01],
        candidate_adjacent_weights=[0.001],
        max_batches=1,
        subset_size=4,
        subsets_per_sample=1,
        adjacent_beta_angstrom=0.25,
        min_length=None,
        max_length=None,
        device=torch.device("cpu"),
    )
    assert result["candidate_weights"] == [0.01]
    assert result["candidate_edm_weights"] == [0.01]
    assert result["candidate_adjacent_weights"] == [0.001]
    assert result["rows"]
