"""Operational handoff refusal and independent effective-gate reporting."""

import subprocess

import numpy as np
import pytest
import torch
from finalize_e014_status import summary_table
from report_e014_effective_gates import report
from upgrade_e014_terminal_guard import ADDED_LINE, FROZEN_PROBE_SOURCE, OLD_LINE, RUNNER, verify_guard_only_change

from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e014_runtime import noise_seed


def test_terminal_handoff_refuses_any_other_change():
    old = subprocess.check_output(["git", "show", f"{FROZEN_PROBE_SOURCE}:{RUNNER}"])
    new = old.replace(OLD_LINE.encode(), (OLD_LINE + ADDED_LINE).encode())
    verify_guard_only_change(old, new)
    with pytest.raises(ValueError, match="only the audited"):
        verify_guard_only_change(old, new + b"\n# unrelated source change\n")


def test_summary_handles_real_group_schema():
    table = summary_table(
        dict(groups=[dict(dimension="timestep", value=490, protein_condition_denominator=500, collapse_count=7)]),
        "timestep",
        ["collapse_count"],
    )
    assert "490 | 500 | 7" in table


def test_effective_gate_supplement_matches_live_forward_hooks(tmp_path):
    model = E014CapacityDenoiser().eval()
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(model=model.state_dict()), checkpoint)
    cfg = dict(diffusion_steps=500, coordinate_scale_angstrom=12.22820347644835, noise_timesteps=[50, 490])
    row = dict(sample_id="synthetic", length=20, length_stratum="20-64")

    class Cache:
        def get(self, _identity):
            return np.random.default_rng(14).normal(size=(20, 3)).astype(np.float32)

    cache = Cache()
    result = report(checkpoint, [row], "synthetic", cache, cfg, tmp_path / "gates", device="cpu")
    x, lengths, mask = coordinate_batch([row], cache, cfg["coordinate_scale_angstrom"], device="cpu")
    diffusion = CoordinateVPDiffusion(500)
    for timestep in cfg["noise_timesteps"]:
        t = torch.tensor([timestep])
        generator = torch.Generator().manual_seed(noise_seed("e014-paired-validation-v1", "synthetic", timestep))
        batch = diffusion.make_training_batch(x, mask, timesteps=t, generator=generator)
        _, components = model.telemetry(batch.noisy_coordinates, t, lengths, mask)
        component = components[0]
        expected = component["gate"] * component["reliability"] * (1 - t[0] / 499).square()
        observed = next(g for g in result["groups"] if g["dimension"] == "timestep" and g["value"] == timestep)
        assert observed["effective_gate_mean"] == pytest.approx(float(expected.mean()), abs=1e-7)
        assert observed["effective_gate_median"] == pytest.approx(float(expected.median()), abs=1e-7)
