import importlib.util
from pathlib import Path

import pytest
import torch

from protein_distance_diffusion.evaluation.e013_one_step_benchmark import adjudicate, diagnostics
from protein_distance_diffusion.training.e013_losses import coordinate_v_loss


def test_equal_protein_weight():
    mask = torch.tensor([[True, False, False], [True, True, True]])
    prediction = torch.tensor(
        [[[2.0, 2, 2], [100.0, 100, 100], [100.0, 100, 100]], [[1.0, 1, 1], [1.0, 1, 1], [1.0, 1, 1]]]
    )
    assert coordinate_v_loss(prediction, torch.zeros_like(prediction), mask) == 2.5


def test_reference_diagnostics_and_gates():
    x = torch.randn(20, 3)
    x -= x.mean(0)
    r = diagnostics(x, x, x, x, 12.22820347644835)
    assert r["v_mse"] == 0 and r["local_mse_angstrom2"] == 0
    assert r["aligned_rmsd_angstrom"] < 1e-4
    assert r["torsion_mse"] == 0 and not r["collapsed"]
    control = [dict(r, stratum="20-64", v_mse=1.0, local_mse_angstrom2=1.0, torsion_mse=1.0)]
    candidate = [dict(control[0], v_mse=0.9, local_mse_angstrom2=0.9)]
    gate = dict(aggregate=0.05, length=0.1, improvement=0.05)
    assert adjudicate(control, candidate, gate)["passed"]
    candidate[0]["collapsed"] = True
    assert not adjudicate(control, candidate, gate)["passed"]


def test_generated_paths_reject_non_d(tmp_path):
    source = Path(__file__).parents[1] / "scripts/run_e013_unconditional_equivariant_denoiser.py"
    spec = importlib.util.spec_from_file_location("e013_runner", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError):
        module.d_path("/tmp/proteingen_forbidden")
    with pytest.raises(ValueError):
        module.d_path("/mnt/c/proteingen_forbidden")
    assert str(module.d_path("/mnt/d/Simone/proteinGen/e013_artifacts")).startswith("/mnt/d/")


def test_future_auxiliaries_disabled_at_max_noise():
    from protein_distance_diffusion.training.e013_losses import low_noise_geometry_auxiliaries

    p = torch.randn(2, 20, 3, requires_grad=True)
    q = torch.randn_like(p)
    losses = low_noise_geometry_auxiliaries(p, q, torch.ones(2, 20, dtype=torch.bool), torch.tensor([499, 499]))
    assert all(float(v.detach()) == 0 for v in losses.values())
    sum(losses.values()).backward()
    assert torch.isfinite(p.grad).all()
