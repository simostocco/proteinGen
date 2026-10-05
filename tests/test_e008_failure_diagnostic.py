import torch

from scripts import diagnose_e008_failure as diagnostic


def test_diagnostic_plan_validates_pinned_evidence_without_cuda():
    cfg = diagnostic.load_config("configs/e008_failure_diagnostic_v1.yaml")
    assert cfg["updates"] == 1000
    assert cfg["authorization"]["authorizes_2000_update_pilot"] is False


def test_aligned_metrics_are_rigid_transform_invariant():
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 1.0]])
    q = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    y = x @ q + torch.tensor([3.0, -2.0, 4.0])
    metrics = diagnostic._metrics(y, x)
    assert metrics["aligned_coordinate_rmse_angstrom"] < 1e-5
    assert abs(metrics["error_linear_slope_angstrom_per_residue"]) < 1e-5
    assert len(metrics["normalized_position_quartile_rmse_angstrom"]) == 4
