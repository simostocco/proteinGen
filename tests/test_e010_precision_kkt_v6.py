import pytest
import torch

from protein_distance_diffusion.training import e010_precision_kkt_v6 as audit
from protein_distance_diffusion.training.e010_cartesian_oracle_v5 import trajectory
from protein_distance_diffusion.training.e010_phase4d_objective_v2 import smooth_bounded_local


@pytest.mark.parametrize("v", [[0, 0, 0], [0.03, -0.02, 0.01], [10, 20, -5]])
def test_radial_jacobian_and_eigenvalues(v):
    v = torch.tensor(v, dtype=torch.float64, requires_grad=True)
    info = audit.radial_geometry(v)
    actual = torch.autograd.functional.jacobian(lambda x: smooth_bounded_local(x, 0.04), v)
    torch.testing.assert_close(actual, info["jacobian"], atol=1e-14, rtol=1e-10)
    expected = torch.stack([info["radial_eigenvalue"], info["a"], info["a"]])
    torch.testing.assert_close(torch.linalg.eigvalsh(actual), expected, atol=1e-14, rtol=1e-9)
    torch.testing.assert_close(info["condition_number"], 1 + v.square().sum() / 0.04**2)


def geometry():
    t = torch.arange(9, dtype=torch.float64)
    p = torch.stack([2 * t, t.sin(), t.cos()], -1)[None]
    return dict(pg=p, source=p + 0.2, target=p + 0.3, mask=torch.ones(1, 9, dtype=torch.bool))


def test_float64_objective_parity_and_decomposition():
    b = geometry()
    p = b["pg"].float().double()
    old = audit.objective(p, b)
    pure = audit.objective(p, b, pure_float64=True)
    for key in old:
        torch.testing.assert_close(old[key], pure[key], atol=1e-15, rtol=1e-14)
    assert pure["total"] == pure["local"] + 16.8 * pure["cartesian"] + 2 * pure["chiral"]


def test_correction_gradient_extraction_chain_rule_and_no_input_mutation():
    b = geometry()
    before = b["pg"].clone()
    v = torch.full((4, 1, 9, 3), 0.01, dtype=torch.float64, requires_grad=True)
    tr = trajectory(b["pg"], b["mask"], v / 0.04)
    loss = audit.objective(tr["prediction"], b, pure_float64=True)["total"]
    gv = torch.autograd.grad(loss, v, retain_graph=True)[0]
    corrections = torch.stack([s["delta"].detach() for s in tr["steps"]]).requires_grad_()
    direct = audit.correction_trajectory(b["pg"], b["mask"], corrections)
    torch.testing.assert_close(direct["prediction"], tr["prediction"], atol=0, rtol=0)
    gd = torch.autograd.grad(audit.objective(direct["prediction"], b, pure_float64=True)["total"], corrections)[0]
    chain = torch.einsum("...ij,...j->...i", audit.radial_geometry(v)["jacobian"], gd)
    torch.testing.assert_close(gv, chain, atol=1e-14, rtol=1e-12)
    assert b["pg"].grad is None
    torch.testing.assert_close(b["pg"], before, atol=0, rtol=0)


def test_float64_directional_finite_differences():
    b = geometry()
    d = torch.full((4, 1, 9, 3), 0.001, dtype=torch.float64, requires_grad=True)

    def fn(d):
        p = audit.correction_trajectory(b["pg"], b["mask"], d)["prediction"]
        return audit.objective(p, b, pure_float64=True)["total"]

    g = torch.autograd.grad(fn(d), d)[0]
    checks = audit.directional_checks(fn, d.detach(), g, [1e-6, 1e-5, 1e-4])
    assert all(row["passed"] for row in checks)


def test_kkt_boundary_interior_and_fixed_cases():
    d = torch.tensor([[0.04, 0, 0], [0.04, 0, 0], [0.04, 0, 0], [0.02, 0, 0], [0, 0, 0]], dtype=torch.float64)
    g = torch.tensor([[-2.0, 0, 0], [2.0, 0, 0], [-2.0, 3, 0], [-2.0, 0, 0], [4.0, 5, 6]], dtype=torch.float64)
    r = audit.kkt(d, g, torch.tensor([True, True, True, True, False]))
    torch.testing.assert_close(r["residual_norm"], torch.tensor([0.0, 2, 3, 2, 0], dtype=torch.float64))
    assert r["multiplier"][0] == 25
    assert r["feasible_radial_residual"][0] == 0
    assert r["tangent_residual"][2] == 3


def test_shadow_perturbations_feasible_and_descend():
    d = torch.tensor([[0.04, 0, 0], [0.01, 0, 0]], dtype=torch.float64)
    g = torch.tensor([[-2.0, 3, 0], [1.0, 0, 0]], dtype=torch.float64)
    for tangent in [False, True]:
        direction = audit.shadow_direction(d, g, torch.ones(2, dtype=torch.bool), tangential=tangent)
        new = audit.project_ball(d + 1e-6 * direction)
        assert new.norm(dim=-1).max() <= 0.040000000000001
        assert ((new - d) * g).sum() < 0


def test_initialization_directions_deterministic_no_neural_parameters():
    state = torch.get_rng_state().clone()
    a = audit.deterministic_directions((4, 1, 9, 3))
    b = audit.deterministic_directions((4, 1, 9, 3))
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
        torch.testing.assert_close(x.norm(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.equal(state, torch.get_rng_state())


def test_physical_trajectory_exact_historical_arithmetic():
    b = geometry()
    z = torch.arange(108, dtype=torch.float64).reshape(4, 1, 9, 3) / 13
    old = trajectory(b["pg"], b["mask"], z)
    new = audit.physical_trajectory(b["pg"], b["mask"], 0.04 * z)
    for a, c in zip(old["states"], new["states"], strict=True):
        torch.testing.assert_close(a, c, atol=0, rtol=0)


def test_precision_replay_exact_solver_code_and_optimizer_contract():
    from types import FunctionType

    from protein_distance_diffusion.training.e010_cartesian_oracle_v5 import solve
    from scripts.replay_e010_precision_kkt_v6 import pure_components

    clone = FunctionType(solve.__code__, {**solve.__globals__, "objective_components": pure_components})
    assert clone.__code__ is solve.__code__
    assert {k: v for k, v in clone.__globals__.items() if k != "objective_components"} == {
        k: v for k, v in solve.__globals__.items() if k != "objective_components"
    }
    assert solve.__globals__["objective_components"] is not pure_components
