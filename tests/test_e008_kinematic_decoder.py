import math

import torch

from protein_distance_diffusion.models.e008_kinematic_decoder import (
    BackboneKinematicDecoder,
    cartesian_to_internal,
    decoder_parameter_count,
    internal_to_cartesian,
)


def _trace(n: int, dtype=torch.float64) -> torch.Tensor:
    angles = torch.linspace(1.5, 2.2, n - 2, dtype=dtype)
    torsions = torch.linspace(-2.5, 2.6, n - 3, dtype=dtype)
    seed = torch.tensor([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [5.0, math.sqrt(3.8**2 - 1.2**2), 0.0]], dtype=dtype)
    return internal_to_cartesian(seed, angles, torsions, bond_length=3.8)


def test_cartesian_internal_round_trip_and_bond_geometry():
    for n in (20, 64, 128, 256, 384, 500):
        xyz = _trace(n)
        bonds, angles, torsions = cartesian_to_internal(xyz)
        rebuilt = internal_to_cartesian(xyz[:3], angles, torsions, bond_length=3.8)
        assert torch.allclose(torch.linalg.vector_norm(xyz[1:] - xyz[:-1], dim=-1), bonds, atol=1e-9)
        assert torch.allclose(
            torch.linalg.vector_norm(rebuilt[1:] - rebuilt[:-1], dim=-1),
            torch.full((n - 1,), 3.8, dtype=xyz.dtype),
            atol=1e-9,
        )
        # Internal-coordinate reconstruction is rigidly identical, including handedness.
        assert torch.allclose(xyz, rebuilt, atol=1e-8)


def test_signed_torsion_changes_sign_under_reflection():
    xyz = _trace(32)
    _, _, torsion = cartesian_to_internal(xyz)
    reflected = xyz * torch.tensor([-1.0, 1.0, 1.0], dtype=xyz.dtype)
    _, _, reflected_torsion = cartesian_to_internal(reflected)
    assert torch.allclose(torch.sin(reflected_torsion), -torch.sin(torsion), atol=1e-8)
    assert torch.allclose(torch.cos(reflected_torsion), torch.cos(torsion), atol=1e-8)


def test_round_trip_preserves_variable_observed_bond_lengths():
    xyz = _trace(96)
    bonds, angles, torsions = cartesian_to_internal(xyz)
    varied = bonds * torch.linspace(0.96, 1.04, len(bonds), dtype=bonds.dtype)
    source = internal_to_cartesian(xyz[:3], angles, torsions, bond_lengths=varied)
    got_bonds, got_angles, got_torsions = cartesian_to_internal(source)
    rebuilt = internal_to_cartesian(source[:3], got_angles, got_torsions, bond_lengths=got_bonds)
    assert torch.allclose(got_bonds, varied, atol=1e-9)
    assert torch.allclose(rebuilt, source, atol=1e-8)


def test_decoder_masks_padding_and_guarantees_bonds():
    model = BackboneKinematicDecoder(width=16, layers=1, pair_rbf_bins=4)
    valid = _trace(32, dtype=torch.float32)
    torch.manual_seed(12)
    valid = valid + torch.randn_like(valid) * 0.08
    coarse = torch.zeros(1, 40, 3)
    coarse[0, :32] = valid
    mask = torch.zeros(1, 40, dtype=torch.bool)
    mask[0, :32] = True
    output = model(coarse, mask)
    xyz = output["coordinates"][0]
    bond = torch.linalg.vector_norm(xyz[1:32] - xyz[:31], dim=-1)
    assert torch.allclose(bond, torch.full_like(bond, 3.8), atol=2e-5)
    assert torch.count_nonzero(xyz[32:]) == 0
    assert torch.allclose(xyz[:32].mean(0), torch.zeros(3), atol=2e-5)
    assert decoder_parameter_count(model) < 250_000


def test_decoder_equivariant_to_proper_rigid_transform():
    torch.manual_seed(3)
    model = BackboneKinematicDecoder(width=16, layers=1, pair_rbf_bins=4).eval()
    xyz = _trace(24, dtype=torch.float32)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, -1] *= -1
    shift = torch.tensor([4.0, -2.0, 8.0])
    mask = torch.ones(1, 24, dtype=torch.bool)
    a = model(xyz[None], mask)["coordinates"]
    b = model((xyz @ q + shift)[None], mask)["coordinates"]
    assert torch.allclose(b, a @ q, atol=2e-4, rtol=2e-4)


def test_forward_kinematics_and_decoder_have_finite_gradients():
    model = BackboneKinematicDecoder(width=16, layers=1, pair_rbf_bins=4)
    xyz = _trace(20, dtype=torch.float32).detach().requires_grad_(True)
    mask = torch.ones(1, 20, dtype=torch.bool)
    output = model(xyz[None], mask)["coordinates"]
    loss = output.square().mean()
    loss.backward()
    assert xyz.grad is not None and torch.isfinite(xyz.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
