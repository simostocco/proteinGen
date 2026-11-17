"""Portable checks for the selected E016/E017 geometry and SAC primitives."""

import math

import pytest
import torch

from protein_joint_sac.contracts import AA, BuildAction
from protein_joint_sac.geometry.ca_builder import inverse, mask_for_step, replay
from protein_joint_sac.models.hybrid_distributions import GeometryDistribution
from protein_joint_sac.rl.soft_targets import discrete_objective, soft_target


def test_signed_geometry_roundtrip_and_gradients():
    actions = []
    for i in range(20):
        mask = mask_for_step(i)
        g = torch.tensor([3.8, 1.9, (-1) ** i * 0.7], dtype=torch.float64)
        g = g * torch.tensor(mask)
        actions.append(BuildAction(AA[i], g, mask))
    state = replay(20, actions)
    for i, action in enumerate(actions):
        g, mask = inverse(state.ca[: i + 1])
        torch.testing.assert_close(g, action.geometry, atol=1e-11, rtol=1e-11)
        assert mask == action.mask
    mu = torch.zeros(3, requires_grad=True)
    dist = GeometryDistribution(mu, torch.zeros(3), (True, True, True))
    g, logp = dist.rsample()
    torch.testing.assert_close(logp, dist.log_prob(g))
    logp.backward()
    assert torch.isfinite(mu.grad).all()
    assert torch.isneginf(dist.log_prob(torch.tensor([-1.0, 1.0, 0.0])))


def test_sac_terminal_mask_and_exact_discrete_entropy():
    result = soft_target(
        torch.tensor([2.0, 2.0]), torch.tensor([True, False]),
        torch.tensor([float('nan'), 5.0]), torch.zeros(2), torch.zeros(2), 0.9, 0.1, 0.1,
    )
    torch.testing.assert_close(result, torch.tensor([2.0, 6.5]))
    loss = discrete_objective(torch.zeros(20), torch.ones(20), 0.1)
    assert float(loss) == pytest.approx(-1 - 0.1 * math.log(20))
    with pytest.raises(ValueError, match='discount'):
        soft_target(0, True, 0, 0, 0, 1.1, 0, 0)
