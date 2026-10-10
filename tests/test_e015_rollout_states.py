import torch

from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion, center_coordinates
from protein_distance_diffusion.training.e015_rollout_states import induced_state


class Model(torch.nn.Module):
    def forward(self, x, t, lengths, mask, xprev=None, available=None):
        return x * 0.1


def test_implied_velocity_reconstructs_known_clean_target():
    torch.manual_seed(15)
    mask = torch.ones(2, 10, dtype=torch.bool)
    x0 = center_coordinates(torch.randn(2, 10, 3), mask)
    xt = center_coordinates(torch.randn(2, 10, 3), mask)
    diffusion = CoordinateVPDiffusion()
    state, t, target = induced_state(
        Model(), diffusion, xt, torch.tensor([400, 250]), x0, torch.tensor([10, 10]), mask, 3
    )
    recovered = diffusion.reconstruct_x0(state, t, target, mask)
    assert torch.allclose(recovered, x0, atol=1e-6)
    assert not state.requires_grad and not target.requires_grad
    assert torch.equal(t, torch.tensor([397, 247]))
