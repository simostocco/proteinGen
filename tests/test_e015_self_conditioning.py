import inspect

import torch

from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e015_self_conditioning import make_self_condition


class Model(torch.nn.Module):
    def forward(self, x, t, lengths, mask):
        return x * 0.1


def test_no_target_api_and_detachment():
    assert "target" not in inspect.signature(make_self_condition).parameters
    x = torch.randn(2, 8, 3, requires_grad=True)
    mask = torch.ones(2, 8, dtype=torch.bool)
    t = torch.tensor([50, 400])
    diffusion = CoordinateVPDiffusion()
    previous, available = make_self_condition(Model(), diffusion, x, t, torch.tensor([8, 8]), mask, True)
    assert not previous.requires_grad and available.all()
    expected = diffusion.reconstruct_x0(x, t, Model()(x, t, None, mask), mask)
    assert torch.equal(previous, expected)
    previous, available = make_self_condition(Model(), diffusion, x, t, None, mask, False)
    assert torch.count_nonzero(previous) == 0 and not available.any()
