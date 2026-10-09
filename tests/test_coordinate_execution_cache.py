"""Execution-cache isolation, original state preservation and current-X refresh."""

import torch

from protein_distance_diffusion.models.coordinate_execution_cache import _SCOPE, CachedCoordinateDenoiser
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser


def test_cache_same_state_and_current_geometry():
    torch.set_num_threads(1)
    torch.manual_seed(143)
    original = E014CapacityDenoiser()
    cached = CachedCoordinateDenoiser()
    cached.load_state_dict(original.state_dict())
    assert sum(p.numel() for p in cached.parameters()) == 980794
    assert cached.state_dict().keys() == original.state_dict().keys()
    x = torch.randn(1, 20, 3)
    args = (torch.tensor([50]), torch.tensor([20]), torch.ones(1, 20, dtype=torch.bool))
    for coordinates in (x, x * 2, x + torch.tensor([3.0, -2.0, 1.0])):
        assert torch.equal(original(coordinates, *args), cached(coordinates, *args))
        assert _SCOPE.get() is None
    output = cached(x, *args)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(cached(x @ rotation.T + 2, *args), output @ rotation.T, atol=5e-5, rtol=5e-5)
