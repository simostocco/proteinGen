import torch
from test_e015_parity_equivariance import small_model

from protein_distance_diffusion.models.e015_multiscale_parity_denoiser import graph, pool_mean


def test_n500_parent_mapping_and_partial_pool():
    x = torch.arange(500).float()[:, None].expand(-1, 3)
    medium = pool_mean(x)
    coarse = pool_mean(medium)
    assert medium.shape == (125, 3) and coarse.shape == (32, 3)
    assert coarse[-1, 0] == 497.5
    model = small_model()
    _, _, _, _, shapes = model.trunk_single(torch.randn(500, 3), torch.tensor(50))
    assert shapes == [500, 125, 32]


def test_graph_sorted_unique_and_selfcondition_provenance():
    torch.manual_seed(1)
    x = torch.randn(20, 3)
    previous = torch.randn(20, 3)
    i, j, f, _ = graph(x, previous, 4, 4)
    ids = i * 20 + j
    assert torch.equal(ids, torch.unique(ids, sorted=True))
    assert f[:, -2].sum() > 0 and f[:, -1].all()
    assert all(bool(((i == z) & (j == z + 1)).any()) for z in range(19))
