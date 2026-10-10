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


def test_packed_batch_equals_independent_proteins_with_masks_and_context():
    model = small_model()
    torch.manual_seed(22)
    x = torch.randn(2, 25, 3)
    previous = torch.randn_like(x)
    lengths = torch.tensor([20, 25])
    mask = torch.arange(25)[None] < lengths[:, None]
    t = torch.tensor([5, 250])
    available = torch.tensor([True, False])
    packed = model(x, t, lengths, mask, previous, available)
    for b, n in enumerate(lengths.tolist()):
        expected, _ = model.single(x[b, :n], t[b], previous[b, :n] if available[b] else None)
        assert torch.allclose(packed[b, :n], expected, atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(packed[0, 20:]) == 0
