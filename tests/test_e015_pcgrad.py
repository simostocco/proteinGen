import torch

from protein_distance_diffusion.training.e015_pcgrad import backward_objectives, combine_gradients


def test_projection_matches_full_vector_and_missing_gradients():
    gv = [torch.tensor([1.0, 0.0]), torch.tensor([2.0]), None]
    gs = [torch.tensor([-2.0, 1.0]), None, torch.tensor([3.0])]
    result, stats = combine_gradients(gv, gs)
    assert torch.allclose(result[0], torch.tensor([-0.6, 1.0]))
    assert torch.equal(result[1], gv[1])
    assert torch.equal(result[2], gs[2])
    assert stats["conflict"]


def test_accumulation_and_two_objectives():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    for _ in range(2):
        backward_objectives(p.sum(), -p[0], [p], accumulation_weight=0.5)
    assert torch.allclose(p.grad, torch.tensor([0.5, 1.5]))


def test_nonconflicting_unchanged():
    g, stats = combine_gradients([torch.tensor([1.0])], [torch.tensor([2.0])])
    assert g[0].item() == 3 and not stats["conflict"]
