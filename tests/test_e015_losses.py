import torch

from protein_distance_diffusion.training.e015_losses import per_protein_losses


def test_equal_protein_padding_and_finite_gradients():
    torch.manual_seed(15)
    x = torch.randn(2, 20, 3)
    mask = torch.arange(20)[None] < torch.tensor([7, 20])[:, None]
    p = (x + 0.01 * torch.randn_like(x)).requires_grad_()
    all_losses = per_protein_losses(p, x, mask)
    one = per_protein_losses(p[:1, :7], x[:1, :7], mask[:1, :7])
    for key in all_losses:
        assert torch.allclose(all_losses[key][0], one[key][0])
    sum(v.mean() for v in all_losses.values()).backward()
    assert torch.isfinite(p.grad).all()
    assert torch.count_nonzero(p.grad[0, 7:]) == 0
