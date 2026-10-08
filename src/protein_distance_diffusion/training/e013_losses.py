"""Equal-protein v loss; geometry auxiliaries remain disabled by default."""
import torch


def coordinate_v_loss(prediction, target, mask):
    per_protein = ((prediction-target).square()*mask[..., None]).sum((1, 2))/(3*mask.sum(1))
    return per_protein.mean()


def local_distance_loss(prediction, target, mask):
    losses = []
    for k in (1, 2, 3):
        valid = mask[:, k:] & mask[:, :-k]
        error = ((prediction[:, k:]-prediction[:, :-k]).norm(dim=-1)
                 -(target[:, k:]-target[:, :-k]).norm(dim=-1)).square()
        losses.append(((error*valid).sum(1)/valid.sum(1).clamp_min(1)).mean())
    return torch.stack(losses).mean()
