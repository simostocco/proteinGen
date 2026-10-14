import torch
from ..contracts import BCPrefixTarget, AA


def diagnostic_bc_loss(actor, target, geometry_weight=1.0):
    if type(target) != BCPrefixTarget:
        raise ValueError("separate supervision target required")
    h = actor.encode(target.state)
    aa_loss = torch.nn.functional.cross_entropy(
        actor.aa_head(h)[None], torch.tensor([AA.index(target.next_action.residue)])
    )
    geom_loss = -actor.geometry_distribution(target.state, target.next_action.residue, h).log_prob(
        target.next_action.geometry
    )
    return aa_loss + geometry_weight * geom_loss, {
        "aa_NLL": aa_loss,
        "geometry_NLL": geom_loss,
        "scope": "DIAGNOSTIC_ONLY_NOT_PRODUCTION_TRAINING",
    }
