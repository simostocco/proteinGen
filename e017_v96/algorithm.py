"""One intervention: explicit burial-region token; no added parameters."""
import torch

VERSION = 'EXPLICIT_BURIAL_REGION_TOKEN_1'


def augment(batch):
    """Pure prefix-public deterministic observation map, used before actor forward.

    Context layout is the sealed V9.1 layout. Existing motif tokens and all
    context, priors, masks, action transforms and likelihoods are preserved.
    """
    context = batch['context']
    token = batch['goals'].new_zeros((len(context), 1, 36))
    token[:, 0, 0] = -1  # distinct from nonnegative motif-role indices
    token[:, 0, 1] = context[:, 7] - context[:, 6]
    token[:, 0, 2] = context[:, 6]  # requested start/N
    token[:, 0, 3] = context[:, 7]  # requested end/N
    token[:, 0, 4] = context[:, 2]  # requested burial maximum
    token[:, 0, 5] = context[:, 3]  # original contrast requirement
    token[:, 0, 32] = (context[:, 6] + context[:, 7]) / 2
    token[:, 0, 33] = 1
    token[:, 0, 35] = 1
    out = dict(batch)
    out['goals'] = torch.cat([batch['goals'], token], dim=1)
    out['goal_valid'] = torch.cat([batch['goal_valid'], torch.ones((len(context), 1), dtype=torch.bool, device=context.device)], dim=1)
    return out


def install(actor):
    """Attach only to the independent actor, never J1 or the immutable parent.

    Both inference and full-trace gradient replay call this same module. Hook
    installation is a required checkpoint binding, not serialized model state.
    """
    assert not getattr(actor, '_v96_burial_token_installed', False)
    def hook(module, args):
        return (augment(args[0]),)
    handle = actor.register_forward_pre_hook(hook)
    actor._v96_burial_token_installed = True
    return handle
