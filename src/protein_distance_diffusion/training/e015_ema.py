"""FP32 EMA with explicit architecture consistency and buffer copying."""

import copy

import torch


def make_ema(model):
    ema = copy.deepcopy(model).float().eval()
    ema.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema, model, decay=0.9999):
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must lie in [0,1)")
    source = dict(model.named_parameters())
    destination = dict(ema.named_parameters())
    if source.keys() != destination.keys():
        raise ValueError("EMA architecture mismatch")
    for name, p in destination.items():
        p.lerp_(source[name].detach().float(), 1 - decay)
    source_buffers = dict(model.named_buffers())
    for name, buffer in ema.named_buffers():
        buffer.copy_(source_buffers[name])
