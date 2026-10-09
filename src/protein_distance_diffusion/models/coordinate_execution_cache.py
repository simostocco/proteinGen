"""Opt-in execution caches for unchanged E013/E014 mathematics.

The original functions remain the reference. Only static edges/RoPE and repeated
current-state edge features within ONE _single call are reused. No X-dependent
feature survives a forward call. No parameter or optimizer state is added.
"""

import contextvars
import math

import torch

from protein_distance_diffusion.models import e013_unconditional_equivariant_denoiser as core
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser

_SCOPE = contextvars.ContextVar("coordinate_execution_scope", default=None)
_EDGES = core.edges
_FEATURES = core.edge_features
_ROPE = core.VectorBlock.rope
_STATIC = {}


def cached_edges(x, k=24, dense=False):
    if _SCOPE.get() is None or (k and not dense):
        return _EDGES(x, k, dense)
    key = (len(x), x.device, k, dense)
    if key not in _STATIC:
        _STATIC[key] = _EDGES(x, k, dense)
    return _STATIC[key]


def cached_features(x, i, j, spatial):
    scope = _SCOPE.get()
    if scope is None:
        return _FEATURES(x, i, j, spatial)
    key = (id(x), id(i), id(j), id(spatial))
    if key not in scope:
        scope[key] = _FEATURES(x, i, j, spatial)
    return scope[key]


def cached_rope(self, z):
    if _SCOPE.get() is None:
        return _ROPE(self, z)
    key = ("rope", len(z), z.device, z.dtype)
    if key not in _STATIC:
        p = torch.arange(len(z), device=z.device, dtype=z.dtype)
        freq = torch.exp(-math.log(10000) * torch.arange(16, device=z.device, dtype=z.dtype) / 16)
        phase = p[:, None] * freq
        _STATIC[key] = phase.cos(), phase.sin()
    cosine, sine = _STATIC[key]
    a, b = z[:, 0::2], z[:, 1::2]
    return torch.stack([a * cosine - b * sine, a * sine + b * cosine], -1).flatten(-2)


class CachedCoordinateDenoiser(E014CapacityDenoiser):
    """Identical model state/forward operations with opt-in computation reuse."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        core.edges = cached_edges
        core.edge_features = cached_features
        core.VectorBlock.rope = cached_rope

    def _single(self, x, timestep):
        token = _SCOPE.set({})
        try:
            return super()._single(x, timestep)
        finally:
            _SCOPE.reset(token)
