"""Differentiable signed C-alpha internal coordinates, version ca_internal_signed_v1."""

import math
import torch
from ..contracts import BuildState, BuildAction

EPS = 1e-10


def mask_for_step(t):
    return (t >= 1, t >= 2, t >= 3)


def unit(v):
    norm = torch.linalg.vector_norm(v)
    if float(norm.detach()) < EPS:
        raise ValueError("degenerate bond/frame; no invented orientation")
    return v / norm


def dihedral(p0, p1, p2, p3):
    u = unit(p2 - p1)
    v = -(p1 - p0)
    v = unit(v - (v * u).sum() * u)
    w = p3 - p2
    w = unit(w - (w * u).sum() * u)
    return torch.atan2((torch.linalg.cross(u, v) * w).sum(), (v * w).sum())


def next_ca(ca, g):
    t = len(ca)
    d, theta, tau = g
    if t == 0:
        return g * 0
    if float(d.detach()) <= 0:
        raise ValueError("positive distance support")
    if t >= 2 and not 0 < float(theta.detach()) < math.pi:
        raise ValueError("angle open (0,pi); collinearity unsupported")
    if t >= 3 and not -math.pi <= float(tau.detach()) <= math.pi:
        raise ValueError("signed torsion [-pi,pi]")
    if t == 1:
        return ca[-1] + d * torch.tensor([1.0, 0.0, 0.0], dtype=ca.dtype, device=ca.device)
    u = unit(ca[-1] - ca[-2])
    if t == 2:
        # canonical bootstrap gauge; only the first three positions use a fixed global axis.
        e = torch.tensor([0.0, 1.0, 0.0], dtype=ca.dtype, device=ca.device)
        e = unit(e - (e * u).sum() * u)
    else:
        e = ca[-3] - ca[-2]
        e = unit(e - (e * u).sum() * u)
    n = torch.linalg.cross(u, e)
    return ca[-1] + d * (-torch.cos(theta) * u + torch.sin(theta) * (torch.cos(tau) * e + torch.sin(tau) * n))


def append(state, action):
    if type(state) != BuildState or type(action) != BuildAction or state.t >= state.N:
        raise ValueError("closed state/action and unfinished episode")
    if action.mask != mask_for_step(state.t) or not state.valid.all():
        raise ValueError("mask or missing observed prefix cannot build a complete joint action")
    if action.geometry.device != state.ca.device:
        raise ValueError("state/action device mismatch; transfer explicitly")
    x = next_ca(state.ca, action.geometry.to(state.ca.dtype))
    return BuildState(state.residues + (action.residue,), torch.cat([state.ca, x[None]], 0), state.N, plan=state.plan)


def inverse(ca):
    if ca.ndim != 2 or ca.shape[1] != 3 or not torch.isfinite(ca).all():
        raise ValueError("finite mapped coordinates")
    t = len(ca) - 1
    g = torch.zeros(3, dtype=ca.dtype, device=ca.device)
    if t >= 1:
        g[0] = torch.linalg.vector_norm(ca[-1] - ca[-2])
    if t >= 2:
        u = unit(ca[-2] - ca[-3])
        v = unit(ca[-1] - ca[-2])
        g[1] = torch.acos(torch.clamp((-u * v).sum(), -1.0, 1.0))
    if t >= 3:
        g[2] = dihedral(*ca[-4:])
    return g, mask_for_step(t)


def action_from_positions(residue, ca):
    g, mask = inverse(ca)
    return BuildAction(residue, g, mask)


def replay(N, actions, *, device="cpu"):
    s = BuildState.empty(N, device=device)
    for a in actions:
        s = append(s, BuildAction(a.residue, a.geometry.to(device), a.mask))
    return s
