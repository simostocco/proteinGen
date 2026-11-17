"""E013 structure-only scalar/vector coordinate-v model. No sequence inputs."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(a, b, c):
    return nn.Sequential(nn.Linear(a, b), nn.SiLU(), nn.Linear(b, c))


def radial_basis(d):
    # Exact E007 basis: centers 0..5, gamma=1/spacing**2 (overlapping Gaussians).
    centers = torch.linspace(0, 5, 16, device=d.device, dtype=d.dtype)
    return torch.exp(-(d[..., None] - centers).square() / (centers[1] - centers[0]).square())


def edges(x, k=24, dense=False):
    """Sorted unique directed edges, spatial provenance; ties use chain index.

    Single unpadded chain. cdist and stable sort cost O(N²) space and
    O(N² log N) time; message computation remains O((k+6)N).
    """
    n = len(x)
    ids = torch.arange(n, device=x.device)
    sep = ids[None, :] - ids[:, None]
    polymer = (sep.abs() >= 1) & (sep.abs() <= 3)
    spatial = torch.zeros_like(polymer)
    if dense:
        spatial = sep != 0
    elif k:
        with torch.no_grad():
            d = torch.cdist(x, x).masked_fill(sep == 0, float("inf"))
            nearest = torch.argsort(d, dim=-1, stable=True)[:, : min(k, n - 1)]
            spatial.scatter_(1, nearest, True)
    i, j = (polymer | spatial).nonzero(as_tuple=True)
    return i, j, spatial[i, j]


def edge_features(x, i, j, spatial):
    r = x[j] - x[i]
    d = torch.linalg.vector_norm(r, dim=-1)
    sep = (j - i).to(x.dtype)
    n = len(x)
    f = torch.cat(
        [
            radial_basis(d),
            (sep / max(n - 1, 1))[:, None],
            (torch.log1p(sep.abs()) / math.log(max(n, 2)))[:, None],
            torch.stack([sep.abs() == z for z in (1, 2, 3)], -1).to(x.dtype),
            spatial[:, None].to(x.dtype),
        ],
        -1,
    )
    return f, r / (d[:, None] + 1e-6)


def local_geometry(x):
    """32 features, frame columns e1/e2/e3; invalid frames are zero.

    Six distances + 18 frame coordinates + angle + torsion sin/cos + q
    + frame validity + torsion validity + sin(angle) + neighbor coverage.
    """
    n = len(x)
    ids = torch.arange(n, device=x.device)

    def offset(k):
        valid = (ids + k >= 0) & (ids + k < n)
        return (x[(ids + k).clamp(0, n - 1)] - x) * valid[:, None], valid

    left, vl = offset(-1)
    b, vr = offset(1)
    a = -left
    an, bn = a.norm(dim=-1), b.norm(dim=-1)
    e1 = b / bn.clamp_min(1e-8)[:, None]
    projected = a - (a * e1).sum(-1)[:, None] * e1
    pn = projected.norm(dim=-1)
    assess = vl & vr & (an > 1e-6) & (bn > 1e-6) & (pn > 1e-6)
    e2 = projected / pn.clamp_min(1e-8)[:, None]
    frame = torch.stack([e1, e2, torch.linalg.cross(e1, e2)], -1) * assess[:, None, None]
    cross = torch.linalg.cross(a, b)
    geom = cross.norm(dim=-1) / (an * bn).clamp_min(1e-8)
    angle = (a * b).sum(-1) / (an * bn).clamp_min(1e-8)
    two, vtwo = offset(2)
    c = two - b
    cross2 = torch.linalg.cross(b, c)
    c1n, c2n = cross.norm(dim=-1), cross2.norm(dim=-1)
    torsion_valid = assess & vtwo & (c1n > 1e-6) & (c2n > 1e-6)
    n1 = cross / c1n.clamp_min(1e-8)[:, None]
    n2 = cross2 / c2n.clamp_min(1e-8)[:, None]
    cosine = (n1 * n2).sum(-1) * torsion_valid
    sine = (torch.linalg.cross(n1, n2) * e1).sum(-1) * torsion_valid
    q = (cross * c).sum(-1) / (an * bn * c.norm(dim=-1)).clamp_min(1e-8) * torsion_valid
    neighbors = [offset(k) for k in (-3, -2, -1, 1, 2, 3)]
    distances = torch.stack([r.norm(dim=-1) for r, _ in neighbors], -1)
    coords = torch.cat([torch.einsum("nac,na->nc", frame, r) for r, _ in neighbors], -1)
    coverage = torch.stack([v for _, v in neighbors], -1).to(x.dtype).mean(-1)
    extra = torch.stack([angle, sine, cosine, q, assess, torsion_valid, geom, coverage], -1).to(x.dtype)
    return torch.cat([distances, coords, extra], -1), frame, geom * assess


class VectorBlock(nn.Module):
    def __init__(self, cs=128, cv=32):
        super().__init__()
        self.norm = nn.LayerNorm(cs)
        self.message = mlp(2 * cs + 22 + 64 + cv, cs, cs + 2 * cv + 1)
        self.update = mlp(2 * cs + cv, cs, cs)
        self.mix = nn.Linear(cv, cv, bias=False)
        self.query = nn.Linear(cs, 32, bias=False)
        self.key = nn.Linear(cs, 32, bias=False)
        self.cs, self.cv = cs, cv

    def rope(self, z):
        p = torch.arange(len(z), device=z.device, dtype=z.dtype)
        freq = torch.exp(-math.log(10000) * torch.arange(16, device=z.device, dtype=z.dtype) / 16)
        phase = p[:, None] * freq
        a, b = z[:, 0::2], z[:, 1::2]
        return torch.stack([a * phase.cos() - b * phase.sin(), a * phase.sin() + b * phase.cos()], -1).flatten(-2)

    def forward(self, h, v, x, time, graph):
        i, j, spatial = graph
        ef, u = edge_features(x, i, j, spatial)
        hn = self.norm(h)
        # Magnitude RMS scaling is SO(3) equivariant; never normalize xyz axes.
        vn = v / (1 + v.square().sum(-1).mean(-1, keepdim=True)).sqrt()[..., None]
        inv = (vn[i] * u[:, None]).sum(-1)
        m = self.message(torch.cat([hn[i], hn[j], ef, time.expand(len(i), -1), inv], -1))
        sm, a, b, logits = torch.split(m, [self.cs, self.cv, self.cv, 1], -1)
        q, key = self.rope(self.query(hn)), self.rope(self.key(hn))
        logits = logits[:, 0] + (q[i] * key[j]).sum(-1) / math.sqrt(32)
        maxima = logits.new_full((len(h),), -torch.inf)
        maxima.scatter_reduce_(0, i, logits, reduce="amax", include_self=True)
        weights = (logits - maxima[i]).exp()
        denom = weights.new_zeros(len(h)).index_add_(0, i, weights)
        weights = weights / denom[i].clamp_min(1e-8)
        vm = a[..., None] * u[:, None] + b.tanh()[..., None] * vn[j]
        va = torch.zeros_like(v).index_add_(0, i, weights[:, None, None] * vm)
        sa = torch.zeros_like(h).index_add_(0, i, weights[:, None] * sm)
        v = v + 0.1 * self.mix(va.transpose(-1, -2)).transpose(-1, -2)
        h = h + 0.1 * self.update(torch.cat([hn, sa, v.norm(dim=-1)], -1))
        return h, v


class E013UnconditionalEquivariantDenoiser(nn.Module):
    def __init__(self, scalar_channels=128, vector_channels=32, spatial_k=24, coarse_stride=8, diffusion_steps=500):
        super().__init__()
        cs, cv = scalar_channels, vector_channels
        self.cs, self.cv, self.k, self.stride = cs, cv, spatial_k, coarse_stride
        self.diffusion_steps = diffusion_steps
        self.time = mlp(64, 64, 64)
        self.length = mlp(3, 16, 16)
        self.input = nn.Linear(85, cs)
        self.blocks = nn.ModuleList([VectorBlock(cs, cv) for _ in range(4)])
        self.global_blocks = nn.ModuleList([VectorBlock(cs, cv) for _ in range(2)])
        self.broadcast_h = nn.Linear(cs, cs)
        self.broadcast_v = nn.Linear(cv, cv, bias=False)
        self.head = mlp(cs, cs, cv)
        self.local_input = nn.Linear(32, 128)
        self.local_blocks = nn.ModuleList([mlp(256, 128, 128) for _ in range(2)])
        self.local_head = nn.Linear(128, 3)
        self.gate = mlp(65, 64, 1)

    def _single(self, x, timestep):
        n = len(x)
        x = x - x.mean(0)
        freq = torch.exp(-math.log(10000) * torch.arange(32, device=x.device, dtype=x.dtype) / 32)
        phase = timestep.to(x.dtype) * freq
        time = self.time(torch.cat([phase.sin(), phase.cos()]))
        length = self.length(x.new_tensor([n / 500, math.log(n) / math.log(500), 1 / math.sqrt(n)]))
        ids = torch.arange(n, device=x.device, dtype=x.dtype)
        pos = torch.stack(
            [
                ids / max(n - 1, 1),
                (-ids / 8).exp(),
                (-(n - 1 - ids) / 8).exp(),
                (-ids / 32).exp(),
                (-(n - 1 - ids) / 32).exp(),
            ],
            -1,
        )
        h = self.input(torch.cat([time.expand(n, -1), length.expand(n, -1), pos], -1))
        v = x.new_zeros(n, self.cv, 3)
        graph = edges(x, self.k)
        for block in self.blocks:
            h, v = block(h, v, x, time, graph)
        segment = torch.arange(n, device=x.device) // self.stride
        m = math.ceil(n / self.stride)
        counts = torch.bincount(segment, minlength=m).to(x.dtype)

        def pool(z):
            return z.new_zeros((m,) + z.shape[1:]).index_add_(0, segment, z) / counts.reshape(
                (m,) + (1,) * (z.ndim - 1)
            )

        hc, vc, xc = pool(h), pool(v), pool(x)
        coarse_graph = edges(xc, dense=True)
        for block in self.global_blocks:
            hc, vc = block(hc, vc, xc, time, coarse_graph)
        h = h + self.broadcast_h(hc)[segment]
        v = v + self.broadcast_v(vc.transpose(-1, -2)).transpose(-1, -2)[segment]
        global_v = (self.head(h)[..., None] * v).sum(1) / math.sqrt(self.cv)
        features, frame, reliability = local_geometry(x)
        lh = F.silu(self.local_input(features))
        pi, pj, _ = edges(x, k=0)
        degree = torch.bincount(pi, minlength=n).clamp_min(1).to(x.dtype)
        for block in self.local_blocks:
            neighbors = torch.zeros_like(lh).index_add_(0, pi, lh[pj]) / degree[:, None]
            lh = lh + 0.1 * block(torch.cat([lh, neighbors], -1))
        local_v = torch.einsum("nac,nc->na", frame, self.local_head(lh))
        gate = torch.sigmoid(self.gate(torch.cat([time.expand(n, -1), reliability[:, None]], -1)))[:, 0]
        # Explicit smooth low-noise envelope; ablate only in a later experiment.
        envelope = (1 - timestep.to(x.dtype) / (self.diffusion_steps - 1)).clamp(0, 1).square()
        out = global_v + (gate * reliability * envelope)[:, None] * local_v
        return out - out.mean(0)

    def forward(self, noisy_coordinates, timesteps, lengths, residue_mask):
        b, n, xyz = noisy_coordinates.shape
        if xyz != 3 or residue_mask.shape != (b, n) or lengths.shape != (b,) or timesteps.shape != (b,):
            raise ValueError("invalid coordinate input shapes")
        expected = torch.arange(n, device=lengths.device)[None] < lengths[:, None]
        if not torch.equal(residue_mask.bool(), expected) or bool((lengths < 2).any()):
            raise ValueError("requires contiguous valid prefix with N>=2")
        if bool(((timesteps < 0) | (timesteps >= self.diffusion_steps)).any()):
            raise ValueError("invalid timestep")
        outputs = [
            F.pad(self._single(noisy_coordinates[z, : int(lengths[z])], timesteps[z]), (0, 0, 0, n - int(lengths[z])))
            for z in range(b)
        ]
        return torch.stack(outputs)
