"""Phase 4D: frozen E010 with an SO(3)-equivariant, parity-sensitive local branch."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from .e010_global_equivariant import GlobalEquivariantResidual

SOURCE_SHA256 = "f5211cbc1be5092175ce15b9761a4efba761d242310287cd6b1e04df6a6744ef"
OFFSETS = (-3, -2, -1, 1, 2, 3)
FEATURE_DIM = 41
TORSION_SIN_INDEX = 34
TORSION_COS_INDEX = 35
PSEUDOSCALAR_INDEX = 36
EPS = 1e-6  # Å for lengths; dimensionless for normalized angular assessability


def neighbor(tensor, offset):
    """Sequence shift with zero outside the tensor; never wrap endpoints."""
    out = torch.zeros_like(tensor)
    n = tensor.shape[1]
    if abs(offset) < n:
        if offset > 0:
            out[:, :-offset] = tensor[:, offset:]
        else:
            out[:, -offset:] = tensor[:, :offset]
    return out


def local_representation(coords, mask, eps=EPS):
    """Columns e1/e2/e3; reflection sends frame to Q F diag(1,1,-1).

    Central contiguous triple required. Both bonds >eps Å and sin(bond turn)
    >eps. No fallback axes. Feature layout: distances[6], availability[6],
    local xyz[18], cos(b_prev,b_next), position/N/C metadata[3], sin/cos torsion,
    normalized triple product, torsion availability, frame availability,
    bond-angle availability, triple-product availability (41 total).
    Torsion uses bonds (i-1→i, i→i+1, i+1→i+2), atan2 convention
    sin=dot(cross(n1,n2),unit(b)), cos=dot(n1,n2), n1=a×b, n2=b×c.
    """
    if coords.ndim != 3 or coords.shape[-1] != 3 or mask.shape != coords.shape[:2]:
        raise ValueError("expected coords[B,N,3], mask[B,N]")
    mask = mask.bool()
    x = torch.where(mask[..., None], coords, 0)
    prev, nxt = neighbor(x, -1), neighbor(x, 1)
    a, b = x - prev, nxt - x
    an, bn = a.norm(dim=-1), b.norm(dim=-1)
    triple = mask & neighbor(mask, -1) & neighbor(mask, 1)
    e1 = b / bn.clamp_min(eps)[..., None]
    v = a - (a * e1).sum(-1, keepdim=True) * e1
    vn = v.norm(dim=-1)
    eligible = triple & (an > eps) & (bn > eps) & (vn / an.clamp_min(eps) > eps)
    e2 = v / vn.clamp_min(eps * eps)[..., None]
    frame = torch.stack((e1, e2, torch.cross(e1, e2, dim=-1)), -1)
    frame = torch.where(eligible[..., None, None], frame, 0)
    distances, available, relative = [], [], []
    for k in OFFSETS:
        ok = mask & neighbor(mask, k)
        d = neighbor(x, k) - x
        distances.append(torch.where(ok, d.norm(dim=-1), 0))
        available.append(ok.to(x.dtype))
        relative.append(torch.where(ok[..., None], torch.einsum("bnji,bnj->bni", frame, d), 0))
    angle_ok = triple & (an > eps) & (bn > eps)
    angle = torch.where(angle_ok, (a * b).sum(-1) / (an * bn).clamp_min(eps**2), 0)
    c = neighbor(x, 2) - nxt
    cn = c.norm(dim=-1)
    n1, n2 = torch.cross(a, b, dim=-1), torch.cross(b, c, dim=-1)
    n1n, n2n = n1.norm(dim=-1), n2.norm(dim=-1)
    four = triple & neighbor(mask, 2)
    torsion_ok = (
        four
        & (an > eps)
        & (bn > eps)
        & (cn > eps)
        & (n1n / (an * bn).clamp_min(eps**2) > eps)
        & (n2n / (bn * cn).clamp_min(eps**2) > eps)
    )
    u1, u2 = n1 / n1n.clamp_min(eps**2)[..., None], n2 / n2n.clamp_min(eps**2)[..., None]
    sine = (torch.cross(u1, u2, dim=-1) * e1).sum(-1)
    cosine = (u1 * u2).sum(-1)
    pseu_ok = four & (an > eps) & (bn > eps) & (cn > eps)
    pseu = (n1 * c).sum(-1) / (an * bn * cn).clamp_min(eps**3)
    pos = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[None].expand(mask.shape)
    # Require right padding: holes cannot silently alter terminal metadata.
    if torch.any(mask & (~mask).cumsum(1).bool()):
        raise ValueError("v1 requires right-padded contiguous chains")
    lengths = mask.sum(1, keepdim=True)
    metadata = torch.stack((pos / (lengths - 1).clamp_min(1), pos / 500, (lengths - 1 - pos).clamp_min(0) / 500), -1)
    signed = torch.stack(
        (
            torch.where(torsion_ok, sine, 0),
            torch.where(torsion_ok, cosine, 0),
            torch.where(pseu_ok, pseu, 0),
            torsion_ok,
            eligible,
            angle_ok,
            pseu_ok,
        ),
        -1,
    ).to(x.dtype)
    features = torch.cat(
        (
            torch.stack(distances, -1),
            torch.stack(available, -1),
            torch.cat(relative, -1),
            angle[..., None],
            metadata,
            signed,
        ),
        -1,
    )
    features = torch.where(eligible[..., None], features, 0)
    return {
        "features": features,
        "frame": frame,
        "eligible": eligible,
        "interior": triple,
        "degenerate": triple & ~eligible,
        "torsion_assessable": torsion_ok & eligible,
        "pseudoscalar_assessable": pseu_ok & eligible,
    }


class LocalBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.messages = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in OFFSETS)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(), nn.Linear(4 * width, width))
        self.out_norm = nn.LayerNorm(width)

    def forward(self, h, valid):
        count = sum(neighbor(valid, k).to(h.dtype) for k in OFFSETS).clamp_min(1)
        msg = sum(layer(neighbor(h, k)) for k, layer in zip(OFFSETS, self.messages, strict=True)) / count[..., None]
        h = self.norm(h + msg) * valid[..., None]
        return self.out_norm(h + self.ff(h)) * valid[..., None]


class LocalGeometryBranch(nn.Module):
    def __init__(self, width=128, blocks=4):
        super().__init__()
        self.input = nn.Linear(FEATURE_DIM, width)
        self.blocks = nn.ModuleList(LocalBlock(width) for _ in range(blocks))
        self.head = nn.Linear(width, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, pg, mask):
        rep = local_representation(pg.detach(), mask)
        valid = rep["eligible"]
        h = self.input(rep["features"]) * valid[..., None]
        for block in self.blocks:
            h = block(h, valid)
        u = self.head(h) * valid[..., None]
        delta = torch.einsum("bnij,bnj->bni", rep["frame"], u)
        return {**rep, "local_vector": u, "delta": delta, "prediction": pg.detach() + delta}


class FrozenGlobalLocal(nn.Module):
    def __init__(self, global_model, local=None):
        super().__init__()
        self.global_model = global_model.requires_grad_(False).eval()
        self.local = local if local is not None else LocalGeometryBranch()

    def train(self, mode=True):
        super().train(mode)
        self.global_model.eval()
        return self

    def forward(self, coords, mask):
        # Historical E010 rejects empty proteins: bypass only those rows.
        active = mask.any(1)
        pg = coords.detach().clone()
        with torch.no_grad():
            if active.any():
                pg[active] = self.global_model(coords[active], mask[active])["prediction"]
        return {**self.local(pg, mask), "global_prediction": pg.detach()}


def load_frozen_hybrid(path):
    """CPU only; exact Phase 4B bytes required before deserializing trusted state."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != SOURCE_SHA256:
        raise ValueError("Phase 4B checkpoint SHA256 mismatch")
    model = GlobalEquivariantResidual(width=416, layers=6, heads=8, vector_channels=64)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    if sum(p.numel() for p in model.parameters()) != 12844352:
        raise ValueError("E010 Large topology mismatch")
    hybrid = FrozenGlobalLocal(model)
    hybrid.source_metadata = {k: v for k, v in payload.items() if isinstance(v, (int, str, float))}
    return hybrid
