"""Small invariant C-alpha repair decoder with exact virtual-bond lengths."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _unit(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)


def cartesian_to_internal(coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract bond lengths, interior C-alpha angles, and signed torsions."""
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3 or coordinates.shape[0] < 3:
        raise ValueError("coordinates must have shape [N,3], N >= 3")
    p = coordinates
    bonds = torch.linalg.vector_norm(p[1:] - p[:-1], dim=-1)
    left = _unit(p[:-2] - p[1:-1])
    right = _unit(p[2:] - p[1:-1])
    angles = torch.atan2(
        torch.linalg.vector_norm(torch.linalg.cross(left, right, dim=-1), dim=-1), (left * right).sum(-1)
    )
    if p.shape[0] < 4:
        return bonds, angles, p.new_empty((0,))
    b0 = -(p[1:-2] - p[:-3])
    b1 = _unit(p[2:-1] - p[1:-2])
    b2 = p[3:] - p[2:-1]
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    torsions = torch.atan2((torch.linalg.cross(b1, v, dim=-1) * w).sum(-1), (v * w).sum(-1))
    return bonds, angles, torsions


def internal_to_cartesian(
    seed: torch.Tensor,
    angles: torch.Tensor,
    torsions: torch.Tensor,
    *,
    bond_length: float = 3.8,
    bond_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct a chain from its first three points and internal coordinates.

    ``angles[i]`` is the virtual angle at residue i+1. ``torsions[i]`` uses
    the signed convention returned by :func:`cartesian_to_internal`.
    """
    if seed.ndim != 2 or seed.shape[0] != 3 or seed.shape[-1] != 3:
        raise ValueError("seed must have shape [3,3]")
    n = angles.numel() + 2
    if torsions.numel() != max(n - 3, 0):
        raise ValueError("torsion count must equal N-3")
    lengths = (
        torch.full((n - 1,), float(bond_length), dtype=seed.dtype, device=seed.device)
        if bond_lengths is None
        else bond_lengths
    )
    if lengths.shape != (n - 1,):
        raise ValueError("bond_lengths must have shape [N-1]")
    p0 = seed[0]
    first_direction = _unit(seed[1] - seed[0])
    second_direction = _unit(seed[2] - seed[1])
    p1 = p0 + lengths[0] * first_direction
    p2 = p1 + lengths[1] * second_direction
    out = [p0, p1, p2]
    for i in range(3, n):
        a, b, c = out[-3:]
        e1 = _unit(c - b)
        normal = _unit(torch.linalg.cross(_unit(a - b), e1, dim=-1))
        e2 = torch.linalg.cross(normal, e1, dim=-1)
        theta = angles[i - 2]
        # The +pi converts the signed-dihedral convention into the NeRF frame.
        phi = torsions[i - 3] + math.pi
        direction = -torch.cos(theta) * e1 + torch.sin(theta) * (torch.cos(phi) * e2 + torch.sin(phi) * normal)
        out.append(c + lengths[i - 1] * direction)
    return torch.stack(out)


def _pair_features(coords: torch.Tensor, mask: torch.Tensor, bins: int) -> torch.Tensor:
    """Per-residue distance-RBF summaries over four tertiary sequence bands."""
    n = coords.shape[0]
    distances = torch.cdist(coords.float()[None], coords.float()[None])[0]
    centers = torch.linspace(0.0, 32.0, bins, device=coords.device, dtype=distances.dtype)
    width = 32.0 / max(bins - 1, 1)
    radial = torch.exp(-((distances[..., None] - centers) / width).square())
    idx = torch.arange(n, device=coords.device)
    sep = (idx[:, None] - idx[None, :]).abs()
    bands = ((sep >= 4) & (sep <= 8), (sep >= 9) & (sep <= 16), (sep >= 17) & (sep <= 32), sep >= 33)
    valid_pairs = mask[:, None] & mask[None, :]
    features = []
    for band in bands:
        selected = band & valid_pairs
        pooled = (radial * selected[..., None]).sum(1) / selected.sum(1).clamp_min(1)[:, None]
        features.append(pooled)
    return torch.cat(features, dim=-1)


def _invariant_features(coords: torch.Tensor, mask: torch.Tensor, bins: int) -> torch.Tensor:
    bonds, angles, torsions = cartesian_to_internal(coords)
    n = coords.shape[0]
    # Place angle/torsion descriptors at the residue whose geometry they affect.
    local = coords.new_zeros((n, 11))
    local[1:-1, 0] = torch.cos(angles)
    local[1:-1, 1] = torch.sin(angles)
    if torsions.numel():
        local[2:-1, 2] = torch.cos(torsions)
        local[2:-1, 3] = torch.sin(torsions)
    for k in range(1, 5):
        if n > k:
            d = torch.linalg.vector_norm(coords[k:] - coords[:-k], dim=-1)
            local[k:, 3 + k] = d / 10.0
    # Signed scalar triple products are invariant under SE(3) and retain handedness.
    if n > 3:
        v1, v2, v3 = coords[1:-2] - coords[:-3], coords[2:-1] - coords[1:-2], coords[3:] - coords[2:-1]
        triple = (torch.linalg.cross(v1, v2, dim=-1) * v3).sum(-1)
        local[1:-2, 8] = triple / (
            torch.linalg.vector_norm(v1, dim=-1)
            * torch.linalg.vector_norm(v2, dim=-1)
            * torch.linalg.vector_norm(v3, dim=-1)
        ).clamp_min(1e-6)
    local[:, 9] = torch.linspace(0, 1, n, device=coords.device, dtype=coords.dtype)
    local[:, 10] = math.log(max(n, 1)) / math.log(500.0)
    return torch.cat((local, _pair_features(coords, mask, bins).to(coords.dtype)), dim=-1)


class _Residual1d(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(), nn.Conv1d(width, width, 5, padding=2)
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + y) * mask[:, :, None]


class BackboneKinematicDecoder(nn.Module):
    """Map a coarse centered C-alpha trace to a fixed-bond-length trace.

    All learned inputs are SE(3)-invariant scalars. The reconstruction uses
    differentiable forward kinematics; adjacent geometry is exact by design.
    """

    def __init__(
        self,
        *,
        width: int = 64,
        layers: int = 4,
        pair_rbf_bins: int = 8,
        bond_length_angstrom: float = 3.8,
        max_length: int = 500,
    ) -> None:
        super().__init__()
        if width < 8 or layers < 1 or pair_rbf_bins < 4 or not 3.6 <= bond_length_angstrom <= 4.0:
            raise ValueError("invalid E008 decoder dimensions or bond length")
        self.width, self.pair_rbf_bins = width, pair_rbf_bins
        self.bond_length_angstrom, self.max_length = float(bond_length_angstrom), max_length
        input_width = 11 + 4 * pair_rbf_bins
        self.input = nn.Linear(input_width, width)
        self.blocks = nn.ModuleList(_Residual1d(width) for _ in range(layers))
        self.output = nn.Linear(width, 4)
        # Start as identity in internal-coordinate space.
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.copy_(torch.tensor([1.0, 0.0, 1.0, 0.0]))

    def forward(self, coarse: torch.Tensor, residue_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if coarse.ndim != 3 or coarse.shape[-1] != 3 or residue_mask.shape != coarse.shape[:2]:
            raise ValueError("coarse/mask must have shapes [B,N,3] and [B,N]")
        batch, side, _ = coarse.shape
        if side > self.max_length:
            raise ValueError("sequence exceeds configured maximum length")
        outputs, deltas = [], []
        for b in range(batch):
            n = int(residue_mask[b].sum().item())
            if n < 4 or not bool(residue_mask[b, :n].all()) or bool(residue_mask[b, n:].any()):
                raise ValueError("mask must be a contiguous true prefix of length >= 4")
            source = coarse[b, :n]
            source = source - source.mean(0, keepdim=True)
            _, angle, torsion = cartesian_to_internal(source)
            feat = _invariant_features(source, residue_mask[b, :n].bool(), self.pair_rbf_bins)[None]
            m = residue_mask[b : b + 1, :n].to(feat.dtype)
            h = self.input(feat) * m[:, :, None]
            for block in self.blocks:
                h = block(h, m)
            raw = self.output(h[0])
            angle_pair = F.normalize(raw[1:-1, :2], dim=-1, eps=1e-6)
            torsion_pair = F.normalize(raw[2:-1, 2:], dim=-1, eps=1e-6)
            delta_angle = torch.atan2(angle_pair[:, 1], angle_pair[:, 0])
            delta_torsion = torch.atan2(torsion_pair[:, 1], torsion_pair[:, 0])
            # The initialized (1,0) pairs represent zero angle corrections.
            theta = (angle + delta_angle).clamp(1e-4, math.pi - 1e-4)
            phi = torch.atan2(torch.sin(torsion + delta_torsion), torch.cos(torsion + delta_torsion))
            rebuilt = internal_to_cartesian(source[:3], theta, phi, bond_length=self.bond_length_angstrom)
            rebuilt = rebuilt - rebuilt.mean(0, keepdim=True)
            # Rigid alignment preserves exact internal geometry and centers on coarse.
            rebuilt = _kabsch_align(rebuilt, source)
            outputs.append(F.pad(rebuilt, (0, 0, 0, side - n)))
            angle_padded = torch.cat((delta_angle.new_zeros(1), delta_angle, delta_angle.new_zeros(side - n + 1)))
            torsion_padded = torch.cat(
                (delta_torsion.new_zeros(2), delta_torsion, delta_torsion.new_zeros(side - n + 1))
            )
            deltas.append((angle_padded, torsion_padded))
        return {
            "coordinates": torch.stack(outputs),
            "angle_correction": torch.stack([x[0] for x in deltas]),
            "torsion_correction": torch.stack([x[1] for x in deltas]),
        }


def _kabsch_align(moving: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable proper-rotation alignment; both traces must be centered."""
    covariance = moving.transpose(0, 1) @ target
    u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
    sign = torch.where(torch.linalg.det(u @ vh) < 0, -1.0, 1.0).to(moving.dtype)
    correction = torch.diag(torch.stack((moving.new_tensor(1), moving.new_tensor(1), sign)))
    rotation = u @ correction @ vh
    return moving @ rotation


def decoder_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def geometry_native_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    coarse: torch.Tensor,
    residue_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Internal and global reconstruction objectives; no adjacent-distance loss."""
    components: dict[str, list[torch.Tensor]] = {
        k: []
        for k in (
            "internal",
            "kabsch_coordinate",
            "i_plus_2",
            "i_plus_3",
            "long_range_pair",
            "contact_map",
            "chirality",
            "radius_of_gyration",
        )
    }
    for b in range(prediction.shape[0]):
        n = int(residue_mask[b].sum().item())
        pred, truth, _source = prediction[b, :n], target[b, :n], coarse[b, :n]
        _, pa, pt = cartesian_to_internal(pred)
        _, ta, tt = cartesian_to_internal(truth)
        components["internal"].append((1 - torch.cos(pa - ta)).mean() + (1 - torch.cos(pt - tt)).mean())
        aligned = _kabsch_align(pred - pred.mean(0), truth - truth.mean(0))
        components["kabsch_coordinate"].append(
            torch.linalg.vector_norm(aligned - (truth - truth.mean(0)), dim=-1).mean()
        )
        for offset, key in ((2, "i_plus_2"), (3, "i_plus_3")):
            pd = torch.linalg.vector_norm(pred[offset:] - pred[:-offset], dim=-1)
            td = torch.linalg.vector_norm(truth[offset:] - truth[:-offset], dim=-1)
            components[key].append(F.smooth_l1_loss(pd, td))
        sep = torch.arange(n, device=pred.device)
        pair = (sep[:, None] - sep[None, :]).abs() >= 8
        pair = torch.triu(pair, diagonal=1)
        pd = torch.cdist(pred.float()[None], pred.float()[None])[0]
        td = torch.cdist(truth.float()[None], truth.float()[None])[0]
        components["long_range_pair"].append(F.smooth_l1_loss(pd[pair], td[pair]))
        pcontact = torch.sigmoid((8.0 - pd) / 0.5)
        tcontact = (td < 8.0).to(pcontact.dtype)
        components["contact_map"].append(F.binary_cross_entropy(pcontact[pair].clamp(1e-6, 1 - 1e-6), tcontact[pair]))
        pv1, pv2, pv3 = pred[1:-2] - pred[:-3], pred[2:-1] - pred[1:-2], pred[3:] - pred[2:-1]
        tv1, tv2, tv3 = truth[1:-2] - truth[:-3], truth[2:-1] - truth[1:-2], truth[3:] - truth[2:-1]
        pchir = (torch.linalg.cross(pv1, pv2, dim=-1) * pv3).sum(-1) / (
            torch.linalg.vector_norm(pv1, dim=-1)
            * torch.linalg.vector_norm(pv2, dim=-1)
            * torch.linalg.vector_norm(pv3, dim=-1)
        ).clamp_min(1e-6)
        tchir = (torch.linalg.cross(tv1, tv2, dim=-1) * tv3).sum(-1) / (
            torch.linalg.vector_norm(tv1, dim=-1)
            * torch.linalg.vector_norm(tv2, dim=-1)
            * torch.linalg.vector_norm(tv3, dim=-1)
        ).clamp_min(1e-6)
        components["chirality"].append(F.relu(0.05 - torch.sign(tchir) * pchir).mean())
        prg = torch.linalg.vector_norm(pred - pred.mean(0), dim=-1).square().mean().sqrt()
        trg = torch.linalg.vector_norm(truth - truth.mean(0), dim=-1).square().mean().sqrt()
        components["radius_of_gyration"].append(F.smooth_l1_loss(prg, trg))
    return {name: torch.stack(values).mean() for name, values in components.items()}
