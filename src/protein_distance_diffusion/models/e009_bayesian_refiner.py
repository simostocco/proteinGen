"""Bayesian SE(3)-equivariant C-alpha coordinate refinement for E009."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _log_mix(x: torch.Tensor, logits: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    z = (x[..., None] - loc) / scale
    return torch.logsumexp(
        F.log_softmax(logits, -1) - 0.5 * z.square() - torch.log(scale) - 0.5 * math.log(2 * math.pi), -1
    )


class GeometryPrior(nn.Module):
    """Normalized bond Gaussian, bounded logistic-normal angle, and circular von Mises mixtures."""

    def __init__(
        self,
        components: int = 3,
        *,
        bond_components: int | None = None,
        angle_components: int | None = None,
        torsion_components: int | None = None,
    ):
        super().__init__()
        bond_components = bond_components or components
        angle_components = angle_components or components
        torsion_components = torsion_components or components
        self.bond_logits = nn.Parameter(torch.zeros(bond_components))
        self.bond_loc = nn.Parameter(torch.zeros(bond_components))
        self.bond_logscale = nn.Parameter(torch.zeros(bond_components))
        self.angle_logits = nn.Parameter(torch.zeros(angle_components))
        self.angle_loc = nn.Parameter(torch.zeros(angle_components))
        self.angle_logscale = nn.Parameter(torch.zeros(angle_components))
        self.torsion_logits = nn.Parameter(torch.zeros(torsion_components))
        self.torsion_loc = nn.Parameter(torch.zeros(torsion_components))
        self.torsion_logk = nn.Parameter(torch.zeros(torsion_components))

    def log_prob(self, bonds: torch.Tensor, angles: torch.Tensor, torsions: torch.Tensor) -> dict[str, torch.Tensor]:
        bond = _log_mix(bonds, self.bond_logits, self.bond_loc, self.bond_logscale.exp().clamp_min(1e-3))
        eps = 1e-6
        u = (angles / math.pi).clamp(eps, 1 - eps)
        y = bounded_angle_transform(angles, eps)
        angle = (
            _log_mix(y, self.angle_logits, self.angle_loc, self.angle_logscale.exp().clamp_min(1e-3))
            - math.log(math.pi)
            - torch.log(u)
            - torch.log1p(-u)
        )
        k = self.torsion_logk.exp().clamp(1e-4, 100)
        norm = math.log(2 * math.pi) + torch.log(torch.special.i0e(k)) + k
        vm = k * torch.cos(torsions[..., None] - self.torsion_loc) - norm
        torsion = torch.logsumexp(F.log_softmax(self.torsion_logits, -1) + vm, -1)
        return {"bond": bond, "angle": angle, "torsion": torsion}


def bounded_angle_transform(angle: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Map the open physical interval (0, pi) to R without endpoint overflow."""
    if not 0 < epsilon < 0.01:
        raise ValueError("angle transform epsilon must be in (0, 0.01)")
    u = (angle / math.pi).clamp(epsilon, 1.0 - epsilon)
    return torch.log(u) - torch.log1p(-u)


def _log_i0(kappa: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.special.i0e(kappa)) + kappa


def fit_gaussian_mixture_stream(
    chunks, components: int, *, max_iter: int = 300, tolerance: float = 1e-7, variance_floor: float = 1e-5
):
    """Deterministic batch-streaming EM for a univariate Gaussian mixture."""
    import numpy as np

    factory = chunks if callable(chunks) else lambda: iter(chunks)
    first = np.asarray(next(factory()), dtype=np.float64).reshape(-1)
    if first.size == 0:
        raise ValueError("empty fitting observations")

    # A callable chunk source is reread each EM pass without retaining observations.
    def arrays():
        for c in factory():
            a = np.asarray(c, dtype=np.float64).reshape(-1)
            if a.size and not np.isfinite(a).all():
                raise ValueError("non-finite fitting observations")
            if a.size:
                yield a

    total = sum(len(a) for a in arrays())
    if total < components:
        raise ValueError("fewer observations than mixture components")
    sample_parts = []
    stride = max(1, total // 100000)
    offset = 0
    for a in arrays():
        first_index = (-offset) % stride
        sample_parts.append(a[first_index::stride])
        offset += len(a)
    sample = np.concatenate(sample_parts)
    loc = np.quantile(sample, (np.arange(components) + 0.5) / components)
    scale = np.full(components, max(float(np.std(sample)), 0.05))
    weight = np.full(components, 1 / components)
    prev = None
    history = []
    for _step in range(max_iter):
        sw = np.zeros(components)
        sx = sw.copy()
        sx2 = sw.copy()
        ll = 0.0
        for x in arrays():
            lp = (
                np.log(weight)[None, :]
                - 0.5 * ((x[:, None] - loc) / scale) ** 2
                - np.log(scale)
                - 0.5 * np.log(2 * np.pi)
            )
            mx = lp.max(1, keepdims=True)
            logden = mx[:, 0] + np.log(np.exp(lp - mx).sum(1))
            r = np.exp(lp - logden[:, None])
            sw += r.sum(0)
            sx += (r * x[:, None]).sum(0)
            sx2 += (r * x[:, None] ** 2).sum(0)
            ll += logden.sum()
        weight = np.maximum(sw / total, 1e-12)
        weight /= weight.sum()
        loc = sx / sw.clip(1e-12)
        var = np.maximum(sx2 / sw.clip(1e-12) - loc**2, variance_floor)
        scale = np.sqrt(var)
        history.append(float(ll / total))
        if prev is not None and abs(history[-1] - prev) <= tolerance * max(1.0, abs(prev)):
            break
        prev = history[-1]
    order = np.argsort(loc)
    return {
        "weights": weight[order],
        "means": loc[order],
        "scales": scale[order],
        "iterations": _step + 1,
        "converged": _step + 1 < max_iter
        or (len(history) > 1 and abs(history[-1] - history[-2]) <= tolerance * max(1.0, abs(history[-2]))),
        "training_log_likelihood": history[-1],
        "history": history,
    }


def fit_von_mises_mixture_stream(chunks, components: int, *, max_iter: int = 300, tolerance: float = 1e-7):
    """Deterministic EM for circular von Mises mixtures with stable Bessel normalization."""
    import numpy as np

    factory = chunks if callable(chunks) else lambda: iter(chunks)

    def arrays():
        for c in factory():
            a = np.asarray(c, dtype=np.float64).reshape(-1)
            if a.size and not np.isfinite(a).all():
                raise ValueError("non-finite fitting observations")
            if a.size:
                yield (a + np.pi) % (2 * np.pi) - np.pi

    total = sum(len(a) for a in arrays())
    if total < components:
        raise ValueError("fewer observations than mixture components")
    sample_parts = []
    stride = max(1, total // 100000)
    offset = 0
    for a in arrays():
        first_index = (-offset) % stride
        sample_parts.append(a[first_index::stride])
        offset += len(a)
    sample = np.concatenate(sample_parts)
    loc = np.quantile(np.sort(sample), (np.arange(components) + 0.5) / components)
    weight = np.full(components, 1 / components)
    kappa = np.full(components, 1.0)
    prev = None
    history = []
    for _step in range(max_iter):
        sw = np.zeros(components)
        sc = sw.copy()
        ss = sw.copy()
        ll = 0.0
        lognorm = np.log(2 * np.pi) + np.log(np.i0(kappa))
        for x in arrays():
            lp = np.log(weight)[None, :] + kappa[None, :] * np.cos(x[:, None] - loc) - lognorm[None, :]
            mx = lp.max(1, keepdims=True)
            z = mx[:, 0] + np.log(np.exp(lp - mx).sum(1))
            r = np.exp(lp - z[:, None])
            sw += r.sum(0)
            sc += (r * np.cos(x[:, None])).sum(0)
            ss += (r * np.sin(x[:, None])).sum(0)
            ll += z.sum()
        weight = np.maximum(sw / total, 1e-12)
        weight /= weight.sum()
        loc = np.arctan2(ss, sc)
        R = np.sqrt(sc**2 + ss**2) / sw.clip(1e-12)
        kappa = np.empty_like(R)
        low = R < 0.53
        middle = (R >= 0.53) & (R < 0.85)
        high = R >= 0.85
        kappa[low] = 2 * R[low] + R[low] ** 3 + 5 * R[low] ** 5 / 6
        kappa[middle] = -0.4 + 1.39 * R[middle] + 0.43 / (1 - R[middle])
        denom = np.maximum(1e-12, R[high] ** 3 - 4 * R[high] ** 2 + 3 * R[high])
        kappa[high] = 1 / denom
        kappa = np.clip(kappa, 1e-4, 100)
        history.append(float(ll / total))
        if prev is not None and abs(history[-1] - prev) <= tolerance * max(1.0, abs(prev)):
            break
        prev = history[-1]
    order = np.argsort((loc + np.pi) % (2 * np.pi))
    return {
        "weights": weight[order],
        "means": loc[order],
        "concentrations": kappa[order],
        "iterations": _step + 1,
        "converged": _step + 1 < max_iter
        or (len(history) > 1 and abs(history[-1] - history[-2]) <= tolerance * max(1.0, abs(history[-2]))),
        "training_log_likelihood": history[-1],
        "history": history,
    }


class _EquivariantBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.edge = nn.Sequential(
            nn.Linear(2 * width + 20, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width + 1)
        )
        self.node = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, h, x, mask, idx, sep):
        # idx [B,N,K], ordered neighbor indices; all coordinate outputs are sums of relative vectors.
        b, n, k = idx.shape
        batch = torch.arange(b, device=x.device)[:, None, None]
        xj = x[batch, idx]
        hj = h[batch, idx]
        delta = xj - x[:, :, None, :]
        d = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
        rbf = torch.exp(-((d[..., None] - torch.linspace(0, 32, 16, device=x.device)) / 2).square())
        sf = torch.stack((sep.float() / 500, torch.log1p(sep.float()) / 7, (sep == 1).float(), (sep == 2).float()), -1)
        ef = torch.cat((h[:, :, None, :].expand_as(hj), hj, rbf, sf), -1)
        out = self.edge(ef)
        valid = mask[:, :, None] * mask[batch, idx]
        msg = out[..., :-1] * valid[..., None]
        agg = msg.sum(2) / valid.sum(2, keepdim=True).clamp_min(1)
        h = self.norm(h + self.node(torch.cat((h, agg), -1))) * mask[..., None]
        coeff = torch.tanh(out[..., -1]) * valid / (valid.sum(2, keepdim=True).clamp_min(1))
        vec = (coeff[..., None] * delta / d[..., None]).sum(2)
        return h, vec


class BayesianSE3Refiner(nn.Module):
    """Seven-block invariant-message network with equivariant mean and invariant scales."""

    def __init__(self, width: int = 128, layers: int = 7, max_length: int = 500, sigma_bounds=(0.03, 3.0)):
        super().__init__()
        if width < 32 or not 6 <= layers <= 8 or max_length < 3 or not 0 < sigma_bounds[0] < sigma_bounds[1]:
            raise ValueError("invalid E009 model configuration")
        self.max_length = max_length
        self.sigma_bounds = sigma_bounds
        self.input = nn.Linear(4, width)
        self.blocks = nn.ModuleList(_EquivariantBlock(width) for _ in range(layers))
        self.vector_gate = nn.Parameter(torch.zeros(layers))
        self.mean = nn.Linear(width, 1)
        self.scale = nn.Linear(width, 1)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.scale.weight)
        nn.init.constant_(self.scale.bias, -2.0)

    def _neighbors(self, x, mask):
        b, n, _ = x.shape
        device = x.device
        ix = torch.arange(n, device=device)
        dist = torch.cdist(x.float(), x.float())
        dist = dist.masked_fill(~(mask[:, None, :].bool()), 1e6)
        k = min(24, n)
        kn = dist.topk(k, dim=-1, largest=False).indices
        offsets = torch.stack([(ix + o).clamp(0, n - 1) for o in (-3, -2, -1, 1, 2, 3)], -1)
        idx = torch.cat((offsets[None].expand(b, -1, -1), kn), -1)
        valid = mask[:, None, :].expand(-1, n, -1).gather(2, idx)
        return idx, valid

    def forward(self, coarse, mask, chain_offset=None):
        if coarse.ndim != 3 or coarse.shape[-1] != 3 or mask.shape != coarse.shape[:2]:
            raise ValueError("coarse/mask shapes must be [B,N,3]/[B,N]")
        b, n, _ = coarse.shape
        if n > self.max_length:
            raise ValueError("sequence exceeds configured max_length")
        if chain_offset is not None and chain_offset.shape != mask.shape:
            raise ValueError("chain_offset must have shape [B,N]")
        if not torch.isfinite(coarse[mask.bool()]).all():
            raise ValueError("valid coarse coordinates must be finite")
        m = mask.to(coarse.dtype)
        lengths = m.sum(1).clamp_min(1)
        safe_coarse = torch.where(mask.bool()[..., None], coarse, torch.zeros_like(coarse))
        center = (safe_coarse * m[..., None]).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)[..., None]
        x = safe_coarse - center
        ix = torch.arange(n, device=x.device)
        local = torch.stack(
            (
                ix.float()[None, :].expand(b, -1) / (lengths - 1).clamp_min(1)[:, None],
                torch.log(lengths.clamp_min(2))[:, None].expand(-1, n) / math.log(self.max_length),
                m,
                torch.zeros((b, n), device=x.device) if chain_offset is None else chain_offset.to(x.dtype),
            ),
            -1,
        )
        idx, _ = self._neighbors(x, m)
        sep = (idx - ix[None, :, None]).abs()
        h = self.input(local) * m[..., None]
        corr = torch.zeros_like(x)
        for block, g in zip(self.blocks, self.vector_gate, strict=True):
            h, v = block(h, x, m, idx, sep)
            corr = corr + torch.sigmoid(g) * v
        # Center correction so global translation behavior remains explicit.
        corr = corr - (corr * m[..., None]).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)[..., None]
        mu = x + corr * self.mean(h) + center
        lo, hi = self.sigma_bounds
        sigma = lo + (hi - lo) * torch.sigmoid(self.scale(h))
        log_sigma = sigma.log()
        return {
            "mean": mu * m[..., None],
            "coarse": safe_coarse,
            "log_sigma": log_sigma.squeeze(-1),
            "sigma": sigma.squeeze(-1),
        }

    @staticmethod
    def sample(posterior, noise=None):
        eps = torch.randn_like(posterior["mean"]) if noise is None else noise
        return posterior["mean"] + posterior["sigma"][..., None] * eps


def normalized_refiner_losses(posterior, target, prior: GeometryPrior, mask, weights=None):
    """Per-residue normalized objective; each named term is separately reported."""
    w = {
        "coordinate_nll": 1.0,
        "geometry_nll": 1.0,
        "long_range_pair_nll": 0.2,
        "contact_nll": 0.2,
        "radius_of_gyration_nll": 0.2,
        "chirality_nll": 0.1,
        "posterior_kl": 0.01,
        **(weights or {}),
    }
    m = mask.to(target.dtype)
    target_clean = torch.where(mask.bool()[..., None], target, torch.zeros_like(target))
    count = m.sum().clamp_min(1)
    sample = BayesianSE3Refiner.sample(posterior)
    err = (target_clean - posterior["mean"]) / posterior["sigma"][..., None]
    coord = (
        (0.5 * err.square() + posterior["log_sigma"][..., None] + 0.5 * math.log(2 * math.pi)) * m[..., None]
    ).sum() / (3 * count)
    # Analytic KL to an isotropic unit prior centered at the coarse input.
    delta = posterior["mean"] - posterior["coarse"]
    kl = (0.5 * (3 * posterior["sigma"].square() + delta.square().sum(-1)) - 3 * posterior["log_sigma"] - 1.5) * m

    def geom_terms(x):
        bond = torch.linalg.vector_norm(x[:, 1:] - x[:, :-1], dim=-1).clamp_min(1e-6)
        u = F.normalize(x[:, :-2] - x[:, 1:-1], dim=-1)
        v = F.normalize(x[:, 2:] - x[:, 1:-1], dim=-1)
        ang = torch.atan2(torch.linalg.vector_norm(torch.cross(u, v, dim=-1), dim=-1), (u * v).sum(-1))
        a = x[:, 1:-2] - x[:, :-3]
        b = x[:, 2:-1] - x[:, 1:-2]
        c = x[:, 3:] - x[:, 2:-1]
        b = F.normalize(b, dim=-1)
        aa = a - (a * b).sum(-1, keepdim=True) * b
        cc = c - (c * b).sum(-1, keepdim=True) * b
        tor = torch.atan2((torch.cross(b, aa, dim=-1) * cc).sum(-1), (aa * cc).sum(-1))
        lp = prior.log_prob(bond, ang, tor)
        bm = (mask[:, 1:] * mask[:, :-1]).to(x.dtype)
        am = (mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]).to(x.dtype)
        tm = (mask[:, 3:] * mask[:, 2:-1] * mask[:, 1:-2] * mask[:, :-3]).to(x.dtype)
        means = [
            (lp["bond"] * bm).sum() / bm.sum().clamp_min(1),
            (lp["angle"] * am).sum() / am.sum().clamp_min(1),
            (lp["torsion"] * tm).sum() / tm.sum().clamp_min(1),
        ]
        return [-z for z in means]

    bond_nll, angle_nll, torsion_nll = geom_terms(sample)
    geometry = (bond_nll + angle_nll + torsion_nll) / 3
    # Pair-distance Gaussian likelihood and binary contact likelihood against target.
    pd = torch.cdist(sample.float(), sample.float())
    td = torch.cdist(target_clean.float(), target_clean.float())
    n = target.shape[1]
    ix = torch.arange(n, device=target.device)
    sep = (ix[:, None] - ix[None, :]).abs()
    pairmask = mask[:, :, None] * mask[:, None, :]
    longmask = pairmask * (sep[None] >= 8)
    long_n = longmask.sum().clamp_min(1)
    long_pair = (((pd - td).square() / (2 * 2.0**2) + math.log(2.0 * math.sqrt(2 * math.pi))) * longmask).sum() / long_n
    contact_target = (td < 8.0).to(target.dtype)
    logits = (8.0 - pd) / 1.0
    contact = F.binary_cross_entropy_with_logits(logits, contact_target, reduction="none")
    contact = (contact * longmask).sum() / long_n

    def rg(x):
        x = torch.where(mask.bool()[..., None], x, torch.zeros_like(x))
        cen = (x * m[..., None]).sum(1, keepdim=True) / count
        return torch.sqrt((((x - cen).square().sum(-1)) * m).sum(1) / count)

    rg_loss = (
        ((rg(sample) - rg(target_clean)).square() / (2 * 2.0**2)) + math.log(2.0 * math.sqrt(2 * math.pi))
    ).mean()

    # Signed scalar triple products encode local handedness and are reflection-sensitive.
    def chirality(x):
        v1 = x[:, 1:-2] - x[:, :-3]
        v2 = x[:, 2:-1] - x[:, 1:-2]
        v3 = x[:, 3:] - x[:, 2:-1]
        return (torch.cross(v1, v2, dim=-1) * v3).sum(-1)

    ct = chirality(target_clean)
    cp = chirality(sample)
    cm = mask[:, 3:] * mask[:, :-3]
    chirality_loss = (F.softplus(-(ct * cp).sign() * cp.abs().clamp(max=20)) * cm).sum() / cm.sum().clamp_min(1)
    terms = {
        "coordinate_nll": coord,
        "geometry_nll": geometry,
        "bond_prior_nll": bond_nll,
        "angle_prior_nll": angle_nll,
        "torsion_prior_nll": torsion_nll,
        "long_range_pair_nll": long_pair,
        "contact_nll": contact,
        "radius_of_gyration_nll": rg_loss,
        "chirality_nll": chirality_loss,
        "posterior_kl": (kl.sum() / count),
    }
    total = sum(w[name] * terms[name] for name in w)
    return {"total": total, **terms}
