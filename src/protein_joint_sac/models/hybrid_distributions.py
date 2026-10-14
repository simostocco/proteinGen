"""Exp/sigmoid/tanh Gaussian reference; periodic seam is explicit, not hidden."""

import math
import torch
from torch.nn import functional as F
from torch.distributions import Normal


class GeometryDistribution:
    def __init__(self, mu, log_std, mask):
        if mu.shape != (3,) or log_std.shape != (3,):
            raise ValueError("geometry distribution shape")
        self.mu = mu
        self.log_std = log_std.clamp(-5, 2)
        self.base = Normal(mu, self.log_std.exp())
        self.mask = torch.tensor(mask, dtype=torch.bool, device=mu.device)

    def transform(self, u):
        g = torch.stack([u[0].exp(), math.pi * torch.sigmoid(u[1]), math.pi * torch.tanh(u[2])])
        return torch.where(self.mask, g, torch.zeros_like(g))

    def log_jacobian(self, u):
        return torch.stack(
            [
                u[0],
                math.log(math.pi) - F.softplus(-u[1]) - F.softplus(u[1]),
                math.log(math.pi) + 2 * (math.log(2) - u[2] - F.softplus(-2 * u[2])),
            ]
        )

    def rsample(self):
        u = self.base.rsample()
        g = self.transform(u)
        logp = torch.where(self.mask, self.base.log_prob(u) - self.log_jacobian(u), torch.zeros_like(u)).sum()
        return g, logp

    def log_prob(self, g):
        if g.shape != (3,):
            raise ValueError("action geometry shape")
        if not self.mask.any():
            return self.mu.sum() * 0
        valid = torch.stack([g[0] > 0, (g[1] > 0) & (g[1] < math.pi), (g[2] > -math.pi) & (g[2] < math.pi)])
        if not bool((valid | ~self.mask).all()):
            return self.mu.sum() * 0 + float("-inf")
        d = torch.where(self.mask[0], g[0], g.new_tensor(1.0))
        p = torch.where(self.mask[1], g[1] / math.pi, g.new_tensor(0.5))
        v = torch.where(self.mask[2], g[2] / math.pi, g.new_tensor(0.0))
        u = torch.stack([d.log(), torch.logit(p), torch.atanh(v)])
        return torch.where(self.mask, self.base.log_prob(u) - self.log_jacobian(u), torch.zeros_like(u)).sum()

    def wrapped_log_prob(self, g):
        # Principal angle seam: density tends to zero at +/-pi. This is a known baseline artifact.
        if self.mask[2]:
            g = torch.stack([g[0], g[1], torch.remainder(g[2] + math.pi, 2 * math.pi) - math.pi])
        return self.log_prob(g)
