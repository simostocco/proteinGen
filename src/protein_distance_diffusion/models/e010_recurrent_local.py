"""Phase 4D v3: capacity-only ladder, shared bounded local recurrence."""

import torch
from torch.utils.checkpoint import checkpoint

from ..training.e010_phase4d_objective_v2 import smooth_bounded_local
from .e010_hybrid_local import LocalGeometryBranch, local_representation

VARIANTS = {"S": (128, 4), "M": (192, 6), "L": (256, 8)}
COUNTS = {"S": 927875, "M": 3115587, "L": 7369987}
K = 4
S_MAX = 0.04


class RecurrentLocalRefiner(LocalGeometryBranch):
    def __init__(self, variant="S", *, activation_checkpoint=True):
        if variant not in VARIANTS:
            raise ValueError("only S/M/L are registered")
        width, blocks = VARIANTS[variant]
        super().__init__(width, blocks)
        self.variant = variant
        self.activation_checkpoint = activation_checkpoint
        if sum(p.numel() for p in self.parameters()) != COUNTS[variant]:
            raise ValueError("capacity count mismatch")

    def step(self, current, mask):
        # Preserve differentiability through all four geometry recomputations.
        rep = local_representation(current, mask)
        valid = rep["eligible"]
        h = self.input(rep["features"]) * valid[..., None]
        for block in self.blocks:
            h = (
                checkpoint(block, h, valid, use_reentrant=False)
                if self.activation_checkpoint and torch.is_grad_enabled()
                else block(h, valid)
            )
        raw = self.head(h) * valid[..., None]
        bounded = smooth_bounded_local(raw, S_MAX)
        # Inward rounding safeguard for finite-precision saturation at s_max.
        limit = torch.nextafter(raw.new_tensor(S_MAX), raw.new_tensor(0.0))
        size = bounded.norm(dim=-1, keepdim=True)
        bounded = bounded * torch.minimum(torch.ones_like(size), limit / size.clamp_min(1e-12))
        delta = torch.einsum("bnij,bnj->bni", rep["frame"], bounded)
        return {**rep, "raw_local": raw, "bounded_local": bounded, "delta": delta, "prediction": current + delta}

    def forward(self, pg, mask):
        current = pg.detach()
        states = [current]
        steps = []
        for _ in range(K):
            out = self.step(current, mask)
            current = out["prediction"]
            states.append(current)
            steps.append(out)
        return {
            "prediction": current,
            "states": states,
            "steps": steps,
            "delta": current - pg.detach(),
            "eligible": steps[0]["eligible"],
            "degenerate": steps[0]["degenerate"],
        }
