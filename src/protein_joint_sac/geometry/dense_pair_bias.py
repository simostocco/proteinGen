"""Smooth all-present-prefix pairs; hard contacts do not drive Q gradients."""

import torch
from torch import nn


class DensePairBias(nn.Module):
    def __init__(self, heads, chemistry=True):
        super().__init__()
        self.chemistry = chemistry
        self.register_buffer("rbf_centers", torch.linspace(0, 20, 8))
        self.net = nn.Sequential(nn.Linear(25, 32), nn.SiLU(), nn.Linear(32, heads))

    def forward(self, f):
        d = f["pair_distance_A"]
        t = len(d)
        dtype = d.dtype
        rbf = torch.exp(-((d[..., None] - self.rbf_centers.to(dtype)) / 3.0).square())
        i = torch.arange(t, device=d.device)
        sep = ((i[:, None] - i[None, :]) / 500.0)[..., None]
        ori = f["pair_local_direction"]
        rot = f["pair_relative_rotation"].reshape(t, t, 9)
        chemical = (
            (f["chemistry"] @ f["chemistry"].T / 100.0)[..., None] if self.chemistry else torch.zeros_like(d[..., None])
        )
        masks = torch.stack(
            [f["pair_valid"], f["frame_valid"][:, None].expand(t, t), f["frame_valid"][None, :].expand(t, t)], -1
        ).to(dtype)
        value = torch.cat([rbf, sep, ori, rot, chemical, masks], -1)
        return self.net(value).permute(2, 0, 1) * f["pair_valid"]
