"""E013 capacity configuration without changing the frozen E013 implementation."""

import math

import torch
from torch import nn

from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import (
    E013UnconditionalEquivariantDenoiser,
    VectorBlock,
    local_geometry,
    mlp,
)

LADDER = {
    "medium": dict(scalar_channels=192, vector_channels=48, sparse_blocks=5, global_blocks=2, local_hidden=192),
    "large": dict(scalar_channels=256, vector_channels=64, sparse_blocks=6, global_blocks=3, local_hidden=256),
    "xl": dict(scalar_channels=320, vector_channels=80, sparse_blocks=6, global_blocks=3, local_hidden=320),
}


class E014CapacityDenoiser(E013UnconditionalEquivariantDenoiser):
    """Only width/depth change; inherit all E013 forward mathematics unchanged."""

    def __init__(
        self,
        scalar_channels=128,
        vector_channels=32,
        sparse_blocks=4,
        global_blocks=2,
        local_hidden=128,
        spatial_k=24,
        coarse_stride=8,
        diffusion_steps=500,
    ):
        super().__init__(scalar_channels, vector_channels, spatial_k, coarse_stride, diffusion_steps)
        if sparse_blocks != 4:
            self.blocks = nn.ModuleList([VectorBlock(scalar_channels, vector_channels) for _ in range(sparse_blocks)])
        if global_blocks != 2:
            self.global_blocks = nn.ModuleList(
                [VectorBlock(scalar_channels, vector_channels) for _ in range(global_blocks)]
            )
        if local_hidden != 128:
            self.local_input = nn.Linear(32, local_hidden)
            self.local_blocks = nn.ModuleList([mlp(2 * local_hidden, local_hidden, local_hidden) for _ in range(2)])
            self.local_head = nn.Linear(local_hidden, 3)

    @torch.no_grad()
    def telemetry(self, noisy_coordinates, timesteps, lengths, residue_mask):
        """Read-only hooks; no extra inference inputs, no training hooks installed."""
        captured = {key: [] for key in ("sparse_v", "coarse_v", "weights", "local_u", "logits")}

        def save(key, output):
            captured[key].append(output.detach())

        hooks = [
            self.blocks[-1].register_forward_hook(lambda _m, _i, out: save("sparse_v", out[1])),
            self.broadcast_v.register_forward_hook(lambda _m, _i, out: save("coarse_v", out)),
            self.head.register_forward_hook(lambda _m, _i, out: save("weights", out)),
            self.local_head.register_forward_hook(lambda _m, _i, out: save("local_u", out)),
            self.gate.register_forward_hook(lambda _m, _i, out: save("logits", out)),
        ]
        try:
            prediction = self(noisy_coordinates, timesteps, lengths, residue_mask)
        finally:
            for hook in hooks:
                hook.remove()
        results = []
        for z, length in enumerate(lengths.tolist()):
            x = noisy_coordinates[z, :length]
            x = x - x.mean(0)
            features, frame, reliability = local_geometry(x)
            segment = torch.arange(length, device=x.device) // self.stride
            v = captured["sparse_v"][z] + captured["coarse_v"][z].transpose(-1, -2)[segment]
            global_v = (captured["weights"][z][..., None] * v).sum(1) / math.sqrt(self.cv)
            gate = captured["logits"][z].sigmoid()[:, 0]
            envelope = (1 - timesteps[z].to(x.dtype) / (self.diffusion_steps - 1)).clamp(0, 1).square()
            local_v = torch.einsum("nac,nc->na", frame, captured["local_u"][z])
            effective = (gate * reliability * envelope)[:, None] * local_v
            # Center both additive components so their sum exactly matches output.
            global_v, effective = global_v - global_v.mean(0), effective - effective.mean(0)
            results.append(
                dict(
                    global_v=global_v,
                    local_effective=effective,
                    gate=gate,
                    reliability=reliability,
                    assessable=features[:, 28].bool(),
                )
            )
        return prediction, results
