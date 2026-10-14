"""Full bidirectional prefix attention with dense learned additive bias and query token."""

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .rope import apply_rope
from ..geometry.local_frames import features
from ..geometry.dense_pair_bias import DensePairBias
from ..contracts import BuildState


@dataclass(frozen=True)
class ModelConfig:
    width: int = 32
    heads: int = 4
    layers: int = 2
    ffn_mult: int = 2
    pair_bias: bool = True
    chemistry: bool = True
    backend: str = "manual"

    def __post_init__(self):
        if (
            self.width % self.heads
            or (self.width // self.heads) % 2
            or self.layers < 1
            or self.ffn_mult < 2
            or self.backend not in ["manual", "sdpa"]
        ):
            raise ValueError("validated model configuration")


class DenseBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.norm = nn.LayerNorm(c.width)
        self.qkv = nn.Linear(c.width, c.width * 3)
        self.out = nn.Linear(c.width, c.width)
        self.norm2 = nn.LayerNorm(c.width)
        self.ff = nn.Sequential(
            nn.Linear(c.width, c.width * c.ffn_mult), nn.GELU(), nn.Linear(c.width * c.ffn_mult, c.width)
        )

    def forward(self, x, bias):
        L = len(x)
        h = self.c.heads
        dh = self.c.width // h
        z = self.qkv(self.norm(x)).view(L, 3, h, dh)
        q, k, v = [z[:, i].permute(1, 0, 2) for i in range(3)]
        positions = torch.arange(L, device=x.device)
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)
        if self.c.backend == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)
        else:
            out = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh) + bias, -1) @ v
        return (
            self.ff(self.norm2(x + self.out(out.permute(1, 0, 2).reshape(L, -1))))
            + x
            + self.out(out.permute(1, 0, 2).reshape(L, -1))
        )


class PrefixEncoder(nn.Module):
    def __init__(self, c=ModelConfig(), sequence_only=False):
        super().__init__()
        self.config = c
        self.sequence_only = sequence_only
        self.aa = nn.Embedding(20, c.width)
        self.token = nn.Linear(22 + 5 + 3 + 1 + 3 + 3 + 1, c.width)
        self.context = nn.Sequential(nn.Linear(2, c.width), nn.SiLU(), nn.Linear(c.width, c.width))
        self.tip = nn.Parameter(torch.zeros(c.width))
        self.pair = DensePairBias(c.heads, chemistry=c.chemistry)
        self.blocks = nn.ModuleList([DenseBlock(c) for _ in range(c.layers)])
        self.final = nn.LayerNorm(c.width)

    def forward(self, state):
        if type(state) != BuildState:
            raise ValueError("only literal build prefix; no edit/oracle/native fields")
        if state.ca.device != self.tip.device:
            raise ValueError("model/state device mismatch; transfer explicitly")
        f = features(state)
        dtype = self.tip.dtype
        t = state.t
        geom = torch.stack(
            [
                f["trailing_CA_bond_A"],
                f["trailing_CA_bend_rad"],
                f["trailing_CA_virtual_torsion_rad"].sin() * f["CA_torsion_valid"],
                f["trailing_CA_virtual_torsion_rad"].cos() * f["CA_torsion_valid"],
                f["prefix_crowding"],
            ],
            -1,
        )
        masks = torch.stack([f["CA_bond_valid"], f["CA_bend_valid"], f["CA_torsion_valid"]], -1)
        chem = f["chemistry"] if self.config.chemistry else torch.zeros_like(f["chemistry"])
        allfeat = torch.cat(
            [
                chem,
                geom,
                masks,
                f["frame_valid"][:, None],
                torch.stack(
                    [
                        f["prefix_crowding"],
                        f["prefix_exposure_proxy"],
                        f["prefix_nonpolar_packing_proxy"]
                        if self.config.chemistry
                        else torch.zeros_like(f["prefix_crowding"]),
                    ],
                    -1,
                ),
                f["generated_terminal_flags"],
                f["CA_valid"][:, None],
            ],
            -1,
        ).to(dtype)
        if self.sequence_only:
            # Geometry and geometry-validity/missingness all removed, including pair bias.
            allfeat = torch.cat(
                [
                    chem,
                    torch.zeros((t, 12), dtype=chem.dtype, device=state.ca.device),
                    f["generated_terminal_flags"],
                    torch.zeros((t, 1), dtype=chem.dtype, device=state.ca.device),
                ],
                -1,
            ).to(dtype)
        context = self.context(torch.tensor([state.N / 500.0, state.t / state.N], dtype=dtype, device=state.ca.device))
        tokens = self.aa(f["token_ids"]) + self.token(allfeat) + context
        x = torch.cat([tokens, (self.tip + context)[None]], 0)
        bias = torch.zeros((self.config.heads, t + 1, t + 1), dtype=dtype, device=state.ca.device)
        if t and self.config.pair_bias and not self.sequence_only:
            bias[:, :t, :t] = self.pair(
                {k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in f.items()}
            )
        for block in self.blocks:
            x = block(x, bias)
        return self.final(x[-1])
