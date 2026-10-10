"""O(3)-typed multiscale core and approved SO(3)-only chiral output.

Only current coordinates, generated context, chain order, lengths/masks and time
enter inference. Cartesian components never enter scalar MLPs.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import local_geometry, radial_basis
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion, center_coordinates


@dataclass(frozen=True)
class State:
    h: torch.Tensor
    v: torch.Tensor
    p: torch.Tensor
    a: torch.Tensor


def vector_linear(layer, vector):
    return layer(vector.transpose(-1, -2)).transpose(-1, -2)


def pool_mean(value, stride=4):
    n = len(value)
    padding = (-n) % stride
    padded = torch.cat([value, value.new_zeros((padding, *value.shape[1:]))])
    summed = padded.reshape(-1, stride, *value.shape[1:]).sum(1)
    counts = torch.arange(0, n, stride, device=value.device).sub(n).neg().clamp_max(stride)
    return summed / counts.reshape((-1,) + (1,) * (value.ndim - 1))


def pool_state(state):
    return State(*(pool_mean(value) for value in (state.h, state.v, state.p, state.a)))


def graph(x, previous, k, k_self, dense=False):
    n = len(x)
    ids = torch.arange(n, device=x.device)
    separation = ids[None, :] - ids[:, None]
    polymer = (separation.abs() >= 1) & (separation.abs() <= 3)
    current = torch.zeros_like(polymer)
    sc = torch.zeros_like(polymer)
    with torch.no_grad():
        if dense:
            current = separation != 0
        elif k:
            distances = torch.cdist(x.float(), x.float()).masked_fill(separation == 0, float("inf"))
            current.scatter_(1, torch.argsort(distances, stable=True)[:, : min(k, n - 1)], True)
        if previous is not None and k_self:
            distances = torch.cdist(previous.float(), previous.float()).masked_fill(separation == 0, float("inf"))
            sc.scatter_(1, torch.argsort(distances, stable=True)[:, : min(k_self, n - 1)], True)
    i, j = (polymer | current | sc).nonzero(as_tuple=True)
    relative = x[j] - x[i]
    distance = relative.norm(dim=-1)
    unit = relative / distance[:, None].clamp_min(1e-6)
    prev_distance = distance.new_zeros(len(i)) if previous is None else (previous[j] - previous[i]).norm(dim=-1)
    sep = separation[i, j]
    features = torch.cat(
        [
            radial_basis(distance),
            radial_basis(prev_distance),
            (sep.float() / max(n - 1, 1))[:, None],
            torch.stack([sep.abs() == z for z in (1, 2, 3)], -1).float(),
            current[i, j, None].float(),
            sc[i, j, None].float(),
            torch.full((len(i), 1), previous is not None, device=x.device).float(),
        ],
        -1,
    )
    return i, j, features, unit


def rope(values):
    n, width = values.shape
    angle = torch.arange(n, device=values.device).float()[:, None] / (
        10000 ** (torch.arange(0, width, 2, device=values.device).float() / width)
    )
    pairs = values.float().reshape(n, -1, 2)
    return torch.stack(
        [
            pairs[..., 0] * angle.cos() - pairs[..., 1] * angle.sin(),
            pairs[..., 0] * angle.sin() + pairs[..., 1] * angle.cos(),
        ],
        -1,
    ).reshape(n, width)


class TypedBlock(nn.Module):
    def __init__(self, cs, cv, cp, ca, ratio=2, attention=False):
        super().__init__()
        self.widths = cs, cv, cp, ca
        self.norm = nn.LayerNorm(cs)
        self.message = nn.Sequential(
            nn.Linear(2 * cs + 39 + 64 + cv + ca + cp, ratio * cs),
            nn.SiLU(),
            nn.Linear(ratio * cs, cs + 2 * cv + cp + 2 * ca),
        )
        self.feed = nn.Sequential(nn.LayerNorm(cs), nn.Linear(cs, 2 * cs), nn.SiLU(), nn.Linear(2 * cs, cs))
        self.vmix = nn.Linear(cv, cv, bias=False)
        self.amix = nn.Linear(ca, ca, bias=False)
        self.pmix = nn.Linear(cp, cp, bias=False)
        self.a_to_v = nn.Linear(ca, cv, bias=False)
        self.p_to_v = nn.Linear(cp, cv, bias=False)
        self.v_to_a = nn.Linear(cv, ca, bias=False)
        self.p_to_a = nn.Linear(cp, ca, bias=False)
        self.v_to_p = nn.Linear(cv, cp, bias=False)
        self.a_to_p = nn.Linear(ca, cp, bias=False)
        self.attention = attention
        if attention:
            self.query = nn.Linear(cs, 64, bias=False)
            self.key = nn.Linear(cs, 64, bias=False)
            self.attention_bias = nn.Linear(39, 1)

    def forward(self, h, v, p, a, i, j, features, unit, time, gate):
        cs, cv, cp, ca = self.widths
        # Normalizations are even, scalar-valued and computed in FP32.
        vn = v.float().square().sum(-1).add(1e-8).sqrt()
        an = a.float().square().sum(-1).add(1e-8).sqrt()
        hv = self.norm(h)
        inputs = torch.cat([hv[i], hv[j], features, time.expand(len(i), -1), vn[j], an[j], p[j].float().square()], -1)
        weights = self.message(inputs)
        hs, vs, vr, ps, av, ar = weights.split((cs, cv, cv, cp, ca, ca), -1)
        vv = vector_linear(self.vmix, v)
        aa = vector_linear(self.amix, a)
        pp = self.pmix(p)
        va = vector_linear(self.a_to_v, a)
        pa = self.p_to_v(p)
        pv = self.p_to_a(p)
        vi = vector_linear(self.v_to_a, v)
        parity_gate = gate[i, None]
        vm = vs[..., None].tanh() * vv[j] + vr[..., None].tanh() * unit[:, None]
        vm = vm + parity_gate[..., None] * pa[j, :, None] * va[j]
        am = av[..., None].tanh() * aa[j] + ar[..., None].tanh() * torch.linalg.cross(vi[i], vi[j])
        am = am + parity_gate[..., None] * pv[j, :, None] * unit[:, None]
        pseudoscalar = (vector_linear(self.v_to_p, v)[i] * vector_linear(self.a_to_p, a)[j]).sum(-1)
        pm = ps.tanh() * pp[j] + parity_gate * pseudoscalar
        if self.attention:
            query, key = rope(self.query(hv)), rope(self.key(hv))
            logits = (query[i] * key[j]).sum(-1) / 8 + self.attention_bias(features)[:, 0].float()
            maxes = logits.new_full((len(h),), -torch.inf).scatter_reduce(
                0, i, logits, reduce="amax", include_self=True
            )
            exp = (logits - maxes[i]).exp()
            totals = logits.new_zeros(len(h)).index_add(0, i, exp)
            edge_weight = exp / totals[i].clamp_min(1e-8)
        else:
            degree = torch.bincount(i, minlength=len(h)).clamp_min(1)
            edge_weight = degree[i].float().reciprocal()

        def aggregate(value):
            value = value.float() * edge_weight.reshape((-1,) + (1,) * (value.ndim - 1))
            return value.new_zeros((len(h), *value.shape[1:])).index_add(0, i, value)

        h = h + 0.1 * aggregate(hs)
        h = h + 0.1 * self.feed(h)
        v = v + 0.1 * aggregate(vm)
        a = a + 0.1 * aggregate(am) * gate[:, None, None]
        p = p + 0.1 * aggregate(pm) * gate[:, None]
        # Smooth invariant bounded normalizations; odd tanh preserves P parity.
        v = v / (1 + v.float().square().sum(-1, keepdim=True)).sqrt()
        a = a / (1 + a.float().square().sum(-1, keepdim=True)).sqrt()
        p = p.tanh()
        return h, v, p, a


class Broadcast(nn.Module):
    def __init__(self, cs, cv, cp, ca):
        super().__init__()
        self.h = nn.Linear(cs, cs)
        self.v = nn.Linear(cv, cv, bias=False)
        self.p = nn.Linear(cp, cp, bias=False)
        self.a = nn.Linear(ca, ca, bias=False)
        self.gate = nn.Linear(2 * cs, 4)

    def forward(self, child, parent):
        mapping = torch.arange(len(child.h), device=child.h.device) // 4
        gates = self.gate(torch.cat([child.h, parent.h[mapping]], -1)).sigmoid()
        return State(
            child.h + gates[:, :1] * self.h(parent.h)[mapping],
            child.v + gates[:, 1, None, None] * vector_linear(self.v, parent.v)[mapping],
            child.p + gates[:, 2, None] * self.p(parent.p)[mapping],
            child.a + gates[:, 3, None, None] * vector_linear(self.a, parent.a)[mapping],
        )


class E015MultiscaleParityDenoiser(nn.Module):
    def __init__(
        self,
        scalar_channels=384,
        vector_channels=96,
        pseudoscalar_channels=48,
        axial_channels=48,
        fine_blocks=8,
        medium_blocks=4,
        coarse_blocks=4,
        mlp_ratio=2,
        k_current=24,
        k_self=16,
        activation_checkpointing=True,
        chiral_head=True,
        parity=True,
        multiscale=True,
        self_conditioning=True,
        parity_midpoint=0.0,
        gate_width=1.0,
    ):
        super().__init__()
        cs, cv, cp, ca = scalar_channels, vector_channels, pseudoscalar_channels, axial_channels
        self.widths = cs, cv, cp, ca
        self.k_current, self.k_self = k_current, k_self
        self.activation_checkpointing, self.chiral_enabled = activation_checkpointing, chiral_head
        self.parity_enabled, self.multiscale_enabled, self.self_conditioning_enabled = (
            parity,
            multiscale,
            self_conditioning,
        )
        self.register_buffer("parity_midpoint", torch.tensor(float(parity_midpoint)))
        self.gate_width = gate_width
        self.register_buffer(
            "logsnr",
            (CoordinateVPDiffusion().alphas_cumprod.log() - torch.log1p(-CoordinateVPDiffusion().alphas_cumprod)),
        )
        self.time = nn.Sequential(nn.Linear(33, 128), nn.SiLU(), nn.Linear(128, 64))
        self.input = nn.Sequential(nn.Linear(76, cs), nn.SiLU(), nn.Linear(cs, cs))
        self.vinit = nn.Linear(3, cv, bias=False)
        self.ainit = nn.Linear(1, ca, bias=False)
        self.pinit = nn.Linear(2, cp, bias=False)
        self.fine = nn.ModuleList([TypedBlock(cs, cv, cp, ca, mlp_ratio) for _ in range(fine_blocks)])
        self.medium = nn.ModuleList(
            [TypedBlock(cs, cv, cp, ca, mlp_ratio, attention=True) for _ in range(medium_blocks)]
        )
        self.coarse = nn.ModuleList(
            [TypedBlock(cs, cv, cp, ca, mlp_ratio, attention=True) for _ in range(coarse_blocks)]
        )
        self.up_medium = Broadcast(cs, cv, cp, ca)
        self.up_fine = Broadcast(cs, cv, cp, ca)
        self.polar_v = nn.Linear(cs, cv)
        self.polar_p = nn.Linear(cp, ca, bias=False)
        self.chiral_a = nn.Linear(cs, ca)
        self.chiral_p = nn.Linear(cp, cv, bias=False)
        self.local = nn.Sequential(nn.Linear(cs, cs), nn.SiLU(), nn.Linear(cs, 3))
        # Nonzero small weights preserve gradient flow from the first update.
        for layer in (self.chiral_a, self.chiral_p, self.local[-1]):
            nn.init.normal_(layer.weight, std=1e-5)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def time_features(self, t):
        frequencies = torch.exp(torch.arange(16, device=t.device).float() * (-math.log(10000) / 16))
        phase = t.float() * frequencies
        return self.time(torch.cat([phase.sin(), phase.cos(), self.logsnr[t].reshape(1).clamp(-20, 20) / 20]))

    def gate(self, t, reliability):
        return (
            (self.logsnr[t] - self.parity_midpoint) / self.gate_width + 2 * (reliability - 1)
        ).sigmoid() * reliability

    def run_blocks(self, state, blocks, x, prev, time, gate, dense=False):
        i, j, features, unit = graph(
            x, prev, self.k_current if not dense else 0, self.k_self if prev is not None else 0, dense
        )
        if not len(i):
            return state
        for block in blocks:
            args = (state.h, state.v, state.p, state.a, i, j, features, unit, time, gate)
            if self.training and self.activation_checkpointing and torch.is_grad_enabled():
                outputs = checkpoint(block, *args, use_reentrant=False)
            else:
                outputs = block(*args)
            state = State(*outputs)
        return state

    def trunk_single(self, x, t, previous=None):
        x = x.float() - x.float().mean(0)
        if previous is not None:
            previous = previous.float() - previous.float().mean(0)
        features, frame, reliability = local_geometry(x)
        time = self.time_features(t)
        sc = x.new_zeros(x.shape) if previous is None else previous - x
        geometry = torch.cat(
            [
                features[:, :6],
                features[:, 24:25],
                features[:, 26:27],
                features[:, 27:28].square(),
                sc.norm(dim=-1, keepdim=True),
                torch.full((len(x), 1), previous is not None, device=x.device),
                torch.full((len(x), 1), len(x) / 500, device=x.device),
            ],
            -1,
        )
        h = self.input(torch.cat([geometry, time.expand(len(x), -1)], -1))
        seeds = torch.stack([frame[:, :, 0], frame[:, :, 1], sc], 1)
        v = vector_linear(self.vinit, seeds)
        a = vector_linear(self.ainit, frame[:, :, 2][:, None])
        p = self.pinit(features[:, [25, 27]])
        gate = self.gate(t, reliability) if self.parity_enabled else torch.zeros_like(reliability)
        a, p = a * gate[:, None, None], p * gate[:, None]
        state = self.run_blocks(State(h, v, p, a), self.fine, x, previous, time, gate)
        shapes = [len(x)]
        if self.multiscale_enabled:
            mx, mp = pool_mean(x), None if previous is None else pool_mean(previous)
            medium = pool_state(state)
            mgate = pool_mean(gate)
            medium = self.run_blocks(medium, self.medium, mx, mp, time, mgate)
            cx, cprev = pool_mean(mx), None if mp is None else pool_mean(mp)
            coarse = self.run_blocks(pool_state(medium), self.coarse, cx, cprev, time, pool_mean(mgate), dense=True)
            medium = self.up_medium(medium, coarse)
            state = self.up_fine(state, medium)
            shapes += [len(mx), len(cx)]
        return state, frame, reliability, gate, shapes

    def single(self, x, t, previous=None):
        state, frame, reliability, gate, shapes = self.trunk_single(x, t, previous)
        h, v, p, a = state.h, state.v, state.p, state.a
        polar = (self.polar_v(h)[..., None] * v).sum(1) / math.sqrt(self.widths[1])
        polar = polar + (self.polar_p(p)[..., None] * a).sum(1) / math.sqrt(self.widths[3])
        chiral = (self.chiral_a(h)[..., None] * a).sum(1) / math.sqrt(self.widths[3])
        chiral = chiral + (self.chiral_p(p)[..., None] * v).sum(1) / math.sqrt(self.widths[1])
        local = torch.einsum("nac,nc->na", frame, self.local(h).float())
        if not self.chiral_enabled:
            chiral, local = torch.zeros_like(chiral), torch.zeros_like(local)
        effective_chiral, effective_local = gate[:, None] * chiral, gate[:, None] * local

        def center(z):
            return z - z.mean(0)

        polar, effective_chiral, effective_local = map(center, (polar, effective_chiral, effective_local))
        return polar + effective_chiral + effective_local, dict(
            global_v=polar,
            chiral_effective=effective_chiral,
            local_effective=effective_local,
            gate=gate,
            reliability=reliability,
            assessable=features_assess(frame),
            shapes=shapes,
        )

    def forward(self, noisy_coordinates, timesteps, lengths, residue_mask, x0_prev=None, sc_available=None):
        return self.with_components(noisy_coordinates, timesteps, lengths, residue_mask, x0_prev, sc_available)[0]

    def with_components(self, noisy_coordinates, timesteps, lengths, residue_mask, x0_prev=None, sc_available=None):
        if noisy_coordinates.shape[-1] != 3 or not torch.equal(lengths, residue_mask.sum(1)):
            raise ValueError("Invalid coordinate mask/length contract")
        output = torch.zeros_like(noisy_coordinates, dtype=torch.float32)
        components = []
        for b, n in enumerate(lengths.tolist()):
            if not torch.all(residue_mask[b, :n]) or torch.any(residue_mask[b, n:]):
                raise ValueError("E015 requires contiguous prefix masks")
            previous = None
            if (
                self.self_conditioning_enabled
                and x0_prev is not None
                and sc_available is not None
                and bool(sc_available[b])
            ):
                previous = x0_prev[b, :n].detach()
            value, component = self.single(noisy_coordinates[b, :n], timesteps[b], previous)
            output[b, :n] = value
            components.append(component)
        return center_coordinates(output, residue_mask), components


def features_assess(frame):
    return frame.square().sum((1, 2)) > 0
