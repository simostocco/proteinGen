"""Torch counterpart of approved prefix-only reference geometry, preserving gradients."""

import torch
from .ca_builder import inverse
from ..chemistry.reference_adapter import chemistry_table, load


def _features(state):
    x = state.ca
    valid = state.valid
    t = state.t
    dtype = x.dtype
    z = x.new_zeros
    safe = torch.where(valid[:, None], x, z(x.shape))
    center = safe.sum(0) / valid.sum().clamp_min(1)
    centered = torch.where(valid[:, None], x - center, z(x.shape))
    # Vectorized trailing geometry: same signed definitions, no Python per-residue
    # autograd graphs or scalar CUDA synchronizations. Invalid atoms stay masked.
    bonds=z((t,)); bends=z((t,)); torsions=z((t,)); frame=z((t,3,3))
    bm=torch.zeros(t,dtype=torch.bool,device=x.device); am=bm.clone(); tm=bm.clone(); frame_valid=bm.clone()
    if t>=2:
        b=x[1:]-x[:-1]; bn=torch.linalg.vector_norm(b,dim=-1)
        bm[1:]=valid[1:] & valid[:-1]
        bonds[1:]=torch.where(bm[1:],bn,z(bn.shape))
    if t>=3:
        a=b[:-1]; c=b[1:]; an=bn[:-1]; cn=bn[1:]
        cross=torch.linalg.cross(a,c); crossn=torch.linalg.vector_norm(cross,dim=-1)
        trip=valid[:-2]&valid[1:-1]&valid[2:]
        av=trip & (an>1e-8)&(cn>1e-8)
        bend=torch.atan2(torch.where(av,crossn,z(crossn.shape)),torch.where(av,-(a*c).sum(-1),torch.ones_like(crossn)))
        bends[2:]=torch.where(av,bend,z(bend.shape));am[2:]=av
        fv=av & (crossn>1e-8)
        ex=c/cn.clamp_min(1e-10)[:,None];ez=cross/crossn.clamp_min(1e-10)[:,None];ey=torch.linalg.cross(ez,ex)
        frame[2:]=torch.where(fv[:,None,None],torch.stack([ex,ey,ez],dim=1),z((len(ex),3,3)));frame_valid[2:]=fv
    if t>=4:
        u=b[1:-1]/bn[1:-1].clamp_min(1e-10)[:,None]
        v=-b[:-2];v=v-(v*u).sum(-1)[:,None]*u
        w=b[2:];w=w-(w*u).sum(-1)[:,None]*u
        vn=torch.linalg.vector_norm(v,dim=-1);wn=torch.linalg.vector_norm(w,dim=-1)
        tv=(valid[:-3]&valid[1:-2]&valid[2:-1]&valid[3:] & (bn[:-2]>=1e-10)&(bn[1:-1]>=1e-10)&(bn[2:]>=1e-10)&(vn>=1e-10)&(wn>=1e-10))
        vh=v/vn.clamp_min(1e-10)[:,None];wh=w/wn.clamp_min(1e-10)[:,None]
        sine=(torch.linalg.cross(u,vh)*wh).sum(-1);cosine=(vh*wh).sum(-1)
        # Avoid undefined atan2(0,0) derivative even for invalid masked frames.
        sine=torch.where(tv,sine,z(sine.shape));cosine=torch.where(tv,cosine,torch.ones_like(cosine))
        torsions[3:]=torch.where(tv,torch.atan2(sine,cosine),z(sine.shape));tm[3:]=tv
    stack=lambda rows,shape:rows
    ids = torch.tensor([load().INDEX[a] for a in state.residues], dtype=torch.long, device=x.device)
    chem = torch.as_tensor(chemistry_table(), dtype=dtype, device=x.device)[ids]
    vec = x[None, :, :] - x[:, None, :]
    dist = torch.linalg.vector_norm(vec, dim=-1)
    pair_valid = valid[:, None] & valid[None, :]
    direction = vec / dist.clamp_min(1e-8)[..., None]
    orient = torch.einsum("iab,ijb->ija", frame, direction)
    relrot = torch.einsum("iab,jcb->ijac", frame, frame)
    idx = torch.arange(t, device=x.device)
    longsep = (idx[:, None] - idx[None, :]).abs() >= 3
    w = torch.relu(1 - dist / 8.0).square() * pair_valid * longsep
    crowd = w.sum(-1)
    ci = load().CHEM_NAMES.index("carbon_only_sidechain")
    packing = (w * chem[:, ci, None] * chem[None, :, ci]).sum(-1)
    terminals = torch.zeros((t, 3), dtype=torch.bool, device=x.device)
    if t:
        terminals[0, 0] = True
        terminals[-1, 1] = t == state.N
        terminals[-1, 2] = t < state.N
    return {
        "token_ids": ids,
        "chemistry": chem,
        "identity_onehot": torch.nn.functional.one_hot(ids, num_classes=20).to(dtype),
        "CA_centered_A": centered,
        "CA_valid": valid,
        "right_handed_frame": frame,
        "frame_valid": frame_valid,
        "trailing_CA_bond_A": stack(bonds, (0,)),
        "CA_bond_valid": bm,
        "trailing_CA_bend_rad": stack(bends, (0,)),
        "CA_bend_valid": am,
        "trailing_CA_virtual_torsion_rad": stack(torsions, (0,)),
        "CA_torsion_valid": tm,
        "pair_distance_A": dist,
        "pair_valid": pair_valid,
        "pair_local_direction": orient,
        "pair_relative_rotation": relrot,
        "prefix_contact8A": (dist <= 8) & pair_valid,
        "prefix_crowding": crowd,
        "prefix_exposure_proxy": torch.where(valid, 1 / (1 + crowd), torch.zeros_like(crowd)),
        "prefix_nonpolar_packing_proxy": packing,
        "generated_terminal_flags": terminals,
        "plan_present": False,
        "generated_plan_values": z((8,)),
    }

from contextlib import contextmanager
from contextvars import ContextVar
_PREFIX_CACHE=ContextVar('E016_prefix_geometry_cache',default=None)
@contextmanager
def reuse_static_prefix(state):
    if state.ca.requires_grad:raise ValueError('only detached replay prefix can be cached')
    entry=(state,state.ca._version,state.valid._version,_features(state))
    token=_PREFIX_CACHE.set(entry)
    try:yield
    finally:_PREFIX_CACHE.reset(token)
def features(state):
    c=_PREFIX_CACHE.get()
    if c is not None and c[0] is state:
        if (state.ca._version,state.valid._version)!=c[1:3]:raise ValueError('cached prefix was mutated')
        return c[3]
    return _features(state)
