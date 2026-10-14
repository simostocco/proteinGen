"""Torch counterpart of approved prefix-only reference geometry, preserving gradients."""

import torch
from .ca_builder import inverse
from ..chemistry.reference_adapter import chemistry_table, load


def features(state):
    x = state.ca
    valid = state.valid
    t = state.t
    dtype = x.dtype
    z = x.new_zeros
    safe = torch.where(valid[:, None], x, z(x.shape))
    center = safe.sum(0) / valid.sum().clamp_min(1)
    centered = torch.where(valid[:, None], x - center, z(x.shape))
    bonds = []
    bends = []
    torsions = []
    frames = []
    fv = []
    bm = []
    am = []
    tm = []
    for i in range(t):
        bond = z(())
        bend = z(())
        torsion = z(())
        frame = z((3, 3))
        bvalid = avalid = tvalid = fvalid = False
        if i >= 1 and bool(valid[i - 1 : i + 1].all()):
            bond = torch.linalg.vector_norm(x[i] - x[i - 1])
            bvalid = True
        if i >= 2 and bool(valid[i - 2 : i + 1].all()):
            a = x[i - 1] - x[i - 2]
            b = x[i] - x[i - 1]
            an = torch.linalg.vector_norm(a)
            bn = torch.linalg.vector_norm(b)
            cross = torch.linalg.cross(a, b)
            cn = torch.linalg.vector_norm(cross)
            if float(an.detach()) > 1e-8 and float(bn.detach()) > 1e-8:
                bend = torch.atan2(cn, -(a * b).sum())
                avalid = True
            if min(float(an.detach()), float(bn.detach()), float(cn.detach())) > 1e-8:
                ex = b / bn
                ez = cross / cn
                ey = torch.linalg.cross(ez, ex)
                frame = torch.stack([ex, ey, ez])
                fvalid = True
        if i >= 3 and bool(valid[i - 3 : i + 1].all()):
            try:
                torsion = inverse(x[i - 3 : i + 1])[0][2]
                tvalid = True
            except ValueError:
                pass
        bonds.append(bond)
        bends.append(bend)
        torsions.append(torsion)
        frames.append(frame)
        fv.append(fvalid)
        bm.append(bvalid)
        am.append(avalid)
        tm.append(tvalid)
    stack = lambda rows, shape: torch.stack(rows) if rows else z(shape)
    frame = stack(frames, (0, 3, 3))
    frame_valid = torch.tensor(fv, dtype=torch.bool, device=x.device)
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
        "CA_bond_valid": torch.tensor(bm, dtype=torch.bool, device=x.device),
        "trailing_CA_bend_rad": stack(bends, (0,)),
        "CA_bend_valid": torch.tensor(am, dtype=torch.bool, device=x.device),
        "trailing_CA_virtual_torsion_rad": stack(torsions, (0,)),
        "CA_torsion_valid": torch.tensor(tm, dtype=torch.bool, device=x.device),
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
