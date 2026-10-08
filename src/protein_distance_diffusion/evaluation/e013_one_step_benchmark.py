"""Reference-associated diagnostics; never supplied to model inference."""
import torch
from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import local_geometry


def diagnostics(pred, target, vhat, vstar, scale):
    # One unpadded protein. Proper-rotation Kabsch; no reflection alignment.
    p, q = pred*scale, target*scale
    u, _, vh = torch.linalg.svd(p.T@q)
    correction = torch.eye(3, device=p.device)
    correction[-1, -1] = torch.linalg.det(u@vh)
    rotation = u@correction@vh
    local = [float(((p[k:]-p[:-k]).norm(dim=-1)-(q[k:]-q[:-k]).norm(dim=-1)).square().mean()) for k in (1, 2, 3)]
    pf, _, _ = local_geometry(pred)
    tf, _, _ = local_geometry(target)
    assess = (pf[:, 29] > 0) & (tf[:, 29] > 0)
    sign_assess = assess & (tf[:, 27].abs() > 1e-3)
    torsion = (pf[:, 25:27]-tf[:, 25:27]).square().sum(-1)
    distances = torch.cdist(p, p)
    ids = torch.arange(len(p), device=p.device)
    sep = (ids[:, None]-ids[None]).abs()
    nonlocal_pairs = sep > 3
    long_pairs = sep >= 12
    bonds = (p[1:]-p[:-1]).norm(dim=-1)
    rg = p.square().sum(-1).mean().sqrt()
    trg = q.square().sum(-1).mean().sqrt()
    return dict(v_mse=float((vhat-vstar).square().mean()),
                x0_mse_angstrom2=float((p-q).square().mean()),
                aligned_rmsd_angstrom=float((p@rotation-q).square().sum(-1).mean().sqrt()),
                local_mse_angstrom2=sum(local)/3,
                **{f'offset_{k+1}_mse_angstrom2': val for k, val in enumerate(local)},
                adjacent_mean_angstrom=float(bonds.mean()), adjacent_std_angstrom=float(bonds.std()),
                adjacent_quantiles_angstrom=[float(z) for z in torch.quantile(bonds, bonds.new_tensor([.05,.5,.95]))],
                torsion_assessed=int(assess.sum()), torsion_mse=float(torsion[assess].mean()) if assess.any() else None,
                triple_sign_assessed=int(sign_assess.sum()),
                triple_inversion_fraction=float(((pf[:, 27]*tf[:, 27] < 0)[sign_assess]).float().mean()) if sign_assess.any() else None,
                clash_fraction=float((distances[nonlocal_pairs] < 3).float().mean()),
                radius_of_gyration_angstrom=float(rg), radius_ratio=float(rg/trg),
                long_range_contact_density=float((distances[long_pairs] < 8).float().mean()) if long_pairs.any() else None,
                finite=bool(torch.isfinite(p).all()), collapsed=bool(rg < .25*trg))


def summarize(rows, field):
    values = [r[field] for r in rows if r[field] is not None]
    return sum(values)/len(values) if values else None


def adjudicate(control, candidate, tolerances):
    ratio = lambda a, b: a/max(b, 1e-12)
    aggregate = ratio(summarize(candidate, 'v_mse'), summarize(control, 'v_mse'))
    local = ratio(summarize(candidate, 'local_mse_angstrom2'), summarize(control, 'local_mse_angstrom2'))
    chiral = ratio(summarize(candidate, 'torsion_mse'), summarize(control, 'torsion_mse'))
    length_ratios = {}
    for stratum in sorted({r['stratum'] for r in candidate}):
        length_ratios[stratum] = ratio(summarize([r for r in candidate if r['stratum']==stratum], 'v_mse'),
                                      summarize([r for r in control if r['stratum']==stratum], 'v_mse'))
    gates = dict(finite=all(r['finite'] for r in candidate), no_collapse=not any(r['collapsed'] for r in candidate),
                 aggregate_noninferiority=aggregate <= 1+tolerances['aggregate'],
                 failure_family_improvement=min(local, chiral) <= 1-tolerances['improvement'],
                 length_nonregression=max(length_ratios.values()) <= 1+tolerances['length'])
    return dict(passed=all(gates.values()), gates=gates, v_ratio=aggregate, local_ratio=local,
                torsion_ratio=chiral, length_v_ratios=length_ratios)
