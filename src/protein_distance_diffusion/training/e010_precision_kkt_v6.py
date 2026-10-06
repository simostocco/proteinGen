"""Read-only numerical audit of the fixed Phase 4D constrained objective."""

import torch

from ..models.e010_hybrid_local import local_representation
from .e010_phase4d_objective_v2 import smooth_bounded_local
from .e010_recurrent_capacity import objective_components

RADIUS = 0.04


def physical_trajectory(pg, mask, v):
    """Historical v5 arithmetic with physical leaves, avoiding z roundtrip."""
    current = pg.detach()
    states, steps = [current], []
    for vector in v:
        with torch.no_grad():
            valid = local_representation(current, mask)["eligible"]
        delta = smooth_bounded_local(vector, RADIUS)
        limit = torch.nextafter(delta.new_tensor(RADIUS), delta.new_tensor(0.0))
        size = delta.norm(dim=-1, keepdim=True)
        delta = delta * torch.minimum(torch.ones_like(size), limit / size.clamp_min(1e-12))
        delta = delta * valid[..., None]
        current = current + delta
        states.append(current)
        steps.append(dict(delta=delta, eligible=valid))
    return dict(prediction=current, states=states, steps=steps)


def radial_geometry(v):
    size = v.norm(dim=-1)
    a = torch.rsqrt(1 + (size / RADIUS).square())
    unit = v / size.clamp_min(1e-300)[..., None]
    eye = torch.eye(3, dtype=v.dtype, device=v.device)
    jacobian = a[..., None, None] * eye + (a.pow(3) - a)[..., None, None] * unit[..., :, None] * unit[..., None, :]
    return dict(
        v_norm=size,
        delta_norm=a * size,
        saturation=a * size / RADIUS,
        a=a,
        radial_eigenvalue=a.pow(3),
        tangential_eigenvalue=a,
        condition_number=1 / a.square(),
        jacobian=jacobian,
    )


def objective(pred, b, *, pure_float64=False, examples_total=60, quartets_total=13029):
    """Same reductions/terms; pure mode removes only historical Cartesian float cast.

    Include the historical 1e-5 source-residual contribution. No new objective,
    coefficient, frozen-quartet or eligibility change is introduced.
    """
    old = objective_components(pred, b, examples_total=examples_total, quartets_total=quartets_total)
    if not pure_float64:
        return old
    p = torch.where(b["mask"][..., None], pred, 0)
    y = torch.where(b["mask"][..., None], b["target"], 0)
    s = torch.where(b["mask"][..., None], b["source"], 0)
    mask = b["mask"]
    per = (((p - y).square().mean(-1) + 1e-5 * (p - s).square().mean(-1)) * mask).sum(1) / mask.sum(1).clamp_min(1)
    cart = per.mean() * pred.shape[0] / examples_total
    # Reuse exact historical local and chiral tensors.
    return dict(
        local=old["local"], cartesian=cart, chiral=old["chiral"], total=old["local"] + 16.8 * cart + 2 * old["chiral"]
    )


def correction_trajectory(pg, mask, corrections):
    """Independent Cartesian correction-space leaves, current eligibility only."""
    p = pg.detach()
    states, eligibility = [p], []
    for delta in corrections:
        with torch.no_grad():
            valid = local_representation(p, mask)["eligible"]
        p = p + delta * valid[..., None]
        eligibility.append(valid)
        states.append(p)
    return dict(prediction=p, states=states, eligible=torch.stack(eligibility))


def kkt(delta, gradient, eligible, boundary_relative_tolerance=1e-6):
    """Active ball: minimize ||g+2 lambda delta|| over lambda>=0.

    Positive radial gradient allows feasible inward descent; negative radial
    gradient points to infeasible outward descent and admits a multiplier.
    Inactive/invalid corrections are fixed, so their residual is zero.
    """
    size = delta.norm(dim=-1)
    unit = delta / size.clamp_min(1e-300)[..., None]
    radial = (gradient * unit).sum(-1)
    tangent = gradient - radial[..., None] * unit
    active = eligible & (size >= RADIUS * (1 - boundary_relative_tolerance))
    multiplier = torch.where(active, (-radial / (2 * size.clamp_min(1e-300))).clamp_min(0), 0)
    residual = torch.where(eligible[..., None], gradient + 2 * multiplier[..., None] * delta, 0)
    feasible_radial = torch.where(active, radial.clamp_min(0), radial)
    return dict(
        active=active,
        multiplier=multiplier,
        radial_gradient=radial,
        tangent_gradient=tangent,
        tangent_residual=tangent.norm(dim=-1) * eligible,
        feasible_radial_residual=feasible_radial.abs() * eligible,
        residual=residual,
        residual_norm=residual.norm(dim=-1),
    )


def project_ball(delta):
    size = delta.norm(dim=-1, keepdim=True)
    return delta * torch.minimum(torch.ones_like(size), RADIUS / size.clamp_min(1e-300))


def shadow_direction(delta, gradient, eligible, *, tangential=False):
    info = kkt(delta, gradient, eligible)
    direction = -info["residual"]
    if tangential:
        direction = torch.where(info["active"][..., None], -info["tangent_gradient"], 0)
    maximum = direction.norm(dim=-1).max()
    return direction / maximum.clamp_min(1e-300)


def deterministic_directions(shape, dtype=torch.float64):
    """No RNG; three fixed globally normalized directions."""
    i = torch.arange(torch.tensor(shape).prod().item(), dtype=dtype).reshape(shape)
    return [
        x / x.norm()
        for x in [
            torch.where(i.remainder(2) == 0, torch.ones_like(i), -torch.ones_like(i)),
            torch.sin(i + 1),
            torch.cos((i + 1) * 1.7),
        ]
    ]


def directional_checks(function, point, gradient, epsilons):
    rows = []
    for j, d in enumerate(deterministic_directions(point.shape, point.dtype)):
        derivative = float((gradient * d).sum())
        for eps in epsilons:
            fd = float((function(point + eps * d) - function(point - eps * d)) / (2 * eps))
            error = abs(fd - derivative)
            rows.append(
                dict(
                    direction=j,
                    epsilon=eps,
                    autograd=derivative,
                    finite_difference=fd,
                    absolute_error=error,
                    relative_error=error / max(abs(fd), abs(derivative), 1e-10),
                    passed=error <= 1e-10 + 1e-5 * max(abs(fd), abs(derivative)),
                )
            )
    return rows
