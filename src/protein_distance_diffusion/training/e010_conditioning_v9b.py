"""Fixed-state physical KKT analysis. Never optimizes coordinates or calls a model."""

import numpy as np
import torch
from scipy.optimize import nnls

from . import e010_no_new_inversion_v9 as v9
from .e010_precision_kkt_v6 import correction_trajectory

S = 0.04


def spectrum(a):
    a = np.asarray(a, dtype=np.float64)
    sv = np.linalg.svd(a, compute_uv=False)
    threshold = max(a.shape, default=0) * np.finfo(float).eps * (sv[0] if len(sv) else 0)
    good = sv[sv > threshold]
    norms = np.linalg.norm(a, axis=1)
    unit = a / np.maximum(norms[:, None], 1e-300)
    gram = unit @ unit.T
    near = np.argwhere(np.triu(np.abs(gram) > 1 - 1e-8, 1))
    return dict(
        rows=len(a),
        rank=len(good),
        singular_values=sv.tolist(),
        largest=float(good[0]) if len(good) else 0.0,
        smallest_nonzero=float(good[-1]) if len(good) else 0.0,
        condition=float(good[0] / good[-1]) if len(good) else None,
        rank_threshold=float(threshold),
        nearly_parallel_pairs=near.tolist(),
        row_norm_min=float(norms.min()) if len(norms) else 0.0,
        row_norm_max=float(norms.max()) if len(norms) else 0.0,
    )


def cone_projection(g, a):
    """Project -g onto A d <= 0 via nonnegative least-squares dual. No coordinate steps."""
    g = np.asarray(g, float).reshape(-1)
    a = np.asarray(a, float).reshape(-1, len(g))
    if not len(a):
        return -g, np.zeros(0)
    norms = np.linalg.norm(a, axis=1)
    scaled = a / np.maximum(norms[:, None], 1e-300)
    mu_scaled, _ = nnls(scaled.T, -g, maxiter=max(1000, 10 * len(a)))
    mu = mu_scaled / np.maximum(norms, 1e-300)
    return -(g + a.T @ mu), mu


def radial(v):
    norms = np.linalg.norm(v, axis=-1)
    a = 1 / np.sqrt(1 + norms**2 / S**2)
    return dict(v_norm=norms, saturation=norms * a / S, tangential=a, radial=a**3, condition=1 / a**2)


def physical(b, delta):
    o = v9.Oracle(b)
    tr = correction_trajectory(o.b["pg"], o.b["mask"], delta)
    vals = v9.v8.terms(tr["prediction"], o.b, o.frozen)
    f = vals["local"] / o.baseline["local"]
    c = torch.cat([v9.v8.normalized_constraints(vals, o.baseline), o.quartets(tr["prediction"])])
    return f, c, tr, o


def physical_jacobians(b, delta):
    d = delta.detach().clone().requires_grad_()
    f, c, tr, o = physical(b, d)
    g = torch.autograd.grad(f, d, retain_graph=True)[0].detach().numpy().reshape(-1)
    j = torch.stack([torch.autograd.grad(ci, d, retain_graph=True)[0].reshape(-1) for ci in c]).detach().numpy()
    return d, f, c, tr, o, g, j


def active_system(delta, eligible, c, j, slack=1e-5, boundary=1e-4):
    """Normalized inequalities plus physical ball normals; all use c<=0."""
    d = np.asarray(delta).reshape(-1, 3)
    valid = np.asarray(eligible).reshape(-1)
    norm = np.linalg.norm(d, axis=-1)
    ids = np.flatnonzero(np.asarray(c) >= -slack)
    balls = np.flatnonzero(valid & (norm >= S * (1 - boundary)))
    a = [j[i] * S for i in ids]
    for i in balls:
        row = np.zeros_like(d)
        row[i] = d[i] / max(norm[i], 1e-300)
        a.append(row.reshape(-1))
    return np.asarray(a).reshape(-1, j.shape[1]), ids, balls


def components(delta, eligible, g, j, c, mu, boundary=1e-4):
    d = np.asarray(delta).reshape(-1, 3)
    valid = np.asarray(eligible).reshape(-1)
    grad = np.asarray(g).reshape(-1, 3)
    lag = (g + j.T @ mu).reshape(-1, 3)
    norms = np.linalg.norm(d, axis=-1)
    unit = d / np.maximum(norms[:, None], 1e-300)
    active = valid & (norms >= S * (1 - boundary))
    ball_mu = np.maximum(-np.sum(lag * unit, axis=-1) * S, 0) * active
    residual = lag + ball_mu[:, None] * unit / S
    residual *= valid[:, None]
    comp = np.abs(mu * c)
    ball_comp = ball_mu * np.abs(norms / S - 1)
    return dict(
        physical_normalized_stationarity_max=float(np.linalg.norm(residual * S, axis=-1).max()),
        physical_normalized_stationarity_l2=float(np.linalg.norm(residual * S)),
        normalized_gradient_norm=float(np.linalg.norm(grad * S)),
        lagrangian_gradient_norm=float(np.linalg.norm(lag)),
        residual=residual,
        ball_multipliers=ball_mu,
        normalized_primal=max(0.0, float(np.max(c)), float(np.max(norms / S - 1))),
        normalized_complementarity=max(float(comp.max()), float(ball_comp.max())),
        normalized_dual=max(0.0, float(-mu.min()), float(-ball_mu.min())),
        normalized_tangential_max=float(
            np.linalg.norm((lag - np.sum(lag * unit, axis=-1)[:, None] * unit) * S * valid[:, None], axis=-1).max()
        ),
        normalized_feasible_radial_max=float((np.maximum(np.sum(lag * unit, axis=-1), 0) * S * active).max()),
    )


def project_balls(delta):
    size = np.linalg.norm(delta, axis=-1, keepdims=True)
    limit = np.nextafter(S, 0.0)
    return delta * np.minimum(1.0, limit / np.maximum(size, 1e-300))


def finite_checks(b, delta, g, j, mu, indices, epsilons, phases, atol=1e-8, rtol=1e-5):
    records = []
    for phase in phases:
        direction = np.sin(np.arange(delta.numel()) + 1 + phase)
        direction /= np.linalg.norm(direction)
        shape = tuple(delta.shape)
        predicted = np.r_[g @ direction, j[indices] @ direction, (g + j.T @ mu) @ direction]
        for eps in epsilons:
            plus = physical(b, delta.detach() + torch.from_numpy((eps * direction).reshape(shape)))[:2]
            minus = physical(b, delta.detach() - torch.from_numpy((eps * direction).reshape(shape)))[:2]
            fp, cp = [x.detach().numpy() for x in plus]
            fm, cm = [x.detach().numpy() for x in minus]
            fd = np.r_[
                (fp - fm) / (2 * eps), (cp[indices] - cm[indices]) / (2 * eps), ((fp - fm) + mu @ (cp - cm)) / (2 * eps)
            ]
            error = np.abs(fd - predicted)
            records.append(
                dict(
                    phase=phase,
                    epsilon_angstrom=eps,
                    maximum_absolute_error=float(error.max()),
                    maximum_relative_error=float((error / np.maximum(np.abs(predicted), atol)).max()),
                    maximum_scaled_error=float((error / (atol + rtol * np.abs(predicted))).max()),
                    passed=bool((error <= atol + rtol * np.abs(predicted)).all()),
                )
            )
    return records
