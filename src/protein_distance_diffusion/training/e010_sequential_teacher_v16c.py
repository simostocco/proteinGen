"""V16C precision-corrected sequential teacher lineage; V16A stays immutable."""
import hashlib
from contextlib import contextmanager
import time
import resource
import numpy as np
import torch
from scipy.optimize import minimize
from . import e010_local_feasibility_v7 as metrics
from .e010_no_new_inversion_v9 import QuartetConstraints
from .e010_phase4d_objective_v2 import freeze_chirality
from ..models.e010_hybrid_local import local_representation

S_MAX = 0.10
K_MAX = 16
MATERIALITY = 1e-6  # Frozen V9B normalized local-MSE loss-sensitivity budget.
SETTINGS = dict(maxiter=1000, ftol=1e-12, disp=False)


@contextmanager
def float64_defaults():
    """Scope PyTorch's det-backward scalar factories to float64 and restore.

    Teacher workers are single-threaded isolated processes. No package or
    persistent process setting changes; explicit state/mask casts remain required.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def project(delta):
    """Radial closed-ball projection, then first feasible uniform inward ULP factor.

    Exact stored NumPy physical norm is authoritative. Already feasible vectors
    remain bitwise unchanged. This only repairs correction-ball arithmetic.
    """
    original = np.asarray(delta, dtype=np.float64)
    if not np.isfinite(original).all():
        raise ValueError('Nonfinite correction')
    out = original.copy().reshape(-1, 3)
    norms = np.linalg.norm(out, axis=1)
    outside = norms > S_MAX
    out[outside] *= (S_MAX / norms[outside])[:, None]
    rounded = out.copy()
    factor = np.ones(len(out))
    ulps = np.zeros(len(out), dtype=np.int64)
    for _ in range(128):
        bad = np.linalg.norm(out, axis=1) > S_MAX
        if not bad.any():
            break
        factor[bad] = np.nextafter(factor[bad], 0.0)
        ulps[bad] += 1
        out[bad] = rounded[bad] * factor[bad, None]
    else:
        raise RuntimeError('Ball numerical guard failed')
    return out.reshape(original.shape), dict(
        radial_projected=int(outside.sum()), inward_guarded=int((ulps > 0).sum()),
        max_inward_factor_ulps=int(ulps.max(initial=0)),
        max_inward_adjustment_angstrom=float(np.linalg.norm(out-rounded, axis=1).max(initial=0)),
        max_total_projection_angstrom=float(np.linalg.norm(out-original.reshape(-1,3), axis=1).max(initial=0)))


class Context:
    def __init__(self, b):
        metrics.require64(b['pg'], b['target'], b['source'])
        self.b = b
        self.frozen = freeze_chirality(b['pg'], b['target'], b['mask'])
        self.baseline = {k: v.detach() for k,v in metrics.terms(b['pg'], b, self.frozen).items()}
        self.quartets = QuartetConstraints(b)

    def values(self, p):
        return metrics.terms(p, self.b, self.frozen)

    def constraints(self, p):
        return torch.cat((metrics.normalized_constraints(self.values(p), self.baseline), self.quartets(p)))

    @torch.no_grad()
    def safety(self, p):
        c = self.constraints(p).numpy()
        q = self.quartets.telemetry(p, 1e-5)
        radius = (p[self.b['mask']]-p[self.b['mask']].mean(0)).norm(dim=-1).square().mean().sqrt()
        ok = bool(np.isfinite(c).all() and (c[:2] <= 1e-8).all() and (c[2:] <= 0).all()
                  and q['new_inversions'] == 0 and q['assessability_lost'] == 0
                  and torch.isfinite(p).all() and radius >= 1e-3)
        return dict(passes=ok, global_constraints=c[:2].tolist(),
                    maximum_extra_constraint=float(c[2:].max()), quartets=q)


class OneStep:
    def __init__(self, context, current):
        self.context, self.current = context, current.detach().clone()
        with torch.no_grad():
            rep = local_representation(self.current, context.b['mask'])
        self.eligible, self.frame = rep['eligible'], rep['frame']
        self.shape = tuple(current.shape)
        self.n = current.numel()
        self.last = None
        metrics.require64(self.current)
        self.scale = self.current.new_tensor(S_MAX)
        self.eligibility_arithmetic = self.eligible.to(dtype=self.current.dtype, device=self.current.device)

    def point(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.last is None or not np.array_equal(x, self.last[0]):
            z = torch.from_numpy(x.copy()).requires_grad_()
            p = self.current + self.scale*z.reshape(self.shape)*self.eligibility_arithmetic[..., None]
            f = self.context.values(p)['local']/self.context.baseline['local']
            c = self.context.constraints(p)
            self.last = (x.copy(), z, p, f, c)
        return self.last

    def fun(self, x):
        return float(self.point(x)[3].detach())

    @float64_defaults()
    def jac(self, x):
        _,z,_,f,_ = self.point(x)
        return torch.autograd.grad(f,z,retain_graph=True)[0].numpy()

    def cfun(self, x):
        return -self.point(x)[4].detach().numpy()

    @float64_defaults()
    def cjac(self, x):
        _,z,p,_,c = self.point(x)
        first = torch.stack([torch.autograd.grad(v,z,retain_graph=True)[0] for v in c[:2]])
        rest = torch.func.jacrev(self.context.quartets)(p.detach()).reshape(len(c)-2,-1)
        chain = (self.scale*self.eligibility_arithmetic[...,None].expand(self.shape)).reshape(-1)
        return -torch.cat((first, rest*chain)).numpy()

    def ball(self, x):
        return 1-np.square(np.asarray(x, dtype=np.float64).reshape(-1,3)).sum(1)

    def ball_jac(self, x):
        z = np.asarray(x, dtype=np.float64).reshape(-1,3)
        j = np.zeros((len(z),len(x)), dtype=np.float64)
        rows = np.arange(len(z))[:,None]
        cols = 3*rows+np.arange(3)[None,:]
        j[rows,cols] = -2*z
        return j


def solve_step(context, current, save_action=None):
    """Zero-start SLSQP; choose best true-feasible projected major/trial iterate."""
    step = OneStep(context,current)
    assert context.safety(current)['passes']
    start_loss = step.fun(np.zeros(step.n))
    best = dict(loss=start_loss, norm=0.0, iteration=0, evaluation=0,
                delta=np.zeros(step.shape), guard=project(np.zeros(step.shape))[1])
    iteration = evaluations = feasible = 0
    max_guard = max_projection = 0.0
    history = []
    started = time.perf_counter()

    def consider(x):
        nonlocal evaluations, feasible, best, max_guard, max_projection
        evaluations += 1
        if not np.isfinite(x).all():
            return
        delta, guard = project(S_MAX*np.asarray(x).reshape(step.shape)*step.eligible.numpy()[...,None])
        max_guard = max(max_guard, guard['max_inward_adjustment_angstrom'])
        max_projection = max(max_projection, guard['max_total_projection_angstrom'])
        p = current+torch.from_numpy(delta)
        with torch.no_grad():
            safety = context.safety(p)
            loss = float(context.values(p)['local']/context.baseline['local'])
        if safety['passes'] and np.isfinite(loss) and loss <= start_loss:
            feasible += 1
            norm = float(np.linalg.norm(delta))
            key = (loss,norm,iteration,evaluations)
            if key < (best['loss'],best['norm'],best['iteration'],best['evaluation']):
                best = dict(loss=loss,norm=norm,iteration=iteration,evaluation=evaluations,delta=delta.copy(),guard=guard)

    def fun(x):
        value = step.fun(x)
        consider(x)  # Includes line-search trial iterates, not only major iterates.
        return value

    def callback(x):
        nonlocal iteration
        iteration += 1
        consider(x)
        history.append(dict(iteration=iteration, best_normalized_local=best['loss']))

    result = minimize(fun,np.zeros(step.n),jac=step.jac,method='SLSQP',
                      constraints=[dict(type='ineq',fun=step.cfun,jac=step.cjac),
                                   dict(type='ineq',fun=step.ball,jac=step.ball_jac)],
                      callback=callback,options=SETTINGS)
    consider(result.x)
    material = start_loss-best['loss'] > MATERIALITY
    delta = best['delta'] if material else np.zeros(step.shape)
    # Save the accepted action BEFORE optional telemetry can fail.
    if save_action is not None:
        save_action(delta)
    return delta, dict(iterations=int(result.nit),success=bool(result.success),status=int(result.status),
                       message=str(result.message),function_evaluations=int(result.nfev),
                       jacobian_evaluations=int(result.njev), best_iteration=best['iteration'],
                       best_evaluation=best['evaluation'],feasible_candidates=feasible,
                       start_normalized_local=start_loss,best_normalized_local=best['loss'],
                       zero_action=not material,accepted_normalized_local=best['loss'] if material else start_loss,
                       maximum_inward_guard_angstrom=max_guard,maximum_total_projection_angstrom=max_projection,
                       selected_projection=best['guard'] if material else project(delta)[1],
                       runtime_seconds=time.perf_counter()-started,
                       peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       history=history, optional_solver_multipliers_available=hasattr(result,'multipliers'))


def validate(b):
    context = Context(b)
    step = OneStep(context,b['pg'])
    x = np.sin(np.arange(step.n)*0.7)*0.03
    rows = []
    for phase in (0.,0.7,1.3):
        d = np.cos(np.arange(step.n)+phase); d /= np.linalg.norm(d)
        jf, jc = float(step.jac(x)@d), step.cjac(x)@d
        for eps in (1e-4,):  # V16B predeclared validation rule; tolerances unchanged.
            fd = (step.fun(x+eps*d)-step.fun(x-eps*d))/(2*eps)
            cd = (step.cfun(x+eps*d)-step.cfun(x-eps*d))/(2*eps)
            error = np.abs(cd-jc)
            assert abs(fd-jf) <= 1e-8+1e-5*abs(jf)
            assert np.all(error <= 1e-8+1e-5*np.abs(jc))
            rows.append(dict(phase=phase,epsilon=eps,objective_error=abs(fd-jf),constraint_max_error=float(error.max())))
    assert context.safety(b['pg'])['passes']
    return dict(variable_count=step.n,quartets=context.quartets.counts, directional_checks=rows,
                baseline_feasible=True,zero_prediction_exact=bool(torch.equal(step.point(np.zeros(step.n))[2],b['pg'])))


class Float64Trace(torch.utils._python_dispatch.TorchDispatchMode):
    """Opt-in operation-wide floating dtype assertion; zero overhead unless enabled.

    Includes nested historical geometry/loss helpers and autograd construction.
    Integers and Boolean masks remain valid. Non-strict mode records old defects.
    """
    def __init__(self, strict=True):
        super().__init__()
        self.strict = strict
        self.operations = {}
        self.violations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        from torch.utils._pytree import tree_leaves
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = str(func)
        record = self.operations.setdefault(name, dict(calls=0, floating_dtypes=[], devices=[]))
        record['calls'] += 1
        for location, values in [('input',(args,kwargs)),('output',out)]:
            for v in tree_leaves(values):
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    dtype,device = str(v.dtype),str(v.device)
                    if dtype not in record['floating_dtypes']: record['floating_dtypes'].append(dtype)
                    if device not in record['devices']: record['devices'].append(device)
                    if v.dtype != torch.float64 or v.device.type != 'cpu':
                        violation=dict(operation=name,location=location,dtype=dtype,device=device,shape=list(v.shape))
                        self.violations.append(violation)
                        if self.strict:
                            raise AssertionError(f'Non-float64 numerical intermediate: {violation}')
        return out
