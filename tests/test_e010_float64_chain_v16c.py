import numpy as np
import pytest
import torch
from protein_distance_diffusion.training import e010_sequential_teacher_v16a as old
from protein_distance_diffusion.training import e010_sequential_teacher_v16c as fixed
from protein_distance_diffusion.training import e010_derivative_resolution_v16b as fd
from scripts.recover_e010_conditioning_v9b import synthetic

@pytest.fixture(autouse=True)
def forbid_optimizers(monkeypatch):
    def blocked(*a,**k): raise AssertionError('No optimization in V16C')
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize,'minimize',blocked)
    monkeypatch.setattr(old,'minimize',blocked)
    monkeypatch.setattr(fixed,'minimize',blocked)
    torch.set_num_threads(1)


def test_scale_mask_and_value_parity():
    b=synthetic(12)
    a=old.OneStep(old.Context(b),b['pg']); s=fixed.OneStep(fixed.Context(b),b['pg'])
    assert s.scale.dtype==torch.float64 and s.scale.item()==.1
    assert s.eligible.dtype==torch.bool and torch.equal(s.eligible,a.eligible)
    assert torch.equal(s.eligibility_arithmetic,s.eligible.double())
    x=fd.state(12)
    assert s.fun(x)==a.fun(x) and np.array_equal(s.cfun(x),a.cfun(x))
    assert torch.equal(s.point(x)[2],a.point(x)[2])


def test_strict_trace_detects_old_and_passes_fixed():
    b=synthetic(12); x=fd.state(12)
    with fixed.Float64Trace(strict=False) as historic_trace:
        old.OneStep(old.Context(b),b['pg']).cjac(x)
    assert historic_trace.violations
    with fixed.Float64Trace() as trace:
        s=fixed.OneStep(fixed.Context(b),b['pg']); s.jac(x); s.cjac(x); s.ball(x); s.ball_jac(x)
        s.context.safety(s.point(x)[2].detach())
    assert not trace.violations
    assert torch.get_default_dtype()==torch.float32


def test_all_three_analytic_directions_and_primary_fd():
    b=synthetic(32); s=fixed.OneStep(fixed.Context(b),b['pg']); x=fd.state(32)
    for phase in fd.PHASES:
        d=fd.direction(32,phase)
        analytic=np.r_[s.jac(x)@d,s.cjac(x)@d]
        assert np.allclose(analytic,fd.direct_analytic(s,x,d),atol=1e-12,rtol=1e-10)
        center=lambda q:np.r_[s.fun(q),s.cfun(q)]
        derivative,_,_=fd.centered(center,x,d,1e-4)
        assert fd.errors(derivative,analytic)['passes'].all()
    assert fixed.validate(b)['baseline_feasible']


def test_solver_and_science_unchanged():
    assert fixed.SETTINGS==old.SETTINGS
    assert fixed.S_MAX==old.S_MAX and fixed.K_MAX==old.K_MAX
    assert fixed.MATERIALITY==old.MATERIALITY
    assert fixed.QuartetConstraints is old.QuartetConstraints
    assert fixed.metrics is old.metrics
    assert fixed.project(np.array([[.11,0.,0.]]))[0].dtype==np.float64
