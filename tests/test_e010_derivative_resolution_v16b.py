import numpy as np
import torch
from protein_distance_diffusion.training import e010_derivative_resolution_v16b as fd
from scripts.recover_e010_conditioning_v9b import synthetic


def test_fixed_state_and_directions():
    n=500
    assert np.array_equal(fd.state(n),np.sin(np.arange(1500)*.7)*.03)
    for phase in fd.PHASES:
        d=np.cos(np.arange(1500)+phase); d/=np.linalg.norm(d)
        assert np.array_equal(fd.direction(n,phase),d)


def test_centered_and_richardson():
    x=np.array([.7]); d=np.array([1.])
    f=lambda v:v**3
    a,_,_=fd.centered(f,x,d,.01)
    b,_,_=fd.centered(f,x,d,.005)
    assert np.allclose((4*b-a)/3,3*x*x,atol=1e-13)


def test_physical_scale_and_analytic_paths():
    torch.set_num_threads(1)
    b=synthetic(12); s=fd.historical.OneStep(fd.historical.Context(b),b['pg'])
    x,d=fd.state(12),fd.direction(12,.7)
    h=1e-6
    p=fd.physical(s,x,d,h)
    direct=.1*h*d.reshape(s.shape)*s.eligible.numpy()[...,None]
    assert p['intended_max_angstrom']==float(np.linalg.norm(direct,axis=-1).max())
    exact=np.r_[s.jac(x)@d,s.cjac(x)@d]
    independent=fd.direct_analytic(s,x,d)
    assert np.allclose(exact[:3],independent[:3],atol=1e-12,rtol=1e-10)
    assert not np.allclose(exact[3:],independent[3:],atol=1e-12,rtol=1e-10)
    assert np.allclose(exact[3:],independent[3:]*(float(np.float32(.1))/.1),atol=1e-12,rtol=1e-10)


def test_no_optimizer_and_no_mutation(monkeypatch):
    torch.set_num_threads(1)
    def forbidden(*a,**k): raise AssertionError('Optimizer forbidden')
    monkeypatch.setattr(fd.historical,'minimize',forbidden)
    b=synthetic(12); copies={k:v.clone() for k,v in b.items()}
    s=fd.historical.OneStep(fd.historical.Context(b),b['pg'])
    row,arrays=fd.audit_direction(s,fd.state(12),fd.direction(12,0.),0.)
    assert len(row['samples'])==13 and len(row['historical_checks'])==3
    assert all(torch.equal(b[k],v) for k,v in copies.items())
    assert row['proposed_future_h']==1e-4
    assert not row['analytic_paths_pass']
    assert row['chain_factor_model_max_error']<1e-12
