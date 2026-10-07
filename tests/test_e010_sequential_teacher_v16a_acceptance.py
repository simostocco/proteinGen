"""Acceptance/geometry regressions independent of SLSQP success telemetry."""
import numpy as np
import torch
from scipy.optimize import OptimizeResult
from protein_distance_diffusion.training import e010_sequential_teacher_v16a as t
from scripts.recover_e010_conditioning_v9b import synthetic


def test_ball_jacobian_and_direct_mapping():
    torch.set_num_threads(1)
    b=synthetic(12); s=t.OneStep(t.Context(b),b['pg'])
    x=np.sin(np.arange(s.n))*.2
    d=np.cos(np.arange(s.n)); d/=np.linalg.norm(d)
    eps=1e-6
    assert np.allclose(s.ball_jac(x)@d,(s.ball(x+eps*d)-s.ball(x-eps*d))/(2*eps),atol=1e-10)
    assert torch.equal(s.point(x)[2],b['pg']+t.S_MAX*torch.from_numpy(x).reshape(s.shape)*s.eligible[...,None])
    changed={**b,'target':b['target']+.01*torch.sin(torch.arange(12,dtype=torch.float64))[None,:,None]}
    assert float(t.Context(changed).baseline['local']) != float(t.Context(b).baseline['local'])


def test_unsafe_best_trial_rejected_and_endpoint_ownership(monkeypatch):
    torch.set_num_threads(1)
    b=synthetic(12); context=t.Context(b); prior=context.quartets.q0.clone()
    s=t.OneStep(context,b['pg']); x=np.zeros(s.n)
    # Trials may be infeasible even if solver claims success; zero stays available.
    def false_success(fun,x,**kw):
        fun(x)
        return OptimizeResult(x=np.full(s.n,np.nan),nit=1,nfev=1,njev=1,
                              success=True,status=0,message='mocked invalid output')
    monkeypatch.setattr(t,'minimize',false_success)
    a,log=t.solve_step(context,b['pg'])
    assert (a==0).all() and log['zero_action'] and log['success']
    assert torch.equal(context.quartets.q0,prior)
    assert context.safety(b['pg']+torch.from_numpy(a))['passes']


def test_fixed_state_hashes_and_zero_updates():
    b=synthetic(12); original=b['pg'].clone()
    for _ in range(16):
        b['pg']=b['pg']+torch.zeros_like(b['pg'])
        assert t.digest(b['pg'].numpy())==t.digest(original.numpy())
    assert torch.equal(b['pg'],original)
