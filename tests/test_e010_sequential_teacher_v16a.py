import numpy as np
import pytest
import torch
from protein_distance_diffusion.training import e010_sequential_teacher_v16a as t
from scripts.recover_e010_conditioning_v9b import synthetic

@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_projection_and_inward_guard():
    rng=np.random.default_rng(123)
    a=rng.normal(size=(5000,3)); a*=.1/np.linalg.norm(a,axis=1)[:,None]
    a[:100]*=2
    p,info=t.project(a)
    assert (np.linalg.norm(p,axis=1)<=.1).all()
    safe=np.linalg.norm(a,axis=1)<=.1
    assert np.array_equal(a[safe],p[safe])
    assert info['max_inward_adjustment_angstrom']<1e-15
    assert np.array_equal(t.project(p)[0],p)


def test_one_step_zero_and_variable_count():
    b=synthetic(12); c=t.Context(b); s=t.OneStep(c,b['pg'])
    assert s.n==36
    assert torch.equal(s.point(np.zeros(s.n))[2],b['pg'])
    assert not s.eligible[:,0].any() and not s.eligible[:,-1].any()
    p=s.point(np.ones(s.n)*.2)[2]
    assert torch.equal(p[:,[0,-1]],b['pg'][:,[0,-1]])
    assert c.safety(b['pg'])['passes']


def test_derivatives_privileged_target_and_reference_constraints():
    b=synthetic(12); result=t.validate(b)
    assert len(result['directional_checks'])==9
    c=t.Context(b); x=b['pg']+.001*torch.sin(b['pg'])
    expected=torch.cat((t.metrics.normalized_constraints(c.values(x),c.baseline),c.quartets(x)))
    assert torch.equal(c.constraints(x),expected)
    b2={**b,'target':b['target']+.001}
    assert not torch.equal(c.values(x)['local'],t.Context(b2).values(x)['local'])


def test_sequential_best_feasible_and_frames():
    b=synthetic(12); c=t.Context(b)
    a,log=t.solve_step(c,b['pg'])
    p=b['pg']+torch.from_numpy(a)
    assert c.safety(p)['passes']
    assert c.values(p)['local']<=c.baseline['local']
    s=t.OneStep(c,b['pg']); u=torch.einsum('bnji,bnj->bni',s.frame,torch.from_numpy(a))
    reconstructed=torch.einsum('bnij,bnj->bni',s.frame,u)
    assert torch.allclose(reconstructed,torch.from_numpy(a),atol=1e-15,rtol=1e-13)
    assert log['best_normalized_local']<=1
    assert (np.linalg.norm(a,axis=-1)<=.1).all()
    a2,log2=t.solve_step(c,b['pg'])
    assert np.array_equal(a,a2)
    assert log['iterations']==log2['iterations']
    second=t.OneStep(c,p)
    assert torch.equal(second.current,p)
    assert not torch.equal(second.frame,s.frame)


def test_noop_and_telemetry_failure_preserves_saved_action(monkeypatch):
    b=synthetic(12); b['target']=b['pg'].clone()
    # A zero local denominator is outside the historical panel; use mocked solver on normal context.
    b=synthetic(12); c=t.Context(b); saved=[]
    from scipy.optimize import OptimizeResult
    def zero(fun,x,**kw):
        fun(x)
        return OptimizeResult(x=x,nit=0,success=False,status=9,message='cap',nfev=1,njev=0)
    monkeypatch.setattr(t,'minimize',zero)
    a,log=t.solve_step(c,b['pg'],lambda a:saved.append(a.copy()))
    assert np.array_equal(a,np.zeros_like(a)) and log['zero_action']
    assert len(saved)==1
    assert not log['optional_solver_multipliers_available']
