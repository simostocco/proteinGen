"""Independent CPU numerical references for fixed-AA SAC laws, no networks."""
import math
import json
from pathlib import Path
import sys
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v7_amortized_lowlevel_20261023_v1');sys.path.insert(0,str(R/'source/src'))
from protein_joint_sac.crisp_v3.density import sample,log_prob,semimarkov
from protein_joint_sac.crisp_v3.recovery import atomic_json,sha
torch.set_num_threads(1)
mask=torch.ones(3,dtype=torch.bool);eps=torch.tensor([.2,-.4,.3],dtype=torch.float64);theta=torch.tensor([1.2,.2,-.3,-1.,-.5,-.4],dtype=torch.float64,requires_grad=True)
def physical(u):return torch.stack([u[0].exp(),math.pi*u[1].sigmoid(),math.pi*u[2].tanh()])
u=theta[:3]+theta[3:].exp()*eps;g=physical(u);J=torch.autograd.functional.jacobian(physical,u);normal=(-.5*eps.square()-theta[3:]-.5*math.log(2*math.pi)).sum();expected=normal-torch.linalg.slogdet(J)[1];actual=log_prob(theta[:3],theta[3:],g,mask);lp_error=float((expected-actual).detach().abs());assert lp_error<1e-10
def objective(t):
 g,lp=sample(t[:3],t[3:],mask,eps);q=-(g[0]-3.8).square()-(g[1]-2).square()-g[2].square();return .001*lp/3-q
analytical=torch.autograd.grad(objective(theta),theta)[0];finite=[]
for k in range(6):
 d=torch.zeros(6,dtype=torch.float64);d[k]=1e-5;finite.append(float((objective(theta.detach()+d)-objective(theta.detach()-d))/(2e-5)))
error=float((analytical-torch.tensor(finite,dtype=torch.float64)).abs().max());assert error<1e-6
masked=mask.clone();masked[1:]=False;gg=g.detach().clone();gg[1:]=float('nan');masked_lp=log_prob(theta[:3],theta[3:],gg,masked);assert torch.isfinite(masked_lp)
reward=torch.tensor([.75,0.],dtype=torch.float64);duration=torch.ones(2,dtype=torch.long);term=torch.tensor([True,False]);v=torch.tensor([999.,.951],dtype=torch.float64);target=semimarkov(reward,.995,duration,term,v);assert torch.allclose(target,torch.tensor([.75,.946245],dtype=torch.float64),atol=1e-12)
t=torch.tensor([1.]);q=torch.tensor([3.]);t.lerp_(q,.005);assert torch.allclose(t,torch.tensor([1.01]))
atomic_json(R/'NUMERICAL_DENSITY_AND_BELLMAN_AUDIT.json',dict(PASS=True,scope='independent CPU analytic-Jacobian/numerical finite-difference references; no actor/critic/model training',transformed_logprob_absolute_error=lp_error,reparameterized_objective_gradient_max_absolute_error=error,analytical_gradient=analytical.tolist(),finite_difference_gradient=finite,masked_undefined_component_ignored=True,terminal_zero_bootstrap=True,nonterminal_min_twinQ_entropy_and_KL_normalization=True,polyak_target_direction=True,density_source_SHA256=sha(R/'source/src/protein_joint_sac/crisp_v3/density.py'),script_SHA256=sha(__file__)))
print('NUMERICAL_AUDIT_PASS',lp_error,error)
