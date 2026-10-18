import copy,math
import pytest
import torch
from torch import nn
from protein_joint_sac.models.geometry_inference import corrected_action,legacy_action
from protein_joint_sac.models.hybrid_distributions import GeometryDistribution
from protein_joint_sac.geometry.ca_builder import next_ca,dihedral
from protein_joint_sac.rl.group_clipping import groups,combine

@pytest.mark.parametrize('z',[-1e6,-100.,-10.,-9.,0.,9.,10.,100.,1e6])
def test_torsion_support_inverse_and_handedness(z):
 mu=torch.tensor([math.log(3.8),0.,z]);ls=torch.zeros(3);eps=torch.zeros(3)
 g,info=corrected_action(mu,ls,(True,True,True),eps)
 assert g.dtype==torch.float64 and -math.pi<float(g[2])<math.pi
 assert math.isfinite(info['represented_action_log_density']) and math.isfinite(info['latent_action_log_density'])
 assert z==0 or math.copysign(1,float(g[2]))==math.copysign(1,z)
 ca=torch.tensor([[0.,0.,0.],[3.8,0.,0.],[3.8,3.8,0.]],dtype=torch.float64)
 point=next_ca(ca,g)
 assert torch.isfinite(point).all()
 recovered=dihedral(ca[0],ca[1],ca[2],point)
 assert abs(float(recovered-g[2]))<1e-12

@pytest.mark.parametrize('z',[-10000.,10000.])
def test_angle_endpoints_are_explicit_and_invertible(z):
 g,info=corrected_action(torch.tensor([math.log(3.8),z,0.]),torch.zeros(3),(True,True,True),torch.zeros(3))
 assert 0<float(g[1])<math.pi and math.isfinite(info['represented_action_log_density'])
 assert info['adjusted_components'][1]

def test_legacy_and_interior_density_unchanged():
 mu=torch.tensor([1.3,.2,.4]);ls=torch.tensor([-.2,-.4,-.5]);mask=(True,True,True)
 torch.manual_seed(42);original=GeometryDistribution(mu,ls,mask).rsample()[0]
 torch.manual_seed(42);eps=torch.randn(3)
 assert torch.equal(original,legacy_action(mu,ls,mask,eps))
 g,info=corrected_action(mu,ls,mask,eps)
 d=GeometryDistribution(mu.double(),ls.double(),mask)
 assert torch.equal(g,d.transform(mu.double()+ls.double().exp()*eps.double()))
 assert not any(info['adjusted_components'])
 assert info['latent_action_log_density']==pytest.approx(info['represented_action_log_density'],abs=1e-10)
 # Saturated legacy behavior is retained, not retroactively repaired.
 legacy=legacy_action(torch.tensor([1.,0.,10.]),torch.zeros(3),mask,torch.zeros(3))
 assert float(legacy.double()[2])>math.pi

def test_rigid_invariance_and_no_sign_flip():
 ca=torch.tensor([[0.,0.,0.],[3.8,0.,0.],[3.8,3.8,0.]],dtype=torch.float64)
 g,_=corrected_action(torch.tensor([1.3,.3,-.7]),torch.zeros(3),(True,True,True),torch.zeros(3))
 Q=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64);shift=torch.tensor([2.,4.,-7.])
 assert torch.allclose(next_ca(ca@Q.T+shift,g),next_ca(ca,g)@Q.T+shift,atol=1e-12)

def test_invalid_distance_not_fabricated():
 with pytest.raises(FloatingPointError):corrected_action(torch.tensor([1000.,0.,0.]),torch.zeros(3),(True,True,True),torch.zeros(3))

def toy():
 m=nn.Module();m.encoder=nn.Linear(2,2);m.aa_head=nn.Linear(2,2);m.geom=nn.Linear(2,3);return m

def assign(m,G=1.):
 aa={n:(torch.ones_like(p) if not n.startswith('geom.') else None) for n,p in m.named_parameters()}
 for n,p in m.named_parameters():p.grad=None if n.startswith('aa_head.') else torch.full_like(p,.5 if n.startswith('encoder.') else G)
 return aa

def test_unique_groups_and_extreme_geometry_independence():
 m=toy();gs,names=groups(m);assert sum(len(v) for v in gs.values())==len(list(m.parameters()))
 first=combine(m,assign(m,1.),{'S':2.,'A':1.,'G':3.});a=[p.grad.clone() for n,p in gs['A']]
 second=combine(m,assign(m,1e6),{'S':2.,'A':1.,'G':3.})
 assert all(torch.equal(x,p.grad) for x,(n,p) in zip(a,gs['A']))
 assert second['geometry_to_AA_shared_ratio']<=.5+1e-7
 assert second['old_whole_model_norm1_scale']<first['old_whole_model_norm1_scale']
 assert second['group_after']['G']==pytest.approx(3.,rel=1e-6)
 assert second['group_scale']['A']==first['group_scale']['A']

def test_nonfinite_and_unclassified_or_alias_stop():
 m=toy();aa=assign(m);m.geom.weight.grad[0,0]=float('nan')
 with pytest.raises(FloatingPointError):combine(m,aa,{'S':2.,'A':2.,'G':2.})
 m=toy();m.extra=nn.Parameter(torch.ones(1))
 with pytest.raises(ValueError):groups(m)
 m=toy();m.alias=m.encoder
 with pytest.raises(ValueError):groups(m)

def test_adamw_state_restart_after_group_clipping():
 torch.manual_seed(12);m=toy();opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
 combine(m,assign(m),{'S':2.,'A':1.,'G':3.});opt.step()
 state=copy.deepcopy(m.state_dict());os=copy.deepcopy(opt.state_dict())
 combine(m,assign(m,1e4),{'S':2.,'A':1.,'G':3.});opt.step();expected=copy.deepcopy(m.state_dict())
 m.load_state_dict(state);opt.load_state_dict(os);combine(m,assign(m,1e4),{'S':2.,'A':1.,'G':3.});opt.step()
 assert all(torch.equal(v,expected[n]) for n,v in m.state_dict().items())
 assert all(torch.isfinite(p).all() for p in m.parameters())


def test_reference_jacobian_matches_actual_transform_derivative():
 d=GeometryDistribution(torch.zeros(3,dtype=torch.float64),torch.zeros(3,dtype=torch.float64),(True,True,True))
 for z in [-5.,0.,5.]:
  u=torch.tensor([.5,z,z],dtype=torch.float64,requires_grad=True)
  J=torch.autograd.functional.jacobian(d.transform,u)
  assert torch.allclose(J.diag().abs().log(),d.log_jacobian(u),atol=1e-10,rtol=1e-10)
