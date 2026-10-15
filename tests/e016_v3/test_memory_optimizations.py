import importlib.util,os,copy
from pathlib import Path
import pytest,torch
from protein_joint_sac.contracts import BuildState,BuildAction
from protein_joint_sac.geometry.local_frames import features,reuse_static_prefix
from protein_joint_sac.geometry.ca_builder import replay,mask_for_step
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.build_critic import TwinCritics
from protein_joint_sac.rl.hybrid_sac_build import HybridSAC
P=Path('/mnt/d/Simone/proteinGen_data/e016_cuda_n64_pdb_certification_20261014_v1/source/src/protein_joint_sac/geometry/local_frames.py')
# Load the actual old function body with its existing relative dependencies.
ns={'__name__':'protein_joint_sac.geometry._reference_loop','__package__':'protein_joint_sac.geometry'};exec(compile(P.read_text(),str(P),'exec'),ns);old=ns['features']
def state(t=12):
 acts=tuple(BuildAction('GPCAV'[i%5],torch.tensor([3.8 if i else 0.,1.9 if i>=2 else 0.,.4 if i>=3 else 0.],dtype=torch.float64),mask_for_step(i)) for i in range(t))
 return replay(64,acts)
@pytest.mark.parametrize('t',[0,1,2,3,4,12,63])
def test_vectorized_exact_reference(t):
 s=state(t);a=features(s);b=old(s)
 for k in a:torch.testing.assert_close(a[k],b[k],atol=2e-12,rtol=2e-12)
def test_masks_degeneracy_and_coordinate_gradients():
 for s in [state(),BuildState(tuple('GPCAV'),torch.zeros(5,3,dtype=torch.float64),20),BuildState(state(5).residues,state(5).ca,20,valid=torch.tensor([True,False,True,True,True]))]:
  a=features(s);b=old(s)
  for k in a:torch.testing.assert_close(a[k],b[k],atol=2e-12,rtol=2e-12)
 s=state();x=s.ca.clone().requires_grad_();s=BuildState(s.residues,x,64)
 a=features(s);b=old(s)
 keys=['trailing_CA_bend_rad','trailing_CA_virtual_torsion_rad','pair_relative_rotation','pair_local_direction']
 ga=torch.autograd.grad(sum(a[k].square().sum() for k in keys),x,retain_graph=True)[0]
 gb=torch.autograd.grad(sum(b[k].square().sum() for k in keys),x)[0]
 torch.testing.assert_close(ga,gb,atol=1e-10,rtol=1e-10)
def test_cache_scoped_and_mutation_rejected():
 s=state();a=features(s)
 with reuse_static_prefix(s):
  assert features(s) is features(s)
  for k in a:torch.testing.assert_close(a[k],features(s)[k])
  s.ca.add_(1)
  with pytest.raises(ValueError):features(s)
 assert features(s) is not features(s)
def test_streamed_cache_preserves_exact_actor_and_candidate_gradients():
 torch.manual_seed(91);actor=JointActor();q=TwinCritics();alg=HybridSAC(actor,q);s=state(4)
 for p in q.parameters():p.requires_grad_(False)
 torch.manual_seed(93);loss=alg.actor_loss(s);loss.backward();expected={n:p.grad.clone() for n,p in actor.named_parameters() if p.grad is not None}
 actor.zero_grad(set_to_none=True);torch.manual_seed(93);value=alg.backward_actor_exact_streamed(s)
 assert abs(value-float(loss.detach()))<2e-6
 for n,p in actor.named_parameters():
  if n in expected:torch.testing.assert_close(p.grad,expected[n],atol=3e-6,rtol=3e-5)
