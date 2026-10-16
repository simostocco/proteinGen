import copy
import pytest,torch
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.sequence_only_actor import SequenceOnlyActor
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.models.batched_bc import collate,encode_batch,losses
from protein_joint_sac.data.engineering_fixtures import records
from protein_joint_sac.data.certified_unified import canonical_teacher_prefix
from protein_joint_sac.contracts import AA

def fixtures():
 r=records()[0];return [canonical_teacher_prefix(r['sequence'],r['coordinates'],t,r['source_sha']) for t in [0,1,2,3,12]]
@pytest.mark.parametrize('mode,chem',[('sequence',False),('joint',False),('joint',True)])
def test_reference_logits_losses_gradients_and_optimizer(mode,chem):
 torch.manual_seed(43);c=ModelConfig(width=16,heads=2,layers=1,ffn_mult=4,chemistry=chem);a=JointActor(c) if mode=='joint' else SequenceOnlyActor(c);b=copy.deepcopy(a);targets=fixtures();states=[q.state for q in targets];batch=collate(states,'cpu');labels=torch.tensor([AA.index(q.next_action.residue) for q in targets]);g=torch.stack([q.next_action.geometry for q in targets]) if mode=='joint' else None
 h=encode_batch(b,batch);ref=torch.stack([a.encode(s) for s in states]);torch.testing.assert_close(h,ref,atol=2e-6,rtol=2e-5)
 old=[]
 for q in targets:
  hs=a.encode(q.state);aa=torch.nn.functional.cross_entropy(a.aa_head(hs)[None],torch.tensor([AA.index(q.next_action.residue)]));geo=-a.geometry_distribution(q.state,q.next_action.residue,hs).log_prob(q.next_action.geometry)/max(1,sum(q.next_action.mask)) if mode=='joint' else aa*0;old.append(aa+geo)
 old=torch.stack(old).mean();aa,geo,_=losses(b,batch,labels,g);new=(aa+geo).mean();torch.testing.assert_close(old,new,atol=2e-6,rtol=2e-5);old.backward();new.backward()
 for p,q in zip(a.parameters(),b.parameters()):
  if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4)
 oa=torch.optim.AdamW(a.parameters(),lr=3e-4);ob=torch.optim.AdamW(b.parameters(),lr=3e-4);oa.step();ob.step()
 for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=2e-5,rtol=1e-3)

def test_padding_suffix_nan_and_other_example_invariance():
 torch.manual_seed(2);m=JointActor(ModelConfig(width=16,heads=2,layers=1));ts=fixtures();s=ts[2].state;a=collate([s],'cpu');b=collate([s,ts[-1].state],'cpu');base=encode_batch(m,a)[0];torch.testing.assert_close(base,encode_batch(m,b)[0],atol=2e-6,rtol=2e-5)
 b['x'][0,s.t:]=float('nan');b['ids'][0,s.t:]=19;torch.testing.assert_close(base,encode_batch(m,b)[0],atol=2e-6,rtol=2e-5)

def test_activation_checkpoint_gradient_parity():
 torch.manual_seed(3);a=JointActor(ModelConfig(width=16,heads=2,layers=2));b=copy.deepcopy(a);states=[q.state for q in fixtures()];x=collate(states,'cpu');encode_batch(a,x).sum().backward();encode_batch(b,x,True).sum().backward()
 for p,q in zip(a.parameters(),b.parameters()):
  if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)
