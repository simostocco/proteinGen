import torch,pytest
from protein_joint_sac.data.certified_unified import canonical_teacher_prefix,canonical_native_prefix
from protein_joint_sac.contracts import BuildAction
from protein_joint_sac.geometry.ca_builder import replay,append,mask_for_step
from protein_joint_sac.geometry.local_frames import features

def native():
 seq='GPCAV'*4
 acts=tuple(BuildAction(a,torch.tensor([3.8 if i else 0.,1.9 if i>=2 else 0.,.3 if i>=3 else 0.],dtype=torch.float64),mask_for_step(i)) for i,a in enumerate(seq))
 x=replay(20,acts).ca
 rot=torch.linalg.qr(torch.tensor([[.5,.7,.8],[1.,.1,.3],[.3,.6,.9]],dtype=torch.float64))[0]
 if torch.linalg.det(rot)<0:rot[:,0]*=-1
 return seq,x@rot+torch.tensor([100.,-30.,80.])
@pytest.mark.parametrize('t',[0,1,2,3,12,19])
def test_native_bootstrap_and_targets_replay_consistently(t):
 seq,x=native();ex=canonical_teacher_prefix(seq,x,t,'a'*64)
 nxt=append(ex.state,ex.next_action)
 expected=canonical_teacher_prefix(seq,x,t+1,'a'*64).state if t<19 else None
 if expected is not None:torch.testing.assert_close(nxt.ca,expected.ca,atol=1e-10,rtol=1e-10)
@pytest.mark.parametrize('t',[1,2,3,12])
def test_suffix_coordinates_and_tokens_cannot_change_encoder_prefix(t):
 seq,x=native();before=canonical_native_prefix(seq,x,t)
 new=x.clone();new[t:]=torch.randn_like(new[t:])*1000
 after=canonical_native_prefix(seq[:t]+'W'*(20-t),new,t)
 assert before.residues==after.residues
 for k,v in features(before).items():
  other=features(after)[k]
  if isinstance(v,torch.Tensor):torch.testing.assert_close(v,other,atol=0,rtol=0)

def test_actor_prefix_does_not_read_even_invalid_future_labels():
 seq,x=native();before=canonical_native_prefix(seq,x,3);x[3:]=float('nan')
 after=canonical_native_prefix(seq[:3]+'X'*17,x,3)
 torch.testing.assert_close(before.ca,after.ca,atol=0,rtol=0)
