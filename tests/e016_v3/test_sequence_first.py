import copy,math
import numpy as np
import pytest,torch
from protein_joint_sac.rl.sequence_first import *
from protein_joint_sac.rl.global_geometry import prefix_batch
from protein_joint_sac.models.batched_bc import encode_batch,losses
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.geometry.ca_builder import inverse
from test_global_geometry import fixture,net
def test_coverage_cycles_and_resume():
 c=Coverage();out=[c.position("a",23,"obs") for _ in range(23)]
 assert len(set(out))==23
 s=c.state_dict();expected=[c.position("a",23,"other") for _ in range(30)]
 d=Coverage();d.load_state_dict(s);assert expected==[d.position("a",23,"other") for _ in range(30)]
 assert d.report()['distinct_identity_positions']==23
def test_anchor_positions_and_empty():
 assert anchors(torch.zeros(30,dtype=torch.bool),30)==[]
 a=anchors(torch.ones(80,dtype=torch.bool),80)
 assert all(80-j>=3 for j in a) and any(80-j>=24 for j in a)
def test_energy_score_formula_and_gradients():
 s=torch.tensor([[0.,1.],[2.,3.],[4.,5.],[6.,7.]],requires_grad=True);y=torch.tensor([1.,2.])
 e=energy_score(s,y);manual=(s-y).norm(dim=-1).mean()-sum((s[i]-s[j]).norm() for i in range(4) for j in range(4) if i!=j)/24
 torch.testing.assert_close(e,manual);e.backward();assert s.grad.isfinite().all()
def test_dropout_eval_and_original_equivalence():
 x,ids=fixture();m=net();b=prefix_batch(x[:8],ids[:8],len(x));m.eval()
 set_dropout(m,0);a=encode_batch(m,b);set_dropout(m,.15);torch.testing.assert_close(a,encode_batch(m,b),atol=0,rtol=0)
 m.train();torch.manual_seed(5);a=encode_batch(m,b);torch.manual_seed(5);torch.testing.assert_close(a,encode_batch(m,b),atol=0,rtol=0)
 assert not torch.equal(a,encode_batch(m,b))
def test_augmentation_placeholder_invariance_and_targets():
 x,ids=fixture();m=net().eval();b=prefix_batch(x[:15],ids[:15],len(x));torch.manual_seed(2);b=augment(b,1.)
 assert b['valid'][0,-3:].all() and b['ca_observed'].all() and not b['valid'][0,:-3].any()
 a=encode_batch(m,b);bb=dict(b);bb['x']=b['x'].clone();bb['x'][~bb['valid']]=1e6
 torch.testing.assert_close(a,encode_batch(m,bb),atol=0,rtol=0)
def test_multi_prefix_logits_and_gradients():
 x,ids=fixture();m=net().eval()
 from protein_joint_sac.models.batched_bc import collate
 from protein_joint_sac.contracts import BuildState,AA
 states=[BuildState(tuple(AA[i%20] for i in range(t)),x[:t],len(x)) for t in [0,2,5,12]]
 b=collate(states,'cpu');a=encode_batch(m,b);loss=a.square().mean(1).sum();loss.backward();g=[None if p.grad is None else p.grad.clone() for p in m.parameters()]
 m.zero_grad(set_to_none=True);out=[]
 for s in states:
  h=encode_batch(m,collate([s],'cpu'));out.append(h);h.square().mean().backward()
 torch.testing.assert_close(a,torch.cat(out),atol=1e-6,rtol=1e-5)
 for p,v in zip(m.parameters(),g):
  if v is not None:torch.testing.assert_close(p.grad,v,atol=2e-6,rtol=3e-4)
def test_nonlocal_rotation_and_gradients():
 x,ids=fixture();m=net().eval();b=prefix_batch(x[:10],ids[:10],len(x));y=ids[10:11]
 torch.manual_seed(9);v,c=direct_nonlocal(m,b,y,x[10:11],K=4);v.backward()
 assert c and m.geom[-1].weight.grad.norm()>0 and m.encoder.blocks[0].qkv.weight.grad.norm()>0
 q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64));q[:,0]*=torch.linalg.det(q);delta=torch.tensor([8.,2.,3.])
 bb=prefix_batch(x[:10]@q+delta,ids[:10],len(x));torch.manual_seed(9);w,_=direct_nonlocal(m,bb,y,x[10:11]@q+delta)
 torch.testing.assert_close(v,w,atol=1e-6,rtol=1e-5)
def test_dropout_recompute_rng_parity():
 from protein_joint_sac.rl.global_geometry import rollout
 x,ids=fixture();m=net().train();set_dropout(m,.15);torch.manual_seed(7)
 a=rollout(m,x[:5],ids[:9],len(x),4,x[:9],precision="FP32",recompute=False);a['Rg_loss'].backward()
 g=[None if p.grad is None else p.grad.clone() for p in m.parameters()];m.zero_grad(set_to_none=True);torch.manual_seed(7)
 b=rollout(m,x[:5],ids[:9],len(x),4,x[:9],precision="FP32",recompute=True);b['Rg_loss'].backward()
 torch.testing.assert_close(a['coordinates'],b['coordinates'],atol=0,rtol=0)
 for p,v in zip(m.parameters(),g):
  if v is not None:torch.testing.assert_close(p.grad,v,atol=1e-6,rtol=1e-4)
