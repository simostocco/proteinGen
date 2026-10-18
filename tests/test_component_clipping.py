import torch
from types import SimpleNamespace
from torch import nn
from protein_joint_sac.rl.component_clipping import combine,blocks
class Fixture(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=nn.Linear(2,2);self.encoder.config=SimpleNamespace(width=2);self.aa_head=nn.Linear(2,20);self.selected_aa=nn.Embedding(20,2);self.geom=nn.Sequential(nn.Linear(4,2),nn.SiLU(),nn.Linear(2,6))
def trial(d):
 m=Fixture();aa={n:None for n,p in m.named_parameters()}
 for n,p in m.named_parameters():p.grad=torch.zeros_like(p)
 aa['encoder.weight']=torch.ones_like(m.encoder.weight);m.encoder.weight.grad.fill_(10)
 aa['aa_head.weight']=torch.ones_like(m.aa_head.weight);m.geom[2].weight.grad.fill_(1);m.geom[2].weight.grad[[0,3]]=d
 out=combine(m,aa,{k:2. for k in blocks(m)[0]});return m,out

def test_distance_spike_cannot_scale_torsion_or_AA():
 a,x=trial(1);b,y=trial(1e12)
 assert torch.equal(a.geom[2].weight.grad[[2,5]],b.geom[2].weight.grad[[2,5]])
 assert torch.equal(a.aa_head.weight.grad,b.aa_head.weight.grad)
 assert x['group_scale']['torsion']==y['group_scale']['torsion']
 assert y['geometry_to_AA_shared_ratio']<=.500001
 assert not y['G_head_scaled_by_shared_cap'] and not y['whole_model_clip_applied']
 assert sum(y['entry_counts'].values())==sum(p.numel() for p in b.parameters())
def test_shared_zero_AA_does_not_zero_geometry_heads():
 m=Fixture();aa={n:None for n,p in m.named_parameters()}
 for n,p in m.named_parameters():p.grad=torch.ones_like(p)
 for p in m.aa_head.parameters():p.grad.zero_()
 o=combine(m,aa);assert o['geometry_shared_scale']==0
 assert m.geom[2].weight.grad.abs().sum()>0
 assert m.encoder.weight.grad.abs().sum()==0
def test_nonfinite_component_is_an_emergency_stop():
 import pytest
 m=Fixture();aa={n:None for n,p in m.named_parameters()}
 for n,p in m.named_parameters():p.grad=torch.zeros_like(p)
 m.geom[2].weight.grad[2,0]=float('nan')
 before={n:p.detach().clone() for n,p in m.named_parameters()}
 with pytest.raises(FloatingPointError):combine(m,aa)
 assert all(torch.equal(before[n],p) for n,p in m.named_parameters())
