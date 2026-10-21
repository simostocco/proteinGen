import hashlib
from pathlib import Path
import torch
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.models.batched_bc import encode_batch
MAIN=Path('/mnt/d/Simone/proteinGen_data/e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt')
SHA='881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412'
def file_sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def tensor_sha(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
class FrozenPrior:
 """Intentionally NOT a child of the trainable system/optimizer."""
 def __init__(self,device='cpu'):
  if file_sha(MAIN)!=SHA:raise ValueError('J1 checkpoint hash')
  w=torch.load(MAIN,map_location='cpu',weights_only=False)['model'];m=JointActor(ModelConfig(width=352,heads=8,layers=8,ffn_mult=w['encoder.blocks.0.ff.0.weight'].shape[0]//352,pair_bias=True,chemistry=False,backend='sdpa'));m.load_state_dict(w,strict=True);self.model=m.to(device).eval().requires_grad_(False);self.identity=tensor_sha(m)
  assert sum(p.numel() for p in m.parameters())==12351746
 @torch.no_grad()
 def __call__(self,b):
  if self.model.training:raise RuntimeError('J1 must stay in eval mode')
  prepared=None
  if getattr(self,'shared_prefix_features',False):
   from protein_joint_sac.models.batched_bc import features_batch
   features,pair,present=features_batch(b,chemistry=True);b['_execution_actor_features']=(features,pair,present)
   original=features.clone();original[...,:22]=0.;original[...,33]=0.
   pv=pair[0].clone();pv[...,21]=0.;prepared=(original,(pv,pair[1]),present)
  h=encode_batch(self.model,b,prepared=prepared);B=len(h);ids=torch.arange(20,device=h.device)[None].expand(B,20)
  z=self.model.geom(torch.cat([h[:,None].expand(B,20,352),self.model.selected_aa(ids),h.new_zeros(B,20,22)],-1)).float()
  return dict(h=h.float(),logits=self.model.aa_head(h).float(),mu=z[:,:,:3],ls=z[:,:,3:].clamp(-5,2))
 def verify(self,optimizers=()):
  ids={id(p) for p in self.model.parameters()}
  for o in optimizers:
   if any(id(p) in ids for g in o.param_groups for p in g['params']):raise AssertionError('J1 in optimizer')
  assert all(not p.requires_grad and p.grad is None for p in self.model.parameters())
  assert tensor_sha(self.model)==self.identity,'frozen J1 parameter/buffer changed'
