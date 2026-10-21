from pathlib import Path
import json,hashlib,shutil,os
import numpy as np,torch
R=Path(__file__).resolve().parents[3];D=R.parent;P=D/'pdb_supervised_split_50id80cov_20261016_v1'
MAIN=D/'e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt';MAIN_SHA='881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(n,x):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.part');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def read(n):return json.loads((R/n).read_text())
def guard():
 if shutil.disk_usage(R).free<10*2**30+200*2**20:raise RuntimeError('10GiB reserve plus200MiB staging')
def frame(x):
 a=x[-1]-x[-2];a=a/np.linalg.norm(a);c=np.cross(x[-2]-x[-3],a);c=c/np.linalg.norm(c);b=np.cross(c,a);return np.stack([a,b,c])
def load_actor(device='cuda'):
 from protein_joint_sac.models.joint_actor import JointActor
 from protein_joint_sac.models.dense_geometry_attention import ModelConfig
 st=torch.load(MAIN,map_location='cpu',weights_only=False);w=st['model'];cfg=ModelConfig(width=352,heads=8,layers=8,ffn_mult=w['encoder.blocks.0.ff.0.weight'].shape[0]//352,pair_bias=True,chemistry=False,backend='sdpa');m=JointActor(cfg);m.load_state_dict(w,strict=True);assert sum(p.numel()for p in m.parameters())==12351746;m.eval().requires_grad_(False);return m.to(device)
def tensor_sha(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
