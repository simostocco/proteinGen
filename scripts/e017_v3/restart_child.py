import argparse,json,hashlib
from pathlib import Path
import torch
from engineering import batch
from protein_joint_sac.crisp_v3.prior import FrozenPrior,tensor_sha,SHA
from protein_joint_sac.crisp_v3.networks import System
from protein_joint_sac.crisp_v3.optimization import Learner
from protein_joint_sac.crisp_v3.storage import restore
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.use_deterministic_algorithms(True);torch.manual_seed(1703);prior=FrozenPrior('cuda');s=System().cuda();learner=Learner(s);b=batch(prior);restore(a.checkpoint,s,learner,{},dict(profile='development',reference=SHA,engineering=True))
with torch.autocast('cuda',dtype=torch.bfloat16):r=learner.supervised(b,torch.tensor([2],device='cuda'),torch.tensor([[3.8,1.8,-.7]],device='cuda',dtype=torch.float64),torch.ones(1,3,device='cuda',dtype=torch.bool))
h=hashlib.sha256()
for k,st in learner.optimizers['actor'].state_dict()['state'].items():
 for key,v in st.items():h.update(str((k,key)).encode());h.update(v.cpu().numpy().tobytes())
prior.verify(learner.optimizers.values());out=dict(model_and_targets=tensor_sha(s),AdamW=h.hexdigest(),next_rng=torch.rand(4,device='cuda').cpu().tolist(),loss=r['loss'],J1=prior.identity)
with open(a.output,'x') as f:json.dump(out,f,indent=2)
print(json.dumps(out),flush=True)
