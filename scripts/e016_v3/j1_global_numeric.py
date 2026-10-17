"""BF16/FP32 and eager/recompute checks on the actual J1 weights; no updates."""
import json,math
import torch
import numpy as np
import j1_global as j
from protein_joint_sac.rl.global_geometry import rollout
from protein_joint_sac.data.accepted_supervised import CoordinateReader
from protein_joint_sac.data.scaled_supervised import canonical
def run():
 j.core.setup();data=j.Data(False);panel=json.load(open(j.R/'configs/EVALUATION_PANELS.json'))['TRAIN'];h,o,_=next(v for v in panel if len(data.seqs[v[0]])>=32)
 row=data.rows[o];reader=CoordinateReader(data.proposal.external);x=torch.tensor(canonical(reader.read(row)),device='cuda');ids=torch.tensor([j.AA.index(v) for v in data.seqs[h]],device='cuda');m,s=j.parent_model();del s;out={};grads={}
 for mode,precision,recompute in [('FP32','FP32',False),('BF16','BF16',False),('RECOMPUTED','BF16',True)]:
  torch.manual_seed(913);r=rollout(m,x[:8],ids[:12],len(ids),4,x[:12],precision=precision,recompute=recompute)
  loss=r['Rg_loss']+r['clash_loss'];g=torch.autograd.grad(loss,list(m.parameters()),allow_unused=True)
  grads[mode]=[v.detach().cpu().double() if v is not None else None for v in g];out[mode]={'Rg_loss':float(r['Rg_loss'].detach()),'clash_loss':float(r['clash_loss'].detach()),'coordinates':r['coordinates'].detach().cpu().tolist()};del r,g
 def err(a,b):
  return math.sqrt(sum(float((x-y).square().sum()) for x,y in zip(grads[a],grads[b]) if x is not None))/max(1e-20,math.sqrt(sum(float(x.square().sum()) for x in grads[a] if x is not None)))
 bf=err('FP32','BF16');cp=err('BF16','RECOMPUTED');coord=float(np.max(np.abs(np.array(out['FP32']['coordinates'])-np.array(out['BF16']['coordinates']))))
 receipt={'status':'PASS' if bf<.15 and cp<1e-5 and coord<.1 else 'FAIL','BF16_FP32_relative_gradient_L2':bf,'eager_recompute_relative_gradient_L2':cp,'BF16_FP32_max_coordinate_error_A':coord,'thresholds':{'BF16_grad_relative':.15,'recompute_grad_relative':1e-5,'coordinate_max_A':.1},'horizon':4,'sequence_hash':h,'observation_id':o,'checkpoint_sha256':j.PARENT_SHA,'results':out}
 j.write(j.R/'tests/GLOBAL_PRECISION_PARITY.json',receipt)
 if receipt['status']!='PASS':raise ValueError('global precision parity preflight failed')
 print(json.dumps({k:v for k,v in receipt.items() if k!='results'}),flush=True)
if __name__=='__main__':run()
