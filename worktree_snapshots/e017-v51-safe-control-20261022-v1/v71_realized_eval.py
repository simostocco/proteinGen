"""Read-only, stage-complete realized-return assessment of fixed critics."""
from pathlib import Path
import json
import runpy
import sys
import time
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v71_expected_return_calibration_20261023_v1')
V7=R.parent/'e017_v7_amortized_lowlevel_20261023_v1'
st=runpy.run_path(str(R/'source/scripts/e017_v71/study.py'))

def main():
 assert (R/'CAMPAIGN_COMPLETED.json').exists()
 e=st['v']['Engine']();e.actor.requires_grad_(False);fp=e.parent_fingerprint;qs=[st['LowQ'](st['PROFILES']['research']['low_q']).cuda().eval() for _ in range(2)];data=np.load(R/'results/realized.npz');idx=np.flatnonzero(data['split']=='CONFIRM');ep=st['episodes']();pool=st['taskmap']();outputs={};start=time.monotonic()
 selected=st['read'](R/'CRITIC_FINAL.json')['selected'];spec=[('V7_WARM',V7/'checkpoints/C_00000_stop.pt','critic'),('V71_SELECTED',Path(selected['path']),'critics')]
 for label,path,key in spec:
  ck=torch.load(path,map_location='cpu',weights_only=False)
  for q,w in zip(qs,ck[key]):q.load_state_dict(w)
  del ck
  values=[]
  with torch.no_grad():
   for rowi in idx:
    ei,k=map(int,data['index'][rowi]);episode=ep[ei];t=pool[episode['task_id']];x=st['v']['reconstruct'](t,episode,k);b=e.batch(t,x);a=torch.tensor([st['AA'].index(t['sequence'][len(x)])],device='cuda');g=torch.tensor([episode['geometry'][k]],device='cuda',dtype=torch.float64);mask=torch.ones(1,3,device='cuda',dtype=torch.bool)
    with torch.autocast('cuda',dtype=torch.bfloat16):values.append([float(q(b,a,g,mask)) for q in qs])
    if len(values)%512==0:print('REALIZED_CALIBRATION',label,len(values),len(idx),flush=True)
  outputs[label]=values
 e.verify();assert st['state_digest'](e.actor.state_dict())==fp
 st['save']('REALIZED_CRITIC_CONFIRMATION.json',dict(rows=idx.tolist(),predictions=outputs,critic_selected_SHA256=selected['sha256'],groups=64,transitions=len(idx),elapsed_seconds=time.monotonic()-start,network_calls=dict(e.calls),actor_updates=0,scope='all4stages of8fixed-parent trajectories per held-outTRAINconfirmation identity; no actor/critic optimizer and no future features'))
if __name__=='__main__':main()
