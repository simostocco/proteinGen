"""Read-only matched-goal diagnostic for the existing RWR250 actor."""
from pathlib import Path
import runpy
import numpy as np
import torch
import time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v71_expected_return_calibration_20261023_v1');V7=R.parent/'e017_v7_amortized_lowlevel_20261023_v1'
s=runpy.run_path(str(R/'source/scripts/e017_v71/study.py'));v=s['v']
def main():
 assert (R/'CAMPAIGN_COMPLETED.json').exists();start=time.monotonic();e=v['Engine']();e.actor.requires_grad_(False);path=V7/'checkpoints/B_00250_stop.pt';st=torch.load(path,map_location='cpu',weights_only=False);e.actor.load_state_dict(st['actor']);del st;fp=s['state_digest'](e.actor.state_dict());pool=v['tasks']('CONFIRM');rows=[]
 for h in sorted({t['identity'] for t in pool}):
  tt=[t for t in pool if t['identity']==h];tt.sort(key=lambda t:0 if t['variant']=='native' else 1);assert len(tt)==2
  for i in range(2):
   seed=v['ns'](h,i,'MATCHED_GOAL_DIAGNOSTIC');a,b=[v['rollout'](e,t,seed,'B250_READ_ONLY_GOAL') for t in tt];pa,pb=np.array(a['tail'][-1]),np.array(b['tail'][-1]);ta,tb=np.array(tt[0]['target']),np.array(tt[1]['target'])
   rows.append(dict(identity=h,group=tt[0]['protected_group_id'],seed=seed,correct=bool((pb-pa).dot(tb-ta)>0),own_minus_cross_A=float((np.linalg.norm(pa-ta)+np.linalg.norm(pb-tb)-np.linalg.norm(pa-tb)-np.linalg.norm(pb-ta))/2)))
 e.verify();assert s['state_digest'](e.actor.state_dict())==fp;s['save']('RWR_GOAL_SENSITIVITY.json',dict(records=rows,checkpoint_SHA256=s['sha'](path),actor_frozen=True,actor_updates=0,elapsed_seconds=time.monotonic()-start,network_calls=dict(e.calls),matched_parent_B25_source=str(V7/'CAUSAL_DIAGNOSTICS.json')))
if __name__=='__main__':main()
