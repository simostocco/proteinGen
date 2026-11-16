"""Predeclared bounded final-only goal/early-action controls. No optimization."""
from pathlib import Path
import json
import runpy
import sys
import time

import numpy as np
import torch

R=Path('/mnt/d/Simone/proteinGen_data/e017_v7_amortized_lowlevel_20261023_v1')
v=runpy.run_path(str(R/'source/scripts/e017_v7/study.py'))
def main():
 assert (R/'CAMPAIGN_COMPLETED.json').exists()
 assert not (R/'CAUSAL_DIAGNOSTICS.json').exists()
 e=v['Engine']();e.actor.requires_grad_(False);parent={k:a.cpu().clone() for k,a in e.actor.state_dict().items()};pool=v['tasks']('CONFIRM');identities=sorted({t['identity'] for t in pool},key=lambda h:v['rank']('causal'+h));goal=[];traces={};loaded={'A':parent};checkpoint={}
 for label in ['B','C','D']:
  p=R/f'results/{label}_FINAL_STATUS.json'
  if p.exists() and v['read'](p).get('evaluated'):
   rec=v['read'](p)['checkpoint'];assert v['sha'](rec['path'])==rec['sha256'];loaded[label]=torch.load(rec['path'],map_location='cpu',weights_only=False)['actor'];checkpoint[label]=rec
 for label,state in loaded.items():
  e.actor.load_state_dict(state)
  for h in identities:
   pair=[t for t in pool if t['identity']==h];pair.sort(key=lambda t:0 if t['variant']=='native' else 1);assert len(pair)==2
   for i in range(2):
    seed=v['ns'](h,i,'MATCHED_GOAL_DIAGNOSTIC');aa=[v['rollout'](e,t,seed,label) for t in pair]
    for t,a in zip(pair,aa):traces[(label,t['task_id'],i)]=a
    pa,pb=[np.asarray(a['tail'][-1]) for a in aa];ta,tb=[np.asarray(t['target']) for t in pair]
    goal.append(dict(arm=label,identity=h,group=pair[0]['protected_group_id'],seed=seed,correct=bool((pb-pa).dot(tb-ta)>0),own_minus_cross_A=float((np.linalg.norm(pa-ta)+np.linalg.norm(pb-tb)-np.linalg.norm(pa-tb)-np.linalg.norm(pb-ta))/2)))
 early=[]
 for label,state in loaded.items():
  if label=='A':continue
  for h in identities[:16]:
   t=next(t for t in pool if t['identity']==h and t['variant']=='native')
   for i in range(2):
    seed=v['ns'](h,i,'MATCHED_GOAL_DIAGNOSTIC');base=traces[('A',t['task_id'],i)];own=traces[(label,t['task_id'],i)]
    e.actor.load_state_dict(state);hybrid=v['rollout'](e,t,seed,label,delta_step=0,donor=base['latent'])
    e.actor.load_state_dict(parent);reverse=v['rollout'](e,t,seed,'A',delta_step=0,donor=own['latent'])
    early.append(dict(arm=label,task_id=t['task_id'],identity=h,group=t['protected_group_id'],seed=seed,
     parent=base,learned=own,learned_with_parent_first=hybrid,parent_with_learned_first=reverse))
 e.verify();v['save']('CAUSAL_DIAGNOSTICS.json',dict(goal_response=goal,early_action_records=early,checkpoint=checkpoint,
  scope='fixed-noise goal interventions and first latent-action swaps; future actions recomputed from hybrid prefix under recipient policy. Deterministic kinematic mechanism only; not biological causality or a high-level planner.',
  network_calls=dict(e.calls),peak_GPU=torch.cuda.max_memory_allocated(),UTC=time.time()))
 print('CAUSAL_COMPLETED',len(goal),len(early),flush=True)
if __name__=='__main__':main()
