"""Compare existing matched-budget historical J1 states without optimizer calls."""
import json,gc
from pathlib import Path
import numpy as np
import torch
import j1_stabilize_train as t
R=t.R;old=t.O;result={}
for new,prior in [(1000,1500)]:
 p=R/f'runs/J1/checkpoint_{new:05d}.pt';q=old/f'runs/J1/checkpoint_{prior:05d}.pt'
 if not p.exists():continue
 a=t.c.s.load_checked(p);b=t.c.s.load_checked(q);out={'new_update':new,'old_pilot_update':prior,'starting_weights_both_old_pilot500':True,'same_sampler_state':t.c.same(a['sampler'],b['sampler']),'same_global_RNG_state':t.c.same(a['rng'],b['rng']),'same_unused_rollout_RNG':t.c.same(a['rollout_rng'],b['rollout_rng']),'difference':'clipping policy plus LR/scheduler; not an isolated clipping ablation'};del a,b;gc.collect()
 # Frozen per-target likelihood arrays from the completed diagnostic, same panel order.
 D=R.parent/'e016_j1_torsion_diagnostics_storage_20261018_v1';x=np.load(R/f'calibration/VALIDATION_{new:05d}.npz');y=np.load(D/f'arrays/pilot{prior}_VALIDATION.npz');panel=sorted(t.read(R/'configs/EVALUATION_PANELS.json')['VALIDATION'],key=lambda v:(v[2],v[0],v[1]));ids=np.array([h for h,o,pos in panel]);_,ix=np.unique(ids,return_inverse=True);rng=np.random.default_rng(891);boot=rng.integers(2947,size=(500,2947));diff={}
 for j,name in enumerate(['distance','angle','torsion']):
  ok=x['mask'][:,j];v=np.bincount(ix[ok],weights=x['nll'][ok,j]-y['nll'][ok,j])/7;diff[name]={'new_minus_old_mean_NLL':float(v.mean()),'identity_bootstrap95':np.quantile(v[boot].mean(1),[.025,.975]).tolist()}
 v=np.bincount(ix,weights=x['aa'].astype(float)-y['aa'].astype(float))/8;diff['AA']={'new_minus_old_mean_NLL':float(v.mean()),'identity_bootstrap95':np.quantile(v[boot].mean(1),[.025,.975]).tolist()};out['paired_VALIDATION_components']=diff;result[str(new)]=out
trace={}
for p in (R/'runs/J1').glob('updates_*.jsonl'):
 for line in open(p):
  v=json.loads(line);trace[v['new_update']]=v['sample_order_sha256']
prior={}
for p in (old/'runs/J1').glob('updates_*.jsonl'):
 for line in open(p):
  v=json.loads(line);prior[v['pilot_update']]=v['sample_order_sha256']
u=sorted(x for x in trace if x+500 in prior);result['sampling_order']={'comparable_updates':len(u),'matched_hashes':sum(trace[x]==prior[x+500] for x in u),'alignment':'newupdate u == previous pilot update500+u; same weights at old500 start'}
t.write(R/'MATCHED_HISTORICAL_BUDGET_COMPARISON.json',result);print('MATCHED_HISTORY_COMPLETE')
