"""One final independent confirmation, using only previously selected safe states."""
import copy,json,runpy,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1');t=runpy.run_path(str(R/'source/scripts/e017_v9/train.py'));p=t['p'];c=t['c'];read=c['read'];save=c['save'];sha=c['sha']
def confirm():
 arms={m:read(R/(m+'_COMPLETED.json')) for m in ['JOINT','AA_ONLY','GEOMETRY_ONLY'] if (R/(m+'_COMPLETED.json')).exists()};assert arms
 bad=[m for m,z in arms.items() if any(s in z['stop_reason'] for s in ['VALIDITY','KL','NUMERICAL','RESOURCE'])]
 if bad:
  save('CONFIRMATION_COMPARISON.json',dict(status='BLOCKED_BY_SCIENTIFIC_SAFETY_STOP',arms=bad,promotion_PASS=False,confirmation_policy_outcomes_unexposed=True));return
 assert all((R/(m+'_COMPLETED.json')).exists() for m in ['JOINT','AA_ONLY','GEOMETRY_ONLY'])
 assert not (R/'CONFIRMATION_COMPARISON.json').exists();p['baseline']('CONFIRM');baseline=[read(f) for f in sorted((R/'results').glob('CONFIRM_B_V3_[0-9][0-9][0-9].json'))];allresults={}
 for mode,z in arms.items():
  selected=z['selected']
  if selected['update']==0:
   candidates=baseline;record=dict(reused_immutable_parent=True,additional_actor_passes=0)
  else:
   e,parent,opt,scheduler=t['setup'](mode);del opt,scheduler;path=Path(selected['checkpoint']);before=sha(path);state=torch.load(path,map_location='cpu',weights_only=False);e.actor.load_state_dict(state['actor'],strict=True);del state;e.actor.eval();record=t['evaluate'](e,mode,'SELECTED','CONFIRM');e.prior.verify();assert c['state_digest'](parent.state_dict())==e.fp and sha(path)==before;record['checkpoint_SHA256']=before;record['calls']=dict(e.calls);candidates=[read(f) for f in sorted((R/'results').glob(mode+'_SELECTED_CONFIRM_[0-9][0-9][0-9].json'))];del e,parent;torch.cuda.empty_cache()
  assert len(candidates)==len(baseline)==16;summary={}
  for K in [1,2,4,8]:
   x=np.array([int(any(r['metrics']['success'] for r in g['records'][:K])) for g in baseline]);y=np.array([int(any(r['metrics']['success'] for r in g['records'][:K])) for g in candidates]);d=y-x;draw=np.random.default_rng(19917).integers(0,16,(10000,16));bs=d[draw].mean(1);summary[str(K)]=dict(parent_success=int(x.sum()),selected_success=int(y.sum()),denominator=16,difference=float(d.mean()),identity_CI95=np.quantile(bs,[.025,.975]).tolist(),identity_CI98_333=np.quantile(bs,[.008333333333333333,.9916666666666667]).tolist())
  primary=summary['1'];primary_PASS=primary['difference']>=.05 and primary['identity_CI98_333'][0]>0
  allresults[mode]=dict(selected=selected,evaluation=record,success_by_K=summary,primary_K1_PASS=primary_PASS,promotion_PASS=False,promotion_note='No automatic promotion based only on K1: all frozen validity/KL/diversity/goal-sensitivity safeguards require full verification.')
 save('CONFIRMATION_COMPARISON.json',dict(status='COMPLETED_INDEPENDENT_CONFIRMATION',arms=allresults,actor_updates=0,promotion_PASS=False,identity_bootstrap_seed=19917,replicates=10000,goal_identity_disjoint_from_development=True,no_TEST=True,conservative_status='No promotion unless every original safeguard is independently satisfied.'))
if __name__=='__main__':confirm()
