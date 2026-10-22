"""Frozen confirmation and transfer audit, plus original V4 protocol; no updates."""
import sys,pathlib,json,copy,time,collections
import numpy as np
import torch
R=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v5_goal_observability_20261022_v1');sys.path.insert(0,str(R/'source/scripts/e017_v5'));import study as s

def load(e,arm):
 if arm=='A':e.reset();e.actor.eval();e.arm='A';return 0
 ix=json.load(open(R/'checkpoints'/arm/'INDEX.json'));record=ix['latest'];st=torch.load(record['path'],map_location='cpu',weights_only=False);assert s.sha(record['path'])==record['sha256'];e.actor.load_state_dict(st['actor']);e.actor.eval();e.arm=arm;return st['update']

def main():
 e=s.Engine();s.seed(s.CONFIG['seed']);begin=time.monotonic()
 # Reuse immutable shifted-prefix manifests frozen before baseline scoring.
 for H in [2,4,8]:
  for arm in ['A','B','C','D','E']:
   u=load(e,arm);s.evaluate(e,arm,H,u,'CONFIRM','native',32);s.evaluate(e,arm,H,u,'CONFIRM','shifted',32)
 # Equal-update H2 confirmation: evaluation-only snapshots, never resume states or new selections.
 for arm in ['B','C','D','E']:
  rec=json.load(open(R/f'checkpoints/{arm}/EVAL_H2.json'));assert s.sha(rec['path'])==rec['sha256'];st=torch.load(rec['path'],map_location='cpu',weights_only=False);e.actor.load_state_dict(st['actor']);e.actor.eval();e.arm=arm
  s.evaluate(e,arm,2,250,'CONFIRM','native_matched_H2',32);s.evaluate(e,arm,2,250,'CONFIRM','shifted_matched_H2',32)
 # Original V4 held-out tasks retained intact; new metrics are separate, never redefine old outcomes.
 old=json.load(open('/mnt/d/Simone/proteinGen_data/e017_v4_multistep_20261022_v1/manifests/VALIDATION.json'))['tasks'];old=copy.deepcopy(old)
 for t in old:t.update(H=8,task_id=t['identity']+'_V4',variant='native')
 for arm in ['A','B','C','D','E']:
  u=load(e,arm);rr=[];start=time.monotonic()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for t in old:
    for ns in [6101,6102]:
     loss,terms,x,_=e.rollout(t,ns);m=s.metric(t,x,ns);m.update(KL=float(terms['KL']),target=t['target']);rr.append(m)
  s.save(f'results/{arm}_ORIGINAL_V4.json',dict(records=rr,summary=s.summary(rr),seconds=time.monotonic()-start,update=u,scientific_role='historical exposed panel, not new independent confirmation; original V4 task geometry, tolerance and noise'))
 e.verify();del e;torch.cuda.empty_cache()
 # Independent constrained IK on same confirmation goals; no future intermediate coordinates read.
 for H in [2,4,8]:
  for condition in ['native','shifted']:
   pool=s.panel('CONFIRM',H,32) if condition=='native' else json.load(open(R/f'manifests/SHIFTED_CONFIRM_H{H}.json'))['tasks'];rr=[]
   for i,t in enumerate(pool):
    result=s.IK(t);result.update(identity=t['identity'],group=t['protected_group_id'],H=H,variant=t['variant']);rr.append(result)
   s.save(f'results/F_IK_H{H}_{condition}.json',dict(records=rr,task_attempts=len(rr),accepted=sum(r['accepted'] for r in rr),criterion='best accepted of up to2 starts x64 iterations; not stochastic policy KL'))
 s.save('results/FINAL_EVALUATION_COMPLETED.json',dict(seconds=time.monotonic()-begin,J1_frozen=True,no_updates=True));print('FINAL_CONFIRMATION_DONE',flush=True)

if __name__=='__main__':main()
