"""A frozen J1, B immutable pre-V9+legacy adapter, C repaired parent, D repaired RL."""
import collections,copy,json,runpy,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1');OLD=R.parent/'e017_v9_property_steering_20261023_v1';legacy=runpy.run_path(str(OLD/'source/scripts/e017_v9/policy.py'));g=runpy.run_path(str(R/'source/scripts/e017_v91/goals.py'));c=g['c'];read=c['read'];save=c['save'];sha=c['sha'];digest=c['state_digest'];AA=c['AA'];ns=c['ns'];public_batch=g['public_batch'];collate=legacy['collate'];joint_kl=legacy['joint_kl']
class Engine(legacy['Engine']):
 def __init__(self):
  super().__init__();self.parent=copy.deepcopy(self.actor).eval().requires_grad_(False)
 @torch.no_grad()
 def distribution(self,worlds,policy):
  prefix=collate([w.state for w in worlds],'cuda');reference=self.prior(prefix);self.calls['J1_batches']+=1;self.calls['J1_states']+=len(worlds)
  if policy=='A_J1':return reference['logits'],reference['mu'],reference['ls'],reference,None
  batch=legacy['public_batch'](worlds,prefix,reference) if policy=='B_PREV9' else public_batch(worlds,prefix,reference)
  if policy in ['C_GOAL_OFF','D_GOAL_OFF']:
   batch['goals']=batch['goals']*0;batch['goal_valid']=batch['goal_valid']&False;batch['context']=batch['context']*0;batch['subgoal']=batch['subgoal']*0
  model=self.parent if policy in ['B_PREV9','C_FIXED','C_GOAL_OFF'] else self.actor
  logits,mu,ls=model(batch);self.calls['actor_batches']+=1;self.calls['actor_states']+=len(worlds);return logits,mu,ls,reference,batch
 def verify(self,trained=False):
  self.prior.verify();assert self.prior.identity==self.jfp;assert all(torch.equal(v,self.initial_buffers[n]) for n,v in self.prior.model.named_buffers());assert digest(self.parent.state_dict())==self.fp
  if not trained:assert digest(self.actor.state_dict())==self.fp
  assert all(p.grad is None for p in self.prior.model.parameters())

def sample(engine,goal,spec,policy,count=8,namespace='BASELINE'):
 # Original V9 candidate seed namespace deliberately preserved for matching.
 g['production_raw'](goal);return legacy['sample'](engine,goal,spec,policy,count,namespace)
def continue_worlds(engine,worlds,policy):return legacy['continue_worlds'](engine,worlds,policy)
def evaluate(panel,policies,label=None,checkpoint=None):
 assert read(R/'OBSERVABILITY_ADMISSION.json')['PASS'];spec=read(R/'REWARD_SPECIFICATION.json');goals=read(R/f'manifests/{panel}_GOALS.json')['goals'];e=Engine();start=time.perf_counter()
 if checkpoint:
  expected=sha(Path(checkpoint));state=torch.load(checkpoint,map_location='cpu',weights_only=False);e.actor.load_state_dict(state['actor'],strict=True);del state;assert sha(Path(checkpoint))==expected
 for policy in policies:
  for i,goal in enumerate(goals):
   c['POLICY'].require(R,4*2**20);name=f'results/{panel}_{label or policy}_{i:03}.json'
   if (R/name).exists():assert sha(R/name)==read((R/name).with_suffix('.sha.json'))['sha256'];continue
   rows,_=sample(e,goal,spec,policy);save(name,dict(goal=goal,policy=policy,checkpoint=checkpoint,records=rows));save(str(Path(name).with_suffix('.sha.json')),dict(sha256=sha(R/name)));print('EVAL',panel,policy,i+1,len(goals),flush=True)
 e.verify(trained=bool(checkpoint));save(f'{panel}_{label or "_".join(policies)}_COMPLETED.json',dict(policies=policies,checkpoint=checkpoint,goals=len(goals),candidates_per_goal=8,calls=dict(e.calls),elapsed_seconds=time.perf_counter()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),J1_fingerprint=e.jfp,parent_fingerprint=e.fp))
if __name__=='__main__':evaluate(sys.argv[1],sys.argv[2].split(','))
