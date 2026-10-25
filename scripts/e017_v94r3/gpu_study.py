"""Finite frozen-policy engineering comparison; never trains or touches TEST."""
import collections,copy,json,math,runpy,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r3_one_step_feasibility_20261025_v1');OLD=R.parent/'e017_v94r2_corrected_geometry_rl_20261025_v2'
P=runpy.run_path(str(R/'source/scripts/e017_v94r/production.py'));F=runpy.run_path(str(R/'source/scripts/e017_v94r3/shield.py'));E=runpy.run_path(str(R/'source/scripts/e017_v94r/evaluation.py'))
read=P['read'];save=P['compact'];c=P['c'];S=P['S'];START=time.perf_counter()

def sample(e,task,which,seeds):
 g=S['generate'].__wrapped__.__globals__;fn={'A':F['OLD']['extend_checked'],'B':F['F']['extend_checked'],'C':F['extend_checked']}[which]
 g['extend_checked']=fn
 try:return P['sample'](e,task,'C',len(seeds),seeds)
 finally:g['extend_checked']=F['extend_checked']

def summary(groups):
 s=E['summarize'](groups);rows=[r for g in groups for r in g['records']];reason=collections.Counter(x for r in rows for q in r['proposals'] if not q[6] for x in q[8]);accepted=sum(q[6] for r in rows for q in r['proposals'])
 s.update(rejection_reasons=dict(reason),accepted_actions=accepted,proposals_per_accepted=s['proposals']/max(1,accepted),
          actor_batches=None,acceptance_status=collections.Counter(q[9] for r in rows for q in r['proposals'] if len(q)>9))
 return s

def main():
 assert read(R/'FILTER_AND_GRADIENT_CPU_TESTS.json')['PASS'];P['guard']()
 assert not __import__('subprocess').check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 e,_=P['K']['load_engine']('C');e.actor.requires_grad_(False).eval();e.parent.requires_grad_(False);fp=c['state_digest'](e.actor.state_dict());jfp=c['state_digest'](e.prior.model.state_dict());tasks=read(P['V93']/'manifests/FUTURE_FIXED_TRAIN_POOL.json')['tasks'][::2][:4]
 results={};cost={}
 for which in ['A','B','C']:
  groups=[];beg=time.perf_counter();calls0=copy.deepcopy(dict(e.calls))
  for i,task in enumerate(tasks):
   path=R/f'results/FITTING_{which}_{i:02}.json'
   if path.exists():pack=read(path);assert pack['actor_fingerprint']==fp
   else:
    seeds=[S['ns']('R3_FIXED_FITTING',task['goal']['goal_id'],i) for i in range(8)];rows=sample(e,task,which,seeds);pack=dict(task=task,records=rows,actor_fingerprint=fp);save(path,pack)
   groups.append(pack)
  results[which]=summary(groups);cost[which]=dict(wall_seconds=time.perf_counter()-beg,forward_calls={k:e.calls[k]-calls0.get(k,0) for k in e.calls},attempts=32)
  print('FITTING',which,results[which],flush=True)
 save(R/'MATCHED_FITTING_COMPARISON.json',dict(summary=results,cost=cost,identities=4,attempts_per_arm=32,seeds='R3_FIXED_FITTING',all_weights_frozen=True))
 # Freeze implementation before accessing reserved development outcomes.
 save(R/'IMPLEMENTATION_FREEZE.json',dict(source_commit=P['bindings']()['V94_source_commit'],binding=P['bindings'](),
    compared_methods=['A old immediate','B R2 packing','C R3 exact one step'],fitting_identities=[t['source_reference_identity'] for t in tasks],
    development='existing reserved16 TRAIN identities x8; no prospective VALIDATION',counterfactual='R2 update500 frozen actor,6 affected groups x8',optimizer_updates=0))
 groups=[];beg=time.perf_counter();calls0=copy.deepcopy(dict(e.calls))
 devtasks=read(P['V93']/'manifests/TRAIN_TASKS.json')['tasks']
 for i,task in enumerate(devtasks):
  path=R/f'results/DEVELOPMENT_C_{i:02}.json'
  if path.exists():pack=read(path);assert pack['actor_fingerprint']==fp
  else:
   seeds=[S['ns']('CANDIDATE',task['goal']['goal_id'],j) for j in range(8)];rows=sample(e,task,'C',seeds);pack=dict(task=task,records=rows,actor_fingerprint=fp);save(path,pack)
  groups.append(pack);print('DEV',i,'complete',sum(r['metrics']['complete'] for r in pack['records']),flush=True)
 # Frozen A/B at initialization are exact reusable evidence on this panel.
 a=read(R.parent/'e017_v94_fixed_sequence_geometry_rl_20261024_v1/production/diagnostics/DEVELOPMENT_00000.json')['summary']['filtered']
 b=read(OLD/'production/diagnostics/DEVELOPMENT_00000.json')['summary']['filtered']
 save(R/'MATCHED_DEVELOPMENT_COMPARISON.json',dict(summary=dict(A=a,B=b,C=summary(groups)),C_seconds=time.perf_counter()-beg,
    C_forward_calls={k:e.calls[k]-calls0.get(k,0) for k in e.calls},A_B_reused=True,baseline_actor_fingerprint=fp,
    same_tasks_and_seeds=True,no_independent_VALIDATION=True,original_scientific_rules_unchanged=True))
 assert c['state_digest'](e.actor.state_dict())==fp
 # The stopped actor is a counterfactual forensic policy, never an initializer.
 index=read(OLD/'production/RECOVERY_INDEX.json')['latest'];path=OLD/'production'/index['relative'];assert P['sha'](path)==index['SHA256'];state=torch.load(path,map_location='cpu',weights_only=False);e.actor.load_state_dict(state['actor'],strict=True);del state
 stopfp=c['state_digest'](e.actor.state_dict());oldfail=read(OLD/'safety_review_20261025_v1/CANDIDATE_VALIDITY_TRANSITIONS.json')['records'];groupids=sorted({f['group'] for f in oldfail if f['termination']});rescues=[]
 for i in groupids:
  oldpacket=read(OLD/f'production/diagnostics/DEV_00500_filtered_{i:02}.json');task=oldpacket['task'];out=R/f'results/HISTORICAL_500_C_{i:02}.json'
  if out.exists():pack=read(out);assert pack['actor_fingerprint']==stopfp
  else:
   seeds=[r['seed'] for r in oldpacket['records']];rows=sample(e,task,'C',seeds);pack=dict(task=task,records=rows,actor_fingerprint=stopfp);save(out,pack)
  for before,after in zip(oldpacket['records'],pack['records']):
   if before['metrics']['hard_valid']:continue
   preceding=F['OLD']['reconstruct'](before,64,max(0,len(before['geometry'])-1));g=before['geometry'][-1];_,check=F['extend_checked'](preceding,task['fixed_sequence'][preceding.t],g)
   shield=[q for q in after['proposals'] if 'certified_one_step_dead_end' in q[8]]
   first_different=None
   for qa,qb in zip(before['proposals'],after['proposals']):
    if qa[:4]!=qb[:4] or qa[6]!=qb[6]:first_different=dict(old=qa,new=qb);break
   rescues.append(dict(group=i,candidate_index=before['candidate_index'],seed=before['seed'],original_prefix_length=len(before['geometry']),
       preceding_action_certified_rejected='certified_one_step_dead_end' in check['reasons'],
       new_length=len(after['geometry']),new_complete=after['metrics']['complete'],new_hard_valid=after['metrics']['hard_valid'],new_full_success=after['metrics']['success'],
       original_proposals=len(before['proposals']),new_proposals=len(after['proposals']),additional_proposals=len(after['proposals'])-len(before['proposals']),
       new_shield_rejections=shield,first_trace_difference=first_different,new_termination=after['termination']))
  print('HISTORICAL_GROUP',i,flush=True)
 save(R/'HISTORICAL_DEAD_END_REPLAY.json',dict(policy_checkpoint_SHA256=index['SHA256'],actor_fingerprint=stopfp,results=rescues,
    all_seven_failures=len(rescues)==7,matched_candidate_indexed_streams=True,
    limitation='new rejection changes states, live-batch shapes and FP rounding; full new rollouts measured, no rescue inferred from rejection alone'))
 assert c['state_digest'](e.actor.state_dict())==stopfp and c['state_digest'](e.prior.model.state_dict())==jfp
 e.prior.verify();assert c['state_digest'](e.parent.state_dict())==e.fp
 save(R/'GPU_ENGINEERING_COMPLETE.json',dict(PASS=True,protein_optimizer_updates=0,J1_bitwise_unchanged=True,actors_frozen=True,
  attempts_generated=96+128+48,peak_GPU_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.perf_counter()-START,
  historical_rescued_valid=sum(r['new_hard_valid'] for r in rescues),historical_seven=rescues))
 print('R3_FROZEN_STUDY_COMPLETE',flush=True)
if __name__=='__main__':main()
