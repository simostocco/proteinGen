"""CPU-only final V7 analysis, checkpoint verification and twelve publications."""
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

R=Path('/mnt/d/Simone/proteinGen_data/e017_v7_amortized_lowlevel_20261023_v1')
OLD=R.parent/'e017_v51_safe_control_20261022_v1';V6=R.parent/'e017_v6_frozen_parent_candidates_20261023_v1'
sys.path.insert(0,str(R/'source/src'))
from protein_joint_sac.crisp_v3.recovery import sha,state_digest,atomic_json

INPUTS={}
def read(p):
 p=Path(p);INPUTS[str(p)]=sha(p);return json.loads(p.read_text())
def save(n,x):atomic_json(R/n,x)
def text(n,s):
 p=R/n;q=p.with_name(p.name+'.part')
 with q.open('w') as f:f.write(s+'\n');f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def md(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
def fmt(v):return 'not available' if v is None else f'{v:.6f}'
def CI(x,scale=1):return f"{x['estimate']*scale:.4f} [{x['CI95'][0]*scale:.4f}, {x['CI95'][1]*scale:.4f}]"
def loadlines(p):
 p=Path(p)
 if not p.exists():return []
 INPUTS[str(p)]=sha(p);return [json.loads(x) for x in p.read_text().splitlines()]

def bootstrap(values,groups):
 values=np.asarray(values,dtype=float);unique=sorted(set(groups));sums=np.array([values[np.array(groups)==g].sum() for g in unique]);counts=np.array([sum(h==g for h in groups) for g in unique]);draw=np.random.default_rng(177717).integers(0,len(unique),(20000,len(unique)));samples=sums[draw].sum(1)/counts[draw].sum(1)
 return dict(estimate=float(values.mean()),CI95=np.quantile(samples,[.025,.975]).tolist(),CI98_333333=np.quantile(samples,[.05/6,1-.05/6]).tolist(),identities=len(unique),observations=len(values),bootstrap_seed=177717,replicates=20000)

def paired(a,b,key):
 aa={(r['task_id'],r.get('candidate_index',r['seed'])):r for r in a};bb={(r['task_id'],r.get('candidate_index',r['seed'])):r for r in b};assert aa.keys()==bb.keys()
 return bootstrap([float(bb[k][key])-float(aa[k][key]) for k in aa],[aa[k]['group'] for k in aa])

def metrics(records):
 return dict(attempts=len(records),identities=len({r['identity'] for r in records}),success=sum(r['success'] for r in records),valid=sum(r['valid'] for r in records),translation_hits=sum(r['translation_hit'] for r in records),rotation_hits=sum(r['rotation_hit'] for r in records),collision_count=sum(r['clashes_per100']>2 for r in records),position_A=float(np.mean([r['position_A'] for r in records])),orientation_deg=float(np.mean([r['orientation_deg'] for r in records])),KL=float(np.mean([r['KL'] for r in records])))

def byK(records):
 targets={t['task_id']:np.asarray(t['target']) for t in read(R/'manifests/CONFIRM.json')['tasks']}
 def original_cost(r):
  delta=np.abs((np.asarray(r['tail'][-1])-targets[r['task_id']])/3.8)
  return float(np.where(delta<1,.5*delta**2,delta-.5).sum()+5*math.sin(math.radians(r['orientation_deg'])/2)**2)
 pools=collections.defaultdict(list)
 for r in records:pools[r['task_id']].append(r)
 for p in pools.values():p.sort(key=lambda r:r['candidate_index']);assert [r['candidate_index'] for r in p]==list(range(32))
 result={}
 for K in [1,2,4,8,16,32]:
  selected=[];exists=[];valid=[];div=[];single=[];group=[];times=0
  for tid,p in sorted(pools.items()):
   p=p[:K];v=[r for r in p if r['valid']];chosen=min(v,key=lambda r:(max(r['position_A']/8,r['orientation_deg']/35),original_cost(r),r['candidate_index'])) if v else None
   # AtK1 report the actual first draw, including invalid failures, not filtering away cases.
   selected.append(p[0] if K==1 else chosen);single.append(p[0]);exists.append(any(r['success'] for r in p));valid.append(bool(v));group.append(p[0]['group']);times+=sum(r['seconds'] for r in p)
   x=np.array([r['tail'] for r in p]);ix=np.triu_indices(K,1)
   if K>1:div.extend(np.sqrt(((x[:,None]-x[None,:])**2).sum(-1).mean(-1))[ix].tolist())
  success=[bool(r and r['success']) for r in selected];assert success==exists
  present=[r for r in selected if r is not None]
  result[str(K)]=dict(tasks=len(pools),identities=len(set(group)),success_count=sum(success),success=bootstrap(success,group),at_least_one_valid=sum(valid),abstentions=sum(r is None for r in selected),selected_valid=sum(r['valid'] for r in present),selected_collision_count=sum(r['clashes_per100']>2 for r in present),position_A_conditional=float(np.mean([r['position_A'] for r in present])) if present else None,orientation_deg_conditional=float(np.mean([r['orientation_deg'] for r in present])) if present else None,
    selected_proposal_KL_conditional=float(np.mean([r['KL'] for r in present])) if present else None,tail_pair_RMSD_A=float(np.mean(div)) if div else None,
    generation_seconds=times,actor_and_J1_passes_EACH=4*K*len(pools),seconds_per_success_task=times/sum(success) if sum(success) else None,
    task_results=[dict(task_id=tid,group=g,success=s,has_valid=v,selected_index=r['candidate_index'] if r else None) for (tid,_),g,s,v,r in zip(sorted(pools.items()),group,success,valid,selected)])
 return result

def main():
 torch.set_num_threads(1)
 seal=read(R/'SEALED_CONFIG.json');cfg=seal['config'];audit=read(R/'PARENT_ARCHITECTURE_AUDIT.json');census=read(R/'TASK_CENSUS.json')
 assert (R/'CAMPAIGN_COMPLETED.json').exists() or (R/'CAMPAIGN_STOP.json').exists() or (R/'RECOVERY_EXECUTION.json').exists()
 for p,h in seal['input_SHA256'].items():assert sha(p)==h,p;INPUTS[p]=h
 candidates=loadlines(R/'results/TRAIN_candidates.jsonl');assert len({(r['task_id'],r['candidate_index']) for r in candidates})==len(candidates)
 positives=[r for r in candidates if r['success']];invalid=[r for r in candidates if not r['valid']];bypool=collections.defaultdict(list)
 for r in candidates:bypool[r['task_id']].append(r)
 trainstats=dict(candidates=len(candidates),distinct_identities=len({r['identity'] for r in candidates}),successful_candidates=len(positives),successful_identities=len({r['identity'] for r in positives}),invalid_candidates=len(invalid),valid_goal_misses=sum(r['valid'] and not r['success'] for r in candidates),tasks_with_success=sum(any(r['success'] for r in p) for p in bypool.values()),tasks_with_valid=sum(any(r['valid'] for r in p) for p in bypool.values()),reward_mean=float(np.mean([r['reward'] for r in candidates])),reward_std=float(np.std([r['reward'] for r in candidates])),KL_mean=float(np.mean([r['KL'] for r in candidates])),completed_inference_seconds=sum(r['seconds'] for r in candidates),full_behavior_actor_J1_passes_each=4*len(candidates),task_success_fraction={k:sum(r['success'] for r in p)/len(p) for k,p in bypool.items()})
 logs={a:loadlines(R/f'results/{a}_updates.jsonl') for a in ['B','C','D']};online={a:loadlines(R/f'results/{a}_online.jsonl') for a in ['B','C','D']};finished={a:read(R/f'results/{a}_FINISHED.json') if (R/f'results/{a}_FINISHED.json').exists() else None for a in ['B','C','D']}
 stop={a:read(R/f'results/{a}_STOP.json') if (R/f'results/{a}_STOP.json').exists() else None for a in ['B','C','D']}
 evaluations={}
 for p in (R/'results').glob('*.json'):
  d=read(p)
  if 'records' in d and 'summary' in d and 'K1' in d:evaluations[p.stem]=d
 conf={a:{} for a in ['A','B','C','D']}
 for d in evaluations.values():
  arm=d['label'].split('_')[0]
  if arm in conf and d['panel'] in ['CONFIRM','CONFIRM_SHIFTED'] and d['K']==32:conf[arm][d['panel']]=dict(summary=metrics(d['records']),K1=metrics([r for r in d['records'] if r['candidate_index']==0]),by_K=byK(d['records']),version=d['label'])
 comparisons={};paired_bestK={}
 for a in ['B','C','D']:
  for panel in ['CONFIRM','CONFIRM_SHIFTED']:
   if panel not in conf[a] or panel not in conf['A']:continue
   av=next(d for d in evaluations.values() if d['label']=='A' and d['panel']==panel and d['K']==32);bv=next(d for d in evaluations.values() if d['label']==conf[a][panel]['version'] and d['panel']==panel and d['K']==32)
   aa=[r for r in av['records'] if r['candidate_index']==0];bb=[r for r in bv['records'] if r['candidate_index']==0]
   comparisons[f'{a}-A_{panel}_K1']={k:paired(aa,bb,k) for k in ['success','valid','position_A','orientation_deg','KL','joint']}
   paired_bestK[f'{a}-A_{panel}']={}
   for K in ['1','2','4','8','16','32']:
    abase={r['task_id']:r for r in conf['A'][panel]['by_K'][K]['task_results']};brows=conf[a][panel]['by_K'][K]['task_results']
    paired_bestK[f'{a}-A_{panel}'][K]=bootstrap([float(r['success'])-float(abase[r['task_id']]['success']) for r in brows],[r['group'] for r in brows])
  if 'CONFIRM' in conf[a] and 'CONFIRM' in conf['B'] and a!='B':
   aa=next(d for d in evaluations.values() if d['label']==conf['B']['CONFIRM']['version'] and d['panel']=='CONFIRM' and d['K']==32)['records'];bb=next(d for d in evaluations.values() if d['label']==conf[a]['CONFIRM']['version'] and d['panel']=='CONFIRM' and d['K']==32)['records'];comparisons[f'{a}-B_CONFIRM_K1']={k:paired([r for r in aa if r['candidate_index']==0],[r for r in bb if r['candidate_index']==0],k) for k in ['success','valid','position_A','orientation_deg','KL']}
 auxiliary=read(R/'CAUSAL_DIAGNOSTICS.json') if (R/'CAUSAL_DIAGNOSTICS.json').exists() else None
 goal={};early={}
 if auxiliary:
  for a in ['A','B','C','D']:
   rr=[r for r in auxiliary['goal_response'] if r['arm']==a]
   if rr:goal[a]=dict(pairs=len(rr),correct=sum(r['correct'] for r in rr),rate=bootstrap([r['correct'] for r in rr],[r['group'] for r in rr]),own_minus_cross_A=float(np.mean([r['own_minus_cross_A'] for r in rr])))
   rr=[r for r in auxiliary['early_action_records'] if r['arm']==a]
   if rr:
    early[a]={k:dict(success=sum(r[k]['success'] for r in rr),attempts=len(rr),valid=sum(r[k]['valid'] for r in rr),position_A=float(np.mean([r[k]['position_A'] for r in rr])),orientation_deg=float(np.mean([r[k]['orientation_deg'] for r in rr]))) for k in ['parent','learned','learned_with_parent_first','parent_with_learned_first']}
    early[a]['learned_first_effect_on_parent_success']=bootstrap([float(r['parent_with_learned_first']['success'])-float(r['parent']['success']) for r in rr],[r['group'] for r in rr])
    early[a]['parent_first_effect_on_learned_success']=bootstrap([float(r['learned_with_parent_first']['success'])-float(r['learned']['success']) for r in rr],[r['group'] for r in rr])
 checks={};retired=loadlines(R/'results/checkpoint_retirements.jsonl')
 for label in ['ENGINEERING','B','C','D']:
  path=R/f'checkpoints/{label}_INDEX.json'
  if not path.exists():continue
  index=read(path);checks[label]={}
  for role in ['latest','best']:
   rec=index[role]
   if rec is None:checks[label][role]=None;continue
   assert sha(rec['path'])==rec['sha256'];st=torch.load(rec['path'],map_location='cpu',weights_only=False);assert state_digest(st)==rec['state_digest'];assert st['source']==seal['source']
   required=['actor','optimizer','scheduler','dual','online_replay','offline_replay','rng','identity_cursor','critic','target_critic','critic_optimizer','config','sealed_input','parent_SHA','J1_tensor_SHA'];assert all(k in st for k in required)
   steps=sorted({int(v['step']) for v in st['optimizer']['state'].values()});assert steps==[st['update']]
   assert st['scheduler']['last_epoch']==st['update'] and st['identity_cursor']==st['update']*cfg['batch']
   critic_steps=[sorted({int(v['step']) for v in opt['state'].values()}) for opt in st['critic_optimizer']]
   assert st['offline_replay']['sha256']==sha(R/'results/TRAIN_candidates.jsonl')
   if label=='C' and st['update']==0:assert state_digest(st['actor'])==audit['parent_actor_state_digest']
   checks[label][role]=dict(record=rec,CPU_loaded=True,SHA256_verified=True,complete_semantic_digest_verified=True,required_fields=required,optimizer_steps=steps,scheduler_epoch=st['scheduler']['last_epoch'],identity_cursor=st['identity_cursor'],rng_keys=list(st['rng']),online_episodes=len(st['online_replay']),critic_states=len(st['critic']),critic_optimizer_states=len(st['critic_optimizer']),critic_updates=st['critic_updates'],critic_optimizer_steps=critic_steps,critic_fingerprints=[state_digest(q) for q in st['critic']],target_critic_fingerprints=[state_digest(q) for q in st['target_critic']],AMP_scaler='BF16; not used',actor_fingerprint=state_digest(st['actor']),scientifically_eligible=rec['eligible'])
   del st
 assert not list((R/'checkpoints').glob('*.part'))
 recovery=dict(source=seal['source'],data_generation_source=audit['source'],parent_SHA=audit['parent_SHA256'],J1_tensor_SHA=audit['J1_tensor_SHA'],checkpoints=checks,retirements=retired,engineering=read(R/'ENGINEERING_TESTS.json') if (R/'ENGINEERING_TESTS.json').exists() else None,scientific_stops=stop,
  replay_test_reference_fix=read(R/'REPLAY_PRECISION_DIAGNOSIS.json') if (R/'REPLAY_PRECISION_DIAGNOSIS.json').exists() else None,
  exact_replay_scope='full actor/Adam/scheduler/RNG next update replay in same process, CPU deserialization; fresh-process replay not claimed',
  operational_resume_limit='restore primitive and complete payload tested; main train entrypoint is not an automatic existing-index resume driver. Recovery supervisor refuses an interrupted training index pending explicit audit. Scientific STOP records block continuation; no training resume command authorized or executed.',
  interrupted_candidate_collection='1068complete unique fsynced rows validated; continuation skips exact existing task/index pairs; no actor optimization had occurred',partial_checkpoint_files=[])
 costs=dict(TRAIN_collection=trainstats,arms={},source=seal['source'],inference={},limits='instrumented distribution calls exclude cached-observation trust-region actor forwards; counts below separate those where identifiable; interruptions may have uncommitted forward work')
 for a in ['B','C','D']:
  rr=logs[a];f=finished[a]
  totalcritic=(f or {}).get('critic_updates',((checks.get(a) or {}).get('latest') or {}).get('critic_updates',0))
  costs['arms'][a]=dict(finished=f,committed_actor_updates=f['actor_updates'] if f else 0,logged_updates=len(rr),online_logged_episodes=len(online[a]),online_logged_actions=sum(len(r['geometry']) for r in online[a]),online_successes=sum(r['success'] for r in online[a]),distinct_online_identities=len({r['identity'] for r in online[a]}),distinct_update_identities=len({t.split('_')[0] for r in rr for t in r['task_ids']}),clipped_updates=sum(r['gradient_norm']>5 for r in rr),mean_gradient_norm=float(np.mean([r['gradient_norm'] for r in rr])) if rr else None,base_offline_episodes_available=len(candidates),base_replay_transitions=4*len(candidates),actor_task_presentations=4*len(rr),critic_transitions_reused=4*len(rr) if a!='B' else 0,critic_updates_each=totalcritic,critic_calibration_updates_each=250 if (a in ['C','D'] and (R/f'checkpoints/{a}_INDEX.json').exists()) else 0)
  if a=='B' and f and trainstats['tasks_with_valid']==512:
   costs['arms'][a]['corrected_total_actor_forwards_including_trust']=f['network_calls']['actor']+16*len(rr)
   costs['arms'][a]['trust_actor_forwards_uninstrumented']=16*len(rr)
   costs['arms'][a]['training_actor_forwards_excluding_DEV']=40*len(rr)
   costs['arms'][a]['training_J1_forwards_excluding_DEV']=24*len(rr)
  if a in conf:
   costs['inference'][a]={panel:{K:dict(actor_J1_passes_each=v['actor_and_J1_passes_EACH'],measured_generation_seconds=v['generation_seconds'],seconds_per_success_task=v['seconds_per_success_task']) for K,v in dd['by_K'].items()} for panel,dd in conf[a].items()}
 save('TRAINING_AND_INFERENCE_COSTS.json',costs);save('CHECKPOINT_AND_RECOVERY_MANIFEST.json',recovery)
 promoted={}
 for a in ['B','C','D']:
  reasons=[]
  if 'CONFIRM' not in conf[a]:promoted[a]=dict(GO=False,reasons=['no eligible trained checkpoint evaluated']);continue
  comp=comparisons[f'{a}-A_CONFIRM_K1']['success'];cur=conf[a]['CONFIRM'];base=conf['A']['CONFIRM']
  if comp['estimate']<.05 or comp['CI98_333333'][0]<=0:reasons.append('K1 gain lacks declared point/familywise CI support')
  if cur['summary']['KL']>1:reasons.append('native H4 conditional geometry KL>1')
  if cur['K1']['valid']/cur['K1']['attempts']<base['K1']['valid']/base['K1']['attempts']-.05:reasons.append('K1 native validity decline>5pp')
  h2=next((d for d in evaluations.values() if d['label']==cur['version'] and d['panel']=='CONFIRM_H2'),None);ah2=next((d for d in evaluations.values() if d['label']=='A' and d['panel']=='CONFIRM_H2'),None)
  if not h2 or h2['summary']['KL']>1:reasons.append('H2 KL safeguard unverified/failed')
  if h2 and ah2 and h2['K1']['success']/h2['K1']['attempts']<ah2['K1']['success']/ah2['K1']['attempts']-.05:reasons.append('H2 K1 retention drop>5pp')
  if a not in goal or goal[a]['rate']['estimate']<.6:reasons.append('goal-direction gate unverified/failed')
  if 'CONFIRM_SHIFTED' not in conf[a] or conf[a]['CONFIRM_SHIFTED']['K1']['success']<conf['A']['CONFIRM_SHIFTED']['K1']['success']:reasons.append('shifted K1 noninferiority point gate failed/unverified')
  if cur['by_K']['32']['success_count']<base['by_K']['32']['success_count']:reasons.append('K32 success below parent')
  promoted[a]=dict(GO=not reasons,reasons=reasons)
 results=dict(schema='E017_V7_FINAL_v1',source=seal['source'],panels=conf,paired_K1_comparisons=comparisons,paired_selected_success_by_K=paired_bestK,goal_response=goal,early_action_controls=early,decisions=promoted,TRAIN_census=trainstats,confidence_protocol='20,000 paired identity-cluster bootstrap, seed177717, task-weighted ratio; nominal95%, familywise98.333333% three trained-arm primary comparisons; K>1secondary bands pointwise, not simultaneous acrossK; fixed training/random pool limits. Degenerate zero-delta intervals on identical paired outcomes do not prove population equivalence.',candidate_records='original results/*_K32.json plus deterministic task manifests; no duplicated tensors')
 save('K1_AND_BEST_OF_K_COMPARISONS.json',results)
 publish(results,seal,census,audit,trainstats,finished,logs,recovery,costs,evaluations,stop)

def publish(results,seal,census,audit,trainstats,finished,logs,recovery,costs,evaluations,stops):
 cfg=seal['config'];conf=results['panels'];comparisons=results['paired_K1_comparisons'];goal=results['goal_response'];early=results['early_action_controls'];decisions=results['decisions']
 anygo=any(x['GO'] for x in decisions.values());decision='GO for further bounded low-level replication only' if anygo else 'NO-GO for promotion, longer horizons or hierarchical RL'
 calpath=R/'results/C_CRITIC_CALIBRATION.json'
 cal=read(calpath) if calpath.exists() else read(R/'CRITIC_CALIBRATION.json') if (R/'CRITIC_CALIBRATION.json').exists() else None
 gate_audit={}
 if logs['B']:
  first=[r['weighted_NLL'] for r in logs['B'][:25] if r['weighted_NLL'] is not None];last=[r['weighted_NLL'] for r in logs['B'][-25:] if r['weighted_NLL'] is not None]
  aa={t for r in logs['B'][:25] for t in r['task_ids']};bb={t for r in logs['B'][-25:] for t in r['task_ids']}
  gate_audit['RWR']=dict(first25_NLL_mean=float(np.mean(first)),last25_NLL_mean=float(np.mean(last)),reduction=float(np.mean(first)-np.mean(last)),required_reduction=.05*abs(float(np.mean(first))),first_window_identities=len(aa),last_window_identities=len(bb),shared_task_identities=len(aa&bb),interpretation='different task/teacher samples; not an identity-matched measure of learning progress; stop remains binding')
 if cal:
  eps=loadlines(R/'results/TRAIN_candidates.jsonl');pool=read(R/'manifests/TRAIN.json')['tasks'];ids=sorted([t['task_id'] for t in pool],key=lambda h:hashlib.sha256(('7717|calibration'+h).encode()).hexdigest());fit=set(ids[:48]);training=[(ep,k) for ep in eps if ep['task_id'] in fit for k in range(4)];probe=training[::max(1,len(training)//128)][:128]
  gate_audit['critic']=dict(training_stage_counts=dict(collections.Counter(str(k) for _,k in training)),probe_stage_counts=dict(collections.Counter(str(k) for _,k in probe)),training_probe_MSE=cal['train_MSE'],required_training_probe_MSE=.5*cal['train_constant_MSE'],heldout_terminal_MSE=cal['heldout_terminal_MSE'],required_heldout_terminal_MSE=.9*cal['heldout_constant_MSE'],heldout_terminal_both_pass=bool(all(v<=.9*cal['heldout_constant_MSE'] for v in cal['heldout_terminal_MSE'])),interpretation='stride12 of four-step flattened transitions samples only step0; stochastic suffix return variance may limit explained first-step MC variance. Not evidence of failed terminal generalization or universally unusable Q function.')
 save('ADMISSION_DIAGNOSTIC_LIMITATIONS.json',gate_audit)
 eng=read(R/'ENGINEERING_TESTS.json') if (R/'ENGINEERING_TESTS.json').exists() else None
 dgate=read(R/'D_ADMISSION.json') if (R/'D_ADMISSION.json').exists() else dict(admitted=False,reason='campaign did not reach D admission')
 native_rows=[];krows=[]
 for arm,panels in conf.items():
  if 'CONFIRM' not in panels:continue
  q=panels['CONFIRM'];p=q['K1'];native_rows.append([arm,q['version'],f"{p['success']}/{p['attempts']}",f"{p['valid']}/{p['attempts']}",fmt(p['position_A']),fmt(p['orientation_deg']),fmt(p['KL']),fmt(q['summary']['KL'])])
  krows.append([arm,*[q['by_K'][str(K)]['success_count'] for K in [1,2,4,8,16,32]]])
 native_table=md(['Arm','Selected version','K1 success','K1 valid','K1 position Å','K1 orientation °','K1 proposal KL','All32 proposal KL'],native_rows)
 ktable=md(['Arm /64 tasks','K1','K2','K4','K8','K16','K32'],krows)
 devrows=[]
 for d in sorted(evaluations.values(),key=lambda d:(d['label'],d['panel'])):
  if d['panel']!='DEV':continue
  q=d['summary'];k=d['K1'];devrows.append([d['label'],f"{k['success']}/{k['attempts']}",f"{q['valid']}/{q['attempts']}",fmt(q['position_A']),fmt(q['orientation_deg']),fmt(q['KL'])])
 devtable=md(['DEV version','K1 success','All2 valid','All2 position Å','All2 orientation °','All2 conditional KL'],devrows)
 cirows=[]
 for key,values in comparisons.items():
  cirows.append([key,CI(values['success'],100),CI(values['valid'],100),CI(values['position_A']),CI(values['orientation_deg'])])
 citab=md(['Paired comparison','Success Δpp [95%]','Validity Δpp [95%]','Position ΔÅ [95%]','Orientation Δ° [95%]'],cirows)
 decision_table=md(['Arm','Promotion gate','Reasons'],[[a,'GO' if q['GO'] else 'NO-GO','; '.join(q['reasons']) or 'all declared confirmation gates pass'] for a,q in decisions.items()])
 trainrows=[]
 for a,q in costs['arms'].items():
  ff=finished[a];trainrows.append([a,q['committed_actor_updates'],q['critic_calibration_updates_each'],q['critic_updates_each'],q['online_logged_episodes'],q['distinct_online_identities'],q['clipped_updates'],fmt((ff or {}).get('seconds')),(ff or {}).get('reason','not executed')])
 train_table=md(['Arm','Actor updates','Critic calibration /each','Critic total incl. calibration /each','Logged self episodes','Online identities','Clipped updates','Wall seconds','Stop/completion'],trainrows)
 transferrows=[]
 for arm,p in conf.items():
  if 'CONFIRM_SHIFTED' not in p:continue
  q=p['CONFIRM_SHIFTED'];m=q['K1'];transferrows.append([arm,f"{m['success']}/{m['attempts']}",f"{m['valid']}/{m['attempts']}",fmt(m['position_A']),fmt(m['orientation_deg']),fmt(q['summary']['KL']),q['by_K']['32']['success_count']])
 trans_table=md(['Arm shifted','K1 success','K1 valid','K1 position Å','K1 orientation °','All32 conditional KL','K32 success /64'],transferrows)
 goals=md(['Arm','Directional correct/pairs','Rate [95%]','Own−cross Å'],[[a,f"{q['correct']}/{q['pairs']}",CI(q['rate'],100),fmt(q['own_minus_cross_A'])] for a,q in goal.items()])
 censustext=f'''# TRAIN tasks and candidate census — E017 V7 v1

{census['TRAIN_eligible_joint_population']:,}qualified joint TRAIN sequence identities exist;1,014have nativeH4tasks in the previous verified task manifest. This bounded pilot hash-selects512distinct identities, one earliest eligible nativeH4task each, and draws8candidates/task. It does not claim to train on21,475identities or treat task multiplicity as independent proteins. Actual dataset: **{trainstats['candidates']}candidates,{trainstats['distinct_identities']}distinct TRAIN identities**.

Existing qualified TRAIN observation ownership/source hashes were checked against the governed PDB-only split data. Every selected native action witness was reconstructed, original validity and full goal completion rechecked, and original coordinate mapping/source provenance retained. Prefixes and four native AA identities are supplied sequentially; action masks are all three geometry components. Metadata uses full deposited sequence lengths. Complete sequences/prefixes/targets/witnesses/source hashes/task seeds are in manifests/TRAIN.json; no PDB copy was materialized.

Fresh VALIDATION:16DEVgroups/32native-or-synthetic tasks and32CONFIRMgroups/64tasks. All160previously prepared V4/V5validation groups were excluded, including V6's exposed identities. Eleven candidate identities were rejected by geometry/witness checks. DEV/CONFIRM/TRAINprotected groups are disjoint. The alternative goal changes the first witness torsion by0.8rad and is independently validity-tested; it is a synthetic kinematic control, not an experimental alternate conformation. TEST outcomes were not read.

Successful proposals: **{trainstats['successful_candidates']}**, spanning **{trainstats['successful_identities']}identities/tasks**. Invalid: {trainstats['invalid_candidates']}; valid goal misses:{trainstats['valid_goal_misses']}. Tasks with any successful proposal:{trainstats['tasks_with_success']}/512; tasks with any valid proposal:{trainstats['tasks_with_valid']}/512. Mean/std reward={trainstats['reward_mean']:.6f}/{trainstats['reward_std']:.6f}; mean conditional KL={trainstats['KL_mean']:.6f}.

All failures remain in results/TRAIN_candidates.jsonl. Each row stores immutable task locator, seed, behavior-policy version, four latent/transformed actions, transformed log probabilities, masks, four-point tail, errors, validity/clash/Rg/conditional KL and reward. Original prefix/goals are referenced, not repeated. Validation rows never enter RWR or SAC replay. A TRAIN-only critic calibration holdout is separate from scientific VALIDATION.

The interruption left1,068complete unique newline-terminated/fsynced candidates, spanning134identities; their sealed inputs were reverified and exact task/index pairs skipped on continuation. No actor update had started. Resource/source recovery receipts and interrupted logs are preserved. Counts of completed forward work exclude any unknown in-progress draw lost at interruption; no scientifically failed sample was replaced.

Task-level candidate success fractions and counts are in K1_AND_BEST_OF_K_COMPARISONS.json TRAIN_census. Multiple valid teacher solutions are retained and sampled with within-task normalized weights. No VALIDATION candidate is a teacher. TRAIN goals are native, while held-out panels include native and alternate goals; that transfer is not a matched within-TRAIN goal-extrapolation experiment.
'''
 text('TRAIN_TASK_AND_CANDIDATE_CENSUS.md',censustext)
 text('REWARD_AND_CONSTRAINT_SPECIFICATION.md','\n\n'.join([
 '# Reward and constraints — E017 V7 v1',
 'Frozen before model comparison in CONFIG.json/SEALED_CONFIG.json. Main actor remains17,438,114parameters with17,429,134trainable shared/geometry parameters;8,980categorical output parameters frozen. Native AAs fixed; J1 remains12,351,746frozen parameters. No high-level/reachability network is optimized.',
 'Terminal r=valid×[1(full complete geometric success)+0.25J], J=exp(−0.5[(position/8Å)²+(orientation/35°)²]); earlier rewards0. Range[0,1.25]. Both original target conditions and whole-prefix+tail hard validity are independently checked. Invalid episodes receive0, even with a low endpoint distance. This is Cα kinematic control, not folding/stability reward.',
 'RWR teacher weight=valid×[1(success)+0.1J], bounded[0,1.1]. Uniform identity/task sampling; normalize weights within the task over its retained trajectories. Every valid solution can be sampled; no arbitrarily chosen unique teacher. Invalid trajectories get zero teacher weight and remain in evidence/replay. A task with no positive weight uses native BC and conditional KL only, without being silently replaced. The coefficients are fixed bounded design choices; empirical TRAIN reward variation is reported, not fitted on VALIDATION.',
 'Fresh AdamW LR2e−6,25stepwarmup, cosine to2e−7through500, weight decay0.01, clip5. Full actor Adam moments are explicitly initialized at zero before size admission. RWR averages per-component negative transformed log probability over all4actions; native BC coefficient0.1. B and C collect one current-policy episode per actor update, share the same TRAIN task/noise schedule and initial4096behavior trajectories. Within-task sampling prevents higher multiplicity from increasing a goal’s presentation frequency.',
 'Original physical law: d=exp(u0), angle=πsigmoid(u1), signed torsion=πtanh(u2), diagonal3Gaussian conditional on fixed AA. Physical log probability includes all invertible-transform Jacobians. No temperature changes, action clamps, invalid replacements or fabricated boundaries. Conditional geometry KL is mean over3Gaussian components, then states/rollout steps and attempts; it is not an AA-marginal or trajectory-level KL. The immutable reference is J1 at the same prefix state.',
 'Adaptive KL multiplier starts0.1; λ←max(0,λ+0.05(batch_mean_KL−0.8)). Hard held-out mean limit remains1.0. Per-optimizer-step mean distribution KL(new||old) on fixed TRAIN replay observations must≤0.02; if it fails after stepping, restore the last complete durable transaction and preserve stop evidence. This is a sampled training-state trust region, not a proof of global/held-out constraints.',
 'DEVscientific stops: nativeH4orH2meanKL>1; H4validity decline>5pp; H2K1success decline>5pp. DEVat25,100,250,500. Maximum500actorupdates;250→500requires safe DEV K1≥parent plus declared TRAIN-learning condition. Checkpoint selection is eligible maximum DEVK1success, then mean joint score, then earliest. Failure/latest states never become selected solely because their position improves. Confirmation is never used to select.',
 'Confirmation promotion: K1gain≥5pp and positive98.333333%familywise lower bound for3arms vsA; preserved native validity within5pp; H4/H2KL≤1; H2success retention within5pp; directional response≥0.6; generated-prefix K1point not worse; K32not worse. Nominal95%intervals reported separately. No threshold was changed after outcomes.',
 'SAC uses fixed-AA twin-Q Bellman targets with γ0.995, Polyak0.005, fixed transformed-entropy coefficientα0.001normalized by3. Nonterminal target=r+γ[min(Q1target,Q2target)−αlogπ(g′|s′)/3−λKL(s′)]; terminal target is the actual terminal task reward, with zero bootstrap. Actor minimizesαlogπ/3−minQ+λKL+0.1nativeBC. Targets are no-grad; Q parameters freeze during actor gradients while candidate-action gradients remain. Replay contains exact reconstructed observations, actions, policy versions and behavior probabilities. Standard off-policy Bellman SAC needs no trajectory importance-ratio weighting for this Markov replay; RWR is empirical weighted likelihood, not an off-policy policy-gradient estimator.',
 'Critic gate:250TRAIN-only known-return updates; fit MSE≤0.5constant-predictor MSE and held-out-terminal MSE≤0.9constant-predictor MSE for BOTH critics, with nonzero reward variation/finite values. No SAC actor updates before pass. ArmD requires both B/Csafe250+updates, BDEVK1strictly better and Cnot worse; a prior scientific/numerical stop blocks D. No alternative algorithm is silently substituted.',
 'Storage:2GiB operational reserve plus64MiBnear-term growth and incoming full checkpoint; latest/best plus atomic staging. Source/config/task/candidate hashes, model/optimizer/scheduler/RNG/dual/cursor/online replay/reference bindings are saved. BF16 uses no GradScaler. Exact replay tests are same-process CPU deserialization/full-state restore; no fresh-process guarantee is claimed.'
 ]))
 text('REWARD_WEIGHTED_LEARNING_RESULTS.md','\n\n'.join([
 '# Reward-weighted behavioral learning — E017 V7 v1',
 f"ArmB final receipt: `{json.dumps(finished['B'])}`. Stop: `{json.dumps(stops['B'])}`. No result is interpreted as AA sequence-design learning.",train_table,devtable,native_table,ktable,citab,
 'The RWR dataset has task-normalized positive successes plus bounded graded valid near misses, retaining all modes and failures in the archive. Updates average four task presentations. Fresh self-generated episodes augment each task’s retained behavioral dataset with its actual policy version and density. No critic updates occur for B.',
 'The primary question is independently sampled CONFIRMK1full success, not weighted TRAIN likelihood, selected Best-of-K or lower positional error. Any selected checkpoint must independently pass confirmation safeguards. A stopped latest state is preserved but is not retrospectively promoted or evaluated as though selected.',decision_table,
 'Training curve plots: plots/TRAINING_AND_DEV_CURVES.png. RWR weighted NLL windows use different sampled tasks/teachers and are not a fixed-probe likelihood curve; the sealed250continuation condition uses their aggregate change and must be interpreted with this sampling limitation. Full per-update loss/KL/norm/LR/dual and online outcome records are retained.'
 ,f"Admission-statistic audit: `{json.dumps(gate_audit.get('RWR'))}`. The two100identity windows share zero tasks. Thus the failed5%criterion is not proof the policy worsened on a fixed teacher panel. B250 retained DEVvalidity62/64 and meanKL0.217894, with K1success4/32 versus parent3/32, but was excluded by the frozen continuation-stop selection logic. This is not a physical/KL failure. No post hoc B250confirmation or promotion was performed. The selected25state answers only the prospectively selected-model question. A future protocol should prospectively separate continuation readiness from safety eligibility and use an identity-matched training probe."
 ]))
 caltxt='Not reached: no critic-calibration result exists.' if cal is None else json.dumps({k:v for k,v in cal.items() if k not in ['heldout_predictions','heldout_returns']},indent=2)
 text('SAC_TRAINING_AND_CRITIC_CALIBRATION.md','\n\n'.join([
 '# SAC and twin-critic calibration — E017 V7 v1',
 'The original research actor is unchanged in capacity. The original research LowQ shape384wide/7blocks/8heads is used for both independent critics and their separate frozen targets; actual parameter counts and complete measured checkpoint bytes are in results/C_STORAGE_ADMISSION.json when reached. No smaller critic is silently substituted.',
 'Known-return admission trains on48TRAINidentities and evaluates a32TRAINidentity calibration holdout, distinct from the scientific VALIDATION confirmation. Labels are realized delayed terminal returns γ^(remaining_steps)r, with terminal actions having deterministic observed rewards. Nonterminal labels are noisy MonteCarlo continuations, not exact expected Q. Both twin critics must beat the sealed constant-predictor baselines; this tests bounded learnability/calibration, not critic optimality.',
 '```json\n'+caltxt+'\n```',train_table,
 'A failed gate stops ArmC with zero SAC actor updates and a complete calibrated/stopped checkpoint. It is evidence against trusting these critics in this budget/configuration, not proof SAC can never work. No policy-gradient comparator was substituted. If calibration passes, target critics copy calibrated weights, then an exact full-state next-SAC-transaction replay test precedes research actor updates; its receipt is preserved separately.',
 f"Diagnostic limitation: `{json.dumps(gate_audit.get('critic'))}`. Actual critic training used384transitions at each step0/1/2/3, but the strided128example TRAINprobe sampled only step0. Both critics PASS the declared held-out-terminal MSE requirement; the combined admission fails its stricter first-action fit requirement. A first-action Q estimates expected future return from stochastic suffixes; it should not be assumed able to explain most realized-return variance. This is an admission-design limitation, not a retrospective threshold change or a failed held-out-terminal result. No additional SAC actor updates were permitted. Before another SAC study, use stage-stratified probes and repeated suffix trials/expected-return calibration, with a new prospectively justified gate."
 'Bellman terminal masking, γ, transformed densities, reparameterized action gradients, fixed-AA conditioning and compact replay reconstruction are tested in ENGINEERING_TESTS.json where reached. Full replay test covers actor/Adam/scheduler/RNG/dual; C_SAC_REPLAY_TEST.json additionally covers Q/target/critic optimizers if admitted. No fresh-GPU-process replay was claimed. J1 parameters/buffers are independently verified and never enter any optimizer.',
 f"ArmD admission: `{json.dumps(dgate)}`. D does not run automatically after unsafe B/C or insufficient evidence. If executed, it starts from the selected B actor with newly initialized, separately calibrated critics and the same additional actor/environment ceilings as C. B pretraining remains an extra cost, not erased from comparisons.",
 'H2 interpretation: B/C share initial behavior candidates, actor ceilings and one new environment episode/update. C also spends twin-critic/target forwards and calibration updates. Thus interaction/actor-update matching does not prove equal total compute; measured wall time, call instrumentation limits and replay reuse are reported. No SAC superiority claim is made from unmatched optimization budgets or a failed calibration.'
 ]))
 text('GOAL_VALIDITY_KL_ANALYSIS.md','\n\n'.join([
 '# Goal, validity, KL and early-action controls — E017 V7 v1',native_table,ktable,devtable,citab,decision_table,
 'Full goal success is the conjunction of≤8Åposition,≤35°orientation and original whole-prefix+tail validity. Unchanged limits: adjacent2.8504200470168195–4.052024993913842Å, nonlocal contacts<2Å at index separation≥3,clashes/100≤2, normalizedRgwithin1.950656875000882–10.184757892570111, finite/nondegenerate frames. Angle/torsion support is unchanged; no new realism tolerance was invented.',
 'K1 includes every first independent draw, including invalid failures. K>1 uses the V6 joint normalized-goal score over hard-valid candidates; full-success existence equals selector success mathematically. Empty valid sets abstain and count as failed tasks. Selected conditional error/KL means exclude abstentions and are explicitly conditional; task success retains the full denominator. Average proposal KL on selected draws is not the selected distribution’s KL.',
 goals,
 'Goal-direction controls hold prefix/AA/noise fixed while changing native versus synthetic feasible target; correctness is dot(endpoint difference,target difference)>0. Own-minus-cross distance is separately reported, with negative favorable. This uses two matched streams per identity and does not select checkpoints. The auxiliary control script/protocol was sealed before trained actor updates; it adds final-only measurements without revising the primary protocol.',
 'Early-action controls on16hash-ranked confirmation identities, native goals,two matched streams: parent and learned full traces versus each recipient policy receiving the other’s first latent action, then recomputing all subsequent actions under the same standardized noise at the hybrid prefix. This changes an actual intermediate decision, not merely the final error formula. The group-clustered success effects are:',
 '```json\n'+json.dumps(early,indent=2)+'\n```',
 'This measures deterministic kinematic intervention effects within a fixed short horizon. It does not establish biological/thermodynamic causality, a learned positive reachability predictor, or a benefit of high-level planning. It does not isolate final-step variance narrowing from every other policy change; component Gaussian KL and shared-noise controls are descriptive, not a complete variance-causality analysis.',
 'All evaluation outcomes/invalidity summaries and actual complete four-action records are retained in results. Source and checkpoint hashes are in provenance/recovery manifests. A lower TRAIN loss or average positional error cannot override a failed goal/validity/KL gate.'
 ]))
 text('GENERATED_PREFIX_TRANSFER.md','\n\n'.join([
 '# Generated-prefix transfer — E017 V7 v1',trans_table,
 'Two final observed prefix actions are replaced using immutable V3 actor weights, fixed identity-derived seed1101+int(identity[:6],16); both native and synthetic target specifications are retained, without outcome filtering. Input-conditioning limitation: the V7 helper applies its target-relative context adapter even after loading V3 weights, while the original historical V3 prefix helper used zero context/canonical goal tokens. Thus these are V3-weight/V7-input generated-prefix stress cases, not byte-identical original-V3 decoding controls. The prefixes and streams are fixed and shared across V7arms; feasibility may change, and whole-prefix validity includes inherited invalidity. Historical transfer scores must not be pooled with this panel.',
 'The same shifted tasks/noise are shared across model arms. Native and shifted panels use separate deterministic stream namespaces, so a within-model raw native-minus-shifted difference is not an exact same-noise causal prefix-effect estimate. Model-versus-parent comparisons within the shifted panel are matched. No shifted candidate enters training or checkpoint selection.',citab,
 'K1shifted noninferiority is a point safeguard in the sealed protocol;32groups cannot certify broad distribution-shift safety. Failure on a shifted goal does not prove infeasibility, and successful Cα fragments do not imply stable folded proteins. Paired matched-goal and first-action controls are reported separately.'
 ]))
 main=[
 '# E017 V7 final scientific report — v1',
 f'**Decision: {decision}.** The research question is higher first-attempt full-valid H4goal success under unchanged safety. Amino-acid identities remained fixed; this is not de novo sequence design.',
 f"Training source commit `{seal['source']}`; data-generation source `{audit['source']}`; isolated worktree branch`e017-v7-amortized-lowlevel-20261023-v1` on D:. Parent `{audit['parent_SHA256']}` is V5D+250, the exact V6proposal parent. Actor fingerprint`{audit['parent_actor_state_digest']}`; frozen J1 tensor fingerprint`{audit['J1_tensor_SHA']}`. J1fileSHA256`881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412`. Strict architecture restoration and exact historical coordinate parity passed. Pre-comparison refinements initialized full zero Adam moments for realistic storage sizing and added SAC recovery tests. The first engineering attempt stopped before any actor update because its saved-tail reference defaulted toFP32: up to1.7377e-6Åerror versus1.0658e-14ÅwithFP64. Only the reference dtype was corrected; the1e-8tolerance, reward, data, model and scientific thresholds were unchanged. Original failed log/source/seal and diagnosis are preserved under protocol_engineering_fix/v1 and logs.",
 '## Data and bounded design',
 f"512qualified TRAIN identities,8frozen-parent trajectories each; {trainstats['successful_candidates']}full successes across{trainstats['successful_identities']}identities.16freshDEVgroups/32tasks,32freshCONFIRMgroups/64tasks.160earlier prepared validation groups excluded;11rejected witness candidates retained. Existing homology partitions preserved; TEST sealed. Original V6panel remains historical context, not new independent confirmation. See TRAIN_TASK_AND_CANDIDATE_CENSUS.md for exact provenance and interruption recovery.",
 'A frozen parent; B task-normalized reward-weighted behavioral learning; C independent fixed-AA twin-Q SAC after known-return calibration; D conditional RWR→SAC only after B/Cgates. Fresh actor optimizers, common actor ceiling500, DEV25/100/250/500, no automatic extension. B/Cshare offline proposals and new-environment episode schedule; extra critic compute remains explicit. Reward/constraint definitions were frozen before comparisons. No high-level/reachability training or longer horizon.',
 '## Training, stops and selected models',train_table,devtable,
 'A scientific-stop state is durable recovery evidence. Only an eligible DEV-selected checkpoint goes to final confirmation; if none exists, the trained arm has no promoted comparison model and its failed DEV outcomes remain reported. No failed state is silently replaced by a trained success or labeled a selected checkpoint.',
 '## Primary confirmation and success versus K',native_table,ktable,citab,decision_table,
 'Each K1row counts64tasks from32independent protected identities. K>1success is per query, not per candidate or protein fold. All smallerKare prefixes of the same32pool, with task/candidate-indexed seeds shared across model arms.20,000paired protein-cluster bootstrap replicates, seed177717; nominal95%and familywise98.333333%for3trained-arm primary comparisons. Secondary paired success-versusKintervals are in JSON and are not simultaneous bands over6Kvalues. A zero-width paired interval from identical binary outcomes is conditional on this fixed panel/noise pool; it does not prove population equivalence. No training-seed or independent pool-repeat uncertainty is captured. Reused DEV and final confirmation remain within VALIDATION, not TEST.',
 '## Hypothesis answers',
 'H1/amortization and H3/safety are assessed by the paired K1and gate tables above; lower weighted teacher loss is not success. If no arm passes every confirmation criterion, this bounded study is a NO-GO, even if its point estimate improves. H2/RL is unresolved if critic admission fails; lack of SAC actor training is not evidence that RWR beats correctly calibrated SAC. H4transfer uses fresh identities, alternate feasible goals and shifted prefixes; exact results and limits are in GENERATED_PREFIX_TRANSFER.md.',
 goals,
 'Early-action intervention effects and uncertainty are in GOAL_VALIDITY_KL_ANALYSIS.md. A positive hybrid effect is short-horizon kinematic mechanism evidence only; it does not establish readiness for hierarchical planning or biological causality. If absent/unsupported, no useful learnable early-control mechanism is claimed.',
 '## Critic calibration and inference/training efficiency',
 ('Critic gate not reached.' if cal is None else f"Direct SAC critic calibration PASS={cal['PASS']}; trainMSE={cal['train_MSE']} versus constant{cal['train_constant_MSE']}; heldout-terminalMSE={cal['heldout_terminal_MSE']} versus constant{cal['heldout_constant_MSE']}. See SAC_TRAINING_AND_CRITIC_CALIBRATION.md. A failed gate correctly blocks SAC actor updates."),
 'The diagnostics themselves limit interpretation: RWR first/last NLLwindows contain100different TRAINidentities with zero overlap, so their change is not a paired learning curve. The critic TRAINfit probe contains only first-action transitions due to stride12 over a four-step flattened table, while critic training covered all stages. Both terminal-holdout critic MSEgates pass; the stop comes from the first-step fit requirement on noisy realized returns. These weaknesses are recorded in ADMISSION_DIAGNOSTIC_LIMITATIONS.json. They do not authorize changing gates or retrospectively selecting B250, which was physically/KLsafe on DEV but failed continuation. H1at the unselected250state and H2with an admitted SACpolicy remain unresolved.',
 'TRAINING_AND_INFERENCE_COSTS.json separates actor/critic updates, offline candidates, new episodes/actions, distinct identities, replay reuse, wall time and instrumented calls. Distribution-call instrumentation omits cached-observation trust-region actor forwards; do not call those incomplete counters total model FLOPs. Candidate inference times include scoring/CUDA synchronization, exclude journal fsync/setup. No standalone cold-start inference benchmark or equal-FLOP superiority is claimed.',
 '## Recovery and storage',
 'Complete latest/best checkpoints were freshly SHA256/CPU-deserialization/semantic-digest verified, including raw actor, zero/full Adam moments, LR scheduler, RNG, dual, deterministic cursor, online replay and immutable offline replay binding; critics/targets/optimizers included where constructed. CHECKPOINT_AND_RECOVERY_MANIFEST.json gives exact hashes/paths/counters and retired-own-state receipts. All staged writes verify before atomic publication; no partial checkpoint remains. Engineering exact-next-update replay is same-process full-state restoration, not a claimed fresh-process experiment.',
 'Recovery limitation: the payload and restore primitive pass exact next-update replay, but the main train CLI is not an automatic existing-index recovery driver. The supervisor refuses an interrupted training index until explicitly audited, rather than silently starting from zero. All completed scientific stops remain binding and no training-resume command is provided. The recovered interruption occurred during deterministic candidate collection before any actor training; that collection resumed exactly by task/index.',
 'All generated artifacts/caches/tmp stayed on D:. Incoming checkpoint+64MiBgrowth+2GiBreserve governed admission. Existing PDB/UniRef/E010/J1/E016/E017evidence was not deleted or modified. Only newly run-owned superseded latest/best checkpoints could be retired after verified publication. Resource watch records actual RAM/swap/D:space and stages. One heavy GPUworker at a time; final GPUownership and worker-exit evidence are in provenance.',
 '## Historical context, limits and next decision',
 'V6frozen parent produced7/63K1and39/63K32, but these are different exposed goals and cannot be numerically substituted for the new Arows. V5H2goal-relative improvement and V5/V5.1H4KL/clash failures motivate the unchanged safeguards; no V5.1stopped arm initialized V7. Prior reports/checkpoints are preserved. No AlphaFold/external inference occurred.',
 'One training seed,512TRAINidentities,16development/32confirmation groups,500step cap, TRAIN-only critic-fit gate and goal/prefix-control limits constrain conclusions. RWR samples successful native-goal trajectories, not a broad sequence-design distribution. A failed critic pilot establishes only that this calibration/budget was inadequate. Before selecting AAs, independently validate joint categorical+geometry densities/rewards/credit assignment and native-free AAcontrol with partition safety; this experiment supplies no such evidence.',
 f"Next: {('replicate the safe low-level result across independent seeds and goals before greater algorithmic complexity' if anygo else 'repair the diagnostic design prospectively: fixed-identity RWR teacher probes and stage-stratified/repeated-suffix expected-return critic calibration, before spending more actor or SAC updates')}. Do not infer need for SAC, a learned critic, H8or a high-level planner from aspiration alone. No further experiment is launched."
 ]
 text('FINAL_E017_V7_SCIENTIFIC_REPORT.md','\n\n'.join(main))
 text('NEXT_CRISP_RESEARCH_DECISION.md','\n\n'.join([
 '# Next CRISP research decision — E017 V7 v1',f'**{decision}.**',decision_table,
 '1. Successful H4trajectories can be present without being amortized. Use the held-out paired K1rows to decide whether this pilot improved the proposal; candidate-positive counts alone do not answer it.',native_table,
 '2. Reward-weighted sufficiency versus RL depends on a calibrated SAC comparison with matched interactions and accounted compute. If critic admission failed, H2remains open; if SAC ran, its paired C−B result and extra critic cost must both be considered.',
 '3. Any full-goal gain with unsafe validity/KL is rejected. The exact failed criterion and checkpoints remain in the manifests. Repeating updates or relaxing thresholds is not the next step.',
 '4. Generated-prefix transfer must retain the original goal and full unconditional denominator; compare arms within the common shifted panel. Generalization is limited to these32protectedgroups and short controls.',trans_table,
 '5. The first-action hybrid controls distinguish the consequences of swapping an early geometric decision under matched subsequent noise. Supported effects would motivate another controlled low-level causal study, not automatic high-level RL. They do not isolate all variance changes or establish protein causal biology.',
 '6. No longer-horizon/hierarchical/readiness claim follows from a failed/uncertain K1pilot. If one arm passes all gates, replicate it across seeds/new protected goals and compare success per environment/J1compute before H8. If the critic gate fails, investigate value calibration/replay feature adequacy in a bounded TRAIN-only study before SAC; no algorithm is silently substituted.',
 'The most informative next work is a prospectively redesigned admission diagnostic, not longer training: pair RWR teacher likelihoods on fixed identities/actions; separate continuation stopping from safety eligibility; stratify critic probes by remaining horizon and calibrate first-action expected returns using repeated stochastic suffixes. The current terminal holdout already improves over a constant predictor, so an inaccurate blanket conclusion that the critics cannot generalize is unwarranted. Keep this study NO-GO and all stopped states intact; do not lower its frozen thresholds.',
 '7. Native AAs were fixed throughout. Joint AAselection needs a new prospective categorical-control objective, transformed hybrid likelihood/KL checks, native-free generation tests and independent sequence/structure quality controls. No de novo sequence-design, foldability or stability claim is supported here.',
 'Stop all workers and preserve exact states; wait for a separate scientific decision. No training, evaluation or cleanup is authorized beyond this V7study.'
 ]))
 plots(logs,evaluations,conf,cal)
 finalize_provenance(seal,audit,recovery)

def plots(logs,evaluations,conf,cal):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
 for arm,rr in logs.items():
  if rr:
   for axes,key in [(ax[0,0],'loss'),(ax[0,1],'KL'),(ax[1,0],'gradient_norm')]:axes.plot([r['update'] for r in rr],[r[key] for r in rr],label=arm,linewidth=.7)
 for arm in ['B','C','D']:
  dd=[d for d in evaluations.values() if d['panel']=='DEV' and d['label'].startswith(arm) and d['label'][1:].isdigit()]
  dd.sort(key=lambda d:int(d['label'][1:]))
  if dd:ax[1,1].plot([int(d['label'][1:]) for d in dd],[d['K1']['success']/d['K1']['attempts'] for d in dd],marker='o',label=arm)
 for axes,title in zip(ax.flat,['TRAIN objective (arm-dependent)','TRAIN conditional KL','Preclip gradient norm','DEV K1 full-valid success']):
  axes.set_title(title);axes.set_xlabel('actor updates');axes.grid(alpha=.2)
  if axes.get_legend_handles_labels()[0]:axes.legend()
 fig.savefig(R/'plots/TRAINING_AND_DEV_CURVES.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
 for arm,panels in conf.items():
  for axes,panel in zip(ax,['CONFIRM','CONFIRM_SHIFTED']):
   if panel in panels:axes.plot([1,2,4,8,16,32],[panels[panel]['by_K'][str(k)]['success']['estimate'] for k in [1,2,4,8,16,32]],marker='o',label=arm)
 for axes,title in zip(ax,['Native confirmation','Generated-prefix confirmation']):
  axes.set_xscale('log',base=2);axes.set_xticks([1,2,4,8,16,32],labels=[1,2,4,8,16,32]);axes.set_xlabel('nested K');axes.set_ylabel('full-valid query success');axes.set_title(title);axes.set_ylim(0,1);axes.grid(alpha=.2)
  if axes.get_legend_handles_labels()[0]:axes.legend()
 fig.savefig(R/'plots/SUCCESS_VERSUS_K.png',dpi=150);plt.close(fig)
 if cal:
  fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True);ax.plot([r['update'] for r in cal['curves']],[r['loss'] for r in cal['curves']],marker='o');ax.set_xlabel('critic-only calibration updates');ax.set_ylabel('recorded batch twin-Q fit loss');ax.set_title('TRAIN known-return calibration');ax.grid(alpha=.2);fig.savefig(R/'plots/CRITIC_CALIBRATION.png',dpi=150);plt.close(fig)

def finalize_provenance(seal,audit,recovery):
 names=['FINAL_E017_V7_SCIENTIFIC_REPORT.md','TRAIN_TASK_AND_CANDIDATE_CENSUS.md','REWARD_AND_CONSTRAINT_SPECIFICATION.md','REWARD_WEIGHTED_LEARNING_RESULTS.md','SAC_TRAINING_AND_CRITIC_CALIBRATION.md','K1_AND_BEST_OF_K_COMPARISONS.json','GOAL_VALIDITY_KL_ANALYSIS.md','GENERATED_PREFIX_TRANSFER.md','TRAINING_AND_INFERENCE_COSTS.json','CHECKPOINT_AND_RECOVERY_MANIFEST.json','NEXT_CRISP_RESEARCH_DECISION.md']
 for p in (R/'plots').glob('*.png'):names.append(str(p.relative_to(R)))
 INPUTS[str(Path(__file__))]=sha(__file__)
 for p in [R/'AUXILIARY_CONTROL_PROTOCOL.json',R/'analysis_source/causal.py',R/'CAUSAL_DIAGNOSTICS.json',R/'RECOVERY_EXECUTION.json',R/'results/resource_watch.jsonl']:
  if p.exists():INPUTS[str(p)]=sha(p)
 if (R/'NUMERICAL_DENSITY_AND_BELLMAN_AUDIT.json').exists():INPUTS[str(R/'NUMERICAL_DENSITY_AND_BELLMAN_AUDIT.json')]=sha(R/'NUMERICAL_DENSITY_AND_BELLMAN_AUDIT.json')
 j1=Path('/mnt/d/Simone/proteinGen_data/e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt');actualj1=sha(j1);assert actualj1=='881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412';INPUTS[str(j1)]=actualj1
 for p in (R/'protocol_engineering_fix').rglob('*'):
  if p.is_file():INPUTS[str(p)]=sha(p)
 if (R/'REPLAY_PRECISION_DIAGNOSIS.json').exists():INPUTS[str(R/'REPLAY_PRECISION_DIAGNOSIS.json')]=sha(R/'REPLAY_PRECISION_DIAGNOSIS.json')
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not gpu.strip(),gpu
 processes=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  except (PermissionError,FileNotFoundError,ProcessLookupError):continue
  if any(s in cmd for s in ['scripts/e017_v7/study.py','v7_recovery_supervisor.py','v7_causal_staged.py']) and (p/'comm').read_text().startswith('python'):processes.append(dict(pid=int(p.name),cmd=cmd))
 assert not processes,processes
 save('PROVENANCE_SHA256.json',dict(schema='E017_V7_final_provenance_v1',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_source_commit=seal['source'],final_analysis_script_SHA256=sha(__file__),input_SHA256=INPUTS,output_SHA256={n:sha(R/n) for n in names},parent_SHA256=audit['parent_SHA256'],parent_actor_fingerprint=audit['parent_actor_state_digest'],J1_tensor_SHA=audit['J1_tensor_SHA'],checkpoint_SHA256={a:{role:rec['record']['sha256'] if rec else None for role,rec in states.items()} for a,states in recovery['checkpoints'].items()},environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,CUDA_build=torch.version.cuda,cache_and_temp={k:os.environ.get(k) for k in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH']},source_status=subprocess.check_output(['git','-C',str(R/'source'),'status','--short'],text=True)),D_free_bytes_before_provenance=__import__('shutil').disk_usage(R).free,active_research_workers=[],GPU_compute_processes=[],TEST_evaluated=False,J1_modified=False,high_level_updates=0,reachability_updates=0,external_folding=False,deleted_historical_artifacts=0,all_generated_artifacts_on_D=True,self_hash='not included to avoid recursive hash'))
 print(json.dumps(dict(published=names,D_free_bytes=__import__('shutil').disk_usage(R).free)),flush=True)
if __name__=='__main__':main()
