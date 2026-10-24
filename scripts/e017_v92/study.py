"""Bounded frozen-policy diagnostics; no optimizer is constructed."""
import collections,copy,json,math,runpy,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v92_prefix_feasibility_20261024_v1');m=runpy.run_path(str(R/'source/scripts/e017_v92/common.py'));globals().update({k:m[k] for k in ['OLD','V9','p','c','g','AA','SPEC','read','save','sha','ns','prefix_status','extend_checked','motif_info','motif_allowed','known_goal_dead','reconstruct','load_engine','policy_name','verify_engine']})
corrected_action=p['legacy']['corrected_action'];START=time.perf_counter()
def admission():
 free=__import__('shutil').disk_usage(R).free;assert free>2*2**30+32*2**20,('storage admission',free)
 if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<6*2**30
 if (R/'control/STOP').exists():raise RuntimeError('cooperative diagnostic STOP requested')
def world(goal,state=None,seed=0):
 w=c['PropertyWorld'](goal,SPEC,seed)
 if state is not None:w.state=copy.deepcopy(state);w.snapshots={}
 return w
def draws(goal,i):
 rng=np.random.default_rng(c['ns']('BASELINE',goal['goal_id'],i));return [(rng.random(),rng.standard_normal(3)) for _ in range(goal['N'])]
def proposal(logits,mu,ls,t,u,eps,allowed=None):
 lp=logits.double().log_softmax(-1);prob=lp.exp().cpu().numpy();Z=1.
 if allowed is not None:Z=float(prob[allowed].sum());assert Z>0;prob=prob*allowed/Z
 prob=prob/prob.sum();cdf=prob.cumsum();cdf[-1]=1.;a=int(np.searchsorted(cdf,u));geom,diagnostic=corrected_action(mu[a],ls[a],c['mask_for_step'](t),torch.tensor(eps,device='cuda',dtype=torch.float64));return AA[a],geom.cpu().tolist(),dict(latent=diagnostic['latent'],proposal_AA_logprob=float(lp[a]),proposal_geometry_logprob=diagnostic['represented_action_log_density'],AA_mask_mass=Z,masked_AA_logprob=float(lp[a])-math.log(Z),repairs=sum(diagnostic['adjusted_components']))
@torch.no_grad()
def generate(e,goal,pol,method,count=8):
 admission();ws=[world(goal,seed=c['ns']('BASELINE',goal['goal_id'],i)) for i in range(count)];streams=[draws(goal,i) for i in range(count)];rows=[dict(sequence='',geometry=[],latent=[],proposals=[],AA_KL=[],geometry_KL_per_active=[],AA_mask_masses=[],repairs=0,seed=c['ns']('BASELINE',goal['goal_id'],i),candidate_index=i,goal_id=goal['goal_id'],policy=pol,method=method,initial_prefix='EMPTY_CANONICAL_GAUGE',native_coordinates_provided=False,termination=None) for i in range(count)];live=list(range(count));start=time.perf_counter()
 for t in range(goal['N']):
  if not live:break
  logits,mu,ls,ref,_=e.distribution([ws[i] for i in live],policy_name(pol));ak,gk,_=p['joint_kl'](logits,mu,ls,ref,t);nextlive=[]
  for j,i in enumerate(live):
   w=ws[i];r=rows[i];allowed=motif_allowed(''.join(w.state.residues),goal) if method in ['M','VM'] else None
   if allowed is not None and not allowed.any():r['termination']='motif_mask_empty';continue
   accepted=False
   for k in range(8 if method in ['V','VM'] else 1):
    if k==0:u,eps=streams[i][t]
    else:rr=np.random.default_rng(ns('ALT',goal['goal_id'],i,t,k));u,eps=rr.random(),rr.standard_normal(3)
    try:
     aa,geom,info=proposal(logits[j],mu[j],ls[j],t,u,eps,allowed);after,check=extend_checked(w.state,aa,geom);r['proposals'].append(dict(t=t,k=k,aa=aa,geometry=geom,accepted=check['accepted'],reasons=check['reasons'],**info));r['repairs']+=info['repairs']
     if method not in ['V','VM'] or check['accepted']:
      w.state=m['append'](w.state,aa,geom);r['geometry'].append(geom);r['latent'].append(info['latent']);r['AA_KL'].append(float(ak[j]));r['geometry_KL_per_active'].append(float(gk[j]));r['AA_mask_masses'].append(info['AA_mask_mass']);accepted=True;break
    except (ValueError,RuntimeError,FloatingPointError) as error:r['termination']='numerical_failure:'+repr(error);break
   if accepted:nextlive.append(i)
   elif r['termination'] is None:r['termination']='no_immediately_valid_proposal_in_8_attempts'
  live=nextlive
 for w,r in zip(ws,rows):
  r['sequence']=''.join(w.state.residues);r['metrics']=c['score'](r['sequence'],w.state.ca.numpy(),goal,SPEC);r['final_prefix_status']=prefix_status(w.state);r['filtered_density_KL']=None if method in ['V','VM'] else 'AA mask exact; Gaussian proposal KL separate';r['elapsed_group_seconds']=time.perf_counter()-start
 return rows
def publish_group(e,panel,pol,method,index,goal):
 name=f'results/{panel}_{pol}_{method}_{index:03}.json';path=R/name
 if path.exists():assert sha(path)==read(path.with_suffix('.sha.json'))['sha256'];return read(path)['records']
 before=dict(e.calls);rows=generate(e,goal,pol,method);save(name,dict(goal=goal,records=rows,calls_delta={k:e.calls[k]-before.get(k,0) for k in e.calls},no_optimizer=True));save(str(Path(name).with_suffix('.sha.json')),dict(sha256=sha(path)));print('GROUP',panel,pol,method,index,'hard',sum(r['metrics']['hard_valid'] for r in rows),'full',sum(r['metrics']['success'] for r in rows),flush=True);return rows
def tests(e,goals):
 checks=0;import itertools
 for motif in ['EF','FE','EE']:
  goal=dict(goals[0],exposed=[0,4],motif=motif)
  for n in range(4):
   for seq in map(''.join,itertools.product('EFG',repeat=n)):
    mask=motif_allowed(seq,goal)
    for aa in 'EFG':
     brute=any(motif in seq+aa+''.join(suffix) for suffix in itertools.product('EFG',repeat=3-n));assert mask[AA.index(aa)]==brute;checks+=1
 orig,_=p['sample'](e,goals[0],SPEC,'C_FIXED',8);new=generate(e,goals[0],'C','U')
 for a,b in zip(orig,new):assert a['sequence']==b['sequence'] and a['geometry']==b['geometry'] and a['metrics']==b['metrics']
 saved=read(OLD/'results/TRAIN_C_FIXED_000.json')['records']
 for a,b in zip(saved,new):assert a['sequence']==b['sequence'] and a['geometry']==b['geometry'] and a['metrics']==b['metrics']
 save('CORRECTNESS_TESTS.json',dict(PASS=True,DFA_checks=checks,unchanged_sampler_bit_exact_8=True,historical_TRAIN_replay_exact=True,final_N_clash_budget=1,global_Rg_not_prefix_monotone=True,no_optimizer=True));print('TESTS_PASS',checks,flush=True)
def step_trials(e,pol,goal,state,condition,trials=16):
 if not prefix_status(state)['valid']:return None
 logits,mu,ls,ref,_=e.distribution([world(goal,state)],policy_name(pol));rows=[]
 for k in range(trials):
  rr=np.random.default_rng(ns('ONE_STEP',goal['goal_id'],state.t,k));aa,geom,info=proposal(logits[0],mu[0],ls[0],state.t,rr.random(),rr.standard_normal(3));after,check=extend_checked(state,aa,geom);rows.append(dict(aa=aa,geometry=geom,valid=check['accepted'],reasons=check['reasons'],new_clash_pairs=check.get('new_clash_pairs'),**info))
 forced=[]
 for aa in sorted(set(goal['motif'])):
  a=AA.index(aa);probs=logits[0].double().softmax(-1);good=0;reasons=collections.Counter()
  for k in range(8):
   rr=np.random.default_rng(ns('FORCED_AA',goal['goal_id'],state.t,aa,k));geo,_=corrected_action(mu[0,a],ls[0,a],c['mask_for_step'](state.t),torch.tensor(rr.standard_normal(3),device='cuda',dtype=torch.float64));_,check=extend_checked(state,aa,geo.cpu().tolist());good+=check['accepted'];reasons.update(check['reasons'])
  forced.append(dict(aa=aa,policy_probability=float(probs[a]),valid=good,total=8,reasons=dict(reasons),conditional_AA_probe_not_joint_policy=True))
 return dict(policy=pol,goal_id=goal['goal_id'],identity=goal['witness_identity'],condition=condition,t=state.t,goal_type='exposed_before_buried' if goal['exposed'][0]<goal['buried'][0] else 'buried_before_exposed',motif_region=goal['exposed'][0]<=state.t<goal['exposed'][1],prefix_sequence=''.join(state.residues),prefix_status=prefix_status(state),trials=rows,forced_motif_AA=forced)
def one_step(e,pol,goals):
 dest=R/f'ONE_STEP_{pol}.json'
 if dest.exists():return
 witnesses=read(V9/'manifests/GOAL_WITNESSES_PRIVATE.json');allrows=[];omitted=[]
 for gi,goal in enumerate(goals[:6]):
  ownpath=R/f'results/TRAIN_{pol}_U_{gi:03}.json'
  if ownpath.exists():records=read(ownpath)['records']
  elif pol in ['A','B']:records=read(V9/f'results/TRAIN_{"A_J1" if pol=="A" else "B_V3"}_{gi:03}.json')['records']
  else:records=read(OLD/f'results/TRAIN_C_FIXED_{gi:03}.json')['records']
  common=read(OLD/f'results/TRAIN_C_FIXED_{gi:03}.json')['records'];wit=witnesses[goal['goal_id']]
  for t in [8,16,32,48]:
   native=c['BuildState'](tuple(wit['sequence'][:t]),torch.tensor(wit['coordinates'][:t],dtype=torch.float64),64)
   for condition,pool in [('own_generated',records),('common_C_generated',common),('native_TRAIN_control',None)]:
    selected=native if pool is None else next((reconstruct(r,64,t) for r in pool if len(r['geometry'])>=t and prefix_status(reconstruct(r,64,t))['valid']),None)
    if selected is None:omitted.append(dict(goal_id=goal['goal_id'],t=t,condition=condition,reason='no valid prefix among first8; not replaced'));continue
    result=step_trials(e,pol,goal,selected,condition)
    if result:allrows.append(result)
    else:omitted.append(dict(goal_id=goal['goal_id'],t=t,condition=condition,reason='native/control prefix violates unchanged rules'))
 save(dest.name,dict(records=allrows,omitted=omitted,selection='first prefix-valid candidate among8, independent of future outcome',TRAIN_only=True));print('ONE_STEP',pol,len(allrows),flush=True)
def viability(e,goals):
 if (R/'FUTURE_VIABILITY.json').exists():return
 records=[];selection=[]
 for gi,goal in enumerate(goals[:8]):
  pool=read(OLD/f'results/TRAIN_C_FIXED_{gi:03}.json')['records'];chosen=None
  for t in [8,16]:
   for ci,r in enumerate(pool):
    state=reconstruct(r,64,t)
    if prefix_status(state)['valid'] and not known_goal_dead(state,goal):chosen=(state,ci);break
   if chosen:break
  if not chosen:continue
  state,ci=chosen;logits,mu,ls,_,_=e.distribution([world(goal,state)],'C_FIXED');actions=[]
  for k in range(16):
   rr=np.random.default_rng(ns('FUTURE_ACTION',goal['goal_id'],state.t,k));aa,geom,info=proposal(logits[0],mu[0],ls[0],state.t,rr.random(),rr.standard_normal(3));after,check=extend_checked(state,aa,geom)
   if check['accepted']:actions.append((k,aa,geom,after))
   if len(actions)==4:break
  if len(actions)<2:continue
  selection.append(dict(goal_id=goal['goal_id'],identity=goal['witness_identity'],prefix_candidate=ci,t=state.t,actions_considered=k+1,actions_retained=len(actions)))
  for ai,aa,geom,after in actions:
   for split in ['FIT','CONFIRM']:
    ws=[world(goal,after,ns('FUTURE_CONTINUATION',goal['goal_id'],split,j)) for j in range(8)];rolls=p['continue_worlds'](e,ws,'C_FIXED')
    for j,row in enumerate(rolls):records.append(dict(goal_id=goal['goal_id'],identity=goal['witness_identity'],t=state.t,action_index=ai,aa=aa,geometry=geom,split=split,replicate=j,seed=ns('FUTURE_CONTINUATION',goal['goal_id'],split,j),hard=row['metrics']['hard_valid'],backbone=row['metrics']['backbone_valid'],full=row['metrics']['success'],reward=row['metrics']['reward'],exposed=row['metrics'].get('exposed_mean'),buried=row['metrics'].get('buried_mean'),contrast=row['metrics'].get('contrast')))
  print('VIABILITY',len(selection),'records',len(records),flush=True)
  if len(selection)==2:break
 save('FUTURE_VIABILITY.json',dict(prefixes=selection,records=records,continuation_policy='C_FIXED',independent_FIT_CONFIRM=True,common_random_numbers_across_actions=True,no_goal_reachability_proof=True))
def main():
 torch.set_num_threads(1);admission();stage=sys.argv[1];pol=sys.argv[2] if len(sys.argv)>2 else 'C';goals=read(OLD/'manifests/TRAIN_GOALS.json')['goals'];e,_=load_engine(pol)
 if stage=='engineering':tests(e,goals)
 elif stage=='train':
  if pol in ['C','D100']:
   for i,goal in enumerate(goals[:8]):
    for method in ['U','V','M','VM']:publish_group(e,'TRAIN',pol,method,i,goal)
  one_step(e,pol,goals)
  if pol=='C':viability(e,goals)
 elif stage=='confirm':
  protocol=read(R/'FROZEN_PROTOCOL.json');assert protocol['frozen_before_confirmation'];assert read(R/'TRAIN_ADMISSION.json')['PASS'];goals=read(OLD/'manifests/CONFIRM_GOALS.json')['goals']
  for i,goal in enumerate(goals):
   for method in (['V','M','VM'] if pol=='C' else ['U','V','VM']):publish_group(e,'CONFIRM',pol,method,i,goal)
 verify_engine(e);save(f'COMPLETED_{stage}_{pol}.json',dict(complete=True,elapsed_seconds=time.perf_counter()-START,calls=dict(e.calls),peak_GPU_bytes=torch.cuda.max_memory_allocated(),free_bytes=__import__('shutil').disk_usage(R).free,J1_fingerprint=e.jfp,actor_fingerprint=e.audit_actor_fingerprint,parent_fingerprint=e.fp,actor_updates=0,all_parameters_frozen=True));print('COMPLETE',stage,pol,flush=True)
if __name__=='__main__':main()
