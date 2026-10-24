"""V9.1 actor-only RLOO with complete compact rollout evidence and durable replay."""
import copy,functools,hashlib,json,math,os,random,runpy,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1');p=runpy.run_path(str(R/'source/scripts/e017_v91/policy.py'));c=p['c'];read=c['read'];save=c['save'];sha=c['sha'];digest=c['state_digest'];ns=c['ns'];AA=c['AA'];original_sample=p['sample']
from protein_joint_sac.crisp_v3.recovery import atomic_checkpoint,serialized_size
from protein_joint_sac.crisp_v3.density import log_prob

def receipt_guard():
 test=read(R/'GOAL_OBSERVABILITY_TESTS.json');assert test['PASS'] and all(sha(R/'source/scripts/e017_v91'/n)==h for n,h in test['source_SHA256'].items());assert read(R/'CONTROLLABILITY_ADMISSION.json')['PASS'];protocol=read(R/'FROZEN_STUDY_PROTOCOL.json');assert sha(R/'source/scripts/e017_v91/goals.py')==protocol['goal_schema_SHA256'];assert sha(R/'REWARD_SPECIFICATION.json')==protocol['task_reward_unchanged_SHA256'];assert c['LIMITS']==read(R/'REWARD_SPECIFICATION.json')['hard_backbone_limits'];return protocol

def setup():
 torch.manual_seed(9917);np.random.seed(9917);random.seed(9917);e=p['Engine']();e.actor.requires_grad_(True);e.actor.train();opt=torch.optim.AdamW(e.actor.parameters(),lr=1e-6,weight_decay=0.)
 for v in e.actor.parameters():opt.state[v]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(v),exp_avg_sq=torch.zeros_like(v))
 return e,e.parent,opt,torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)

@functools.lru_cache(None)
def scales():return {read(path)['goal']['goal_id']:max(.05,float(np.std([r['metrics']['reward'] for r in read(path)['records']]))) for path in (R/'results').glob('TRAIN_C_FIXED_[0-9][0-9][0-9].json')}

def cached_sample(e,goal,spec,policy,count,namespace):
 if not (namespace.startswith('RL_') or namespace.startswith('ENGINEERING_')):return original_sample(e,goal,spec,policy,count,namespace)
 root=R/'rollouts';root.mkdir(exist_ok=True);path=root/(namespace+'.json');fingerprint=digest(e.actor.state_dict());binding=dict(behavior_actor_fingerprint=fingerprint,goal_SHA256=sha(R/'manifests/TRAIN_GOALS.json'),reward_SHA256=sha(R/'REWARD_SPECIFICATION.json'),schema_SHA256=sha(R/'source/scripts/e017_v91/goals.py'),goal_id=goal['goal_id'],namespace=namespace,parent_SHA256=c['PSHA'])
 if path.exists():
  meta=read(path.with_suffix('.sha.json'));assert sha(path)==meta['sha256'];saved=read(path);assert saved['binding']==binding,'cached rollout is not on-policy for recovered weights';return saved['records'],[]
 c['POLICY'].require(R,2*2**20);rows,worlds=original_sample(e,goal,spec,policy,count,namespace)
 for row in rows:row['policy']='D_RL_repaired';row['behavior_actor_fingerprint']=fingerprint
 save(str(path.relative_to(R)),dict(binding=binding,goal=goal,records=rows,source='complete generated trajectories before optimizer update, no native prefix',unique_collected_trajectories=len(rows),residue_decisions=sum(len(r['geometry']) for r in rows)));save(str(path.with_suffix('.sha.json').relative_to(R)),dict(sha256=sha(path),committed=True,trajectories=len(rows),residue_decisions=sum(len(r['geometry']) for r in rows),goal_id=goal['goal_id'],namespace=namespace));return rows,worlds
p['sample']=cached_sample

def outputs(e,parent,batch,mode):
 assert mode=='JOINT';return e.actor(batch)


def rngstate():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())

def restore_rng(s):torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda']);np.random.set_state(s['numpy']);random.setstate(s['python'])

def cpu(v):
 if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
 if isinstance(v,dict):return {k:cpu(z) for k,z in v.items()}
 if isinstance(v,list):return [cpu(z) for z in v]
 if isinstance(v,tuple):return tuple(cpu(z) for z in v)
 return copy.deepcopy(v)

def update(e,parent,opt,scheduler,mode,goal,spec,update_index,namespace='RL'):
 e.actor.eval();rows,_=p['sample'](e,goal,spec,'B_V3',8,namespace+'_'+str(update_index))
 if any(r['repairs'] or r['support_failure'] or not r['metrics']['complete'] for r in rows):raise FloatingPointError('incomplete or repaired rollout; strict-density training stopped')
 rewards=np.array([r['metrics']['reward'] for r in rows]);adv=(rewards-(rewards.sum()-rewards)/7)/scales()[goal['goal_id']];adv=np.clip(adv,-5,5);coeff=torch.tensor(adv,device='cuda');worlds=[c['PropertyWorld'](goal,spec,r['seed']) for r in rows];opt.zero_grad(set_to_none=True);e.actor.train();losses=[];AAKL=[];GKL=[];probes=[];density_error=0.
 for t in range(64):
  batchprefix=p['collate']([w.state for w in worlds],'cuda')
  with torch.no_grad():ref=e.prior(batchprefix)
  batch=p['public_batch'](worlds,batchprefix,ref);logits,mu,ls=outputs(e,parent,batch,mode);idx=torch.tensor([AA.index(r['sequence'][t]) for r in rows],device='cuda');ar=torch.arange(8,device='cuda');mask=torch.tensor(c['mask_for_step'](t),device='cuda').expand(8,3);g=torch.tensor([r['geometry'][t] for r in rows],device='cuda',dtype=torch.float64);lpaa=logits.double().log_softmax(-1)[ar,idx];lpg=log_prob(mu[ar,idx].double(),ls[ar,idx].double(),g,mask);joint=lpaa+lpg;beh=torch.tensor([r['AA_logprob'][t]+r['geometry_logprob'][t] for r in rows],device='cuda');density_error=max(density_error,float((joint-beh).abs().max().detach()))
  aa,gkl,_=p['joint_kl'](logits,mu,ls,ref,t);term=(-coeff*joint).mean()/64+(.1*aa.mean()+.1*gkl.mean())/64
  if not torch.isfinite(term):raise FloatingPointError('nonfinite PG loss')
  term.backward();losses.append(float(term.detach()));AAKL.append(float(aa.mean().detach()));GKL.append(float(gkl.mean().detach()))
  if t in [0,16,32,48,63]:
   probe_worlds=[]
   for w in worlds:
    z=c['PropertyWorld'](goal,spec,w.seed);z.state=copy.deepcopy(w.state);probe_worlds.append(z)
   probes.append((probe_worlds,dict(logits=logits.detach().clone(),mu=mu.detach().clone(),ls=ls.detach().clone())))
  for w,r in zip(worlds,rows):w.append(r['sequence'][t],r['geometry'][t])
 if density_error>2e-5:raise ValueError('behavior/recomputed transformed density mismatch '+str(density_error))
 grad=float(torch.nn.utils.clip_grad_norm_(e.actor.parameters(),5.));assert math.isfinite(grad)
 before=cpu(e.actor.state_dict());opt_before=cpu(opt.state_dict());sched_before=copy.deepcopy(scheduler.state_dict());opt.step();scheduler.step();e.actor.eval();maxstep=0.
 with torch.no_grad():
  for ws,old in probes:
   logits,mu,ls,_,_=e.distribution(ws,'B_V3');_,_,joint=p['joint_kl'](logits,mu,ls,old,ws[0].state.t);
   if not torch.isfinite(joint).all():
    e.actor.load_state_dict(before);opt.load_state_dict(opt_before);scheduler.load_state_dict(sched_before);raise FloatingPointError('nonfinite post-step trust probe rolled back')
   maxstep=max(maxstep,float(joint.max()))
 if maxstep>.02:
  e.actor.load_state_dict(before);opt.load_state_dict(opt_before);scheduler.load_state_dict(sched_before);raise ValueError('trust-region update rolled back '+str(maxstep))
 return dict(update=update_index,goal_id=goal['goal_id'],identity=goal['witness_identity'],reward_mean=float(rewards.mean()),reward_SD=float(rewards.std()),full_success=sum(r['metrics']['success'] for r in rows),backbone_valid=sum(r['metrics']['backbone_valid'] for r in rows),hard_valid=sum(r['metrics']['hard_valid'] for r in rows),loss=sum(losses),gradient_norm=grad,clipped=grad>5,AA_KL=float(np.mean(AAKL)),geometry_KL=float(np.mean(GKL[1:])),step_joint_KL_max=maxstep,density_recompute_max_error=density_error,sequence_exposures=8,residue_decisions=512),rows

def train_state_digest(e,opt,scheduler):return digest(dict(actor=e.actor.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),RNG=rngstate()))
def counters(u):
 paths=sorted((R/'rollouts').glob('RL_[0-9]*.sha.json'));groups=[read(path) for path in paths];order=np.random.default_rng(9917).permutation(32);goals=read(R/'manifests/TRAIN_GOALS.json')['goals'];return dict(accepted_actor_updates=u,accepted_trajectories=8*u,accepted_residue_decisions=512*u,unique_collected_trajectories=sum(v['trajectories'] for v in groups),collected_residue_decisions=sum(v['residue_decisions'] for v in groups),distinct_TRAIN_identities=len({goals[order[j%32]]['witness_identity'] for j in range(u)}))
def bundle(e,opt,scheduler,u,selected,reason=None):
 protocol=receipt_guard();return dict(schema='V91_FULL_ACTOR_RECOVERY_1',actor=cpu(e.actor.state_dict()),optimizer=cpu(opt.state_dict()),scheduler=scheduler.state_dict(),AMP=None,RNG=rngstate(),update=u,counters=counters(u),selected=selected,stop_reason=reason,config=protocol,parent_SHA256=c['PSHA'],J1_SHA256=c['J1SHA'],parent_actor_fingerprint=e.fp,J1_fingerprint=e.jfp,goal_schema_SHA256=sha(R/'source/scripts/e017_v91/goals.py'),reward_SHA256=sha(R/'REWARD_SPECIFICATION.json'),task_SHA256={panel:sha(R/f'manifests/{panel}_GOALS.json') for panel in ['TRAIN','DEV','CONFIRM']},source_commit=subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip(),execution_source_SHA256={n:sha(R/'source/scripts/e017_v91'/n) for n in ['core.py','goals.py','policy.py','train.py']},dataset_provenance=read(p['OLD']/'INITIAL_INPUT_AUDIT.json'),sampler=dict(order=np.random.default_rng(9917).permutation(32).tolist(),next_index=u%32,next_update=u+1,next_namespace='RL_'+str(u+1),episode_stream='original V9|9917 hash namespace,goal,candidate; independent candidate random streams'),on_policy_rollouts=dict(root=str(R/'rollouts'),compact_actions_and_behavior_densities_persisted=True,pending=None),recovery_journal='logs/ACCEPTED_UPDATES.jsonl',training_state_digest=train_state_digest(e,opt,scheduler))
def checkpoint(e,opt,scheduler,u,selected,reason=None):
 root=R/'checkpoints/protein';root.mkdir(exist_ok=True);record=atomic_checkpoint(root/f'checkpoint_{u:04}.pt',bundle(e,opt,scheduler,u,selected,reason),c['POLICY']);meta=root/'RECOVERY_INDEX.json';index=read(meta) if meta.exists() else dict(schema='V91_LATEST_BEST_NAMED_1',owned={},no_deletions=True)
 index['owned'][str(u)]=record;index.update(latest=record,best=index['owned'].get(str(selected['update']),record),update=u,selected=selected,stop_reason=reason,automatic_restart=False);save(str(meta.relative_to(R)),index);return record

def engineering():
 protocol=receipt_guard();assert read(R/'INITIAL_QUALITY_ADMISSION.json')['PASS'];e,parent,opt,scheduler=setup();selected=dict(update=0,metrics=read(R/'INITIAL_QUALITY_ADMISSION.json')['C_DEV'],label='C_FIXED');state=bundle(e,opt,scheduler,0,selected);size=serialized_size(state)
 # Six full files cover initial,25,50,100,250 plus an unscheduled stop;
 # replay journals provide exact between-checkpoint recovery without deletion.
 budget=c['POLICY'].require(R,6*size+128*2**20);save('STORAGE_ADMISSION.json',dict(PASS=True,full_checkpoint_bytes=size,maximum_full_checkpoint_files=6,all_named_checkpoints_preserved=True,rollout_and_report_budget_bytes=128*2**20,policy=budget,already_existing_engineering_actor_bytes=sum(q.stat().st_size for q in (R/'checkpoints/engineering').glob('*.pt')),near_growth_is_additional=True))
 record=checkpoint(e,opt,scheduler,0,selected);spec=read(R/'REWARD_SPECIFICATION.json');goal=read(R/'manifests/TRAIN_GOALS.json')['goals'][0];a,rows=update(e,parent,opt,scheduler,'JOINT',goal,spec,1,'ENGINEERING_REPLAY');afterA=train_state_digest(e,opt,scheduler);loaded=torch.load(record['path'],map_location='cpu',weights_only=False);e.actor.load_state_dict(loaded['actor']);opt.load_state_dict(loaded['optimizer']);scheduler.load_state_dict(loaded['scheduler']);restore_rng(loaded['RNG']);b,again=update(e,parent,opt,scheduler,'JOINT',goal,spec,1,'ENGINEERING_REPLAY');afterB=train_state_digest(e,opt,scheduler);assert afterA==afterB and a==b and [r['geometry'] for r in rows]==[r['geometry'] for r in again];e.prior.verify();assert digest(parent.state_dict())==e.fp
 save('RECOVERY_ENGINEERING.json',dict(PASS=True,initial_checkpoint=record,full_checkpoint_bytes=size,bit_exact_model_optimizer_scheduler_RNG_replay=True,exact_cached_on_policy_replay=True,first_gradient=a,discarded_optimizer_executions=2,unique_collected_engineering_trajectories=8,replayed_cached_trajectories=8,residue_decisions=512,engineered_weights_not_used_for_production=True,peak_GPU_bytes=torch.cuda.max_memory_allocated(),source_SHA256=sha(R/'source/scripts/e017_v91/train.py')));print('V91_RECOVERY_ENGINEERING_PASS',flush=True)

def resolve_records(panel,policy):
 if policy in ['A_J1','B_PREV9'] and panel in ['TRAIN','DEV']:
  aliases=read(R/'REUSED_FROZEN_BASELINES.json')['aliases'][panel+'_'+policy];return [read(Path(v['path'])) for v in aliases]
 return [read(path) for path in sorted((R/'results').glob(f'{panel}_{policy}_[0-9][0-9][0-9].json'))]
def summary(groups):
 assert groups;rows=[row for v in groups for row in v['records']];m=[r['metrics'] for r in rows];return dict(goals=len(groups),trajectories=len(rows),K1=sum(v['records'][0]['metrics']['success'] for v in groups),K8=sum(any(r['metrics']['success'] for r in v['records']) for v in groups),backbone_valid=float(np.mean([r['backbone_valid'] for r in m])),hard_valid=float(np.mean([r['hard_valid'] for r in m])),reward=float(np.mean([r['reward'] for r in m])),AA_KL=float(np.mean([z for r in rows for z in r['AA_KL']])),geometry_KL=float(np.mean([z for r in rows for z in r['geometry_KL_per_active'][1:]])),unique_sequence_fraction=len({r['sequence'] for r in rows})/len(rows))
def quality():
 receipt_guard();groups=resolve_records('TRAIN','C_FIXED');assert len(groups)==32;rows=[row for v in groups for row in v['records']];metrics=summary(groups);reasons=[];goals_with_valid=sum(any(r['metrics']['hard_valid'] for r in v['records']) for v in groups);rew=np.array([r['metrics']['reward'] for r in rows]);C=summary(resolve_records('DEV','C_FIXED'));B=summary(resolve_records('DEV','B_PREV9'))
 if np.mean([r['metrics']['complete'] for r in rows])<.95:reasons.append('completion below95%')
 if metrics['backbone_valid']<.2 or metrics['hard_valid']<.1 or goals_with_valid<16:reasons.append('original TRAIN quality gate')
 if rew.std()<.03 or (rew>0).mean()<.1 or any(np.std([r['metrics'].get(key,0) for r in rows])<=1e-8 for key in ['exposed_mean','buried_mean','motif_utility']):reasons.append('uninformative reward/property variation')
 if any(r['repairs'] or r['support_failure'] or not np.isfinite(r['AA_logprob']+r['geometry_logprob']).all() for r in rows):reasons.append('strict density admission')
 if metrics['AA_KL']>.5 or metrics['geometry_KL']>1:reasons.append('original reference KL safeguard')
 if C['backbone_valid']<B['backbone_valid']-.05 or C['hard_valid']<B['hard_valid']-.05:reasons.append('repaired initial DEV validity outside original5pp allowance')
 save('INITIAL_QUALITY_ADMISSION.json',dict(PASS=not reasons,reasons=reasons,C_TRAIN=metrics,C_DEV=C,B_DEV=B,reward_SD=float(rew.std()),nonzero_reward_fraction=float((rew>0).mean()),goals_with_valid=goals_with_valid));print(json.dumps(read(R/'INITIAL_QUALITY_ADMISSION.json')))

def evaluate_model(e,label,panel='DEV'):
 spec=read(R/'REWARD_SPECIFICATION.json');goals=read(R/f'manifests/{panel}_GOALS.json')['goals'];out=[]
 for i,goal in enumerate(goals):
  rows,_=original_sample(e,goal,spec,'D_RL',8,'BASELINE');obj=dict(goal=goal,records=rows,policy='D_RL',label=label);name=f'results/{panel}_{label}_{i:03}.json';save(name,obj);save(str(Path(name).with_suffix('.sha.json')),dict(sha256=sha(R/name)));out.append(obj)
 return summary(out)

def run(resume=False):
 receipt_guard();assert read(R/'RECOVERY_ENGINEERING.json')['PASS'];assert sha(R/'source/scripts/e017_v91/train.py')==read(R/'RECOVERY_ENGINEERING.json')['source_SHA256'];assert not (R/'PRODUCTION_COMPLETED.json').exists();index=read(R/'checkpoints/protein/RECOVERY_INDEX.json');assert index['stop_reason'] is None;checkpoint_path=Path(index['latest']['path']);assert sha(checkpoint_path)==index['latest']['sha256'];state=torch.load(checkpoint_path,map_location='cpu',weights_only=False);assert digest(state)==index['latest']['state_digest'];assert all(sha(R/'source/scripts/e017_v91'/n)==h for n,h in state['execution_source_SHA256'].items());e,parent,opt,scheduler=setup();e.actor.load_state_dict(state['actor']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);restore_rng(state['RNG']);selected=state['selected'];start_u=state['update'];assert resume or start_u==0;goals=read(R/'manifests/TRAIN_GOALS.json')['goals'];order=np.random.default_rng(9917).permutation(32);spec=read(R/'REWARD_SPECIFICATION.json');base=read(R/'INITIAL_QUALITY_ADMISSION.json');B,C=base['B_DEV'],base['C_DEV'];journal=R/'logs/ACCEPTED_UPDATES.jsonl';committed=[json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else [];history=[v['metrics'] for v in committed if v['update']<=start_u];by_update={v['update']:v for v in committed};milestones=read(R/'MILESTONES.json')['evaluations'] if (R/'MILESTONES.json').exists() else [];reason=None;u=start_u;start=time.perf_counter()
 if committed and not resume:raise ValueError('existing journal requires explicit exact replay resume')
 for target in range(start_u+1,251):
  if (R/'PAUSE_REQUEST.json').exists():reason='COOPERATIVE_PAUSE';break
  goal=goals[order[(target-1)%32]]
  try:
   c['POLICY'].require(R,224*2**20);metrics,_=update(e,parent,opt,scheduler,'JOINT',goal,spec,target)
  except (ValueError,FloatingPointError,RuntimeError,OSError) as exc:reason='NUMERICAL_OR_RESOURCE_STOP: '+repr(exc);break
  digest_after=train_state_digest(e,opt,scheduler);entry=dict(update=target,metrics=metrics,training_state_digest=digest_after,rollout_path=str(R/'rollouts'/('RL_'+str(target)+'.json')),rollout_SHA256=sha(R/'rollouts'/('RL_'+str(target)+'.json')))
  if target in by_update:assert entry==by_update[target],'exact interrupted-run replay mismatch'
  else:
   with journal.open('a') as f:f.write(json.dumps(entry)+'\n');f.flush();os.fsync(f.fileno())
  u=target;history.append(metrics);save('LIVE_STATUS.json',dict(status='RUNNING',pid=os.getpid(),update=u,counters=counters(u),last=metrics,automatic_restart=False));
  if u%10==0:print('V91_RLOO',u,metrics['reward_mean'],flush=True)
  if u in [25,50,100,250]:
   existing=next((v for v in milestones if v['update']==u),None);val=evaluate_model(e,f'D_U{u:04}') if existing is None else {k:v for k,v in existing.items() if k!='update'}
   if existing is None:milestones.append(dict(update=u,**val))
   safeguards=val['backbone_valid']>=max(B['backbone_valid'],C['backbone_valid'])-.05 and val['hard_valid']>=max(B['hard_valid'],C['hard_valid'])-.05 and val['AA_KL']<=.5 and val['geometry_KL']<=1
   rank=lambda z:(z['K1'],z['K8'],z['hard_valid'])
   if safeguards and rank(val)>rank(selected['metrics']):selected=dict(update=u,metrics=val,label=f'D_U{u:04}')
   if not safeguards:reason='SCIENTIFIC_STOP_ORIGINAL_VALIDITY_OR_KL_SAFEGUARD'
   elif u==100 and (val['K1']<C['K1'] or np.mean([v['reward_mean'] for v in history[-32:]])<=base['C_TRAIN']['reward']):reason='SCIENTIFIC_STOP_100_NO_TRAIN_GAIN_OR_DEV_K1_DECLINE'
   elif u==250 and val['K1']<=C['K1']:reason='SCIENTIFIC_STOP_250_NO_DEV_K1_GAIN'
   checkpoint(e,opt,scheduler,u,selected,reason);save('MILESTONES.json',dict(evaluations=milestones,selected=selected))
   if reason:break
 e.verify(trained=True);latest=read(R/'checkpoints/protein/RECOVERY_INDEX.json')
 if latest['update']!=u:checkpoint(e,opt,scheduler,u,selected,reason or 'BOUNDED_250_CEILING')
 save('PRODUCTION_COMPLETED.json',dict(status='STOPPED',updates=u,counters=counters(u),stop_reason=reason or 'BOUNDED_250_CEILING',selected=selected,milestones=milestones,elapsed_seconds=time.perf_counter()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),calls=dict(e.calls),parent_fingerprint=e.fp,J1_fingerprint=e.jfp,automatic_restart=False,between_checkpoint_exact_replay_journal=True));save('LIVE_STATUS.json',dict(status='STOPPED',updates=u,stop_reason=reason or 'BOUNDED_250_CEILING',automatic_restart=False));print('V91_STOPPED',u,reason,flush=True)
if __name__=='__main__':
 if sys.argv[1]=='quality':quality()
 elif sys.argv[1]=='engineering':engineering()
 elif sys.argv[1]=='run':run()
 elif sys.argv[1]=='resume':run(True)
