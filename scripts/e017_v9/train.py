"""Bounded critic-free on-policy RLOO; immutable J1 and independent parent."""
import copy,json,math,os,random,runpy,subprocess,sys,time,types
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1');p=runpy.run_path(str(R/'source/scripts/e017_v9/policy.py'));c=p['c'];read=c['read'];save=c['save'];sha=c['sha'];digest=c['state_digest'];ns=c['ns'];AA=c['AA']
from protein_joint_sac.crisp_v3.recovery import atomic_checkpoint,serialized_size
from protein_joint_sac.crisp_v3.density import log_prob
CFG=dict(schema='V9_RLOO_v1',seed=9917,algorithm='episodic on-policy score-function RLOO, no critics',group=8,N=64,lr=1e-6,weight_decay=0.,clip=5.,AA_KL_beta=.1,geometry_mean_active_KL_beta=.1,advantage='(own_reward-mean_other7)/fixed TRAIN goal SD; SD floor.05, clip[-5,5]',loss='-advantage*sum_active_joint_log_density/N, plus mean per-step analytic AA/geometry reference KL; no pathwise reward gradient',reward_discount=1.,optimizer='AdamW fresh per arm',schedule='constant LambdaLR; no old optimizer/cursor restored',update_ceilings={'JOINT':400,'AA_ONLY':50,'GEOMETRY_ONLY':50},milestones=[25,50,100,250,400],per_update_step_TR='maximum joint analytic KL over40 fixed freshly sampled prefix decisions <=.02, otherwise rollback including optimizer/scheduler',scientific_stops='nonfinite/repair; DEV hard/backbone validity declines>5pp; DEV AAmeanKL>.5 or geometrymeanactiveKL>1; at100 DEV K1>=parent plus latest32TRAINgroups meanreward>frozenTRAINmean; at250 DEV K1>parent; no later ceiling extension',selection='greatest DEV K1, then K8, then hardvalid count, then earlier update; initial parent eligible',confirmation='one final evaluation of selected checkpoints only; joint/AA/G familywise98.333333% paired identity interval; >=5pp K1 gain; preserved validity/diversity/KL and K8',source_commit=None)
def rngstate():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
def restore_rng(s):torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda']);np.random.set_state(s['numpy']);random.setstate(s['python'])
def cpu(v):
 if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
 if isinstance(v,dict):return {k:cpu(z) for k,z in v.items()}
 if isinstance(v,list):return [cpu(z) for z in v]
 if isinstance(v,tuple):return tuple(cpu(z) for z in v)
 return copy.deepcopy(v)
def setup(mode):
 e=p['Engine']();parent=copy.deepcopy(e.actor).eval().requires_grad_(False);e.actor.requires_grad_(True);e.actor.train();opt=torch.optim.AdamW(e.actor.parameters(),lr=CFG['lr'],weight_decay=0.)
 # Materialize exact zero Adam moments before sizing/checkpoint admission.
 for v in e.actor.parameters():opt.state[v]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(v),exp_avg_sq=torch.zeros_like(v))
 scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
 original=e.distribution
 @torch.no_grad()
 def distribution(self,worlds,policy):
  logits,mu,ls,ref,batch=original(worlds,'B_V3')
  if mode!='JOINT':
   pa,pm,pl=parent(batch)
   if mode=='AA_ONLY':mu,ls=pm,pl
   else:logits=pa
  return logits,mu,ls,ref,batch
 e.distribution=types.MethodType(distribution,e)
 return e,parent,opt,scheduler
def outputs(e,parent,batch,mode):
 logits,mu,ls=e.actor(batch)
 if mode!='JOINT':
  with torch.no_grad():pa,pm,pl=parent(batch)
  if mode=='AA_ONLY':mu,ls=pm,pl
  else:logits=pa
 return logits,mu,ls
def scales():
 return {read(f)['goal']['goal_id']:max(.05,float(np.std([z['metrics']['reward'] for z in read(f)['records']]))) for f in (R/'results').glob('TRAIN_B_V3_[0-9][0-9][0-9].json')}
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

def bundle(e,opt,scheduler,mode,update_index,counters,selected,stop_reason=None):
 return dict(schema='V9_full_actor_recovery_v1',actor=cpu(e.actor.state_dict()),optimizer=cpu(opt.state_dict()),scheduler=scheduler.state_dict(),AMP=None,RNG=rngstate(),arm=mode,update=update_index,counters=counters,selected=selected,scientific_stop=stop_reason,config=read(R/'RL_PROTOCOL_FROZEN.json'),parent_SHA256=c['PSHA'],J1_SHA256=c['J1SHA'],reward_SHA256=sha(R/'REWARD_SPECIFICATION.json'),task_SHA256={k:sha(R/f'manifests/{k}_GOALS.json') for k in ['TRAIN','DEV','CONFIRM']},deterministic_sampler=dict(order=np.random.default_rng(9917).permutation(32).tolist(),next_index=update_index%32,namespace='RL_'+str(update_index+1)),pending_rollout=None)
def checkpoint(e,opt,scheduler,mode,u,counters,selected,reason=None):
 root=R/'checkpoints'/mode;root.mkdir(exist_ok=True);state=bundle(e,opt,scheduler,mode,u,counters,selected,reason);record=atomic_checkpoint(root/f'update_{u:04}.pt',state,c['POLICY']);save(f'checkpoints/{mode}/LATEST.json',dict(**record,update=u,selected=selected,stop_reason=reason));return record

def evaluate(e,mode,label,panel='DEV'):
 spec=read(R/'REWARD_SPECIFICATION.json');goals=read(R/f'manifests/{panel}_GOALS.json')['goals'];out=[]
 for i,g in enumerate(goals):
  records,_=p['sample'](e,g,spec,'B_V3',8,'BASELINE');name=f'results/{mode}_{label}_{panel}_{i:03}.json';save(name,dict(goal=g,records=records));out.extend(records)
 metrics=[r['metrics'] for r in out];bygoal=[out[i*8:(i+1)*8] for i in range(len(goals))];aakl=[z for r in out for z in r['AA_KL']];gkl=[z for r in out for z in r['geometry_KL_per_active'][1:]];return dict(K1=sum(g[0]['metrics']['success'] for g in bygoal),K8=sum(any(r['metrics']['success'] for r in g) for g in bygoal),goals=len(goals),backbone_valid=float(np.mean([m['backbone_valid'] for m in metrics])),hard_valid=float(np.mean([m['hard_valid'] for m in metrics])),reward=float(np.mean([m['reward'] for m in metrics])),AA_KL=float(np.mean(aakl)),geometry_KL=float(np.mean(gkl)),sequence_unique=len({r['sequence'] for r in out}),trajectories=len(out))
def base_dev():
 groups=[read(f)['records'] for f in sorted((R/'results').glob('DEV_B_V3_[0-9][0-9][0-9].json'))];assert len(groups)==8;ms=[r['metrics'] for g in groups for r in g];return dict(K1=sum(g[0]['metrics']['success'] for g in groups),K8=sum(any(r['metrics']['success'] for r in g) for g in groups),backbone_valid=float(np.mean([m['backbone_valid'] for m in ms])),hard_valid=float(np.mean([m['hard_valid'] for m in ms])))
def engineering():
 assert read(R/'LOW_LEVEL_ADMISSION.json')['PASS'];assert not (R/'RL_PROTOCOL_FROZEN.json').exists();CFG['source_commit']=subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip();CFG['TRAIN_goal_reward_scales']=scales();CFG['reward_SHA256']=sha(R/'REWARD_SPECIFICATION.json');CFG['design_SHA256']=sha(R/'SCIENTIFIC_DESIGN_INITIAL.json');save('RL_PROTOCOL_FROZEN.json',CFG)
 e,parent,opt,scheduler=setup('JOINT');spec=read(R/'REWARD_SPECIFICATION.json');goal=read(R/'manifests/TRAIN_GOALS.json')['goals'][0];initial=digest(e.actor.state_dict());state=bundle(e,opt,scheduler,'ENGINEERING',0,dict(trajectories=0,residue_decisions=0),None);size=serialized_size(state);admission=c['POLICY'].require(R,12*size,near_growth=128*2**20);path=R/'checkpoints/ENGINEERING_INITIAL.pt';record=atomic_checkpoint(path,state,c['POLICY']);a,rows=update(e,parent,opt,scheduler,'JOINT',goal,spec,1,'ENGINEERING');afterA=digest(dict(actor=e.actor.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict()));loaded=torch.load(path,map_location='cpu',weights_only=False);e.actor.load_state_dict(loaded['actor']);opt.load_state_dict(loaded['optimizer']);scheduler.load_state_dict(loaded['scheduler']);restore_rng(loaded['RNG']);b,again=update(e,parent,opt,scheduler,'JOINT',goal,spec,1,'ENGINEERING');afterB=digest(dict(actor=e.actor.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict()));assert afterA==afterB and [r['geometry'] for r in rows]==[r['geometry'] for r in again] and a==b;assert initial!=digest(e.actor.state_dict());e.prior.verify();assert digest(parent.state_dict())==e.fp;save('RL_ENGINEERING.json',dict(PASS=True,checkpoint=record,complete_checkpoint_bytes=size,storage_admission_12_checkpoint_slots=admission,exact_optimizer_scheduler_RNG_replay=True,exact_selfgenerated_rollout_replay=True,first_gradient=a,engineering_updates=2,production_updates=0,engineering_weights_discarded=True,J1_fingerprint=e.jfp,immutable_parent_fingerprint=e.fp,peak_GPU_bytes=torch.cuda.max_memory_allocated()))
 print('RL_ENGINEERING_PASS',flush=True)

def run(mode,resume=False):
 assert read(R/'RL_ENGINEERING.json')['PASS'];assert mode in ['JOINT','AA_ONLY','GEOMETRY_ONLY'];root=R/'checkpoints'/mode;assert resume or not (root/'LATEST.json').exists(),'explicit recovery path required';e,parent,opt,scheduler=setup(mode);spec=read(R/'REWARD_SPECIFICATION.json');goals=read(R/'manifests/TRAIN_GOALS.json')['goals'];order=np.random.default_rng(9917).permutation(32);base=base_dev();frozen=read(R/'LOW_LEVEL_ADMISSION.json');history=[];evaluations=[];selected=dict(update=0,metrics=base,checkpoint='PARENT');counters=dict(trajectories=0,residue_decisions=0,accepted_actor_updates=0,distinct_TRAIN_identities=0);seen=set();
 if resume:
  assert not (R/(mode+'_COMPLETED.json')).exists(),'completed/stopped arm cannot automatically resume'
  meta=read(root/'LATEST.json');assert sha(meta['path'])==meta['sha256'];state=torch.load(meta['path'],map_location='cpu',weights_only=False);assert digest(state)==meta['state_digest'] and state['scientific_stop'] is None;assert state['parent_SHA256']==c['PSHA'] and state['reward_SHA256']==sha(R/'REWARD_SPECIFICATION.json')
  e.actor.load_state_dict(state['actor']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);restore_rng(state['RNG']);counters=state['counters'];selected=state['selected'];first=state['update']+1
  trace=R/f'logs/{mode}_TRAIN.jsonl';history=[json.loads(line) for line in trace.read_text().splitlines() if json.loads(line)['update']<first] if trace.exists() else []
  if trace.exists():
   trace.rename(trace.with_name(trace.name+'.before_recovery_'+str(time.time_ns())));trace.write_text(''.join(json.dumps(v)+'\n' for v in history))
  seen={v['identity'] for v in history};milestone=R/(mode+'_MILESTONES.json');evaluations=[v for v in read(milestone)['evaluations'] if v['update']<first] if milestone.exists() else []
  save(mode+'_RECOVERY_EVENT.json',dict(checkpoint=meta,exact_next_update=first,archived_uncommitted_trace=True))
 else:
  checkpoint(e,opt,scheduler,mode,0,counters,selected);first=1
 start=time.perf_counter();reason=None;u=first-1
 for u in range(first,CFG['update_ceilings'][mode]+1):
  
  if (R/'PAUSE_REQUEST.json').exists():reason='COOPERATIVE_PAUSE';u-=1;break
  goal=goals[order[(u-1)%32]]
  try:
   c['POLICY'].require(R,224*2**20);row,_=update(e,parent,opt,scheduler,mode,goal,spec,u)
  except (ValueError,FloatingPointError,RuntimeError,OSError) as ex:reason='NUMERICAL_OR_RESOURCE_STOP: '+repr(ex);u-=1;break
  counters['trajectories']+=8;counters['residue_decisions']+=512;counters['accepted_actor_updates']=u;seen.add(goal['witness_identity']);counters['distinct_TRAIN_identities']=len(seen);history.append(row)
  with (R/f'logs/{mode}_TRAIN.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  save(mode+'_LIVE_STATUS.json',dict(status='RUNNING',pid=os.getpid(),update=u,counters=counters,latest=row,automatic_restart=False))
  if u%10==0:print('RLOO',mode,u,row['reward_mean'],flush=True)
  if u in CFG['milestones']:
   val=evaluate(e,mode,f'U{u:04}');evaluations.append(dict(update=u,**val));rank=lambda v:(v['K1'],v['K8'],v['hard_valid']);prior=selected['metrics']
   if val['backbone_valid']>=base['backbone_valid']-.05 and val['hard_valid']>=base['hard_valid']-.05 and val['AA_KL']<=.5 and val['geometry_KL']<=1 and rank(val)>rank(prior):selected=dict(update=u,metrics=val,checkpoint=str(root/f'update_{u:04}.pt'))
   if val['backbone_valid']<base['backbone_valid']-.05 or val['hard_valid']<base['hard_valid']-.05:reason='SCIENTIFIC_STOP_VALIDITY_DECLINE'
   elif val['AA_KL']>.5 or val['geometry_KL']>1:reason='SCIENTIFIC_STOP_KL'
   elif u==100 and (val['K1']<base['K1'] or np.mean([h['reward_mean'] for h in history[-32:]])<=frozen['reward_mean']):reason='SCIENTIFIC_STOP_100_NO_TRAIN_GAIN_OR_DEV_K1_DECLINE'
   elif u==250 and val['K1']<=base['K1']:reason='SCIENTIFIC_STOP_250_NO_DEV_K1_GAIN'
   checkpoint(e,opt,scheduler,mode,u,counters,selected,reason)
   save(mode+'_MILESTONES.json',dict(evaluations=evaluations,selected=selected))
   if reason:break
 e.prior.verify();assert digest(parent.state_dict())==e.fp
 latest=read(root/'LATEST.json')
 if latest['update']!=u:checkpoint(e,opt,scheduler,mode,u,counters,selected,reason or 'BOUNDED_CEILING_REACHED')
 save(mode+'_COMPLETED.json',dict(status='STOPPED',stop_reason=reason or 'BOUNDED_CEILING_REACHED',updates=u,counters=counters,selected=selected,milestones=evaluations,elapsed_seconds=time.perf_counter()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),calls=dict(e.calls),J1_fingerprint=e.jfp,parent_fingerprint=e.fp,automatic_restart=False))
 save(mode+'_LIVE_STATUS.json',dict(status='STOPPED',update=u,counters=counters,stop_reason=reason or 'BOUNDED_CEILING_REACHED',automatic_restart=False));print('COMPLETE',mode,u,reason,flush=True)
if __name__=='__main__':
 if sys.argv[1]=='engineering':engineering()
 elif sys.argv[1].startswith('resume_'):run(sys.argv[1][7:],resume=True)
 else:run(sys.argv[1])
