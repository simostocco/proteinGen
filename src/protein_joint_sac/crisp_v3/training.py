"""Future training orchestration. All execution requires a separate scoped authorization."""
import copy,json,math,random,shutil,time,uuid,errno,sqlite3,warnings
from pathlib import Path
import torch
from torch.nn import functional as F
from protein_joint_sac.models.batched_bc import collate
from protein_joint_sac.geometry.ca_builder import next_ca
from .density import sample
from .networks import System,prepare
from .prior import FrozenPrior,SHA,file_sha,tensor_sha
from .optimization import Learner,supervised,finish
from .storage import Replay,digest,payload,restore
from .recovery import StoragePolicy,StoragePause,RollingRecovery,atomic_json
from .data import Population,CoverageSampler
from .goals import Goals
from .pipeline import native_goal_batch,observation,collect,replay_example
from .environment import World,goal_tokens,reconstruct

def seal_config(config):
 if config['reference_sha256']!=SHA:raise ValueError('reference mismatch')
 if config['profile'] not in ['development','research','large']:raise ValueError('profile')
 if config['stage'] not in ['supervised','low_sac','high_sac','reachability']:raise ValueError('stage')
 if config.get('shapes') is not None:
  with torch.device('meta'):System(config['profile'],config['shapes'])
 if config['updates']<1 or config['effective_AA_batch']<1:raise ValueError('update/batch')
 return digest(config)

def authorization(path,config):
 if path is None:raise PermissionError('Future substantive training requires separate authorization')
 a=json.load(open(path))
 import subprocess
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[3],text=True).strip()
 if a.get('source_commit')!=commit:raise PermissionError('authorization source commit mismatch')
 if a.get('config_sha256')!=seal_config(config) or a.get('reference_sha256')!=SHA or a.get('stage')!=config['stage'] or int(a.get('max_updates',0))<config['updates'] or not a.get('approved'):raise PermissionError('authorization must name exact configuration, stage, reference and update allowance')

def geometric_proximity(b,mu,ls,mask):
 """One-step guide only for an explicit target at the current next residue.
 This does NOT differentiate an ungenerated suffix or use hidden native atoms.
 """
 terms=[]
 for i in range(len(mu)):
  t=int(b['prefix']['t'][i]);N=int(b['prefix']['N'][i]);indices=(b['goals'][i,:,32]*N).round().long();eligible=b['goal_valid'][i]&b['goals'][i,:,33].bool()&(indices==t)
  if not eligible.any() or t<3 or not mask[i].all() or not b['prefix']['valid'][i,:t].all():continue
  g,_=sample(mu[i:i+1],ls[i:i+1],mask[i:i+1]);x=next_ca(b['prefix']['x'][i,:t],g[0]);target=b['goals'][i,eligible,:3].double()*50;dist=(x-target).square().sum(-1);terms.append((dist/(64+dist)).mean())
 return torch.stack(terms).mean() if terms else (mu.sum()+ls.sum())*0

def supervised_update(system,learner,prior,pop,sampler,geometry_sampler,goals,config):
 # Gradients accumulate over independent literal-prefix microbatches; one optimizer step.
 m=system.models['actor'];o=learner.optimizers['actor'];o.zero_grad(set_to_none=True);device=next(m.parameters()).device;B=config['effective_AA_batch'];micro=config['microbatch'];weights=config['loss_weights'];report={};parts=[]
 records=sampler.draw(B)
 for start in range(0,B,micro):
  subset=records[start:start+micro];p,aa,g,mask,_=pop.batch(subset,device);ref=prior(p);zero=torch.zeros(len(aa),1,36,device=device);v=torch.zeros(len(aa),1,device=device,dtype=torch.bool);b=prepare(p,ref,zero,v)
  loss,terms=supervised(system,b,aa,g,mask,{**weights,'geometry':0.,'goal':0.});scaled=loss*len(subset)/B;scaled.backward();parts.append(float(scaled.detach()))
 for _ in range(config['complete_geometry_decisions']):
  # Independent identity-first complete structure stream. Never supplied by AA-only samples.
  p,aa,g,mask,_=pop.batch(geometry_sampler.geometry(1),device);ref=prior(p);b=prepare(p,ref,torch.zeros(1,1,36,device=device),torch.zeros(1,1,device=device,dtype=torch.bool));loss,_=supervised(system,b,aa,g,mask,{**weights,'aa':0.,'goal':0.});scaled=loss/config['complete_geometry_decisions'];scaled.backward();parts.append(float(scaled.detach()))
 for _ in range(config['goal_demonstrations']):
  b,aa,g,mask=native_goal_batch(pop,goals,prior,device);loss,_=supervised(system,b,aa,g,mask,weights,lambda mu,ls,valid:geometric_proximity(b,mu,ls,valid));scaled=loss/config['goal_demonstrations'];scaled.backward();parts.append(float(scaled.detach()))
 report['gradient_groups']={k:math.sqrt(sum(float(p.grad.float().square().sum()) for p in group.parameters() if p.grad is not None)) for k,group in [('encoder',m.encoder),('AA',m.aa),('geometry',m.geometry)]};report['gradient_norm']=finish(o,m.parameters(),config['gradient_clip']);learner.steps['actor']+=1;learner.schedulers['actor'].step();report['loss']=sum(parts);report['AA_decisions']=B;return report

def validate_actor(system,prior,panel,device,config):
 """Read-only frozen caller-supplied native panel. No automatic TEST construction."""
 from protein_joint_sac.contracts import AA,BuildState
 from protein_joint_sac.data.scaled_supervised import internal
 m=system.models['actor'];was=m.training;m.eval();rows=[]
 with torch.no_grad():
  for r in panel:
   if r['split'] not in ['TRAIN','VALIDATION']:raise ValueError('TEST forbidden')
   seq=r['sequence'];x=torch.tensor(r['CA'],dtype=torch.float64);g=torch.tensor(internal(x.numpy()));N=len(seq)
   for t in r['positions']:
    p=collate([BuildState(tuple(seq[:t]),x[:t],N)],device);b=prepare(p,prior(p),torch.zeros(1,1,36,device=device),torch.zeros(1,1,dtype=torch.bool,device=device));aa=torch.tensor([AA.index(seq[t])],device=device);loss,terms=supervised(system,b,aa,g[t:t+1].to(device),torch.tensor([[t>=1,t>=2,t>=3]],device=device),config['loss_weights']);rows.append(dict(identity=r['identity'],split=r['split'],t=t,**{k:float(v) for k,v in terms.items()}))
 m.train(was);return rows

def bc_regularizer(system,prior,pop,sampler,geometry_sampler,config):
 device=next(system.parameters()).device;p,aa,g,mask,_=pop.batch(sampler.draw(config['BC_microbatch']),device);b=prepare(p,prior(p),torch.zeros(len(aa),1,36,device=device),torch.zeros(len(aa),1,device=device,dtype=torch.bool));loss,_=supervised(system,b,aa,g,mask,{**config['loss_weights'],'geometry':0.,'goal':0.});pg,yg,gg,mg,_=pop.batch(geometry_sampler.geometry(1),device);bg=prepare(pg,prior(pg),torch.zeros(1,1,36,device=device),torch.zeros(1,1,device=device,dtype=torch.bool));gloss,_=supervised(system,bg,yg,gg,mg,{**config['loss_weights'],'aa':0.,'goal':0.});return config['BC_weight']*loss+config['geometry_BC_weight']*gloss

def execute(config,root,grant,resume=None):
 authorization(grant,config);root=Path(root).resolve();D=Path('/mnt/d/Simone/proteinGen_data').resolve()
 if not root.is_relative_to(D):raise ValueError('all training artifacts must reside on D:')
 if config['stage']!='supervised' and not resume and not config.get('initial_checkpoint'):raise ValueError('later stages require an explicit full-state initialization checkpoint')
 root.mkdir(parents=True,exist_ok=True)
 policy=StoragePolicy(**config.get('storage_policy',{}))
 # Existing latest recovery resumes in its OWN run, avoiding abandoned live
 # replay copies. Historical/pre-v2 paths always fork and are never adopted.
 owned_resume=False
 if resume:
  candidate=Path(resume).absolute().parent.parent
  if (candidate/'STORAGE_OWNER.json').exists() and (candidate/'RECOVERY_INDEX.json').exists():
   index=json.load(open(candidate/'RECOVERY_INDEX.json'))
   if index['latest'] and index['latest']['checkpoint']==str(Path(resume).absolute()):run=candidate;owned_resume=True
 if not owned_resume:
  policy.startup(root);run=root/('run_'+uuid.uuid4().hex);run.mkdir()
 manager=RollingRecovery(run,policy,create=not owned_resume)
 manager._recover_pending();manager.collect_superseded()
 torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);torch.use_deterministic_algorithms(True)
 prior=FrozenPrior('cuda');system=System(config['profile'],config.get('shapes')).cuda();learner=Learner(system,**config['optimizer']);pop=Population();sampler=CoverageSampler(pop,config['seed']);geometry_sampler=CoverageSampler(pop,config['seed']+1);goals=Goals(pop,config['seed']+2,json.load(open(config['native_limits'])));samplers=dict(AA=sampler,geometry=geometry_sampler,goals=goals);replay=Replay(run/'replay.sqlite');counters={'updates':0,'episodes':0};limits=json.load(open(config['native_limits']));replay_rng=random.Random(config['seed']+3)
 data_manifest=pop.membership();effective_config={**config,'data_manifest':data_manifest,'data_input_hashes':pop.input_hashes(),'native_limits_sha256':file_sha(config['native_limits']),'validation_panel_sha256':file_sha(config['validation_panel']) if config.get('validation_panel') else None,'source_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[3],text=True).strip()}
 if config.get('initial_checkpoint') and not resume:
  initial=config['initial_checkpoint']
  if file_sha(initial['path'])!=initial['sha256']:raise ValueError('initialization checkpoint SHA')
  parent=torch.load(initial['path'],map_location='cpu',weights_only=False)
  if parent['j1_sha256']!=SHA or parent['profile']!=system.profile:raise ValueError('initialization reference/profile')
  if parent['config'].get('data_input_hashes')!=effective_config['data_input_hashes']:raise ValueError('stage initialization data provenance')
  system.load_state_dict(parent['system'],strict=True)
  for k,o in learner.optimizers.items():
   o.load_state_dict(parent['optimizers'][k])
   for g in o.param_groups:g['lr']=config['optimizer']['lr']
  for k,v in samplers.items():v.load_state_dict(parent['samplers'][k])
  counters['episodes']=parent['counters'].get('episodes',0)
  if parent['replay_snapshot']:
   snap=parent['replay_snapshot'];assert file_sha(snap['path'])==snap['sha256'];replay.close();import sqlite3
   source=sqlite3.connect(snap['path']);dest=sqlite3.connect(run/'replay.sqlite');source.backup(dest);source.close();dest.close();replay=Replay(run/'replay.sqlite')
 if resume:
  state=restore(resume,system,learner,samplers,effective_config);counters=state['counters'];replay_rng.setstate(counters['replay_rng']);replay.close();import sqlite3
  assert file_sha(state['replay_snapshot']['path'])==state['replay_snapshot']['sha256'];source=sqlite3.connect(state['replay_snapshot']['path']);dest=sqlite3.connect(run/'replay.sqlite');source.backup(dest);source.close();dest.close();replay=Replay(run/'replay.sqlite')
 if owned_resume:
  if json.load(open(run/'CONFIG.json'))!=effective_config:raise ValueError('same-run resume configuration mismatch')
 else:(run/'CONFIG.json').write_text(json.dumps(effective_config,indent=2))
 history=open(run/'updates.jsonl','a' if owned_resume else 'x')
 replay.set_write_guard(lambda n:manager.runway(replay,n))
 last_checkpoint_time=time.monotonic()
 def checkpoint(validation=None,near_growth=None):
  nonlocal last_checkpoint_time
  counters['replay_rng']=replay_rng.getstate();prior.verify(learner.optimizers.values())
  out=manager.publish(replay,lambda snap:payload(system,learner,samplers,effective_config,snap,counters),counters['updates'],validation,near_growth);last_checkpoint_time=time.monotonic();return out
 at_boundary=True
 try:
  if manager.current()['latest'] is None:checkpoint()
  for update in range(counters['updates'],config['updates']):
   manager.runway(replay);at_boundary=False
   t=time.monotonic();stage=config['stage'];system.train();goals.stage=min(len(__import__('protein_joint_sac.crisp_v3.goals',fromlist=['CURRICULUM']).CURRICULUM)-1,update//config['curriculum_every'])
   with torch.autocast('cuda',dtype=torch.bfloat16):
    if stage=='supervised':result=supervised_update(system,learner,prior,pop,sampler,geometry_sampler,goals,config)
    else:
     if update%config['collect_every']==0:
      task=goals.draw();policy=tensor_sha(system.models['actor'])+':'+tensor_sha(system.models['high_actor']);ep=collect(system,prior,task,limits,replay,policy,config['seed']+counters['episodes'],config['operational_events']);counters['episodes']+=1
     desired='high' if stage=='high_sac' else 'low'
     try:r=replay.sample(1,replay_rng,level=desired)[0]
     except ValueError:r=None
     if r is None:result={'skipped':'no matching resolved replay'}
     else:
      b,nb,worlds=replay_example(prior,replay,r,limits,'cuda');a=r['action'];high=stage=='high_sac';aa=torch.tensor([a['option'] if high else a['aa']],device='cuda');g=torch.tensor([a.get('geometry',[3.8,1.8,0.])],device='cuda',dtype=torch.float64);mask=torch.ones(1,3,device='cuda',dtype=torch.bool)
      if stage=='reachability':
       ep,events=replay.read(r['episode']);final=reconstruct(ep['initial'],events,ep['goal'],limits,ep['initial'].get('seed',1))
       if final.truncated or not final.terminated:result={'skipped':'censored reachability'}
       else:
        from .environment import assess
        result=learner.reach(b,torch.tensor([replay.policy_slot(r['policy'])],device='cuda'),torch.tensor([float(assess(final.state,ep['goal'],limits)['success'])],device='cuda'),torch.tensor([True],device='cuda'))
      else:
       result={}
       if update%config['critic_every']==0:result['critics']=learner.critic(b,nb,aa,g,mask,mask,torch.tensor([r['reward']],device='cuda'),torch.tensor([r['duration']],device='cuda'),torch.tensor([r['terminated']],device='cuda'),high,worlds[1].action_mask(for_bootstrap=True)[None].cuda() if high and not worlds[1].terminated else torch.ones(1,9,device='cuda',dtype=torch.bool))
       if update%config['actor_every']==0:result['actor']=learner.high_actor(b,worlds[0].action_mask()[None].cuda()) if high else learner.low_actor(b,mask,lambda:bc_regularizer(system,prior,pop,sampler,geometry_sampler,config))
       system.polyak(config['target_tau'])
   counters['updates']=update+1;counters['replay_rng']=replay_rng.getstate();at_boundary=True
   result.update(update=update+1,seconds=time.monotonic()-t)
   if (update+1)%config.get('log_every',10)==0 or update+1==config['updates']:
    line=json.dumps(result)+'\n';manager.runway(replay,len(line.encode()));history.write(line);history.flush()
   validation=None
   if config.get('validation_panel') and ((update+1)%config['validation_every']==0 or update+1 in config.get('validation_milestones',[])):
    panel=json.load(open(config['validation_panel']));scores=validate_actor(system,prior,panel,'cuda',config)
    groups={}
    for r in scores:
     if r['split']=='VALIDATION':groups.setdefault(r['identity'],[]).append(r['aa'])
    if groups:
     value=sum(sum(v)/len(v) for v in groups.values())/len(groups)
     validation=dict(metric='identity_weighted_validation_AA_NLL',value=value,mode='min',panel_sha256=effective_config['validation_panel_sha256'],identities=len(groups),decisions=sum(map(len,groups.values())))
   if validation or (update+1)%config['checkpoint_every']==0 or update+1==config['updates'] or time.monotonic()-last_checkpoint_time>=config.get('checkpoint_wall_seconds',900):checkpoint(validation)
 except (OSError,sqlite3.Error) as error:
  if not isinstance(error,StoragePause) and getattr(error,'errno',None) not in [errno.ENOSPC,errno.EDQUOT] and getattr(error,'sqlite_errorcode',None)!=sqlite3.SQLITE_FULL:raise
  warnings.warn('Training paused for storage; inspect PAUSED_STORAGE.json before resuming',RuntimeWarning)
  # Normal forecast pressure is detected BEFORE an update; save the coherent
  # current state with no further growth allowance, then return PAUSED.
  emergency=None;save_error=None
  if at_boundary:
   try:emergency=checkpoint(near_growth=0)
   except (StoragePause,OSError) as failure:save_error=str(failure)
  latest=manager.current()['latest']
  atomic_json(run/'PAUSED_STORAGE.json',dict(status='PAUSED_STORAGE',reason=str(error),current_completed_updates=counters['updates'],current_update_incomplete=not at_boundary,exact_current_state_saved=bool(emergency),resume_checkpoint=latest['checkpoint'] if latest else None,resume_update=latest['update'] if latest else None,emergency_save_error=save_error))
 finally:history.close();replay.close();prior.verify(learner.optimizers.values())
 return str(run)
