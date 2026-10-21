"""Future training orchestration. All execution requires a separate scoped authorization."""
import copy,json,math,random,shutil,time,uuid
from pathlib import Path
import torch
from torch.nn import functional as F
from protein_joint_sac.models.batched_bc import collate
from protein_joint_sac.geometry.ca_builder import next_ca
from .density import sample
from .networks import System,prepare
from .prior import FrozenPrior,SHA,file_sha,tensor_sha
from .optimization import Learner,supervised,finish
from .storage import Replay,digest,payload,atomic_save,restore
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
 root.mkdir(parents=True,exist_ok=True);run=root/('run_'+uuid.uuid4().hex);run.mkdir();cfgsha=seal_config(config)
 # Fail before allocation if two complete checkpoints cannot be staged safely.
 with torch.device('meta'):prototype=System(config['profile'],config.get('shapes'));counts=prototype.counts()
 estimate=counts['total_trainable']*12+sum(counts['target_nontrainable'].values())*4+64*2**20
 if shutil.disk_usage(root).free<10*2**30+2*estimate:raise OSError('insufficient two-generation atomic recovery headroom; do not lower model capacity silently')
 torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);torch.use_deterministic_algorithms(True)
 prior=FrozenPrior('cuda');system=System(config['profile'],config.get('shapes')).cuda();learner=Learner(system,**config['optimizer']);pop=Population();sampler=CoverageSampler(pop,config['seed']);geometry_sampler=CoverageSampler(pop,config['seed']+1);goals=Goals(pop,config['seed']+2,json.load(open(config['native_limits'])));samplers=dict(AA=sampler,geometry=geometry_sampler,goals=goals);replay=Replay(run/'replay.sqlite');counters={'updates':0,'episodes':0};limits=json.load(open(config['native_limits']));replay_rng=random.Random(config['seed']+3)
 data_manifest=pop.membership();effective_config={**config,'data_manifest':data_manifest,'source_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[3],text=True).strip()}
 if config.get('initial_checkpoint') and not resume:
  initial=config['initial_checkpoint']
  if file_sha(initial['path'])!=initial['sha256']:raise ValueError('initialization checkpoint SHA')
  parent=torch.load(initial['path'],map_location='cpu',weights_only=False)
  if parent['j1_sha256']!=SHA or parent['profile']!=system.profile:raise ValueError('initialization reference/profile')
  system.load_state_dict(parent['system'],strict=True)
  for k,o in learner.optimizers.items():
   o.load_state_dict(parent['optimizers'][k])
   for g in o.param_groups:g['lr']=config['optimizer']['lr']
  for k,v in samplers.items():v.load_state_dict(parent['samplers'][k])
 if resume:
  state=restore(resume,system,learner,samplers,effective_config);counters=state['counters'];replay_rng.setstate(counters['replay_rng']);replay.close();import sqlite3
  assert file_sha(state['replay_snapshot']['path'])==state['replay_snapshot']['sha256'];source=sqlite3.connect(state['replay_snapshot']['path']);dest=sqlite3.connect(run/'replay.sqlite');source.backup(dest);source.close();dest.close();replay=Replay(run/'replay.sqlite')
 (run/'CONFIG.json').write_text(json.dumps(effective_config,indent=2));history=open(run/'updates.jsonl','x');previous=None
 try:
  for update in range(counters['updates'],config['updates']):
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
   counters['updates']=update+1;counters['replay_rng']=replay_rng.getstate();result.update(update=update+1,seconds=time.monotonic()-t);history.write(json.dumps(result)+'\n');history.flush()
   if config.get('validation_panel') and (update+1)%config['validation_every']==0:
    panel=json.load(open(config['validation_panel']));scores=validate_actor(system,prior,panel,'cuda',config);(run/f'validation_{update+1:08d}.json').write_text(json.dumps(scores))
   if (update+1)%config['checkpoint_every']==0 or update+1==config['updates']:
    prior.verify(learner.optimizers.values());snapshot=run/f'replay_{update+1:08d}.sqlite';replay.snapshot(snapshot);snap=dict(path=str(snapshot),sha256=file_sha(snapshot));path=run/f'checkpoint_{update+1:08d}.pt';atomic_save(path,payload(system,learner,samplers,effective_config,snap,counters),estimate)
    manifest=dict(checkpoint=str(path),sha256=file_sha(path),replay=snap,config_sha256=digest(effective_config),source_commit=effective_config['source_commit'],update=update+1)
    pointer=run/'LATEST_RECOVERY.json.part'
    with open(pointer,'w') as f:json.dump(manifest,f,indent=2);f.flush();__import__('os').fsync(f.fileno())
    pointer.replace(run/'LATEST_RECOVERY.json')
    # Bounded retention only for checkpoints created by this run, retain latest plus selected outside this loop.
    if previous and not config['retain_all_checkpoints']:
     for p in previous:p.unlink()
    previous=(path,snapshot)
 finally:history.close();replay.close();prior.verify(learner.optimizers.values())
 return str(run)
