"""V9.7 authorized orchestration; qualified sampler/loss/Runtime stay unchanged."""
from pathlib import Path
import collections,copy,datetime,fcntl,gzip,hashlib,json,math,os,runpy,shutil,signal,subprocess,sys,time,traceback
import numpy as np
import torch
BASE=Path('/mnt/d/Simone/proteinGen_data/e017_v97_function_preserving_gate_20261031_v1')
PAIR=BASE/'paired_production_20261031_v1'
INIT=BASE/'initialization_admission_20261031_v1'
WT=BASE.parent/'e017_v97_engineering_worktree_20261031_v1'
OLD=BASE.parent/'e017_v96_paired_burial_control_20261029_v1'
ENG=BASE.parent/'e017_v96_burial_conditioning_20261029_v1'
R3=BASE.parent/'e017_v94r3_substantial_geometry_rl_20261026_v1'
ARMS={a:PAIR/f'arm_{a}' for a in ['A','B']}
sys.path.insert(0,str(WT/'e017_v97'))
import core97 as Q
from safety97 import Monitor,SafetyPause,LOOKS,N,ALPHA
from gate97 import GATE_SCALE
read,sha,now=Q.read,Q.sha,Q.now
def atomic(path,obj,immutable=False):Q.put(path,obj,immutable)
def event(kind,**kw):
 row=dict(UTC=now(),kind=kind,**kw)
 with (PAIR/'runtime/EVENTS.jsonl').open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(row),flush=True)
def no_gpu_worker():
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True)
 assert not p.stdout.strip(),('CONFLICTING_GPU_WORKLOAD',p.stdout)
def execution():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def verify():
 x=read(PAIR/'FROZEN_SOURCE_AND_INPUTS.json')
 for p,h in x['files'].items():assert sha(p)==h,('PRODUCTION_SOURCE_INPUT_CHANGED',p)
 Q.verify()
 return x
def resource_guard():
 reserve=read(PAIR/'PRODUCTION_PLAN.json')['minimum_free_launch_bytes']
 assert shutil.disk_usage(PAIR).free>reserve,'STORAGE_STOP'
 assert not (PAIR/'control/STOP').exists(),'COOPERATIVE_STOP'
 if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<6*2**30,'VRAM_STOP'
def factory(arm):
 root=ARMS[arm];P,F=Q.factory(arm,root/'production');cfg=read(root/'TRAINING_CONFIGURATION_AND_SEEDS.json')
 def binding():return dict(Q.binding(),arm=arm,production_frozen_inputs_SHA256=sha(PAIR/'FROZEN_SOURCE_AND_INPUTS.json'),production_configuration_SHA256=sha(root/'TRAINING_CONFIGURATION_AND_SEEDS.json'),initializer_SHA256=read(INIT/f'arm_{arm}/RESULT.json')['initializer']['checkpoint_SHA256'])
 def guard():
  resource_guard();assert not (root/'control/STOP').exists(),'ARM_COOPERATIVE_STOP'
 g=P['Runtime'].__init__.__globals__;g.update(ROOT=root,SPEC=cfg,bindings=binding,guard=guard)
 P.update(ROOT=root,SPEC=cfg,bindings=binding)
 return P,F
_LEGACY=None
def legacy():
 global _LEGACY
 if _LEGACY is None:
  _LEGACY=runpy.run_path(str(OLD/'source/paired.py'))
  _LEGACY['baseline'].__globals__.update(PAIR=PAIR,ARMS=ARMS,event=event,atomic=atomic)
 return _LEGACY
def goal_score(*args):return legacy()['goal_score'](*args)
def development(P,rt,arm):return legacy()['baseline'](P,rt,arm)
def verify_publication(root,relative=False):
 x=read(root/'PROVENANCE_SHA256.json')
 for p,v in x['files'].items():assert sha(root/p if relative else p)==v['SHA256'],('PUBLICATION_HASH',p)
 return len(x['files'])
def prepare():
 assert not PAIR.exists(),'EXISTING_PRODUCTION_ROOT_MUST_USE_RECOVERY'
 Q.verify();no_gpu_worker();initial=read(INIT/'PAIRED_PRODUCTION_READINESS.json');assert initial['V97_INITIALIZATION_ADMISSION']=='PASS' and initial['READY_FOR_PAIRED_5000_UPDATE_TRAINING']
 counts=dict(engineering=verify_publication(BASE),initialization=verify_publication(INIT,True))
 assert read(INIT/'runtime/PUBLICATION_VERIFICATION.json')['PASS']
 for a in ARMS:
  x=read(INIT/f'arm_{a}/RESULT.json');assert x['INITIALIZATION_SAFETY']=='PASS' and x['neural_weights_unchanged'] and x['production_updates']==0
  assert read(BASE/f'RECOVERY_{a}.json')['PASS']
 # Retain one immutable checkpoint per new safety look, in addition to the
 # qualified rolling/named staging envelope; never bind a look to a reused slot.
 extra_safety_images=12*read(BASE/'RESOURCE_AND_RUNTIME_ESTIMATE.json')['estimate']['checkpoint_bytes']
 reserve=read(INIT/'EXECUTION_BINDING.json')['minimum_D_reserve_bytes']+extra_safety_images;assert shutil.disk_usage(BASE).free>reserve
 PAIR.mkdir();(PAIR/'source').mkdir();(PAIR/'runtime/tmp').mkdir(parents=True);(PAIR/'runtime/cache').mkdir();(PAIR/'control').mkdir()
 local=Path('/mnt/d/Simone/proteinGen/e017_v97_production')
 for name in ['production97.py','tests97.py','paired.py']:shutil.copyfile(local/name,PAIR/'source'/name)
 # Published algorithms reused with infrastructure/label adaptations only.
 for name in ['evaluate.py','analyze.py']:
  code=(OLD/'source'/name).read_text()
  if name=='evaluate.py':
   start=code.index('def load_model(');end=code.index('def main():',start)
   code=code[:start]+'from production97 import load_model\n'+code[end:]
   code=code.replace("assert shutil.disk_usage(PAIR).free>2*2**30+256*2**20,'STORAGE_RESERVE'","resource_guard()")
  else:
   code=code.replace('Final controlled V9.6 scientific recommendation','Final controlled V9.7 scientific recommendation')
   code=code.replace('qualification/ADMISSION.json','ACTUAL_INITIAL_RECOVERY_ROUNDTRIP.json and the linked V9.7 qualification')
  (PAIR/'source'/name).write_text(code)
 launcher=(BASE/'run.sh').read_text().replace(str(BASE),str(PAIR));(PAIR/'run.sh').write_text(launcher);(PAIR/'run.sh').chmod(0o755)
 for name in ['VALIDATION_PANEL_MANIFEST.json','SUBSTANTIAL_PAIRED_TRAINING_PROTOCOL.md','VALIDATION_COMPARISON_PROTOCOL.md','SAFETY_MONITOR_STATISTICAL_DESIGN.md']:shutil.copyfile(BASE/name,PAIR/name)
 train=read(BASE/'TRAINING_SCHEDULE.json')['updates'];fit=read(BASE/'TRAINING_POPULATION.json');pop=read(BASE/'MONITOR_POPULATION.json');mon=read(BASE/'MONITOR_SCHEDULE.json')['looks'];val=read(PAIR/'VALIDATION_PANEL_MANIFEST.json')
 assert len(train)==5000 and len(fit['tasks'])==612 and len(pop['tasks'])==256 and len(val['tasks'])==1024 and len(val['diagnostics'])==80
 assert len({t['goal']['protected_group'] for t in val['tasks']})==512
 assert not {t['source_reference_identity'] for t in fit['tasks']}&{t['source_reference_identity'] for t in pop['tasks']}
 fitting_seeds={s for v in train for s in v['seeds']};monitor_seeds={v['candidate_seed'] for row in mon.values() for v in row};assert len(fitting_seeds)==80000 and not fitting_seeds&monitor_seeds
 for arm,root in ARMS.items():
  root.mkdir();(root/'control').mkdir();(root/'runtime').mkdir();(root/'production').mkdir()
  cfg=read(BASE/f'FUTURE_ARM_{arm}_CONFIGURATION.json');cfg.update(experiment='E017_V97_MATCHED_SUBSTANTIAL_PRODUCTION',production_authorized=True,production_authorization='Explicit current user authorization; initialization PASS both; qualified immutable implementation')
  assert cfg['updates']==5000 and cfg['group_size']==16 and cfg['generation_attempts']==80000 and cfg['gate_scale']==GATE_SCALE
  atomic(root/'TRAINING_CONFIGURATION_AND_SEEDS.json',cfg,True)
  atomic(root/'PRODUCTION_AUTHORIZATION.json',dict(explicit_review_authorized=True,UTC=now(),configuration_SHA256=sha(root/'TRAINING_CONFIGURATION_AND_SEEDS.json'),initialization_result_SHA256=sha(INIT/f'arm_{arm}/RESULT.json'),budget_updates=5000,budget_attempts=80000),True)
 a=read(INIT/'arm_A/RESULT.json');b=read(INIT/'arm_B/RESULT.json');estimate=read(BASE/'RESOURCE_AND_RUNTIME_ESTIMATE.json')['estimate']
 ah=a['seconds_this_process']/3600;bh=b['seconds_this_process']*4096/4080/3600
 runtime=dict(fitting_hours_A=estimate['A_training_seconds']/3600,fitting_hours_B=estimate['B_training_seconds']/3600,remaining_safety_looks_per_arm=6,remaining_safety_attempts=49152,safety_hours_A=6*ah,safety_hours_B=6*bh,final_validation_attempts=89088,final_validation_hours_range=[60690/3600,60690/3600*1.25],development_attempts_per_look=352,development_looks_per_arm=7,development_hours_estimate=1.5,assumptions='fit: qualified engineering throughput; safety: completed look0 measured wall time, fresh suffix corrected; validation: historical full panel plus up to25percent gated overhead; changing actor/rejection rates can alter cost')
 runtime['total_hours_range']=[runtime['fitting_hours_A']+runtime['fitting_hours_B']+runtime['safety_hours_A']+runtime['safety_hours_B']+h+runtime['development_hours_estimate'] for h in runtime['final_validation_hours_range']]
 atomic(PAIR/'PRODUCTION_PLAN.json',dict(UTC=now(),updates_per_arm=5000,attempts_per_arm=80000,arms=['A','B'],sequential=True,minimum_free_launch_bytes=reserve,extra_immutable_safety_checkpoint_budget_bytes=extra_safety_images,operational_reserve_bytes=2*2**30,safety_looks=LOOKS,completed_initialization_looks_reused_as_admission=True,monitoring_poll_seconds=60,monitoring_report_seconds=1800,arm_B_launch_contract='A completed5000 with terminal safety PASS and GPU release; early A stop pauses paired execution for review',checkpoint_selection='final5000 primary; best descriptive only',runtime=runtime,TEST=False),True)
 files={str(p):sha(p) for p in (PAIR/'source').glob('*.py')}
 for p in [PAIR/'run.sh',PAIR/'PRODUCTION_PLAN.json',PAIR/'VALIDATION_PANEL_MANIFEST.json',BASE/'TRAINING_SCHEDULE.json',BASE/'MONITOR_SCHEDULE.json',BASE/'TRAINING_POPULATION.json',BASE/'MONITOR_POPULATION.json',BASE/'SOURCE_BINDING.json',BASE/'PROVENANCE_SHA256.json',INIT/'PROVENANCE_SHA256.json',INIT/'PAIRED_PRODUCTION_READINESS.json',BASE/'SUBSTANTIAL_PAIRED_TRAINING_PROTOCOL.md',BASE/'VALIDATION_COMPARISON_PROTOCOL.md',BASE/'qualified_train/manifests/FUTURE_FIXED_TRAIN_POOL.json',BASE/'qualified_train/manifests/TRAIN_TASKS.json']+[p for a in ARMS.values() for p in [a/'TRAINING_CONFIGURATION_AND_SEEDS.json',a/'PRODUCTION_AUTHORIZATION.json']]:files[str(p)]=sha(p)
 # Bind the reused runtime/likelihood/evaluation source as well as new adapters.
 hist=read(OLD/'FROZEN_SOURCE_AND_INPUTS.json');files.update(hist['files'])
 for a in ARMS:
  for name in ['RESULT.json','INITIALIZER_CERTIFICATION.json']:files[str(INIT/f'arm_{a}'/name)]=sha(INIT/f'arm_{a}'/name)
 atomic(PAIR/'FROZEN_SOURCE_AND_INPUTS.json',dict(files=files,historical_models=read(BASE/'SOURCE_BINDING.json')['models'],qualified_source_binding_SHA256=sha(BASE/'SOURCE_BINDING.json')),True)
 atomic(PAIR/'PREPARATION.json',dict(PASS=True,UTC=now(),verified_publication_files=counts,D_free_bytes=shutil.disk_usage(PAIR).free,production_updates=0,fit_groups=306,monitor_groups=128,validation_groups=512,primary_goals=1024,diagnostic_groups=80),True)
 print('PRODUCTION_PREPARED',json.dumps(runtime),flush=True)
def check_initializer(P,rt,arm):
 state=torch.load(R3/'production/checkpoints/milestone_05000.pt',map_location='cpu',weights_only=False)['actor'];live=rt.e.actor.state_dict();keys=set(live)-({'burial_gate'} if arm=='B' else set())
 assert keys==set(state) and all(torch.equal(live[k].cpu(),state[k]) for k in state)
 assert rt.update==0 and not rt.counters and all(float(s['step'])==0 and not s['exp_avg'].count_nonzero() and not s['exp_avg_sq'].count_nonzero() for s in rt.opt.state.values())
 if arm=='B':assert float(rt.e.actor.burial_gate)==0.
 assert sum(p.numel() for p in rt.e.actor.parameters())==17438114+(arm=='B')
 rt.e.prior.verify();assert P['c']['state_digest'](rt.e.parent.state_dict())==rt.e.fp
def draws():return dict(cpu=torch.rand(8).tolist(),cuda=torch.rand(8,device='cuda').cpu().tolist(),numpy=np.random.random(8).tolist(),python=__import__('random').random())
def admit(arm):
 verify();resource_guard();execution();no_gpu_worker();root=ARMS[arm];P,F=factory(arm);rt=P['Runtime'](root/'production');check_initializer(P,rt,arm)
 digest=P['state_digest'](rt.state());d=draws();rt.restore();assert P['state_digest'](rt.state())==digest and draws()==d;rt.restore()
 report=development(P,rt,arm);initial=read(INIT/f'arm_{arm}/RESULT.json')
 # Only original KL/diversity sentinel guards; validity admission is full4096.
 assert report['filtered']['geometry_KL_per_active']==initial['sentinel']['geometry_KL_per_active']
 assert report['filtered']['structural_pair_RMSD']==initial['sentinel']['structural_pair_RMSD']
 assert initial['INITIALIZATION_SAFETY']=='PASS';rt.baseline=report;rt.best=dict(update=0,**report['filtered']);rt.publish('DEVELOPMENT_BEST')
 rt.restore();assert rt.update==0;check_initializer(P,rt,arm)
 atomic(root/'ACTUAL_INITIAL_RECOVERY_ROUNDTRIP.json',dict(PASS=True,UTC=now(),arm=arm,initializer_exact=True,gate_zero=(arm=='B'),fresh_optimizer_scheduler=True,full_actual_state_RNG_restored=True,initialization_safety_SHA256=sha(INIT/f'arm_{arm}/RESULT.json'),checkpoint=read(rt.root/'RECOVERY_INDEX.json')['latest'],production_optimizer_updates=0),True)
 event('ACTUAL_PRODUCTION_ADMISSION_PASS',arm=arm,production_updates=0)
class GateSensor:
 def __init__(self,actor):
  self.actor=actor;self.handles=[];self.reset()
  if hasattr(actor,'burial_gate'):
   self.handles=[actor.geometry.register_forward_hook(self.forward),actor.burial_gate.register_hook(self.gradient)]
 def reset(self):self.pending=None;self.sum2=0.;self.weighted_sum2=0.;self.num=0;self.grad_sum=0.;self.grad_calls=0
 def gradient(self,g):self.grad_sum+=float(g.detach());self.grad_calls+=1
 def forward(self,module,args,out):
  # Passive measurement of the two existing forwards. No new network call/RNG.
  x=out.detach().float().tanh()[...,:3]
  if self.pending is None:self.pending=x;return
  delta=(x-self.pending)*x.new_tensor([.15,1.,1.]);sum2=float(delta.double().square().sum());self.sum2+=sum2;self.weighted_sum2+=math.tanh(float(self.actor.burial_gate.detach())/GATE_SCALE)**2*sum2;self.num+=delta.numel();self.pending=None
 def summary(self):
  if not hasattr(self.actor,'burial_gate'):return None
  assert self.pending is None;a=float(self.actor.burial_gate.detach());g=math.tanh(a/GATE_SCALE);rms=math.sqrt(self.sum2/max(1,self.num))
  return dict(a=a,gamma=g,gate_gradient_preclip_sum=self.grad_sum,gradient_contributions=self.grad_calls,gate_gradient_postclip=float(self.actor.burial_gate.grad) if self.actor.burial_gate.grad is not None else None,saturated=abs(g)>=.99,burial_mean_delta_RMS_all_geometry_heads=rms,gate_scaled_mean_delta_RMS=math.sqrt(self.weighted_sum2/max(1,self.num)),contribution_scope='all20AA mean heads across existing forwards using gate at each forward; not a goal-steering metric')
def metric(rt,P,arm,u,sensor=None):
 packet=read(rt.root/f'rollouts/u{u:06d}.json');rows=packet['records'];g=packet['task']['goal'];m=[r['metrics'] for r in rows];commit=read(rt.root/f'commits/u{u:06d}.json');v=commit['logical']
 value=dict(update=u,identity=packet['task']['source_reference_identity'],difficulty=g['difficulty'],reward=v['stats']['reward_mean'],exposure=sum(x.get('exposed_mean',-1)>=g['exposure_min'] for x in m),burial=sum(x.get('buried_mean',2)<=g['burial_max'] for x in m),success=sum(x['success'] for x in m),reward_components_descriptive={k:float(np.mean([x[k] for x in m if k in x])) for k in ['motif_success','target_error'] if any(k in x for x in m)},packing=sum(x['packing_valid'] for x in m),clash_failures=sum(x['complete'] and 'nonlocal_clashes' in x['geometry']['failures'] for x in m),grad_norm=v['gradient_norm'],clipped=v['gradient_norm']>5,entropy=v['diagnostics']['latent_Gaussian_entropy_per_active_dimension'],advantage=v['diagnostics'],step_KL=v['step_KL'],stats=v['stats'],seconds=commit['seconds'],gate=sensor.summary() if sensor else None)
 path=rt.root/f'metrics/u{u:06d}.json'
 if path.exists():
  old=read(path);assert {k:val for k,val in old.items() if k!='gate'}=={k:val for k,val in value.items() if k!='gate'}
 else:atomic(path,value,True)
 if value['gate'] and value['gate']['saturated'] and u%50==0:event('GATE_SATURATION_WARNING',arm=arm,update=u,gate=value['gate'])
def safety(rt,P,arm,u):
 path=rt.root/f'safety/look_{u:05d}';result=path/'RESULT.json'
 fp=P['c']['state_digest'](rt.e.actor.state_dict())
 if result.exists():
  x=read(result);assert x['actor_fingerprint']==fp and x['look']==u
  cp=x['checkpoint'];assert sha(rt.root/cp['relative'])==cp['SHA256']
  m=Monitor(path/'monitor',arm,u,cp['SHA256'],actor_fingerprint=fp);assert m.frontier==4096 and m.backbone==x['backbone'] and m.hard==x['hard']
  sent=x['sentinel'];b=rt.baseline['filtered'];kw=dict(initial_hard=b['hard_valid'],KL=sent['geometry_KL_per_active'],diversity=sent['structural_pair_RMSD'],initial_diversity=b['structural_pair_RMSD'])
  assert m.decision(**kw)==x['decision']
  atomic(path/'PUBLICATION_VERIFICATION.json',dict(PASS=True,RESULT_SHA256=sha(result),frontier=4096,actor_unchanged=True,checkpoint_SHA256=cp['SHA256'],source_binding_SHA256=sha(BASE/'SOURCE_BINDING.json')),True)
  if x['decision']['status']!='PASS':raise SafetyPause(x['decision']['status'])
  return
 resource_guard();event('SCHEDULED_SAFETY_LOOK',arm=arm,update=u,attempts=4096)
 if (path/'CHECKPOINT.json').exists():
  cp=read(path/'CHECKPOINT.json');assert sha(rt.root/cp['relative'])==cp['SHA256'] and read(rt.root/'RECOVERY_INDEX.json')['latest']['SHA256']==cp['SHA256'],'INTERRUPTED_SAFETY_CHECKPOINT_MISMATCH'
 else:
  target=path/'CHECKPOINT.pt';target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():rt.publish('SAFETY_LOOK_BOUNDARY')
  cp=read(rt.root/'RECOVERY_INDEX.json')['latest']
  if not target.exists():
   try:os.link(rt.root/cp['relative'],target)
   except OSError:
    temp=path/('CHECKPOINT.copy_part_'+str(time.time_ns()))
    with (rt.root/cp['relative']).open('rb') as src,temp.open('xb') as dst:shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
    os.replace(temp,target)
  assert sha(target)==cp['SHA256'];cp=dict(cp,rolling_relative=cp['relative'],relative=str(target.relative_to(rt.root)));atomic(path/'CHECKPOINT.json',cp,True)
 report=development(P,rt,arm);sent=report['filtered'];initial=read(INIT/f'arm_{arm}/RESULT.json');b=rt.baseline['filtered']
 m=Monitor(path/'monitor',arm,u,cp['SHA256'],actor_fingerprint=fp);started=time.monotonic();entry=m.frontier
 while m.frontier<N:
  resource_guard();assert not (ARMS[arm]/'control/STOP').exists(),'ARM_COOPERATIVE_STOP'
  m.collect(P,rt.e,max_new=min(16,N-m.frontier))
  atomic(path/'PROGRESS.json',dict(UTC=now(),arm=arm,look=u,committed=m.frontier,total=N,backbone=m.backbone,hard=m.hard,PID=os.getpid(),seconds_this_process=time.monotonic()-started))
  atomic(ARMS[arm]/'runtime/PHASE.json',dict(UTC=now(),state='SAFETY_EVALUATION',update=u,committed=m.frontier,total=N,PID=os.getpid()))
 checked=Monitor(path/'monitor',arm,u,cp['SHA256'],actor_fingerprint=fp);assert checked.frontier==4096 and checked.backbone==m.backbone and checked.hard==m.hard
 # Historical hard/diversity preservation uses the original fixed-development baseline.
 kw=dict(initial_hard=b['hard_valid'],KL=sent['geometry_KL_per_active'],diversity=sent['structural_pair_RMSD'],initial_diversity=b['structural_pair_RMSD'])
 counts=collections.Counter()
 for i in range(N):
  row=read(m.root/f'rows/{i:05d}.json');counts['completed']+=row['complete'];counts['aborted']+=row['aborted']
 assert counts['completed']+counts['aborted']==4096
 d=m.decision(**kw);x=dict(UTC=now(),arm=arm,look=u,actor_fingerprint=fp,checkpoint=cp,decision=d,attempts=4096,backbone=m.backbone,hard=m.hard,completed=counts['completed'],aborted=counts['aborted'],sentinel=sent,recovered_prefix=entry,seconds_this_process=time.monotonic()-started)
 # Enforcement publishes a bound decision first; complete non-PASS is never extended.
 try:m.enforce(**kw)
 except SafetyPause:assert d['status'] in ['BREACH','INCONCLUSIVE']
 atomic(result,x,True);atomic(path/'PUBLICATION_VERIFICATION.json',dict(PASS=True,RESULT_SHA256=sha(result),frontier=4096,actor_unchanged=P['c']['state_digest'](rt.e.actor.state_dict())==fp,checkpoint_SHA256=cp['SHA256'],source_binding_SHA256=sha(BASE/'SOURCE_BINDING.json')),True)
 event('SAFETY_DECISION',arm=arm,update=u,status=d['status'],backbone=m.backbone,hard=m.hard,CI=d['backbone_CI'])
 if d['status']!='PASS':raise SafetyPause(d['status'])
 if (sent['K1'],sent['K8'],sent['hard_valid'])>(rt.best['K1'],rt.best['K8'],rt.best['hard_valid']):rt.best=dict(update=u,**sent);rt.publish('DEVELOPMENT_BEST')
def required_look_before_update(u):
 assert 1<=u<=5000
 return max(k for k in LOOKS if k<u)
def require_safety_before_update(arm,u):
 look=required_look_before_update(u)
 if look==0:
  p=INIT/f'arm_{arm}/RESULT.json';assert sha(p)==read(ARMS[arm]/'PRODUCTION_AUTHORIZATION.json')['initialization_result_SHA256'];assert read(p)['INITIALIZATION_SAFETY']=='PASS'
 else:
  path=ARMS[arm]/f'production/safety/look_{look:05d}';p=path/'RESULT.json';v=read(path/'PUBLICATION_VERIFICATION.json')
  assert v['PASS'] and sha(p)==v['RESULT_SHA256'] and read(p)['decision']['status']=='PASS','PRIOR_SAFETY_NOT_PASS'
def training(arm):
 verify();resource_guard();execution();no_gpu_worker();root=ARMS[arm]
 assert read(PAIR/'PRODUCTION_ADMISSION.json')['PASS'] and read(root/'ACTUAL_INITIAL_RECOVERY_ROUNDTRIP.json')['PASS']
 assert not (root/'production/STOP_RECORD.json').exists(),'REGISTERED_STOP_MUST_NOT_BE_RESTARTED'
 lock=(root/'control/WORKER.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 P,F=factory(arm);rt=P['Runtime'](root/'production');assert rt.baseline is not None
 if rt.update==0:check_initializer(P,rt,arm)
 (rt.root/'metrics').mkdir(exist_ok=True);plan=read(BASE/'TRAINING_SCHEDULE.json')['updates'];sensor=GateSensor(rt.e.actor)
 def callback(u,record=True):
  if record:metric(rt,P,arm,u,sensor)
  if u in LOOKS[1:]:safety(rt,P,arm,u)
  if u%50==0:rt.publish('ROLLING')
 def step(u):
  require_safety_before_update(arm,u)
  task=rt.task(u);s=plan[u-1];assert task['goal']['goal_id']==s['goal_id'] and task['source_reference_identity']==s['identity']
  assert [P['S']['ns']('SUBSTANTIAL',P['SPEC']['seed'],u,i) for i in range(16)]==s['seeds']
  sensor.reset();return rt.step(u)
 start=time.monotonic();status='COMPLETED';reason=None;code=0
 event('TRAINING_STARTED',arm=arm,PID=os.getpid(),committed_update=rt.update)
 try:
  commits=sorted((rt.root/'commits').glob('u*.json'));end=int(commits[-1].stem[1:]) if commits else 0;assert len(commits)==end
  # Reconcile completed safety callback at checkpoint frontier before any optimizer step.
  if rt.update in LOOKS[1:]:callback(rt.update,False)
  for u in range(rt.update+1,end+1):step(u);callback(u)
  for u in range(rt.update+1,5001):
   resource_guard();atomic(root/'runtime/PHASE.json',dict(UTC=now(),state='FITTING',update=rt.update,next_update=u,PID=os.getpid()))
   step(u);callback(u)
   if u in [1000,2000,3000,4000,5000]:event('UPDATE_MILESTONE',arm=arm,update=u)
  rt.publish('COMPLETED_DECLARED_BUDGET');atomic(rt.root/'TRAINING_COMPLETED.json',dict(update=rt.update,counters=dict(rt.counters)),True)
 except SafetyPause as e:
  status='SAFETY_STOPPED' if str(e)=='BREACH' else 'SAFETY_INCONCLUSIVE';reason=str(e);code=20
  rt.publish('REGISTERED_SAFETY_STOP');atomic(rt.root/'STOP_RECORD.json',dict(UTC=now(),arm=arm,update=rt.update,status=status,reason=reason,thresholds_unchanged=True),True)
 except BaseException as e:
  reason=repr(e)
  if any(t in reason for t in ['STEP_KL_STABILITY_STOP','GEOMETRY_KL_STABILITY_STOP','nonfinite gradient']):
   status='SAFETY_STOPPED';code=20
   # Reject the uncommitted unsafe step and restore/replay only durable commitments.
   rt.restore();cs=sorted((rt.root/'commits').glob('u*.json'));end=int(cs[-1].stem[1:]) if cs else 0
   for u in range(rt.update+1,end+1):step(u)
   rt.publish('REGISTERED_STEP_SAFETY_STOP');atomic(rt.root/'STOP_RECORD.json',dict(UTC=now(),arm=arm,update=rt.update,status=status,reason=reason),True)
  else:
   atomic(root/'runtime/WORKER_FAILURE.json',dict(UTC=now(),error=reason,traceback=traceback.format_exc(),automatic_retry=False,committed_update=rt.update));raise
 cp=read(rt.root/'RECOVERY_INDEX.json')['latest'];assert sha(rt.root/cp['relative'])==cp['SHA256'];rollouts=list((rt.root/'rollouts').glob('u*.json'))
 outcome=dict(UTC=now(),arm=arm,status=status,exit_code=code,PID=os.getpid(),updates=rt.update,committed_attempts=rt.counters['attempts'],completed_trajectories=rt.counters['complete'],aborted_trajectories=rt.counters['aborts'],all_generated_training_attempts=16*len(rollouts),seconds=time.monotonic()-start,checkpoint=cp,training_seed=P['SPEC']['seed'],J1_frozen=True,reason=reason)
 atomic(root/'TRAINING_OUTCOME.json',outcome,True);event('TRAINING_EXIT',**outcome);raise SystemExit(code)
def load_model(P,F,e,model):
 arm=model[0]
 if model.endswith('0'):
  path=R3/'production/checkpoints/milestone_05000.pt';state=torch.load(path,map_location='cpu',weights_only=False)['actor']
  if arm=='B':state=dict(state,burial_gate=torch.zeros_like(e.actor.burial_gate.detach().cpu()))
  e.actor.load_state_dict(state,strict=True);meta=dict(path=str(path),SHA256=sha(path),update=0,certified_initializer_update=5000)
 else:
  v=read(ARMS[arm]/'TRAINING_OUTCOME.json')['checkpoint'];path=ARMS[arm]/'production'/v['relative'];assert sha(path)==v['SHA256'];state=torch.load(path,map_location='cpu',weights_only=False)
  assert P['state_digest'](state)==v['state_digest'] and state['binding']==P['bindings']();e.actor.load_state_dict(state['actor'],strict=True);meta=dict(path=str(path),SHA256=v['SHA256'],update=state['update'])
 e.actor.eval().requires_grad_(False);meta['actor_fingerprint']=P['c']['state_digest'](e.actor.state_dict());return meta
def monitor():
 lock=(PAIR/'control/MONITOR.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);last=0.;last_events=0
 while not (PAIR/'control/MONITOR_STOP').exists():
  rows=[]
  for arm,root in ARMS.items():
   live=root/'production/LIVE_STATUS.json';x=read(live) if live.exists() else dict(state='QUEUED',update=0);m=root/f"production/metrics/u{x.get('update',0):06d}.json"
   if m.exists():x['latest_metrics']=read(m)
   phase=root/'runtime/PHASE.json'
   if phase.exists():x['phase']=read(phase)
   dev=sorted((root/'production/diagnostics').glob('DEVELOPMENT_*.json'))
   if dev:x['fixed_development']=read(dev[-1])
   rows.append(dict(arm=arm,**x))
  gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],capture_output=True,text=True)
  ctrl=read(PAIR/'runtime/CONTROLLER_STATUS.json') if (PAIR/'runtime/CONTROLLER_STATUS.json').exists() else {}
  x=dict(UTC=now(),PID=os.getpid(),controller=ctrl,D_free_bytes=shutil.disk_usage(PAIR).free,GPU=gpu.stdout.strip(),GPU_query_exit=gpu.returncode,arms=rows)
  atomic(PAIR/'runtime/MONITOR_LATEST.json',x)
  with (PAIR/'runtime/MONITOR.jsonl').open('a') as f:f.write(json.dumps(x,separators=(',',':'))+'\n');f.flush()
  events=PAIR/'runtime/EVENTS.jsonl'
  if events.exists():
   with events.open() as f:
    f.seek(last_events)
    for line in f:print('EVENT',line.strip(),flush=True)
    last_events=f.tell()
  if time.monotonic()-last>=1800:
   last=time.monotonic();print('THIRTY_MINUTE_PROGRESS',json.dumps(x),flush=True)
  if (PAIR/'runtime/CONTROLLER_EXIT.json').exists():break
  time.sleep(60)
 atomic(PAIR/'runtime/MONITOR_EXIT.json',dict(UTC=now(),PID=os.getpid(),reason='CONTROLLER_TERMINAL' if (PAIR/'runtime/CONTROLLER_EXIT.json').exists() else 'MONITOR_ONLY_STOP'))
def run_child(mode,arm=None):
 resource_guard();no_gpu_worker();label=mode+(('_'+arm) if arm else '');log=(PAIR/f'runtime/{label}.log').open('a')
 cmd=[str(PAIR/'run.sh'),str(PAIR/'source/production97.py'),mode]+([arm] if arm else [])
 if mode in ['evaluate','analyze']:cmd=[str(PAIR/'run.sh'),str(PAIR/f'source/{mode}.py')]
 p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 atomic(PAIR/'runtime/CONTROLLER_STATUS.json',dict(UTC=now(),mode=mode,arm=arm,worker_PID=p.pid,controller_PID=os.getpid()))
 atomic(PAIR/f'runtime/{label}_LAUNCH.json',dict(UTC=now(),PID=p.pid,command=cmd))
 event('WORKER_LAUNCH',mode=mode,arm=arm,PID=p.pid)
 while p.poll() is None:
  time.sleep(60)
  if (PAIR/'runtime/MONITOR_FAILURE.json').exists():atomic(PAIR/'control/STOP',dict(reason='MONITOR_FAILURE'))
 log.close()
 for _ in range(30):
  try:no_gpu_worker();break
  except AssertionError:time.sleep(1)
 no_gpu_worker();event('GPU_RELEASE_VERIFIED',arm=arm,mode=mode,exit_code=p.returncode);return p.returncode
def controller():
 lock=(PAIR/'control/CONTROLLER.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);verify();resource_guard();no_gpu_worker()
 ml=(PAIR/'runtime/monitor.log').open('a');mon=subprocess.Popen([str(PAIR/'run.sh'),str(PAIR/'source/production97.py'),'monitor'],stdin=subprocess.DEVNULL,stdout=ml,stderr=subprocess.STDOUT)
 atomic(PAIR/'runtime/MONITOR_LAUNCH.json',dict(UTC=now(),PID=mon.pid,poll_seconds=60,report_seconds=1800));event('CONTROLLER_STARTED',PID=os.getpid(),monitor_PID=mon.pid)
 try:
  assert read(PAIR/'ORCHESTRATION_QUALIFICATION.json')['PASS']
  for arm,root in ARMS.items():
   if not (root/'ACTUAL_INITIAL_RECOVERY_ROUNDTRIP.json').exists():assert run_child('admit',arm)==0,('ACTUAL_ADMISSION_FAILED',arm)
  atomic(PAIR/'PRODUCTION_ADMISSION.json',dict(PASS=True,UTC=now(),both_actual_zero_update_recovery_PASS=True,initialization_safety_PASS_both=True,source_binding_SHA256=sha(PAIR/'FROZEN_SOURCE_AND_INPUTS.json'),D_free_bytes=shutil.disk_usage(PAIR).free),True)
  for arm,root in ARMS.items():
   if (root/'TRAINING_OUTCOME.json').exists():code=read(root/'TRAINING_OUTCOME.json')['exit_code']
   else:code=run_child('train',arm)
   if code==20:
    atomic(PAIR/'runtime/CONTROLLER_EXIT.json',dict(UTC=now(),status='REGISTERED_SAFETY_PAUSE',arm=arm,exit_code=20,next_action='Review preserved stop; no automatic other-arm start or retry'),True);return
   assert code==0,('ARM_INFRASTRUCTURE_STOP',arm,code)
   assert read(root/'TRAINING_OUTCOME.json')['updates']==5000
  assert run_child('evaluate')==0,'FINAL_EVALUATION_FAILED'
  assert run_child('analyze')==0,'FINAL_PUBLICATION_FAILED'
  atomic(PAIR/'runtime/CONTROLLER_EXIT.json',dict(UTC=now(),status='COMPLETED',exit_code=0),True);event('CAMPAIGN_COMPLETED')
 except BaseException as e:
  atomic(PAIR/'runtime/CONTROLLER_FAILURE.json',dict(UTC=now(),error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
  atomic(PAIR/'runtime/CONTROLLER_EXIT.json',dict(UTC=now(),status='INFRASTRUCTURE_STOPPED',exit_code=1,error=repr(e)),True);raise
 finally:
  mon.wait(timeout=70);ml.close()
if __name__=='__main__':
 mode=sys.argv[1]
 try:{'prepare':prepare,'admit':lambda:admit(sys.argv[2]),'train':lambda:training(sys.argv[2]),'controller':controller,'monitor':monitor}[mode]()
 except BaseException as e:
  if mode=='monitor':atomic(PAIR/'runtime/MONITOR_FAILURE.json',dict(UTC=now(),error=repr(e),traceback=traceback.format_exc()))
  raise
