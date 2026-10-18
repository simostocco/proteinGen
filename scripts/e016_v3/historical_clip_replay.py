"""One <=2500-update J1 lineage; full source state, grouped clipping, no long rollouts."""
import collections,copy,gc,json,math,os,random,resource,shutil,signal,subprocess,threading,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
import j1_sequence_first as s
from protein_joint_sac.rl.group_clipping import groups,combine,VERSION as GRAD_VERSION
from protein_joint_sac.models.geometry_inference import corrected_action,PROTOCOL
from protein_joint_sac.models.batched_bc import collate,encode_batch
from protein_joint_sac.contracts import AA,BuildState,BuildAction
from protein_joint_sac.geometry.ca_builder import append,mask_for_step
from partial_finalize import metrics
from j1_global_finalize import metric_summary,paired_ci
R=Path('/mnt/d/Simone/proteinGen_data/e016_j1_clip_precision_pilot_20261018_v1')
OLD=Path('/mnt/d/Simone/proteinGen_data/e016_j1_sequence_first_20261018_v1')
SOURCE=OLD/'runs/J1/checkpoint_05000.pt';SOURCE_SHA='462459bf8c688d755505a6f9636f6b5f7b6c810191086e1b00b3cc002868df02'
s.R=R;s.j.R=R;s.j.core.R=R
write=s.write
STOP=threading.Event();LAST=[time.monotonic()]
def read(p):return json.load(open(p))
def guard():
 free=shutil.disk_usage(R).free
 if free<10*2**30+210*2**20:raise RuntimeError('10GiB reserve plus210MiB checkpoint staging unavailable')
 mem={v.split(':')[0]:int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith(('MemAvailable:','MemTotal:','SwapFree:','SwapTotal:'))}
 if mem['MemAvailable']<.1*mem['MemTotal']:raise RuntimeError('host RAM headroom below10percent')
 return {'D_free_bytes':free,'RSS_highwater_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_allocated_peak':torch.cuda.max_memory_allocated(),'GPU_reserved_peak':torch.cuda.max_memory_reserved(),**mem}
def status(phase,**kw):write(R/'HEARTBEAT.json',{'phase':phase,'time':time.time(),**kw})
def sha(p):return s.file_sha(Path(p))
def finite_state(opt):return all(bool(torch.isfinite(v).all()) for q in opt.state.values() for v in q.values() if torch.is_tensor(v))
class Engine:
 def __init__(self,data,c):
  torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
  self.data=data;self.c=c;self.name='J1';self.joint=True;self.model=s.j.core.model('J1');s.set_dropout(self.model,.15)
  state=s.load_checked(SOURCE)
  if sha(SOURCE)!=SOURCE_SHA or state['additional_update']!=5000:raise ValueError('source mismatch')
  self.model.load_state_dict(state['model'],strict=True)
  self.opt=torch.optim.AdamW(self.model.parameters(),lr=c['learning_rate'],weight_decay=.01,fused=True);self.opt.load_state_dict(state['optimizer'])
  if {float(v['step']) for v in self.opt.state.values()}!={35000.}:raise ValueError('source moments/counter')
  for g in self.opt.param_groups:g['lr']=c['learning_rate'];g['initial_lr']=c['learning_rate']
  def factor(k):
   if k<100:return (k+1)/100
   return .2+.8*.5*(1+math.cos(math.pi*min(1.,(k-100)/2400)))
  self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.opt,factor)
  self.sampler=s.Sampler(data);self.sampler.load_state_dict(state['sampler']);self.source_counts=dict(self.sampler.counts);self.source_coverage=self.sampler.report();self.rollrng=random.Random();self.rollrng.setstate(state['rollout_rng'])
  self.update=0;self.nonlocal_targets=0;self.history=[];self.best=0;self.elapsed=0.;self.start=time.monotonic()
  self.lineage={'version':'j1_clip_precision_pilot_v1','source_checkpoint':str(SOURCE),'source_sha256':SOURCE_SHA,'source_total_updates':35000,'config':c,'package_sha256':s.code_hash(),'entrypoint_sha256':sha(__file__),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip(),'data_lineage':state['lineage'],'panel_sha256':sha(R/'configs/EVALUATION_PANELS.json'),'geometry_version':'ca_internal_signed_v1','inference_protocol':PROTOCOL,'gradient_policy':GRAD_VERSION,'normalization_changes':False,'TEST':'NOT_USED'}
  s.restore_rng(state['rng']);del state
 def backward(self):
  self.model.train();self.opt.zero_grad(set_to_none=True);aaex,geoex=self.sampler.draw();aa_value=geo_value=0.;parts_sum=np.zeros(3)
  for off in range(0,64,16):
   b,y,g=self.data.batch(aaex[off:off+16]);b=s.augment(b,.075)
   with torch.autocast('cuda',dtype=torch.bfloat16):h=encode_batch(self.model,b);a=F.cross_entropy(self.model.aa_head(h).float(),y,reduction='sum')/64
   a.backward();aa_value+=float(a.detach());del a,h,b,y,g
  aa={n:None if p.grad is None else p.grad.detach().clone() for n,p in self.model.named_parameters()};self.opt.zero_grad(set_to_none=True)
  for off in range(0,32,16):
   b,y,g=self.data.batch(geoex[off:off+16]);b=s.augment(b,.075);a,local,parts=s.loss_call(self.model,b,y,g)
   (self.c['lambda_local']*local.sum()/32).backward();geo_value+=float(local.detach().sum())/32;parts_sum+=parts.detach().sum(0).cpu().numpy();del a,local,parts,b,y,g
  extra={}
  if (self.update+1)%2==0:
   ex=geoex[-4:];b,y,g=self.data.batch(ex);b=s.augment(b,.075);targets=torch.tensor(np.asarray([self.data.base.ca[o][t] for h,o,t in ex]),device='cuda')
   nl,counts=s.direct_nonlocal(self.model,b,y,targets,K=4,seed=180042)
   (self.c['lambda_nonlocal']*nl).backward();self.nonlocal_targets+=len(counts);extra={'nonlocal_energy_score':float(nl.detach()),'nonlocal_targets':len(counts)};del nl,targets,b,y,g
  return aa,{'AA_NLL_stochastic':aa_value,'local_geometry_NLL':geo_value,'component_mean_NLL':(parts_sum/32).tolist(),'sample_order_sha256':s.j.digest([aaex,geoex]),**extra}
 def step(self):
  guard();start=time.monotonic();aa,out=self.backward();diag=combine(self.model,aa,self.c['group_limits']);del aa
  measure=(self.update+1)%25==0;before={n:p.detach().clone() for n,p in self.model.named_parameters()} if measure else None
  self.opt.step();self.scheduler.step();self.update+=1
  if not all(bool(torch.isfinite(p).all()) for p in self.model.parameters()):raise FloatingPointError('nonfinite updated model')
  if measure:
   if not finite_state(self.opt):raise FloatingPointError('nonfinite optimizer moments')
   diag['parameter_update_L2_by_group']={k:math.sqrt(sum(float((p.detach()-before[n]).double().square().sum()) for n,p in v)) for k,v in groups(self.model)[0].items()};del before
  torch.cuda.synchronize();LAST[0]=time.monotonic()
  return {'pilot_update':self.update,'total_optimizer_updates':35000+self.update,'seconds':time.monotonic()-start,'learning_rate_used':self.scheduler.get_last_lr()[0] if self.update==0 else self.c['learning_rate']*self.scheduler.lr_lambdas[0](self.update-1),'sampler_cursor':self.sampler.cursor,'nonlocal_targets_total':self.nonlocal_targets,'differentiated_rollout_residues':0,**out,**diag}
 def checkpoint(self,path,model_only=False):
  guard();payload={'lineage':self.lineage,'model':self.model.state_dict(),'pilot_update':self.update,'total_optimizer_updates':35000+self.update,'history':self.history}
  if not model_only:payload.update(optimizer=self.opt.state_dict(),scheduler=self.scheduler.state_dict(),rng=s.rng_state(),sampler=self.sampler.state_dict(),rollout_rng=self.rollrng.getstate(),nonlocal_targets=self.nonlocal_targets,elapsed=self.elapsed+time.monotonic()-self.start,best=self.best)
  return s.save_atomic(path,payload)
 def resume(self,path):
  st=s.load_checked(path)
  if st['lineage']!=self.lineage:raise ValueError('pilot lineage/config/source mismatch')
  if st['scheduler']['last_epoch']!=st['pilot_update'] or st['sampler']['cursor']!=5000+st['pilot_update'] or {float(x['step']) for x in st['optimizer']['state'].values()}!={35000.+st['pilot_update']}:raise ValueError('recovery counter inconsistency')
  self.model.load_state_dict(st['model'],strict=True);self.opt.load_state_dict(st['optimizer']);self.scheduler.load_state_dict(st['scheduler']);self.sampler.load_state_dict(st['sampler']);self.rollrng.setstate(st['rollout_rng']);self.update=st['pilot_update'];self.nonlocal_targets=st['nonlocal_targets'];self.history=st['history'];self.best=st['best'];self.elapsed=st['elapsed'];self.start=time.monotonic();s.restore_rng(st['rng'])

def calibrate(data,c):
 e=Engine(data,c);rows=[];bands=collections.Counter()
 for i in range(12):
  aa,out=e.backward();d=combine(e.model,aa);rows.append(d);del aa;e.update+=1
  status('TRAIN_ONLY_CLIP_CALIBRATION',batch=i+1)
 limits={k:float(np.quantile([d['group_before'][k] for d in rows],.9)) for k in ['S','A','G']}
 gs,names=groups(e.model);result={'policy':'90th percentile of12 actual64AA/32geometry TRAIN batches, before independent clipping; shared auxiliary budget0.5 already applied','batches':12,'rows':rows,'group_limits':limits,'group_parameter_names':names,'parameter_counts':{k:sum(p.numel() for n,p in v) for k,v in gs.items()},'total_trainable_parameters':sum(p.numel() for p in e.model.parameters() if p.requires_grad),'main_optimizer_updates':0,'real_optimizer_updates':0,'sampler_coverage_diagnostic_only':e.sampler.report(),'no_threshold_fit_on_VALIDATION':True}
 if result['total_trainable_parameters']!=12351746:raise ValueError('model capacity changed')
 write(R/'GROUP_GRADIENT_CLIPPING_REPORT.json',result);del e;gc.collect();torch.cuda.empty_cache();return limits

def same(a,b):
 if torch.is_tensor(a) and torch.is_tensor(b):return torch.equal(a.detach().cpu(),b.detach().cpu())
 if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def recovery_test(data,c):
 e=Engine(data,c);[e.step() for _ in range(2)];p=R/'runtime/preflight_restart.pt';e.checkpoint(p)
 expected=[e.step() for _ in range(2)];end={'model':s.cpu_copy(e.model.state_dict()),'optimizer':s.cpu_copy(e.opt.state_dict()),'scheduler':copy.deepcopy(e.scheduler.state_dict()),'sampler':copy.deepcopy(e.sampler.state_dict()),'rng':s.rng_state()};del e;gc.collect();torch.cuda.empty_cache()
 e=Engine(data,c);e.resume(p);actual=[e.step() for _ in range(2)]
 checks={'model':same(end['model'],e.model.state_dict()),'optimizer':same(end['optimizer'],e.opt.state_dict()),'scheduler':same(end['scheduler'],e.scheduler.state_dict()),'sampler':same(end['sampler'],e.sampler.state_dict()),'RNG':same(end['rng'],s.rng_state()),'sample_order':all(a['sample_order_sha256']==b['sample_order_sha256'] for a,b in zip(actual,expected))}
 errors={k:max(abs(a[k]-b[k]) for a,b in zip(actual,expected)) for k in ['AA_NLL_stochastic','local_geometry_NLL','AA_shared_L2','geometry_shared_scale']}
 result={'status':'PASS' if all(checks.values()) and max(errors.values())==0 else 'FAIL','checks':checks,'errors':errors,'engineering_optimizer_calls':6,'main_optimizer_calls':0,'checkpoint_sha256':sha(p),'checkpoint_bytes':p.stat().st_size,'retention':'ephemeral preflight full checkpoint removed after verified proof; receipt/hash retained; no historical artifact touched'}
 write(R/'GRADIENT_POLICY_TESTS.json',result)
 if result['status']!='PASS':raise RuntimeError('real deterministic restart failed')
 p.unlink();p.with_name(p.name+'.sha256').unlink();del end,e;gc.collect();torch.cuda.empty_cache()

@torch.no_grad()
def generate(model,checkpoint,tag,n=4):
 """Fixed live batch4, CPU saved Gaussian eps/uniforms; protocol-bound cache."""
 model.eval();s.set_dropout(model,.15);cp_sha=sha(checkpoint);path=R/'generation'/f'{tag}_{PROTOCOL}.json';rows=read(path) if path.exists() else []
 expected={(N,k) for N in [64,128,256,384,500] for k in range(n)}
 if any(v['checkpoint_sha256']!=cp_sha or v['protocol']!=PROTOCOL for v in rows):raise ValueError('generation cache identity')
 for N in [64,128,256,384,500]:
  for first in range(0,n,4):
   keys={(v['N'],v['sample_index']) for v in rows}
   if all((N,k) in keys for k in range(first,first+4)):continue
   if any((N,k) in keys for k in range(first,first+4)):raise ValueError('incomplete published generation batch')
   seeds=[2026101700+100*N+k for k in range(first,first+4)];draws=[]
   for k,seed in zip(range(first,first+4),seeds):
    dp=R/'generation'/f'draws_N{N}_{k:02}.npz'
    if not dp.exists():
     rng=np.random.default_rng(seed);np.savez_compressed(dp,uniform=rng.random(N),epsilon=rng.standard_normal((N,3)))
    with np.load(dp) as z:draws.append((z['uniform'].copy(),z['epsilon'].copy(),sha(dp)))
   states=[BuildState.empty(N) for _ in seeds];fails={};counts=np.zeros((4,3),int);max_discrep=np.zeros(4);start=time.monotonic()
   for t in range(N):
    active=[i for i in range(4) if i not in fails]
    if not active:break
    # Failed rows are padded with another active prefix; no cross-row attention.
    ix=[i if i in active else active[0] for i in range(4)];b=collate([states[i] for i in ix],'cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):h=encode_batch(model,b);logits=model.aa_head(h).float()
    probs=logits.double().softmax(-1).cpu().numpy();choices=[]
    for i in range(4):choices.append(min(19,int(np.searchsorted(np.cumsum(probs[i]),draws[i][0][t],side='right'))))
    labels=torch.tensor(choices,device='cuda');chem=torch.zeros_like(model.chemistry[labels])
    with torch.autocast('cuda',enabled=False):p=model.geom(torch.cat([h.float(),model.selected_aa(labels).float(),chem.float()],-1)).float().cpu()
    for i in active:
     try:
      action,info=corrected_action(p[i,:3],p[i,3:],mask_for_step(t),torch.tensor(draws[i][1][t]));counts[i]+=np.array(info['adjusted_components'],int);max_discrep[i]=max(max_discrep[i],abs(info['latent_action_log_density']-info['represented_action_log_density']))
      states[i]=append(states[i],BuildAction(AA[choices[i]],action,mask_for_step(t)))
     except (ValueError,FloatingPointError) as ex:fails[i]={'status':'GENERATION_GEOMETRY_FAILURE','error':str(ex),'completed_residues':states[i].t}
    LAST[0]=time.monotonic()
   for i,k in enumerate(range(first,first+4)):
    v={'N':N,'sample_index':k,'seed':seeds[i],'checkpoint_sha256':cp_sha,'protocol':PROTOCOL,'live_batch':4,'draws_sha256':draws[i][2],'boundary_repairs':counts[i].tolist(),'max_latent_representative_logdensity_difference':float(max_discrep[i]),'seconds_batch':time.monotonic()-start,'seconds':(time.monotonic()-start)/4}
    if i in fails:v.update(fails[i])
    else:v.update(status='COMPLETE',sequence=''.join(states[i].residues),generated_CA_A=states[i].ca.tolist(),metrics=metrics(states[i].ca.tolist()))
    rows.append(v)
   write(path,rows);status('CORRECTED_GENERATION',model=tag,N=N,completed_samples=len(rows))
 return rows

def assess(e,data):
 state=s.rng_state()
 with s.historical_evaluation():
  panels=read(R/'configs/EVALUATION_PANELS.json');tr=s.j.evaluate(e.model,data.base,panels['TRAIN'],'TRAIN');va=s.j.evaluate(e.model,data.base,panels['VALIDATION'],'VALIDATION')
  from partial_train import Engine as PE
  part=PE.evaluate(e,read(R/'configs/PARTIAL_VALIDATION_PANEL.json'),'PARTIAL')
  from j1_sequence_assess import partial_eval
  acc=partial_eval(e.model,data,sorted(read(R/'configs/PARTIAL_VALIDATION_PANEL.json'),key=lambda x:(x[2],x[0],x[1])),'VALIDATION')
 s.restore_rng(state);LAST[0]=time.monotonic()
 return {'pilot_update':e.update,'TRAIN':tr,'VALIDATION':va,'PARTIAL_VALIDATION':part,'PARTIAL_VALIDATION_AA_ACCURACY':acc}

def rotate_checkpoints(e,model=False):
 d=R/'runs/J1';path=d/f'checkpoint_{e.update:05}.pt';e.checkpoint(path);s.load_checked(path)
 if model:e.checkpoint(d/f'model_{e.update:05}.pt',True)
 keep={e.update,e.best}
 for old in d.glob('checkpoint_*.pt'):
  u=int(old.stem.split('_')[1])
  if u in keep:continue
  # Only bounded transient recovery states created by this new controller.
  with (R/'RETENTION.jsonl').open('a') as f:f.write(json.dumps({'path':str(old),'sha256':sha(old),'update':u,'reason':'verified newer full recovery; all evaluated candidates retain model snapshot'})+'\n')
  old.unlink();old.with_name(old.name+'.sha256').unlink()
 return path

def run():
 if (R/'runs/J1/COMPLETED.json').exists():print('ALREADY_STOPPED_NO_AUTOMATIC_RESUME');return
 s.j.core.setup();guard();status('DATA_LOADING');data=s.PartialData();write(R/'DATA_RECEIPT.json',data.receipt)
 c=read(R/'configs/PROPOSED.json')
 if not (R/'configs/FROZEN.json').exists():
  c['group_limits']=calibrate(data,c);write(R/'configs/FROZEN.json',c)
 else:c=read(R/'configs/FROZEN.json')
 if not (R/'GRADIENT_POLICY_TESTS.json').exists():recovery_test(data,c)
 if read(R/'GRADIENT_POLICY_TESTS.json')['status']!='PASS':raise RuntimeError('recovery preflight gate')
 e=Engine(data,c);existing=sorted((R/'runs/J1').glob('checkpoint_*.pt'))
 if existing:e.resume(existing[-1])
 if not e.history:
  base=read(OLD/'runs/J1/evaluation_05000.json');base['pilot_update']=0;write(R/'runs/J1/evaluation_00000.json',base);e.history=[base]
  status('BASELINE_CORRECTED_GENERATION')
  with s.historical_evaluation():
   parent,state=s.j.parent_model();del state;generate(parent,s.j.PARENT,'fallback',4);del parent;gc.collect();torch.cuda.empty_cache();generate(e.model,SOURCE,'source5000',4)
  # Baseline generation RNG is independent; restore source global state exactly.
  s.restore_rng(s.load_checked(SOURCE)['rng'])
 def stop(*args):STOP.set()
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 def watchdog():
  while not STOP.wait(30):
   if time.monotonic()-LAST[0]>900:STOP.set()
 LAST[0]=time.monotonic();threading.Thread(target=watchdog,daemon=True).start();reason=None;torch.cuda.reset_peak_memory_stats()
 with (R/'runs/J1'/f'updates_{e.update:05}_{int(time.time())}.jsonl').open('x') as log:
  while e.update<2500 and not STOP.is_set():
   m=e.step();log.write(json.dumps(m)+'\n')
   if e.update%25==0:log.flush();status('TRAINING',**m,**guard());print(json.dumps(m),flush=True)
   if e.update in [500,1000,1500,2500]:
    result=assess(e,data);e.history.append(result);write(R/'runs/J1'/f'evaluation_{e.update:05}.json',result)
    va=result['VALIDATION'];parent=read(OLD/'SEQUENCE_BASELINES.json')['J1']['VALIDATION'];gap=va['AA_NLL']-result['TRAIN']['AA_NLL']
    if va['AA_NLL']<e.history[e.best//500 if e.best else 0]['VALIDATION']['AA_NLL']:e.best=e.update
    recent=e.history[-2:]
    if e.update>=1000 and all(v['VALIDATION']['AA_NLL']>parent['AA_NLL']+.03 or v['VALIDATION']['geometry_NLL']>parent['geometry_NLL']+.20 for v in recent):reason='two consecutive frozen AA/local safety violations'
    write(R/'TRAIN_VALIDATION_PERPLEXITY.json',e.history);cp=rotate_checkpoints(e,True)
    state=s.rng_state()
    with s.historical_evaluation():rows=generate(e.model,cp,f'pilot{e.update:05}',4)
    s.restore_rng(state)
    ok=[v for v in rows if 'metrics' in v]
    if len(ok)!=20 or np.mean([v['metrics']['clashes_per100_residues'] for v in ok])>6.:reason='corrected generation finite/severe-clash safety safeguard'
    with (R/'EXPERIMENT_DECISIONS.md').open('a') as f:f.write(f"\nMilestone pilot+{e.update}: VALIDATION PPL{va['perplexity']:.6f}, local{va['geometry_NLL']:.6f}, gap{gap:.6f}, stop={reason}. No threshold change.\n")
    if reason:break
   elif e.update%250==0:rotate_checkpoints(e)
  final=R/'runs/J1'/f'checkpoint_{e.update:05}.pt'
  if not final.exists():e.checkpoint(final)
  write(R/'runs/J1/COMPLETED.json',{'pilot_updates':e.update,'total_optimizer_updates':35000+e.update,'latest_checkpoint':str(final),'latest_sha256':sha(final),'best_AA_update':e.best,'reason':reason or ('signal/watchdog' if STOP.is_set() else 'authorized2500maximum'),'elapsed_seconds_including_evaluation':e.elapsed+time.monotonic()-e.start,'coverage':e.sampler.report(),'source_counts':e.source_counts,'nonlocal_targets':e.nonlocal_targets,'rollout_residues':0,'resources':guard()})
 status('PILOT_STOPPED',update=e.update,reason=reason or 'authorized maximum');print('SHORT_PILOT_STOPPED',flush=True)
if __name__=='__main__':run()
