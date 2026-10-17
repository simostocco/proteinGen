"""Four authorized joint continuations; VALIDATION only, no TEST, AF, RL, or E015."""
import argparse,collections,copy,hashlib,json,math,os,resource,shutil,signal,threading,time
from pathlib import Path
import numpy as np
import torch
import scaled_bc as core
from scaled_train import Engine as PreviousEngine,cpu_copy,maxdiff,rng_digest
from protein_joint_sac.data.partial_continuation import PartialData,ContinuationSampler,R
from protein_joint_sac.data.partial_geometry import VERSION
from protein_joint_sac.data.certified_unified import file_sha
from protein_joint_sac.rl.checkpointing import save_atomic,load_checked
from protein_joint_sac.rl.bc_training import rng_state,restore_rng,code_hash
from protein_joint_sac.contracts import digest,GEOMETRY_VERSION
from protein_joint_sac.chemistry.reference_adapter import ref
OLD=Path('/mnt/d/Simone/proteinGen_data/e016_scaled10m_training_20261016_v1');core.R=R
MILESTONES=[2500,5000,10000,15000,20000];STOP=threading.Event();LAST=[time.monotonic()]
def write(p,x):core.write(p,x)
def health():
 d=core.guard(400*2**20);m={v.split(':')[0]:int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith(('MemAvailable:','MemTotal:'))}
 if m['MemAvailable']<.1*m['MemTotal']:raise RuntimeError('actual host RAM headroom below10percent')
 d.update(m);return d
def stopped(*_):STOP.set()
def watchdog():
 while not STOP.wait(30):
  if time.monotonic()-LAST[0]>600:write(R/'WATCHDOG_STOP.json',{'reason':'no progress for600seconds'});STOP.set()
class Engine:
 def __init__(self,arm,data,config):
  self.arm=arm;self.name=arm[:2];self.data=data;self.c=config;self.joint=True;self.model=core.model(self.name);self.opt=torch.optim.AdamW(self.model.parameters(),lr=config['learning_rate'],weight_decay=.01,fused=True)
  p=OLD/'runs'/self.name/'checkpoint_10000.pt'
  if file_sha(p)!=config['parent_sha256']:raise ValueError('parent checkpoint SHA mismatch')
  parent=load_checked(p);self.model.load_state_dict(parent['model'],strict=True);self.opt.load_state_dict(parent['optimizer'])
  for g in self.opt.param_groups:g['lr']=config['learning_rate'];g['initial_lr']=config['learning_rate']
  def factor(k):
   if k<config['warmup_updates']:return (k+1)/config['warmup_updates']
   phase=min(1.,(k-config['warmup_updates'])/(config['max_updates']-config['warmup_updates']));return .1+.9*.5*(1+math.cos(math.pi*phase))
  self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.opt,factor);self.sampler=ContinuationSampler(data,parent['sampler'],arm.endswith('PARTIAL'),config['partial_geometry_oversample']);del parent
  self.update=0;self.validation=[];self.start=time.monotonic();self.elapsed_prior=0.;self.band_counts=np.zeros(5,dtype=np.int64);self.anomaly_exposure=0
  self.flagged={r['sequence_hash'] for r in json.load(open(core.PILOT/'priors/COORDINATE_EXTREMES.json'))['examples']}
  quality=R/'view/OBSERVED_GEOMETRY_DIAGNOSTICS.json'
  if quality.exists():self.flagged.update(r['sequence_hash'] for r in json.load(open(quality)) if r['split']=='TRAIN')
  self.lineage={'arm':arm,'config':config,'package_sha256':code_hash(),'entrypoint_sha256':file_sha(__file__),'core_entrypoint_sha256':file_sha(core.__file__),'parent_checkpoint':str(p),'geometry_version':GEOMETRY_VERSION,'mask_version':VERSION,'chemistry':ref().__dict__,'view_sha256':file_sha(R/'view/VIEW_FROZEN.json'),'validation_sha256':file_sha(R/'configs/VALIDATION_PANELS.json'),'optimizer':'RESTORED_PARENT_ADAMW_MOMENTS_NEW_SCHEDULE','TEST':'NEVER_EVALUATED_OR_USED_FOR_DEVELOPMENT','torch':str(torch.__version__),'precision':'BF16','normalizers':'unchanged TRAIN priors; no new fitted scaler'}
 def step(self):
  if self.update>=self.c['max_updates']:raise ValueError('continuation ceiling')
  health();a=time.monotonic();examples=self.sampler.draw(64);ordered=sorted(examples,key=lambda e:(e[2],e[0],e[1]));self.model.train();self.opt.zero_grad(set_to_none=True);sums=np.zeros(5);prep=0.;calc=0.
  for i in range(0,64,self.c['live_batch']):
   q=time.monotonic();b,y,g=self.data.batch(ordered[i:i+self.c['live_batch']]);prep+=time.monotonic()-q;q=time.monotonic();loss,aa,ge,parts=core.loss_call(self.model,b,y,g,self.c);(loss.sum()/64).backward();calc+=time.monotonic()-q;sums+=np.array([float(aa.detach().sum()),float(ge.detach().sum()),*parts.detach().sum(0).cpu().tolist()]);del loss,aa,ge,parts,b,y,g
  norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True);q=time.monotonic();self.opt.step();self.scheduler.step();torch.cuda.synchronize();optimizer=time.monotonic()-q;self.update+=1
  if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError('nonfinite parameters')
  for h,o,t in examples:
   N=len(self.data.seqs[h]);band=next(i for i,(lo,hi) in enumerate([(20,64),(65,128),(129,256),(257,384),(385,500)]) if lo<=N<=hi);self.band_counts[band]+=1;self.anomaly_exposure+=int(h in self.flagged)
  LAST[0]=time.monotonic()
  return {'arm':self.arm,'update':self.update,'total_lineage_updates':10000+self.update,'AA_NLL':sums[0]/64,'geometry_NLL':sums[1]/64,'component_NLL_sums':sums[2:].tolist(),'gradient_norm_preclip':float(norm),'learning_rate':self.opt.param_groups[0]['lr'],'seconds':time.monotonic()-a,'data_seconds':prep,'forward_backward_seconds':calc,'optimizer_seconds':optimizer,'decisions':self.sampler.cursor,'prefix_tokens':self.sampler.tokens,'distinct_additional_TRAIN_identities':len(self.sampler.seen),'branch_decisions':dict(self.sampler.branch_counts),'geometry_component_targets':self.sampler.component_counts.tolist(),'AA_targets_at_missing_CA':self.sampler.missing_aa,'length_exposures':self.band_counts.tolist(),'anomaly_identity_decisions':self.anomaly_exposure,'sample_order_sha256':digest(examples),'complete_stream_sha256':self.sampler.complete_stream_sha256,'partial_stream_sha256':self.sampler.partial_stream_sha256}
 @torch.no_grad()
 def evaluate(self,panel,kind):
  self.model.eval();start=time.monotonic();by=collections.defaultdict(lambda:np.zeros(6));bands={str(i):np.zeros(9) for i in range(5)};coverage=collections.defaultdict(lambda:np.zeros(9));tot=np.zeros(5);counts=np.zeros(3,dtype=np.int64);missing_AA=0
  ordered=sorted(panel,key=lambda e:(e[2],e[0],e[1]))
  for i in range(0,len(ordered),self.c['live_batch']):
   ex=ordered[i:i+self.c['live_batch']]
   if any(self.data.proposal.sequences[h]['split']!='VALIDATION' for h,o,t in ex):raise ValueError('VALIDATION-only evaluation')
   b,y,g=self.data.batch(ex);loss,aa,ge,parts=core.loss_call(self.model,b,y,g,self.c);mask=b['geometry_target_valid'];tot+=np.array([float(aa.sum()),float(ge.sum()),*parts.sum(0).cpu().tolist()]);counts+=mask.sum(0).cpu().numpy()
   for (h,o,t),a,c,parts_row,mask_row in zip(ex,aa.cpu().tolist(),ge.cpu().tolist(),parts.cpu().tolist(),mask.cpu().tolist()):
    n=len(self.data.seqs[h]);k=next(j for j,(lo,hi) in enumerate([(20,64),(65,128),(129,256),(257,384),(385,500)]) if lo<=n<=hi);v=np.array([a,c,1.,*parts_row,*mask_row]);by[h]+=np.array([a,c,1.,*mask_row]);bands[str(k)]+=v
    frac=float(self.data.cache[o]['seen'].mean()) if o in self.data.cache else 1.;bucket='1_COMPLETE' if frac==1 else ('0-.25' if frac<.25 else '.25-.50' if frac<.5 else '.50-.75' if frac<.75 else '.75-1');coverage[bucket]+=v;missing_AA+=int(o in self.data.cache and not self.data.cache[o]['seen'][t])
   LAST[0]=time.monotonic()
  identities=np.array([v[:2]/v[2] for v in by.values()]);rng=np.random.default_rng(742);boot=np.array([identities[rng.integers(len(identities),size=len(identities))].mean(0) for _ in range(500)]);n=len(panel)
  return {'additional_update':self.update,'population':kind,'identities':len(by),'prefix_decisions':n,'AA_NLL':float(tot[0]/n),'geometry_valid_component_mean_per_decision':float(tot[1]/n),'selection_objective':float((tot[0]+tot[1])/n),'component_NLL':[float(x/n) if n else None for x,n in zip(tot[2:],counts)],'component_target_counts':counts.tolist(),'identity_bootstrap_95pct_AA_and_geometry':np.quantile(boot,[.025,.975],axis=0).tolist(),'uncertainty_limit':'identity resampling; protected-family correlations remain','per_identity_AA_geometry_mean':{h:(v[:2]/v[2]).tolist() for h,v in by.items()},'AA_targets_at_missing_CA':missing_AA,'length_bands':{k:{'decisions':int(v[2]),'AA_NLL':float(v[0]/max(1,v[2])),'geometry_NLL':float(v[1]/max(1,v[2])),'component_counts':v[6:].astype(int).tolist(),'component_NLL':[float(x/n) if n else None for x,n in zip(v[3:6],v[6:])]} for k,v in bands.items()},'observed_coverage_buckets':{k:{'decisions':int(v[2]),'AA_NLL':float(v[0]/max(1,v[2])),'geometry_NLL':float(v[1]/max(1,v[2])),'component_counts':v[6:].astype(int).tolist()} for k,v in coverage.items()},'panel_sha256':digest(panel),'seconds':time.monotonic()-start}
 def checkpoint(self,path):
  if path.exists():raise ValueError('immutable checkpoint exists')
  health();return save_atomic(path,{'lineage':self.lineage,'model':self.model.state_dict(),'optimizer':self.opt.state_dict(),'scheduler':self.scheduler.state_dict(),'rng':rng_state(),'sampler':self.sampler.state_dict(),'additional_update':self.update,'validation':self.validation,'elapsed':self.elapsed_prior+time.monotonic()-self.start,'length_exposures':self.band_counts.tolist(),'anomaly_exposure':self.anomaly_exposure})
 def resume(self,path):
  s=load_checked(path)
  if s['lineage']!=self.lineage:raise ValueError('continuation checkpoint lineage mismatch')
  self.model.load_state_dict(s['model']);self.opt.load_state_dict(s['optimizer']);self.scheduler.load_state_dict(s['scheduler']);self.sampler.load_state_dict(s['sampler']);self.update=s['additional_update'];self.validation=s['validation'];self.elapsed_prior=s['elapsed'];self.start=time.monotonic();self.band_counts=np.array(s['length_exposures']);self.anomaly_exposure=s['anomaly_exposure'];restore_rng(s['rng'])
 def generate(self,N,seed):return PreviousEngine.generate(self,N,seed)

def retain_best_latest(d,engine):
 best=min(engine.validation,key=lambda v:v['objective'])['update'];keep={engine.update,best}
 for p in sorted(d.glob('checkpoint_*.pt')):
  u=int(p.stem.split('_')[-1])
  if u in keep:continue
  if p.is_symlink() or p.stat().st_nlink!=1 or p.parent.resolve()!=d.resolve() or not p.resolve().is_relative_to(R/'runs'):raise RuntimeError('unsafe new-checkpoint retention path')
  expected=p.with_name(p.name+'.sha256').read_text().strip()
  if file_sha(p)!=expected:raise RuntimeError('superseded new checkpoint checksum mismatch')
  receipt={'path':str(p),'sha256':expected,'bytes':p.stat().st_size,'reason':'frozen retention of new-task best+latest only; milestone metrics/SHA retained','kept_updates':sorted(keep),'D_free_before':shutil.disk_usage(R).free}
  p.unlink();p.with_name(p.name+'.sha256').unlink();receipt['D_free_after']=shutil.disk_usage(R).free
  with (d/'CHECKPOINT_RETENTION.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')

def run(arm):
 core.setup();config=json.load(open(R/'configs/CONTINUATION_FROZEN.json'))[arm];gate=json.load(open(R/'PREFLIGHT_PASS.json'))
 if gate['status']!='PASS' or gate['config_sha256']!=file_sha(R/'configs/CONTINUATION_FROZEN.json') or gate['view_sha256']!=file_sha(R/'view/VIEW_FROZEN.json') or gate['package_sha256']!=code_hash() or gate['training_entrypoint_sha256']!=file_sha(__file__) or gate['core_entrypoint_sha256']!=file_sha(core.__file__):raise ValueError('required CPU/GPU/data preflight gate')
 d=R/'runs'/arm;d.mkdir(exist_ok=True)
 if (d/'COMPLETED.json').exists():
  receipt=json.loads((d/'COMPLETED.json').read_text())
  if receipt['config']!=config or file_sha(receipt['selected_checkpoint'])!=receipt['selected_sha256']:raise ValueError('completed run integrity mismatch')
  print('ALREADY_COMPLETE '+arm,flush=True);return
 write(R/'HEARTBEAT.json',{'arm':arm,'phase':'DATA_LOADING','timestamp':time.time()});data=PartialData();write(d/'DATA_RECEIPT.json',data.receipt);engine=Engine(arm,data,config);panels=json.load(open(R/'configs/VALIDATION_PANELS.json'));complete=panels['complete'];partial=panels['partial'];existing=sorted(d.glob('checkpoint_*.pt'))
 if existing:
  rejected=[]
  for cp in reversed(existing):
   try:engine.resume(cp);break
   except (OSError,RuntimeError,ValueError,EOFError) as ex:rejected.append({'path':str(cp),'reason':repr(ex)})
  else:raise RuntimeError('no valid durable continuation checkpoint; preserve failed files')
  if rejected:write(d/f'RECOVERY_REJECTED_{int(time.time())}.json',rejected)
 else:
  state=rng_state();full=engine.evaluate(complete,'COMMON_COMPLETE_VALIDATION');part=engine.evaluate(partial,'PARTIAL_VALIDATION');restore_rng(state);write(d/'validation_00000.json',{'complete':full,'partial':part});engine.validation.append({'update':0,'objective':full['selection_objective']})
 signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped);LAST[0]=time.monotonic();threading.Thread(target=watchdog,daemon=True).start();torch.cuda.reset_peak_memory_stats()
 path=d/f'updates_from_{engine.update:05}_{int(time.time())}.jsonl'
 with path.open('x') as log:
  while engine.update<20000 and not STOP.is_set():
   m=engine.step();log.write(json.dumps(m)+'\n')
   if engine.update%25==0:log.flush();write(R/'HEARTBEAT.json',{'phase':'TRAINING','timestamp':time.time(),**m,**health()});print(json.dumps(m),flush=True)
   if engine.update%500==0 and engine.update not in MILESTONES:
    log.flush();cp=d/f'checkpoint_{engine.update:05}.pt';engine.checkpoint(cp);load_checked(cp);retain_best_latest(d,engine)
   if engine.update in MILESTONES:
    log.flush();state=rng_state();full=engine.evaluate(complete,'COMMON_COMPLETE_VALIDATION');part=engine.evaluate(partial,'PARTIAL_VALIDATION');write(d/f'validation_{engine.update:05}.json',{'complete':full,'partial':part});engine.validation.append({'update':engine.update,'objective':full['selection_objective']});generated=[]
    for N in [64,128,256,384,500]:generated.append(engine.generate(N,830000+engine.update+N));LAST[0]=time.monotonic()
    write(d/f'generation_{engine.update:05}.json',generated);restore_rng(state);cp=d/f'checkpoint_{engine.update:05}.pt';sha=engine.checkpoint(cp);load_checked(cp);write(d/f'milestone_{engine.update:05}.json',{'checkpoint':str(cp),'sha256':sha,'metrics':m,'resources':health()})
    retain_best_latest(d,engine)
    vals=engine.validation
    if len(vals)>=4 and all(v['objective']>min(z['objective'] for z in vals[:-3])+1e-4 for v in vals[-3:]):write(d/'EARLY_STOP.json',{'reason':'three consecutive common-complete validation milestones worse than earlier best','additional_update':engine.update});break
  if STOP.is_set():
   cp=d/f'checkpoint_{engine.update:05}.pt'
   if not cp.exists():engine.checkpoint(cp)
   write(d/'STOPPED.json',{'additional_update':engine.update,'checkpoint':str(cp)});raise RuntimeError('graceful stop; do not advance another arm')
 best=min(engine.validation,key=lambda x:x['objective']);cp=(d/f"checkpoint_{best['update']:05}.pt") if best['update'] else OLD/'runs'/engine.name/'checkpoint_10000.pt'
 write(d/'COMPLETED.json',{'arm':arm,'additional_updates':engine.update,'total_lineage_updates':10000+engine.update,'additional_prefix_decisions':engine.sampler.cursor,'distinct_additional_TRAIN_identities':len(engine.sampler.seen),'branch_decisions':dict(engine.sampler.branch_counts),'geometry_component_targets':engine.sampler.component_counts.tolist(),'AA_at_missing_CA':engine.sampler.missing_aa,'selected_checkpoint':str(cp),'selected_sha256':file_sha(cp),'best_common_complete_objective':best['objective'],'elapsed_seconds':engine.elapsed_prior+time.monotonic()-engine.start,'resources':health(),'config':config})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['J1-FULL','J1-PARTIAL','J2-FULL','J2-PARTIAL'],required=True);args=ap.parse_args()
 try:run(args.arm)
 except Exception as ex:write(R/'BLOCKED.json',{'arm':args.arm,'error':repr(ex),'timestamp':time.time()});raise
