"""One recoverable J1 lineage with periodic differentiable global geometry losses."""
import collections,copy,json,math,os,random,signal,threading,time
from pathlib import Path
import numpy as np
import torch
import j1_global as j
from protein_joint_sac.data.partial_continuation import PartialData
from protein_joint_sac.data.scaled_supervised import Sampler
from protein_joint_sac.rl.global_geometry import rollout,VERSION
from protein_joint_sac.rl.bc_training import rng_state,restore_rng,code_hash
from protein_joint_sac.rl.checkpointing import save_atomic,load_checked
from protein_joint_sac.data.certified_unified import file_sha
from scaled_train import Engine as GenerationEngine,cpu_copy,maxdiff,rng_digest
STOP=threading.Event();LAST=[time.monotonic()]
def stop(*_):STOP.set()
def watchdog():
 while not STOP.wait(30):
  if time.monotonic()-LAST[0]>600:STOP.set()
class MixedSampler:
 def __init__(self,data,parent):
  self.data=data;self.complete=Sampler(data.base);self.complete.load_state_dict(parent['complete'])
  self.ri=random.Random(20261018);self.ro=random.Random(20261019);self.rt=random.Random(20261020)
  self.cursor=0;self.seen=set();self.counts=collections.Counter();self.tokens=0;self.bands=[0]*5
 def draw(self):
  out=self.complete.draw(56)
  for _ in range(8):
   h=self.ri.choice(self.data.ids['TRAIN']);o=self.ro.choice(self.data.owners['TRAIN'][h]);t=self.rt.randrange(len(self.data.seqs[h]));out.append((h,o,t))
  for h,o,t in out:
   partial=o in self.data.rows;self.cursor+=1;self.seen.add(h);self.tokens+=t;self.bands[j.band(len(self.data.seqs[h]))]+=1;self.counts['partial' if partial else 'complete']+=1
   if partial and not self.data.cache[o]['seen'][t]:self.counts['partial_AA_at_missing_CA']+=1
  return out
 def state_dict(self):
  return {'complete':self.complete.state_dict(),'rng':[r.getstate() for r in [self.ri,self.ro,self.rt]],'cursor':self.cursor,'seen':sorted(self.seen),'counts':dict(self.counts),'tokens':self.tokens,'bands':self.bands.copy()}
 def load_state_dict(self,x):
  self.complete.load_state_dict(x['complete'])
  for r,s in zip([self.ri,self.ro,self.rt],x['rng']):r.setstate(s)
  self.cursor=x['cursor'];self.seen=set(x['seen']);self.counts=collections.Counter(x['counts']);self.tokens=x['tokens'];self.bands=x['bands']
class Engine:
 def __init__(self,data,config):
  if config['effective_batch']!=64 or config['partial_decisions_per_batch']!=8 or 64%config['live_batch']:raise ValueError('frozen effective batch and87.5/12.5 mixture required')
  self.data=data;self.c=config;self.name='J1';self.joint=True;self.model,parent=j.parent_model()
  self.opt=torch.optim.AdamW(self.model.parameters(),lr=config['learning_rate'],weight_decay=.01,fused=True);self.opt.load_state_dict(parent['optimizer'])
  if any(float(s['step'])!=30000 for s in self.opt.state.values()):raise ValueError('parent optimizer update count')
  for g in self.opt.param_groups:g['lr']=config['learning_rate'];g['initial_lr']=config['learning_rate']
  def factor(k):
   if k<config['warmup_updates']:return (k+1)/config['warmup_updates']
   p=min(1.,(k-config['warmup_updates'])/(config['max_updates']-config['warmup_updates']))
   return .1+.9*.5*(1+math.cos(math.pi*p))
  self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.opt,factor);self.sampler=MixedSampler(data,parent['sampler']);restore_rng(parent['rng']);del parent
  self.rollrng=random.Random(config['rollout_seed']);self.update=0;self.rollout_residues=0;self.history=[];self.elapsed=0.;self.start=time.monotonic();self.best=0;self.geometry_best=0
  self.lineage={'config':config,'parent_sha256':j.PARENT_SHA,'package_sha256':code_hash(),'entrypoint_sha256':file_sha(__file__),'baseline_entrypoint_sha256':file_sha(j.__file__),'split_sha256':file_sha(j.core.P/'PROPOSAL_MANIFEST.json'),'view_sha256':file_sha(j.OLD/'view/VIEW_FROZEN.json'),'panels_sha256':file_sha(j.R/'configs/EVALUATION_PANELS.json'),'baseline_sha256':file_sha(j.R/'SEQUENCE_BASELINES.json'),'geometry_version':'ca_internal_signed_v1','rollout_version':VERSION,'chemistry':'DISABLED; existing registry unchanged','prior_sha256':{n:file_sha(j.core.PILOT/'priors'/n) for n in ['GEOMETRY_PRIORS.json','SEQUENCE_STATS.json']},'torch':str(torch.__version__),'TEST':'NOT_USED'}
 def step(self):
  j.guard();start=time.monotonic();examples=self.sampler.draw();ordered=sorted(examples,key=lambda e:(e[2],e[0],e[1]));self.model.train();self.opt.zero_grad(set_to_none=True);tot=np.zeros(5);components=np.zeros(3,int);prep=0.
  for i in range(0,64,self.c['live_batch']):
   q=time.monotonic();ex=ordered[i:i+self.c['live_batch']];b,y,g=self.data.batch(ex);prep+=time.monotonic()-q
   for k,(h,o,t) in enumerate(ex):
    if o in self.data.rows:b['geometry_target_valid'][k]=False
   loss,aa,ge,part=j.core.loss_call(self.model,b,y,g,self.c);(loss.sum()/64).backward()
   tot+=np.array([float(aa.detach().sum()),float(ge.detach().sum()),*part.detach().sum(0).cpu().tolist()]);components+=b['geometry_target_valid'].sum(0).cpu().numpy()
   del b,y,g,loss,aa,ge,part
  extra={};global_seconds=0.
  if (self.update+1)%self.c['rollout_period']==0:
   q=time.monotonic();h=self.rollrng.choice(self.data.base.jointids['TRAIN']);o=self.rollrng.choice(self.data.base.owners[h]);N=len(self.data.seqs[h]);H=min(self.c['rollout_horizon'],N-3);t=self.rollrng.randrange(3,N-H+1)
   ref=torch.tensor(self.data.base.ca[o][:t+H],device='cuda');ids=torch.tensor([j.AA.index(v) for v in self.data.seqs[h][:t+H]],device='cuda')
   r=rollout(self.model,ref[:t],ids,N,H,ref,recompute=True,clash_radius=self.c['clash_radius_A'])
   ramp=min(1.,(self.update+1)/self.c['rollout_weight_ramp_updates']);aux=ramp*(self.c['lambda_Rg']*r['Rg_loss']+self.c['lambda_clash']*r['clash_loss']);aux.backward()
   extra={'rollout_horizon':H,'rollout_start':t,'rollout_sequence_hash':h,'rollout_observation':o,'Rg_loss':float(r['Rg_loss'].detach()),'clash_loss':float(r['clash_loss'].detach()),'auxiliary_weighted_loss':float(aux.detach()),'auxiliary_ramp':ramp,'generated_prefix_AA_NLL':float(r['generated_prefix_AA_NLL'].detach())};self.rollout_residues+=H
   del r,aux,ref,ids;global_seconds=time.monotonic()-q
  norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True);lr=self.opt.param_groups[0]['lr'];self.opt.step();self.scheduler.step();torch.cuda.synchronize();self.update+=1
  if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError('nonfinite model')
  LAST[0]=time.monotonic()
  return {'additional_update':self.update,'total_lineage_updates':30000+self.update,'AA_NLL':tot[0]/64,'geometry_NLL':tot[1]/64,'component_NLL_sums':tot[2:].tolist(),'component_valid_counts':components.tolist(),'gradient_norm_preclip':float(norm),'learning_rate':lr,'seconds':time.monotonic()-start,'data_seconds':prep,'global_seconds':global_seconds,'ordinary_decisions':self.sampler.cursor,'rollout_residues':self.rollout_residues,'branch_counts':dict(self.sampler.counts),'distinct_identities':len(self.sampler.seen),'length_exposures':self.sampler.bands,'sample_order_sha256':j.digest(examples),**extra}
 def checkpoint(self,path):
  if path.exists():raise ValueError('never overwrite durable checkpoint')
  j.guard();return save_atomic(path,{'lineage':self.lineage,'model':self.model.state_dict(),'optimizer':self.opt.state_dict(),'scheduler':self.scheduler.state_dict(),'sampler':self.sampler.state_dict(),'rng':rng_state(),'rollout_rng':self.rollrng.getstate(),'additional_update':self.update,'rollout_residues':self.rollout_residues,'history':self.history,'best':self.best,'geometry_best':self.geometry_best,'elapsed':self.elapsed+time.monotonic()-self.start})
 def resume(self,path):
  s=load_checked(path)
  if s['lineage']!=self.lineage:raise ValueError('lineage/source/data/config changed')
  self.model.load_state_dict(s['model']);self.opt.load_state_dict(s['optimizer']);self.scheduler.load_state_dict(s['scheduler']);self.sampler.load_state_dict(s['sampler']);self.rollrng.setstate(s['rollout_rng']);self.update=s['additional_update'];self.rollout_residues=s['rollout_residues'];self.history=s['history'];self.best=s['best'];self.geometry_best=s['geometry_best'];self.elapsed=s['elapsed'];self.start=time.monotonic();restore_rng(s['rng'])
 def generate(self,N,seed):return GenerationEngine.generate(self,N,seed)
def prune(d,engine):
 keep={engine.update,engine.best,engine.geometry_best}
 for p in sorted(d.glob('checkpoint_*.pt')):
  u=int(p.stem.split('_')[1])
  if u in keep:continue
  if p.is_symlink() or p.stat().st_nlink!=1 or not p.resolve().is_relative_to(j.R/'runs'):raise ValueError('unsafe retention path')
  sha=p.with_name(p.name+'.sha256').read_text().strip()
  if file_sha(p)!=sha:raise ValueError('retention checksum')
  receipt={'update':u,'path':str(p),'sha256':sha,'bytes':p.stat().st_size,'reason':'new-lineage capped best+geometry+latest retention; historical files untouched'}
  p.unlink();p.with_name(p.name+'.sha256').unlink()
  with (d/'RETENTION.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
def proof():
 j.core.setup();c=json.load(open(j.R/'configs/CONTINUATION_FROZEN.json'));data=PartialData();e=Engine(data,c);rows=[]
 # Separate engineering preflight, discarded weights; verify a global-loss optimizer step.
 for _ in range(15):rows.append(e.step())
 cp=j.R/'tests'/f'engineering_update15_{int(time.time())}.pt';e.checkpoint(cp);expected=e.step();model=cpu_copy(e.model.state_dict());opt=cpu_copy(e.opt.state_dict());sampler=e.sampler.state_dict();schedule=copy.deepcopy(e.scheduler.state_dict());rng=rng_digest();roll=e.rollrng.getstate()
 restored=Engine(data,c);restored.resume(cp);actual=restored.step()
 pe=maxdiff(model,restored.model.state_dict());oe=maxdiff(opt,restored.opt.state_dict())
 same=sampler==restored.sampler.state_dict() and schedule==restored.scheduler.state_dict() and roll==restored.rollrng.getstate() and rng==rng_digest() and expected['sample_order_sha256']==actual['sample_order_sha256']
 out={'status':'PASS' if same and pe<2e-5 and oe<1e-4 else 'FAIL','parameter_max_abs_error':pe,'optimizer_max_abs_error':oe,'sampler_scheduler_RNG_and_rollout_equal':same,'AA_NLL_abs_error':abs(expected['AA_NLL']-actual['AA_NLL']),'Rg_loss_abs_error':abs(expected['Rg_loss']-actual['Rg_loss']),'clash_loss_abs_error':abs(expected['clash_loss']-actual['clash_loss']),'scope':'12.35M REAL COMPLETE+PARTIAL AA BATCH; ACTUAL REPARAMETERIZED GLOBAL LOSS AT UPDATE16','engineering_only_optimizer_updates':17,'main_optimizer_updates':0,'resources':j.guard(),'config_sha256':file_sha(j.R/'configs/CONTINUATION_FROZEN.json'),'package_sha256':code_hash(),'training_entrypoint_sha256':file_sha(__file__),'baseline_entrypoint_sha256':file_sha(j.__file__)}
 j.write(j.R/'tests/GLOBAL_RESTART.json',out);j.write(j.R/'tests/ENGINEERING_UPDATES.json',rows+[expected,actual])
 if out['status']!='PASS':raise RuntimeError('global optimizer restart failed')
 print('GLOBAL_RESTART_PASS',flush=True)
def generation_summary(engine,u,n=2):
 from partial_finalize import metrics
 j.core.R=j.R # importing legacy metrics must not change the new artifact root
 rows=[]
 for N in [64,128,256,384,500]:
  for k in range(n):
   v=engine.generate(N,2026101700+100*N+k)
   if v['status']=='COMPLETE':v['metrics']=metrics(v['generated_CA_A'])
   rows.append(v);LAST[0]=time.monotonic()
 j.write(j.R/'runs/J1'/f'generation_{u:05}.json',rows)
 good=[v for v in rows if 'metrics' in v]
 native=json.load(open(j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json'))['native']['TRAIN']
 rgerr=[];cl=[]
 for N in [64,128,256,384,500]:
  target=np.median([v['metrics']['Rg_over_N_cuberoot'] for v in native[str(N)]['rows']])
  ss=[v['metrics'] for v in good if v['N']==N]
  rgerr.extend(abs(math.log(v['Rg_over_N_cuberoot']/target)) for v in ss);cl.extend(v['clashes_per100_residues'] for v in ss)
 return {'successes':len(good),'samples':len(rows),'clashes_per100_mean':float(np.mean(cl)),'mean_absolute_log_Rg_ratio_to_TRAIN_median':float(np.mean(rgerr))}
def run():
 j.core.setup();c=json.load(open(j.R/'configs/CONTINUATION_FROZEN.json'));gate=json.load(open(j.R/'tests/GLOBAL_RESTART.json'))
 if gate['status']!='PASS' or gate['config_sha256']!=file_sha(j.R/'configs/CONTINUATION_FROZEN.json') or gate['package_sha256']!=code_hash() or gate['training_entrypoint_sha256']!=file_sha(__file__) or gate['baseline_entrypoint_sha256']!=file_sha(j.__file__):raise ValueError('preflight source/config gate')
 precision=json.load(open(j.R/'tests/GLOBAL_PRECISION_PARITY.json'))
 if precision['status']!='PASS':raise ValueError('global precision gate')
 j.write(j.R/'HEARTBEAT.json',{'phase':'DATA_LOADING','timestamp':time.time()})
 data=PartialData();engine=Engine(data,c);d=j.R/'runs/J1';d.mkdir(parents=True,exist_ok=True);panels=json.load(open(j.R/'configs/EVALUATION_PANELS.json'));baseline=json.load(open(j.R/'SEQUENCE_BASELINES.json'))['J1'];existing=sorted(d.glob('checkpoint_*.pt'),key=lambda p:int(p.stem.split('_')[1]))
 if existing:
  failures=[]
  for p in reversed(existing):
   try:engine.resume(p);break
   except (OSError,ValueError,RuntimeError,EOFError) as ex:failures.append({'path':str(p),'error':repr(ex)})
  else:raise ValueError('no valid checkpoint; preserve all failed attempts')
  j.write(d/f'RECOVERY_{int(time.time())}.json',{'verified_update':engine.update,'checkpoint':str(p),'rejected':failures,'note':'updates after last durable state may have been lost; replay from restored sampler/RNG; raw attempts retained'})
 else:engine.history=[{'update':0,'TRAIN':{k:baseline['TRAIN'][k] for k in ['AA_NLL','perplexity','geometry_NLL']},'VALIDATION':{k:baseline['VALIDATION'][k] for k in ['AA_NLL','perplexity','geometry_NLL']},'eligible':True}]
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);LAST[0]=time.monotonic();threading.Thread(target=watchdog,daemon=True).start();torch.cuda.reset_peak_memory_stats();stop_reason=None
 with (d/f'updates_from_{engine.update:05}_{int(time.time())}.jsonl').open('x') as log:
  while engine.update<c['max_updates'] and not STOP.is_set():
   m=engine.step();log.write(json.dumps(m)+'\n')
   if engine.update%25==0:
    log.flush();j.write(j.R/'HEARTBEAT.json',{'phase':'TRAINING','timestamp':time.time(),**m,**j.guard()});print(json.dumps(m),flush=True)
   if engine.update in c['milestones']:
    log.flush();state=rng_state();train=j.evaluate(engine.model,data.base,panels['TRAIN'],'TRAIN');LAST[0]=time.monotonic();val=j.evaluate(engine.model,data.base,panels['VALIDATION'],'VALIDATION');LAST[0]=time.monotonic()
    gen=generation_summary(engine,engine.update);restore_rng(state);g=c['guardrails'];eligible=val['AA_NLL']<=g['AA_NLL_max'] and val['geometry_NLL']<=g['local_geometry_max'] and gen['successes']==gen['samples']
    # A severe increase in close overlaps also disqualifies a candidate.
    parentgen=json.load(open(j.OLD/'runs/J1-FULL/SELECTED_GENERATION.json'));basecl=float(np.mean([v['independent_CA_metrics']['clashes_per100_residues'] for v in parentgen if v['sample_index']<2]))
    eligible=eligible and gen['clashes_per100_mean']<=g['severe_clash_ratio']*basecl+g['severe_clash_additive_per100']
    entry={'update':engine.update,'TRAIN':{k:train[k] for k in ['AA_NLL','perplexity','geometry_NLL']},'VALIDATION':{k:val[k] for k in ['AA_NLL','perplexity','geometry_NLL']},'generation':gen,'eligible':eligible};engine.history.append(entry);j.write(d/f'evaluation_{engine.update:05}.json',{'TRAIN':train,'VALIDATION':val,'generation':gen,'eligible':eligible})
    currentbest=next(v for v in engine.history if v['update']==engine.best)
    if eligible and val['AA_NLL']<currentbest['VALIDATION']['AA_NLL']:engine.best=engine.update
    geom_history=[v for v in engine.history if v.get('eligible') and 'generation' in v]
    if geom_history:engine.geometry_best=min(geom_history,key=lambda v:v['generation']['mean_absolute_log_Rg_ratio_to_TRAIN_median'])['update']
    recent=engine.history[-2:]
    if len(recent)==2 and all(v['VALIDATION']['AA_NLL']>baseline['VALIDATION']['AA_NLL']+g['emergency_AA_NLL_delta'] or v['VALIDATION']['geometry_NLL']>baseline['VALIDATION']['geometry_NLL']+g['emergency_local_geometry_delta'] for v in recent):stop_reason='two consecutive severe sequence/local-geometry guardrail violations'
    if len(engine.history)>=5 and engine.best==0 and all(v['VALIDATION']['AA_NLL']>baseline['VALIDATION']['AA_NLL']+.015 for v in engine.history[-3:]):stop_reason='sustained unacceptable sequence deterioration; original parent retained'
    if engine.update==20000:
     prior=next(v for v in engine.history if v['update']==10000);progress=val['AA_NLL']<=prior['VALIDATION']['AA_NLL']-.001 or gen['mean_absolute_log_Rg_ratio_to_TRAIN_median']<=.9*prior['generation']['mean_absolute_log_Rg_ratio_to_TRAIN_median'] or gen['clashes_per100_mean']<=.9*prior['generation']['clashes_per100_mean']
     if not (eligible and progress):stop_reason='no useful VALIDATION/geometry progress qualifying for extension after20000'
    j.write(j.R/'TRAIN_VALIDATION_PERPLEXITY_HISTORY.json',engine.history)
   if engine.update%250==0 or stop_reason:
    cp=d/f'checkpoint_{engine.update:05}.pt';sha=engine.checkpoint(cp);saved=load_checked(cp)
    if saved['additional_update']!=engine.update:raise ValueError('durable checkpoint update mismatch')
    del saved;prune(d,engine);j.write(d/f'receipt_{engine.update:05}.json',{'update':engine.update,'checkpoint':str(cp),'sha256':sha,'resources':j.guard(),'stop_reason':stop_reason})
   if stop_reason:break
  if STOP.is_set():
   cp=d/f'checkpoint_{engine.update:05}.pt'
   if not cp.exists():engine.checkpoint(cp)
   j.write(d/'STOPPED.json',{'update':engine.update,'reason':'signal/watchdog','checkpoint':str(cp)});raise RuntimeError('safely stopped; resume required')
 selected=j.PARENT if engine.best==0 else d/f'checkpoint_{engine.best:05}.pt';latest=d/f'checkpoint_{engine.update:05}.pt'
 j.write(d/'COMPLETED.json',{'status':'FINISHED_AUTHORIZED_CONTINUATION','additional_updates':engine.update,'total_lineage_updates':30000+engine.update,'ordinary_decisions':engine.sampler.cursor,'rollout_residues':engine.rollout_residues,'branch_counts':dict(engine.sampler.counts),'distinct_new_exposed_identities':len(engine.sampler.seen),'selected_update':engine.best,'selected_checkpoint':str(selected),'selected_sha256':file_sha(selected),'latest_checkpoint':str(latest),'latest_sha256':file_sha(latest),'stop_reason':stop_reason or 'maximum authorized updates reached','history':engine.history,'elapsed_seconds':engine.elapsed+time.monotonic()-engine.start,'resources':j.guard(),'config':c})
 print('J1_GLOBAL_MAIN_COMPLETE',flush=True)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['proof','train']);a=ap.parse_args()
 try:proof() if a.stage=='proof' else run()
 except BaseException as ex:
  j.write(j.R/f'FAILED_{a.stage}_{int(time.time())}.json',{'error':repr(ex),'timestamp':time.time()});raise
