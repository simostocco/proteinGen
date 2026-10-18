"""Single sequence-first J1 continuation. Real-data engineering preflight and staged training."""
import collections,copy,gc,json,math,os,random,signal,subprocess,threading,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
import j1_global as j
from protein_joint_sac.data.partial_continuation import PartialData
from protein_joint_sac.rl.sequence_first import *
from protein_joint_sac.models.batched_bc import losses,encode_batch
from protein_joint_sac.rl.bc_training import rng_state,restore_rng,code_hash
from protein_joint_sac.rl.checkpointing import save_atomic,load_checked
from protein_joint_sac.data.certified_unified import file_sha
from scaled_train import Engine as Generator,cpu_copy,maxdiff,rng_digest
R=Path('/mnt/d/Simone/proteinGen_data/e016_j1_sequence_first_20261018_v1')
PREVIOUS=Path('/mnt/d/Simone/proteinGen_data/e016_j1_global_pretraining_20261017_v1')
j.R=R;j.core.R=R
write=j.write
STOP=threading.Event();LAST=[time.monotonic()]
def status(phase,**kw):write(R/'HEARTBEAT.json',{'phase':phase,'timestamp':time.time(),**kw})
def guard():
 j.guard()
 if STOP.is_set():raise RuntimeError("graceful stop requested")
def loss_call(model,b,y,g):
 with torch.autocast('cuda',dtype=torch.bfloat16):
  return losses(model,b,y,g,geometry_fp32=True)
class Sampler:
 def __init__(self,data):
  self.data=data;self.r=random.Random(180042);self.cover=Coverage(42);self.cursor=0;self.counts=collections.Counter();self.orders={};self.cursors={}
  for name,ids in [('complete',data.base.jointids['TRAIN']),('partial',data.ids['TRAIN']),('geometry',data.base.jointids['TRAIN'])]:
   self.orders[name]=list(ids);self.r.shuffle(self.orders[name]);self.cursors[name]=0
 def pick(self,kind,aa=True):
  k=self.cursors[kind]
  if k==len(self.orders[kind]):self.r.shuffle(self.orders[kind]);k=0
  h=self.orders[kind][k];self.cursors[kind]=k+1
  owners=self.data.owners['TRAIN'][h] if kind=='partial' else self.data.base.owners[h]
  o=self.r.choice(owners);N=len(self.data.seqs[h]);t=self.cover.position(h,N,o,0 if aa else 3,"AA" if aa else "GEOMETRY")
  if aa:
   self.counts[kind+'_AA']+=1
   if kind=='partial' and not self.data.cache[o]['seen'][t]:self.counts['partial_AA_missing_CA']+=1
  else:self.counts['complete_geometry_decisions']+=1
  return h,o,t
 def draw(self):
  partial=[38,39,38,39,38][self.cursor%5]
  aa=[self.pick('partial') for _ in range(partial)]+[self.pick('complete') for _ in range(64-partial)]
  ge=[self.pick('geometry',False) for _ in range(32)];self.cursor+=1
  return sorted(aa,key=lambda e:(e[2],e[0],e[1])),sorted(ge,key=lambda e:(e[2],e[0],e[1]))
 def state_dict(self):
  return {'rng':self.r.getstate(),'coverage':self.cover.state_dict(),'cursor':self.cursor,'counts':dict(self.counts),'orders':copy.deepcopy(self.orders),'cursors':self.cursors.copy()}
 def load_state_dict(self,x):
  self.r.setstate(x['rng']);self.cover.load_state_dict(x['coverage']);self.cursor=x['cursor'];self.counts=collections.Counter(x['counts']);self.orders=copy.deepcopy(x['orders']);self.cursors=x['cursors'].copy()
 def report(self):return {'updates':self.cursor,'exposures':dict(self.counts),**self.cover.report()}
class Engine:
 def __init__(self,data,c):
  self.data=data;self.c=c;self.name='J1';self.joint=True;self.model,parent=j.parent_model();set_dropout(self.model,.15)
  self.opt=torch.optim.AdamW(self.model.parameters(),lr=c['learning_rate'],weight_decay=.01,fused=True);self.opt.load_state_dict(parent['optimizer'])
  if {float(s['step']) for s in self.opt.state.values()}!={30000.}:raise ValueError('parent optimizer mismatch')
  for group in self.opt.param_groups:group['lr']=c['learning_rate'];group['initial_lr']=c['learning_rate']
  def factor(k):
   if k<250:return (k+1)/250
   return .1+.9*.5*(1+math.cos(math.pi*min(1,(k-250)/(c['max_updates']-250))))
  self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.opt,factor);restore_rng(parent['rng']);del parent
  self.sampler=Sampler(data);self.rollrng=random.Random(180720);self.update=0;self.residues=0;self.nonlocal_targets=0;self.history=[];self.best=0;self.elapsed=0.;self.start=time.monotonic()
  self.lineage={'version':VERSION,'config':c,'package_sha256':code_hash(),'entrypoint_sha256':file_sha(__file__),'parent_sha256':j.PARENT_SHA,'split_sha256':file_sha(j.core.P/'PROPOSAL_MANIFEST.json'),'view_sha256':file_sha(j.OLD/'view/VIEW_FROZEN.json'),'panels_sha256':file_sha(R/'configs/EVALUATION_PANELS.json'),'geometry_version':'ca_internal_signed_v1','chemistry':'DISABLED','TEST':'NOT_USED'}
 def step(self,force_stage=None,force_horizon=None):
  guard();start=time.monotonic();self.model.train();self.opt.zero_grad(set_to_none=True);aaex,geoex=self.sampler.draw();AA=GEO=0.;component=np.zeros(3);gnonlocal=0.;counts=[];stage=force_stage or (1 if self.update<2500 else 2 if self.update<5000 else 3)
  for off in range(0,64,16):
   b,y,g=self.data.batch(aaex[off:off+16]);b=augment(b,self.c['mask_probability'])
   # AA-only exposure stream; no duplicated AA loss on geometry-only examples.
   with torch.autocast('cuda',dtype=torch.bfloat16):h=encode_batch(self.model,b);aa=F.cross_entropy(self.model.aa_head(h).float(),y,reduction='sum')/64
   aa.backward();AA+=float(aa.detach());del b,y,g,h,aa
  for off in range(0,32,16):
   ex=geoex[off:off+16];b,y,g=self.data.batch(ex);b=augment(b,self.c['mask_probability'])
   aa,geo,parts=loss_call(self.model,b,y,g);obj=self.c['lambda_local']*geo.sum()/32;obj.backward()
   GEO+=float(geo.detach().sum())/32;component+=parts.detach().sum(0).cpu().numpy();del b,y,g,obj,geo,parts,aa
  extra={};auxstart=time.monotonic()
  if stage>=2 and ((self.update+1)%self.c['nonlocal_period']==0 or force_stage):
   ex=geoex[-4:];b,y,g=self.data.batch(ex);b=augment(b,self.c['mask_probability'])
   targets=torch.tensor(np.asarray([self.data.base.ca[o][t] for h,o,t in ex]),device='cuda')
   nl,counts=direct_nonlocal(self.model,b,y,targets,K=4,seed=180042)
   ramp=min(1.,max(0.,(self.update+1-2500)/500)) if not force_stage else 1.
   (self.c['lambda_nonlocal']*ramp*nl).backward();gnonlocal=float(nl.detach());self.nonlocal_targets+=len(counts)
   extra.update(nonlocal_energy_score=gnonlocal,nonlocal_anchor_counts=counts,nonlocal_ramp=ramp);del nl,b,y,g,targets
  if stage>=3 and ((self.update+1)%self.c['rollout_period']==0 or force_stage):
   H=force_horizon or self.c['horizons'][(self.update//self.c['rollout_period'])%len(self.c['horizons'])]
   candidates=[h for h in self.data.base.jointids['TRAIN'] if len(self.data.seqs[h])>=H+3]
   h=self.rollrng.choice(candidates);o=self.rollrng.choice(self.data.base.owners[h]);N=len(self.data.seqs[h]);t=self.rollrng.randrange(3,N-H+1)
   ref=torch.tensor(self.data.base.ca[o][:t+H],device='cuda');ids=torch.tensor([j.AA.index(a) for a in self.data.seqs[h][:t+H]],device='cuda')
   r=generated_bundle(self.model,ref[:t],ids,N,H,ref);ramp=min(1.,max(0.,(self.update+1-5000)/1000)) if not force_stage else 1.
   aux=ramp*(self.c['lambda_Rg']*r['Rg']+self.c['lambda_clash']*r['clash']+self.c['lambda_pair']*r['pair']);aux.backward();self.residues+=r['residues']
   extra.update(rollout_horizon=H,rollout_paths=2,rollout_rg=float(r['Rg'].detach()),rollout_clash=float(r['clash'].detach()),rollout_pair_energy=float(r['pair'].detach()),rollout_ramp=ramp,rollout_start=t,rollout_identity=h);del r,aux,ref,ids
  pre=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True);lr=self.opt.param_groups[0]['lr'];self.opt.step();self.scheduler.step();self.update+=1;torch.cuda.synchronize()
  if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError('nonfinite parameters')
  LAST[0]=time.monotonic()
  return {'additional_update':self.update,'total_optimizer_updates':30000+self.update,'stage':stage,'AA_NLL_stochastic_batch':AA,'local_geometry_NLL':GEO,'geometry_component_mean':(component/32).tolist(),'gradient_norm_preclip':float(pre),'learning_rate':lr,'seconds':time.monotonic()-start,'auxiliary_seconds':time.monotonic()-auxstart,'ordinary_AA_decisions':self.update*64,'geometry_decisions':self.update*32,'nonlocal_targets':self.nonlocal_targets,'rollout_residues':self.residues,'sample_order_sha256':j.digest([aaex,geoex]),**extra}
 def checkpoint(self,path):
  if path.exists():raise ValueError('immutable checkpoint exists')
  j.guard()
  return save_atomic(path,{'lineage':self.lineage,'model':self.model.state_dict(),'optimizer':self.opt.state_dict(),'scheduler':self.scheduler.state_dict(),'sampler':self.sampler.state_dict(),'rng':rng_state(),'rollout_rng':self.rollrng.getstate(),'additional_update':self.update,'residues':self.residues,'nonlocal_targets':self.nonlocal_targets,'history':self.history,'best':self.best,'elapsed':self.elapsed+time.monotonic()-self.start})
 def resume(self,path):
  s=load_checked(path)
  if s['lineage']!=self.lineage:raise ValueError('lineage integrity mismatch')
  self.model.load_state_dict(s['model'],strict=True);self.opt.load_state_dict(s['optimizer']);self.scheduler.load_state_dict(s['scheduler']);self.sampler.load_state_dict(s['sampler']);self.rollrng.setstate(s['rollout_rng']);self.update=s['additional_update'];self.residues=s['residues'];self.nonlocal_targets=s['nonlocal_targets'];self.history=s['history'];self.best=s['best'];self.elapsed=s['elapsed'];self.start=time.monotonic();restore_rng(s['rng'])
 @torch.no_grad()
 def generate(self,N,seed):return Generator.generate(self,N,seed)
def checkpoint_prune(engine):
 d=R/'runs/J1';keep={engine.update,engine.best}
 # Preserve explicit stage boundaries as compact model-only states, full latest/best restart states.
 for p in d.glob('checkpoint_*.pt'):
  u=int(p.stem.split('_')[1])
  if u in keep:continue
  if p.is_symlink() or p.stat().st_nlink!=1 or not p.resolve().is_relative_to(d):raise ValueError('unsafe new-lineage retention')
  receipt={'update':u,'path':str(p),'sha256':file_sha(p),'reason':'bounded new-lineage best/latest retention; milestones retain model-only states and receipts'}
  with (d/'RETENTION.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
  p.unlink();p.with_name(p.name+'.sha256').unlink()
def generation(engine,u,n=2):
 from partial_finalize import metrics
 j.core.R=R
 rows=[]
 for N in [64,128,256,384,500]:
  for k in range(n):
   v=engine.generate(N,2026101700+100*N+k);v.update(sample_index=k,checkpoint_update=u)
   if v['status']=='COMPLETE':v['metrics']=metrics(v['generated_CA_A'])
   rows.append(v);LAST[0]=time.monotonic()
 write(R/'runs/J1'/f'generation_{u:05}.json',rows)
 return rows
def assess(engine,data,panels):
 state=rng_state();train=j.evaluate(engine.model,data.base,panels['TRAIN'],'TRAIN');LAST[0]=time.monotonic();val=j.evaluate(engine.model,data.base,panels['VALIDATION'],'VALIDATION');LAST[0]=time.monotonic()
 from partial_train import Engine as PE
 partial=PE.evaluate(engine,json.load(open(R/'configs/PARTIAL_VALIDATION_PANEL.json')),'PARTIAL')
 j.core.R=R;LAST[0]=time.monotonic();gen=generation(engine,engine.update);restore_rng(state)
 return {'TRAIN':train,'VALIDATION':val,'PARTIAL_VALIDATION':partial,'generation':gen}
def train():
 j.core.setup();c=json.load(open(R/'configs/FROZEN.json'));gate=json.load(open(R/'PREFLIGHT_PASS.json'))
 if gate['package_sha256']!=code_hash() or gate['entrypoint_sha256']!=file_sha(__file__) or gate['config_sha256']!=file_sha(R/'configs/FROZEN.json'):raise ValueError('preflight source/config mismatch')
 status('DATA_LOADING');data=PartialData();e=Engine(data,c);d=R/'runs/J1';panels=json.load(open(R/'configs/EVALUATION_PANELS.json'))
 existing=sorted(d.glob('checkpoint_*.pt'))
 if existing:
  rejected=[]
  for p in reversed(existing):
   try:e.resume(p);break
   except Exception as ex:rejected.append({'path':str(p),'error':repr(ex)})
  else:raise RuntimeError('no verified recovery state')
  write(R/f'RECOVERY_{int(time.time())}.json',{'checkpoint':str(p),'rejected':rejected,'resumed_update':e.update})
 baseline=json.load(open(R/'SEQUENCE_BASELINES.json'))['J1'];status('TRAINING',update=e.update)
 if not e.history:e.history=[{'update':0,'TRAIN':baseline['TRAIN'],'VALIDATION':baseline['VALIDATION']}]
 def stop(*a):STOP.set()
 signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 def watch():
  while not STOP.wait(30):
   if time.monotonic()-LAST[0]>900:STOP.set()
 LAST[0]=time.monotonic();threading.Thread(target=watch,daemon=True).start();torch.cuda.reset_peak_memory_stats();reason=None
 with (d/f'updates_from_{e.update:05}_{int(time.time())}.jsonl').open('x') as log:
  while e.update<c['max_updates'] and not STOP.is_set():
   m=e.step();log.write(json.dumps(m)+'\n')
   if e.update%25==0:log.flush();status('TRAINING',**m,**j.guard());print(json.dumps(m),flush=True)
   if e.update in c['milestones']:
    log.flush();result=assess(e,data,panels);write(d/f'evaluation_{e.update:05}.json',result)
    entry={'update':e.update,'TRAIN':result['TRAIN'],'VALIDATION':result['VALIDATION'],'PARTIAL_VALIDATION':result['PARTIAL_VALIDATION']}
    e.history.append(entry);val=result['VALIDATION'];delta=val['AA_NLL']-baseline['VALIDATION']['AA_NLL'];local=val['geometry_NLL']-baseline['VALIDATION']['geometry_NLL']
    finitegen=len([r for r in result['generation'] if r['status']=='COMPLETE'])==10
    clashes=np.mean([r['metrics']['clashes_per100_residues'] for r in result['generation'] if 'metrics' in r])
    eligible=delta<0 and local<=c['guardrails']['local_delta'] and finitegen and clashes<c['guardrails']['severe_clash_mean']
    entry['eligible']=eligible
    best=next(v for v in e.history if v['update']==e.best)
    if eligible and val['AA_NLL']<best['VALIDATION']['AA_NLL']:e.best=e.update
    recent=e.history[-2:]
    if len(e.history)>=3 and all(x['VALIDATION']['AA_NLL']>baseline['VALIDATION']['AA_NLL']+c['guardrails']['AA_delta'] or x['VALIDATION']['geometry_NLL']>baseline['VALIDATION']['geometry_NLL']+c['guardrails']['local_delta'] for x in recent):reason='two consecutive predefined AA/local guard violations'
    if not finitegen or clashes>c['guardrails']['severe_clash_mean']:reason='generation finite/clash safeguard failed'
    if e.update==20000:
     earlier=next(x for x in e.history if x['update']==10000)
     progress=val['AA_NLL']<earlier['VALIDATION']['AA_NLL']-.002
     if not (eligible and progress):reason='no qualifying validation progress for optional extension beyond20000'
    write(R/'TRAIN_VALIDATION_PERPLEXITY_HISTORY.json',e.history)
    # Compact source-bound stage snapshots; optimizer recovery retained via best/latest.
    if e.update in [2500,5000,20000]:save_atomic(d/f'stage_model_{e.update:05}.pt',{'model':e.model.state_dict(),'lineage':e.lineage,'additional_update':e.update})
    with (R/'EXPERIMENT_DECISIONS.md').open('a') as f:f.write(f"\nMilestone+{e.update}: AA NLL delta{delta:.6f}; local delta{local:.6f}; eligible{eligible}; stop{reason}. Stage interventions are sequential, not causal ablations.\n")
   if e.update%250==0 or reason:
    cp=d/f'checkpoint_{e.update:05}.pt';e.checkpoint(cp);load_checked(cp);checkpoint_prune(e)
    write(R/'POSITION_COVERAGE_REPORT.json',e.sampler.report());write(R/'SAMPLING_MIXTURE_AND_EXPOSURES.json',{'AA':dict(e.sampler.counts),'rollout_residues':e.residues,'nonlocal_targets':e.nonlocal_targets})
   if reason:break
  if STOP.is_set():
   cp=d/f'checkpoint_{e.update:05}.pt'
   if not cp.exists():e.checkpoint(cp)
   write(d/'STOPPED.json',{'update':e.update,'checkpoint':str(cp),'reason':'signal/watchdog'})
   return
 write(d/'COMPLETED.json',{'status':'COMPLETED_OR_PREDECLARED_STOP','additional_updates':e.update,'total_optimizer_updates':30000+e.update,'best_update':e.best,'stop_reason':reason or 'maximum authorized updates','latest_checkpoint':str(d/f'checkpoint_{e.update:05}.pt'),'latest_sha256':file_sha(d/f'checkpoint_{e.update:05}.pt'),'elapsed_seconds':e.elapsed+time.monotonic()-e.start,'coverage':e.sampler.report(),'rollout_residues':e.residues,'nonlocal_targets':e.nonlocal_targets,'resources':j.guard(),'lineage':e.lineage})
 status('MAIN_COMPLETE',update=e.update);print('SEQUENCE_FIRST_MAIN_COMPLETE',flush=True)
if __name__=='__main__':train()
