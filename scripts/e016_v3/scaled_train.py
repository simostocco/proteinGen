"""Authorized scratch scaled supervised controller. 10k maximum; milestone-only immutable retention."""
import argparse,copy,json,math,os,random,resource,signal,threading,time,hashlib
from pathlib import Path
import numpy as np
import torch
import scaled_bc as core
from scaled_bc import R,model,make,loss_call,autocast,guard,write,gpu,MILESTONES
from protein_joint_sac.data.scaled_supervised import Data,Sampler,P,PILOT
from protein_joint_sac.models.batched_bc import collate,encode_batch
from protein_joint_sac.contracts import AA,BuildState,BuildAction,GeneratedProtein,GEOMETRY_VERSION,digest
from protein_joint_sac.geometry.ca_builder import append,mask_for_step
from protein_joint_sac.models.hybrid_distributions import GeometryDistribution
from protein_joint_sac.evaluation.bc_metrics import trace_metrics,sequence_diversity
from protein_joint_sac.rl.bc_training import rng_state,restore_rng,code_hash
from protein_joint_sac.rl.checkpointing import save_atomic,load_checked
from protein_joint_sac.data.certified_unified import file_sha
from protein_joint_sac.chemistry.reference_adapter import ref
STOP=threading.Event();LAST=[time.monotonic()]
def stop(signum=None,frame=None):STOP.set()
def watchdog():
 while not STOP.wait(30):
  if time.monotonic()-LAST[0]>600:
   write(R/'WATCHDOG_STOP.json',{'reason':'no update progress for600seconds','timestamp':time.time()});STOP.set()
def cpu_copy(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu_copy(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [cpu_copy(v) for v in x]
 return x
def maxdiff(a,b):
 if isinstance(a,torch.Tensor):return float((a-b.detach().cpu()).abs().max())
 if isinstance(a,dict):return max([maxdiff(a[k],b[k]) for k in a] or [0.])
 if isinstance(a,(tuple,list)):return max([maxdiff(x,y) for x,y in zip(a,b)] or [0.])
 return 0. if a==b else float('inf')
def rng_digest():return digest(cpu_convert(rng_state()))
def cpu_convert(x):
 if isinstance(x,torch.Tensor):return x.cpu().tolist()
 if isinstance(x,dict):return {k:cpu_convert(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [cpu_convert(v) for v in x]
 return x

class Engine:
 def __init__(self,name,data,config):
  self.name=name;self.data=data;self.c=config;self.joint=name.startswith('J');self.model=model(name);self.opt=torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad),lr=config['learning_rate'],weight_decay=config['weight_decay'],fused=True)
  def factor(k):
   if k<config['warmup_updates']:return (k+1)/config['warmup_updates']
   return .5*(1+math.cos(math.pi*min(1.,(k-config['warmup_updates'])/(config['max_updates']-config['warmup_updates']))))
  self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.opt,factor);self.sampler=Sampler(data,name=='S2',42);self.update=0;self.validation=[];self.start=time.monotonic();self.elapsed_prior=0.;self.band_counts=np.zeros(5,dtype=np.int64);self.flagged=0
  self.flagged_ids={r['sequence_hash'] for r in json.load(open(PILOT/'priors/COORDINATE_EXTREMES.json'))['examples']}
  self.lineage={'config':config,'code_sha256':code_hash(),'entrypoint_sha256':file_sha(__file__),'core_sha256':file_sha(Path(core.__file__)),'geometry_version':GEOMETRY_VERSION,'chemistry':ref().__dict__,'validation_panel_sha256':file_sha(R/'configs/VALIDATION_PANELS.json'),'initialization':'SCRATCH_SEED42_NO_OLD_WEIGHTS','normalizers':None,'scaler':None,'torch':str(torch.__version__),'cuda':torch.version.cuda}
 def step(self):
  if self.update>=self.c['max_updates']:raise ValueError('10k update ceiling')
  guard();a=time.monotonic();examples=self.sampler.draw(self.c['effective_batch']);ordered=sorted(examples,key=lambda e:(e[2],e[0],e[1] or ''));self.model.train();self.opt.zero_grad(set_to_none=True);sums=np.zeros(5);prep_seconds=0.;forward_backward=0.
  for start in range(0,len(ordered),self.c['live_batch']):
   ex=ordered[start:start+self.c['live_batch']];q=time.monotonic();b,y,g=make(self.data,ex,self.joint);prep_seconds+=time.monotonic()-q;q=time.monotonic();v,aa,ge,parts=loss_call(self.model,b,y,g,self.c);(v.sum()/len(ordered)).backward();forward_backward+=time.monotonic()-q
   sums+=np.array([float(aa.detach().sum()),float(ge.detach().sum()),*parts.detach().sum(0).cpu().tolist()]);del b,y,g,v,aa,ge,parts
  norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.c['clip_norm'],error_if_nonfinite=True);q=time.monotonic();self.opt.step();self.scheduler.step();torch.cuda.synchronize();opt_seconds=time.monotonic()-q;self.update+=1
  if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError('nonfinite model parameter')
  for h,o,t in examples:
   N=len(self.data.seqs[h]);idx=next(i for i,(lo,hi) in enumerate([(20,64),(65,128),(129,256),(257,384),(385,500)]) if lo<=N<=hi);self.band_counts[idx]+=1;self.flagged+=int(h in self.flagged_ids)
  LAST[0]=time.monotonic()
  return {'update':self.update,'AA_NLL':sums[0]/len(examples),'geometry_component_mean_NLL':sums[1]/len(examples),'geometry_component_NLL_sums':sums[2:].tolist(),'gradient_norm_preclip':float(norm),'learning_rate':self.opt.param_groups[0]['lr'],'seconds':time.monotonic()-a,'data_seconds':prep_seconds,'forward_backward_host_seconds':forward_backward,'optimizer_sync_seconds':opt_seconds,'decisions':self.sampler.cursor,'prefix_tokens':self.sampler.tokens,'distinct_TRAIN_identities':len(self.sampler.seen),'sample_order_sha256':digest(examples),'length_exposures':self.band_counts.tolist(),'anomaly_identity_decisions':self.flagged}
 def checkpoint(self,path):
  if path.exists():raise ValueError('immutable checkpoint exists')
  guard(400*2**20)
  self.last_durable=path
  return save_atomic(path,{'lineage':self.lineage,'model':self.model.state_dict(),'optimizer':self.opt.state_dict(),'scheduler':self.scheduler.state_dict(),'rng':rng_state(),'sampler':self.sampler.state_dict(),'update':self.update,'validation':json.loads(json.dumps(self.validation,allow_nan=False)),'elapsed':self.elapsed_prior+time.monotonic()-self.start,'length_exposures':self.band_counts.tolist(),'flagged_decisions':self.flagged})
 def resume(self,path):
  s=load_checked(path);self.last_durable=path
  if s['lineage']!=self.lineage:raise ValueError('checkpoint source/config/dataset/feature provenance changed')
  self.model.load_state_dict(s['model']);self.opt.load_state_dict(s['optimizer']);self.scheduler.load_state_dict(s['scheduler']);self.sampler.load_state_dict(s['sampler']);self.update=s['update'];self.validation=s['validation'];self.elapsed_prior=s['elapsed'];self.start=time.monotonic();self.band_counts=np.array(s['length_exposures']);self.flagged=s['flagged_decisions'];restore_rng(s['rng'])
 @torch.no_grad()
 def evaluate(self,panel):
  self.model.eval();sums=np.zeros(5);counts=np.zeros(3);bands={str(i):{'n':0,'AA':0.,'geometry':0.} for i in range(5)};start=time.monotonic();valid_ids=set()
  for h,o,t in panel:
   if self.data.proposal.sequences[h]['split']!='VALIDATION':raise ValueError('ordinary evaluation is VALIDATION-only')
  # Length sorting affects execution only, not identity/prefix weights.
  ordered=sorted(panel,key=lambda e:e[2]);live=self.c['live_batch']
  for i in range(0,len(ordered),live):
   ex=ordered[i:i+live];b,y,g=make(self.data,ex,self.joint);v,aa,ge,parts=loss_call(self.model,b,y,g,self.c);sums+=np.array([float(aa.sum()),float(ge.sum()),*parts.sum(0).cpu().tolist()]);counts+=np.array([sum(t>=d for _,_,t in ex) for d in [1,2,3]]) if self.joint else 0
   av=aa.cpu().tolist();gv=ge.cpu().tolist()
   for (h,o,t),a,c in zip(ex,av,gv):
    N=len(self.data.seqs[h]);idx=next(j for j,(lo,hi) in enumerate([(20,64),(65,128),(129,256),(257,384),(385,500)]) if lo<=N<=hi);bands[str(idx)]['n']+=1;bands[str(idx)]['AA']+=a;bands[str(idx)]['geometry']+=c;valid_ids.add(h)
  for b in bands.values():
   b['AA']/=max(1,b['n']);b['geometry']/=max(1,b['n'])
  n=len(panel);return {'update':self.update,'AA_NLL':sums[0]/n,'perplexity':math.exp(sums[0]/n),'geometry_component_mean_NLL':sums[1]/n,'selection_loss':(sums[0]+sums[1])/n,'component_NLL':(sums[2:]/np.maximum(counts,1)).tolist(),'component_counts':counts.tolist(),'prefixes':n,'identities':len(valid_ids),'panel_sha256':digest(panel),'length_bands':bands,'seconds':time.monotonic()-start}
 @torch.no_grad()
 def generate(self,N,seed):
  self.model.eval();start=time.monotonic();state=BuildState.empty(N);result={'N':N,'seed':seed}
  try:
   torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
   for _ in range(N):
    b=collate([state],'cuda')
    with autocast(self.c['precision']):
     h=encode_batch(self.model,b);i=int(torch.distributions.Categorical(logits=self.model.aa_head(h)[0].float()).sample())
     if self.joint:
      chem=self.model.chemistry[i] if self.c['name']=='J2' else torch.zeros_like(self.model.chemistry[i])
      if self.c.get('geometry_projection_fp32',False):
       with torch.autocast('cuda',enabled=False):p=self.model.geom(torch.cat([h[0].float(),self.model.selected_aa.weight[i].float(),chem.float()])).float()
      else:p=self.model.geom(torch.cat([h[0],self.model.selected_aa.weight[i],chem])).float()
    if self.joint:
     g,_=GeometryDistribution(p[:3],p[3:],mask_for_step(state.t)).rsample();state=append(state,BuildAction(AA[i],g.cpu().double(),mask_for_step(state.t)))
    else:
     residues=state.residues+(AA[i],);state=BuildState(residues,torch.zeros(len(residues),3,dtype=torch.float64),N,torch.zeros(len(residues),dtype=torch.bool))
   result.update({'status':'COMPLETE','sequence':''.join(state.residues),'geometry':trace_metrics(GeneratedProtein(state)) if self.joint else None,'generated_CA_A':state.ca.tolist() if self.joint else None})
  except (ValueError,FloatingPointError) as e:result.update({'status':'GENERATION_GEOMETRY_FAILURE','error':str(e),'completed_residues':state.t})
  result['seconds']=time.monotonic()-start;return result

def run(name):
 core.setup();configs=json.load(open(R/'configs/FROZEN.json'));c=configs[name]
 receipt=json.load(open(R/'PREFLIGHT_PASS.json'))
 if receipt['status']!='PASS' or receipt['configs_sha256']!=file_sha(R/'configs/FROZEN.json'):raise ValueError('frozen preflight/config mismatch')
 if json.load(open(R/'preflight/SCALED_RESTART.json'))['status']!='PASS':raise ValueError('12M restart preflight required')
 report={};write(R/'HEARTBEAT.json',{'name':name,'status':'DATA_LOADING','timestamp':time.time()});data=Data(name.startswith('J'),report);d=R/'runs'/name;d.mkdir(exist_ok=True);write(d/'DATA_RECEIPT.json',report);eng=Engine(name,data,c);panels=json.load(open(R/'configs/VALIDATION_PANELS.json'));panel=panels['full' if name=='S2' else 'matched'];existing=sorted(d.glob('checkpoint_[0-9]*.pt'),key=lambda p:int(p.stem.split('_')[-1]));segment=f'updates_from_{eng.update:05}_{int(time.time())}.jsonl'
 if (d/'RECOVERY_START.json').exists():
  recovery=json.load(open(d/'RECOVERY_START.json'))
  if not existing or int(existing[-1].stem.split('_')[-1])<=recovery['update']:existing=[Path(recovery['path'])]
 if existing:eng.resume(existing[-1]);segment=f'updates_from_{eng.update:05}_{int(time.time())}.jsonl'
 else:
  state=rng_state();metric=eng.evaluate(panel);restore_rng(state);eng.validation.append(metric);write(d/'validation_00000.json',metric)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);LAST[0]=time.monotonic();threading.Thread(target=watchdog,daemon=True).start();torch.cuda.reset_peak_memory_stats();reference_resume=None
 with (d/segment).open('x') as log:
  while eng.update<10000 and not STOP.is_set():
   m=eng.step();log.write(json.dumps(m)+'\n')
   if eng.update%25==0:
    log.flush();write(R/'HEARTBEAT.json',{'name':name,'status':'TRAINING','timestamp':time.time(),**m,**guard()});print(json.dumps({'name':name,**m,**guard()}),flush=True)
   if eng.update==251 and not (d/'RESUME_VERIFICATION.json').exists():
    expected_model=cpu_copy(eng.model.state_dict());expected_opt=cpu_copy(eng.opt.state_dict());expected_scheduler=copy.deepcopy(eng.scheduler.state_dict());expected_sampler=copy.deepcopy(eng.sampler.state_dict());expected_rng=rng_digest();resumed=Engine(name,data,c);resumed.resume(eng.last_durable);actual=resumed.step();pe=maxdiff(expected_model,resumed.model.state_dict());oe=maxdiff(expected_opt,resumed.opt.state_dict());same=actual['sample_order_sha256']==m['sample_order_sha256'] and resumed.scheduler.state_dict()==expected_scheduler and resumed.sampler.state_dict()==expected_sampler and rng_digest()==expected_rng
    receipt={'status':'PASS' if same and pe<2e-5 and oe<1e-4 else 'FAIL','parameter_max_abs_error':pe,'optimizer_max_abs_error':oe,'scheduler_sampler_rng_and_examples_equal':same,'loss_abs_error':abs(actual['AA_NLL']-m['AA_NLL']),'scope':'12M_REAL_PDB_UPDATE251_RECONSTRUCTION','precision':c['precision']};write(d/'RESUME_VERIFICATION.json',receipt);del resumed,expected_model,expected_opt;torch.cuda.empty_cache()
    if receipt['status']!='PASS':raise RuntimeError('scaled restart verification failed')
   if eng.update in MILESTONES:
    log.flush();state=rng_state();metric=eng.evaluate(panel);eng.validation.append(metric);write(d/f'validation_{eng.update:05}.json',metric)
    if name=='S2':write(d/f'matched_validation_{eng.update:05}.json',eng.evaluate(panels['matched']))
    generated=[eng.generate(N,42+eng.update*10+j) for j,N in enumerate([20,64,128,256,384,500])];restore_rng(state);write(d/f'generation_{eng.update:05}.json',generated);p=d/f'checkpoint_{eng.update:05}.pt';h=eng.checkpoint(p);load_checked(p)
    write(d/f'milestone_{eng.update:05}.json',{'status':'VERIFIED_DURABLE','checkpoint':str(p),'checkpoint_sha256':h,'validation':metric,'elapsed_seconds':eng.elapsed_prior+time.monotonic()-eng.start,'device':gpu(),**m,**guard()});print('MILESTONE '+name+' '+str(eng.update),flush=True);LAST[0]=time.monotonic()
  if STOP.is_set() and eng.update<10000:
   p=d/f'checkpoint_{eng.update:05}.pt'
   if not p.exists():eng.checkpoint(p)
   write(d/'STOPPED.json',{'update':eng.update,'reason':'graceful signal/watchdog','checkpoint':str(p)});raise RuntimeError('graceful stop; do not advance next model')
 best=min((x for x in eng.validation if x['update'] in MILESTONES),key=lambda x:x['selection_loss'] if name.startswith('J') else x['AA_NLL']);checkpoint=d/f"checkpoint_{best['update']:05}.pt"
 if (d/'RECOVERY_START.json').exists():
  recovery=json.load(open(d/'RECOVERY_START.json'))
  if recovery['update']==best['update']:checkpoint=Path(recovery['path'])
 write(d/'COMPLETED.json',{'status':'COMPLETED_10000_UPDATES','name':name,'updates':eng.update,'decisions':eng.sampler.cursor,'distinct_TRAIN_identities':len(eng.sampler.seen),'prefix_tokens':eng.sampler.tokens,'length_exposures':eng.band_counts.tolist(),'anomaly_identity_decisions':eng.flagged,'best_validation':best,'final_validation':eng.validation[-1],'selected_checkpoint':str(checkpoint),'selected_sha256':file_sha(checkpoint),'elapsed_seconds':eng.elapsed_prior+time.monotonic()-eng.start,'resources':guard(),'config':c});print('TRAINING_COMPLETE '+name,flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--name',required=True,choices=['S1','J1','J2','S2']);q=a.parse_args()
 try:run(q.name)
 except BaseException as e:
  write(R/'BLOCKED.json',{'model':q.name,'exception':type(e).__name__,'message':str(e),'timestamp':time.time(),'resources':guard(0)});raise
