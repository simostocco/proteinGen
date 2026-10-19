"""One bounded AA-head-only J1 lineage. No shared/geometry optimizer parameters."""
import collections,copy,gc,hashlib,json,math,os,signal,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import j1_full_sequence as m
R=m.R;D=R.parent;read=m.read;write=m.write;sha=m.sha;s=m.s;c=m.c
FROOT=D/'e016_j1_full_sequence_20261018_v1';O=D/'e016_j1_clip_precision_pilot_20261018_v1'
def tensor_sha(x):
 x=x.detach().cpu().contiguous();return hashlib.sha256(str(x.dtype).encode()+str(list(x.shape)).encode()+x.numpy().tobytes()).hexdigest()
def frozen_snapshot(model):return {n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith('aa_head.')}
def norm(ps):return math.sqrt(sum(float(p.grad.double().square().sum()) for p in ps if p.grad is not None))
class Sampler(m.ExpandedSampler):
 def draw(self):
  aa=[]
  for k,n in zip(['complete','partial_only','unused'],self.QUOTAS[self.cursor%5]):aa.extend(self.pick(k) for _ in range(n))
  self.cursor+=1;return sorted(aa,key=lambda e:(e[2],e[0],e[1] or ''))
class Engine:
 def __init__(self,data):
  torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
  assert sha(m.SOURCE)==m.SOURCE_SHA
  st=s.load_checked(m.SOURCE);self.data=data;self.model=s.j.core.model('J1');self.model.load_state_dict(st['model'],strict=True);s.set_dropout(self.model,.15)
  names=[];ptr={}
  for n,p in self.model.named_parameters(remove_duplicate=False):
   assert p.data_ptr() not in ptr,('aliased parameter',n,ptr.get(p.data_ptr()));ptr[p.data_ptr()]=n;p.requires_grad_(n.startswith('aa_head.'))
   if p.requires_grad:names.append(n)
  assert names==['aa_head.weight','aa_head.bias'];assert self.model.aa_head.weight.shape==(20,352)
  self.params=[p for p in self.model.parameters() if p.requires_grad];self.frozen=frozen_snapshot(self.model);self.original_head={n:p.detach().cpu().clone() for n,p in self.model.named_parameters() if p.requires_grad}
  assert sum(p.numel() for p in self.params)==7060 and sum(p.numel() for p in self.model.parameters())==12351746
  self.c={'version':'j1_AA_head_only_v1','source_checkpoint':str(m.SOURCE),'source_sha256':m.SOURCE_SHA,'historical_updates_at_start':35500,'head_parameters':names,'trainable_parameters':7060,'model_parameters':12351746,'optimizer':'fresh head-only AdamW','learning_rate':5e-5,'terminal_learning_rate':5e-6,'warmup':50,'decay_end':1000,'weight_decay':.01,'head_clip_L2':read(m.T/'configs/FROZEN.json')['block_limits']['A'],'dropout':.15,'input_augmentation':.075,'AA_quota_cycle':Sampler.QUOTAS,'geometry_training_decisions':0,'loss':'AA_CE_ONLY','initial_updates':500,'max_updates':1000,'extension':{'free_VAL_NLL_min_gain':.01,'paired95_upper_max':0,'native_VAL_max_regression':.01,'free_gap_max_increase':.05,'recent250to500_min_gain':.001},'generation':'ca_fp64_interior_v1;32attempts per length; original safeguards unchanged','panels':{n:sha(R/'configs'/n) for n in ['EVALUATION_PANELS.json','GEOMETRY_FREE_PANELS.json','GENERATED_PREFIX_PANEL.json']},'sampler':'fresh deterministic category orders, inherited parent position coverage; no geometry draws','TEST':'NOT_USED'}
  self.opt=torch.optim.AdamW(self.params,lr=5e-5,weight_decay=.01,fused=True)
  def factor(k):return (k+1)/50 if k<50 else .1+.9*.5*(1+math.cos(math.pi*min(1.,(k-50)/950)))
  self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.opt,factor);self.sampler=Sampler(data,st['sampler']);self.update=0;self.history=[];self.start=time.monotonic();self.elapsed=0.;self.stop=False
  self.lineage={'config':self.c,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip(),'code_sha256':s.code_hash(),'entrypoint_sha256':sha(__file__),'frozen_manifest':{n:tensor_sha(v) for n,v in self.frozen.items()},'source_optimizer_policy':'not restored; new restricted optimizer; original preserved'}
  inventory=[{'name':n,'shape':list(p.shape),'parameters':p.numel(),'trainable':p.requires_grad,'sha256_before':tensor_sha(p)} for n,p in self.model.named_parameters()]
  write(R/'AA_HEAD_PARAMETER_INVENTORY.json',{'model_parameters':12351746,'trained_parameters':7060,'frozen_parameters':12344686,'parameters':inventory,'buffers':[{'name':n,'shape':list(v.shape),'sha256':tensor_sha(v)} for n,v in self.model.named_buffers()],'no_tied_parameter_storage':True,'independent_geometry':'geom(cat(hidden, selected_aa[true_target], chemistry)); no AA-head logits/weights consumed'})
  write(R/'HEAD_ONLY_OPTIMIZER_CONFIG.json',self.c);s.restore_rng(st['rng']);del st
 def verify(self):
  for n,v in self.model.state_dict().items():
   if n in self.frozen:assert torch.equal(v.detach().cpu(),self.frozen[n]),'frozen tensor changed: '+n
  assert all(p.grad is None for n,p in self.model.named_parameters() if not p.requires_grad)
  assert {id(p) for g in self.opt.param_groups for p in g['params']}=={id(p) for p in self.params}
  return {'all_non_AA_parameters_bitwise_equal':True,'all_buffers_bitwise_equal':True,'frozen_max_abs_difference':0.,'frozen_gradient_entries':0,'head_changed':any(not torch.equal(p.detach().cpu(),self.original_head[n]) for n,p in self.model.named_parameters() if p.requires_grad)}
 def step(self):
  m.guard();start=time.monotonic();self.model.train();self.opt.zero_grad(set_to_none=True);examples=self.sampler.draw();loss=0.;lr=self.opt.param_groups[0]['lr']
  for off in range(0,64,16):
   b,y,_=self.data.batch(examples[off:off+16]);b=m.augment_available(b,.075)
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):h=m.encode_batch(self.model,b)
   with torch.autocast('cuda',dtype=torch.bfloat16):ce=F.cross_entropy(self.model.aa_head(h).float(),y,reduction='sum')/64
   ce.backward();loss+=float(ce.detach())
  raw=norm(self.params)
  if not math.isfinite(raw) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()):raise FloatingPointError('nonfinite head gradient')
  clipped=float(torch.nn.utils.clip_grad_norm_(self.params,self.c['head_clip_L2'],error_if_nonfinite=True));post=norm(self.params);before=[p.detach().clone() for p in self.params];self.opt.step();self.scheduler.step();self.update+=1
  if not c.finite_state(self.opt) or any(not bool(torch.isfinite(p).all()) for p in self.params):raise FloatingPointError('nonfinite optimizer')
  delta=math.sqrt(sum(float((p.detach()-b).double().square().sum()) for p,b in zip(self.params,before)));torch.cuda.synchronize();c.LAST[0]=time.monotonic()
  return {'new_update':self.update,'AA_NLL_stochastic':loss,'AA_head_gradient_L2_before':raw,'AA_head_gradient_L2_after':post,'clip_scale':min(1.,self.c['head_clip_L2']/max(raw,1e-30)),'head_parameter_update_L2':delta,'learning_rate':lr,'seconds':time.monotonic()-start,'sample_order_sha256':s.j.digest(examples),'frozen_shared_gradient_L2':0.,'geometry_training_targets':0}
 def checkpoint(self):
  self.verify();path=R/'runs/J1'/f'checkpoint_{self.update:05d}.pt'
  payload={'lineage':self.lineage,'model':self.model.state_dict(),'AA_head':self.model.aa_head.state_dict(),'optimizer':self.opt.state_dict(),'scheduler':self.scheduler.state_dict(),'sampler':self.sampler.state_dict(),'rng':s.rng_state(),'new_update':self.update,'historical_optimizer_updates_at_start':35500,'total_model_optimizer_updates':35500+self.update,'history':self.history,'elapsed':self.elapsed+time.monotonic()-self.start,'frozen_verification':self.verify()}
  if path.exists():
   q=s.load_checked(path);assert c.same(q['model'],payload['model']) and q['new_update']==self.update
  else:s.save_atomic(path,payload)
  q=s.load_checked(path);assert c.same(q['model'],self.model.state_dict()) and c.same(q['optimizer'],self.opt.state_dict())
  return path
 def resume(self,path):
  st=s.load_checked(path);assert st['lineage']==self.lineage;u=st['new_update'];assert st['scheduler']['last_epoch']==u and st['sampler']['cursor']==u
  assert {int(v['step']) for v in st['optimizer']['state'].values()}=={u}
  self.model.load_state_dict(st['model']);self.opt.load_state_dict(st['optimizer']);self.scheduler.load_state_dict(st['scheduler']);self.sampler.load_state_dict(st['sampler']);self.update=u;self.history=st['history'];self.elapsed=st['elapsed'];self.start=time.monotonic();s.restore_rng(st['rng']);self.verify()

def preflight(e):
 rng=s.rng_state();sam=copy.deepcopy(e.sampler.state_dict());e.model.eval();ex=[]
 for k in ['complete','partial_only','unused']:
  for h in e.data.populations['TRAIN'][k][:8]:
   o=None if k=='unused' else (e.data.owners['TRAIN'][h][0] if k=='partial_only' else e.data.base.owners[h][0]);ex.append((h,o,min(32,len(e.data.seqs[h])-1)))
 b,y,g=e.data.batch(ex);from protein_joint_sac.rl.sequence_first import geometry_parameters
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):h=m.encode_batch(e.model,b);log=e.model.aa_head(h);gp=geometry_parameters(e.model,h,y)
 e.proof_batch=(b,y,g);e.proof_geometry=gp.detach().cpu().clone()
 only=[x for x in ex if x[1] is None];a,ay,_=e.data.aa_only_batch(only);a['x'].fill_(float('nan'))
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):l1=e.model.aa_head(m.encode_batch(e.model,a))
 a['x'].fill_(1e12)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):l2=e.model.aa_head(m.encode_batch(e.model,a))
 assert torch.equal(l1,l2) and torch.isfinite(l1).all()
 # Canonical suffix and target remain outside encoder conditioning.
 hh,oo,tt=only[0];original=e.data.seqs[hh];e.data.seqs[hh]=original[:tt]+''.join(m.AA[(m.AA.index(ch)+1)%20] for ch in original[tt:]);a2,_,_=e.data.aa_only_batch(only);e.data.seqs[hh]=original
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):l3=e.model.aa_head(m.encode_batch(e.model,a2))
 assert torch.equal(l2,l3)
 e.model.train();e.opt.zero_grad(set_to_none=True)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):h=m.encode_batch(e.model,b)
 with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(e.model.aa_head(h).float(),y)
 loss.backward();raw=norm(e.params);assert raw>0 and math.isfinite(raw);assert e.verify()['frozen_gradient_entries']==0;e.opt.zero_grad(set_to_none=True)
 # Quota and no-geometry-stream resume proof without an optimizer step.
 x=[e.sampler.draw() for _ in range(2)];st=copy.deepcopy(e.sampler.state_dict());want=[e.sampler.draw() for _ in range(3)];assert [e.sampler.counts[k] for k in ['complete','partial_only','unused']]==[128,96,96];assert e.sampler.counts['geometry']==0;e.sampler.load_state_dict(st);assert [e.sampler.draw() for _ in range(3)]==want;e.sampler.load_state_dict(sam);s.restore_rng(rng)
 write(R/'HEAD_PREFLIGHT_TESTS.json',{'status':'PASS','train_gradient_L2':raw,'clip_limit':e.c['head_clip_L2'],'no_frozen_gradients':True,'all_missing_finite':True,'masked_NaN_or_large_placeholder_invariant':True,'future_AA_suffix_invariant':True,'mixed_complete_partial_unused':True,'quota_and_sampler_resume_exact':True,'geometry_stream_decisions':0,'optimizer_steps_before_baseline':0,'legacy_mask_tests':str(FROOT/'AA_ONLY_INPUT_MASK_TESTS.json')})
 write(R/'DECISIONS_FROZEN_BEFORE_BASELINE.json',{'config_sha256':sha(R/'HEAD_ONLY_OPTIMIZER_CONFIG.json'),'source_commit':e.lineage['source_commit'],'panels':e.c['panels'],'TRAIN_preflight_head_gradient':raw,'schedule_retained':True})

def geom_proof(e):
 from protein_joint_sac.rl.sequence_first import geometry_parameters
 e.model.eval();b,y,g=e.proof_batch
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):h=m.encode_batch(e.model,b);now=geometry_parameters(e.model,h,y)
 assert torch.equal(now.cpu(),e.proof_geometry)
 return {'conditional_geometry_parameters_bitwise_equal':True,'maximum_absolute_prediction_difference':0.,**e.verify()}

def assess(e):
 z=m.assess(e)
 for role in ['FREE_TRAIN','FREE_VALIDATION']:
  for v in z[role]['categories'].values():v['decisions']=4*v['identities']
  for v in z[role]['length_bands'].values():v['identities']=v['decisions'];v['decisions']*=4
 z['frozen_geometry_verification']=geom_proof(e)
 if e.history:
  for role in ['NATIVE_TRAIN','NATIVE_VALIDATION']:
   for key in ['geometry_NLL','component_NLL','component_counts']:assert z[role][key]==e.history[0][role][key],('geometry changed',role,key)
 write(R/'runs/J1'/f'evaluation_{e.update:05d}.json',z);e.history.append(z);write(R/'TRAIN_VALIDATION_AA_METRICS.json',e.history);return z

def generation_decision(e,cp):
 rows=m.generation(e,cp,32);native=read(s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json')['native'];fallback=read(O/'generation/fallback_ca_fp64_interior_v1.json');parent=read(O/'generation/final00500_ca_fp64_interior_v1.json');ss=m.old.metric_summary(rows);bs=m.old.metric_summary(parent);fails=m.old.check_candidate(ss,m.old.metric_summary(fallback),native)
 for n,z in ss.items():
  if z['long_contact_fraction']['mean']<.8*bs[n]['long_contact_fraction']['mean']:fails.append(n+' contacts below .8 accepted parent')
 out={'update':e.update,'samples_per_length':32,'summary':ss,'failures':fails,'passes':not fails,'parent_summary':bs};write(R/'generation'/f'decision_{e.update:05d}.json',out);return out

def decision(row,base,previous=None,generation=None):
 v=row['FREE_VALIDATION'];b=base['FREE_VALIDATION'];keys=sorted(v['per_identity']);ci=m.old.paired_ci([v['per_identity'][h]['AA_NLL']-b['per_identity'][h]['AA_NLL'] for h in keys]);fail=[]
 if v['AA_NLL']>b['AA_NLL']-.01:fail.append('free VAL NLL gain < .01')
 if ci['bootstrap_95pct'][1]>=0:fail.append('paired identity CI does not support gain')
 if row['NATIVE_VALIDATION']['AA_NLL']>base['NATIVE_VALIDATION']['AA_NLL']+.01:fail.append('native VAL NLL regression > .01')
 if row['free_gap']>base['free_gap']+.05:fail.append('free gap increase > .05')
 if row['NATIVE_VALIDATION']['geometry_NLL']!=base['NATIVE_VALIDATION']['geometry_NLL']:fail.append('frozen geometry changed')
 if previous and v['AA_NLL']>previous['FREE_VALIDATION']['AA_NLL']-.001:fail.append('recent held-out gain < .001')
 if generation:fail.extend(generation['failures'])
 else:fail.append('final generation assessment pending')
 return {'passes':not fail,'failures':fail,'paired_free_VAL_NLL_change':ci}

def restart(e):
 a=[e.step() for _ in range(2)];cp=e.checkpoint();b=[e.step() for _ in range(2)];ref={k:copy.deepcopy(v) for k,v in {'model':s.cpu_copy(e.model.state_dict()),'optimizer':s.cpu_copy(e.opt.state_dict()),'scheduler':e.scheduler.state_dict(),'sampler':e.sampler.state_dict(),'rng':s.rng_state()}.items()};e.resume(cp);actual=[e.step() for _ in range(2)];cur={'model':e.model.state_dict(),'optimizer':e.opt.state_dict(),'scheduler':e.scheduler.state_dict(),'sampler':e.sampler.state_dict(),'rng':s.rng_state()};checks={k:c.same(v,cur[k]) for k,v in ref.items()};checks['sample_order']=[x['sample_order_sha256'] for x in b]==[x['sample_order_sha256'] for x in actual];assert all(checks.values());write(R/'HEAD_GPU_RESTART_TEST.json',{'status':'PASS','checks':checks,'unique_main_updates':4,'reexecuted_proof_updates':2,'frozen_proof':geom_proof(e)});e.checkpoint();return a+actual

def run():
 assert not (R/'runs/J1/COMPLETED.json').exists(),'completed lineage: no automatic restart'
 s.j.core.setup();m.guard();m.status('HEAD_DATA_LOADING');data=m.ExpandedData();write(R/'DATA_RECEIPT.json',data.receipt)
 e=Engine(data);preflight(e);paths=sorted((R/'runs/J1').glob('checkpoint_*.pt'))
 if paths:
  e.resume(paths[-1]);base=e.history[0];initial=[]
 else:
  base=assess(e)
  assert abs(base['FREE_VALIDATION']['perplexity']-17.925712)<2e-5 and abs(base['NATIVE_VALIDATION']['perplexity']-15.597933)<2e-5
  e.checkpoint();initial=[]
 if e.update==0:initial=restart(e)
 ceiling=1000 if (R/'EXTENSION_DECISION.json').exists() and read(R/'EXTENSION_DECISION.json')['passes'] else 500;reason=None
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:setattr(e,'stop',True))
 torch.cuda.reset_peak_memory_stats()
 with (R/'runs/J1/updates.jsonl').open('a') as log:
  for row in initial:log.write(json.dumps(row)+'\n')
  while e.update<ceiling and not e.stop:
   if (R/'STOP_REQUESTED').exists():reason='safe stop requested';break
   row=e.step();log.write(json.dumps(row)+'\n')
   if e.update%25==0:log.flush();m.status('HEAD_TRAINING',**row,**m.guard())
   if e.update%100==0 and e.update%250!=0:e.checkpoint()
   if e.update%250==0:
    result=assess(e);cp=e.checkpoint();gen=generation_decision(e,cp) if e.update>=500 else None
    verdict=decision(result,base,generation=gen);write(R/f'ASSESSMENT_{e.update:05d}.json',verdict)
    if e.update==500:
     ext=decision(result,base,e.history[-2],gen);write(R/'EXTENSION_DECISION.json',ext)
     if ext['passes']:ceiling=1000
     else:reason='500-update pilot complete; predefined extension conditions failed'
    if e.update>=750 and (result['FREE_VALIDATION']['AA_NLL']>base['FREE_VALIDATION']['AA_NLL']+.02 or not gen['passes']):reason='held-out or generation safeguard'
    with (R/'EXPERIMENT_DECISIONS.md').open('a') as f:f.write(f"\n+{e.update}: native VAL PPL {result['NATIVE_VALIDATION']['perplexity']}; geometry-free VAL PPL {result['FREE_VALIDATION']['perplexity']}; free gap {result['free_gap']}; assessment {verdict}.\n")
    if reason:break
  cp=e.checkpoint();write(R/'FROZEN_PARAMETER_EQUIVALENCE.json',geom_proof(e));write(R/'runs/J1/COMPLETED.json',{'new_head_optimizer_updates':e.update,'historical_updates_at_start':35500,'total_model_optimizer_updates':35500+e.update,'reexecuted_proof_updates':2,'latest_checkpoint':str(cp),'latest_sha256':sha(cp),'stop_reason':reason or ('safe signal' if e.stop else 'authorized maximum'),'sampler':e.sampler.report(),'elapsed_including_evaluation_seconds':e.elapsed+time.monotonic()-e.start,'resources':m.guard()});m.status('HEAD_TRAINING_STOPPED',update=e.update,reason=reason)
if __name__=='__main__':run()
