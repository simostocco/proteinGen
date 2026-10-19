"""Full J1 continuation with training-only masked reconstruction and paired geometry consistency."""
import collections,copy,gc,json,math,os,random,signal,subprocess,threading,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import j1_full_sequence as m
from protein_joint_sac.models.bidirectional_multitask import SpanReadout,SpanSampler,span_batch,span_loss,masked_states
from protein_joint_sac.rl.group_clipping import norm
R=m.R;D=R.parent;s=m.s;c=m.c;read=m.read;write=m.write;sha=m.sha;O=D/'e016_j1_clip_precision_pilot_20261018_v1';PREV=D/'e016_j1_full_sequence_20261018_v1';MILESTONES=[1000,2500,5000,7500,10000,15000,20000]
STOP=threading.Event()
def shared_norm(grads):return norm([g for n,g in grads.items() if n.startswith('encoder.') and g is not None])
def snapshot(model):return {n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
def clear(model):
 for p in model.parameters():p.grad=None

def initial_config():
 cfg=m.initial_config();cfg.update(source_total_optimizer_updates=35500,source_prior_additional_update=None,version='j1_bidirectional_multitask_v1',primary_updates=10000,max_updates=20000,learning_rate=1e-5,terminal_learning_rate=2e-6,warmup_updates=200,decay_end=10000,extension_LR='constant terminal LR through20k; no reheat',aux_peak_LR=3e-5,aux_terminal_LR=6e-6,lambda_masked=.20,lambda_consistency=.025,sequence_aux_shared_cap=.35,geometry_primary_AA_shared_cap=.5,masked_examples=16,masked_microbatch=8,paired_targets=8,dropout=.15,AA_mixture={'complete':.4,'partial_only':.3,'unused':.3},milestones=MILESTONES,rollout_period=None,horizons=[],retention='immutable evaluated milestones plus two alternating rolling recovery slots; no historical file touched',resource_runtime_gate_hours=6.)
 cfg['selection_new']={'free_VAL_target_NLL':math.log(17.303353)-.02,'paired95_vs_previous_research_upper_max':0.,'native_VAL_extra_max':.005,'free_gap_extra_base':.05,'free_gap_extra_credit_max':.10,'native_gap_extra_max':.05,'local_NLL_max':-.10796287496467151,'generation_clash_factor_vs_parent':1.35,'generation_clash_extra':.25,'contacts_ratio_min':.8,'relative_Rg_range':[.65,1.35],'native_Rg_error_factor':1.35,'native_Rg_error_extra':.10,'sequence_Hamming_ratio_min':.9,'structure_diversity_ratio_min':.7,'unique_sequences_min_32':30,'complete_min_32':32,'uncertainty':'paired identity/sample bootstrap2000; fixed point thresholds never relaxed after evaluation; known homology dependence remains'}
 cfg['extension']={'requires_new_selection_pass':True,'free_VAL_NLL_gain_7500to10000_min':.002,'native_VAL_extra_max':.005,'runtime_projection_gate_hours':6.,'disk_reserve_GiB':10}
 cfg['early_stop_new']={'after5000_min_recent_free_VAL_gain':.002,'alternative_native_VAL_gain':.002,'two_milestone_free_VAL_regression_vs_start':.03,'two_milestone_local_regression_vs_start':.15,'severe_clashes_per100_min':10.,'severe_clashes_parent_factor':3.,'severe_clashes_parent_extra':2.,'severe_Rg_relative_range':[.3,2.5],'all_nonfinite_emergency_stop':True}
 return cfg

def prepare(data):
 assert sha(m.SOURCE)==m.SOURCE_SHA
 for role in ['TRAIN','VALIDATION']:
  sets=[set(v) for v in data.populations[role].values()];assert set.union(*sets)==set(data.base.ids[role]);assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
 report=read(PREV/'ALL_TRAIN_IDENTITY_ADMISSION.json');assert report['source_sha256']==m.SOURCE_SHA;write(R/'ALL_TRAIN_IDENTITY_ADMISSION.json',report|{'new_lineage_population_reconciled':True,'no_split_or_geometry_admission_change':True})
 if not (R/'configs/FROZEN.json').exists():write(R/'configs/FROZEN.json',initial_config())
 # New auxiliary evaluation is fixed before learning, matched TRAIN category×length cells.
 if not (R/'configs/MASKED_EVALUATION.json').exists():
  sampler=SpanSampler(data,2026101945,'VALIDATION');rng=random.Random(2026101944);va=[]
  for band in range(5):
   ids=[h for h in data.base.ids['VALIDATION'] if s.j.band(len(data.seqs[h]))==band];rng.shuffle(ids);assert len(ids)>=200;va.extend(ids[:200])
  tr=[]
  for kind in ['complete','partial_only','unused']:
   for band in range(5):
    n=sum(h in set(data.populations['VALIDATION'][kind]) and s.j.band(len(data.seqs[h]))==band for h in va);ids=[h for h in data.populations['TRAIN'][kind] if s.j.band(len(data.seqs[h]))==band];rng.shuffle(ids);assert len(ids)>=n;tr.extend(ids[:n])
  write(R/'configs/MASKED_EVALUATION.json',{role:[sampler.plan(h,record=False) for h in sorted(ids)] for role,ids in [('TRAIN',tr),('VALIDATION',va)]})

class Engine(m.Engine):
 def __init__(self,data,cfg):
  super().__init__(data,cfg)
  with torch.random.fork_rng(devices=[0]):torch.manual_seed(2026101942);self.aux=SpanReadout(self.model).cuda()
  self.opt.add_param_group({'params':list(self.aux.parameters()),'lr':cfg['aux_peak_LR'],'initial_lr':cfg['aux_peak_LR']})
  for i,pg in enumerate(self.opt.param_groups):pg['initial_lr']=cfg['learning_rate'] if i==0 else cfg['aux_peak_LR'];pg['lr']=pg['initial_lr']
  def factor(k):return (k+1)/200 if k<200 else .2+.8*.5*(1+math.cos(math.pi*min(1.,(k-200)/9800)))
  self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.opt,factor);self.mask_sampler=SpanSampler(data);self.pair_exposures=0;self.consistency_stats=[];self.mask_micro=cfg['masked_microbatch'];self.best_sequence=0;self.micro_profile=[]
  self.lineage.update(version=cfg['version'],config=cfg,entrypoint_sha256=sha(__file__),auxiliary_parameters=sum(p.numel() for p in self.aux.parameters()),production_parameters=sum(p.numel() for p in self.model.parameters()),auxiliary_init='independent AA-head copy; MASK embedding mean input-AA embedding; no teacher/distillation',AA_task='64 target decisions40/30/30; selected qualified targets average native/masked CE, not extra identities',source_total_optimizer_updates=35500)
  if (R/'DECISIONS_FROZEN_BEFORE_BASELINE.json').exists():self.lineage['sealed_config_sha256']=sha(R/'configs/FROZEN.json')
  assert self.lineage['auxiliary_parameters']==7412 and self.lineage['production_parameters']==12351746
 def backward(self):
  self.model.train();self.aux.train();self.opt.zero_grad(set_to_none=True);aaex,geoex=self.sampler.draw();complete=set(self.data.populations['TRAIN']['complete']);eligible=[x for x in aaex if x[0] in complete and x[2]>=4];pairs=[eligible[int(i)] for i in np.linspace(0,len(eligible)-1,min(self.c['paired_targets'],len(eligible))).astype(int)] if eligible else [];pairset=set(pairs);rest=[x for x in aaex if x not in pairset];av=0.;gv=0.;cs=np.zeros(3);klvalue=0.
  for off in range(0,len(rest),16):
   b,y,_=self.data.batch(rest[off:off+16]);b=m.augment_available(b,.075)
   with torch.autocast('cuda',dtype=torch.bfloat16):h=m.encode_batch(self.model,b);loss=F.cross_entropy(self.model.aa_head(h).float(),y,reduction='sum')/64
   loss.backward();av+=float(loss.detach());del loss,h,b,y
  kl=None
  if pairs:
   b,y,_=self.data.batch(pairs);before=s.rng_state()
   with torch.autocast('cuda',dtype=torch.bfloat16):h=m.encode_batch(self.model,b);logits=self.model.aa_head(h).float();la=.5*F.cross_entropy(logits,y,reduction='sum')/64
   teacher=logits.detach().softmax(-1);la.backward();av+=float(la.detach());del h,logits,la;after=s.rng_state();s.restore_rng(before);bm={k:v.clone() for k,v in b.items()};bm['ca_observed']=b['valid'].clone();bm['ca_aug_visible']=torch.zeros_like(b['valid']);bm['valid'].zero_();bm['bond_valid'].zero_();bm['geometry_target_valid'].zero_()
   with torch.autocast('cuda',dtype=torch.bfloat16):h=m.encode_batch(self.model,bm);logits=self.model.aa_head(h).float();lb=.5*F.cross_entropy(logits,y,reduction='sum')/64
   kl=F.kl_div(logits.log_softmax(-1),teacher,reduction='batchmean');lb.backward(retain_graph=True);av+=float(lb.detach());klvalue=float(kl.detach());del h,lb,b,bm,y,logits;s.restore_rng(after)
  primary=snapshot(self.model);an=shared_norm(primary);clear(self.model)
  if kl is not None:(self.c['lambda_consistency']*kl).backward();del kl
  plans=self.mask_sampler.draw(self.c['masked_examples']);ml=0.;correct=targets=0
  for off in range(0,len(plans),self.mask_micro):
   batch=span_batch(self.data.seqs,plans[off:off+self.mask_micro])
   with torch.autocast('cuda',dtype=torch.bfloat16):loss,logits,mask=span_loss(self.model,self.aux,batch);objective=self.c['lambda_masked']*loss.sum()/len(plans)
   objective.backward();ml+=float(loss.detach().sum())/len(plans);correct+=int((logits.detach().argmax(-1).eq(batch[2])&mask).sum());targets+=int(mask.sum());del loss,logits,objective,batch
  auxgrad=snapshot(self.model);auxnorm=shared_norm(auxgrad);beta=min(1.,self.c['sequence_aux_shared_cap']*an/max(auxnorm,1e-30));dot=sum(float((primary[n].float()*v.float()).sum()) for n,v in auxgrad.items() if n.startswith('encoder.') and v is not None and primary[n] is not None)
  sequence={n:(None if primary[n] is None and auxgrad[n] is None else (torch.zeros_like(p) if primary[n] is None else primary[n])+(torch.zeros_like(p) if auxgrad[n] is None else beta*auxgrad[n])) for n,p in self.model.named_parameters()};del primary,auxgrad;clear(self.model)
  for off in range(0,32,16):
   b,y,g=self.data.batch(geoex[off:off+16]);b=m.augment_available(b,.075);a,local,parts=s.loss_call(self.model,b,y,g);(self.c['lambda_local']*local.sum()/32).backward();gv+=float(local.detach().sum())/32;cs+=parts.detach().sum(0).cpu().numpy();del a,local,parts,b,y,g
  extra={}
  if (self.update+1)%2==0:
   ex=geoex[-4:];b,y,g=self.data.batch(ex);b=m.augment_available(b,.075);target=torch.tensor(np.asarray([self.data.base.ca[o][t] for h,o,t in ex]),device='cuda');nl,counts=s.direct_nonlocal(self.model,b,y,target,K=4,seed=180042);(self.c['lambda_nonlocal']*nl).backward();self.nonlocal_targets+=len(counts);extra={'nonlocal_energy_score':float(nl.detach()),'nonlocal_targets':len(counts)};del nl,b,y,g,target
  self.pair_exposures+=len(pairs)
  return sequence,{'AA_NLL_stochastic':av,'masked_NLL_stochastic':ml,'masked_correct':correct,'masked_targets':targets,'paired_targets':len(pairs),'consistency_KL':klvalue,'primary_AA_shared_L2':an,'sequence_aux_shared_raw_L2':auxnorm,'sequence_aux_scale':beta,'sequence_aux_to_primary_ratio':beta*auxnorm/max(an,1e-30),'primary_aux_shared_cosine':dot/max(an*auxnorm,1e-30),'local_geometry_NLL':gv,'component_mean_NLL':(cs/32).tolist(),'sample_order_sha256':s.j.digest([aaex,geoex,plans]),**extra}
 def step(self):
  m.guard();start=time.monotonic();lr=[pg['lr'] for pg in self.opt.param_groups];seq,out=self.backward();sn=shared_norm(seq);stats=m.combine(self.model,seq,self.c['block_limits'],ratio=.5*out['primary_AA_shared_L2']/max(sn,1e-30));del seq
  for p in self.aux.parameters():
   if p.grad is not None and not bool(torch.isfinite(p.grad).all()):raise FloatingPointError('nonfinite auxiliary gradient')
  ar=float(torch.nn.utils.clip_grad_norm_(self.aux.readout.parameters(),self.c['block_limits']['A'],error_if_nonfinite=True));am=float(torch.nn.utils.clip_grad_norm_([self.aux.mask_embedding],1.,error_if_nonfinite=True));measure=(self.update+1)%100==0
  before={n:p.detach().clone() for n,p in self.model.named_parameters()} if measure else None;self.opt.step();self.scheduler.step();self.update+=1
  if not all(bool(torch.isfinite(p).all()) for p in list(self.model.parameters())+list(self.aux.parameters())) or not c.finite_state(self.opt):raise FloatingPointError('nonfinite update')
  if measure:
   bs,_=m.blocks(self.model);stats['parameter_update_L2_by_block']={k:math.sqrt(sum(float((p.detach()-before[n] if rows is None else (p.detach()-before[n])[list(rows)]).double().square().sum()) for n,p,rows in v)) for k,v in bs.items()}
  torch.cuda.synchronize();c.LAST[0]=time.monotonic();return {'new_update':self.update,'total_optimizer_updates':35500+self.update,'seconds':time.monotonic()-start,'learning_rates':lr,'aux_head_gradient_L2_before_clip':ar,'MASK_embedding_gradient_L2_before_clip':am,'geometry_to_primary_AA_shared_ratio':stats['geometry_to_AA_shared_ratio']*sn/max(out['primary_AA_shared_L2'],1e-30),**out,**stats}
 def payload(self):return {'lineage':self.lineage,'model':self.model.state_dict(),'auxiliary':self.aux.state_dict(),'optimizer':self.opt.state_dict(),'scheduler':self.scheduler.state_dict(),'rng':s.rng_state(),'sampler':self.sampler.state_dict(),'masked_sampler':self.mask_sampler.state_dict(),'rollout_rng':self.rollrng.getstate(),'new_update':self.update,'total_optimizer_updates':35500+self.update,'nonlocal_targets':self.nonlocal_targets,'paired_targets':self.pair_exposures,'evaluation_updates':[v['new_update'] for v in self.history],'evaluation_sha256':{str(v['new_update']):sha(R/'runs/J1'/f"evaluation_{v['new_update']:05d}.json") for v in self.history},'best_sequence':self.best_sequence,'elapsed':self.elapsed+time.monotonic()-self.start}
 def checkpoint(self,path=None):
  c.guard(700*2**20);path=Path(path) if path else R/'runs/J1'/f'checkpoint_{self.update:05d}.pt';payload=self.payload()
  if path.exists() and path.name.startswith('checkpoint_'):
   st=s.load_checked(path);assert st['new_update']==self.update and c.same(st['model'],self.model.state_dict());return path
  digest=s.save_atomic(path,payload);st=s.load_checked(path);assert c.same(st['model'],self.model.state_dict()) and c.same(st['auxiliary'],self.aux.state_dict()) and st['new_update']==self.update
  with (R/'CHECKPOINT_PUBLICATIONS.jsonl').open('a') as f:f.write(json.dumps({'path':str(path),'new_update':self.update,'sha256':digest,'bytes':path.stat().st_size,'rolling':path.name.startswith('recovery_')})+'\n')
  return path
 def resume(self,path):
  st=s.load_checked(path);assert st['lineage']==self.lineage,'lineage mismatch';u=st['new_update'];assert st['scheduler']['last_epoch']==u and st['sampler']['cursor']==u
  for i,pg in enumerate(st['optimizer']['param_groups']):
   expected=35500+u if i==0 else u;assert {int(st['optimizer']['state'][k]['step']) for k in pg['params'] if k in st['optimizer']['state']}==({expected} if expected else set())
  self.model.load_state_dict(st['model']);self.aux.load_state_dict(st['auxiliary']);self.opt.load_state_dict(st['optimizer']);self.scheduler.load_state_dict(st['scheduler']);self.sampler.load_state_dict(st['sampler']);self.mask_sampler.load_state_dict(st['masked_sampler']);self.rollrng.setstate(st['rollout_rng']);self.update=u;self.nonlocal_targets=st['nonlocal_targets'];self.pair_exposures=st['paired_targets'];self.best_sequence=st['best_sequence'];self.elapsed=st['elapsed'];self.start=time.monotonic();self.history=[]
  for v in st['evaluation_updates']:
   p=R/'runs/J1'/f'evaluation_{v:05d}.json';assert sha(p)==st['evaluation_sha256'][str(v)];self.history.append(read(p))
  s.restore_rng(st['rng'])

@torch.no_grad()
def masked_eval(e,role):
 e.model.eval();e.aux.eval();plans=read(R/'configs/MASKED_EVALUATION.json')[role];plans=sorted(plans,key=lambda p:len(e.data.seqs[p[0]]));by={};correct=targets=0;bands=collections.defaultdict(list)
 for off in range(0,len(plans),e.mask_micro):
  subset=plans[off:off+e.mask_micro];batch=span_batch(e.data.seqs,subset)
  with torch.autocast('cuda',dtype=torch.bfloat16):ls,logits,mask=span_loss(e.model,e.aux,batch)
  ok=(logits.argmax(-1).eq(batch[2])&mask);correct+=int(ok.sum());targets+=int(mask.sum())
  for (h,pos),v,cnt in zip(subset,ls.cpu().tolist(),ok.sum(1).cpu().tolist()):by[h]={'NLL':v,'correct':cnt,'targets':len(pos)};bands[str(s.j.band(len(e.data.seqs[h])))].append(v)
 return {'role':role,'identities':len(by),'masked_targets':targets,'identity_mean_NLL':float(np.mean([v['NLL'] for v in by.values()])),'masked_accuracy':correct/targets,'per_identity':by,'length_bands_NLL':{k:float(np.mean(v)) for k,v in bands.items()},'label':'BIDIRECTIONAL MASK RECONSTRUCTION; NOT AUTOREGRESSIVE PERPLEXITY'}

def assess(e):
 rng=s.rng_state();row=m.assess(e)
 for role in ['FREE_TRAIN','FREE_VALIDATION']:
  for z in row[role]['categories'].values():z['decisions']=4*z['identities']
  for z in row[role]['length_bands'].values():z['identities']=z['decisions'];z['decisions']*=4
 row['MASKED_TRAIN']=masked_eval(e,'TRAIN');row['MASKED_VALIDATION']=masked_eval(e,'VALIDATION');s.restore_rng(rng);return row

def generation(e,path,n=8):return m.generation(e,path,n)
def generation_check(rows,n=32):
 parent=read(O/'generation/final00500_ca_fp64_interior_v1.json');ps=m.old.metric_summary(parent);ss=m.old.metric_summary(rows);native=read(s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json')['native'];fails=[];thresholds={}
 for N,z in ss.items():
  b=ps[N]
  if z['complete_valid']<2:fails.append(N+' incomplete generation');continue
  clash=1.35*b['clashes_per100_residues']['mean']+.25;ref=float(np.median([v['metrics']['Rg_over_N_cuberoot'] for v in native['TRAIN'][N]['rows']]));old=abs(math.log(b['Rg_over_N_cuberoot']['mean']/ref));new=abs(math.log(z['Rg_over_N_cuberoot']['mean']/ref));ratio=z['Rg_A']['mean']/b['Rg_A']['mean'];thresholds[N]={'clashes_per100_max':clash,'contacts_min':.8*b['long_contact_fraction']['mean'],'Rg_ratio_min':.65,'Rg_ratio_max':1.35,'native_logRg_error_max':1.35*old+.10}
  if z['complete_valid']!=n:fails.append(N+' incomplete generation')
  if z['complete_valid']<2:continue
  if z['unique_sequences']<math.ceil(n*30/32):fails.append(N+' sequence uniqueness')
  if z['clashes_per100_residues']['mean']>clash:fails.append(N+' sequence-first clash limit')
  if z['long_contact_fraction']['mean']<.8*b['long_contact_fraction']['mean']:fails.append(N+' contact loss')
  if not .65<=ratio<=1.35 or new>1.35*old+.10:fails.append(N+' compactness')
  if z['mean_pair_sequence_Hamming_fraction']<.9*b['mean_pair_sequence_Hamming_fraction']:fails.append(N+' sequence diversity')
  if z['mean_pair_Rg_normalized_distance_fingerprint_RMS']<.7*b['mean_pair_Rg_normalized_distance_fingerprint_RMS']:fails.append(N+' structure diversity')
 return {'passes':not fails,'failures':fails,'summary':ss,'thresholds':thresholds,'samples_per_length':n}

def decision(row,base,gen=None):
 cfg=read(R/'configs/FROZEN.json')['selection_new'];research=read(PREV/'runs/J1/evaluation_02500.json');v=row['FREE_VALIDATION'];keys=sorted(v['per_identity']);ci=m.old.paired_ci([v['per_identity'][h]['AA_NLL']-research['FREE_VALIDATION']['per_identity'][h]['AA_NLL'] for h in keys]);gain=base['FREE_VALIDATION']['AA_NLL']-v['AA_NLL'];fails=[]
 if v['AA_NLL']>cfg['free_VAL_target_NLL']:fails.append('free VAL target not reached')
 if ci['bootstrap_95pct'][1]>=0:fails.append('paired gain vs previous research unsupported')
 if row['NATIVE_VALIDATION']['AA_NLL']>base['NATIVE_VALIDATION']['AA_NLL']+.005:fails.append('native VAL not competitive')
 if row['free_gap']>base['free_gap']+.05+min(.1,max(0.,gain)):fails.append('free gap beyond compensated allowance')
 if row['native_gap']>base['native_gap']+.05:fails.append('native gap >.05extra')
 if row['NATIVE_VALIDATION']['geometry_NLL']>cfg['local_NLL_max']:fails.append('new local allowance exceeded')
 if gen:fails.extend(gen['failures'])
 else:fails.append('final32 generation assessment pending')
 return {'passes':not fails,'failures':fails,'paired_vs_previous_research':ci,'free_VAL_gain_vs_parent':gain}

def run():
 assert not (R/'runs/J1/COMPLETED.json').exists(),'already stopped; no automatic resume'
 s.j.core.setup();m.guard();m.status('MULTITASK_DATA_LOADING');data=m.ExpandedData();write(R/'DATA_RECEIPT.json',data.receipt);prepare(data);cfg=read(R/'configs/FROZEN.json');e=Engine(data,cfg)
 paths=list((R/'runs/J1').glob('checkpoint_*.pt'))+list((R/'runs/J1').glob('recovery_slot_*.pt'))
 if paths:
  valid=[]
  for p in paths:
   try:st=s.load_checked(p);valid.append((st['new_update'],p));del st
   except Exception as err:write(R/'runtime'/('INVALID_'+p.name+'.json'),{'path':str(p),'error':str(err),'preserved':True})
  if not valid:raise ValueError('no valid recovery state')
  e.resume(max(valid,key=lambda z:z[0])[1])
 else:
  from j1_multitask_preflight import preflight
  preflight(e);write(R/'configs/FROZEN.json',e.c);e.lineage['config']=e.c;e.lineage['sealed_config_sha256']=sha(R/'configs/FROZEN.json');write(R/'DECISIONS_FROZEN_BEFORE_BASELINE.json',{'config_sha256':sha(R/'configs/FROZEN.json'),'source_commit':e.lineage['source_commit'],'before_first_main_evaluation':True,'selection_new':e.c['selection_new'],'MLM_validation_panel_sha256':sha(R/'configs/MASKED_EVALUATION.json')})
  base=assess(e);assert abs(base['FREE_VALIDATION']['perplexity']-17.925712)<2e-5;assert abs(base['NATIVE_VALIDATION']['geometry_NLL']+.127962875)<2e-6;e.history=[base];write(R/'runs/J1/evaluation_00000.json',base);e.checkpoint()
 ceiling=20000 if (R/'EXTENSION_DECISION.json').exists() and read(R/'EXTENSION_DECISION.json')['passes'] else 10000;reason=None
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:STOP.set())
 c.LAST[0]=time.monotonic()
 def watchdog():
  while not STOP.wait(30):
   if time.monotonic()-c.LAST[0]>1800:write(R/'WATCHDOG_STOP.json',{'no_progress_seconds':1800});STOP.set()
 threading.Thread(target=watchdog,daemon=True).start();torch.cuda.reset_peak_memory_stats()
 with (R/'runs/J1'/f'updates_from_{e.update:05d}_{int(time.time())}.jsonl').open('x') as log:
  while e.update<ceiling and not STOP.is_set():
   if (R/'STOP_REQUESTED').exists():reason='safe stop requested';break
   row=e.step();log.write(json.dumps(row)+'\n')
   if e.update%25==0:log.flush();m.status('MULTITASK_TRAINING',**row,**m.guard())
   if e.update in MILESTONES:
    result=assess(e);e.history.append(result);write(R/'runs/J1'/f'evaluation_{e.update:05d}.json',result);write(R/'SEQUENCE_TRAIN_VALIDATION_METRICS.json',e.history)
    if not e.best_sequence or result['FREE_VALIDATION']['AA_NLL']<next(v for v in e.history if v['new_update']==e.best_sequence)['FREE_VALIDATION']['AA_NLL']:e.best_sequence=e.update
    cp=e.checkpoint();rows=generation(e,cp,8);check=generation_check(rows,8);write(R/'generation'/f'assessment_{e.update:05d}_8.json',check);verdict=decision(result,e.history[0]);write(R/f'SELECTION_{e.update:05d}.json',verdict)
    sums=check['summary'];parent=m.old.metric_summary(read(O/'generation/final00500_ca_fp64_interior_v1.json'))
    if any(z['complete_valid']<8 or z['clashes_per100_residues']['mean']>max(10.,3*parent[n]['clashes_per100_residues']['mean']+2) or not .3<=z['Rg_A']['mean']/parent[n]['Rg_A']['mean']<=2.5 for n,z in sums.items()):reason='severe generation safeguard'
    if len(e.history)>=3 and all(v['FREE_VALIDATION']['AA_NLL']>e.history[0]['FREE_VALIDATION']['AA_NLL']+.03 or v['NATIVE_VALIDATION']['geometry_NLL']>e.history[0]['NATIVE_VALIDATION']['geometry_NLL']+.15 for v in e.history[-2:]):reason='two-milestone severe AA/local regression'
    if e.update>=5000:
     previous=e.history[-2];free_progress=previous['FREE_VALIDATION']['AA_NLL']-result['FREE_VALIDATION']['AA_NLL'];native_progress=previous['NATIVE_VALIDATION']['AA_NLL']-result['NATIVE_VALIDATION']['AA_NLL']
     if free_progress<.002 and not (native_progress>=.002 and free_progress>=0):reason='held-out next-AA plateau after5000'
    if e.update==10000 and not reason:
     full=generation(e,cp,32);gcheck=generation_check(full,32);ext=decision(result,e.history[0],gcheck);prev=next(v for v in e.history if v['new_update']==7500)
     if prev['FREE_VALIDATION']['AA_NLL']-result['FREE_VALIDATION']['AA_NLL']<.002:ext['failures'].append('recent7500to10000 gain<.002');ext['passes']=False
     if m.guard()['D_free_bytes']<10*2**30+2*cp.stat().st_size+900*2**20:ext['failures'].append('extension checkpoint headroom');ext['passes']=False
     write(R/'EXTENSION_DECISION.json',ext)
     if ext['passes']:ceiling=20000
     else:reason='10k completed; extension criteria not satisfied'
    with (R/'EXPERIMENT_DECISIONS.md').open('a') as f:f.write(f"\n+{e.update}: free VAL PPL {result['FREE_VALIDATION']['perplexity']}; native VAL PPL {result['NATIVE_VALIDATION']['perplexity']}; local {result['NATIVE_VALIDATION']['geometry_NLL']}; MLM VAL {result['MASKED_VALIDATION']['identity_mean_NLL']}; reason {reason}.\n")
    if reason:break
   elif e.update%500==0:e.checkpoint(R/'runs/J1'/f'recovery_slot_{(e.update//500)%2}.pt')
  cp=e.checkpoint();write(R/'runs/J1/COMPLETED.json',{'new_optimizer_updates':e.update,'historical_updates_at_start':35500,'total_optimizer_updates':35500+e.update,'latest_checkpoint':str(cp),'latest_sha256':sha(cp),'best_sequence_update':e.best_sequence,'stop_reason':reason or ('signal/watchdog' if STOP.is_set() else 'authorized maximum'),'next_AA_sampler':e.sampler.report(),'masked_sampler':e.mask_sampler.report(),'paired_target_exposures':e.pair_exposures,'nonlocal_targets':e.nonlocal_targets,'differentiated_long_rollout_residues':0,'elapsed_with_evaluations':e.elapsed+time.monotonic()-e.start,'resources':m.guard()});m.status('MULTITASK_TRAINING_STOPPED',update=e.update,reason=reason)
if __name__=='__main__':run()
