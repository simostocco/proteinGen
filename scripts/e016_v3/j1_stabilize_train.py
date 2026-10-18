"""One evidence-gated J1 continuation. Exact source-state provenance; no other models/trainers."""
import collections,copy,gc,json,math,os,random,signal,subprocess,threading,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
import j1_clip_pilot as c
import j1_torsion_diagnostic as diag
from partial_train import Engine as PE
from j1_sequence_assess import partial_eval,check_candidate
from j1_global_finalize import metric_summary,paired_ci
from protein_joint_sac.rl.component_clipping import combine,blocks,VERSION
from protein_joint_sac.models.batched_bc import losses
R=Path(__file__).resolve().parents[3];O=R.parent/'e016_j1_clip_precision_pilot_20261018_v1'
c.R=R;c.s.R=R;c.s.j.R=R;c.s.j.core.R=R
write=c.write;read=c.read;sha=c.sha;STOP=threading.Event();MILESTONES=[500,1000,2500,5000,7500,10000]

def status(phase,**kw):c.status(phase,**kw);print(json.dumps({'phase':phase,**kw}),flush=True)
def safe():
 out=c.guard(350*2**20)
 if (R/'STOP_REQUESTED').exists():STOP.set()
 return out

def frozen():
 cfg=dict(read(O/'configs/FROZEN.json'));cfg.update(name='J1',version=VERSION,max_updates=10000,primary_updates=5000,learning_rate=1e-5,terminal_learning_rate=2e-6,warmup_updates=100,decay_end=5000,extension_LR='constant2e-6 after5000; no reheat',gradient_policy=VERSION,group_limits=None)
 cfg['extension']={'VAL_PPL_max':15.490720,'AA_paired_NLL95_upper_max':0.,'AA_improvement_vs2500_min_nats':.001,'local_NLL_max':-.113708968,'torsion_delta_vs_start_max':.02,'distance_angle_delta_vs_start_max':.02,'torsion95coverage_max_drop':.01,'torsion_top1pct_contribution_max_factor':1.10,'torsion_top1pct_contribution_max_extra':.01,'gap_max_extra_vs_start':.02,'contacts_min_ratio_vs_start':.8,'full32_per_length_frozen_safeguards':True}
 cfg['promotion']={'AA_NLL_improvement_vs_start_min':.002,'paired_AA95_upper_max':0.,'gap_extra_vs_start_max':.02,'local_NLL_max':-.113708968,'torsion_mean_delta_max':.02,'distance_angle_delta_max':.02,'torsion95coverage_max_drop':.01,'torsion_coverage_absolute_error_sum_max_extra':.03,'torsion_tail1pct_max_factor':1.10,'torsion_tail1pct_max_extra':.01,'long_contact_ratio_min_vs_start':.8,'all_original32_per_length_generation_safeguards':True}
 cfg['retention']='all evaluated milestones full-state + latest250-step recovery; retire only verified superseded NEW nonmilestone recovery files'
 return cfg

class Engine(c.Engine):
 def __init__(self,data,cfg):
  torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
  self.data=data;self.c=cfg;self.name='J1';self.joint=True;self.model=c.s.j.core.model('J1');c.s.set_dropout(self.model,.15)
  receipt=read(R/'replay/RECONSTRUCTION.json');self.exact=receipt['status']=='EXACT_RECONSTRUCTION';origin=Path(receipt['checkpoint']) if self.exact else O/'runs/J1/model_00500.pt';state=c.s.load_checked(origin)
  self.model.load_state_dict(state['model'],strict=True);self.opt=torch.optim.AdamW(self.model.parameters(),lr=cfg['learning_rate'],weight_decay=.01,fused=True)
  if self.exact:self.opt.load_state_dict(state['optimizer'])
  for g in self.opt.param_groups:g['lr']=cfg['learning_rate'];g['initial_lr']=cfg['learning_rate']
  def factor(k):
   if k<100:return (k+1)/100
   return .2+.8*.5*(1+math.cos(math.pi*min(1.,(k-100)/4900)))
  self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.opt,factor);self.sampler=c.s.Sampler(data)
  if self.exact:self.sampler.load_state_dict(state['sampler'])
  else:
   parent=c.s.load_checked(c.SOURCE);self.sampler.load_state_dict(parent['sampler']);del parent
   for _ in range(500):self.sampler.draw()
  self.source_counts=dict(self.sampler.counts);self.source_coverage=self.sampler.report();self.source_visited=dict(self.sampler.cover.visited) if hasattr(self.sampler.cover,'visited') else dict(self.sampler.state_dict()['coverage']['visited'])
  self.rollrng=random.Random();self.rollrng.setstate(state['rollout_rng'] if self.exact else random.Random(180042).getstate());self.update=0;self.nonlocal_targets=0;self.history=[];self.best=0;self.elapsed=0.;self.start=time.monotonic()
  self.lineage={'version':VERSION,'config':cfg,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip(),'package_sha256':c.s.code_hash(),'entrypoint_sha256':sha(__file__),'selected_weights_sha256':sha(O/'runs/J1/model_00500.pt'),'origin':str(origin),'origin_sha256':sha(origin),'exact_optimizer_reconstruction':self.exact,'historical_updates_at_start':35500,'parent_lineage':state['lineage'],'panels_sha256':sha(R/'configs/EVALUATION_PANELS.json'),'geometry_version':'ca_internal_signed_v1','chemistry':'DISABLED','inference_protocol':'ca_fp64_interior_v1','normalization':'UNCHANGED','TEST':'NOT_USED'}
  if self.exact:c.s.restore_rng(state['rng'])
  else:torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);random.seed(42)
  del state
 def step(self):
  safe();start=time.monotonic();lr=self.opt.param_groups[0]['lr'];aa,out=self.backward();stats=combine(self.model,aa,self.c['block_limits']);del aa
  measure=(self.update+1)%25==0;before={n:p.detach().clone() for n,p in self.model.named_parameters()} if measure else None
  self.opt.step();self.scheduler.step();self.update+=1
  if not all(bool(torch.isfinite(p).all()) for p in self.model.parameters()):raise FloatingPointError('nonfinite model update')
  if measure:
   if not c.finite_state(self.opt):raise FloatingPointError('nonfinite optimizer moments')
   bb,_=blocks(self.model);stats['parameter_update_L2_by_block']={k:math.sqrt(sum(float((p.detach()-before[n] if rows is None else (p.detach()-before[n])[list(rows)]).double().square().sum()) for n,p,rows in v)) for k,v in bb.items()};del before
  torch.cuda.synchronize();c.LAST[0]=time.monotonic()
  return {'new_update':self.update,'total_optimizer_updates':35500+self.update,'seconds':time.monotonic()-start,'learning_rate':lr,'sampler_cursor':self.sampler.cursor,'nonlocal_targets_total':self.nonlocal_targets,'differentiated_rollout_residues':0,**out,**stats}
 def checkpoint(self,path):
  safe();payload={'lineage':self.lineage,'model':self.model.state_dict(),'optimizer':self.opt.state_dict(),'scheduler':self.scheduler.state_dict(),'rng':c.s.rng_state(),'sampler':self.sampler.state_dict(),'rollout_rng':self.rollrng.getstate(),'new_update':self.update,'total_optimizer_updates':35500+self.update,'nonlocal_targets':self.nonlocal_targets,'history':self.history,'best':self.best,'source_counts':self.source_counts,'source_coverage':self.source_coverage,'source_visited':self.source_visited,'elapsed':self.elapsed+time.monotonic()-self.start}
  if Path(path).exists():return sha(path)
  digest=c.s.save_atomic(path,payload);st=c.s.load_checked(path)
  if st['new_update']!=self.update or not c.same(st['model'],self.model.state_dict()):raise ValueError('atomic checkpoint verification')
  return digest
 def resume(self,path):
  st=c.s.load_checked(path)
  if st['lineage']!=self.lineage:raise ValueError('lineage changed; do not silently resume')
  if st['scheduler']['last_epoch']!=st['new_update'] or st['sampler']['cursor']!=5500+st['new_update']:raise ValueError('scheduler/sampler counter mismatch')
  expected=35500+st['new_update'] if self.exact else st['new_update']
  if {int(float(v['step'])) for v in st['optimizer']['state'].values()}!={expected}:raise ValueError('optimizer counter mismatch')
  self.model.load_state_dict(st['model']);self.opt.load_state_dict(st['optimizer']);self.scheduler.load_state_dict(st['scheduler']);self.sampler.load_state_dict(st['sampler']);self.rollrng.setstate(st['rollout_rng']);self.update=st['new_update'];self.nonlocal_targets=st['nonlocal_targets'];self.history=st['history'];self.best=st['best'];self.elapsed=st['elapsed'];self.source_counts=st['source_counts'];self.source_coverage=st['source_coverage'];self.source_visited=st['source_visited'];self.start=time.monotonic();c.s.restore_rng(st['rng'])

@torch.no_grad()
def complete_eval(e,panel,role):
 m=e.model;m.eval();data=e.data.base;ordered=sorted(panel,key=lambda x:(x[2],x[0],x[1]));rows=collections.defaultdict(list);by=collections.defaultdict(list);bands=collections.defaultdict(list);positions=collections.defaultdict(list);start=time.monotonic()
 for off in range(0,len(ordered),16):
  ex=ordered[off:off+16]
  if any(data.proposal.sequences[h]['split']!=role for h,o,t in ex):raise ValueError('split mismatch')
  b,y,g=c.s.j.core.make(data,ex,True)
  with torch.autocast('cuda',dtype=torch.bfloat16):hstate=c.s.encode_batch(m,b);logits=m.aa_head(hstate).float();aa=F.cross_entropy(logits,y,reduction='none')
  from protein_joint_sac.rl.sequence_first import geometry_parameters
  p=geometry_parameters(m,hstate,y).float();mu=p[:,:3];ls=p[:,3:].clamp(-5,2);mask=b['t'][:,None]>=torch.arange(1,4,device='cuda')[None];gg=g.double();u=torch.stack([torch.where(mask[:,0],gg[:,0],1.).log(),torch.logit(torch.where(mask[:,1],gg[:,1]/math.pi,.5)),torch.atanh(torch.where(mask[:,2],gg[:,2]/math.pi,0.))],-1);jac=torch.stack([u[:,0],math.log(math.pi)-F.softplus(-u[:,1])-F.softplus(u[:,1]),math.log(math.pi)+2*(math.log(2)-u[:,2]-F.softplus(-2*u[:,2]))],-1);nll=torch.where(mask,.5*((u-mu)/ls.exp()).square()+ls+.5*math.log(2*math.pi)+jac,0.);geo=nll.sum(1)/mask.sum(1).clamp_min(1)
  if off==4096:
   with torch.autocast('cuda',dtype=torch.bfloat16):ra,rg,rp=losses(m,b,y,g,geometry_fp32=True)
   if not torch.equal(aa,ra) or not torch.equal(nll,rp):raise ValueError('reference density/AA parity failed')
  cor=logits.topk(3,dim=-1).indices.eq(y[:,None]).cpu().numpy()
  for key,v in [('aa',aa),('nll',nll),('mask',mask),('p',p),('u',u),('g',g)]:rows[key].append(v.cpu().numpy())
  for (identity,o,t),a,z,cr in zip(ex,aa.cpu().tolist(),geo.cpu().tolist(),cor):
   v=[a,z,float(cr[0]),float(cr.any())];by[identity].append(v);bands[str(c.s.j.band(len(data.seqs[identity])))].append(v);positions[str(min(9,int(10*t/len(data.seqs[identity]))))].append(v)
  c.LAST[0]=time.monotonic()
 z={k:np.concatenate(v) for k,v in rows.items()};z.update(N=np.array([len(data.seqs[h]) for h,o,t in ordered]),t=np.array([t for h,o,t in ordered]),anomaly=np.zeros(len(ordered),bool));summary=diag.summarize(z,ordered)
 def short(values):
  v=np.asarray(values).mean(0);return {'decisions':len(values),'AA_NLL':float(v[0]),'perplexity':math.exp(float(v[0])),'geometry_NLL':float(v[1]),'top1':float(v[2]),'top3':float(v[3])}
 result=short([v for vv in by.values() for v in vv]);result.update(role=role,identities=len(by),component_NLL=[summary['components'][k]['target_weighted']['mean'] for k in ['distance','angle','torsion']],component_counts=[summary['components'][k]['target_weighted']['n'] for k in ['distance','angle','torsion']],per_identity={h:short(v) for h,v in by.items()},length_bands={k:short(v) for k,v in bands.items()},prefix_position_deciles={k:short(v) for k,v in positions.items()},seconds=time.monotonic()-start)
 (R/'calibration').mkdir(exist_ok=True);np.savez_compressed(R/'calibration'/f'{role}_{e.update:05d}.npz',**z);write(R/'calibration'/f'{role}_{e.update:05d}.json',summary)
 return result

def assess(e):
 rng=c.s.rng_state()
 with c.s.historical_evaluation():
  panels=read(R/'configs/EVALUATION_PANELS.json');tr=complete_eval(e,panels['TRAIN'],'TRAIN');va=complete_eval(e,panels['VALIDATION'],'VALIDATION');part=PE.evaluate(e,read(R/'configs/PARTIAL_VALIDATION_PANEL.json'),'PARTIAL');acc=partial_eval(e.model,e.data,sorted(read(R/'configs/PARTIAL_VALIDATION_PANEL.json'),key=lambda x:(x[2],x[0],x[1])),'VALIDATION')
 c.s.restore_rng(rng);return {'new_update':e.update,'TRAIN':tr,'VALIDATION':va,'PARTIAL_VALIDATION':part,'PARTIAL_VALIDATION_AA_ACCURACY':acc}

def native_checks(row):
 orig=read(c.OLD/'SEQUENCE_BASELINES.json')['J1'];va=row['VALIDATION'];tr=row['TRAIN'];fail=[]
 if va['AA_NLL']>=orig['VALIDATION']['AA_NLL']:fail.append('AA validation safeguard')
 if va['geometry_NLL']>orig['VALIDATION']['geometry_NLL']+.10:fail.append('local geometry safeguard')
 if va['AA_NLL']-tr['AA_NLL']>orig['VALIDATION']['AA_NLL']-orig['TRAIN']['AA_NLL']:fail.append('gap safeguard')
 return fail

def generation(e,path,n):
 rng=c.s.rng_state()
 with c.s.historical_evaluation():rows=c.generate(e.model,path,f'new{e.update:05}',n)
 c.s.restore_rng(rng);return rows

def calibrate(data,cfg):
 e=Engine(data,cfg);rows=[];aa_bands=collections.Counter();geo_bands=collections.Counter();start=time.monotonic()
 # Actual64AA/32complete-geometry sampling, dropout.15, augmentation.075; no optimizer updates.
 draw=e.sampler.draw
 def tracked():
  a,g=draw()
  aa_bands.update(c.s.j.band(len(data.seqs[h])) for h,o,t in a);geo_bands.update(c.s.j.band(len(data.seqs[h])) for h,o,t in g);return a,g
 e.sampler.draw=tracked
 for i in range(16):
  a,loss=e.backward();row=combine(e.model,a);rows.append(row);e.model.zero_grad(set_to_none=True);e.update+=1;status('TRAIN_ONLY_BLOCK_CALIBRATION',batch=i+1)
 if set(aa_bands)!=set(range(5)) or set(geo_bands)!=set(range(5)):raise ValueError('calibration lacks length coverage')
 limits={k:float(np.quantile([v['group_before'][k] for v in rows],.9)) for k in rows[0]['group_before']}
 if any(not math.isfinite(v) or v<=0 for v in limits.values()):raise ValueError('invalid calibrated block limit')
 write(R/'BLOCK_CALIBRATION.json',{'rows':rows,'limits':limits,'AA_length_bands':dict(aa_bands),'geometry_length_bands':dict(geo_bands),'source':'16 actual TRAIN-only batches, p90 separate norms before block clipping, shared cap already applied onlyS','AA_decisions':1024,'geometry_decisions':512,'optimizer_steps':0,'seconds':time.monotonic()-start});del e;gc.collect();torch.cuda.empty_cache();return limits

def checkpoint(e):
 path=R/'runs/J1'/f'checkpoint_{e.update:05d}.pt';e.checkpoint(path)
 for p in (R/'runs/J1').glob('checkpoint_*.pt'):
  u=int(p.stem.split('_')[-1])
  if u in MILESTONES or u in [e.update,e.best,4]:continue
  digest=sha(p);c.s.load_checked(p)
  with (R/'NEW_RETENTION.jsonl').open('a') as f:f.write(json.dumps({'retired_new_transient':str(p),'sha256':digest,'update':u,'verified_successor':str(path)})+'\n')
  p.unlink();p.with_name(p.name+'.sha256').unlink()
 return path

def restart_proof(e):
 first=[e.step() for _ in range(2)];p=R/'runs/J1/checkpoint_00002.pt';e.checkpoint(p);reference=[e.step() for _ in range(2)];end={'model':c.s.cpu_copy(e.model.state_dict()),'optimizer':c.s.cpu_copy(e.opt.state_dict()),'scheduler':copy.deepcopy(e.scheduler.state_dict()),'sampler':copy.deepcopy(e.sampler.state_dict()),'rng':c.s.rng_state()};e.resume(p);actual=[e.step() for _ in range(2)]
 checks={'model':c.same(end['model'],e.model.state_dict()),'optimizer':c.same(end['optimizer'],e.opt.state_dict()),'scheduler':c.same(end['scheduler'],e.scheduler.state_dict()),'sampler':c.same(end['sampler'],e.sampler.state_dict()),'RNG':c.same(end['rng'],c.s.rng_state()),'sample_order':all(a['sample_order_sha256']==b['sample_order_sha256'] for a,b in zip(reference,actual))}
 errors={k:max(abs(a[k]-b[k]) for a,b in zip(reference,actual)) for k in ['AA_NLL_stochastic','local_geometry_NLL','AA_shared_L2','geometry_shared_scale']};write(R/'COMPONENT_RESTART_TEST.json',{'status':'PASS' if all(checks.values()) and max(errors.values())==0 else 'FAIL','checks':checks,'errors':errors,'main_unique_updates':4,'additional_reexecuted_update_calls':2,'actual_optimizer_calls':6,'checkpoint2_sha256':sha(p)})
 if not all(checks.values()) or max(errors.values())!=0:raise ValueError('GPU deterministic restart failed')
 checkpoint(e);return first+actual

def extension(row,history,rows):
 rules=read(R/'configs/FROZEN.json')['extension'];base=history[0];prev=next(v for v in history if v['new_update']==2500);va=row['VALIDATION'];tr=row['TRAIN'];ids=sorted(va['per_identity']);ci=paired_ci([va['per_identity'][h]['AA_NLL']-base['VALIDATION']['per_identity'][h]['AA_NLL'] for h in ids]);cal=read(R/'calibration/VALIDATION_05000.json');bcal=read(R/'calibration/VALIDATION_00000.json');tau=cal['torsion_calibration'];bt=bcal['torsion_calibration'];native=read(c.s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json')['native'];source=read(O/'generation/pilot00500_ca_fp64_interior_v1.json');fallback=read(O/'generation/fallback_ca_fp64_interior_v1.json');ss=metric_summary(rows);bs=metric_summary(source);fails=native_checks(row)+check_candidate(ss,metric_summary(fallback),native)
 def require(ok,text):
  if not ok:fails.append(text)
 require(va['perplexity']<rules['VAL_PPL_max'],'PPL target not reached');require(ci['bootstrap_95pct'][1]<0,'AA paired improvement not supported');require(va['AA_NLL']<prev['VALIDATION']['AA_NLL']-.001,'no useful2500to5000 AA progress');require(va['AA_NLL']-tr['AA_NLL']<=base['VALIDATION']['AA_NLL']-base['TRAIN']['AA_NLL']+.02,'gap grew>.02');require(va['component_NLL'][2]<=base['VALIDATION']['component_NLL'][2]+.02,'torsion mean deterioration');require(all(a<=b+.02 for a,b in zip(va['component_NLL'][:2],base['VALIDATION']['component_NLL'][:2])),'distance/angle deterioration');require(tau['coverage50_80_95']['95']>=bt['coverage50_80_95']['95']-.01,'torsion tail coverage deterioration');require(cal['components']['torsion']['worst_tail']['0.01']['contribution_to_mean']<=1.10*bcal['components']['torsion']['worst_tail']['0.01']['contribution_to_mean']+.01,'torsion tail loss deterioration')
 for N,v in ss.items():require(v['long_contact_fraction']['mean']>=.8*bs[N]['long_contact_fraction']['mean'],N+' contact loss')
 result={'approved_by_frozen_criteria':not fails,'failures':fails,'paired_AA_NLL_change':ci,'criteria_frozen_before_main':True};write(R/'EXTENSION_DECISION.json',result);return not fails

def run():
 if (R/'runs/J1/COMPLETED.json').exists():raise RuntimeError('already stopped; no automatic resume')
 if not (R/'replay/RECONSTRUCTION.json').exists():raise ValueError('replay decision required')
 c.s.j.core.setup();safe();status('DATA_LOADING');data=c.s.PartialData();write(R/'DATA_RECEIPT.json',data.receipt)
 if not (R/'configs/FROZEN.json').exists():
  cfg=frozen();cfg['block_limits']=calibrate(data,cfg);write(R/'configs/FROZEN.json',cfg)
 else:cfg=read(R/'configs/FROZEN.json')
 e=Engine(data,cfg);history=[];recent=sorted((R/'runs/J1').glob('checkpoint_*.pt'));initial=[]
 if recent:e.resume(recent[-1]);history=e.history
 else:
  base=assess(e);old=next(v for v in read(O/'TRAIN_VALIDATION_PERPLEXITY.json') if v['pilot_update']==500)
  if abs(base['VALIDATION']['AA_NLL']-old['VALIDATION']['AA_NLL'])>2e-6 or abs(base['VALIDATION']['geometry_NLL']-old['VALIDATION']['geometry_NLL'])>2e-6:raise ValueError('frozen evaluation baseline parity failed')
  history=[base];e.history=history;write(R/'runs/J1/evaluation_00000.json',base);initial=restart_proof(e)
 e.history=history
 def stopping(*_):STOP.set()
 signal.signal(signal.SIGINT,stopping);signal.signal(signal.SIGTERM,stopping)
 def watch():
  while not STOP.wait(30):
   if time.monotonic()-c.LAST[0]>900:STOP.set();write(R/'WATCHDOG_STOP.json',{'reason':'no progress900seconds'})
 c.LAST[0]=time.monotonic();threading.Thread(target=watch,daemon=True).start();ceiling=10000 if (R/'EXTENSION_DECISION.json').exists() and read(R/'EXTENSION_DECISION.json')['approved_by_frozen_criteria'] else 5000;reason=None;torch.cuda.reset_peak_memory_stats()
 with (R/'runs/J1'/f'updates_{e.update:05d}_{int(time.time())}.jsonl').open('x') as log:
  for v in initial:log.write(json.dumps(v)+'\n')
  while e.update<ceiling and not STOP.is_set():
   v=e.step();log.write(json.dumps(v)+'\n')
   if e.update%25==0:log.flush();status('TRAINING',**v,**safe())
   if e.update in MILESTONES:
    result=assess(e);history.append(result);e.history=history;write(R/'runs/J1'/f'evaluation_{e.update:05d}.json',result);write(R/'TRAIN_VALIDATION_PERPLEXITY.json',history)
    qualifies=not native_checks(result)
    if qualifies and (not e.best or result['VALIDATION']['AA_NLL']<next(x for x in history if x['new_update']==e.best)['VALIDATION']['AA_NLL']):e.best=e.update
    cp=checkpoint(e);rows=generation(e,cp,32 if e.update>=5000 else 4);ok=[x for x in rows if 'metrics' in x]
    if len(ok)!=len(rows) or np.mean([x['metrics']['clashes_per100_residues'] for x in ok])>6:reason='frozen generation finite/severe-clash safety'
    parent=read(c.OLD/'SEQUENCE_BASELINES.json')['J1']['VALIDATION']
    if e.update>=1000 and all(x['VALIDATION']['AA_NLL']>parent['AA_NLL']+.03 or x['VALIDATION']['geometry_NLL']>parent['geometry_NLL']+.20 for x in history[-2:]):reason='two consecutive frozen AA/local safety violations'
    with (R/'EXPERIMENT_DECISIONS.md').open('a') as f:f.write(f"\nMilestone+{e.update}: VAL PPL{result['VALIDATION']['perplexity']}, local{result['VALIDATION']['geometry_NLL']}, native_eligible{qualifies}, stop{reason}. No criteria change.\n")
    if e.update==5000 and not reason:
     if extension(result,history,rows):ceiling=10000
     else:reason='5000 complete; frozen extension criteria not met'
    if reason:break
   elif e.update%250==0:checkpoint(e)
  cp=checkpoint(e);write(R/'runs/J1/COMPLETED.json',{'new_optimizer_updates':e.update,'historical_optimizer_updates_at_start':35500,'historical_replay_updates':500,'restart_reexecuted_updates':2,'total_optimizer_updates':35500+e.update,'latest_checkpoint':str(cp),'latest_sha256':sha(cp),'best_native_update':e.best,'stop_reason':reason or ('signal/watchdog/request' if STOP.is_set() else 'authorized maximum'),'elapsed_seconds_including_evaluation':e.elapsed+time.monotonic()-e.start,'counts_at_start':e.source_counts,'counts_at_end':dict(e.sampler.counts),'coverage_at_end':e.sampler.report(),'nonlocal_targets':e.nonlocal_targets,'differentiated_rollout_residues':0,'resources':safe()});status('TRAINING_STOPPED',update=e.update,reason=reason)
if __name__=='__main__':run()
