"""Single J1 pilot: full TRAIN AA coverage, unchanged independently qualified geometry."""
import collections,copy,gc,json,math,os,random,signal,subprocess,threading,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
import j1_stabilize_train as old
from protein_joint_sac.data.aa_only_supervised import ExpandedData,ExpandedSampler,augment_available
from protein_joint_sac.data.partial_continuation import PartialData
from protein_joint_sac.models.batched_bc import collate,encode_batch,losses,features_batch
from protein_joint_sac.rl.component_clipping import combine,blocks
from protein_joint_sac.models.geometry_inference import corrected_action
from protein_joint_sac.contracts import AA,BuildState,BuildAction
from protein_joint_sac.geometry.ca_builder import append,mask_for_step
R=Path(__file__).resolve().parents[3];D=R.parent;T=D/'e016_j1_torsion_stabilization_20261018_v1';P=D/'pdb_supervised_split_50id80cov_20261016_v1';CENSUS=D/'e016_unused_train_census_20261018_v1';SOURCE=T/'replay/checkpoint_00500_full.pt';SOURCE_SHA='9208263300fb9e88519ff0d40e6d9d29a9744b709d9a6181819cd0894b593000'
c=old.c;s=c.s;write=c.write;read=c.read;sha=c.sha;old.R=R;c.R=R;s.R=R;s.j.R=R;s.j.core.R=R
STOP=threading.Event();MILESTONES=[500,1000,2500,5000]
def status(phase,**kw):c.status(phase,**kw);print(json.dumps({'phase':phase,**kw}),flush=True)
def guard():
 if (R/'STOP_REQUESTED').exists():STOP.set()
 return c.guard(450*2**20)
def initial_config():
 cfg=read(T/'configs/FROZEN.json');cfg.update(version='j1_full_sequence_AA_v1',primary_updates=2500,max_updates=5000,learning_rate=1e-5,terminal_learning_rate=2e-6,warmup_updates=100,decay_end=2500,milestones=MILESTONES,optimizer='RESTORED_35500_ADAMW_MOMENTS_NEW_PHASE_SCHEDULER',AA_mixture={'complete':.4,'partial_only':.3,'unused':.3},AA_quota_cycle=ExpandedSampler.QUOTAS,rollout_period=None,horizons=[],retention='all evaluated full states plus latest 250-step recovery; only superseded new nonmilestone states can be retired')
 cfg['extension']={'free_AA_min_NLL_improvement':.003,'paired_identity_95_upper_below':0.,'native_AA_NLL_max_extra_vs_start':.01,'native_gap_max_extra':.02,'free_gap_max_extra':.02,'local_NLL_max':-.113708968,'component_NLL_max_extra_vs_start':.02,'long_contacts_min_ratio_vs_start':.8,'full32_original_generation_safeguards':True,'require_1000_to2500_free_AA_improvement':.001}
 cfg['early_stop_new']='two consecutive milestone geometry-free VAL NLL > start+.02; retain all original AA/local/severe generation safety rules'
 cfg['geometry_free_evaluation']={'positions_per_identity':4,'selection_seed':2026101842,'TRAIN_matched_to_VAL':'exact category-by-length-band identity counts, uniform identities within cells','scoring':'identity mean of four prefix CE; exp(mean identity NLL)','geometry':'all unavailable, no observation access'}
 cfg['promotion'].update(free_AA_min_NLL_improvement=.003,free_gap_extra_max=.02)
 return cfg

def prepare(data):
 if sha(SOURCE)!=SOURCE_SHA:raise ValueError('source checkpoint hash mismatch')
 seqs=data.proposal.sequences;cats=data.populations
 for role in ['TRAIN','VALIDATION']:
  sets=[set(v) for v in cats[role].values()];assert set.union(*sets)==set(data.base.ids[role]);assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
  for h in set.union(*sets):
   r=seqs[h];assert r['split']==role and r['canonical_full_sequence']==data.seqs[h] and r['length']==len(data.seqs[h]) and set(data.seqs[h])<=set(AA)
 report={'status':'PASS','source_checkpoint':str(SOURCE),'source_sha256':SOURCE_SHA,'TRAIN_populations':{k:len(v) for k,v in cats['TRAIN'].items()},'VALIDATION_populations':{k:len(v) for k,v in cats['VALIDATION'].items()},'unused_AA_labels':sum(len(data.seqs[h]) for h in cats['TRAIN']['unused']),'split_manifest_sha256':sha(P/'PROPOSAL_MANIFEST.json'),'sequence_assignments_sha256':sha(P/'SEQUENCE_ASSIGNMENTS.parquet'),'partial_view_sha256':sha(s.j.OLD/'view/VIEW_FROZEN.json'),'census_seal_sha256':sha(CENSUS/'PROVENANCE_SHA256.json'),'unused_coordinates_loaded':0,'scientific_split_changed':False,'withheld_geometry':'Structural records exist; withheld because structural qualification failed or was not established.'}
 write(R/'ALL_TRAIN_IDENTITY_ADMISSION.json',report)
 cfgpath=R/'configs/FROZEN.json'
 if not cfgpath.exists():write(cfgpath,initial_config())
 rng=random.Random(2026101842);train_ids=[]
 for kind in ['complete','partial_only','unused']:
  for band in range(5):
   val=[h for h in cats['VALIDATION'][kind] if s.j.band(len(data.seqs[h]))==band];tr=[h for h in cats['TRAIN'][kind] if s.j.band(len(data.seqs[h]))==band];rng.shuffle(tr)
   if len(tr)<len(val):raise ValueError('matched TRAIN cell too small')
   train_ids.extend(tr[:len(val)])
 def panel(ids):
  out=[]
  for h in sorted(ids):
   n=len(data.seqs[h]);r=random.Random(int(s.j.digest(['free_panel_v1',h]),16));pos=[r.randrange(k*n//4,(k+1)*n//4) for k in range(4)];assert len(set(pos))==4;out.extend((h,None,t) for t in pos)
  return out
 panels={'TRAIN':panel(train_ids),'VALIDATION':panel(data.base.ids['VALIDATION'])}
 assert [len(panels[k]) for k in panels]==[59100,59100]
 fp=R/'configs/GEOMETRY_FREE_PANELS.json'
 if fp.exists():assert read(fp)==json.loads(json.dumps(panels))
 else:write(fp,panels)
 # Two complete validation identities per length band; all their true-AA-conditioned geometry is generated from BOS.
 robust=[]
 for band in range(5):robust.extend([h for h in sorted(data.base.jointids['VALIDATION']) if s.j.band(len(data.seqs[h]))==band][:2])
 write(R/'configs/GENERATED_PREFIX_PANEL.json',{'identities':robust,'seed':2026101881,'geometry':'sampled corrected FP64 from BOS; true AA introduced only after each prediction; all future native coordinates withheld','metrics':'all next-AA targets after generated-prefix accumulation, separate from standard panels'})
 receipt={'config_sha256':sha(cfgpath),'free_panels_sha256':sha(fp),'native_panels_sha256':sha(R/'configs/EVALUATION_PANELS.json'),'before_first_new_geometry_free_VALIDATION':not (R/'runs/J1/evaluation_00000.json').exists(),'extension_rules':read(cfgpath)['extension']}
 if (R/'DECISIONS_FROZEN_BEFORE_BASELINE.json').exists():
  prior=read(R/'DECISIONS_FROZEN_BEFORE_BASELINE.json');assert prior['config_sha256']==receipt['config_sha256'] and prior['free_panels_sha256']==receipt['free_panels_sha256']
 else:write(R/'DECISIONS_FROZEN_BEFORE_BASELINE.json',receipt)

class Engine(c.Engine):
 def __init__(self,data,cfg):
  torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
  self.data=data;self.c=cfg;self.name='J1';self.joint=True;self.model=s.j.core.model('J1');s.set_dropout(self.model,.15);state=s.load_checked(SOURCE)
  if sha(SOURCE)!=SOURCE_SHA or {float(v['step']) for v in state['optimizer']['state'].values()}!={35500.}:raise ValueError('selected full source state mismatch')
  self.model.load_state_dict(state['model'],strict=True);assert sum(p.numel() for p in self.model.parameters())==12351746
  self.opt=torch.optim.AdamW(self.model.parameters(),lr=cfg['learning_rate'],weight_decay=.01,fused=True);self.opt.load_state_dict(state['optimizer'])
  for pg in self.opt.param_groups:pg['lr']=cfg['learning_rate'];pg['initial_lr']=cfg['learning_rate']
  def factor(k):
   if k<100:return (k+1)/100
   return .2+.8*.5*(1+math.cos(math.pi*min(1.,(k-100)/2400)))
  self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.opt,factor);self.sampler=ExpandedSampler(data,state['sampler']);self.rollrng=random.Random();self.rollrng.setstate(state['rollout_rng']);self.update=0;self.nonlocal_targets=0;self.history=[];self.best=0;self.elapsed=0.;self.start=time.monotonic()
  self.lineage={'version':'j1_full_sequence_AA_v1','config':cfg,'source_checkpoint':str(SOURCE),'source_sha256':SOURCE_SHA,'historical_updates_at_start':35500,'parent_lineage':state['lineage'],'historical_scheduler':state['scheduler'],'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip(),'package_sha256':s.code_hash(),'entrypoint_sha256':sha(__file__),'admission_sha256':sha(R/'ALL_TRAIN_IDENTITY_ADMISSION.json'),'native_panel_sha256':sha(R/'configs/EVALUATION_PANELS.json'),'free_panel_sha256':sha(R/'configs/GEOMETRY_FREE_PANELS.json'),'sampler_initialization':'new category RNG/orders; original identity-position coverage retained; no state invented','inference_protocol':'ca_fp64_interior_v1','TEST':'NOT_USED','normalization':'UNCHANGED'}
  s.restore_rng(state['rng']);del state
 def backward(self):
  self.model.train();self.opt.zero_grad(set_to_none=True);aaex,geoex=self.sampler.draw();av=gv=0.;cs=np.zeros(3)
  for off in range(0,64,16):
   b,y,g=self.data.batch(aaex[off:off+16]);b=augment_available(b,.075)
   with torch.autocast('cuda',dtype=torch.bfloat16):h=encode_batch(self.model,b);a=F.cross_entropy(self.model.aa_head(h).float(),y,reduction='sum')/64
   a.backward();av+=float(a.detach());del a,h,b,y,g
  aa={n:None if p.grad is None else p.grad.detach().clone() for n,p in self.model.named_parameters()};self.opt.zero_grad(set_to_none=True)
  for off in range(0,32,16):
   b,y,g=self.data.batch(geoex[off:off+16]);b=augment_available(b,.075);a,local,parts=s.loss_call(self.model,b,y,g);(self.c['lambda_local']*local.sum()/32).backward();gv+=float(local.detach().sum())/32;cs+=parts.detach().sum(0).cpu().numpy();del a,local,parts,b,y,g
  extra={}
  if (self.update+1)%2==0:
   ex=geoex[-4:];b,y,g=self.data.batch(ex);b=augment_available(b,.075);targets=torch.tensor(np.asarray([self.data.base.ca[o][t] for h,o,t in ex]),device='cuda');nl,counts=s.direct_nonlocal(self.model,b,y,targets,K=4,seed=180042);(self.c['lambda_nonlocal']*nl).backward();self.nonlocal_targets+=len(counts);extra={'nonlocal_energy_score':float(nl.detach()),'nonlocal_targets':len(counts)};del nl,targets,b,y,g
  return aa,{'AA_NLL_stochastic':av,'local_geometry_NLL':gv,'component_mean_NLL':(cs/32).tolist(),'sample_order_sha256':s.j.digest([aaex,geoex]),**extra}
 def step(self):
  guard();start=time.monotonic();lr=self.opt.param_groups[0]['lr'];aa,out=self.backward();stats=combine(self.model,aa,self.c['block_limits']);del aa;self.opt.step();self.scheduler.step();self.update+=1
  if not all(bool(torch.isfinite(p).all()) for p in self.model.parameters()) or not c.finite_state(self.opt):raise FloatingPointError('nonfinite update or AdamW moment')
  torch.cuda.synchronize();c.LAST[0]=time.monotonic();return {'new_update':self.update,'total_optimizer_updates':35500+self.update,'seconds':time.monotonic()-start,'learning_rate':lr,'nonlocal_targets_total':self.nonlocal_targets,'differentiated_rollout_residues':0,**out,**stats}
 def checkpoint(self,path):
  guard();payload={'lineage':self.lineage,'model':self.model.state_dict(),'optimizer':self.opt.state_dict(),'scheduler':self.scheduler.state_dict(),'rng':s.rng_state(),'sampler':self.sampler.state_dict(),'rollout_rng':self.rollrng.getstate(),'new_update':self.update,'total_optimizer_updates':35500+self.update,'nonlocal_targets':self.nonlocal_targets,'history':self.history,'best':self.best,'elapsed':self.elapsed+time.monotonic()-self.start}
  if Path(path).exists():
   st=s.load_checked(path)
   if st['new_update']!=self.update or not c.same(st['model'],self.model.state_dict()):raise ValueError('checkpoint name collision')
   return sha(path)
  digest=s.save_atomic(path,payload);st=s.load_checked(path)
  if not c.same(st['model'],self.model.state_dict()) or st['new_update']!=self.update:raise ValueError('atomic checkpoint mismatch')
  return digest
 def resume(self,path):
  st=s.load_checked(path)
  if st['lineage']!=self.lineage:raise ValueError('lineage changed')
  if st['scheduler']['last_epoch']!=st['new_update'] or st['sampler']['cursor']!=st['new_update'] or {float(v['step']) for v in st['optimizer']['state'].values()}!={35500.+st['new_update']}:raise ValueError('recovery counters inconsistent')
  self.model.load_state_dict(st['model']);self.opt.load_state_dict(st['optimizer']);self.scheduler.load_state_dict(st['scheduler']);self.sampler.load_state_dict(st['sampler']);self.rollrng.setstate(st['rollout_rng']);self.update=st['new_update'];self.nonlocal_targets=st['nonlocal_targets'];self.history=st['history'];self.best=st['best'];self.elapsed=st['elapsed'];self.start=time.monotonic();s.restore_rng(st['rng'])

@torch.no_grad()
def free_eval(e,panel,role):
 e.model.eval();ordered=sorted(panel,key=lambda v:(v[2],v[0]));by=collections.defaultdict(list);meta={h:k for k,ids in e.data.populations[role].items() for h in ids};start=time.monotonic()
 for off in range(0,len(ordered),16):
  ex=ordered[off:off+16]
  if any(e.data.proposal.sequences[h]['split']!=role for h,o,t in ex):raise ValueError('free eval membership mismatch')
  b,y,_=e.data.aa_only_batch(ex)
  with torch.autocast('cuda',dtype=torch.bfloat16):h=encode_batch(e.model,b);logits=e.model.aa_head(h).float()
  ce=F.cross_entropy(logits,y,reduction='none');ok=logits.topk(3,dim=-1).indices.eq(y[:,None])
  for (identity,o,t),v,cor in zip(ex,ce.cpu().tolist(),ok.cpu().tolist()):by[identity].append([v,float(cor[0]),float(any(cor)),t])
  c.LAST[0]=time.monotonic()
 def summarize(v):
  a=np.array(v);z=a.mean(0);return {'decisions':len(v),'AA_NLL':float(z[0]),'perplexity':math.exp(float(z[0])),'top1':float(z[1]),'top3':float(z[2])}
 per={h:summarize(v) for h,v in by.items()};means=[(h,np.mean(v,axis=0)) for h,v in by.items()];result=summarize([x for h,x in means]);result['decisions']=len(panel);result.update(identities=len(by),role=role,geometry='UNAVAILABLE_ALL_MASKS_FALSE',per_identity=per,categories={k:summarize([v for h,v in means if meta[h]==k])|{'identities':sum(meta[h]==k for h,v in means)} for k in ['complete','partial_only','unused']},length_bands={str(k):summarize([v for h,v in means if s.j.band(len(e.data.seqs[h]))==k]) for k in range(5)},prefix_deciles={str(k):summarize([v for h,vv in by.items() for v in vv if min(9,int(10*v[3]/len(e.data.seqs[h])))==k]) for k in range(10)},seconds=time.monotonic()-start)
 return result

@torch.no_grad()
def robust_eval(e):
 e.model.eval();panel=read(R/'configs/GENERATED_PREFIX_PANEL.json');ids=panel['identities'];states=[BuildState.empty(len(e.data.seqs[h])) for h in ids];rngs=[np.random.default_rng(panel['seed']+k) for k in range(len(ids))];rows={h:[] for h in ids};fail=[];start=time.monotonic()
 for t in range(max(len(e.data.seqs[h]) for h in ids)):
  ix=[k for k,h in enumerate(ids) if states[k].t==t and t<len(e.data.seqs[h]) and h not in fail]
  if not ix:continue
  b=collate([states[k] for k in ix],'cuda');labels=torch.tensor([AA.index(e.data.seqs[ids[k]][t]) for k in ix],device='cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):h=encode_batch(e.model,b);logits=e.model.aa_head(h).float()
  from protein_joint_sac.rl.sequence_first import geometry_parameters
  p=geometry_parameters(e.model,h,labels).cpu();ce=F.cross_entropy(logits,labels,reduction='none').cpu().tolist()
  for j,k in enumerate(ix):
   identity=ids[k];rows[identity].append(ce[j]);noise=torch.tensor(rngs[k].standard_normal(3))
   try:act,_=corrected_action(p[j,:3],p[j,3:],mask_for_step(t),noise);states[k]=append(states[k],BuildAction(e.data.seqs[identity][t],act,mask_for_step(t)))
   except (ValueError,FloatingPointError):fail.append(identity)
  c.LAST[0]=time.monotonic()
 vals={h:float(np.mean(v)) for h,v in rows.items() if v};nll=float(np.mean(list(vals.values())));return {'identities':len(ids),'decisions':sum(map(len,rows.values())),'AA_NLL':nll,'perplexity':math.exp(nll),'per_identity':vals,'failures':fail,'protocol':panel,'seconds':time.monotonic()-start,'label':'TRUE-AA-CONDITIONED GENERATED GEOMETRY, NOT NATIVE-PREFIX PPL OR FREE JOINT GENERATION'}

def assess(e):
 rng=s.rng_state();start=time.monotonic()
 with s.historical_evaluation():
  native=read(R/'configs/EVALUATION_PANELS.json');free=read(R/'configs/GEOMETRY_FREE_PANELS.json');tr=old.complete_eval(e,native['TRAIN'],'TRAIN');status('EVALUATION_NATIVE_TRAIN_DONE',update=e.update);va=old.complete_eval(e,native['VALIDATION'],'VALIDATION');status('EVALUATION_NATIVE_VALIDATION_DONE',update=e.update);ft=free_eval(e,free['TRAIN'],'TRAIN');status('EVALUATION_FREE_TRAIN_DONE',update=e.update);fv=free_eval(e,free['VALIDATION'],'VALIDATION');status('EVALUATION_FREE_VALIDATION_DONE',update=e.update);robust=robust_eval(e)
 s.restore_rng(rng);return {'new_update':e.update,'NATIVE_TRAIN':tr,'NATIVE_VALIDATION':va,'FREE_TRAIN':ft,'FREE_VALIDATION':fv,'GENERATED_PREFIX':robust,'native_gap':va['AA_NLL']-tr['AA_NLL'],'free_gap':fv['AA_NLL']-ft['AA_NLL'],'seconds':time.monotonic()-start}

def mask_tests(e):
 rng=s.rng_state();sampler=copy.deepcopy(e.sampler.state_dict());m=e.model;m.eval();d=e.data;ex=[]
 for k in ['complete','partial_only','unused']:
  for h in d.populations['TRAIN'][k][:4]:o=None if k=='unused' else (d.owners['TRAIN'][h][0] if k=='partial_only' else d.base.owners[h][0]);ex.append((h,o,min(32,len(d.seqs[h])-1)))
 b,y,g=d.batch(ex);existing=[v for v in ex if v[1] is not None];oldb,oldy,oldg=PartialData.batch(d,existing);newb,newy,newg=d.batch(existing);assert c.same(oldb,newb) and torch.equal(oldy,newy) and torch.equal(oldg,newg)
 only=[v for v in ex if v[1] is None];a,labels,geom=d.aa_only_batch(only);a['x'].fill_(1e8);_,pair,present=features_batch(a,False,False);assert not pair[1].any() and present.any() and not a['bond_valid'].any()
 with torch.autocast('cuda',dtype=torch.bfloat16):log1=m.aa_head(encode_batch(m,a));aa,gg,pp=losses(m,a,labels,geom,geometry_fp32=True)
 assert not gg.any() and not pp.any();m.zero_grad(set_to_none=True);aa.mean().backward();AAshared=math.sqrt(sum(float(p.grad.double().square().sum()) for n,p in m.named_parameters() if n.startswith('encoder.') and p.grad is not None));AAhead=math.sqrt(sum(float(p.grad.double().square().sum()) for n,p in m.named_parameters() if n.startswith('aa_head.') and p.grad is not None));Gnorm=sum(float(p.grad.abs().sum()) for n,p in m.named_parameters() if n.startswith('geom.') and p.grad is not None);assert AAshared>0 and AAhead>0 and Gnorm==0
 m.zero_grad(set_to_none=True);a['x'].fill_(float('nan'))
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):log2=m.aa_head(encode_batch(m,a))
 assert torch.equal(log1,log2) and torch.isfinite(log2).all()
 # Adversarial future-label test: a genuine suffix change cannot enter the AA-only encoder.
 source_sequence=d.seqs[only[0][0]];hh,oo,tt=only[0]
 d.seqs[hh]=source_sequence[:tt]+''.join(AA[(AA.index(ch)+1)%20] for ch in source_sequence[tt:])
 future_b,_,_=d.aa_only_batch(only);d.seqs[hh]=source_sequence
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):future_logits=m.aa_head(encode_batch(m,future_b))
 assert torch.equal(log2,future_logits)
 # Native future-coordinate changes remain outside a complete literal prefix as well.
 hh,oo,tt=existing[0];saved=d.base.ca[oo][tt:].copy();d.base.ca[oo][tt:]=1e9
 future_complete,_,_=d.batch(existing);d.base.ca[oo][tt:]=saved
 assert torch.equal(future_complete['x'],newb['x'])
 with torch.autocast('cuda',dtype=torch.bfloat16):ma,mg,mp=losses(m,b,y,g,geometry_fp32=True)
 (ma.mean()+mg.mean()).backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());assert not mg[-4:].any();m.zero_grad(set_to_none=True)
 # New sampler exact cycle quotas and restart, with no optimizer calls.
 first=[e.sampler.draw() for _ in range(2)];st=copy.deepcopy(e.sampler.state_dict());expected=[e.sampler.draw() for _ in range(3)];report=e.sampler.report();assert [report['counts'][k] for k in ['complete','partial_only','unused']]==[128,96,96];e.sampler.load_state_dict(st);actual=[e.sampler.draw() for _ in range(3)];assert actual==expected
 e.sampler.load_state_dict(sampler);s.restore_rng(rng)
 write(R/'AA_ONLY_INPUT_MASK_TESTS.json',{'status':'PASS','actual_GPU':'RTX5060','checkpoint_parameters':sum(p.numel() for p in m.parameters()),'AA_only_shared_gradient_L2':AAshared,'AA_head_gradient_L2':AAhead,'geometry_head_gradient_L1_from_AA_only':Gnorm,'masked_placeholder_NaN_logits_exact':True,'mixed_batch_forward_backward_finite':True,'AA_only_geometry_loss_zero':True,'existing_complete_partial_batch_exact':True,'sampler_resume_exact':True,'five_update_quota':[128,96,96],'optimizer_calls':0,'future_AA_suffix_mutation_exact':True,'future_native_coordinate_suffix_excluded':True,'future_isolation':'Only canonical[:t] is supplied by Data.target(joint=False); no unused coordinate record loaded; N is declared input','prior_CPU_25_checks':str(CENSUS/'AA_ONLY_FORWARD_SMOKE_TEST.json')})

def checkpoint(e):
 cp=R/'runs/J1'/f'checkpoint_{e.update:05d}.pt';e.checkpoint(cp)
 for p in (R/'runs/J1').glob('checkpoint_*.pt'):
  u=int(p.stem.split('_')[-1])
  if u in MILESTONES or u in [0,2,4,e.update,e.best]:continue
  h=sha(p);s.load_checked(p)
  with (R/'NEW_RETENTION.jsonl').open('a') as f:f.write(json.dumps({'path':str(p),'sha256':h,'reason':'verified superseded NEW nonmilestone checkpoint','successor':str(cp)})+'\n')
  p.unlink();p.with_name(p.name+'.sha256').unlink()
 return cp

def restart_proof(e):
 a=[e.step() for _ in range(2)];p=checkpoint(e);ref=[e.step() for _ in range(2)];st={'model':s.cpu_copy(e.model.state_dict()),'optimizer':s.cpu_copy(e.opt.state_dict()),'scheduler':copy.deepcopy(e.scheduler.state_dict()),'sampler':copy.deepcopy(e.sampler.state_dict()),'rng':s.rng_state()};e.resume(p);actual=[e.step() for _ in range(2)];checks={'model':c.same(st['model'],e.model.state_dict()),'optimizer':c.same(st['optimizer'],e.opt.state_dict()),'scheduler':c.same(st['scheduler'],e.scheduler.state_dict()),'sampler':c.same(st['sampler'],e.sampler.state_dict()),'RNG':c.same(st['rng'],s.rng_state()),'order':all(x['sample_order_sha256']==y['sample_order_sha256'] for x,y in zip(ref,actual))}
 if not all(checks.values()):raise ValueError('GPU interruption/resume failed')
 write(R/'GPU_RESTART_TEST.json',{'status':'PASS','checks':checks,'main_unique_updates':4,'reexecuted_proof_updates':2,'physical_optimizer_calls':6,'checkpoint2_sha256':sha(p)});checkpoint(e);return a+actual

def generation(e,cp,n):return old.generation(e,cp,n)
def decision(row,base,previous=None,full_generation=None):
 cfg=read(R/'configs/FROZEN.json');v=row['FREE_VALIDATION'];b=base['FREE_VALIDATION'];keys=sorted(v['per_identity']);ci=old.paired_ci([v['per_identity'][h]['AA_NLL']-b['per_identity'][h]['AA_NLL'] for h in keys]);rules=cfg['extension'];fail=[]
 def req(ok,why):
  if not ok:fail.append(why)
 req(v['AA_NLL']<=b['AA_NLL']-.003,'geometry-free AA gain<.003');req(ci['bootstrap_95pct'][1]<0,'geometry-free paired95 does not support gain');req(row['NATIVE_VALIDATION']['AA_NLL']<=base['NATIVE_VALIDATION']['AA_NLL']+.01,'native AA deterioration>.01');req(row['free_gap']<=base['free_gap']+.02,'geometry-free gap increased>.02');req(row['native_gap']<=base['native_gap']+.02,'native gap increased>.02');req(row['NATIVE_VALIDATION']['geometry_NLL']<=-.113708968,'frozen local geometry failure');req(all(a<=b+.02 for a,b in zip(row['NATIVE_VALIDATION']['component_NLL'],base['NATIVE_VALIDATION']['component_NLL'])),'local component deterioration>.02')
 if previous:req(v['AA_NLL']<=previous['FREE_VALIDATION']['AA_NLL']-.001,'no useful1000to2500 free-AA gain')
 if full_generation:
  native=read(s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json')['native'];fallback=read(c.O/'generation/fallback_ca_fp64_interior_v1.json') if hasattr(c,'O') else read(c.R.parent/'e016_j1_clip_precision_pilot_20261018_v1/generation/fallback_ca_fp64_interior_v1.json');source=read(c.R.parent/'e016_j1_clip_precision_pilot_20261018_v1/generation/pilot00500_ca_fp64_interior_v1.json');ss=old.metric_summary(full_generation);fail.extend(old.check_candidate(ss,old.metric_summary(fallback),native));bs=old.metric_summary(source)
  for n,z in ss.items():req(z['long_contact_fraction']['mean']>=.8*bs[n]['long_contact_fraction']['mean'],n+' contacts below.8parent')
 return {'passes':not fail,'failures':fail,'paired_geometry_free_AA_NLL_change':ci}

def run(stage='all'):
 s.j.core.setup();guard();status('DATA_LOADING');data=ExpandedData();write(R/'DATA_RECEIPT.json',data.receipt);prepare(data);cfg=read(R/'configs/FROZEN.json');e=Engine(data,cfg)
 if stage=='tests':mask_tests(e);return
 if (R/'runs/J1/COMPLETED.json').exists():raise ValueError('already stopped: no automatic resume')
 if stage=='baseline' or not (R/'AA_ONLY_INPUT_MASK_TESTS.json').exists():mask_tests(e)
 history=[];initial=[];paths=sorted((R/'runs/J1').glob('checkpoint_*.pt'))
 if paths:e.resume(paths[-1]);history=e.history
 else:
  baseline=assess(e);oldbase=next(v for v in read(c.R.parent/'e016_j1_clip_precision_pilot_20261018_v1/TRAIN_VALIDATION_PERPLEXITY.json') if v['pilot_update']==500)
  if abs(baseline['NATIVE_VALIDATION']['AA_NLL']-oldbase['VALIDATION']['AA_NLL'])>2e-6 or abs(baseline['NATIVE_VALIDATION']['geometry_NLL']-oldbase['VALIDATION']['geometry_NLL'])>2e-6:raise ValueError('native baseline parity failed')
  history=[baseline];e.history=history;write(R/'runs/J1/evaluation_00000.json',baseline);checkpoint(e)
 if stage=='baseline':return
 if e.update==0:initial=restart_proof(e)
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:STOP.set())
 def watch():
  while not STOP.wait(30):
   if time.monotonic()-c.LAST[0]>1200:STOP.set();write(R/'WATCHDOG_STOP.json',{'no_progress_seconds':1200})
 c.LAST[0]=time.monotonic();threading.Thread(target=watch,daemon=True).start();ceiling=5000 if (R/'EXTENSION_DECISION.json').exists() and read(R/'EXTENSION_DECISION.json')['passes'] else 2500;reason=None;torch.cuda.reset_peak_memory_stats()
 with (R/'runs/J1'/f'updates_{e.update:05d}_{int(time.time())}.jsonl').open('x') as log:
  for row in initial:log.write(json.dumps(row)+'\n')
  while e.update<ceiling and not STOP.is_set():
   row=e.step();log.write(json.dumps(row)+'\n')
   if e.update%25==0:log.flush();status('TRAINING',**row,**guard())
   if e.update in MILESTONES:
    result=assess(e);history.append(result);e.history=history;write(R/'runs/J1'/f'evaluation_{e.update:05d}.json',result);write(R/'SEQUENCE_TRAIN_VALIDATION_METRICS.json',history)
    eligible=decision(result,history[0])
    if eligible['passes'] and (not e.best or result['FREE_VALIDATION']['AA_NLL']<next(v for v in history if v['new_update']==e.best)['FREE_VALIDATION']['AA_NLL']):e.best=e.update
    cp=checkpoint(e);gen=generation(e,cp,32 if e.update>=2500 else 4);ok=[r for r in gen if 'metrics' in r]
    if len(ok)!=len(gen) or np.mean([r['metrics']['clashes_per100_residues'] for r in ok])>6:reason='frozen finite/completion/severe-clash safety'
    parent=read(c.OLD/'SEQUENCE_BASELINES.json')['J1']['VALIDATION']
    if e.update>=1000 and all(r['NATIVE_VALIDATION']['AA_NLL']>parent['AA_NLL']+.03 or r['NATIVE_VALIDATION']['geometry_NLL']>parent['geometry_NLL']+.2 for r in history[-2:]):reason='two consecutive original AA/local safety violations'
    if e.update>=1000 and all(r['FREE_VALIDATION']['AA_NLL']>history[0]['FREE_VALIDATION']['AA_NLL']+.02 for r in history[-2:]):reason='two consecutive geometry-free AA regressions'
    if e.update==2500 and not reason:
     ext=decision(result,history[0],next(r for r in history if r['new_update']==1000),gen);write(R/'EXTENSION_DECISION.json',ext)
     if ext['passes']:ceiling=5000
     else:reason='initial2500 complete; predefined extension criteria not met'
    with (R/'EXPERIMENT_DECISIONS.md').open('a') as f:f.write(f"\nMilestone+{e.update}: nativePPL {result['NATIVE_VALIDATION']['perplexity']}, freePPL {result['FREE_VALIDATION']['perplexity']}, local {result['NATIVE_VALIDATION']['geometry_NLL']}; stop {reason}.\n")
    if reason:break
   elif e.update%250==0:checkpoint(e)
  cp=checkpoint(e);write(R/'runs/J1/COMPLETED.json',{'new_optimizer_updates':e.update,'historical_optimizer_updates_at_start':35500,'total_optimizer_updates':35500+e.update,'proof_reexecuted_updates':2,'latest_checkpoint':str(cp),'latest_sha256':sha(cp),'best_native_and_free_provisional_update':e.best,'stop_reason':reason or ('signal/watchdog' if STOP.is_set() else 'authorized maximum'),'elapsed_seconds_including_evaluation':e.elapsed+time.monotonic()-e.start,'sampler':e.sampler.report(),'nonlocal_targets':e.nonlocal_targets,'differentiated_rollout_residues':0,'resources':guard()});status('TRAINING_STOPPED',update=e.update,reason=reason)
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('stage',nargs='?',default='all',choices=['tests','baseline','all']);run(a.parse_args().stage)
