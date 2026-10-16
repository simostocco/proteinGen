"""Bounded real-PDB supervised pilot; no TEST evaluation, RL or folding oracle."""
import argparse,collections,copy,gzip,hashlib,json,math,os,resource,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from protein_joint_sac.contracts import AA,GEOMETRY_VERSION,digest
from protein_joint_sac.data.accepted_supervised import admit,AcceptedPopulation,CoordinateReader,CLAIM
from protein_joint_sac.data.pdb_split_proposal import PDBSplitProposal
from protein_joint_sac.data.certified_unified import file_sha
from protein_joint_sac.rl.bc_training import BCConfig,BCTrainer,code_hash,rng_state
from protein_joint_sac.chemistry.reference_adapter import ref,load
from protein_joint_sac.evaluation.bc_metrics import generate
R=Path('/mnt/d/Simone/proteinGen_data/e016_real_pdb_pilots_20261016_v1')
P=Path('/mnt/d/Simone/proteinGen_data/pdb_supervised_split_50id80cov_20261016_v1')
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.part');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(tmp,p)
def guard(extra=64*2**20):
 if shutil.disk_usage(R).free<10*2**30+extra:raise RuntimeError('disk safety reserve')
 size=sum(p.stat().st_size for base in [R/'runs',R/'priors'] for p in base.rglob('*') if p.is_file())
 if size>2*2**30:raise RuntimeError('combined persistent pilot/prior cap')
 return {'free_bytes':shutil.disk_usage(R).free,'pilot_prior_bytes':size,'rss_highwater_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
def rng_digest():
 def convert(x):
  if isinstance(x,torch.Tensor):return x.cpu().tolist()
  if isinstance(x,dict):return {k:convert(v) for k,v in x.items()}
  if isinstance(x,(tuple,list)):return [convert(v) for v in x]
  return x
 return digest(convert(rng_state()))
def band(n):return next(x for x,(a,b) in enumerate([(20,64),(65,128),(129,256),(257,384),(385,500)]) if a<=n<=b)
def internal(x):
 b=np.diff(x,axis=0);d=np.linalg.norm(b,axis=1)
 theta=np.arctan2(np.linalg.norm(np.cross(b[:-1],b[1:]),axis=1),-np.sum(b[:-1]*b[1:],axis=1))
 u=b[1:-1]/d[1:-1,None];v=-b[:-2];v=v-(v*u).sum(1)[:,None]*u;w=b[2:]-(b[2:]*u).sum(1)[:,None]*u
 vn=np.linalg.norm(v,axis=1);wn=np.linalg.norm(w,axis=1)
 v=v/vn[:,None];w=w/wn[:,None];tau=np.arctan2((np.cross(u,v)*w).sum(1),(v*w).sum(1))
 if not np.isfinite(x).all() or np.any(d<=0) or np.any(theta<=0) or np.any(theta>=np.pi) or np.any(abs(tau)>=np.pi) or np.any(vn<1e-10) or np.any(wn<1e-10):raise ValueError('unsupported/degenerate observed geometry')
 return d,theta,tau

def fit():
 start=time.monotonic();guard(512*2**20)
 rec=json.load(open(R/'ADMISSION.json')) if (R/'ADMISSION.json').exists() else admit(P,R/'ADMISSION.json');p=PDBSplitProposal(P)
 if rec['proposal_manifest_sha256']!=file_sha(P/'PROPOSAL_MANIFEST.json'):raise ValueError('admission changed')
 seqs=list(p.iter_sequences('TRAIN'));seqhash=hashlib.sha256('\n'.join(sorted(s['sequence_id'] for s in seqs)).encode()).hexdigest()
 c=collections.Counter();lengths=collections.Counter();bigram=np.zeros((20,20),dtype=np.int64)
 for row in seqs:
  s=row['canonical_full_sequence'];c.update(s);lengths[str(band(len(s)))]+=1
  for a,b in zip(s[:-1],s[1:]):bigram[AA.index(a),AA.index(b)]+=1
 n=sum(c.values());freq={a:c[a]/n for a in AA}
 write(R/'priors/SEQUENCE_STATS.json',{'scope':'TRAIN_ONLY_UNIQUE_FULL_STRINGS','identities':len(seqs),'membership_sha256':seqhash,'residue_count':n,'counts':dict(c),'frequency':freq,'bigram_counts':bigram.tolist(),'bigram_pseudocount':0,'length_bands':dict(lengths),'admission_sha256':file_sha(R/'ADMISSION.json')})
 train_members={s['sequence_id'] for s in seqs};rows=list(p.iter_joint('TRAIN'));mult=collections.Counter(x['sequence_hash'] for x in rows)
 names=['d_A','log_d','theta_rad','logit_theta_over_pi','tau_rad','atanh_tau_over_pi','sin_tau','cos_tau']
 stats={a:{k:np.zeros(4) for k in names} for a in ['GLOBAL',*AA]};extrema={k:[float('inf'),float('-inf')] for k in names};hist={k:np.zeros(128) for k in names};ranges={'d_A':(0.,10.),'log_d':(-4.,4.),'theta_rad':(0.,np.pi),'logit_theta_over_pi':(-12.,12.),'tau_rad':(-np.pi,np.pi),'atanh_tau_over_pi':(-12.,12.),'sin_tau':(-1.,1.),'cos_tau':(-1.,1.)};overflow=collections.Counter();reader=CoordinateReader(p.external)
 positions=collections.Counter();processed=0;fitids=set()
 with (R/'priors/TRAIN_GEOMETRY_INPUTS.jsonl').open('x') as ledger:
  for row,x in reader.stream(rows):
   if row['split']!='TRAIN' or row['role']!='TRAINING' or row['sequence_hash'] not in train_members:raise ValueError('fit membership leakage')
   s=p.sequences[row['sequence_hash']]['canonical_full_sequence']
   if x.shape!=(len(s),3):raise ValueError('full mapped polymer mismatch')
   d,theta,tau=internal(x)
   values=[d,np.log(d),theta,np.log(theta/(np.pi-theta)),tau,np.arctanh(tau/np.pi),np.sin(tau),np.cos(tau)]
   for key,v,offset in zip(names,values,[1,1,2,2,3,3,3,3]):
    w=1/(mult[row['sequence_hash']]*len(v));letters=np.asarray(list(s[offset:]));extrema[key]=[min(extrema[key][0],float(v.min())),max(extrema[key][1],float(v.max()))]
    lo,hi=ranges[key];overflow[key]+=int(((v<lo)|(v>hi)).sum());hist[key]+=np.histogram(v,bins=128,range=(lo,hi))[0]*w;positions[key]+=len(v)
    for a in ['GLOBAL',*AA]:
     z=v if a=='GLOBAL' else v[letters==a];stats[a][key]+=np.array([len(z)*w,z.sum()*w,np.square(z).sum()*w,len(z)])
   ledger.write(json.dumps({k:row[k] for k in ['observation_id','sequence_hash','split','role','coordinate_path','coordinate_row','source_sha']})+'\n');processed+=1;fitids.add(row['sequence_hash'])
   if processed%2000==0:print(json.dumps({'fit_observations':processed,'elapsed_seconds':time.monotonic()-start,**guard()}),flush=True)
 out={a:{k:{'weighted_mass':float(v[0]),'mean':float(v[1]/v[0]) if v[0] else None,'variance':float(max(0,v[2]/v[0]-(v[1]/v[0])**2)) if v[0] else None,'observed_positions':int(v[3]),'valid':bool(v[0])} for k,v in st.items()} for a,st in stats.items()}
 trainids={s['sequence_id'] for s in seqs};heldout={s['sequence_id'] for split in ['VALIDATION','TEST'] for s in p.iter_sequences(split)}
 if fitids&heldout or not fitids<=trainids or len(fitids)!=21475 or processed!=63315:raise ValueError('prior census/membership contradiction')
 payload={'version':'limited-train-geometry-v1','status':'TRAIN_ONLY_FITTED_DESCRIPTIVE','observations':processed,'identities':len(fitids),'heldout_fit_intersection':0,'admission_sha256':file_sha(R/'ADMISSION.json'),'inputs_sha256':file_sha(R/'priors/TRAIN_GEOMETRY_INPUTS.jsonl'),'geometry_version':GEOMETRY_VERSION,'chemistry':ref().__dict__,'allowed_consumers':['head initialization if separately versioned','offline plausibility diagnostics','future explicit geometry regularization'],'head_initialization_applied':False,'weighting':'uniform identity then uniform observation then uniform valid position within each component; residue conditioning renormalizes weighted mass','statistics':out,'extrema':extrema,'histograms':{k:{'range':ranges[k],'bins':v.tolist(),'overflow_positions':overflow[k]} for k,v in hist.items()},'circular_torsion':{'mean_direction_rad':math.atan2(out['GLOBAL']['sin_tau']['mean'],out['GLOBAL']['cos_tau']['mean']),'resultant_length':math.hypot(out['GLOBAL']['sin_tau']['mean'],out['GLOBAL']['cos_tau']['mean'])},'seconds':time.monotonic()-start,'resources':guard(),'source_files_verified':len(reader.checked),'not_claimed':['rotamers','atomic energies','folding stability','universal distance limits']}
 write(R/'priors/GEOMETRY_PRIORS.json',payload);write(R/'PRIOR_FIT_RESULT.json',{k:v for k,v in payload.items() if k not in ['statistics','histograms']});print('PRIORS_PASS',flush=True)

def configurations():
 out={}
 for name,mode,matched,chem in [('S1','sequence',True,False),('S2','sequence',False,False),('J1','joint',True,False),('J2','joint',True,True)]:
  out[name]=BCConfig(mode=mode,width=128,heads=4,layers=4,chemistry=chem,seed=42,batch_size=32,learning_rate=3e-4,weight_decay=.01,clip_norm=1.,scheduler_updates=100,device='cuda',scope='LIMITED_PDB_PILOT',geometry_normalization='valid_component_mean',population_view='MATCHED_JOINT' if matched else 'ALL_SEQUENCES',validation_identity_limit=20,admission_sha256=file_sha(R/'ADMISSION.json'),prior_sha256=file_sha(R/'priors/GEOMETRY_PRIORS.json'),entrypoint_sha256=file_sha(__file__))
 return out

def validation(trainer):
 pop=trainer.population;groups=pop.groups['VALIDATION'];chosen=[]
 for b in range(5):
  chosen.extend(sorted(h for h,ii in groups.items() if band(len(pop.rows[ii[0]]['sequence']))==b)[:4])
 metrics=[];trainer.model.eval();start=time.monotonic()
 with torch.no_grad():
  for h in chosen:
   # Bounded frozen pilot panel, one deterministic observation per identity; not full validation.
   i=groups[h][0];L=len(pop.rows[i]['sequence'])
   for t in sorted(set([0,1,2,L//2,L-1])):
    loss,m=trainer.loss(i,t);metrics.append({'sequence_hash':h,'observation_id':pop.rows[i]['observation_id'],'t':t,'length_band':band(L),'loss':float(loss),**m})
  summary={'aa_nll':float(np.mean([m['aa_nll'] for m in metrics])),'geometry_nll_component_normalized':float(np.mean([m['geometry_loss'] for m in metrics])),'selection_loss':float(np.mean([m['loss'] for m in metrics])),'identities':len(chosen),'available_identities':len(groups),'prefixes':len(metrics),'panel_identity_sha256':hashlib.sha256('\n'.join(chosen).encode()).hexdigest(),'seconds':time.monotonic()-start,'weighting':'uniform panel identity and selected prefix; first observation per identity','scope':'BOUNDED_VALIDATION_PANEL_NOT_FULL_POPULATION','length_bands':dict(collections.Counter(str(m['length_band']) for m in metrics)),'details':metrics}
  summary['aa_perplexity']=math.exp(summary['aa_nll'])
 freq=json.load(open(R/'priors/SEQUENCE_STATS.json'))['frequency']
 summary['TRAIN_full_frequency_baseline_nll']=float(np.mean([-math.log(freq[pop.rows[next(i for i in groups[m['sequence_hash']])]['sequence'][m['t']]]) for m in metrics]))
 return summary

def gradient_diagnostic(trainer):
 pop=trainer.population;i=pop.groups['TRAIN'][sorted(pop.groups['TRAIN'])[0]][0];t=min(12,len(pop.rows[i]['sequence'])-1)
 state,label,g=pop.target(i,t,trainer.config.mode,trainer.config.device);h=trainer.model.encode(state)
 aa=torch.nn.functional.cross_entropy(trainer.model.aa_head(h)[None],torch.tensor([label],device=h.device));params=list(trainer.model.parameters())
 def norm(grads):return math.sqrt(sum(float(x.detach().double().square().sum()) for x in grads if x is not None))
 an=norm(torch.autograd.grad(aa,params,retain_graph=g is not None,allow_unused=True));gn=None;parts=None
 if g is not None:
  dist=trainer.model.geometry_distribution(state,AA[label],h);geom=-dist.log_prob(g)/max(1,int(dist.mask.sum()))
  gn=norm(torch.autograd.grad(geom,params,allow_unused=True))
  u=torch.stack([g[0].log(),torch.logit(g[1]/math.pi),torch.atanh(g[2]/math.pi)])
  parts=(-dist.base.log_prob(u)+dist.log_jacobian(u)).detach().cpu().tolist()
 return {'sequence_hash':pop.rows[i]['sequence_hash'],'prefix_length':t,'AA_gradient_l2':an,'component_normalized_geometry_gradient_l2':gn,'geometry_component_NLL_d_theta_tau':parts,'scope':'ONE_TRAIN_PREFIX_GRADIENT_SCALE_DIAGNOSTIC_NO_OPTIMIZER_STEP'}

def pilot(name,updates):
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU substitute')
 if not 3<=updates<=100:raise ValueError('bounded updates3..100')
 guard(512*2**20);c=configurations()[name];d=R/'runs'/name;d.mkdir(exist_ok=False);start=time.monotonic()
 from dataclasses import asdict
 write(R/'configs'/f'{name}.json',asdict(c));pop=AcceptedPopulation(R/'ADMISSION.json',c.mode,name!='S2');trainer=BCTrainer(c,pop);torch.cuda.reset_peak_memory_stats()
 write(d/'GRADIENT_SCALE.json',gradient_diagnostic(trainer));initial=validation(trainer);write(d/'VALIDATION_INITIAL.json',initial)
 log=[];counts=collections.Counter();identities=set();prefixes=0;token_exposure=0;checkpoint=None;reference=None;resume=None
 with (d/'UPDATES.jsonl').open('x') as f:
  for k in range(updates):
   if time.monotonic()-start>=1750:break
   guard();torch.cuda.synchronize();tick=time.monotonic();m=trainer.step();torch.cuda.synchronize();m['seconds']=time.monotonic()-tick;m['cuda_allocated_peak']=torch.cuda.max_memory_allocated();m['cuda_reserved_peak']=torch.cuda.max_memory_reserved();m.update(guard());m['batch_length_bands']=dict(collections.Counter(str(band(len(pop.rows[i]['sequence']))) for i,t in m['samples']))
   for i,t in m['samples']:counts[str(band(len(pop.rows[i]['sequence'])))]+=1;identities.add(pop.rows[i]['sequence_hash']);prefixes+=1;token_exposure+=t
   f.write(json.dumps(m)+'\n');f.flush();log.append(m);print(json.dumps({'mode':name,**{k:v for k,v in m.items() if k!='samples'}}),flush=True)
   if name=='J2' and k==1:checkpoint=d/'resume_at_2.pt';trainer.checkpoint(checkpoint)
   if name=='J2' and k==2:reference={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()};reference_metric=m;reference_opt=copy.deepcopy(trainer.optimizer.state_dict());reference_scheduler=copy.deepcopy(trainer.scheduler.state_dict());reference_sampler=copy.deepcopy(trainer.sampler.state_dict());reference_rng=rng_digest()
 final=validation(trainer);write(d/'VALIDATION_FINAL.json',final);trainer.checkpoint(d/f'checkpoint_{trainer.update:03}.pt')
 if name=='J2' and reference is not None:
  # Fresh model/optimizer/sampler, no reuse of trainer RNG; actual next stochastic update.
  live=trainer;trainer=BCTrainer(c,pop);trainer.resume(checkpoint);reproduced=trainer.step();diff=max(float((v.cpu()-reference[k]).abs().max()) for k,v in trainer.model.state_dict().items());same=reproduced['samples']==reference_metric['samples'];lossdiff=abs(reproduced['aa_nll']-reference_metric['aa_nll']);opt_error=0.
  for idx,st in reference_opt['state'].items():
   for key,v in st.items():
    z=trainer.optimizer.state_dict()['state'][idx][key]
    if isinstance(v,torch.Tensor):opt_error=max(opt_error,float((v.cpu()-z.cpu()).abs().max()))
  recovery_states=(trainer.scheduler.state_dict()==reference_scheduler and trainer.sampler.state_dict()==reference_sampler and rng_digest()==reference_rng)
  resume={'device':'cuda','checkpoint_update':2,'compared_update':3,'same_stochastic_examples':same,'max_parameter_abs_error':diff,'aa_nll_abs_error':lossdiff,'optimizer_state_max_error':opt_error,'scheduler_sampler_rng_equal':recovery_states,'status':'PASS' if same and diff==0 and lossdiff==0 and opt_error==0 and recovery_states else 'FAIL'};write(d/'RESUME_COMPARISON.json',resume)
  if resume['status']!='PASS':raise RuntimeError('exact real-data restart failed')
  trainer=live
 generation=[]
 for n in [20,64]:
  trainer.model.eval()
  try:generation.append({'N':n,'status':'PASS',**generate(trainer.model,n,c.mode)})
  except (ValueError,RuntimeError,FloatingPointError) as e:
   generation.append({'N':n,'status':'NUMERICAL_OR_GEOMETRY_FAILURE','error':str(e)})
   write(d/'GENERATION_FAILURE.json',generation);raise
 write(d/'GENERATION.json',generation)
 result={'name':name,'status':'REAL_PDB_PILOT_PASS','training_scope':'TRAINED_NOT_CONVERGED','config':asdict(c),'parameters':sum(p.numel() for p in trainer.model.parameters()),'train_identities':len(pop.groups['TRAIN']),'validation_available_identities':len(pop.groups['VALIDATION']),'rows':len(pop.rows),'updates':trainer.update,'effective_batch':32,'live_prefix_microbatch':1,'prefix_decisions':prefixes,'prefix_residues_encoded':token_exposure,'unique_train_identities_exposed':len(identities),'length_band_prefix_exposures':dict(counts),'initial_validation':{k:v for k,v in initial.items() if k!='details'},'final_validation':{k:v for k,v in final.items() if k!='details'},'update_seconds_mean':float(np.mean([m['seconds'] for m in log])),'update_seconds_median':float(np.median([m['seconds'] for m in log])),'prefix_decisions_per_second':prefixes/sum(m['seconds'] for m in log),'prefix_residues_per_second':token_exposure/sum(m['seconds'] for m in log),'first_train_aa_nll':log[0]['aa_nll'],'last_train_aa_nll':log[-1]['aa_nll'],'gradient_norm_max':max(m['grad_norm'] for m in log),'cuda_allocated_peak':max(m['cuda_allocated_peak'] for m in log),'cuda_reserved_peak':max(m['cuda_reserved_peak'] for m in log),'checkpoint_bytes':sum(p.stat().st_size for p in d.glob('*.pt')),'seconds_total':time.monotonic()-start,'resume':resume,'geometry_prior_initialization':'NOT_APPLIED_REFERENCE_RETAINED','chemistry_features':list(load().APPROVED_NAMES) if c.chemistry else [],'code_sha256':code_hash(),'admission_sha256':file_sha(R/'ADMISSION.json'),'priors_sha256':file_sha(R/'priors/GEOMETRY_PRIORS.json'),'resources':guard(),'TEST_metrics_computed':False,'AF_calls':0}
 write(d/'RESULT.json',result);print('PILOT_PASS '+name,flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['fit','pilot']);a.add_argument('--mode',choices=['S1','S2','J1','J2']);a.add_argument('--updates',type=int,default=12);q=a.parse_args()
 if q.stage=='fit':fit()
 else:pilot(q.mode,q.updates)
