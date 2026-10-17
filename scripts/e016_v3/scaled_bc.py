"""10M+ scratch PDB BC, bounded profiling and durable milestone controller. No RL/oracle."""
import argparse,collections,copy,hashlib,json,math,os,random,resource,shutil,signal,subprocess,threading,time
from contextlib import nullcontext
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from protein_joint_sac.contracts import AA,BuildState,BuildAction,GeneratedProtein,GEOMETRY_VERSION,digest
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.sequence_only_actor import SequenceOnlyActor
from protein_joint_sac.models.batched_bc import collate,losses,encode_batch,features_batch
from protein_joint_sac.models.hybrid_distributions import GeometryDistribution
from protein_joint_sac.geometry.ca_builder import append,mask_for_step
from protein_joint_sac.evaluation.bc_metrics import trace_metrics,sequence_diversity
from protein_joint_sac.data.scaled_supervised import Data,Sampler,P,PILOT,internal,canonical
from protein_joint_sac.data.certified_unified import file_sha,canonical_teacher_prefix
from protein_joint_sac.chemistry.reference_adapter import ref,load
from protein_joint_sac.rl.bc_training import rng_state,restore_rng,code_hash
from protein_joint_sac.rl.checkpointing import save_atomic,load_checked
R=Path('/mnt/d/Simone/proteinGen_data/e016_scaled10m_training_20261016_v1')
MILESTONES=[250,1000,2000,5000,7500,10000]
STOP=threading.Event()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);part=path.with_name(path.name+'.part');part.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(part,path)
def size(path):return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
def guard(staging=400*2**20):
 free=shutil.disk_usage(R).free
 if free<10*2**30+staging:raise RuntimeError('disk safety reserve plus staging insufficient')
 if Path('/proc/pressure/memory').exists():
  pressure=Path('/proc/pressure/memory').read_text()
  if float(pressure.split('avg10=')[1].split()[0])>20:raise RuntimeError('severe host memory pressure')
 return {'D_free_bytes':free,'RSS_highwater_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,'reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None}
def gpu():
 return subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
def setup():
 torch.set_num_threads(6);torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
def model(name,device='cuda'):
 torch.manual_seed(42);c=ModelConfig(width=352,heads=8,layers=8,ffn_mult=4,chemistry=name=='J2',backend='sdpa');m=JointActor(c) if name.startswith('J') else SequenceOnlyActor(c)
 if not name.startswith('J'):
  for p in m.encoder.pair.parameters():p.requires_grad_(False)
 n=sum(p.numel() for p in m.parameters() if p.requires_grad)
 if not 10000000<=n<=15000000:raise RuntimeError('10-15M independently trainable parameter requirement')
 return m.to(device)
def make(data,examples,joint):
 states=[];labels=[];g=[]
 for h,o,t in examples:
  s,a,v=data.target(h,o,t,joint);states.append(s);labels.append(a)
  if joint:g.append(v)
 return collate(states,'cuda'),torch.tensor(labels,device='cuda'),torch.stack(g).cuda() if joint else None
def autocast(precision):return torch.autocast('cuda',dtype=torch.bfloat16) if precision=='BF16' else nullcontext()
def loss_call(m,b,y,g,c):
 with autocast(c['precision']):aa,geom,components=losses(m,b,y,g,c.get('activation_checkpoint',False),geometry_fp32=c.get('geometry_projection_fp32',False))
 return aa+geom,aa,geom,components

def preflight():
 setup();guard(5*2**30);counts={}
 for name in ['S1','S2','J1','J2']:
  m=model(name);counts[name]={'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'stored_parameters':sum(p.numel() for p in m.parameters()),'architecture':asdict(m.encoder.config)};del m
 write(R/'MODEL_PARAMETER_COUNTS.json',counts)
 report={};data=Data(True,report);write(R/'preflight/DATA_INGESTION.json',report)
 # Fast gauge parity checked across actual sources, bootstrap and suffix boundaries.
 checks=[]
 for h in data.jointids['TRAIN'][:8]:
  o=data.owners[h][0];x=data.ca[o];seq=data.seqs[h]
  for t in [0,1,2,3,min(32,len(seq)-1)]:
   a,_,g=data.target(h,o,t,True);b=canonical_teacher_prefix(seq,x,t,data.rows[o]['source_sha']);err=float((a.ca-b.state.ca).abs().max()) if t else 0.
   if err>1e-9 or not torch.allclose(g,b.next_action.geometry,atol=1e-9,rtol=1e-9):raise ValueError('fast canonical gauge/reference mismatch')
   checks.append(err)
 write(R/'preflight/REAL_GAUGE_PARITY.json',{'status':'PASS','checks':len(checks),'max_error_A':max(checks)})
 # Freeze initialization criterion before testing it; preserve original distribution family.
 criterion={'TRAIN_panel_targets':256,'seed':931,'required_mean_NLL_reduction_fraction':.10,'maximum_extreme_component_NLL':100,'decision':'PENDING','TEST_used':False}
 write(R/'preflight/PRIOR_INITIALIZATION_CRITERION.json',criterion)
 priors=json.load(open(PILOT/'priors/GEOMETRY_PRIORS.json'))['statistics']['GLOBAL'];mu=torch.tensor([priors[k]['mean'] for k in ['log_d','logit_theta_over_pi','atanh_tau_over_pi']]);ls=torch.tensor([.5*math.log(priors[k]['variance']) for k in ['log_d','logit_theta_over_pi','atanh_tau_over_pi']]);flags=json.load(open(PILOT/'priors/COORDINATE_EXTREMES.json'))['examples'];extreme=[]
 for f in flags:
  o=f['observation_id'];t=f['positions_1based'][1]-1;g=torch.from_numpy(data.g[o][t]);dist=GeometryDistribution(mu,ls,mask_for_step(t));extreme.append(float(-dist.log_prob(g)/max(1,sum(mask_for_step(t)))))
 # Source extremes make empirical distance variance dangerously overconfident if any exceeds threshold.
 decision={'applied':False,'reason':'TRAIN source-extreme conditional NLL exceeds predeclared100 threshold' if max(extreme)>100 else 'ordinary initialization retained unless complete preflight advantage verified','transformed_mu':mu.tolist(),'transformed_log_std':ls.tolist(),'source_extreme_normalized_NLL':extreme,'statistics_sha256':file_sha(PILOT/'priors/GEOMETRY_PRIORS.json'),'same_for_J1_J2':True}
 write(R/'preflight/PRIOR_INITIALIZATION_DECISION.json',decision)
 records=[]
 for name in ['S1','J2']:
  m=model(name);opt=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=0.,weight_decay=.01,fused=True)
  for L in [64,128,256,384,500]:
   candidates=sorted(data.jointids['TRAIN'],key=lambda h:(abs(len(data.seqs[h])-L),h))[:64];base=[(h,data.owners[h][0],min(L-1,len(data.seqs[h])-1)) for h in candidates]
   for B in [1,2,4,8,16,32,64]:
    for precision in ['BF16','FP32']:
     cfg={'precision':precision,'activation_checkpoint':False};row={'name':name,'requested_length':L,'batch':B,**cfg}
     try:
      torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();a=time.monotonic();b,y,g=make(data,base[:B],name.startswith('J'));torch.cuda.synchronize();row['data_seconds']=time.monotonic()-a
      a=time.monotonic();prepared=features_batch(b,m.encoder.config.chemistry,m.encoder.sequence_only);torch.cuda.synchronize();row['geometry_pair_seconds']=time.monotonic()-a;del prepared
      opt.zero_grad(set_to_none=True);a=time.monotonic();v,aa,ge,_=loss_call(m,b,y,g,cfg);torch.cuda.synchronize();row['forward_seconds']=time.monotonic()-a;a=time.monotonic();v.mean().backward();torch.cuda.synchronize();row['backward_seconds']=time.monotonic()-a
      norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);a=time.monotonic();opt.step();torch.cuda.synchronize();row['optimizer_seconds']=time.monotonic()-a;row['decisions_per_second']=B/(row['forward_seconds']+row['backward_seconds']+row['optimizer_seconds']+row['data_seconds']);row['loss']=float(v.mean());row['gradient_norm']=float(norm);row['status']='PASS';row.update(guard());row['device']=gpu();del b,y,g,v,aa,ge
     except torch.cuda.OutOfMemoryError as e:
      row['status']='CUDA_OOM_PROFILE_ONLY';row['error']=str(e)[:300];opt.zero_grad(set_to_none=True)
     records.append(row);write(R/'preflight/BATCH_PROFILE.json',records);print(json.dumps(row),flush=True)
     torch.cuda.empty_cache()
     if row['status']!='PASS':break
    if row['status']!='PASS':break
  del opt,m;torch.cuda.empty_cache()
 # Select best throughput at longest-prefix case with adequate measured device headroom.
 configs={}
 for name in ['S1','J1','J2','S2']:
  family='J2' if name.startswith('J') else 'S1';passed=[r for r in records if r['name']==family and r['requested_length']==500 and r['status']=='PASS'];best=max(passed,key=lambda r:r['decisions_per_second'])
  configs[name]={'version':'scaled-bc-v1','name':name,'width':352,'layers':8,'heads':8,'ffn_mult':4,'max_length':500,'effective_batch':64,'live_batch':min(64,best['batch']),'precision':best['precision'],'activation_checkpoint':False,'seed':42,'max_updates':10000,'warmup_updates':250,'learning_rate':.0003,'weight_decay':.01,'clip_norm':1.,'geometry_weight':1.,'geometry_normalization':'valid_component_mean','dropout':0.,'source_quality':'RETAIN_ADMITTED_SOURCES_NO_SILENT_EXCLUSION','prior_initialization':False,'milestones':MILESTONES,'validation_panel_identities':320,'validation_prefixes_per_identity':4,'TEST':'ONE_FINAL_EVENT_ONLY_AFTER_ALL_SELECTIONS_FROZEN','sampler':'independent identity/position/observation RNG streams; same seed/population for S1/J1/J2','architecture_sha256':digest(counts[name]['architecture']),'dataset_manifest_sha256':file_sha(P/'PROPOSAL_MANIFEST.json'),'prior_sha256':file_sha(PILOT/'priors/GEOMETRY_PRIORS.json')}
 # Strict BF16 validation against FP32 before admitting mixed precision.
 m=model('J2');sample=Sampler(data).draw(8);b,y,g=make(data,sample,True);fp=loss_call(m,b,y,g,{'precision':'FP32'})[0].mean();fp.backward();grads=[p.grad.detach().clone() if p.grad is not None else None for p in m.parameters()];m.zero_grad(set_to_none=True);bf=loss_call(m,b,y,g,{'precision':'BF16'})[0].mean();bf.backward();err=math.sqrt(sum(float((p.grad-q).square().sum()) for p,q in zip(m.parameters(),grads) if q is not None))/max(1e-20,math.sqrt(sum(float(q.square().sum()) for q in grads if q is not None)));diff=abs(float(fp)-float(bf));ok=err<.05 and diff<.05
 write(R/'preflight/BF16_PARITY.json',{'status':'PASS' if ok else 'FP32_FALLBACK','relative_gradient_L2':err,'loss_absolute_error':diff,'thresholds':{'grad_relative_L2':.05,'loss_absolute':.05}})
 if not ok:
  for name,c in configs.items():
   family='J2' if name.startswith('J') else 'S1';passed=[r for r in records if r['name']==family and r['requested_length']==500 and r['status']=='PASS' and r['precision']=='FP32'];best=max(passed,key=lambda r:r['decisions_per_second']);c['precision']='FP32';c['live_batch']=min(64,best['batch'])
 del m;torch.cuda.empty_cache()
 write(R/'configs/FROZEN.json',configs)
 write(R/'configs/VALIDATION_PANELS.json',{'matched':data.panel(True),'full':data.panel(False),'role':'DEVELOPMENT_ONLY','TEST_read':False})
 write(R/'PREFLIGHT_PASS.json',{'status':'PASS','configs_sha256':file_sha(R/'configs/FROZEN.json'),'data':report,'projected_checkpoint_bytes':4*6*counts['J2']['stored_parameters']*12,'reserve_required':10*2**30,'resources':guard(4*2**30)})
 print('SCALED_PREFLIGHT_PASS',flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['preflight']);q=a.parse_args();preflight()
