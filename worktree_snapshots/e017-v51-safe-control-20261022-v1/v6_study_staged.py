"""Frozen E017 V6 candidate diagnostic. No optimizer, training, or TEST reader."""
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

R=Path('/mnt/d/Simone/proteinGen_data/e017_v6_frozen_parent_candidates_20261023_v1')
OLD=R.parent/'e017_v51_safe_control_20261022_v1'
V5=R.parent/'e017_v5_goal_observability_20261022_v1'
PARENT=V5/'checkpoints/D/checkpoint_00250.pt'
PARENT_SHA='0b2f0b68783e4234f44c3b1041a5c1f59f04191af79b13232553b511168de265'
STOP=False

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def save(name,obj):
 p=R/name;tmp=p.with_name(p.name+'.part')
 with tmp.open('w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
 fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def free():
 s=os.statvfs(R);return s.f_bavail*s.f_frsize
def seed(task,i):
 return [7101,7102][i] if i<2 else int(hashlib.sha256(f'E017V6|7617|{task}|{i}'.encode()).hexdigest()[:15],16)
def tasks():return read(OLD/'manifests/AUDIT_H4_native.json')['tasks']
def loadrows():
 p=R/'results/candidates.jsonl'
 return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []

def initialize():
 assert not (R/'PROTOCOL.json').exists()
 ts=tasks();old=read(OLD/'results/INITIAL_00000_AUDIT_H4_native.json')
 assert len(ts)==63 and len(old['records'])==126 and old['summary']['success']==20
 assert len({t['protected_group_id'] for t in ts})==32
 tr=read(V5/'manifests/TRAIN.json')['tasks'];assert not {t['protected_group_id'] for t in ts}&{t['protected_group_id'] for t in tr}
 assert sha(PARENT)==PARENT_SHA
 inputs={str(p):sha(p) for p in [PARENT,OLD/'manifests/AUDIT_H4_native.json',OLD/'results/INITIAL_00000_AUDIT_H4_native.json',OLD/'FINAL_E017_V51_SCIENTIFIC_REPORT.md',OLD/'NEXT_RESEARCH_DECISION.md',V5/'manifests/TRAIN.json',R/'source/scripts/e017_v6/study.py']}
 save('PROTOCOL.json',dict(schema='e017_v6_frozen_candidates_v1',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  source_commit=subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip(),
  panel='reused V5.1 AUDIT native H4; existing protected VALIDATION, exploratory inference diagnostic, not new independent confirmation',
  tasks=63,identities=32,groups=32,K=[1,2,4,8,16,32],candidate_indices='0..31; indices0/1 reused seeds7101/7102, then task-indexed SHA256 seeds',seed_rule='E017V6|7617|task_id|candidate_index SHA256 first15hex',
  shared_legacy_noise='historical streams7101/7102 shared across tasks; later streams independently task-indexed; paired targets share legacy noise by design',
  parent_SHA256=PARENT_SHA,parent=str(PARENT),temperature=1.,horizon=4,precision='BF16 neural; FP64 geometry; unchanged historical law',
  A='first candidate only (same baseline across K)',B='minimum position_A then candidate_index; ignore validity',
  C='valid candidates only; minimize max(position_A/8,orientation_deg/35), then original Huber+5sin^2(theta/2) score, then candidate_index; no valid candidate => abstention/failure',
  D='any full valid success in first K; C attains D exactly because max-normalized-error<=1 iff both goal thresholds pass',
  additional_selector='original goal-cost minimum among valid candidates, exploratory, may miss full success despite its presence',
  early_prefix_rule='exploratory irreversible first nonlocal collision (one exceeds final clash limit for22/24-residue fragments), spacing/support/degenerate frame; contour bound distance>remaining*max_bond+8 proves impossible to hit position while preserving max_bond; never used to alter pools',
  limits=read(OLD/'HISTORICAL_H4_COLLISION_AND_KL_ANALYSIS.json')['summary']['limits'],
  primary='K32 C versus B paired full-success delta, identity-clustered95%CI; other K descriptive with paired familywise99.166667% six-K bands',
  bootstrap_seed=176617,bootstrap_replicates=20000,storage_reserve=2*2**30,maximum_new_candidates=1890,maximum_worker_seconds=7200,
  proposal_KL='record per step and per Gaussian action component; selected-law KL not inferred; no policy promotion',
  safety='stop on storage<2GiB+64MiB, nonfinite/support/parity/hash/frozen-state failure, GPU allocation>6GiB; retain durable completed rows; no optimizer or TEST',input_SHA256=inputs))
 save('RECOVERY.json',dict(status='READY',next_task=0,completed_candidates=0,no_training=True))
 print('SEALED',len(ts),126,'reuse',1890,'new',flush=True)

def early(t,tail,lim):
 import numpy as np
 x=np.asarray(t['native_prefix']+tail);start=t['t'];out=None
 for k in range(1,5):
  y=x[:start+k];d=np.linalg.norm(y[-1]-y[-2]);cross=np.linalg.norm(np.cross(y[-1]-y[-2],y[-2]-y[-3]));dd=np.linalg.norm(y[-1]-y[:-3],axis=-1)
  reasons=[]
  if d<lim['adjacent_low_A'] or d>lim['adjacent_high_A']:reasons.append('spacing')
  if cross<=1e-8:reasons.append('degenerate_frame')
  if bool((dd<2).any()):reasons.append('nonlocal_collision')
  if np.linalg.norm(y[-1]-np.asarray(t['target']))>(4-k)*lim['adjacent_high_A']+8:reasons.append('necessary_position_bound')
  if reasons and out is None:out=dict(step=k,reasons=reasons,saved_forward_steps=4-k)
 return out

def worker():
 import numpy as np
 import torch
 sys.path.insert(0,str(R/'source/scripts/e017_v5'))
 import study as s
 from protein_joint_sac.crisp_v3.prior import tensor_sha
 from protein_joint_sac.crisp_v3.environment import frame
 from protein_joint_sac.crisp_v3.recovery import state_digest
 config=read(R/'PROTOCOL.json')
 for p,h in config['input_SHA256'].items():assert sha(p)==h,p
 assert subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip()==config['source_commit']
 assert free()>2*2**30+64*2**20
 start=time.monotonic();torch.set_num_threads(1);s.seed(7617)
 e=s.Engine();st=torch.load(PARENT,map_location='cpu',weights_only=False);e.actor.load_state_dict(st['actor'],strict=True);del st
 e.arm='D';e.actor.eval().requires_grad_(False);e.prior.model.eval().requires_grad_(False)
 initial=state_digest(e.actor.state_dict());j1=e.prior.identity
 assert sum(p.numel() for p in e.actor.parameters())==17438114
 assert not any(p.requires_grad for p in e.actor.parameters())
 lim=config['limits'];ts=tasks();legacy={(r['task_id'],r['seed']):r for r in read(OLD/'results/INITIAL_00000_AUDIT_H4_native.json')['records']}
 # No BC diagnostic forward: exactly four actor/J1 calls per fresh continuation.
 def draw(t,ns):
  began=time.monotonic();torch.cuda.synchronize();x=s.xprefix(t);eps=torch.randn(4,3,dtype=torch.float64,generator=torch.Generator().manual_seed(ns)).cuda();ks=[];klcomponents=[]
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for k in range(4):
    mu,ls,rm,rl,*_=e.predict(t,x)
    components=rl-ls+.5*((ls.exp()/rl.exp()).square()+((mu-rm)/rl.exp()).square())-.5
    klcomponents.append(components.double().cpu().tolist());ks.append(float(components.mean()))
    g=s.physical(mu.double()+ls.double().exp()*eps[k])
    if not bool(torch.isfinite(g).all()) or not bool((g[0]>0)&(g[1]>0)&(g[1]<math.pi)&(g[2].abs()<math.pi)):raise FloatingPointError('nonfinite/support; no redraw')
    x=torch.cat([x,s.next_ca(x,g)[None]],0)
   r=s.metric(t,x,ns)
  torch.cuda.synchronize();r.update(KL=float(np.mean(ks)),step_KL=ks,KL_components_by_step=klcomponents,seconds=time.monotonic()-began)
  return r
 # Reproduce only two existing candidates as bounded source/architecture parity checks.
 if not (R/'PARITY.json').exists():
  parity=[]
  for ns in [7101,7102]:
   a=draw(ts[0],ns);b=legacy[(ts[0]['task_id'],ns)]
   error=float(np.max(np.abs(np.asarray(a['tail'])-np.asarray(b['tail']))));assert error<1e-8,(error,ns)
   assert a['valid']==b['valid'] and a['success']==b['success'] and abs(a['KL']-b['KL'])<1e-6
   parity.append(dict(seed=ns,tail_max_absolute_error_A=error,position_error_A=abs(a['position_A']-b['position_A']),seconds=a['seconds']))
  save('PARITY.json',dict(PASS=True,records=parity,actor_SHA=initial,J1_tensor_SHA=j1,strict_load=True,actor_parameters=17438114,J1_parameters=12351746,optimizer_constructed=False,extra_actor_and_J1_passes_each=8))
 existing=loadrows();done={(r['task_id'],r['candidate_index']) for r in existing};assert len(done)==len(existing)
 log=open(R/'results/candidates.jsonl','a',buffering=1)
 try:
  for ti,t in enumerate(ts):
   for i in range(32):
    if (t['task_id'],i) in done:continue
    if STOP:raise InterruptedError('cooperative stop at candidate boundary')
    if time.monotonic()-start>config['maximum_worker_seconds']:raise TimeoutError('bounded two-hour worker limit')
    if free()<2*2**30+64*2**20:raise OSError('storage reserve')
    ns=seed(t['task_id'],i)
    if i<2:r=dict(legacy[(t['task_id'],ns)]);r.update(seconds=None,KL_components_by_step=None,reused=True)
    else:r=draw(t,ns);r['reused']=False
    r.update(candidate_index=i,seed=ns,early_rule=early(t,r['tail'],lim))
    r['goal_hit']=bool(r['translation_hit'] and r['rotation_hit'])
    r['goal_score']=max(r['position_A']/8,r['orientation_deg']/35)
    # Original goal cost computed from saved errors/coords, not an additional policy call.
    delta=np.abs((np.asarray(r['tail'][-1])-np.asarray(t['target']))/3.8)
    r['original_goal_cost']=float(np.where(delta<1,.5*delta**2,delta-.5).sum()+5*math.sin(math.radians(r['orientation_deg'])/2)**2)
    x=np.asarray(t['native_prefix']+r['tail']);dd=np.linalg.norm(np.diff(x,axis=0),axis=1)
    r['validity_failures']=[]
    if not np.isfinite(x).all():raise FloatingPointError('nonfinite trajectory')
    if dd.min()<lim['adjacent_low_A'] or dd.max()>lim['adjacent_high_A']:r['validity_failures'].append('spacing')
    if r['clashes_per100']>lim['clashes_per100_max']:r['validity_failures'].append('nonlocal_clashes')
    if not lim['Rg_normalized_low']<=r['Rg_normalized']<=lim['Rg_normalized_high']:r['validity_failures'].append('Rg')
    if bool((np.linalg.norm(np.cross(np.diff(x,axis=0)[1:],np.diff(x,axis=0)[:-1]),axis=-1)<=1e-8).any()):r['validity_failures'].append('frame')
    assert bool(r['valid'])==(not r['validity_failures'])
    # Compact original four-point tails permit reproducible diversity and failure analysis.
    compact={k:r[k] for k in ['task_id','identity','group','variant','H','candidate_index','seed','reused','seconds','position_A','orientation_deg','translation_hit','rotation_hit','valid','success','goal_hit','goal_score','original_goal_cost','KL','step_KL','KL_components_by_step','clashes_per100','Rg_normalized','adjacent_min_A','adjacent_max_A','angle_mean_rad','torsion_mean_rad','tail','validity_failures','early_rule']}
    log.write(json.dumps(compact,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno());done.add((t['task_id'],i))
   assert state_digest(e.actor.state_dict())==initial
   save('RECOVERY.json',dict(status='RUNNING',pid=os.getpid(),last_completed_task=ti,completed_candidates=len(done),UTC=time.time(),free_bytes=free(),max_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),max_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),max_worker_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.monotonic()-start,model_frozen=True))
   print('TASK',ti+1,len(ts),'candidates',len(done),'seconds',round(time.monotonic()-start,2),flush=True)
  e.verify();assert state_digest(e.actor.state_dict())==initial and tensor_sha(e.prior.model)==j1
  assert sha(PARENT)==PARENT_SHA
  save('WORKER_COMPLETED.json',dict(complete=True,candidates=len(done),new_candidates=1890,reused_candidates=126,actor_state_digest=initial,J1_tensor_SHA=j1,elapsed_seconds=time.monotonic()-start,free_bytes=free(),GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,UTC=time.time(),no_training=True))
 except BaseException as ex:
  save('WORKER_STOP.json',dict(reason=repr(ex),completed_candidates=len(done),last_complete_row_durable=True,UTC=time.time()));raise
 finally:log.flush();os.fsync(log.fileno());log.close()

def handle(*args):
 global STOP
 STOP=True

if __name__=='__main__':
 signal.signal(signal.SIGTERM,handle);signal.signal(signal.SIGINT,handle)
 {'initialize':initialize,'worker':worker}[sys.argv[1]]()
