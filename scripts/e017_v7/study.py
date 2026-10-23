"""E017 V7 bounded fixed-AA H4 study. No TEST/high-level/reachability updates."""
import argparse
import collections
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch

R=Path('/mnt/d/Simone/proteinGen_data/e017_v7_amortized_lowlevel_20261023_v1')
OLD=R.parent/'e017_v51_safe_control_20261022_v1';V5=R.parent/'e017_v5_goal_observability_20261022_v1';V6=R.parent/'e017_v6_frozen_parent_candidates_20261023_v1'
sys.path.insert(0,str(R/'source/scripts/e017_v5'))
import study as s
from protein_joint_sac.crisp_v3.recovery import sha,state_digest,atomic_json,atomic_checkpoint,serialized_size,StoragePolicy
from protein_joint_sac.crisp_v3.storage import rng_state,set_rng
from protein_joint_sac.crisp_v3.density import sample,log_prob,semimarkov
from protein_joint_sac.crisp_v3.networks import LowQ,PROFILES
from protein_joint_sac.crisp_v3.environment import frame,goal_tokens,assess
from protein_joint_sac.crisp_v3.prior import tensor_sha
from protein_joint_sac.contracts import AA,BuildState

PARENT=V5/'checkpoints/D/checkpoint_00250.pt';PSHA='0b2f0b68783e4234f44c3b1041a5c1f59f04191af79b13232553b511168de265'
POLICY=StoragePolicy(min_headroom_bytes=2*2**30,near_term_growth_bytes=64*2**20,checkpoint_hint_bytes=650*2**20)
STOP=False
CFG=dict(schema='E017_V7_v1',seed=7717,TRAIN_identities=512,TRAIN_K=8,DEV_identities=16,CONFIRM_identities=32,
 milestones=[25,100,250,500],periodic_checkpoint=50,actor_updates_max=500,critic_calibration_updates=250,
 lr=2e-6,lr_floor=2e-7,warmup=25,critic_lr=1e-4,batch=4,clip=5.,gamma=.995,alpha=.001,polyak=.005,
 dual_initial=.1,dual_target=.8,dual_eta=.05,native_BC_weight=.1,trust_region_step_KL=.02,
 eval_K1_seed=17717,eval_K=[1,2,4,8,16,32],KL_limit=1.,validity_allowance=.05,H2_retention_allowance=.05,
 reward='terminal valid*(full_success+.25*exp(-.5*((position/8)^2+(orientation/35)^2))); zero preceding rewards; no invalid positive teacher',
 RWR_weights='valid*(full_success+.1*joint_near_goal); normalize within task over all available trajectories; uniform identity sampling; invalid weight0; no-valid task uses native BC/KL only',
 selection='eligible highest DEV K1 full success, then joint score, then earliest; confirmation never selects',
 continuation250to500='DEV K1 point >= parent and all safety gates; RWR last25 weighted NLL improves >=5% versus first25, SAC critic remains calibrated and TRAIN self-return >= baseline',
 critic_gate='250 bounded MC known-return updates: training MSE <=.5 constant predictor MSE; held-out terminal MSE <=.9 constant predictor MSE; nonfinite fails; no SAC actor step until pass',
 D_gate='B and C eligible with actor updates>=250; B DEV K1 strictly improves and C not worse than parent; no prior numerical/scientific stop; otherwise skip',
 promotion='CONFIRM native K1 gain>=5pp, paired identity-clustered familywise98.333333% lower>0 (three trained arms vs A); H4/H2 mean KL<=1; H4 validity point decline<=5pp; H2 success retention within5pp; goal direction>=.6; generated-prefix K1 point not worse; K32 success not worse; one seed limits generality',
 numerical='nonfinite/support failure: preserve last complete recovery; no redraw; storage/time/CUDA checks; no threshold changes',
 exact_recovery='full model, AdamW, scheduler, dual, RNG, deterministic cursor, compact online replay, immutable candidate/config/task/source/parent hashes; verified atomic latest/best',
 arm_budgets='B/C same shared offline4096behavior trajectories plus one actual current-policy H4episode/update, same task queue and common noise; same actor update cap; C additional twin-critic compute separately counted, no equal-compute superiority claim',
 no_TEST=True,fixed_AA=True,train_high_level=False,train_reachability=False)

def read(p):return json.loads(Path(p).read_text())
def save(n,x):atomic_json(R/n,x)
def commit():return subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip()
def rank(x):return hashlib.sha256(f'7717|{x}'.encode()).hexdigest()
def ns(tid,i,panel='TRAIN'):return int(hashlib.sha256(f'V7|7717|{panel}|{tid}|{i}'.encode()).hexdigest()[:15],16)
def tasks(panel):return read(R/f'manifests/{panel}.json')['tasks']
def budget():
 POLICY.require(R,650*2**20)
 if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()>6*2**30:raise RuntimeError('CUDA6GiB resource gate')
 if STOP:raise InterruptedError('cooperative requested stop')

def prepare():
 from protein_joint_sac.data.scaled_supervised import Data,canonical,internal
 from protein_joint_sac.data.accepted_supervised import CoordinateReader
 from protein_joint_sac.crisp_v3.execution import set_loader
 torch.set_num_threads(1);set_loader('arrow_scalar_lru4');data=Data(False);reader=CoordinateReader(data.proposal.external)
 pool=read(V5/'manifests/TRAIN.json')['tasks'];by=collections.defaultdict(list)
 for t in pool:
  if t['H']==4 and t['variant']=='native':by[t['identity']].append(t)
 selected=sorted(by,key=rank)[:512];train=[]
 for h in selected:
  t=copy.deepcopy(min(by[h],key=lambda t:t['t']));r=data.rows[t['observation']]
  assert r['sequence_hash']==h and r['split']=='TRAIN' and r['source_sha']==t['source_sha']
  x=s.fk(t,torch.tensor(t['native_actions'],dtype=torch.float64));m=s.metric(t,x,0);assert m['valid'] and m['success'] and t['witness_max_error_A']<1e-6
  t.update(task_seed=ns(t['task_id'],0),validity_masks=[[True]*3]*4,source_task_manifest=str(V5/'manifests/TRAIN.json'));train.append(t)
 assert len(train)==512
 save('manifests/TRAIN.json',dict(tasks=train,eligible_existing_native_H4_identities=len(by),selection='hash-ranked512 identities, earliest eligible nativeH4window, existing qualified observation/witness reverified'))
 excluded=set()
 for p in [V5/'manifests/DEV.json',V5/'manifests/CONFIRM.json',R.parent/'e017_v4_multistep_20261022_v1/manifests/VALIDATION.json']:
  excluded.update(t['protected_group_id'] for t in read(p)['tasks'])
 rows=[];reject=[];groups=set(excluded);needed=48
 ids=sorted(data.jointids['VALIDATION'],key=lambda h:rank('freshV7'+h))
 for h in ids:
  if len({t['identity'] for t in rows})==needed:break
  g=data.proposal.sequences[h]['protected_group_id']
  if g in groups or len(data.seqs[h])<24:continue
  o=min(data.owners[h]);r=data.rows[o]
  try:
   xx=canonical(reader.read(r));gg=internal(xx);seq=data.seqs[h];assert len(xx)==len(seq)
   t=dict(identity=h,protected_group_id=g,split='VALIDATION',observation=o,source_sha=r['source_sha'],entry_id=r['entry_id'],coordinate_path=r['coordinate_path'],coordinate_row=r['coordinate_row'],N=len(seq),t=20,H=4,sequence=seq,native_prefix=xx[:20].tolist(),native_actions=gg[20:24].tolist(),hidden_native_future=xx[20:24].tolist(),target=xx[23].tolist(),orientation=frame(torch.tensor(xx[:24])).tolist(),variant='native',task_id=f'{h}_20_4_native',validity_masks=[[True]*3]*4)
   m=s.metric(t,torch.tensor(xx[:24]),0)
   if not m['success']:raise ValueError('native witness fails original validity')
   rebuilt=s.fk(t,torch.tensor(t['native_actions']));err=float((rebuilt-torch.tensor(xx[:24])).abs().max());assert err<1e-6
   t['witness_max_error_A']=err;t['witness_valid']=True;t['task_seed']=ns(t['task_id'],0)
   alt=copy.deepcopy(t);a=torch.tensor(t['native_actions']);a[0,2]=((a[0,2]+.8+math.pi)%(2*math.pi))-math.pi;ax=s.fk(t,a)
   alt.update(target=ax[-1].tolist(),orientation=frame(ax).tolist(),variant='kinematic_alt',task_id=f'{h}_20_4_alt',witness_actions=a.tolist(),experimental_target=False,synthetic_control=True)
   if not s.metric(alt,ax,0)['success']:raise ValueError('alternate witness invalid; pair not admitted')
   alt['task_seed']=ns(alt['task_id'],0);rows.extend([t,alt]);groups.add(g)
  except (ValueError,AssertionError,FloatingPointError) as ex:reject.append(dict(identity=h,reason=str(ex)))
  if len(rows)%16==0:print('PREPARED_NEW_VALIDATION',len(rows)//2,flush=True)
 assert len(rows)==96
 for panel,rr in [('DEV',rows[:32]),('CONFIRM',rows[32:])]:save(f'manifests/{panel}.json',dict(tasks=rr,partition='existing protected VALIDATION',unused_groups_excluded_all_V4_V5_prepared_panels=True))
 for panel in ['DEV','CONFIRM']:
  rr=[]
  for t in tasks(panel):
   z=copy.deepcopy(t);z['H']=2;z['task_id']=t['task_id'].replace('_4_','_2_');a=torch.tensor(t.get('witness_actions',t['native_actions'])[:2]);x=torch.tensor(t['native_prefix'])
   for action in a:x=torch.cat([x,s.next_ca(x,action)[None]],0)
   z.update(target=x[-1].tolist(),orientation=frame(x).tolist(),native_actions=t['native_actions'][:2],hidden_native_future=t['hidden_native_future'][:2]);assert s.metric(z,x,0)['success'];rr.append(z)
  save(f'manifests/{panel}_H2.json',dict(tasks=rr,construction='same source/prefix, two-step valid native/alternate witness'))
 save('manifests/EXCLUSIONS.json',reject)
 parts=[{t['protected_group_id'] for t in tasks(p)} for p in ['TRAIN','DEV','CONFIRM']]
 assert all(not parts[i]&parts[j] for i in range(3) for j in range(i))
 save('TASK_CENSUS.json',dict(TRAIN_identities=512,TRAIN_tasks=512,TRAIN_eligible_joint_population=21475,eligible_existing_H4=len(by),DEV_identities=16,DEV_tasks=32,CONFIRM_identities=32,CONFIRM_tasks=64,previously_prepared_validation_groups_excluded=len(excluded),rejected=len(reject),TEST_evaluated=False))

class Engine(s.Engine):
 def __init__(self):
  super().__init__();assert sha(PARENT)==PSHA;st=torch.load(PARENT,map_location='cpu',weights_only=False);self.actor.load_state_dict(st['actor'],strict=True);del st
  self.arm='D';self.actor.eval();assert state_digest(self.actor.state_dict())==read(V6/'WORKER_COMPLETED.json')['actor_state_digest'];self.parent_fingerprint=state_digest(self.actor.state_dict());self.calls=collections.Counter()
 def batch(self,t,x):
  with torch.no_grad():
   p=self.p(t,x);ref=self.prior_forward(p);self.calls['J1']+=1;gt,gv=goal_tokens(s.goal(t),'cuda');ctx,dp,rot=s.relative(t,x);gt=gt.clone();gt[0,0,:3]=dp.float()/s.CONFIG['position_scale_A'];gt[0,0,3:12]=rot.flatten().float()
   return dict(prefix=p,reference=ref,features=s.features_batch(p,chemistry=True),goals=gt,goal_valid=gv,context=ctx[None].float(),subgoal=x.new_zeros(1).float())
 def distribution(self,t,x):
  b=self.batch(t,x);l,mu,ls=self.actor(b);self.calls['actor']+=1;a=AA.index(t['sequence'][len(x)])
  return mu[0,a],ls[0,a],b['reference']['mu'][0,a],b['reference']['ls'][0,a],b,a

def reward(m):
 j=float(m['valid'])*math.exp(-.5*((m['position_A']/8)**2+(m['orientation_deg']/35)**2))
 return float(m['success'])+.25*j,j

def rollout(e,t,seed,version,delta_step=None,donor=None):
 x=s.xprefix(t);epsilon=torch.randn(t['H'],3,dtype=torch.float64,generator=torch.Generator().manual_seed(seed)).cuda();gs=[];us=[];lp=[];ks=[];kcomponents=[];beg=time.monotonic()
 e.actor.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for k in range(t['H']):
   mu,ls,rm,rl,_,_=e.distribution(t,x);u=mu.double()+ls.double().exp()*epsilon[k]
   if k==delta_step and donor is not None:u=torch.tensor(donor[k],dtype=torch.float64,device='cuda')
   g=s.physical(u)
   if not bool(torch.isfinite(g).all()) or not bool((g[0]>0)&(g[1]>0)&(g[1]<math.pi)&(g[2].abs()<math.pi)):raise FloatingPointError('action support, no redraw')
   lp.append(float(log_prob(mu.double(),ls.double(),g,torch.ones(3,dtype=torch.bool,device='cuda'))));c=rl-ls+.5*((ls.exp()/rl.exp()).square()+((mu-rm)/rl.exp()).square())-.5;ks.append(float(c.mean()));kcomponents.append(c.float().cpu().tolist());gs.append(g.cpu().tolist());us.append(u.cpu().tolist());x=torch.cat([x,s.next_ca(x,g)[None]],0)
  m=s.metric(t,x,seed)
 torch.cuda.synchronize();r,j=reward(m)
 return dict(task_id=t['task_id'],identity=t['identity'],group=t['protected_group_id'],seed=seed,policy_version=version,geometry=gs,latent=us,behavior_log_prob=lp,masks=[[True]*3]*t['H'],tail=m['tail'],position_A=m['position_A'],orientation_deg=m['orientation_deg'],valid=m['valid'],success=m['success'],translation_hit=m['translation_hit'],rotation_hit=m['rotation_hit'],clashes_per100=m['clashes_per100'],Rg_normalized=m['Rg_normalized'],KL=float(np.mean(ks)),step_KL=ks,KL_components=kcomponents,reward=r,joint=j,seconds=time.monotonic()-beg)

def reconstruct(t,ep,k):
 x=s.xprefix(t)
 for action in ep['geometry'][:k]:x=torch.cat([x,s.next_ca(x,torch.tensor(action,device='cuda',dtype=torch.float64))[None]],0)
 return x.detach()

def summary(rr,K=1):
 return dict(attempts=len(rr),identities=len({r['identity'] for r in rr}),success=sum(r['success'] for r in rr),valid=sum(r['valid'] for r in rr),position_A=float(np.mean([r['position_A'] for r in rr])),orientation_deg=float(np.mean([r['orientation_deg'] for r in rr])),KL=float(np.mean([r['KL'] for r in rr])),reward=float(np.mean([r['reward'] for r in rr])),joint=float(np.mean([r['joint'] for r in rr])))

def evaluate(e,label,panel,K=2):
 path=R/f'results/{label}_{panel}_K{K}.json'
 if path.exists():return read(path)
 rr=[]
 for t in tasks(panel):
  budget()
  for i in range(K):rr.append(dict(rollout(e,t,ns(t['task_id'],i,panel),label),candidate_index=i))
  if len(rr)%(8*K)==0:print('EVAL',label,panel,len(rr),flush=True)
 first=[r for r in rr if r['candidate_index']==0];out=dict(label=label,panel=panel,K=K,summary=summary(rr),K1=summary(first),records=rr)
 save(f'results/{label}_{panel}_K{K}.json',out);return out

def seal():
 assert not (R/'SEALED_CONFIG.json').exists()
 inputs={str(p):sha(p) for p in (R/'manifests').glob('*.json')};inputs[str(PARENT)]=sha(PARENT)
 for p in [V6/'FINAL_E017_V6_SCIENTIFIC_REPORT.md',V6/'PROVENANCE_SHA256.json',OLD/'FINAL_E017_V51_SCIENTIFIC_REPORT.md',V5/'FINAL_E017_V5_SCIENTIFIC_REPORT.md',R/'source/scripts/e017_v7/study.py']:inputs[str(p)]=sha(p)
 for p in [Path(s.__file__),Path(s.v.__file__),R/'source/scripts/e017_v5/frozen_science_config.json',R/'source/src/protein_joint_sac/crisp_v3/networks.py',R/'source/src/protein_joint_sac/crisp_v3/prior.py',R/'source/src/protein_joint_sac/crisp_v3/density.py',R/'source/src/protein_joint_sac/crisp_v3/environment.py',R/'source/src/protein_joint_sac/geometry/ca_builder.py',Path('/mnt/d/Simone/proteinGen_data/pdb_supervised_split_50id80cov_20261016_v1/PROPOSAL_MANIFEST.json')]:inputs[str(p)]=sha(p)
 save('CONFIG.json',CFG);save('SEALED_CONFIG.json',dict(source=commit(),config=CFG,input_SHA256=inputs,UTC=time.time()))

def collect():
 e=Engine();e.actor.requires_grad_(False);initial=e.parent_fingerprint;path=R/'results/TRAIN_candidates.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else [];done={(r['task_id'],r['candidate_index']) for r in rows}
 with path.open('a') as f:
  for ti,t in enumerate(tasks('TRAIN')):
   for k in range(8):
    if (t['task_id'],k) in done:continue
    budget();r=dict(rollout(e,t,ns(t['task_id'],k),'A'),candidate_index=k);f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
   if ti%16==0:
    save('LIVE_STATUS.json',dict(stage='TRAIN_CANDIDATES',task=ti+1,total=512,pid=os.getpid(),free_bytes=__import__('shutil').disk_usage(R).free,peak_GPU=torch.cuda.max_memory_allocated()));print('COLLECT',ti+1,512,flush=True)
 e.verify();assert state_digest(e.actor.state_dict())==initial
 save('CANDIDATES_COMPLETED.json',dict(complete=True,candidates=4096,TRAIN_identities=512,actor_fingerprint=initial,J1_tensor_SHA=e.prior.identity,candidate_file_SHA256=sha(path),network_calls=dict(e.calls)))

def handle(*_):
 global STOP
 STOP=True

# Training/calibration entrypoints are defined below before the sealed commit.
def config_check():
 seal=read(R/'SEALED_CONFIG.json');assert seal['source']==commit() and seal['config']==CFG
 for p,h in seal['input_SHA256'].items():assert sha(p)==h,p

def offline():return [json.loads(s) for s in (R/'results/TRAIN_candidates.jsonl').read_text().splitlines()]

def actor_optimizer(e):
 e.actor.requires_grad_(True);e.actor.aa.requires_grad_(False)
 o=torch.optim.AdamW([p for p in e.actor.parameters() if p.requires_grad],lr=CFG['lr'],weight_decay=.01,foreach=False)
 # Explicit fresh zero moments make the very first storage admission measure
 # the complete eventual Adam bundle, rather than an unrealistically empty one.
 for group in o.param_groups:
  for p in group['params']:o.state[p]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
 sc=torch.optim.lr_scheduler.LambdaLR(o,lambda u:min(1.,(u+1)/25)*(.1+.9*(1+math.cos(math.pi*min(u,500)/500))/2))
 return o,sc

def init_critics():
 # Original research LowQ capacity; target nets and all optimizers are separate.
 qs=[LowQ(PROFILES['research']['low_q']).cuda() for _ in range(2)]
 targets=[copy.deepcopy(q).eval().requires_grad_(False) for q in qs]
 opts=[torch.optim.AdamW(q.parameters(),lr=CFG['critic_lr'],weight_decay=.01,foreach=False) for q in qs]
 return qs,targets,opts

def checkpoint(e,label,u,o,sc,dual,online,monitor,qs=None,tq=None,qo=None,reason=None):
 promote=monitor.get('eligible',False) and reason is None and monitor.get('selection_score',[-1,-1])>monitor.get('best_score',[-1,-1])
 if promote:monitor['best_score']=monitor['selection_score']
 state=dict(schema='e017_v7_complete_recovery_v1',actor=e.actor.state_dict(),optimizer=o.state_dict(),scheduler=sc.state_dict(),
  dual=dual,online_replay=online,offline_replay=dict(path=str(R/'results/TRAIN_candidates.jsonl'),sha256=read(R/'CANDIDATES_COMPLETED.json')['candidate_file_SHA256']),
  rng=rng_state(),arm=label,update=u,identity_cursor=u*CFG['batch'],environment_episodes=len(online),environment_actions=4*len(online),
  critic=[q.state_dict() for q in qs] if qs else [],target_critic=[q.state_dict() for q in tq] if tq else [],
  critic_optimizer=[q.state_dict() for q in qo] if qo else [],critic_updates=monitor.get('critic_updates',0),
  config=CFG,source=commit(),sealed_input=read(R/'SEALED_CONFIG.json'),parent=str(PARENT),parent_SHA=PSHA,J1_tensor_SHA=e.prior.identity,
  monitoring=monitor,stop_reason=reason,AMP_scaler=None,fixed_AA=True)
 size=serialized_size(state);POLICY.require(R,size+2**20);p=R/f'checkpoints/{label}_{u:05}{"_stop" if reason else ""}.pt'
 if p.exists():assert False,'immutable checkpoint collision'
 rec=atomic_checkpoint(p,state,POLICY);rec.update(update=u,reason=reason,eligible=monitor.get('eligible',False) and reason is None)
 old=read(R/f'checkpoints/{label}_INDEX.json') if (R/f'checkpoints/{label}_INDEX.json').exists() else dict(latest=None,best=None)
 best=old['best']
 if rec['eligible']:
  if best is None or promote:best=rec
 save(f'checkpoints/{label}_INDEX.json',dict(latest=rec,best=best,serialized_bytes=size))
 retained={rec['path'],best['path'] if best else ''}
 for previous in [old['latest'],old['best']]:
  if previous and previous['path'] not in retained and Path(previous['path']).exists():
   with (R/'results/checkpoint_retirements.jsonl').open('a') as f:f.write(json.dumps(dict(path=previous['path'],sha256=previous['sha256'],replacement=rec['sha256'],reason='own superseded state; latest/best atomically committed and full digest read back'))+'\n')
   Path(previous['path']).unlink()
 return rec

def eligibility(ev,h2,base,b2):
 q=ev['summary'];h=h2['summary'];a=base['summary'];b=b2['summary'];reasons=[]
 if q['KL']>1 or h['KL']>1:reasons.append('conditional geometry KL>1')
 if q['valid']/q['attempts']<a['valid']/a['attempts']-.05:reasons.append('H4 validity drop>5pp')
 if h2['K1']['success']/h2['K1']['attempts']<b2['K1']['success']/b2['K1']['attempts']-.05:reasons.append('H2 K1 retention drop>5pp')
 return dict(eligible=not reasons,reasons=reasons,selection_score=[ev['K1']['success'],ev['K1']['joint']],H4=q,H2=h,H4_K1=ev['K1'],H2_K1=h2['K1'])

def teacher_step(e,t,ep):
 x=s.xprefix(t);terms=[];kl=[];old=[]
 for k in range(4):
  with torch.autocast('cuda',dtype=torch.bfloat16):mu,ls,rm,rl,b,a=e.distribution(t,x)
  g=torch.tensor(ep['geometry'][k],device='cuda',dtype=torch.float64);mask=torch.ones(3,dtype=torch.bool,device='cuda')
  terms.append(-log_prob(mu.double(),ls.double(),g,mask)/3);kl.append(e.kl(mu,ls,rm,rl));old.append((b,a,mu.detach().clone(),ls.detach().clone()))
  x=torch.cat([x,s.next_ca(x,g)[None]],0).detach()
 return torch.stack(terms).mean(),torch.stack(kl).mean(),old

def native_prior(e,t,step):
 x=s.xprefix(t)
 for action in t['native_actions'][:step]:x=torch.cat([x,s.next_ca(x,torch.tensor(action,device='cuda',dtype=torch.float64))[None]],0)
 with torch.autocast('cuda',dtype=torch.bfloat16):mu,ls,rm,rl,_,_=e.distribution(t,x)
 g=torch.tensor(t['native_actions'][step],device='cuda',dtype=torch.float64)
 return -log_prob(mu.double(),ls.double(),g,torch.ones(3,dtype=torch.bool,device='cuda'))/3,e.kl(mu,ls,rm,rl)

def critic_known_return(e,eps,qs,opts,updates=250):
 pool={t['task_id']:t for t in tasks('TRAIN')};ids=sorted(pool,key=lambda h:rank('calibration'+h));fit=set(ids[:48]);hold=set(ids[48:80])
 train=[(ep,k) for ep in eps if ep['task_id'] in fit for k in range(4)];test=[(ep,3) for ep in eps if ep['task_id'] in hold]
 def values(data):
  pred=[];truth=[]
  with torch.no_grad():
   for ep,k in data:
    t=pool[ep['task_id']];x=reconstruct(t,ep,k);b=e.batch(t,x);a=torch.tensor([AA.index(t['sequence'][len(x)])],device='cuda');g=torch.tensor([ep['geometry'][k]],device='cuda',dtype=torch.float64);mask=torch.ones(1,3,device='cuda',dtype=torch.bool)
    with torch.autocast('cuda',dtype=torch.bfloat16):v=[q(b,a,g,mask).float() for q in qs]
    pred.append([float(z) for z in v]);truth.append(ep['reward']*CFG['gamma']**(3-k))
  return np.asarray(pred),np.asarray(truth)
 trprobe=train[::max(1,len(train)//128)][:128];teprobe=test[::max(1,len(test)//128)][:128]
 base_train=np.array([ep['reward']*CFG['gamma']**(3-k) for ep,k in trprobe]);base_test=np.array([ep['reward'] for ep,k in teprobe]);mean=float(np.mean([ep['reward']*CFG['gamma']**(3-k) for ep,k in train]));vtrain=float(np.mean((base_train-mean)**2));vtest=float(np.mean((base_test-mean)**2));curve=[]
 for u in range(updates):
  budget()
  for opt in opts:opt.zero_grad(set_to_none=True)
  losses=[]
  for j in range(4):
   ep,k=train[(u*17+j*101)%len(train)];t=pool[ep['task_id']];x=reconstruct(t,ep,k);b=e.batch(t,x);a=torch.tensor([AA.index(t['sequence'][len(x)])],device='cuda');g=torch.tensor([ep['geometry'][k]],device='cuda',dtype=torch.float64);mask=torch.ones(1,3,device='cuda',dtype=torch.bool);target=ep['reward']*CFG['gamma']**(3-k)
   for q in qs:
    with torch.autocast('cuda',dtype=torch.bfloat16):value=q(b,a,g,mask).float();loss=(value-target).square().mean()/4
    if not torch.isfinite(loss):raise FloatingPointError('critic known-return loss')
    loss.backward();losses.append(float(loss.detach()))
  for q,opt in zip(qs,opts):torch.nn.utils.clip_grad_norm_(q.parameters(),5,error_if_nonfinite=True);opt.step()
  if (u+1)%50==0:curve.append(dict(update=u+1,loss=sum(losses)));print('CRITIC_CALIBRATION',u+1,curve[-1]['loss'],flush=True)
 trainpred,truth=values(trprobe);testpred,tt=values(teprobe);tmse=np.mean((trainpred-truth[:,None])**2,0);hmse=np.mean((testpred-tt[:,None])**2,0)
 passed=bool(vtrain>1e-8 and vtest>1e-8 and np.isfinite(trainpred).all() and np.isfinite(testpred).all() and (tmse<=.5*vtrain).all() and (hmse<=.9*vtest).all())
 out=dict(PASS=passed,updates=updates,critic_updates_each=updates,TRAIN_fit_identities=48,TRAIN_heldout_identities=32,training_probe=len(trprobe),heldout_terminal_probe=len(teprobe),constant_prediction=mean,train_constant_MSE=vtrain,heldout_constant_MSE=vtest,train_MSE=tmse.tolist(),heldout_terminal_MSE=hmse.tolist(),curves=curve,heldout_predictions=testpred.tolist(),heldout_returns=tt.tolist(),reason=None if passed else 'known-return fit/calibration gate failed; no SAC actor updates authorized')
 save('CRITIC_CALIBRATION.json',out);return out

def sac_update(e,qs,tq,qo,pool,replay,u,dual):
 # Fixed-AA specialization of crisp_v3.optimization.Learner. No20-AA mixture.
 losses=[];norms=[]
 for opt in qo:opt.zero_grad(set_to_none=True)
 for j in range(4):
  ep=replay[(u*53+j*97)%len(replay)];k=(u+j)%4;t=pool[ep['task_id']];x=reconstruct(t,ep,k);b=e.batch(t,x);a=torch.tensor([AA.index(t['sequence'][len(x)])],device='cuda');g=torch.tensor([ep['geometry'][k]],device='cuda',dtype=torch.float64);mask=torch.ones(1,3,device='cuda',dtype=torch.bool)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   if k==3:target=torch.tensor([ep['reward']],device='cuda')
   else:
    nx=reconstruct(t,ep,k+1);mu,ls,rm,rl,nb,na=e.distribution(t,nx);ng,nlp=sample(mu,ls,mask[0]);nkl=e.kl(mu,ls,rm,rl);v=torch.minimum(*[q(nb,torch.tensor([na],device='cuda'),ng[None],mask).float() for q in tq])-CFG['alpha']*nlp/3-dual*nkl
    target=semimarkov(torch.zeros(1,device='cuda'),CFG['gamma'],torch.ones(1,device='cuda',dtype=torch.long),torch.zeros(1,device='cuda',dtype=torch.bool),v).float()
  for q in qs:
   with torch.autocast('cuda',dtype=torch.bfloat16):value=q(b,a,g,mask).float();loss=(value-target).square().mean()/4
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite Bellman critic')
   loss.backward();losses.append(float(loss.detach()))
 for q,opt in zip(qs,qo):norms.append(float(torch.nn.utils.clip_grad_norm_(q.parameters(),5,error_if_nonfinite=True)));opt.step()
 with torch.no_grad():
  for q,target in zip(qs,tq):
   for a,b in zip(target.parameters(),q.parameters(),strict=True):a.lerp_(b,CFG['polyak'])
 return dict(critic_loss=sum(losses),critic_gradient_norm=norms)

def policy_update(e,label,o,sc,dual,selected,bytask,pool,u,qs=None):
 e.actor.train()
 o.zero_grad(set_to_none=True);nlls=[];kls=[];cached=[];scalars=[]
 if qs:
  for q in qs:q.requires_grad_(False)
 try:
  for j,tid in enumerate(selected):
   t=pool[tid];valid=bytask[tid];weights=np.array([r['success']+.1*r['joint'] if r['valid'] else 0 for r in valid]);teacher=None
   if weights.sum()>0:teacher=valid[np.random.choice(len(valid),p=weights/weights.sum())]
   if label=='B':
    if teacher is not None:nll,kl,c=teacher_step(e,t,teacher);loss=nll+dual*kl;cached.extend(c);nlls.append(float(nll.detach()));kls.append(float(kl.detach()))
    else:
     nll,kl=native_prior(e,t,u%4);loss=.1*nll+dual*kl;kls.append(float(kl.detach()))
   else:
    ep=valid[(u+j)%len(valid)];k=(u+j)%4;x=reconstruct(t,ep,k)
    with torch.autocast('cuda',dtype=torch.bfloat16):mu,ls,rm,rl,b,a=e.distribution(t,x)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):h=[q.encoder(b) for q in qs]
    ng,nlp=sample(mu,ls,torch.ones(3,dtype=torch.bool,device='cuda'));kl=e.kl(mu,ls,rm,rl)
    with torch.autocast('cuda',dtype=torch.bfloat16):qv=torch.minimum(*[q.candidate(hh,torch.tensor([a],device='cuda'),ng[None],torch.ones(1,3,dtype=torch.bool,device='cuda')).float() for q,hh in zip(qs,h)])
    loss=(CFG['alpha']*nlp/3-qv+dual*kl).mean();kls.append(float(kl.detach()));cached.append((b,a,mu.detach().clone(),ls.detach().clone()))
   bc,nkl=native_prior(e,t,(u+j)%4);loss=(loss+.1*bc)/4
   if not torch.isfinite(loss):raise FloatingPointError('actor nonfinite')
   loss.backward();scalars.append(float(loss.detach()))
 finally:
  if qs:
   for q in qs:q.requires_grad_(True)
 norm=float(torch.nn.utils.clip_grad_norm_(e.actor.parameters(),5,error_if_nonfinite=True));o.step();sc.step();dual=max(0.,dual+.05*(float(np.mean(kls))-.8))
 # Step trust-region is measured on the exact immutable replay observations.
 stepk=[]
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for b,a,om,ol in cached:
   _,m,l=e.actor(b);nm,nl=m[0,a],l[0,a];stepk.append(float((ol-nl+.5*((nl.exp()/ol.exp()).square()+((nm-om)/ol.exp()).square())-.5).mean()))
 if stepk and np.mean(stepk)>.02:raise RuntimeError('per-update training-state trust-region KL>.02; restore prior durable checkpoint')
 return dual,dict(loss=sum(scalars),weighted_NLL=float(np.mean(nlls)) if nlls else None,KL=float(np.mean(kls)),gradient_norm=norm,dual=dual,step_KL=float(np.mean(stepk)) if stepk else 0,lr=sc.get_last_lr()[0])

def restore(e,o,sc,rec,qs=None,tq=None,qo=None):
 assert sha(rec['path'])==rec['sha256'];st=torch.load(rec['path'],map_location='cpu',weights_only=False);assert state_digest(st)==rec['state_digest']
 e.actor.load_state_dict(st['actor']);o.load_state_dict(st['optimizer']);sc.load_state_dict(st['scheduler']);set_rng(st['rng'])
 if qs:
  for q,v in zip(qs,st['critic']):q.load_state_dict(v)
  for q,v in zip(tq,st['target_critic']):q.load_state_dict(v)
  for q,v in zip(qo,st['critic_optimizer']):q.load_state_dict(v)
 return st

def engineering():
 config_check();s.seed(7717);e=Engine();o,sc=actor_optimizer(e);pool={t['task_id']:t for t in tasks('TRAIN')};eps=offline();bt=collections.defaultdict(list)
 for ep in eps:bt[ep['task_id']].append(ep)
 # Physical-law Jacobian, gradient, Bellman terminal masking, and compact replay tests.
 mu=torch.tensor([.9,.2,-.1],dtype=torch.float64,requires_grad=True);ls=torch.tensor([-.5,-.3,-.2],dtype=torch.float64);mask=torch.ones(3,dtype=torch.bool);g,lp=sample(mu,ls,mask,torch.tensor([.2,-.4,.3]));grad=torch.autograd.grad(lp+g.sum(),mu)[0];assert torch.isfinite(grad).all() and grad.abs().sum()>0
 r=torch.tensor([.5,.5]);v=torch.tensor([100.,2.]);term=torch.tensor([True,False]);target=semimarkov(r,.995,torch.ones(2,dtype=torch.long),term,v);assert torch.allclose(target,torch.tensor([.5,2.49]))
 errors=[]
 for ep in eps[::max(1,len(eps)//32)][:32]:
  x=reconstruct(pool[ep['task_id']],ep,4);errors.append(float((x[-4:].cpu()-torch.tensor(ep['tail'],dtype=torch.float64)).abs().max()));assert errors[-1]<1e-8
 # One complete real-actor update, full recovery, exact next-update replay.
 ids=list(pool);dual,row=policy_update(e,'B',o,sc,.1,ids[:4],bt,pool,0);rec=checkpoint(e,'ENGINEERING',1,o,sc,dual,[],dict(eligible=False))
 dual2,row2=policy_update(e,'B',o,sc,dual,ids[4:8],bt,pool,1);signature=state_digest(dict(actor=e.actor.state_dict(),optimizer=o.state_dict(),scheduler=sc.state_dict(),dual=dual2,rng=rng_state()))
 st=restore(e,o,sc,rec);dual3,row3=policy_update(e,'B',o,sc,st['dual'],ids[4:8],bt,pool,1);signature2=state_digest(dict(actor=e.actor.state_dict(),optimizer=o.state_dict(),scheduler=sc.state_dict(),dual=dual3,rng=rng_state()));assert signature==signature2 and row2==row3
 e.verify();save('ENGINEERING_TESTS.json',dict(PASS=True,fixed_AA=True,transformed_density_finite_gradient=True,terminal_and_nonterminal_Bellman=True,replay_max_error_A=max(errors),full_actor_optimizer_scheduler_rng_exact_next_update=True,checkpoint=rec,engineering_actor_executions=3,actor_parameters=sum(p.numel() for p in e.actor.parameters()),actor_trainable_parameters=sum(p.numel() for p in e.actor.parameters() if p.requires_grad),J1_tensor_SHA=e.prior.identity,parent_SHA=PSHA,peak_GPU=torch.cuda.max_memory_allocated(),calls=dict(e.calls)))

def train(label):
 config_check();assert not (R/f'results/{label}_STOP.json').exists();s.seed(7717);e=Engine();o,sc=actor_optimizer(e);eps=offline();pool={t['task_id']:t for t in tasks('TRAIN')};ids=sorted(pool,key=rank);by=collections.defaultdict(list)
 for ep in eps:by[ep['task_id']].append(ep)
 online=[];dual=.1;monitor=dict(eligible=False,critic_updates=0);qs=tq=qo=None;u=0;start=time.monotonic();logs=[];reason=None
 base=read(R/'results/A_DEV_K2.json');b2=read(R/'results/A_DEV_H2_K2.json')
 if label!='B':
  qs,tq,qo=init_critics()
  if label=='D':
   ix=read(R/'checkpoints/B_INDEX.json');assert ix['best'];assert not (R/'results/B_STOP.json').exists() and not (R/'results/C_STOP.json').exists()
   e.actor.load_state_dict(torch.load(ix['best']['path'],map_location='cpu',weights_only=False)['actor'])
  # Fill Adam critic moments before measuring a complete recovery bundle.
  cal=critic_known_return(e,eps,qs,qo)
  for q,t in zip(qs,tq):t.load_state_dict(q.state_dict())
  monitor['critic_updates']=cal['updates'];monitor['calibration_PASS']=cal['PASS']
  reason=None if cal['PASS'] else 'critic known-return calibration failed; SAC actor remains unchanged'
  rec=checkpoint(e,label,0,o,sc,dual,online,monitor,qs,tq,qo,reason)
  save(f'results/{label}_STORAGE_ADMISSION.json',dict(complete_checkpoint_bytes=rec['bytes'],free_bytes=__import__('shutil').disk_usage(R).free,staging_plus_headroom=rec['bytes']+2*2**30+64*2**20,admitted=True,critic_parameters_each=[sum(p.numel() for p in q.parameters()) for q in qs],target_parameters_each=[sum(p.numel() for p in q.parameters()) for q in tq]))
  if reason:
   save(f'results/{label}_STOP.json',dict(reason=reason,actor_updates=0,critic_calibration_updates=250,checkpoint=rec));save(f'results/{label}_FINISHED.json',dict(actor_updates=0,reason=reason,seconds=time.monotonic()-start,network_calls=dict(e.calls),peak_GPU=torch.cuda.max_memory_allocated()));return
  # Exact next fixed-AA SAC transaction from the CPU-deserialized complete
  # calibrated checkpoint; engineering executions never count as research steps.
  def trial():
   report=sac_update(e,qs,tq,qo,pool,eps,0,.1);du,pr=policy_update(e,label,o,sc,.1,ids[:4],by,pool,0,qs)
   sig=state_digest(dict(actor=e.actor.state_dict(),optimizer=o.state_dict(),scheduler=sc.state_dict(),q=[q.state_dict() for q in qs],target=[q.state_dict() for q in tq],qoptimizer=[op.state_dict() for op in qo],dual=du,rng=rng_state()))
   return sig,report,pr
  first=trial();restore(e,o,sc,rec,qs,tq,qo);second=trial();assert first==second;restore(e,o,sc,rec,qs,tq,qo)
  save(f'{label}_SAC_REPLAY_TEST.json',dict(PASS=True,full_state_next_transaction_exact=True,engineering_actor_executions=2,engineering_critic_executions_each=2,checkpoint=rec,scope='CPU deserialization and RNG/full-state restore in same process, no fresh-process guarantee claimed'))
 else:rec=checkpoint(e,label,0,o,sc,dual,online,monitor)
 try:
  for update in range(500):
   budget();u=update
   t=pool[ids[u%len(ids)]];episode=rollout(e,t,ns(t['task_id'],10000+u,'ONLINE'),f'{label}_{u}');online.append(episode);by[t['task_id']].append(episode)
   with (R/f'results/{label}_online.jsonl').open('a') as f:f.write(json.dumps(episode,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
   selected=[ids[(4*u+j)%len(ids)] for j in range(4)];extra={}
   if qs:
    extra=sac_update(e,qs,tq,qo,pool,eps+online,u,dual);monitor['critic_updates']+=1
   dual,row=policy_update(e,label,o,sc,dual,selected,by,pool,u,qs);u=update+1;row.update(update=u,task_ids=selected,online_reward=episode['reward'],online_success=episode['success'],critic_updates=monitor['critic_updates'],**extra);logs.append(row)
   with (R/f'results/{label}_updates.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
   if u in CFG['milestones']:
    ev=evaluate(e,f'{label}{u}','DEV');h2=evaluate(e,f'{label}{u}','DEV_H2');nm=eligibility(ev,h2,base,b2);nm.update(best_score=monitor.get('best_score',[-1,-1]),critic_updates=monitor['critic_updates']);monitor=nm;reason='; '.join(nm['reasons']) or None
    if u==250 and not reason:
     if ev['K1']['success']<base['K1']['success']:reason='predeclared250 continuation gate: DEV K1 below parent'
     elif label=='B':
      first=[x['weighted_NLL'] for x in logs[:25] if x['weighted_NLL'] is not None];last=[x['weighted_NLL'] for x in logs[-25:] if x['weighted_NLL'] is not None]
      if not first or np.mean(first)-np.mean(last)<.05*abs(np.mean(first)):reason='predeclared250 continuation gate: insufficient RWR NLL improvement'
     elif np.mean([x['online_reward'] for x in logs[-50:]])<np.mean([x['reward'] for x in eps]):reason='predeclared250 continuation gate: TRAIN self-return below parent'
    rec=checkpoint(e,label,u,o,sc,dual,online,monitor,qs,tq,qo,reason)
    if reason:break
   elif u%50==0:rec=checkpoint(e,label,u,o,sc,dual,online,monitor,qs,tq,qo)
   if u%25==0:
    save('LIVE_STATUS.json',dict(stage='TRAINING',arm=label,update=u,pid=os.getpid(),seconds=time.monotonic()-start,free_bytes=__import__('shutil').disk_usage(R).free,peak_GPU=torch.cuda.max_memory_allocated()));print('TRAIN',label,u,flush=True)
 except (RuntimeError,FloatingPointError,OSError,InterruptedError) as ex:
  last=read(R/f'checkpoints/{label}_INDEX.json')['latest'];st=restore(e,o,sc,last,qs,tq,qo);u=st['update'];dual=st['dual'];online=st['online_replay'];monitor=st['monitoring'];reason=repr(ex)+'; rolled back to last verified complete transaction';rec=checkpoint(e,label,u,o,sc,dual,online,monitor,qs,tq,qo,reason)
 e.verify()
 if reason:save(f'results/{label}_STOP.json',dict(reason=reason,actor_updates=u,checkpoint=rec,no_invalid_promotion=True))
 save(f'results/{label}_FINISHED.json',dict(actor_updates=u,critic_updates=monitor['critic_updates'],environment_episodes=len(online),environment_actions=4*len(online),reason=reason,seconds=time.monotonic()-start,network_calls=dict(e.calls),peak_GPU=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),J1_frozen=True))

def baseline():
 config_check();s.seed(7717);e=Engine();e.actor.requires_grad_(False);a=evaluate(e,'A','DEV');b=evaluate(e,'A','DEV_H2');e.verify();save('BASELINE_INTEGRITY.json',dict(actor_fingerprint=e.parent_fingerprint,J1_tensor_SHA=e.prior.identity,parent_SHA=PSHA,source=commit(),baseline_KL_pass=bool(a['summary']['KL']<=1 and b['summary']['KL']<=1)))
 if a['summary']['KL']>1 or b['summary']['KL']>1:raise RuntimeError('frozen parent fails original KL safeguard on prospective DEV; no training admission')

def audit():
 e=Engine();e.actor.requires_grad_(False);t=read(OLD/'manifests/AUDIT_H4_native.json')['tasks'][0];records=read(OLD/'results/INITIAL_00000_AUDIT_H4_native.json')['records'];out=[]
 for seed in [7101,7102]:
  a=rollout(e,t,seed,'audit');b=next(r for r in records if r['task_id']==t['task_id'] and r['seed']==seed);err=float(np.max(np.abs(np.array(a['tail'])-np.array(b['tail']))));assert err<1e-8 and abs(a['KL']-b['KL'])<1e-6 and a['success']==b['success'];out.append(dict(seed=seed,tail_error_A=err,KL_error=abs(a['KL']-b['KL'])))
 e.verify();save('PARENT_ARCHITECTURE_AUDIT.json',dict(PASS=True,strict_restore=True,parent_SHA256=PSHA,parent_actor_state_digest=e.parent_fingerprint,J1_tensor_SHA=e.prior.identity,actor_parameters=sum(p.numel() for p in e.actor.parameters()),J1_parameters=sum(p.numel() for p in e.prior.model.parameters()),actor_shape=dict(width=448,layers=7,heads=8),action_distribution='fixed-AA conditional3Gaussian latents with exp/logistic/tanh transforms, temperature1; mean3component conditional KL to J1',exact_legacy_parity=out,source=commit()))

def shifted():
 # Same original frozen-V3 two-action prefix perturbation as V5/V6, no outcome filtering.
 e=Engine();e.actor.load_state_dict(e.initial);e.actor.requires_grad_(False);cache={};out=[]
 for t in tasks('CONFIRM'):
  key=t['identity']
  if key not in cache:
   x=torch.tensor(t['native_prefix'][:-2],device='cuda',dtype=torch.float64);epsilon=torch.randn(2,3,dtype=torch.float64,generator=torch.Generator().manual_seed(1101+int(key[:6],16))).cuda()
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for k in range(2):
     mu,ls,*_=e.distribution(t,x);x=torch.cat([x,s.next_ca(x,s.physical(mu.double()+ls.double().exp()*epsilon[k]))[None]],0)
   cache[key]=x.cpu().tolist()
  z=copy.deepcopy(t);z['native_prefix']=cache[key];z['shift_note']='two original frozen-V3 actions, native target retained; feasibility may change';out.append(z)
 save('manifests/CONFIRM_SHIFTED.json',dict(tasks=out,no_filtering=True,source_policy='immutable original V3',seeds='1101+int(identity[:6],16)'))

def final_evaluate():
 config_check();e=Engine();e.actor.requires_grad_(False)
 for label in ['A','B','C','D']:
  if label=='A':version='A'
  else:
   ixpath=R/f'checkpoints/{label}_INDEX.json'
   if not ixpath.exists():continue
   ix=read(ixpath);rec=ix['best']
   if rec is None:
    save(f'results/{label}_FINAL_STATUS.json',dict(evaluated=False,reason='no eligible trained checkpoint; stopped latest preserved, no retrospective confirmation of an ineligible state'));continue
   assert sha(rec['path'])==rec['sha256'];st=torch.load(rec['path'],map_location='cpu',weights_only=False);e.actor.load_state_dict(st['actor']);version=f'{label}_BEST{rec["update"]}';save(f'results/{label}_FINAL_STATUS.json',dict(evaluated=True,checkpoint=rec,version=version))
  for panel,K in [('CONFIRM',32),('CONFIRM_SHIFTED',32),('CONFIRM_H2',2)]:evaluate(e,version,panel,K)
  e.verify()
 save('FINAL_EVALUATION_COMPLETED.json',dict(complete=True,J1_frozen=True,UTC=time.time()))

def campaign():
 config_check()
 commands=[('audit',),('baseline',),('collect',),('engineering',),('train','--arm','B'),('train','--arm','C')]
 def call(args):
  name='_'.join(args).replace('--','');p=R/f'logs/{name}.log'
  with p.open('x') as f:result=subprocess.run([sys.executable,'-B',str(R/'source/scripts/e017_v7/study.py'),*args],stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
  print('STAGE',name,result.returncode,flush=True)
  if result.returncode:save('CAMPAIGN_STOP.json',dict(stage=name,returncode=result.returncode,UTC=time.time()));return False
  return True
 for args in commands:
  if not call(args):return
 # D is admitted only by the sealed scientific gates; never restart a stopped arm.
 B=read(R/'results/B_FINISHED.json');C=read(R/'results/C_FINISHED.json');admit=False
 if not B['reason'] and not C['reason'] and B['actor_updates']>=250 and C['actor_updates']>=250:
  b=read(R/f'results/B{B["actor_updates"]}_DEV_K2.json');c=read(R/f'results/C{C["actor_updates"]}_DEV_K2.json');a=read(R/'results/A_DEV_K2.json');admit=b['K1']['success']>a['K1']['success'] and c['K1']['success']>=a['K1']['success']
 save('D_ADMISSION.json',dict(admitted=admit,reason='sealed B/C safety/250-update/K1 gates' if admit else 'B/C did not both pass sealed safety/250-update/K1 gates; no D training launched'))
 if admit and not call(('train','--arm','D')):return
 if not call(('final_evaluate',)):return
 save('CAMPAIGN_COMPLETED.json',dict(complete=True,UTC=time.time(),no_TEST=True,no_high_level=True))

if __name__=='__main__':
 signal.signal(signal.SIGTERM,handle);signal.signal(signal.SIGINT,handle);torch.set_num_threads(1)
 parser=argparse.ArgumentParser();parser.add_argument('stage');parser.add_argument('--arm');args=parser.parse_args()
 if args.stage=='train':train(args.arm)
 else:globals()[args.stage]()
