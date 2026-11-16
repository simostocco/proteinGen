"""V7.1 fixed-policy expected-return diagnostics; no actor optimization."""
import collections
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time

import numpy as np
import torch

R=Path('/mnt/d/Simone/proteinGen_data/e017_v71_expected_return_calibration_20261023_v1')
V7=R.parent/'e017_v7_amortized_lowlevel_20261023_v1'
v=runpy.run_path(str(R/'source/scripts/e017_v7/study.py'))
sys.path.insert(0,str(R/'source/src'))
from protein_joint_sac.crisp_v3.recovery import atomic_json,atomic_checkpoint,sha,state_digest,serialized_size,StoragePolicy
from protein_joint_sac.crisp_v3.networks import LowQ,PROFILES
from protein_joint_sac.crisp_v3.storage import rng_state,set_rng
from protein_joint_sac.contracts import AA
STOP=False
CFG=dict(schema='E017_V71_v1',seed=87171,gamma=.995,continuation='V6 parent V5D250 fixed native AA policy; temperature1',
 fit_identities=256,dev_identities=64,confirm_identities=64,expected_fit_identities=32,expected_dev_identities=16,expected_confirm_identities=16,expected_VAL_identities=8,
 fixed_actions_per_state=2,suffix_replicates=16,stages=[0,1,2,3],critic_lr=1e-4,critic_batch=4,clip=5.,polyak=.005,
 milestones=[0,100,250,500,750],maximum_new_updates=750,maximum_total_updates_including_V7=1000,
 warmstart='V7 stopped twin critics only; all48previous fit identities forced into newFIT; old32diagnostic identities excluded from newCONFIRM; fresh critic AdamW; no actor optimizer',
 target='unregularized discounted terminal task return; V7calibration actually used this target. Soft-reference recorded separately with alpha.001,lambda.1,future-only entropy/KL',
 continue250to500='both critics DEV nonterminal expected-reference MSE <= best simple predictor MSE and improve >=10% vs warm-start0; no nonfinite or mean overestimate>.25',
 continue500to750='both critics expected DEV MSE <=.95*best simple baseline, informative action ranking point>=.55 and absolute mean bias<=.1',
 selection='DEV minimum mean twin nonterminal expected-reference MSE; tie earliest. NoCONFIRM/VALIDATIONselection',
 promotion='onTRAIN-CONFIRM and smallVALIDATION: each twin nonterminal expected RMSE<=.9*best simple RMSE; paired identity-bootstrap95%MSE difference upper<0; biasabs<=.1; confident action ranking>=.6 and95%lower>.5 with>=10supportedpairs,>=8groups; no stage MSE>1.1*stage-constant. NoSACupdates evenifGO.',
 supported_pair='nonterminal paired suffix reward difference absolute>=.1 and >1.96*paired MCstandarderror; terminal exact pairs separate',
 simple='stage constants; ridge lambda1 fixed; two-layer64/32MLP, fresh seed,1000CPUupdates, DEVselect100/250/500/1000; no future features',
 RWR='fixed64TRAINidentities all8existing behavior trajectories, evaluate exact same prefixes/actions under parent,+25,+250; event mass from sampled outcomes/importance diagnostics, density not probability',
 RWR_validation='reuse full V7 parent/+25 pools; new +250 diagnostic on sameCONFIRMnative/shifted64tasks,K32,exactV7streams. OldV7stops not reversed or promoted.',
 stage_variant='not added: original context already contains explicit remaining/H and relative pose; no missing-stage information demonstrated',
 reserve_bytes=2*2**30,checkpoint_hint=520*2**20,actor_updates=0,high_level_updates=0,reachability_updates=0,TEST=False)
POLICY=StoragePolicy(min_headroom_bytes=2*2**30,near_term_growth_bytes=64*2**20,checkpoint_hint_bytes=520*2**20)
def read(p):return json.loads(Path(p).read_text())
def save(p,x):atomic_json(R/p,x)
def rank(s):return hashlib.sha256(('87171|'+s).encode()).hexdigest()
def ns(*s):return int(rank('|'.join(map(str,s)))[:15],16)
def source():return subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip()
def budget():
 POLICY.require(R,520*2**20)
 if STOP:raise InterruptedError('cooperative stop at complete boundary')
 if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()>6*2**30:raise RuntimeError('CUDA6GiBgate')
def episodes():return [json.loads(s) for s in (V7/'results/TRAIN_candidates.jsonl').read_text().splitlines()]
def taskmap():return {t['task_id']:t for t in read(V7/'manifests/TRAIN.json')['tasks']}
def unit(a):return a/np.linalg.norm(a)
def frame(x):
 a=unit(x[-1]-x[-2]);n=unit(np.cross(x[-2]-x[-3],a));return np.stack([a,np.cross(n,a),n])
def append(x,g):
 u=unit(x[-1]-x[-2]);e=x[-3]-x[-2];e=unit(e-e.dot(u)*u);n=np.cross(u,e);d,a,t=g
 return np.concatenate([x,(x[-1]+d*(-np.cos(a)*u+np.sin(a)*(np.cos(t)*e+np.sin(t)*n)))[None]])
def pose(t,x):
 dp=frame(x)@(np.asarray(t['target'])-x[-1]);rot=frame(x)@np.asarray(t['orientation']).T;angle=math.acos(np.clip((np.trace(rot)-1)/2,-1,1));return dp,rot,angle
def features(t,x,g,k):
 nx=append(x,g);pre,pr,po=pose(t,x);post,rot,ori=pose(t,nx);d=np.linalg.norm(np.diff(nx,axis=0),axis=1);n=len(nx);ix=np.arange(n);clash=int(((np.linalg.norm(nx[:,None]-nx[None,:],axis=-1)<2)&(ix[:,None]-ix[None,:]>=3)).sum());rg=float(np.sqrt(np.square(nx-nx.mean(0)).sum(-1).mean())/n**(1/3));one=np.zeros(20);one[AA.index(t['sequence'][len(x)])]=1
 latent=[np.log(g[0]),np.log(g[1]/(math.pi-g[1])),np.arctanh(g[2]/math.pi)]
 return np.r_[np.eye(4)[k],(3-k)/4,pre/8,np.linalg.norm(pre)/8,po/math.pi,post/8,np.linalg.norm(post)/8,ori/math.pi,rot.flatten(),g[0]/3.8,g[1]/math.pi,np.sin(g[2]),np.cos(g[2]),latent,d.min()/3.8,d.max()/3.8,clash*100/n,rg/10,float(clash>0),float(d.min()<2.8504200470168195 or d.max()>4.052024993913842),t['N']/500,one]

def prepare():
 pool=taskmap();ep=episodes();oldids=sorted(pool,key=lambda h:hashlib.sha256(('7717|calibration'+h).encode()).hexdigest());oldfit=oldids[:48];oldhold=oldids[48:80];others=sorted(set(pool)-set(oldids[:80]),key=rank)
 fit=oldfit+others[:208];dev=oldhold+others[208:240];confirm=others[240:304];assert [len(fit),len(dev),len(confirm)]==[256,64,64]
 groups=[{pool[h]['protected_group_id'] for h in a} for a in [fit,dev,confirm]];assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
 split={h:n for n,a in [('FIT',fit),('DEV',dev),('CONFIRM',confirm)] for h in a};picked={n:sorted(a,key=lambda h:rank('expected'+h))[:number] for n,a,number in [('FIT',fit,32),('DEV',dev,16),('CONFIRM',confirm,16)]}
 val=[t for t in read(V7/'manifests/CONFIRM.json')['tasks'] if t['variant']=='native'];val=sorted(val,key=lambda t:rank('VAL'+t['task_id']))[:8];picked['VALIDATION']=[t['task_id'] for t in val];shift={t['task_id']:t for t in read(V7/'manifests/CONFIRM_SHIFTED.json')['tasks']};shiftval=[shift[t['task_id']] for t in val];picked['VALIDATION_SHIFTED']=picked['VALIDATION']
 save('manifests/SPLITS.json',dict(task_split=split,FIT=fit,DEV=dev,CONFIRM=confirm,expected=picked,VALIDATION_tasks=val,VALIDATION_SHIFTED_tasks=shiftval,RWR_fixed_ids=sorted(pool,key=lambda h:rank('RWR'+h))[:64],old_V7_fit=oldfit,old_V7_holdout=oldhold))
 X=[];Y=[];stage=[];index=[];part=[];succ=[];err=[]
 for ei,e in enumerate(ep):
  t=pool[e['task_id']]
  for k in range(4):
   x=np.asarray(t['native_prefix']+e['tail'][:k]);g=np.asarray(e['geometry'][k]);nx=append(x,g);err.append(float(np.max(np.abs(nx[-1]-np.asarray(e['tail'][k])))));assert err[-1]<1e-8
   f=features(t,x,g,k);X.append(f);Y.append(e['reward']*.995**(3-k));stage.append(k);index.append([ei,k]);part.append(split.get(e['task_id'],'UNUSED'));succ.append(e['success'])
 np.savez_compressed(R/'results/realized.npz',X=np.asarray(X),y=np.asarray(Y),stage=np.asarray(stage),success=np.asarray(succ),split=np.asarray(part),index=np.asarray(index))
 example=ep[0];t=pool[example['task_id']];k=1;x=np.asarray(t['native_prefix']+example['tail'][:k]);g=np.asarray(example['geometry'][k]);changed=copy.deepcopy(t);changed['hidden_native_future']=[[999.]*3]*4;pos=len(x)+1;changed['sequence']=t['sequence'][:pos]+'A'*(len(t['sequence'])-pos);assert np.array_equal(features(t,x,g,k),features(changed,x,g,k))
 save('RETURN_AUDIT.json',dict(stage_counts=dict(collections.Counter(stage)),max_coordinate_reconstruction_error_A=max(err),return_exponents=[3,2,1,0],original_V7_stage_probe=[128,0,0,0],original_fit_identities_restricted_to_FIT=True,features_dim=len(X[0]),future_coordinates_and_unrevealed_AA_perturbation_invariant=True,features='current prefix+current fixedAA+proposed action+deterministic post-action residual/validity only; no subsequent trajectory or unrevealedAA',unregularized=True,soft_training_in_V7=False))

def seal():
 assert not (R/'PROTOCOL.json').exists();paths=[V7/'FINAL_E017_V7_SCIENTIFIC_REPORT.md',V7/'CRITIC_CALIBRATION.json',V7/'ADMISSION_DIAGNOSTIC_LIMITATIONS.json',V7/'results/TRAIN_candidates.jsonl',V7/'manifests/TRAIN.json',V7/'checkpoints/C_00000_stop.pt',V7/'checkpoints/B_00025.pt',V7/'checkpoints/B_00250_stop.pt',R/'manifests/SPLITS.json',R/'results/realized.npz',R/'source/scripts/e017_v71/study.py']
 save('PROTOCOL.json',dict(config=CFG,source=source(),input_SHA256={str(p):sha(p) for p in paths},UTC=time.time()))
def check():
 p=read(R/'PROTOCOL.json');assert p['source']==source()
 for path,h in p['input_SHA256'].items():assert sha(path)==h,path

def collect_expected():
 start=time.monotonic();check();e=v['Engine']();e.actor.requires_grad_(False);fp=e.parent_fingerprint;pool=taskmap();splits=read(R/'manifests/SPLITS.json');pool.update({t['task_id']:t for t in splits['VALIDATION_tasks']});anchor={r['task_id']:r for r in episodes() if r['candidate_index']==0};rows=[];samplelog=R/'results/suffix_samples.jsonl';path=R/'results/expected_rows.jsonl';done={json.loads(s)['key'] for s in path.read_text().splitlines()} if path.exists() else set()
 # Atomic per state-action means: incomplete suffix groups are evidence but never a committed label.
 with path.open('a') as out,samplelog.open('a') as samples:
  for split,ids in splits['expected'].items():
   for tid in ids:
    t=next(z for z in splits['VALIDATION_SHIFTED_tasks'] if z['task_id']==tid) if split=='VALIDATION_SHIFTED' else pool[tid];base=anchor.get(tid)
    if base is None:base=v['rollout'](e,t,ns('anchor',tid),'V71_parent')
    for k in range(4):
     x=v['reconstruct'](t,base,k)
     with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):mu,ls,rm,rl,*_=e.distribution(t,x)
     for a in range(2):
      key=f'{split}|{tid}|{k}|{a}'
      if key in done:continue
      budget();epsilon=torch.randn(3,dtype=torch.float64,generator=torch.Generator().manual_seed(ns('action',tid,k,a))).cuda();u=mu.double()+ls.double().exp()*epsilon;g,actionlp=v['sample'](mu,ls,torch.ones(3,dtype=torch.bool,device='cuda'),epsilon);g=g.detach();nx=torch.cat([x,v['s'].next_ca(x,g)[None]],0);ret=[];success=[];soft=[];duration=1 if k==3 else 16
      for m in range(duration):
       y=nx.clone();seed=ns('suffix',tid,k,m);eps=torch.randn(4,3,dtype=torch.float64,generator=torch.Generator().manual_seed(seed)).cuda();actions=[];lp=[];kl=[]
       with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for j in range(k+1,4):
         mm,ll,mr,lr,*_=e.distribution(t,y);gg,_=v['sample'](mm,ll,torch.ones(3,dtype=torch.bool,device='cuda'),eps[j]);lprob=float(v['log_prob'](mm.double(),ll.double(),gg,torch.ones(3,dtype=torch.bool,device='cuda')));kk=float(e.kl(mm,ll,mr,lr));lp.append(lprob);kl.append(kk);actions.append(gg.cpu().tolist());y=torch.cat([y,v['s'].next_ca(y,gg)[None]],0)
        metric=v['s'].metric(t,y,seed)
       rr,joint=v['reward'](metric);target=.995**(3-k)*rr;softret=target-sum(.995**(j-k)*(.001*ll/3+.1*kk) for j,ll,kk in zip(range(k+1,4),lp,kl));ret.append(target);soft.append(softret);success.append(bool(metric['success']))
       samples.write(json.dumps(dict(key=key,replicate=m,seed=seed,continuation_policy_SHA=v['PSHA'],reward=rr,success=metric['success'],valid=metric['valid'],return_task=target,return_soft=softret,future_geometry=actions,future_logprob=lp,future_KL=kl))+'\n')
      samples.flush();os.fsync(samples.fileno());xx=x.cpu().numpy();gg=g.cpu().numpy();record=dict(key=key,split=split,task_id=tid,identity=t['identity'],group=t['protected_group_id'],stage=k,action_index=a,anchor_geometry=base['geometry'][:k],geometry=gg.tolist(),latent=u.detach().cpu().tolist(),mask=[True]*3,AA_index=AA.index(t['sequence'][len(x)]),behavior_logprob_fixed_action=float(actionlp),action_seed=ns('action',tid,k,a),features=features(t,xx,gg,k).tolist(),samples=duration,returns=ret,soft_returns=soft,success_labels=success,mean_return=float(np.mean(ret)),return_MC_SE=float(np.std(ret,ddof=1)/math.sqrt(duration)) if duration>1 else 0,mean_soft_return=float(np.mean(soft)),success_probability_empirical=float(np.mean(success)))
      out.write(json.dumps(record,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno());done.add(key)
    save('LIVE_STATUS.json',dict(stage='EXPECTED_RETURNS',split=split,task=tid,completed_pairs=len(done),pid=os.getpid(),free_bytes=__import__('shutil').disk_usage(R).free));print('EXPECTED',split,len(done),flush=True)
 e.verify();assert state_digest(e.actor.state_dict())==fp;save('EXPECTED_COMPLETED.json',dict(complete=True,state_action_pairs=len(done),actor_fingerprint=fp,J1_tensor_SHA=e.prior.identity,network_calls=dict(e.calls),peak_GPU=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start))

class MLP(torch.nn.Module):
 def __init__(self,d):super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(d,64),torch.nn.SiLU(),torch.nn.Linear(64,32),torch.nn.SiLU(),torch.nn.Linear(32,1))
 def forward(self,x):return self.net(x).squeeze(-1)

def simple_fit():
 check();torch.set_num_threads(1);torch.manual_seed(87171);data=np.load(R/'results/realized.npz');fit=data['split']=='FIT';dev=data['split']=='DEV';X=data['X'];y=data['y'];mean=X[fit].mean(0);sd=X[fit].std(0).clip(1e-5);xx=(X-mean)/sd;d=xx.shape[1];aug=np.c_[np.ones(len(X)),xx];ridge=np.linalg.solve(aug[fit].T@aug[fit]+np.eye(d+1),aug[fit].T@y[fit]);constants=[float(y[fit&(data['stage']==k)].mean()) for k in range(4)]
 model=MLP(d);opt=torch.optim.Adam(model.parameters(),lr=.001);best=None;bestloss=float('inf');curves=[];tx=torch.tensor(xx,dtype=torch.float32);ty=torch.tensor(y,dtype=torch.float32);ii=np.flatnonzero(fit);rng=np.random.default_rng(87171)
 for u in range(1000):
  batch=rng.choice(ii,64);opt.zero_grad();loss=(model(tx[batch])-ty[batch]).square().mean();loss.backward();opt.step()
  if u+1 in [100,250,500,1000]:
   with torch.no_grad():val=float((model(tx[dev])-ty[dev]).square().mean())
   curves.append(dict(update=u+1,DEV_MSE=val))
   if val<bestloss:bestloss=val;best=copy.deepcopy(model.state_dict())
 torch.save(dict(MLP=best,mean=mean,sd=sd,ridge=ridge,constants=constants,curves=curves,MLP_parameters=sum(p.numel() for p in model.parameters())),R/'results/simple_models.pt');save('SIMPLE_FIT.json',dict(MLP_parameters=sum(p.numel() for p in model.parameters()),curves=curves,selected_DEV_MSE=bestloss,FIT_rows=int(fit.sum()),DEV_rows=int(dev.sum()),features_dim=d))

def simple_predict(rows):
 st=torch.load(R/'results/simple_models.pt',map_location='cpu',weights_only=False);X=np.array([r['features'] for r in rows]);xx=(X-st['mean'])/st['sd'];m=MLP(X.shape[1]);m.load_state_dict(st['MLP']);m.eval()
 with torch.no_grad():mlp=m(torch.tensor(xx,dtype=torch.float32)).numpy()
 return dict(constant=np.array([st['constants'][r['stage']] for r in rows]),ridge=np.c_[np.ones(len(rows)),xx]@st['ridge'],MLP=mlp)

def handle(*_):
 global STOP
 STOP=True

def expected_rows():return [json.loads(s) for s in (R/'results/expected_rows.jsonl').read_text().splitlines()]
def rank_metrics(pred,rows):
 pairs=collections.defaultdict(dict)
 for z,r in zip(pred,rows):pairs[(r['task_id'],r['stage'])][r['action_index']]=(float(z),r)
 supported=[];allpairs=[]
 for (tid,k),p in pairs.items():
  if set(p)!={0,1}:continue
  a,b=p[0][1],p[1][1];d=np.array(b['returns'])-np.array(a['returns']);delta=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(len(d))) if len(d)>1 else 0;pd=p[1][0]-p[0][0];correct=.5 if pd==0 else float(pd*delta>0)
  rec=dict(task_id=tid,group=a['group'],stage=k,reference_delta=delta,reference_SE=se,predicted_delta=pd,correct=correct)
  if delta!=0:allpairs.append(rec)
  if k<3 and abs(delta)>=.1 and abs(delta)>1.96*se:supported.append(rec)
 return dict(supported_pairs=len(supported),supported_groups=len({r['group'] for r in supported}),accuracy=float(np.mean([r['correct'] for r in supported])) if supported else None,records=supported,all_nontied_pairs=allpairs)
def score(pred,rows):
 p=np.asarray(pred);y=np.array([r['mean_return'] for r in rows]);stage=np.array([r['stage'] for r in rows]);out={}
 for k in [0,1,2,3,'nonterminal','all']:
  mask=np.ones(len(y),dtype=bool) if k=='all' else stage<3 if k=='nonterminal' else stage==k;d=p[mask]-y[mask]
  out[str(k)]=dict(n=int(mask.sum()),MSE=float(np.mean(d*d)),RMSE=float(np.sqrt(np.mean(d*d))),MAE=float(np.mean(abs(d))),bias=float(np.mean(d)),pred_min=float(p[mask].min()),pred_max=float(p[mask].max()))
 out['ranking']=rank_metrics(pred,rows);return out

def point(t,row):
 x=v['s'].xprefix(t)
 for g in row['anchor_geometry']:x=torch.cat([x,v['s'].next_ca(x,torch.tensor(g,device='cuda',dtype=torch.float64))[None]],0)
 return x

def critic_predictions(e,qs,rows,pool):
 result=[]
 with torch.no_grad():
  for row in rows:
   t=next(z for z in read(R/'manifests/SPLITS.json')['VALIDATION_SHIFTED_tasks'] if z['task_id']==row['task_id']) if row['split']=='VALIDATION_SHIFTED' else pool[row['task_id']];x=point(t,row);b=e.batch(t,x);a=torch.tensor([AA.index(t['sequence'][len(x)])],device='cuda');g=torch.tensor([row['geometry']],device='cuda',dtype=torch.float64);mask=torch.ones(1,3,device='cuda',dtype=torch.bool)
   with torch.autocast('cuda',dtype=torch.bfloat16):result.append([float(q(b,a,g,mask)) for q in qs])
 return np.asarray(result)

def critics():
 start=time.monotonic();check();torch.manual_seed(87171);np.random.seed(87171);e=v['Engine']();e.actor.requires_grad_(False);frozen=e.parent_fingerprint;old=torch.load(V7/'checkpoints/C_00000_stop.pt',map_location='cpu',weights_only=False)
 assert state_digest(old['actor'])==frozen
 qs=[LowQ(PROFILES['research']['low_q']).cuda() for _ in range(2)]
 for q,st in zip(qs,old['critic']):q.load_state_dict(st,strict=True)
 del old
 targets=[copy.deepcopy(q).eval().requires_grad_(False) for q in qs];opts=[torch.optim.AdamW(q.parameters(),lr=.0001,weight_decay=.01,foreach=False) for q in qs]
 for q,opt in zip(qs,opts):
  for p in q.parameters():opt.state[p]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
 pool=taskmap();splits=read(R/'manifests/SPLITS.json');pool.update({t['task_id']:t for t in splits['VALIDATION_tasks']});eps=episodes();bytask=collections.defaultdict(list)
 for ep in eps:bytask[ep['task_id']].append(ep)
 expected=expected_rows();dev=[r for r in expected if r['split']=='DEV'];expected_fit=collections.defaultdict(list)
 for r in expected:
  if r['split']=='FIT':expected_fit[(r['task_id'],r['stage'])].append(r)
 simple=simple_predict(dev);simp_scores={name:score(p,dev) for name,p in simple.items()};best_simple=min(simp_scores,key=lambda n:simp_scores[n]['nonterminal']['MSE']);baseMSE=simp_scores[best_simple]['nonterminal']['MSE'];save('DEV_SIMPLE_COMPARISON.json',dict(scores=simp_scores,best=best_simple))
 def step(u):
  for opt in opts:opt.zero_grad(set_to_none=True)
  losses=[];ids=splits['FIT'];used=[]
  for k in range(4):
   tid=ids[(u+k*59)%len(ids)];t=pool[tid];possible=expected_fit[(tid,k)]
   if possible and np.random.rand()<.5:
    row=possible[np.random.randint(len(possible))];x=point(t,row);geometry=row['geometry'];y=row['mean_return'];mode='expected'
   else:
    ep=bytask[tid][np.random.randint(8)];x=v['reconstruct'](t,ep,k);geometry=ep['geometry'][k];y=ep['reward']*.995**(3-k);mode='realized'
   b=e.batch(t,x);a=torch.tensor([AA.index(t['sequence'][len(x)])],device='cuda');g=torch.tensor([geometry],device='cuda',dtype=torch.float64);mask=torch.ones(1,3,device='cuda',dtype=torch.bool)
   for q in qs:
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=(q(b,a,g,mask).float()-y).square().mean()/4
    if not torch.isfinite(loss):raise FloatingPointError('critic loss')
    loss.backward();losses.append(float(loss.detach()))
   used.append(dict(task_id=tid,stage=k,target_kind=mode))
  norms=[]
  for q,opt in zip(qs,opts):norms.append(float(torch.nn.utils.clip_grad_norm_(q.parameters(),5,error_if_nonfinite=True)));opt.step()
  with torch.no_grad():
   for q,t in zip(qs,targets):
    for a,b in zip(t.parameters(),q.parameters(),strict=True):a.lerp_(b,.005)
  return dict(loss=sum(losses),gradient_norms=norms,used=used)
 def state(u,best):
  return dict(schema='V71_complete_critic_only',actor=e.actor.state_dict(),actor_optimizer=None,critics=[q.state_dict() for q in qs],targets=[q.state_dict() for q in targets],optimizers=[o.state_dict() for o in opts],rng=rng_state(),update=u,historical_updates=250,sampler_cursor=u*4,best=best,source=source(),protocol=read(R/'PROTOCOL.json'),replay_reference=dict(realized=str(R/'results/realized.npz'),sha256=sha(R/'results/realized.npz'),expected=str(R/'results/expected_rows.jsonl'),expected_sha256=sha(R/'results/expected_rows.jsonl')),J1_tensor_SHA=e.prior.identity,parent_actor_fingerprint=frozen)
 def restore(rec):
  assert sha(rec['path'])==rec['sha256'];s=torch.load(rec['path'],map_location='cpu',weights_only=False);assert state_digest(s)==rec['state_digest']
  for q,x in zip(qs,s['critics']):q.load_state_dict(x)
  for q,x in zip(targets,s['targets']):q.load_state_dict(x)
  for opt,x in zip(opts,s['optimizers']):opt.load_state_dict(x)
  set_rng(s['rng']);return s
 def publish(u,best,isbest=False,reason=None):
  st=state(u,best);size=serialized_size(st);POLICY.require(R,size+2**20);rec=atomic_checkpoint(R/f'checkpoints/critic_{u:04}{"_stop" if reason else ""}.pt',st,POLICY);rec.update(update=u,reason=reason)
  index=read(R/'checkpoints/INDEX.json') if (R/'checkpoints/INDEX.json').exists() else dict(latest=None,best=None)
  previous=[index['latest'],index['best']];index['latest']=rec
  if isbest:index['best']=rec
  save('checkpoints/INDEX.json',index)
  for p in previous:
   if p and p['path'] not in {rec['path'],index['best']['path'] if index['best'] else ''} and Path(p['path']).exists():
    with (R/'results/retirements.jsonl').open('a') as f:f.write(json.dumps(dict(path=p['path'],sha256=p['sha256'],replacement=rec['sha256'],reason='only own superseded, full replacement verified'))+'\n')
    Path(p['path']).unlink()
  return rec
 initialpath=R/'results/DEV_critics_0000.json'
 if initialpath.exists():initialscores=read(initialpath)['scores']
 else:
  ip=critic_predictions(e,qs,dev,pool);initialscores=[score(ip[:,i],dev) for i in range(2)];save('results/DEV_critics_0000.json',dict(predictions=ip.tolist(),rows=[r['key'] for r in dev],scores=initialscores))
 initial=[s['nonterminal']['MSE'] for s in initialscores]
 latest=R/'checkpoints/INDEX.json';best=float(np.mean(initial));u=0
 if latest.exists():
  assert not (R/'CRITIC_STOP.json').exists();st=restore(read(latest)['latest']);u=st['update'];best=st['best']
 else:
  rec=publish(0,best,True)
  first=step(0);sig=state_digest(state(1,best));restore(rec);second=step(0);sig2=state_digest(state(1,best));assert sig==sig2 and first==second;restore(rec)
  save('RECOVERY_TEST.json',dict(PASS=True,full_critic_optimizer_target_sampler_RNG_next_update_exact=True,scope='same process CPU deserialization',complete_checkpoint_bytes=rec['bytes'],critic_parameters_each=[sum(p.numel() for p in q.parameters()) for q in qs],actor_parameters=sum(p.numel() for p in e.actor.parameters()),actor_updates=0,engineering_critic_executions_each=2))
 curves=[dict(update=0,scores=initialscores,best_simple=best_simple,best_simple_MSE=baseMSE)];reason=None
 try:
  for update in range(u,750):
   budget();beg=time.monotonic();row=step(update);u=update+1;row['seconds']=time.monotonic()-beg
   with (R/'results/critic_updates.jsonl').open('a') as f:f.write(json.dumps(dict(update=u,**row))+'\n');f.flush();os.fsync(f.fileno())
   if u in [1,100,250,500,750]:
    pred=critic_predictions(e,qs,dev,pool);scores=[score(pred[:,i],dev) for i in range(2)]
    mean=float(np.mean([s['nonterminal']['MSE'] for s in scores]));isbest=mean<best
    if isbest:best=mean
    if u==250 and not all(s['nonterminal']['MSE']<=baseMSE and s['nonterminal']['MSE']<=.9*b for s,b in zip(scores,initial)):reason='new250update gate: critics do not outperform best simple DEV expected-return baseline with declared progress'
    if u==500 and not all(s['nonterminal']['MSE']<=.95*baseMSE and abs(s['nonterminal']['bias'])<=.1 and (s['ranking']['accuracy'] or 0)>=.55 for s in scores):reason='new500update gate: insufficient expected-return/ranking advantage'
    if any(abs(s['nonterminal']['bias'])>.25 or s['nonterminal']['pred_max']>3 for s in scores):reason='serious expected-return overestimation/bias gate'
    curves.append(dict(update=u,scores=scores,best_simple=best_simple,best_simple_MSE=baseMSE));save(f'results/DEV_critics_{u:04}.json',dict(predictions=pred.tolist(),rows=[r['key'] for r in dev],scores=scores));publish(u,best,isbest,reason);print('CRITICS',u,mean,reason,flush=True)
    if reason:break
   elif u%50==0:publish(u,best)
   save('LIVE_STATUS.json',dict(stage='CRITICS',update=u,pid=os.getpid(),free_bytes=__import__('shutil').disk_usage(R).free,peak_GPU=torch.cuda.max_memory_allocated()))
 except (RuntimeError,OSError,InterruptedError,FloatingPointError) as ex:
  st=restore(read(latest)['latest']);u=st['update'];best=st['best'];reason=repr(ex)+'; restored last verified complete transaction';publish(u,best,False,reason)
 if reason:save('CRITIC_STOP.json',dict(update=u,reason=reason))
 e.verify();assert state_digest(e.actor.state_dict())==frozen;index=read(latest);bestrec=index['best'] or index['latest'];restore(bestrec)
 predictions={}
 for split in ['FIT','DEV','CONFIRM','VALIDATION','VALIDATION_SHIFTED']:
  rr=[r for r in expected if r['split']==split];pred=critic_predictions(e,qs,rr,pool);tpred=critic_predictions(e,targets,rr,pool);sp=simple_predict(rr);predictions[split]=dict(rows=[r['key'] for r in rr],predictions=dict(simple={k:a.tolist() for k,a in sp.items()},critic1=pred[:,0].tolist(),critic2=pred[:,1].tolist(),target1=tpred[:,0].tolist(),target2=tpred[:,1].tolist()),scores={**{k:score(a,rr) for k,a in sp.items()},'critic1':score(pred[:,0],rr),'critic2':score(pred[:,1],rr)})
 save('CRITIC_FINAL.json',dict(selected=bestrec,curves=curves,partitions=predictions,actor_frozen=True,J1_tensor_SHA=e.prior.identity,actor_fingerprint=frozen,new_updates=u,old_updates=250,network_calls=dict(e.calls),peak_GPU=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start))

def rwr_monitor():
 start=time.monotonic();check();e=v['Engine']();e.actor.requires_grad_(False);splits=read(R/'manifests/SPLITS.json');pool=taskmap();fixed=set(splits['RWR_fixed_ids']);eps=[r for r in episodes() if r['task_id'] in fixed];result={(r['task_id'],r['candidate_index']):dict(task_id=r['task_id'],identity=r['identity'],group=r['group'],candidate_index=r['candidate_index'],success=r['success'],valid=r['valid'],parent_logprob=sum(r['behavior_log_prob'])) for r in eps}
 for label,path in [('B25',V7/'checkpoints/B_00025.pt'),('B250',V7/'checkpoints/B_00250_stop.pt')]:
  st=torch.load(path,map_location='cpu',weights_only=False);e.actor.load_state_dict(st['actor'],strict=True);del st;fingerprint=state_digest(e.actor.state_dict())
  for ei,ep in enumerate(eps):
   t=pool[ep['task_id']];x=v['s'].xprefix(t);lp=[];kl=[]
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for k in range(4):
     mu,ls,rm,rl,*_=e.distribution(t,x);g=torch.tensor(ep['geometry'][k],device='cuda',dtype=torch.float64);lp.append(float(v['log_prob'](mu.double(),ls.double(),g,torch.ones(3,dtype=torch.bool,device='cuda'))));kl.append(float(e.kl(mu,ls,rm,rl)));x=torch.cat([x,v['s'].next_ca(x,g)[None]],0)
   result[(ep['task_id'],ep['candidate_index'])][label+'_logprob']=sum(lp);result[(ep['task_id'],ep['candidate_index'])][label+'_KL']=float(np.mean(kl))
   if (ei+1)%64==0:print('RWR_LIKELIHOOD',label,ei+1,len(eps),flush=True)
  e.verify();assert state_digest(e.actor.state_dict())==fingerprint
 save('RWR_FIXED_TRAIN_LIKELIHOODS.json',dict(records=list(result.values()),tasks=64,candidates=len(eps),scope='point densities on identical parent-generated TRAIN trajectories; event probabilities require sampling or documented importance weighting'))
 # The stopped250 actor is evaluated only as a new read-only diagnostic, never promoted.
 for panel in ['CONFIRM','CONFIRM_SHIFTED']:
  path=R/f'results/B250_{panel}.json'
  if path.exists():continue
  rr=[]
  for ti,t in enumerate(v['tasks'](panel)):
   budget()
   for i in range(32):rr.append(dict(v['rollout'](e,t,v['ns'](t['task_id'],i,panel),'B250_READONLY'),candidate_index=i))
   if (ti+1)%8==0:print('RWR_SAMPLE',panel,ti+1,64,flush=True)
  save(f'results/B250_{panel}.json',dict(records=rr,checkpoint_SHA256=sha(V7/'checkpoints/B_00250_stop.pt'),scope='new V7.1 diagnostic on already-exposed V7validation; no retrospective V7promotion'))
 e.verify();save('RWR_MONITOR_COMPLETED.json',dict(complete=True,network_calls=dict(e.calls),actor_updates=0,J1_tensor_SHA=e.prior.identity,peak_GPU=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start))

def campaign():
 check()
 for stage,marker in [('simple_fit','SIMPLE_FIT.json'),('collect_expected','EXPECTED_COMPLETED.json'),('critics','CRITIC_FINAL.json'),('rwr_monitor','RWR_MONITOR_COMPLETED.json')]:
  if (R/marker).exists():continue
  owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
  if owners.strip():raise RuntimeError('another CUDA owner; no concurrent workload')
  with (R/f'logs/{stage}_{time.time_ns()}.log').open('x') as f:
   child=subprocess.Popen([sys.executable,'-B',str(R/'source/scripts/e017_v71/study.py'),stage],stdout=f,stderr=subprocess.STDOUT);save('EXECUTION.json',dict(stage=stage,pid=child.pid,supervisor_pid=os.getpid(),status='RUNNING',automatic_restart=False,UTC=time.time()));rc=child.wait()
  if rc:save('EXECUTION.json',dict(stage=stage,status='STOPPED',returncode=rc,automatic_restart=False,UTC=time.time()));return
 save('CAMPAIGN_COMPLETED.json',dict(complete=True,actor_updates=0,SAC_updates=0,UTC=time.time()));save('EXECUTION.json',dict(status='COMPLETED',automatic_restart=False,UTC=time.time()))

if __name__=='__main__':
 signal.signal(signal.SIGTERM,handle);signal.signal(signal.SIGINT,handle);torch.set_num_threads(1);globals()[sys.argv[1]]()
