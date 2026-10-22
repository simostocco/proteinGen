"""E017 V5: explicit relative pose, matched feedback gradients and valid IK amortization."""
import os,sys,pathlib,json,hashlib,math,time,random,copy,subprocess,shutil,collections,argparse
import numpy as np
import torch
from torch.nn import functional as F
R=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v5_goal_observability_20261022_v1')
sys.path.insert(0,str(R/'source/scripts/e017_v4'));import experiment as v
from protein_joint_sac.crisp_v3.environment import frame,assess,goal_tokens
from protein_joint_sac.crisp_v3.recovery import atomic_json,atomic_checkpoint,state_digest
from protein_joint_sac.crisp_v3.storage import rng_state,set_rng
from protein_joint_sac.geometry.ca_builder import next_ca
from protein_joint_sac.contracts import BuildState,AA
from protein_joint_sac.models.batched_bc import features_batch
v.ROOT=R
CONFIG=dict(schema='e017_v5_v1',seed=5217,horizons=[2,4,8],TRAIN_identities=1024,DEV_identities=64,CONFIRM_identities=64,prefixes=[20,28],arms=['B','C','D','E'],batch=4,stage_updates={'2':250,'4':500,'8':1250},lr=3e-5,lr_floor=3e-6,warmup=25,weight_decay=.01,clip=1.,weights={'goal':1.,'local':.1,'BC':.1,'KL':.01,'IK':.1},position_scale_A=3.8,orientation_weight=.5,teacher_per_horizon=128,IK_iterations=64,IK_starts=2,IK_orientation_weight=.5,train_native_frequency=.75,eval_seeds=[7101,7102],position_tolerance_A=8.,orientation_tolerance_deg=35.,progression=dict(min_position_gain=.10,min_success_gain=.05,validity_allowance=.05,direction_rate=.60,position_CI_upper_below=0.),scientific_stop='No horizon escalation unless own DEV success+direction+paired position/validity gate passes; matched C/D escalation requires both; no auto-extension',no_RL=True)
_frozen=pathlib.Path(__file__).with_name('frozen_science_config.json')
if _frozen.exists():CONFIG=json.load(open(_frozen))
def save(n,x):atomic_json(R/n,x)
def sha(p):return v.sha(p)
def confighash():return hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()
def seed(n):v.seed(n)
def path(split):return R/'manifests'/f'{split}.json'
def tasks(split,H=None):
 out=json.load(open(path(split)))['tasks'];return [t for t in out if H is None or t['H']==H]
def horizon(t):return t.get('H',8)
def goal(t):return dict(kind='P2',N=t['N'],targets=[dict(index=t['t']+horizon(t)-1,position=t['target'],orientation=t['orientation'],tolerance_A=8.,orientation_tolerance_deg=35.)],motifs={})
def xprefix(t,device='cuda'):return torch.tensor(t['native_prefix'],dtype=torch.float64,device=device)
def physical(z):return torch.stack([z[...,0].exp(),math.pi*z[...,1].sigmoid(),math.pi*z[...,2].tanh()],-1)
def inverse_latent(g):return torch.stack([g[...,0].log(),torch.logit(g[...,1]/math.pi),torch.atanh(g[...,2]/math.pi)],-1)
def fk(t,g):
 x=xprefix(t,g.device)
 for action in g:x=torch.cat([x,next_ca(x,action)[None]],0)
 return x

def relative(t,x):
 # frame() stores basis vectors in ROWS, i.e. Rt=frame(x).T in column convention.
 basis=frame(x);dp=basis@(x.new_tensor(t['target'])-x[-1]);rot=basis@x.new_tensor(t['orientation']).T;remaining=t['t']+horizon(t)-len(x);dist=torch.linalg.vector_norm(dp);d=torch.linalg.vector_norm(x[1:]-x[:-1],dim=-1)
 c=x.new_zeros(64);c[:3]=dp/CONFIG['position_scale_A'];c[3:12]=rot.flatten();c[12]=dist/CONFIG['position_scale_A'];c[13]=remaining/8;c[14]=dist/(CONFIG['position_scale_A']*max(remaining,1));c[15]=torch.relu(dist-remaining*v.LIMITS['adjacent_high_A'])/CONFIG['position_scale_A'];c[16]=d[-1]/CONFIG['position_scale_A'];c[17]=float(bool(d.min()>=v.LIMITS['adjacent_low_A'] and d.max()<=v.LIMITS['adjacent_high_A']));c[18]=1;c[19]=1;c[20]=remaining/float(horizon(t));return c.float(),dp,rot

def metric(t,x,noise):
 g=goal(t);g['N']=len(x);r=assess(BuildState(tuple(t['sequence'][:len(x)]),x,len(x)),g,v.LIMITS);er=r['errors'][0];internal=v.internal(x.detach().cpu().numpy())[t['t']:];r.update(task_id=t['task_id'],identity=t['identity'],group=t['protected_group_id'],H=horizon(t),variant=t.get('variant','native'),seed=noise,position_A=er['position_error_A'],orientation_deg=er['orientation_error_deg'],translation_hit=er['position_error_A']<=8,rotation_hit=er['orientation_error_deg']<=35,finite=bool(torch.isfinite(x).all()),initial_A=float(torch.linalg.vector_norm(xprefix(t,x.device)[-1]-x.new_tensor(t['target']))),adjacent_min_A=float(internal[:,0].min()),adjacent_max_A=float(internal[:,0].max()),angle_mean_rad=float(internal[:,1].mean()),torsion_mean_rad=float(internal[:,2].mean()),tail=x[t['t']:].detach().cpu().tolist());return r

def prepare():
 from protein_joint_sac.crisp_v3.execution import set_loader
 from protein_joint_sac.data.scaled_supervised import Data,canonical,internal
 from protein_joint_sac.data.accepted_supervised import CoordinateReader
 set_loader('arrow_scalar_lru4');data=Data(False);reader=CoordinateReader(data.proposal.external);old=json.load(open('/mnt/d/Simone/proteinGen_data/e017_v4_multistep_20261022_v1/manifests/VALIDATION.json'))['tasks'];excluded_groups={x['protected_group_id'] for x in old};outputs={};reject=[];census={};usedgroups=set();start=time.monotonic()
 for sp,source,count in [('TRAIN','TRAIN',1024),('DEV','VALIDATION',64),('CONFIRM','VALIDATION',64)]:
  ids=[h for h in data.jointids[source] if len(data.seqs[h])>=22];ids.sort(key=lambda h:hashlib.sha256((str(CONFIG['seed'])+source+h).encode()).hexdigest());counts={str(H):sum(len(data.seqs[h])>=t+H for h in ids for t in [20,28]) for H in [2,4,8]};census[sp]=dict(sequence_eligible_joint_identities=len(ids),metadata_potential_native_windows_by_H=counts,not_coordinate_verified_until_selected=True)
  rows=[];chosen=set();observations=set();scanned=0
  for h in ids:
   if len(chosen)>=count:break
   group=data.proposal.sequences[h]['protected_group_id']
   if group in usedgroups or (source=='VALIDATION' and group in excluded_groups):continue
   scanned+=1
   if scanned%32==0:save('results/PREPARATION_PROGRESS.json',dict(panel=sp,scanned=scanned,admitted_identities=len(chosen),tasks=len(rows),exclusions=len(reject),seconds=time.monotonic()-start));print('PREP',sp,scanned,len(chosen),flush=True)
   o=min(data.owners[h]);r=data.rows[o];before=len(rows)
   try:
    assert r['split']==source and r['sequence_hash']==h
    xx=canonical(reader.read(r));gg=internal(xx);seq=data.seqs[h];assert len(xx)==len(seq)
    for t in [20,28]:
     for H in [2,4,8]:
      if len(seq)<t+H:continue
      native=xx[t:t+H];actions=gg[t:t+H];base=dict(identity=h,protected_group_id=group,split=source,panel=sp,observation=o,source_sha=r['source_sha'],entry_id=r['entry_id'],coordinate_path=r['coordinate_path'],coordinate_row=r['coordinate_row'],N=len(seq),t=t,H=H,sequence=seq,native_prefix=xx[:t].tolist(),native_actions=actions.tolist(),target=xx[t+H-1].tolist(),orientation=frame(torch.tensor(xx[:t+H])).tolist(),variant='native',hidden_native_future=native.tolist(),action_masks=[[True,True,True]]*H)
      base['task_id']=f'{h}_{t}_{H}_native';w=torch.tensor(xx[:t+H]);m=metric(base,w,0)
      if not m['valid']:reject.append(dict(identity=h,t=t,H=H,reason='native prefix+H fails unchanged V3 validity'));continue
      rebuilt=fk(base,torch.tensor(actions));err=float((rebuilt-w).abs().max());assert err<1e-6;base['witness_max_error_A']=err;base['witness_valid']=True;rows.append(base)
      # Reproducible kinematic alternate; only valid alternatives admitted, never experimental labels.
      alt=copy.deepcopy(base);a=torch.tensor(actions);a[0,2]=((a[0,2]+.8+math.pi)%(2*math.pi))-math.pi;ax=fk(base,a);alt.update(target=ax[-1].tolist(),orientation=frame(ax).tolist(),variant='kinematic_alt',task_id=f'{h}_{t}_{H}_alt',witness_actions=a.tolist(),experimental_target=False)
      am=metric(alt,ax,0)
      if am['valid']:alt['witness_max_error_A']=0.;rows.append(alt)
      else:reject.append(dict(identity=h,t=t,H=H,reason='synthetic alternate witness invalid',synthetic=True))
   except (ValueError,AssertionError,FloatingPointError) as er:reject.append(dict(identity=h,observation=o,reason=str(er)))
   if len(rows)>before:chosen.add(h);usedgroups.add(group);observations.add(o)
  if len(chosen)!=count:raise RuntimeError(f'insufficient eligible {sp} identities {len(chosen)}')
  outputs[sp]=rows;census[sp].update(selected_identities=len(chosen),selected_observations=len(observations),scanned_identities=scanned,tasks=len(rows),tasks_by_H=dict(collections.Counter(str(x['H']) for x in rows)),native_vs_synthetic=dict(collections.Counter(x['variant'] for x in rows)),length_bands=dict(collections.Counter('<=64' if len(data.seqs[h])<=64 else '65-128' if len(data.seqs[h])<=128 else '129-256' if len(data.seqs[h])<=256 else '>256' for h in chosen)))
  save(f'manifests/{sp}.json',dict(selection='hash-ranked identity-first qualified complete; one observation per identity; prefixes20/28; original split; protected groups disjoint across panels',tasks=rows));print('DATA',sp,census[sp],flush=True)
 assert not {t['protected_group_id'] for t in outputs['TRAIN']}&{t['protected_group_id'] for t in outputs['DEV']+outputs['CONFIRM']}
 save('manifests/EXCLUSIONS.json',reject);save('TASK_CENSUS.json',dict(population=dict(TRAIN_sequence=118219,TRAIN_joint=21475,TRAIN_joint_observations=sum(r['split']=='TRAIN' for r in data.rows.values())),panels=census,historical_V4_validation_excluded_from_new_panels=True,exclusions=len(reject),seconds=time.monotonic()-start,proposal_SHA=sha('/mnt/d/Simone/proteinGen_data/pdb_supervised_split_50id80cov_20261016_v1/PROPOSAL_MANIFEST.json')))

class Engine(v.Engine):
 def __init__(self):super().__init__();self.arm='C';self.buffer_initial={n:b.detach().clone() for n,b in self.prior.model.named_buffers()}
 def verify(self,optimizers=()):
  self.prior.verify(optimizers)
  assert all(torch.equal(b,self.buffer_initial[n]) for n,b in self.prior.model.named_buffers()),'J1 nonpersistent buffer drift'
 def predict(self,t,x,goal_on=True):
  p=self.p(t,x);ref=self.prior_forward(p);gt,gv=goal_tokens(goal(t),'cuda');c=x.new_zeros(1,64).float()
  if self.arm in ['C','D','E']:
   ctx,dp,rot=relative(t,x);c=ctx[None];gt=gt.clone();gt[0,0,:3]=dp.float()/CONFIG['position_scale_A'];gt[0,0,3:12]=rot.flatten().float()
  if not goal_on:gt=gt*0;gv=gv&False;c=c*0
  b=dict(prefix=p,reference=ref,features=features_batch(p,chemistry=True),goals=gt,goal_valid=gv,context=c,subgoal=x.new_zeros(1).float());l,mu,ls=self.actor(b);a=AA.index(t['sequence'][len(x)]);return mu[0,a],ls[0,a],ref['mu'][0,a],ref['ls'][0,a],ref,b
 def guide(self,x,t):
  scaled=(x[-1]-x.new_tensor(t['target']))/CONFIG['position_scale_A'];position=F.huber_loss(scaled,torch.zeros_like(scaled),delta=1.,reduction='sum');rotation=(3-(frame(x)*x.new_tensor(t['orientation'])).sum())/4;return position+CONFIG['orientation_weight']*rotation.clamp(0,1),dict(position=position,orientation=rotation)
 def local(self,x,t):
  start=t['t'];d=torch.linalg.vector_norm(x[start:]-x[start-1:-1],dim=-1);spacing=(F.relu(v.LIMITS['adjacent_low_A']-d).square()+F.relu(d-v.LIMITS['adjacent_high_A']).square()).mean()/.25**2;dist=torch.linalg.vector_norm(x[start:,None]-x[None,:],dim=-1);i=torch.arange(start,len(x),device=x.device)[:,None];j=torch.arange(len(x),device=x.device)[None];return spacing+(F.relu(2-dist).square()*(i-j>=3)).sum()/horizon(t)
 def rollout(self,t,noise_seed,retain=False,route=None,delta=None):
  H=horizon(t);epsilon=torch.randn(H,3,dtype=torch.float64,generator=torch.Generator().manual_seed(noise_seed)).cuda();x=xprefix(t);kl=[];mus=[];lss=[];uu=[]
  for k in range(H):
   inp=x.detach() if (self.arm=='D' if route is None else route=='feedback_detached') else x
   mu,ls,rm,rl,*_=self.predict(t,inp);mus.append(mu);lss.append(ls)
   if retain:mu.retain_grad();ls.retain_grad()
   u=mu.double()+ls.double().exp()*epsilon[k]
   if delta is not None:u=u+delta[k]
   g=physical(u)
   if not torch.isfinite(g).all() or not bool((g[0]>0)&(g[1]>0)&(g[1]<math.pi)&(g[2].abs()<math.pi)):raise FloatingPointError('support violation')
   x=torch.cat([x,next_ca(x,g)[None]],0);kl.append(self.kl(mu,ls,rm,rl));uu.append(u)
  guide,_=self.guide(x,t);bi=noise_seed%H;native=torch.tensor(t['native_prefix']+t['hidden_native_future'][:bi],dtype=torch.float64,device='cuda');bm,bl,*_=self.predict(t,native);ng=torch.tensor(t['native_actions'][bi],dtype=torch.float64,device='cuda');bc=-v.log_prob(bm.double(),bl.double(),ng,torch.ones(3,device='cuda',dtype=torch.bool))/3
  terms=dict(goal=guide,local=self.local(x,t),BC=bc,KL=torch.stack(kl).mean());loss=sum(CONFIG['weights'][k]*val for k,val in terms.items());return loss,terms,x,dict(mus=mus,lss=lss,u=uu)
 def teacher_loss(self,t,record):
  x=xprefix(t);loss=[]
  for g in record['actions']:
   mu,ls,*_=self.predict(t,x);action=torch.tensor(g,dtype=torch.float64,device='cuda');loss.append(-v.log_prob(mu.double(),ls.double(),action,torch.ones(3,device='cuda',dtype=torch.bool))/3);x=torch.cat([x,next_ca(x,action)[None]],0)
  return torch.stack(loss).mean()

def IK(t,iterations=64,starts=2):
 # No native actions or future coordinates read: target + visible prefix only.
 begin=time.monotonic();best=None;attempts=[]
 for s in range(starts):
  z=torch.zeros(horizon(t),3,dtype=torch.float64,requires_grad=True);z.data[:,1]=.4;z.data[:,2]=(.5 if s==0 else -.5);opt=torch.optim.Adam([z],lr=.08)
  for k in range(iterations):
   opt.zero_grad();g=torch.stack([v.LIMITS['adjacent_low_A']+.01+(v.LIMITS['adjacent_high_A']-v.LIMITS['adjacent_low_A']-.02)*z[:,0].sigmoid(),.03+(math.pi-.06)*z[:,1].sigmoid(),math.pi*z[:,2].tanh()],-1);x=fk(t,g);pos=F.huber_loss((x[-1]-x.new_tensor(t['target']))/3.8,torch.zeros(3,dtype=torch.float64),reduction='sum');rot=(3-(frame(x)*x.new_tensor(t['orientation'])).sum())/4;clash=Engine.local(None,x,t);loss=pos+CONFIG['IK_orientation_weight']*rot+2.*clash+((g[:,0]-3.8)/.1).square().mean()+.001*z.square().mean();loss.backward();torch.nn.utils.clip_grad_norm_([z],10);opt.step()
  with torch.no_grad():
   g=torch.stack([v.LIMITS['adjacent_low_A']+.01+(v.LIMITS['adjacent_high_A']-v.LIMITS['adjacent_low_A']-.02)*z[:,0].sigmoid(),.03+(math.pi-.06)*z[:,1].sigmoid(),math.pi*z[:,2].tanh()],-1);x=fk(t,g);m=metric(t,x,s);m.update(actions=g.tolist(),seconds=time.monotonic()-begin,start=s,teacher_accepted=m['success'],failure_reason=None if m['success'] else 'goal_tolerance' if m['valid'] else 'backbone_validity');m['exclusion_codes']=[]
   if not m['translation_hit']:m['exclusion_codes'].append('position_tolerance')
   if not m['rotation_hit']:m['exclusion_codes'].append('orientation_tolerance')
   dd=torch.linalg.vector_norm(x[1:]-x[:-1],dim=-1)
   if dd.min()<v.LIMITS['adjacent_low_A'] or dd.max()>v.LIMITS['adjacent_high_A']:m['exclusion_codes'].append('adjacent_spacing')
   if m['clashes_per100']>v.LIMITS['clashes_per100_max']:m['exclusion_codes'].append('nonlocal_clashes')
   if not v.LIMITS['Rg_normalized_low']<=m['Rg_normalized']<=v.LIMITS['Rg_normalized_high']:m['exclusion_codes'].append('fragment_Rg')
   if not m['finite']:m['exclusion_codes'].append('nonfinite')
   attempts.append(m)
   if m['success'] and (best is None or m['position_A']<best['position_A']):best=m
  # Retain both bounded starts, including multiple valid solutions for stochastic teacher sampling.
 return dict(task_id=t['task_id'],accepted=best is not None,best=best,attempts=attempts,seconds=time.monotonic()-begin)

def teachers():
 pool=tasks('TRAIN');out=[];start=time.monotonic()
 for H in [2,4,8]:
  seen=set();candidates=[]
  for t in sorted([t for t in pool if t['H']==H],key=lambda t:hashlib.sha256(t['task_id'].encode()).hexdigest()):
   if t['identity'] not in seen:seen.add(t['identity']);candidates.append(t)
   if len(candidates)>=CONFIG['teacher_per_horizon']:break
  for i,t in enumerate(candidates):
   out.append(IK(t));
   if (i+1)%16==0:save('results/IK_TEACHER_PROGRESS.json',dict(H=H,done=i+1,seconds=time.monotonic()-start));print('IK',H,i+1,flush=True)
 save('manifests/IK_TEACHERS.json',dict(records=out,accepted=sum(x['accepted'] for x in out),seconds=time.monotonic()-start,scientific_label='synthetic target-conditioned kinematic solutions, not experimental protein structures'))

def panel(split,H,nident=24):
 out=[];ids=sorted({t['identity'] for t in tasks(split,H)})[:nident]
 for h in ids:
  rows=[t for t in tasks(split,H) if t['identity']==h];tmin=min(t['t'] for t in rows);rows=[t for t in rows if t['t']==tmin];out.extend(rows)
 return out

def summary(rr):
 return dict(attempts=len(rr),identities=len({r['identity'] for r in rr}),position_A=float(np.mean([r['position_A'] for r in rr])),median_position_A=float(np.median([r['position_A'] for r in rr])),orientation_deg=float(np.mean([r['orientation_deg'] for r in rr])),success=sum(r['success'] for r in rr),valid=sum(r['valid'] for r in rr),translation_hit=sum(r['translation_hit'] for r in rr),rotation_hit=sum(r['rotation_hit'] for r in rr),finite=sum(r['finite'] for r in rr),KL=float(np.mean([r.get('KL',0.) for r in rr])))
def paired(a,b,key):
 aa={(r['task_id'],r['seed']):r for r in a};d=collections.defaultdict(list)
 for r in b:d[r['group']].append(float(r[key])-float(aa[(r['task_id'],r['seed'])][key]))
 means=np.array([np.mean(x) for x in d.values()]);rng=np.random.default_rng(117);samples=means[rng.integers(0,len(means),(4000,len(means)))].mean(-1);return dict(delta=float(means.mean()),CI95=np.quantile(samples,[.025,.975]).tolist(),groups=len(means))
def direction(rr):
 groups=collections.defaultdict(dict)
 for r in rr:groups[(r['identity'],r['seed'])][r['variant']]=r
 out=[]
 for (h,s),x in groups.items():
  if not {'native','kinematic_alt'}<=x.keys():continue
  a,b=x['native'],x['kinematic_alt'];ta=np.array(a['target']);tb=np.array(b['target']);pa=np.array(a['tail'][-1]);pb=np.array(b['tail'][-1]);dot=float((pb-pa).dot(tb-ta));own=(np.linalg.norm(pa-ta)+np.linalg.norm(pb-tb))/2;cross=(np.linalg.norm(pa-tb)+np.linalg.norm(pb-ta))/2;out.append(dict(identity=h,seed=s,correct=dot>0,own_minus_cross_A=float(own-cross)))
 return dict(pairs=len(out),correct=sum(x['correct'] for x in out),rate=sum(x['correct'] for x in out)/max(1,len(out)),mean_own_minus_cross_A=float(np.mean([x['own_minus_cross_A'] for x in out])) if out else None,records=out)

def evaluate(e,arm,H,update,split='DEV',condition='native',nident=24):
 name=f'{arm}_H{H}_{update:05d}_{split}_{condition}';p=R/'results'/f'{name}.json'
 if p.exists():return json.load(open(p))
 start=time.monotonic();e.actor.eval();e.arm=arm;rr=[];pool=panel(split,H,nident)
 if condition.startswith('shifted'):pool=json.load(open(R/f'manifests/SHIFTED_{split}_H{H}.json'))['tasks']
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for t in pool:
   for ns in CONFIG['eval_seeds']:
    loss,terms,x,_=e.rollout(t,ns);r=metric(t,x,ns);r.update(target=t['target'],KL=float(terms['KL']),loss=float(loss),terms={k:float(x) for k,x in terms.items()});rr.append(r)
 e.verify();out=dict(arm=arm,H=H,update=update,panel=split,condition=condition,seconds=time.monotonic()-start,summary=summary(rr),direction=direction(rr),records=rr);save(f'results/{name}.json',out);print('EVAL',name,out['summary'],out['direction']['rate'],flush=True);return out

def baseline():
 e=Engine();e.arm='A'
 for H in [2,4,8]:
  evaluate(e,'A',H,0);evaluate(e,'C',H,0);evaluate(e,'A',H,0,'DEV','shifted')
 e.verify()

def gate(current,base):
 n=current['summary']['attempts'];a=base['summary'];b=current['summary'];pp=paired(base['records'],current['records'],'position_A');gain=(a['position_A']-b['position_A'])/a['position_A'];sr=b['success']/n-a['success']/a['attempts'];valid=b['valid']/n>=a['valid']/a['attempts']-.05;g=CONFIG['progression'];passed=gain>=g['min_position_gain'] and pp['CI95'][1]<0 and sr>=g['min_success_gain'] and valid and current['direction']['rate']>=g['direction_rate'];return dict(pass_gate=passed,position_gain=gain,position_delta=pp,success_gain=sr,validity_preserved=valid,direction_rate=current['direction']['rate'])

def scheduler(opt):return torch.optim.lr_scheduler.LambdaLR(opt,lambda k:min(1.,(k+1)/25)*(.1+.9*(1+math.cos(math.pi*min(k,2000)/2000))/2))
def optimizer(e):return torch.optim.AdamW([p for p in e.actor.parameters() if p.requires_grad],lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'],foreach=False)
def saveck(e,opt,sched,arm,update,stage,cursor,order,best,monitor):
 e.verify([opt]);state=dict(schema='e017_v5_recovery_v1',actor=e.actor.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),arm=arm,update=update,stage=stage,cursor=cursor,order=order,best=best,monitor=monitor,rng=rng_state(),config=CONFIG,config_hash=confighash(),source=json.load(open(R/'SEALED_CONFIG.json'))['source_commit'],task_hashes={s:sha(path(s)) for s in ['TRAIN','DEV','CONFIRM']},IK_SHA=sha(R/'manifests/IK_TEACHERS.json'),parent=str(v.PARENT),parent_SHA=v.SHA,J1_tensor_SHA=e.prior.identity)
 dr=R/'checkpoints'/arm;dr.mkdir(exist_ok=True);old=json.load(open(dr/'INDEX.json')) if (dr/'INDEX.json').exists() else dict(latest=None,best=None);p=dr/f'checkpoint_{update:05d}.pt';rec=atomic_checkpoint(p,state,v.POLICY);rec.update(update=update,stage=stage);newbest=rec if best.get('update')==update else old['best'];save(f'checkpoints/{arm}/INDEX.json',dict(latest=rec,best=newbest))
 for q in dr.glob('checkpoint_*.pt'):
  if str(q) not in {rec['path'],newbest['path'] if newbest else ''}:
   with (dr/'RETIREMENTS.jsonl').open('a') as f:f.write(json.dumps(dict(path=str(q),SHA=sha(q),reason='own V5 superseded verified rolling checkpoint'))+'\n')
   q.unlink()

def identity_order(pool,seed_value):
 ri=random.Random(seed_value);by=collections.defaultdict(list)
 for i,t in enumerate(pool):by[t['identity']].append(i)
 ids=sorted(by);ri.shuffle(ids);order=[]
 for j,h in enumerate(ids):
  preferred='native' if j%4!=3 else 'kinematic_alt';ii=[i for i in by[h] if pool[i]['variant']==preferred];order.append(ri.choice(ii or by[h]))
 return order

def train(arm,H):
 seal=json.load(open(R/'SEALED_CONFIG.json'));assert seal['config_hash']==confighash() and seal['source_commit']==subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip();seed(CONFIG['seed']);e=Engine();e.arm=arm;opt=optimizer(e);sched=scheduler(opt);dr=R/'checkpoints'/arm;ix=dr/'INDEX.json';update=0;cursor=0;best=dict(score=1e30,update=None);monitor={};pool=tasks('TRAIN',H);order=identity_order(pool,CONFIG['seed']+H)
 if ix.exists():
  q=json.load(open(ix))['latest'];st=torch.load(q['path'],map_location='cpu',weights_only=False);assert sha(q['path'])==q['sha256'] and state_digest(st)==q['state_digest'] and st['config_hash']==confighash();e.actor.load_state_dict(st['actor']);opt.load_state_dict(st['optimizer']);sched.load_state_dict(st['scheduler']);set_rng(st['rng']);update=st['update'];best=st['best'] if st['stage']==H else dict(score=1e30,update=None);monitor=st['monitor'];cursor=st['cursor'] if st['stage']==H else 0;order=st['order'] if st['stage']==H else order
 startupdate={2:0,4:250,8:750}[H];target=startupdate+CONFIG['stage_updates'][str(H)];assert update>=startupdate and update<=target
 if ix.exists() and st['stage']!=H:
  previous=st['stage'];snapshot=dict(actor=st['actor'],arm=arm,H=previous,update=st['update'],schema='evaluation_only_not_recoverable',parent_SHA=v.SHA);record=atomic_checkpoint(dr/f'EVAL_H{previous}.pt',snapshot,v.POLICY);save(f'checkpoints/{arm}/EVAL_H{previous}.json',record)
 teachers_db={r['task_id']:[x for x in r['attempts'] if x['teacher_accepted']] for r in json.load(open(R/'manifests/IK_TEACHERS.json'))['records'] if r['accepted']};teacher_tasks=[t for t in pool if t['task_id'] in teachers_db];assert arm!='E' or teacher_tasks
 log=open(R/f'results/{arm}_updates.jsonl','a',buffering=1);baseline=json.load(open(R/f'results/A_H{H}_00000_DEV_native.json'));stagebegin=time.monotonic()
 for u in range(update,target):
  v.POLICY.require(R,4096);e.actor.train();opt.zero_grad(set_to_none=True);t0=time.monotonic();parts=collections.defaultdict(float);seen=[];variants=[];gradgroups={}
  for j in range(CONFIG['batch']):
   if cursor>=len(order):order=identity_order(pool,CONFIG['seed']+u+H);cursor=0
   # Fixed native/alternate frequency, identity-first order compiled once at data seal below.
   task=pool[order[cursor]];cursor+=1;ns=CONFIG['seed']+100000+4*u+j;seen.append(task['identity']);variants.append(task['variant'])
   with torch.autocast('cuda',dtype=torch.bfloat16):
    loss,terms,x,_=e.rollout(task,ns)
    if arm=='E':
     tt=teacher_tasks[(4*u+j)%len(teacher_tasks)];solutions=teachers_db[tt['task_id']];ikloss=e.teacher_loss(tt,solutions[(u+j)%len(solutions)]);loss=loss+CONFIG['weights']['IK']*ikloss;terms['IK']=ikloss
   (loss/CONFIG['batch']).backward()
   for k,z in terms.items():parts[k]+=float(z.detach())/CONFIG['batch']
  assert all(p.grad is None for p in e.prior.model.parameters())
  for name,mod in [('encoder',e.actor.encoder),('geometry',e.actor.geometry),('residue',e.actor.residue)]:gradgroups[name]=math.sqrt(sum(float(p.grad.float().square().sum()) for p in mod.parameters() if p.grad is not None))
  norm=torch.nn.utils.clip_grad_norm_([p for p in e.actor.parameters() if p.requires_grad],CONFIG['clip'],error_if_nonfinite=True);opt.step();sched.step();torch.cuda.synchronize();log.write(json.dumps(dict(update=u+1,H=H,terms=dict(parts),gradient_norm=float(norm),groups=gradgroups,identities=seen,variants=variants,seconds=time.monotonic()-t0))+'\n')
  if (u+1)%25==0:save(f'results/{arm}_STATUS.json',dict(update=u+1,H=H,status='TRAINING',seconds_last=time.monotonic()-t0));print('TRAIN',arm,H,u+1,flush=True)
  if u+1 in [startupdate+100,target]:
   ev=evaluate(e,arm,H,u+1);n=ev['summary']['attempts'];good=ev['summary']['valid']/n>=baseline['summary']['valid']/baseline['summary']['attempts']-.05;score=ev['summary']['position_A']/3.8+ev['summary']['orientation_deg']/180
   if good and score<best['score']:best=dict(score=score,update=u+1,H=H)
   if ev['summary']['finite']!=n or ev['summary']['KL']>1:raise RuntimeError('sealed finite/KL scientific stop')
   monitor[str(H)]=gate(ev,baseline)
   if u+1==target:
    shifted=evaluate(e,arm,H,u+1,'DEV','shifted');shiftbase=json.load(open(R/f'results/A_H{H}_00000_DEV_shifted.json'));sv=shifted['summary'];sb=shiftbase['summary'];shiftok=sv['position_A']<=sb['position_A']*.95 and sv['valid']/sv['attempts']>=sb['valid']/sb['attempts']-.05;monitor[str(H)]['generated_prefix_gate']=shiftok;monitor[str(H)]['pass_gate']=monitor[str(H)]['pass_gate'] and shiftok
   saveck(e,opt,sched,arm,u+1,H,cursor,order,best,monitor)
  elif (u+1)%100==0:saveck(e,opt,sched,arm,u+1,H,cursor,order,best,monitor)
 log.close();e.verify([opt]);
 # Reuse existing full checkpoint until later training would retire it; no redundant actor copy for stopped arms.
 save(f'checkpoints/{arm}/EVAL_H{H}.json',json.load(open(dr/'INDEX.json'))['latest'])
 save(f'results/{arm}_H{H}_FINISHED.json',dict(update=target,stage_updates=target-startupdate,seconds=time.monotonic()-stagebegin,gate=monitor[str(H)],J1_frozen=True,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--arm',default='C');p.add_argument('--H',type=int,default=2);a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='teachers':teachers()
 elif a.command=='baseline':baseline()
 elif a.command=='train':train(a.arm,a.H)
 else:raise ValueError(a.command)
