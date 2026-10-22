"""Bounded E017 V5.1. Fixed-AA geometric supervision only; no RL optimization."""
import os,sys,pathlib,json,math,time,copy,random,collections,hashlib,signal,subprocess,uuid,shutil,argparse
import numpy as np
import torch
from torch.nn import functional as F
R=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v51_safe_control_20261022_v1');OLD=R.parent/'e017_v5_goal_observability_20261022_v1'
sys.path.insert(0,str(R/'source/scripts/e017_v5'));import study as s
from protein_joint_sac.crisp_v3.recovery import atomic_json,atomic_checkpoint,state_digest,sha,StoragePolicy,StoragePause
from protein_joint_sac.crisp_v3.storage import rng_state,set_rng
from protein_joint_sac.crisp_v3.environment import frame
PARENT=OLD/'checkpoints/D/checkpoint_00250.pt';PARENT_SHA='0b2f0b68783e4234f44c3b1041a5c1f59f04191af79b13232553b511168de265'
POLICY=StoragePolicy(min_headroom_bytes=2*2**30,warning_free_bytes=4*2**30,near_term_growth_bytes=32*2**20,checkpoint_hint_bytes=220*2**20)
DEFAULT=dict(schema='E017_V51_v1',seed=6517,updates=500,batch=4,milestones=[0,100,250,500],periodic_checkpoint=50,lr=1e-5,lr_floor=1e-6,warmup=25,weight_decay=.01,clip=5.,arms=['A','B','C','D'],feedback='D+250 feedback-detached neural inputs; direct FK gradients retained in all arms',native_preference=.75,eval_seeds=[7101,7102],weights=dict(goal=1.,local=.1,BC=.1,KL=.01),collision_margin_A=3.,collision_weight=None,KL_limit=1.,dual_target=.8,dual_eta=.05,dual_initial=.01,hard_validity_allowance=.05,H2_success_allowance=.05,position_tolerance_A=8.,orientation_tolerance_deg=35.,selection='eligible max DEV H4 valid success; tie minimum position/3.8+orientation/180; then earliest. H2 success>=initial-5pp, native H4 validity>=initial-5pp and mean KL<=1; final audit never selects.',primary_comparison='matched+100 H4 updates; stopped models retained but scientifically invalid not promoted',primary_endpoint='paired protected-identity valid H4 success increase: CI95 lower>0 and point>=5pp vs A; alternatively joint valid-and-position/orientation-better endpoint defined in PROTOCOL, no posthoc replacement',no_RL=True)
CONFIG=json.load(open(R/'CONFIG.json')) if (R/'CONFIG.json').exists() else DEFAULT
STOP_REQUEST=None

def save(n,x):atomic_json(R/n,x)
def read(n):return json.load(open(R/n))
def seed(n):s.seed(n)
def configsha():return hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()
def source():return subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip()
def trainpool():return [t for t in json.load(open(OLD/'manifests/TRAIN.json'))['tasks'] if t['H']==4]
def collision(x,t,margin=None):
 """Every generated-to-earlier nonlocal pair once; no self/adjacent pairs. Smooth pre-contact gradient."""
 m=CONFIG['collision_margin_A'] if margin is None else margin;start=t['t'];i=torch.arange(start,len(x),device=x.device)[:,None];j=torch.arange(len(x),device=x.device)[None];mask=i-j>=3;d=torch.sqrt((x[start:,None]-x[None,:]).square().sum(-1)+1e-12)
 return (F.relu(m-d).square()*mask).sum()/t['H']
def collision_details(t,tail):
 x=np.array(t['native_prefix']+tail);start=t['t'];events=[];nearest=[]
 for i in range(start,len(x)):
  dd=np.linalg.norm(x[i]-x[:i-2],axis=1);j=int(dd.argmin());nearest.append(dict(step=i-start+1,distance_A=float(dd[j]),earlier_index=j,pair_type='prefix' if j<start else 'generated'))
  for j in np.flatnonzero(dd<2.):events.append(dict(step=i-start+1,earlier_index=int(j),distance_A=float(dd[j]),pair_type='prefix' if j<start else 'generated'))
 return dict(first_collision_step=min((e['step'] for e in events),default=None),collision_pairs=events,nearest_by_step=nearest,has_collision=bool(events))

class Engine(s.Engine):
 def __init__(self):
  super().__init__();assert sha(PARENT)==PARENT_SHA;st=torch.load(PARENT,map_location='cpu',weights_only=False);self.actor.load_state_dict(st['actor']);del st;self.arm='D';self.label='A';self.dual=CONFIG['dual_initial'];self.actor_buffers={n:b.clone() for n,b in self.actor.named_buffers()}
 def kl(self,mu,ls,rm,rl):
  value=super().kl(mu,ls,rm,rl)
  if not torch.is_grad_enabled() and hasattr(self,'step_KLs'):self.step_KLs.append(float(value))
  return value
 def loss(self,t,ns,retain=False):
  self.step_KLs=[]
  base,terms,x,extra=self.rollout(t,ns,retain=retain,route='feedback_detached');cc=collision(x,t);terms['collision']=cc
  loss=base
  if self.label in ['B','D']:loss=loss+(self.dual-CONFIG['weights']['KL'])*terms['KL']
  if self.label in ['C','D']:loss=loss+CONFIG['collision_weight']*cc
  return loss,terms,x,extra
 def verify(self,optimizers=()):
  super().verify(optimizers);assert all(torch.equal(b,self.actor_buffers[n]) for n,b in self.actor.named_buffers())

def optimizer(e):return torch.optim.AdamW([p for p in e.actor.parameters() if p.requires_grad],lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'],foreach=False)
def scheduler(o):return torch.optim.lr_scheduler.LambdaLR(o,lambda u:min(1.,(u+1)/CONFIG['warmup'])*(.1+.9*(1+math.cos(math.pi*min(u,500)/500))/2))
def update(e,o,sc,pool,order,cursor,u):
 e.transaction_phase='FORWARD';e.actor.train();o.zero_grad(set_to_none=True);parts=collections.defaultdict(float);ids=[];ts=[];tt=time.monotonic()
 for j in range(CONFIG['batch']):
  if cursor>=len(order):order=s.identity_order(pool,CONFIG['seed']+u+4);cursor=0
  t=pool[order[cursor]];cursor+=1;ids.append(t['identity']);ts.append(t['task_id']);ns=CONFIG['seed']+100000+4*u+j
  with torch.autocast('cuda',dtype=torch.bfloat16):loss,terms,_,_=e.loss(t,ns)
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss before optimizer')
  (loss/CONFIG['batch']).backward()
  for k,z in terms.items():parts[k]+=float(z.detach())/CONFIG['batch']
 groups={n:math.sqrt(sum(float(p.grad.float().square().sum()) for p in m.parameters() if p.grad is not None)) for n,m in [('encoder',e.actor.encoder),('geometry',e.actor.geometry),('residue',e.actor.residue)]}
 assert all(p.grad is None for p in e.prior.model.parameters());norm=torch.nn.utils.clip_grad_norm_([p for p in e.actor.parameters() if p.requires_grad],CONFIG['clip'],error_if_nonfinite=True)
 # No stop/validation signal is raised between optimizer and scheduler transaction boundary.
 e.transaction_phase='OPTIMIZER';o.step();sc.step()
 if e.label in ['B','D']:e.dual=max(0.,e.dual+CONFIG['dual_eta']*(parts['KL']-CONFIG['dual_target']))
 torch.cuda.synchronize();return order,cursor,dict(update=u+1,terms=dict(parts),gradient_norm=float(norm),groups=groups,dual=e.dual,identities=ids,task_ids=ts,seconds=time.monotonic()-tt)

def ckstate(e,o,sc,label,u,order,cursor,best,monitor,reason=None):
 e.verify([o]);return dict(schema='e017_v51_complete_recovery_v1',actor=e.actor.state_dict(),optimizer=o.state_dict(),scheduler=sc.state_dict(),sampler=dict(order=order,cursor=cursor),rng=rng_state(),dual=e.dual,arm=label,update=u,best=best,monitor=monitor,stop_reason=reason,config=CONFIG,config_hash=configsha(),source=source(),input_hashes=read('INPUT_MANIFEST.json'),parent=str(PARENT),parent_SHA=PARENT_SHA,J1_tensor_SHA=e.prior.identity,immutable_nonactor_system_parent=dict(path=str(s.v.PARENT),sha256=s.v.SHA),replay=dict(active=False,episodes=0),stage=4)
def publish(e,o,sc,label,u,order,cursor,best,monitor,reason=None):
 state=ckstate(e,o,sc,label,u,order,cursor,best,monitor,reason);dr=R/'checkpoints'/label;dr.mkdir(exist_ok=True);p=dr/f'checkpoint_{u:05d}_{uuid.uuid4().hex[:8]}.pt';rec=atomic_checkpoint(p,state,POLICY);rec.update(update=u,stop_reason=reason,scientifically_eligible=bool(monitor.get('eligible',False)) and reason is None)
 # Readback is mandatory even at scientific stop. It checks all state, including dual multiplier.
 loaded=torch.load(p,map_location='cpu',weights_only=False);assert loaded['update']==u and state_digest(loaded)==rec['state_digest'];del loaded
 old=read(f'checkpoints/{label}/INDEX.json') if (dr/'INDEX.json').exists() else dict(latest=None,best=None);newbest=rec if best.get('update')==u and reason is None else old['best'];
 if newbest:assert sha(newbest['path'])==newbest['sha256']
 save(f'checkpoints/{label}/INDEX.json',dict(latest=rec,best=newbest,stop_reason=reason,source=source()))
 for q in dr.glob('checkpoint_*.pt'):
  if str(q) not in {rec['path'],newbest['path'] if newbest else ''}:
   with (dr/'RETIREMENTS.jsonl').open('a') as f:f.write(json.dumps(dict(path=str(q),sha256=sha(q),replacement=rec['path'],reason='only own V5.1 superseded, replacement readback verified'))+'\n')
   q.unlink()
 return rec

def load_record(e,o,sc,record):
 assert sha(record['path'])==record['sha256'];st=torch.load(record['path'],map_location='cpu',weights_only=False);assert state_digest(st)==record['state_digest'] and st['config_hash']==configsha();assert st['input_hashes']==read('INPUT_MANIFEST.json');e.actor.load_state_dict(st['actor']);o.load_state_dict(st['optimizer']);sc.load_state_dict(st['scheduler']);e.dual=st['dual'];set_rng(st['rng']);return st

def evaluate(e,label,u,panel='DEV',H=4,condition='native'):
 name=f'{label}_{u:05d}_{panel}_H{H}_{condition}';p=R/'results'/f'{name}.json'
 if p.exists():return json.load(open(p))
 tasks=read(f'manifests/{panel}_H{H}_{condition}.json')['tasks'];e.actor.eval();rr=[];tt=time.monotonic()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for t in tasks:
   for ns in CONFIG['eval_seeds']:
    loss,terms,x,extra=e.loss(t,ns);m=s.metric(t,x,ns);detail=collision_details(t,m['tail']);m.update(target=t['target'],loss=float(loss),KL=float(terms['KL']),terms={k:float(z) for k,z in terms.items()},collision=detail,step_KL=e.step_KLs,normalization='mean of three Gaussian components, then H steps');rr.append(m)
 e.verify();out=dict(label=label,update=u,panel=panel,H=H,condition=condition,seconds=time.monotonic()-tt,summary=s.summary(rr),direction=s.direction(rr),records=rr);out['summary']['collision_fraction']=sum(r['collision']['has_collision'] for r in rr)/len(rr);save(f'results/{name}.json',out);print('EVAL',name,out['summary'],flush=True);return out

def eligible(ev,h2,base,b2):
 q=ev['summary'];h=h2['summary'];a=base['summary'];b=b2['summary'];eligible=q['finite']==q['attempts'] and q['KL']<=1 and q['valid']/q['attempts']>=a['valid']/a['attempts']-.05 and h['KL']<=1 and h['success']/h['attempts']>=b['success']/b['attempts']-.05
 return dict(eligible=eligible,H4_KL=q['KL'],H2_KL=h['KL'],H4_valid_rate=q['valid']/q['attempts'],H2_success_rate=h['success']/h['attempts'])
def stopreason(ev,h2,base,b2):
 q=ev['summary'];h=h2['summary'];a=base['summary'];b=b2['summary'];reasons=[]
 if q['finite']!=q['attempts'] or h['finite']!=h['attempts']:reasons.append('nonfinite validation')
 if q['KL']>1 or h['KL']>1:reasons.append('mean conditional geometry KL > 1.0 nat/component')
 if q['valid']/q['attempts']<a['valid']/a['attempts']-.05:reasons.append('H4 backbone validity drop > 5 percentage points')
 if h['success']/h['attempts']<b['success']/b['attempts']-.05:reasons.append('H2 full-success retention drop > 5 percentage points')
 return '; '.join(reasons) or None

def train(label):
 seal=read('SEALED_CONFIG.json');assert seal['config_hash']==configsha() and seal['source']==source();assert not (R/f'results/{label}_STOP.json').exists(),'no automatic continuation after scientific stop'
 seed(CONFIG['seed']);e=Engine();e.label=label;o=optimizer(e);sc=scheduler(o);pool=trainpool();order=s.identity_order(pool,CONFIG['seed']+4);cursor=0;u=0;best=dict(update=None,success=-1,score=1e30);monitor={};dr=R/'checkpoints'/label
 if (dr/'INDEX.json').exists():
  st=load_record(e,o,sc,read(f'checkpoints/{label}/INDEX.json')['latest']);u=st['update'];order=st['sampler']['order'];cursor=st['sampler']['cursor'];best=st['best'];monitor=st['monitor']
 base=read('results/INITIAL_00000_DEV_H4_native.json');b2=read('results/INITIAL_00000_DEV_H2_native.json')
 if not (dr/'INDEX.json').exists():publish(e,o,sc,label,0,order,cursor,best,monitor)
 log=open(R/f'results/{label}_updates.jsonl','a',buffering=1);reason=None;start=time.monotonic()
 try:
  while u<CONFIG['updates']:
   # Reserve the next full atomic checkpoint BEFORE spending the next optimizer step.
   POLICY.require(R,220*2**20);before=dict(rng=rng_state(),order=list(order),cursor=cursor);neworder,newcursor,row=update(e,o,sc,pool,order,cursor,u);order,cursor=neworder,newcursor;u+=1;e.transaction_phase='BOUNDARY';log.write(json.dumps(row)+'\n')
   if u%25==0:save('LIVE_STATUS.json',dict(phase='TRAINING',arm=label,update=u,UTC=time.time(),last_seconds=row['seconds']));print('TRAIN',label,u,flush=True)
   if u in CONFIG['milestones']:
    ev=evaluate(e,label,u);h2=evaluate(e,label,u,H=2);monitor=eligible(ev,h2,base,b2);reason=stopreason(ev,h2,base,b2);q=ev['summary'];score=q['position_A']/3.8+q['orientation_deg']/180
    if not reason and monitor['eligible'] and (q['success']>best['success'] or q['success']==best['success'] and score<best['score']):best=dict(update=u,success=q['success'],score=score)
    # Publish complete stop-state BEFORE making the stop visible or returning.
    rec=publish(e,o,sc,label,u,order,cursor,best,monitor,reason)
    if u==100:
     (R/'evaluation').mkdir(exist_ok=True);er=atomic_checkpoint(R/'evaluation'/f'{label}_matched100.pt',dict(actor=e.actor.state_dict(),update=u,arm=label,dual=e.dual,source=source(),config_hash=configsha(),schema='inference_only_not_recovery',recovery_checkpoint_SHA=rec['sha256']),POLICY);save(f'evaluation/{label}_matched100.json',er)
    if reason:
     save(f'results/{label}_STOP.json',dict(reason=reason,exact_update=u,complete_verified_recovery=rec,no_invalid_promotion=True));break
    evaluate(e,label,u,condition='shifted')
   elif u%CONFIG['periodic_checkpoint']==0:publish(e,o,sc,label,u,order,cursor,best,monitor)
   if STOP_REQUEST:
    reason='graceful operational stop: '+STOP_REQUEST;rec=publish(e,o,sc,label,u,order,cursor,best,monitor,reason);save(f'results/{label}_STOP.json',dict(reason=reason,exact_update=u,complete_verified_recovery=rec));break
 except (FloatingPointError,RuntimeError,StoragePause) as ex:
  # Numerical failures in forward/backward/clipping precede AdamW, so actor/moments are unchanged.
  # Exceptions in an uncommitted AdamW transaction roll back to the verified complete checkpoint.
  reason=repr(ex);last=read(f'checkpoints/{label}/INDEX.json')['latest']
  if getattr(e,'transaction_phase','FORWARD')=='OPTIMIZER' or 'CUDA' in reason or 'out of memory' in reason:
   st=load_record(e,o,sc,last);u=st['update'];order=st['sampler']['order'];cursor=st['sampler']['cursor'];best=st['best'];monitor=st['monitor'];reason+='; rolled back to last verified safe transaction'
  elif getattr(e,'transaction_phase','FORWARD')=='FORWARD':
   order=before['order'];cursor=before['cursor'];set_rng(before['rng'])
  o.zero_grad(set_to_none=True);rec=publish(e,o,sc,label,u,order,cursor,best,monitor,reason);save(f'results/{label}_STOP.json',dict(reason=reason,exact_update=u,complete_verified_recovery=rec,no_invalid_promotion=True))
 finally:log.flush();os.fsync(log.fileno());log.close()
 e.verify([o]);save(f'results/{label}_FINISHED.json',dict(update=u,stop_reason=reason,J1_frozen=True,seconds=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),trainable=sum(p.numel() for p in e.actor.parameters() if p.requires_grad)));print('FINISHED',label,u,reason,flush=True)

def shift_pool(e,pool):
 cache={};out=[]
 # Exact V5 control convention: frozen V3, same two seeded prefix draws, no terminal outcome filtering.
 e.actor.load_state_dict(e.initial);e.arm='A';e.actor.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for t in pool:
   key=(t['identity'],t['t'])
   if key not in cache:
    x=torch.tensor(t['native_prefix'][:-2],dtype=torch.float64,device='cuda');eps=torch.randn(2,3,dtype=torch.float64,generator=torch.Generator().manual_seed(1101+int(t['identity'][:6],16))).cuda()
    for k in range(2):
     mu,ls,*_=e.predict(t,x);x=torch.cat([x,s.next_ca(x,s.physical(mu.double()+ls.double().exp()*eps[k]))[None]],0)
    cache[key]=x.cpu().tolist()
   z=copy.deepcopy(t);z['native_prefix']=cache[key];z['shift_note']='two frozen-V3 generated prefix actions; same target retained; feasibility not guaranteed';out.append(z)
 e.arm='D';return out

def prepare():
 e=Engine();allconf=json.load(open(OLD/'manifests/CONFIRM.json'))['tasks'];allids=sorted({t['identity'] for t in allconf});newids=set(allids[32:]);oldscored={r['identity'] for r in json.load(open(OLD/'results/A_H4_00000_CONFIRM_native.json'))['records']};assert not newids&oldscored and len(newids)==32
 census={}
 for H in [2,4]:
  dev=s.panel('DEV',H,24)
  conf=[]
  for h in sorted(newids):
   rows=[t for t in allconf if t['identity']==h and t['H']==H];mn=min(t['t'] for t in rows);conf.extend(t for t in rows if t['t']==mn)
  for name,pool in [('DEV',dev),('AUDIT',conf)]:
   save(f'manifests/{name}_H{H}_native.json',dict(tasks=pool,role='historical reused development' if name=='DEV' else 'previously unscored last32 protected VALIDATION identities'))
   if H==4:
    shifted=shift_pool(e,pool);save(f'manifests/{name}_H{H}_shifted.json',dict(tasks=shifted,no_terminal_score_filter=True))
   census[f'{name}_H{H}']=dict(tasks=len(pool),identities=len({t['identity'] for t in pool}),groups=len({t['protected_group_id'] for t in pool}),attempts=len(pool)*2)
 trainids={t['identity'] for t in trainpool()};assert not trainids&newids
 inputs={str(OLD/'manifests/TRAIN.json'):sha(OLD/'manifests/TRAIN.json'),str(OLD/'source/scripts/e017_v5/study.py'):sha(OLD/'source/scripts/e017_v5/study.py')};inputs.update({str(p):sha(p) for p in (R/'manifests').glob('*.json')});save('INPUT_MANIFEST.json',inputs);save('TASK_CENSUS.json',dict(panels=census,TRAIN_tasks=len(trainpool()),TRAIN_identities=len(trainids),TEST_evaluated=False,old_confirmation_preserved=True));print('TASKS_READY',census,flush=True)

def historical():
 pool={t['task_id']:t for t in json.load(open(OLD/'manifests/DEV.json'))['tasks']};raw=json.load(open(OLD/'results/C_H4_00350_DEV_native.json'));out=[];sens=[]
 for r in raw['records']:
  t=pool[r['task_id']];d=collision_details(t,r['tail']);z=dict(task_id=t['task_id'],identity=t['identity'],position_A=r['position_A'],orientation_deg=r['orientation_deg'],KL=r['KL'],valid=r['valid'],**d);out.append(z)
  if d['has_collision']:
   x=np.array(t['native_prefix']+r['tail']);g=torch.tensor(s.v.internal(x)[t['t']:],dtype=torch.float64,requires_grad=True);y=s.fk(t,g);ev=d['collision_pairs'][0];step=ev['step']-1;dist=torch.linalg.vector_norm(y[t['t']+step]-y[ev['earlier_index']]);gg=torch.autograd.grad(dist,g)[0]*torch.tensor([.1,.1,.1]);sens.append(dict(task_id=t['task_id'],seed=r['seed'],collision_step=step+1,standardized_component_sensitivity=gg.abs().sum(0).tolist(),per_step_sensitivity=gg.norm(dim=-1).tolist()))
 def corr(a,b):
  aa=np.array(a);bb=np.array(b);return float(np.corrcoef(aa,bb)[0,1]) if aa.std()>0 and bb.std()>0 else None
 summary=dict(attempts=len(out),invalid=sum(not r['valid'] for r in out),colliding=sum(r['has_collision'] for r in out),first_steps=dict(collections.Counter(str(r['first_collision_step']) for r in out if r['has_collision'])),pair_types=dict(collections.Counter(e['pair_type'] for r in out for e in r['collision_pairs'])),position_by_collision={str(c):float(np.mean([r['position_A'] for r in out if r['has_collision']==c])) for c in [False,True]},KL_by_collision={str(c):float(np.mean([r['KL'] for r in out if r['has_collision']==c])) for c in [False,True]},KL_position_correlation=corr([r['KL'] for r in out],[r['position_A'] for r in out]),collision_goal_correlation=corr([int(r['has_collision']) for r in out],[r['position_A'] for r in out]),mean_component_sensitivity=np.mean([r['standardized_component_sensitivity'] for r in sens],axis=0).tolist(),normalization='KL = mean over valid latent distance/angle/torsion, then mean over four steps, then decisions. Shared invertible transforms cancel Jacobians in KL. Fixed AA conditional, not joint categorical KL.',limits=s.v.LIMITS,missing_350_weights='Not reconstructed. Original recorded tails permit collision localization and kinematic action sensitivities only; exact policy Jacobians/KL per-step unavailable at350.')
 save('HISTORICAL_H4_COLLISION_AND_KL_ANALYSIS.json',dict(summary=summary,records=out,kinematic_sensitivities=sens));print('HISTORICAL',summary,flush=True)

def probe():
 seed(CONFIG['seed']);e=Engine();e.actor.eval();pool=trainpool();rows=[];ratios=[];params=[p for p in e.actor.parameters() if p.requires_grad];tt=time.monotonic()
 for i,t in enumerate(pool[:6]):
  with torch.autocast('cuda',dtype=torch.bfloat16):base,terms,x,extra=e.rollout(t,123+i,route='feedback_detached');cp=collision(x,t)
  row=dict(task_id=t['task_id'],losses={k:float(z.detach()) for k,z in terms.items()},collision=float(cp.detach()),gradient_norms={})
  for k,loss in [('goal',terms['goal']),('KL',terms['KL']),('local',.1*terms['local']),('BC',.1*terms['BC']),('collision',cp)]:
   gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);row['gradient_norms'][k]=math.sqrt(sum(float(g.float().square().sum()) for g in gs if g is not None));del gs
  if row['gradient_norms']['collision']>1e-8:ratios.append(.3*row['gradient_norms']['goal']/row['gradient_norms']['collision'])
  rows.append(row)
 assert ratios,'TRAIN probes show no collision gradients; cannot calibrate weight';coef=float(np.clip(np.median(ratios),.05,5.));cfg=copy.deepcopy(DEFAULT);cfg['collision_weight']=coef;save('CONFIG.json',cfg);save('TRAIN_ONLY_GRADIENT_CALIBRATION.json',dict(records=rows,collision_weight=coef,rule='median .3*goal-gradient-norm / rawcollision-gradient-norm, clipped to[.05,5], TRAIN-only first6 tasks; dual target.8 and eta.05 fixed prospectively',seconds=time.monotonic()-tt));e.verify();print('CALIBRATED',coef,flush=True)

def baseeval():
 seed(CONFIG['seed']);e=Engine()
 for H in [2,4]:evaluate(e,'INITIAL',0,H=H)
 evaluate(e,'INITIAL',0,condition='shifted')

def finaleval():
 seed(CONFIG['seed']);e=Engine()
 for name in ['INITIAL','FROZEN_J1']+[a+'_MATCHED100' for a in CONFIG['arms']]+CONFIG['arms']:
  if name=='FROZEN_J1':
   # Actor-free reference distributions: bypass residual actor, preserve exact J1 conditional action law.
   def pred(t,x,goal_on=True):
    ref=e.prior_forward(e.p(t,x));a=s.AA.index(t['sequence'][len(x)]);return ref['mu'][0,a],ref['ls'][0,a],ref['mu'][0,a],ref['ls'][0,a],ref,None
   original=e.predict;e.predict=pred;u=0
  elif name=='INITIAL':
   e.actor.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['actor']);u=0
  else:
   arm=name.split('_')[0]
   if name.endswith('_MATCHED100'):rec=read(f'evaluation/{arm}_matched100.json')
   else:
    ix=read(f'checkpoints/{arm}/INDEX.json');rec=ix['best'] or ix['latest']
   st=torch.load(rec['path'],map_location='cpu',weights_only=False);assert sha(rec['path'])==rec['sha256'];e.actor.load_state_dict(st['actor']);e.dual=st['dual'];u=st['update'];e.label=arm;save(f'results/{name}_AUDIT_CHECKPOINT.json',dict(record=rec,selection='predeclared matched100' if name.endswith('_MATCHED100') else 'best eligible DEV or failure-state latest explicitly unqualified'))
  for H in [2,4]:evaluate(e,name,u,'AUDIT',H)
  evaluate(e,name,u,'AUDIT',4,'shifted')
  if name=='FROZEN_J1':e.predict=original
  e.label='A'
 e.verify();save('FINAL_EVALUATION_STATUS.json',dict(complete=True,J1_frozen=True,no_RL=True))

def handler(sig,frm):
 global STOP_REQUEST
 STOP_REQUEST=signal.Signals(sig).name

if __name__=='__main__':
 signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler);p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--arm');a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='historical':historical()
 elif a.command=='probe':probe()
 elif a.command=='baseline':baseeval()
 elif a.command=='train':train(a.arm)
 elif a.command=='final':finaleval()
 else:raise ValueError(a.command)
