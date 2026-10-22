"""E017 V4: native one-step versus eight-step geometry credit. No SAC/J1 optimization."""
import argparse,copy,hashlib,json,math,os,pathlib,random,shutil,time,subprocess
from collections import Counter,defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from protein_joint_sac.crisp_v3.prior import FrozenPrior,file_sha,tensor_sha
from protein_joint_sac.crisp_v3.networks import LowActor,PROFILES
from protein_joint_sac.models.batched_bc import features_batch,encode_batch
from protein_joint_sac.crisp_v3.environment import frame,goal_tokens,assess
from protein_joint_sac.crisp_v3.density import sample,log_prob
from protein_joint_sac.geometry.ca_builder import next_ca
from protein_joint_sac.data.scaled_supervised import Data,canonical,internal
from protein_joint_sac.data.accepted_supervised import CoordinateReader
from protein_joint_sac.contracts import AA,BuildState
from protein_joint_sac.crisp_v3.recovery import atomic_json,sha,StoragePolicy,state_digest,atomic_checkpoint
from protein_joint_sac.crisp_v3.storage import rng_state,set_rng
ROOT=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v4_multistep_20261022_v1')
PARENT=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_crisp_v3_throughput_20261021_v1/training_runs/run_d3e78a804534473884df42f0f34b7c41/checkpoints/checkpoint_00010000_7d8c5d05.pt')
SHA='ccd11e9788a30a4f3c398b57ee655d24f3e4b636fec875469f080b711ee3e8e9'
LIMITS=json.load(open('/mnt/d/Simone/proteinGen_data/e017_crisp_v3_20261021_v1/configs/NATIVE_CALIBRATION.json'))
POLICY=StoragePolicy(min_headroom_bytes=2*2**30,warning_free_bytes=4*2**30,near_term_growth_bytes=32*2**20,checkpoint_hint_bytes=240*2**20)
CONFIG=dict(schema='e017_v4_8step_v1',horizon=8,TRAIN_identities=64,VALIDATION_identities=32,batch=2,updates=500,milestones=[0,100,250,500],seed=4127,lr=1e-5,lr_floor=1e-6,warmup=25,clip=1.,weights=dict(goal=1.,local=.1,BC=.1,KL=.01),precision='BF16 neural, FP64 geometry; FP32 optimizer',J1='frozen weights, differentiable coordinate inputs',B='V3 one-step native-prefix latent-mean guide, matched native BC and local/KL terms',C='sampled differentiable 8-step terminal guide; native BC on separate native prefix',IK_steps=[8,64],IK_lr=.05,validation_noise_seeds=[6101,6102],selection='lowest held-out mean position error among checkpoints retaining >=Avalidity-0.05; not biological promotion',success='position<=8 A, orientation<=35 deg AND original V3 prefix validity',GO='>=20% validation position improvement vs B, identity paired CI supports gain, joint hit improves and validity decline<=0.05',stop='nonfinite or unsupported action: immediate; completion<.95 or KL>1nat/component at two validations: stop; no automatic extension')

def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def save(name,value):atomic_json(ROOT/name,value)
def dataset(split):return json.load(open(ROOT/'manifests'/f'{split}.json'))['tasks']
def config_hash():return hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()
def task_goal(task,target=None):
 return dict(kind='P2',N=task['N'],targets=[dict(index=task['t']+7,position=task['target'],orientation=task['orientation'],tolerance_A=8.,orientation_tolerance_deg=35.)],motifs={})
def prefix(task,step=0,device='cuda'):
 x=torch.tensor(task['native_prefix'] if step==0 else task['native_prefix']+task['hidden_native_future'][:step],dtype=torch.float64,device=device)
 return x

def prepare_tasks():
 from protein_joint_sac.crisp_v3.execution import set_loader
 set_loader('arrow_scalar_lru4');data=Data(False);reader=CoordinateReader(data.proposal.external);exclusions=[];outputs={}
 for split,count in [('TRAIN',64),('VALIDATION',32)]:
  tasks=[];groups=set();candidates=[h for h in data.jointids[split] if 64<=len(data.seqs[h])<=96];candidates.sort(key=lambda h:hashlib.sha256((str(CONFIG['seed'])+split+h).encode()).hexdigest())
  for h in candidates:
   if len(tasks)==count:break
   group=data.proposal.sequences[h]['protected_group_id']
   if group in groups:continue
   o=min(data.owners[h],key=lambda o:hashlib.sha256((h+o).encode()).hexdigest());r=data.rows[o];t=24 if len(tasks)%2==0 else 32
   try:
    assert r['split']==split and r['sequence_hash']==h
    x=canonical(reader.read(r));g=internal(x);seq=data.seqs[h];assert len(x)==len(seq)
    target=x[t+7];orient=frame(torch.tensor(x[:t+8])).tolist();goal=dict(kind='P2',N=t+8,targets=[dict(index=t+7,position=target.tolist(),orientation=orient,tolerance_A=8.,orientation_tolerance_deg=35.)],motifs={});w=assess(BuildState(tuple(seq[:t+8]),torch.tensor(x[:t+8]),t+8),goal,LIMITS)
    if not w['valid']:raise ValueError('native prefix+8 fails original V3 validity')
    rebuilt=torch.tensor(x[:t]);
    for k in range(8):rebuilt=torch.cat([rebuilt,next_ca(rebuilt,torch.tensor(g[t+k]))[None]],0)
    error=float((rebuilt-torch.tensor(x[:t+8])).abs().max())
    if error>1e-6:raise ValueError('native action roundtrip mapping/support failed')
    tasks.append(dict(identity=h,protected_group_id=group,split=split,observation=o,source_sha=r['source_sha'],entry_id=r['entry_id'],coordinate_path=r['coordinate_path'],coordinate_row=r['coordinate_row'],N=len(seq),t=t,sequence=seq,native_prefix=x[:t].tolist(),hidden_native_future=x[t:t+8].tolist(),native_actions=g[t:t+8].tolist(),target=target.tolist(),orientation=orient,native_witness_max_error_A=error,native_witness=w,notes='Hidden future used only as supervision/witness, never passed to rollout actor'))
    groups.add(group)
   except (ValueError,AssertionError) as e:exclusions.append(dict(split=split,identity=h,observation=o,reason=str(e)))
  if len(tasks)!=count:raise ValueError('insufficient eligible tasks')
  outputs[split]=tasks;save(f'manifests/{split}.json',dict(schema='e017_v4_tasks_v1',split=split,selection='hash-ranked identity-first, unique protected groups, qualified complete, length64-96; native witness filtering only',tasks=tasks,proposal_SHA=file_sha(data.proposal.root/'PROPOSAL_MANIFEST.json') if hasattr(data.proposal,'root') else file_sha('/mnt/d/Simone/proteinGen_data/pdb_supervised_split_50id80cov_20261016_v1/PROPOSAL_MANIFEST.json')))
 assert not {x['identity'] for x in outputs['TRAIN']}&{x['identity'] for x in outputs['VALIDATION']}
 assert not {x['protected_group_id'] for x in outputs['TRAIN']}&{x['protected_group_id'] for x in outputs['VALIDATION']}
 save('manifests/EXCLUSIONS.json',exclusions);save('CONFIG_PROPOSED.json',CONFIG);print('TASKS_PREPARED',len(outputs['TRAIN']),len(outputs['VALIDATION']),len(exclusions),flush=True)

class Engine:
 def __init__(self):
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert sha(PARENT)==SHA;self.prior=FrozenPrior('cuda');s=torch.load(PARENT,map_location='cpu',weights_only=False)
  self.initial={k[len('models.actor.'):]:v.clone() for k,v in s['system'].items() if k.startswith('models.actor.')};del s
  self.actor=LowActor(PROFILES['research']['actor']).cuda();self.reset();self.j1_initial=self.prior.identity
 def reset(self):
  self.actor.load_state_dict(self.initial);self.actor.eval();self.actor.requires_grad_(True);self.actor.aa.requires_grad_(False)
 def p(self,task,x):
  t=len(x);ids=torch.tensor([[AA.index(a) for a in task['sequence'][:t]]],device=x.device);return dict(x=x[None],ids=ids,t=torch.tensor([t],device=x.device),N=torch.tensor([task['N']],device=x.device),valid=torch.ones(1,t,dtype=torch.bool,device=x.device))
 def prior_forward(self,p):
  # Frozen parameters != no_grad. Input derivatives are preserved through all J1 features and layers.
  h=encode_batch(self.prior.model,p,activation_checkpoint=torch.is_grad_enabled());B=len(h);ids=torch.arange(20,device=h.device)[None].expand(B,20);z=self.prior.model.geom(torch.cat([h[:,None].expand(B,20,352),self.prior.model.selected_aa(ids),h.new_zeros(B,20,22)],-1)).float()
  return dict(h=h.float(),logits=self.prior.model.aa_head(h).float(),mu=z[:,:,:3],ls=z[:,:,3:].clamp(-5,2))
 def predict(self,task,x,goal=True):
  p=self.p(task,x);ref=self.prior_forward(p);gt,gv=goal_tokens(task_goal(task),'cuda')
  if not goal:gt=gt*0;gv=gv&False
  b=dict(prefix=p,reference=ref,features=features_batch(p,chemistry=True),goals=gt,goal_valid=gv,context=x.new_zeros(1,64).float(),subgoal=x.new_zeros(1).float());l,mu,ls=self.actor(b);a=AA.index(task['sequence'][len(x)]);return mu[0,a],ls[0,a],ref['mu'][0,a],ref['ls'][0,a],ref,b
 def goal_loss(self,x,task):
  target=x.new_tensor(task['target']);d2=(x[-1]-target).square().sum();position=d2/(64+d2);ori=(3-(frame(x)*x.new_tensor(task['orientation'])).sum())/4
  return position+.5*ori.clamp(0,1),dict(position=position,orientation=ori)
 def kl(self,mu,ls,rm,rl):return (rl-ls+.5*((ls.exp()/rl.exp()).square()+((mu-rm)/rl.exp()).square())-.5).mean()
 def local(self,x,task):
  t=task['t'];d=torch.linalg.vector_norm(x[t:]-x[t-1:-1],dim=-1);spacing=(F.relu(LIMITS['adjacent_low_A']-d).square()+F.relu(d-LIMITS['adjacent_high_A']).square()).mean()/.25**2
  dist=torch.linalg.vector_norm(x[t:,None]-x[None,:],dim=-1);i=torch.arange(t,len(x),device=x.device)[:,None];j=torch.arange(len(x),device=x.device)[None];mask=i-j>=3;clash=(F.relu(2-dist).square()*mask).sum()/8
  return spacing+clash
 def rollout(self,task,noise_seed,arm='C',delta=None,retain=False):
  gen=torch.Generator(device='cpu').manual_seed(noise_seed);epsilon=torch.randn(8,3,dtype=torch.float64,generator=gen,device='cpu').cuda();x=prefix(task);mus=[];lss=[];us=[];kl=[];gs=[];refs=[]
  for k in range(8):
   inp=prefix(task,k) if arm=='B' else x
   mu,ls,rm,rl,ref,b=self.predict(task,inp);mus.append(mu);lss.append(ls);refs.append(ref)
   if retain:mu.retain_grad();ls.retain_grad()
   u=mu.double()+ls.double().exp()*epsilon[k]
   if delta is not None:u=u+delta[k]
   g=torch.stack([u[0].exp(),math.pi*u[1].sigmoid(),math.pi*u[2].tanh()]);
   if not torch.isfinite(g).all() or not bool((g[0]>0)&(g[1]>0)&(g[1]<math.pi)&(g[2].abs()<math.pi)):raise FloatingPointError('unrepresentable transformed draw; no clamp')
   us.append(u);gs.append(g);kl.append(self.kl(mu,ls,rm,rl))
   if arm=='B':
    # Literal V3 deterministic next-target guidance from a native prefix.
    if k==7:
     m=mu.double();mg=torch.stack([m[0].exp(),math.pi*m[1].sigmoid(),math.pi*m[2].tanh()]);x=torch.cat([inp,next_ca(inp,mg)[None]],0)
   else:x=torch.cat([x,next_ca(x,g)[None]],0)
  guide,parts=self.goal_loss(x,task)
  # Same one native-prefix BC decision in each arm; never used as rollout state in C.
  bi=noise_seed%8;ni=prefix(task,bi);bm,bl,_,_,_,_=self.predict(task,ni);ng=torch.tensor(task['native_actions'][bi],device='cuda',dtype=torch.float64);bc=-log_prob(bm.double(),bl.double(),ng,torch.ones(3,device='cuda',dtype=torch.bool))/3
  if arm=='B':
   local_x=prefix(task)
   for g in gs:local_x=torch.cat([local_x,next_ca(local_x,g)[None]],0)
   local=self.local(local_x,task)
  else:local=self.local(x,task)
  terms=dict(goal=guide,local=local,BC=bc,KL=torch.stack(kl).mean());total=sum(CONFIG['weights'][key]*value for key,value in terms.items())
  return total,terms,x,dict(mus=mus,ls=lss,u=us,epsilon=epsilon,refs=refs,goal_parts=parts)
 def evaluate_one(self,task,noise_seed):
  # Evaluation always samples generated prefixes, regardless of training arm.
  _,terms,x,extra=self.rollout(task,noise_seed,'C');seq=task['sequence'][:len(x)];gt=task_goal(task);gt['N']=len(x);s=BuildState(tuple(seq),x,len(x));m=assess(s,gt,LIMITS);er=m['errors'][0];initial=float(torch.linalg.vector_norm(prefix(task)[-1]-x.new_tensor(task['target'])));g=torch.stack([torch.stack([u[0].exp(),math.pi*u[1].sigmoid(),math.pi*u[2].tanh()]) for u in extra['u']]);m.update(identity=task['identity'],seed=noise_seed,split=task['split'],position_error_A=er['position_error_A'],orientation_error_deg=er['orientation_error_deg'],initial_residual_A=initial,improves_initial=er['position_error_A']<initial,target_hit=er['position_error_A']<=8 and er['orientation_error_deg']<=35,KL=float(terms['KL']),local=float(terms['local']),adjacent_min_A=float(g[:,0].min()),adjacent_max_A=float(g[:,0].max()),angle_mean_rad=float(g[:,1].mean()),torsion_mean_rad=float(g[:,2].mean()),finite=bool(torch.isfinite(x).all()),coordinate_tail=x[task['t']:].cpu().tolist());return m
 def evaluate(self,tasks,label,update):
  rows=[];self.actor.eval();start=time.monotonic()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for task in tasks:
    for ns in CONFIG['validation_noise_seeds']:rows.append(self.evaluate_one(task,ns))
  out=dict(arm=label,update=update,seconds=time.monotonic()-start,records=rows,summary=summary(rows));save(f'results/{label}_{update:05d}_{tasks[0]["split"]}.json',out);print('EVAL',label,update,out['summary'],flush=True);return out

def summary(rows):
 return dict(attempts=len(rows),identities=len({x['identity'] for x in rows}),position_error_A=float(np.mean([x['position_error_A'] for x in rows])),orientation_error_deg=float(np.mean([x['orientation_error_deg'] for x in rows])),success=sum(x['success'] for x in rows),target_hit=sum(x['target_hit'] for x in rows),valid=sum(x['valid'] for x in rows),finite=sum(x['finite'] for x in rows),improves_initial=sum(x['improves_initial'] for x in rows),KL=float(np.mean([x['KL'] for x in rows])))

def gradient_preflight():
 seed(CONFIG['seed']);e=Engine();task=dataset('TRAIN')[0];e.actor.eval();torch.cuda.reset_peak_memory_stats();start=time.monotonic();delta=torch.zeros(8,3,dtype=torch.float64,device='cuda',requires_grad=True)
 total,terms,x,z=e.rollout(task,8101,'C',delta);gg=torch.autograd.grad(terms['goal'],delta,retain_graph=True)[0];mg=torch.autograd.grad(terms['goal'],z['mus'],retain_graph=True);pgrad=torch.autograd.grad(terms['goal'],list(e.actor.encoder.parameters()),allow_unused=True,retain_graph=True);norm=float(torch.sqrt(sum(q.square().sum() for q in pgrad if q is not None)))
 errors=[]
 for k in [0,3,7]:
  j=int(gg[k].abs().argmax());h=.002;values=[]
  for sign in [-1,1]:
   d=torch.zeros_like(delta);d[k,j]=sign*h
   with torch.no_grad():_,tt,_,_=e.rollout(task,8101,'C',d);values.append(float(tt['goal']))
  fd=(values[1]-values[0])/(2*h);anal=float(gg[k,j]);rel=abs(fd-anal)/max(.001,abs(fd),abs(anal));errors.append(dict(step=k,channel=j,analytic=anal,finite_difference=fd,relative_error=rel))
 assert torch.isfinite(gg).all() and min(float(torch.linalg.vector_norm(v)) for v in mg)>1e-9
 assert max(x['relative_error'] for x in errors)<.08,errors
 # Input-to-J1 derivative, excluding any J1 weight gradient.
 px=prefix(task).requires_grad_(True);ref=e.prior_forward(e.p(task,px));rgrad=torch.autograd.grad(ref['h'].square().sum(),px)[0];assert torch.isfinite(rgrad).all() and rgrad.abs().max()>0
 norms={}
 for key,term in terms.items():
  grad=torch.autograd.grad(CONFIG['weights'][key]*term,[p for p in e.actor.parameters() if p.requires_grad],retain_graph=True,allow_unused=True);norms[key]=math.sqrt(sum(float(p.float().square().sum()) for p in grad if p is not None))
 del pgrad,grad;total.backward();assert all(p.grad is None for p in e.prior.model.parameters());e.prior.verify();assert all(torch.isfinite(p.grad).all() for p in e.actor.parameters() if p.grad is not None)
 result=dict(PASS=True,FD=errors,latent_action_gradient_norms=gg.norm(dim=-1).detach().cpu().tolist(),policy_mu_gradient_norms=[float(g.norm()) for g in mg],shared_encoder_gradient_norm=norm,J1_input_gradient_norm=float(rgrad.norm()),J1_weights_frozen=True,weighted_loss_gradient_norms=norms,loss_terms={k:float(v.detach()) for k,v in terms.items()},peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-start,trainable_parameters=sum(p.numel() for p in e.actor.parameters() if p.requires_grad))
 save('results/GRADIENT_PREFLIGHT.json',result);print('PREFLIGHT',result,flush=True)

def ckpt(e,opt,sched,arm,update,best,order,cursor,strikes):
 e.prior.verify([opt]);state=dict(schema='e017_v4_recovery_v1',actor=e.actor.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),arm=arm,update=update,best=best,order=order,cursor=cursor,monitor=dict(strikes=strikes),source_commit=json.load(open(ROOT/'SEALED_CONFIG.json'))['source_commit'],rng=rng_state(),config=CONFIG,config_hash=config_hash(),parent=str(PARENT),parent_SHA=SHA,J1_SHA=e.prior.identity,TRAIN_SHA=sha(ROOT/'manifests/TRAIN.json'),VALIDATION_SHA=sha(ROOT/'manifests/VALIDATION.json'))
 path=ROOT/'checkpoints'/arm/f'checkpoint_{update:05d}.pt';path.parent.mkdir(exist_ok=True);idxpath=path.parent/'INDEX.json';old=json.load(open(idxpath)) if idxpath.exists() else {'latest':None,'best':None};record=atomic_checkpoint(path,state,POLICY);record.update(update=update);newbest=record if best.get('update')==update else old['best'];atomic_json(idxpath,dict(latest=record,best=newbest))
 # Retire only this arm's superseded run-owned complete checkpoints, after verified replacement.
 for f in path.parent.glob('checkpoint_*.pt'):
  if str(f) not in {record['path'],newbest['path'] if newbest else ''}:
   with (path.parent/'RETIREMENTS.jsonl').open('a') as log:log.write(json.dumps(dict(path=str(f),sha256=sha(f),reason='verified rolling replacement within V4 run'))+'\n')
   f.unlink()
 return path

def train(arm):
 assert arm in ['B','C'];assert not (ROOT/f'results/{arm}_STOP.json').exists(),'scientific stop requires review';sealed=json.load(open(ROOT/'SEALED_CONFIG.json'));assert sealed['config']==CONFIG and sealed['config_hash']==config_hash();assert subprocess.check_output(['git','-C',str(ROOT/'source'),'rev-parse','HEAD'],text=True).strip()==sealed['source_commit'];seed(CONFIG['seed']);e=Engine();tasks=dataset('TRAIN');val=dataset('VALIDATION');order=list(range(len(tasks)));random.Random(CONFIG['seed']).shuffle(order);cursor=0
 opt=torch.optim.AdamW([p for p in e.actor.parameters() if p.requires_grad],lr=CONFIG['lr'],weight_decay=.01,foreach=False)
 mult=lambda u: min(1.,(u+1)/25)*(.1+.9*(1+math.cos(math.pi*min(u,500)/500))/2)
 sched=torch.optim.lr_scheduler.LambdaLR(opt,mult);best=dict(position=1e30,update=None);start_update=0;strikes=0;idxpath=ROOT/'checkpoints'/arm/'INDEX.json'
 if idxpath.exists():
  ix=json.load(open(idxpath));st=torch.load(ix['latest']['path'],map_location='cpu',weights_only=False);assert sha(ix['latest']['path'])==ix['latest']['sha256'] and st['config_hash']==config_hash();e.actor.load_state_dict(st['actor']);opt.load_state_dict(st['optimizer']);sched.load_state_dict(st['scheduler']);set_rng(st['rng']);order=st['order'];cursor=st['cursor'];best=st['best'];start_update=st['update'];strikes=st['monitor']['strikes']
 if not idxpath.exists():ckpt(e,opt,sched,arm,0,best,order,cursor,strikes)
 log=open(ROOT/f'results/{arm}_updates.jsonl','a',buffering=1);baseline=json.load(open(ROOT/'results/A_00000_VALIDATION.json'))['summary']
 save(f'results/{arm}_STATUS.json',dict(status='RUNNING',completed_updates=start_update))
 for u in range(start_update,500):
  POLICY.require(ROOT,4096);e.actor.train();opt.zero_grad(set_to_none=True);start=time.monotonic();parts=defaultdict(float);components=defaultdict(float);action_gradients=[];ids=[]
  for j in range(2):
   if cursor==len(order):random.Random(CONFIG['seed']+u).shuffle(order);cursor=0
   task=tasks[order[cursor]];cursor+=1;ids.append(task['identity']);ns=CONFIG['seed']+100000+2*u+j
   with torch.autocast('cuda',dtype=torch.bfloat16):loss,terms,_,details=e.rollout(task,ns,arm,retain=(u+1 in [1,100,250,500]))
   (loss/2).backward()
   if u+1 in [1,100,250,500]:action_gradients.append([float(x.grad.norm()) if x.grad is not None else 0. for x in details['mus']])
   for k,v in details['goal_parts'].items():components[k]+=float(v.detach())/2
   for k,v in terms.items():parts[k]+=float(v.detach())/2
  if any(p.grad is not None for p in e.prior.model.parameters()):raise RuntimeError('J1 gradient leak')
  gn=torch.nn.utils.clip_grad_norm_([p for p in e.actor.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step();sched.step();torch.cuda.synchronize();row=dict(update=u+1,loss=sum(CONFIG['weights'][k]*v for k,v in parts.items()),terms=dict(parts),terminal_components=dict(components),action_distribution_gradient_norms=action_gradients,gradient_norm=float(gn),identities=ids,seconds=time.monotonic()-start);log.write(json.dumps(row)+'\n')
  if (u+1)%25==0:
   save(f'results/{arm}_STATUS.json',dict(status='RUNNING',completed_updates=u+1,last_seconds=row['seconds']));print('TRAIN',arm,u+1,row['loss'],row['seconds'],flush=True)
  if u+1 in [100,250,500]:
   ev=e.evaluate(val,arm,u+1);e.evaluate(tasks[:8],arm,u+1);good=ev['summary']['valid']/ev['summary']['attempts']>=baseline['valid']/baseline['attempts']-.05
   if good and ev['summary']['position_error_A']<best['position']:best=dict(position=ev['summary']['position_error_A'],update=u+1)
   bad=ev['summary']['finite']/ev['summary']['attempts']<.95 or ev['summary']['KL']>1;strikes=strikes+1 if bad else 0;ckpt(e,opt,sched,arm,u+1,best,order,cursor,strikes)
   if strikes>=2:save(f'results/{arm}_STOP.json',dict(update=u+1,reason='sealed completion/KL gate'));break
  elif (u+1)%100==0:ckpt(e,opt,sched,arm,u+1,best,order,cursor,strikes)
 log.close();save(f'results/{arm}_STATUS.json',dict(status='COMPLETED' if u+1==500 else 'STOPPED_SCIENTIFIC',completed_updates=u+1));e.prior.verify([opt]);save(f'results/{arm}_FINISHED.json',dict(update=u+1,J1_frozen=True,actor_SHA=tensor_sha(e.actor),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),source_commit=subprocess.check_output(['git','-C',str(ROOT/'source'),'rev-parse','HEAD'],text=True).strip()))

def baseline():
 assert json.load(open(ROOT/'SEALED_CONFIG.json'))['config_hash']==config_hash();e=Engine();e.evaluate(dataset('VALIDATION'),'A',0);e.evaluate(dataset('TRAIN')[:8],'A',0)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','preflight','baseline','train']);ap.add_argument('--arm',choices=['B','C']);args=ap.parse_args()
 if args.command=='prepare':prepare_tasks()
 elif args.command=='preflight':gradient_preflight()
 elif args.command=='baseline':baseline()
 else:train(args.arm)
