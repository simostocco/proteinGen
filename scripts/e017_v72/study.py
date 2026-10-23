"""Bounded H4 counterfactual action ranking; neural actor and J1 always frozen."""
import collections,copy,fcntl,hashlib,json,math,os,random,runpy,shutil,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v72_counterfactual_ranking_20261023_v1')
B=R.parent/'e017_v71_expected_return_calibration_20261023_v1';V7=R.parent/'e017_v7_amortized_lowlevel_20261023_v1'
s=runpy.run_path(str(R/'source/scripts/e017_v71/study.py'));v=s['v'];AA=s['AA'];sha=s['sha'];digest=s['state_digest'];read=s['read']
from protein_joint_sac.crisp_v3.recovery import atomic_json,atomic_checkpoint,StoragePolicy
from protein_joint_sac.crisp_v3.storage import rng_state,set_rng
CFG=dict(schema='E017_V72_v1',seed=87272,fit=32,dev=12,confirm=32,generated_confirm=12,actions=8,stages=[0,1,2,3],replicates_per_bank=16,banks=['EST','ASSESS'],gamma=.995,
 precision='FP32 neural inference, FP64 geometry; temperature1; batch16; same frozen policy across all conditions',
 target='ordinary discounted terminal V7 task reward; soft diagnostic separately future-only alpha.001 lambda.1; no actor optimization',
 E_gate='FIT stages0to2: >=25%states and>=12proteins have EST action pair absdelta>=.1 and>1.96pairedMCSE; no CONFIRM inspection',
 E='same5889parameter MLP warm-startedV71 with frozen V71scaling, Adam lr.0003 batch64,1000CPUupdates; MSE to FIT ESTmeans; DEV ESTselected action return selects100/250/500/1000; noASSESSlabels for fitting/selection',
 primary='oldMLP vs recalibratedE ifexecuted, chosen once by DEV stages0to2 EST mean selected action return, tieoldMLP; beforeCONFIRMoutcomes',
 promotion='CONFIRMstage0 primary: assessedreturn advantage over uniform>=.05 and overheuristic>=.025, pairedprotein95%CIlower>0 forboth; assessed success gainvsuniform>=.03 and notbelowheuristic; supportedpairranking>=.6 with95%lower>.5,>=16pairs>=12groups; selectedvalidity vsheuristic decline<=.05; selectedstateconditionalKL<=1; generatedstage0 assessedreturngainvsuniform>0 andvaliditydecline<=.05. NoSACexecution evenifGO.',
 supported='pair ESTabsdelta>=.1 and>1.96pairedSE; independent ASSESS support reported separately; primary ranking uses pairs supported in both banks with same sign',
 identifiability='nominal paired95%MC contrasts; Bonferroni28contrast normal-approx z3.124 also reported; finiteM limitations; no uncertain ground-truth ranking',
 reserve_bytes=2*2**30,max_GPU_bytes=6*2**30,max_estimated_collection_seconds=5400,maximum_new_MLP_updates=1000,actor_updates=0,large_critic_updates=0,TEST=False)
POLICY=StoragePolicy(min_headroom_bytes=2*2**30,near_term_growth_bytes=128*2**20,checkpoint_hint_bytes=4*2**20)
STOP=False
def save(name,x):atomic_json(R/name,x)
def ns(*x):return int(hashlib.sha256(('87272|'+ '|'.join(map(str,x))).encode()).hexdigest()[:15],16)
def source():return subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip()
def budget():
 POLICY.require(R,4*2**20)
 if STOP:raise InterruptedError('cooperative boundary stop')
 if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()>CFG['max_GPU_bytes']:raise RuntimeError('GPU6GiBgate')
def handle(*_):
 global STOP
 STOP=True
def prepare():
 old=read(B/'manifests/SPLITS.json');pool=s['taskmap']();rank=lambda x:ns('population',x)
 fit=sorted(old['FIT'],key=rank)[:32];dev=sorted(old['DEV'],key=rank)[:12];unused=set(pool)-set(old['FIT']+old['DEV']+old['CONFIRM']);confirm=sorted(unused,key=rank)[:32];assert len(confirm)==32
 groups=[{pool[x]['protected_group_id'] for x in ids} for ids in [fit,dev,confirm]];assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
 save('manifests/TASKS.json',dict(FIT=[pool[x] for x in fit],DEV=[pool[x] for x in dev],CONFIRM=[pool[x] for x in confirm],SHIFT_IDS=sorted(confirm,key=lambda x:ns('shift',x))[:12],selection='TRAIN-only tasks; CONFIRM from128V71unused identities, disjoint all V71FIT/DEV/CONFIRM; protectedgroups disjoint newFIT/DEV/CONFIRM',parent_candidate0_reference=str(V7/'results/TRAIN_candidates.jsonl')))
 inputs=[B/'FINAL_E017_V71_SCIENTIFIC_REPORT.md',B/'PROVENANCE_SHA256.json',B/'results/simple_models.pt',B/'checkpoints/critic_0250_stop.pt',V7/'results/TRAIN_candidates.jsonl',V7/'manifests/TRAIN.json',v['PARENT'],v['s'].__file__,v['s'].v.__file__,R/'source/src/protein_joint_sac/crisp_v3/prior.py',R/'source/src/protein_joint_sac/crisp_v3/environment.py',R/'source/src/protein_joint_sac/crisp_v3/networks.py',R/'source/src/protein_joint_sac/geometry/ca_builder.py',R/'manifests/TASKS.json']
 j1=R.parent/'e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt';inputs.append(j1)
 assert sha(v['PARENT'])==v['PSHA'];assert sha(j1)=='881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412';assert sha(B/'checkpoints/critic_0250_stop.pt')=='35c8375c83547dd68a233f619e7119fe98ee61d6682a7e0552358920fb672b39'
 save('INPUT_AUDIT.json',dict(input_SHA256={str(p):sha(p) for p in inputs},V71_source='1f9a91b226460f163e13c501b9d2e1c524d84a42',actor_SHA=v['PSHA'],actor_parameters=17438114,J1_parameters=12351746,critic_parameters_each=12839497,MLP_parameters=5889,D_free_bytes=shutil.disk_usage(R).free,all_models_frozen=True))
 save('DESIGN.json',CFG)
def seal():
 assert not (R/'PROTOCOL.json').exists();audit=read(R/'INPUT_AUDIT.json');paths={str(p):sha(p) for p in (R/'source/scripts/e017_v72').glob('*.py')};save('PROTOCOL.json',dict(config=CFG,source=source(),input_SHA256={**audit['input_SHA256'],**paths},UTC=time.time(),freeze_before_prospective_outcomes=True))
def check():
 p=read(R/'PROTOCOL.json');assert p['source']==source();assert p['config']==CFG
 for path,h in p['input_SHA256'].items():assert sha(path)==h,path
def next_batch(x,g):
 u=torch.nn.functional.normalize(x[:,-1]-x[:,-2],dim=-1);ee=x[:,-3]-x[:,-2];ee=torch.nn.functional.normalize(ee-(ee*u).sum(-1,keepdim=True)*u,dim=-1);n=torch.linalg.cross(u,ee,dim=-1);d,a,t=g.unbind(-1);direction=-a.cos()[:,None]*u+a.sin()[:,None]*(t.cos()[:,None]*ee+t.sin()[:,None]*n);return torch.cat([x,(x[:,-1]+d[:,None]*direction)[:,None]],dim=1)
class BatchEngine:
 def __init__(self):
  self.e=v['Engine']();self.e.actor.requires_grad_(False);torch.set_num_threads(1);self.fp=digest(self.e.actor.state_dict());self.calls=collections.Counter()
 @torch.no_grad()
 def batch(self,t,x):
  n,L,_=x.shape;ids=torch.tensor([[AA.index(a) for a in t['sequence'][:L]]],device='cuda').expand(n,L);p=dict(x=x,ids=ids,t=torch.full((n,),L,device='cuda'),N=torch.full((n,),t['N'],device='cuda'),valid=torch.ones(n,L,dtype=torch.bool,device='cuda'));ref=self.e.prior_forward(p);self.calls['J1_batches']+=1;self.calls['J1_examples']+=n
  gt,gv=v['goal_tokens'](v['s'].goal(t),'cuda');gt=gt.expand(n,*gt.shape[1:]).clone();gv=gv.expand(n,*gv.shape[1:]);relative=[v['s'].relative(t,xx) for xx in x];gt[:,0,:3]=torch.stack([z[1] for z in relative]).float()/v['s'].CONFIG['position_scale_A'];gt[:,0,3:12]=torch.stack([z[2].flatten() for z in relative]).float();return dict(prefix=p,reference=ref,features=v['s'].features_batch(p,chemistry=True),goals=gt,goal_valid=gv,context=torch.stack([z[0] for z in relative]),subgoal=x.new_zeros(n).float())
 @torch.no_grad()
 def dist(self,t,x):
  b=self.batch(t,x);_,mu,ls=self.e.actor(b);n=len(x);a=AA.index(t['sequence'][x.shape[1]]);self.calls['actor_batches']+=1;self.calls['actor_examples']+=n;return mu[:,a],ls[:,a],b['reference']['mu'][:,a],b['reference']['ls'][:,a],b
 def verify(self):self.e.verify();assert digest(self.e.actor.state_dict())==self.fp
def eps(seed,n):return torch.randn(n,3,dtype=torch.float64,generator=torch.Generator().manual_seed(seed)).cuda()
def actions(mu,ls,noise):return v['s'].physical(mu.double()+ls.double().exp()*noise)
def anchors():return {r['task_id']:r for r in s['episodes']() if r['candidate_index']==0}
def state(t,anchor,k):
 x=v['s'].xprefix(t)
 for g in anchor['geometry'][:k]:x=torch.cat([x,v['s'].next_ca(x,torch.tensor(g,device='cuda',dtype=torch.float64))[None]])
 return x
@torch.no_grad()
def suffix_batch(e,t,x,g,k,bank,rep_ids):
 n=len(rep_ids);xx=next_batch(x[None].expand(n,-1,-1),g[None].expand(n,-1));noise=torch.stack([eps(ns('continuation',t['task_id'],t.get('condition','NATIVE'),k,bank,m),4) for m in rep_ids]);future=[];lps=[];kls=[]
 for j in range(k+1,4):
  mu,ls,rm,rl,_=e.dist(t,xx);gg=actions(mu,ls,noise[:,j]);lp=v['log_prob'](mu.double(),ls.double(),gg,torch.ones(n,3,dtype=torch.bool,device='cuda'));kl=(rl-ls+.5*((ls.exp()/rl.exp()).square()+((mu-rm)/rl.exp()).square())-.5).mean(-1);future.append(gg.cpu().tolist());lps.append(lp.cpu().tolist());kls.append(kl.cpu().tolist());xx=next_batch(xx,gg)
 result=[]
 for i,y in enumerate(xx.cpu()):
  m=v['s'].metric(t,y,0);reward,joint=v['reward'](m);soft=.995**(3-k)*reward-sum(.995**(j-k)*(.001*lps[j-k-1][i]/3+.1*kls[j-k-1][i]) for j in range(k+1,4));result.append(dict(replicate=rep_ids[i],seed=ns('continuation',t['task_id'],t.get('condition','NATIVE'),k,bank,rep_ids[i]),return_task=.995**(3-k)*reward,return_soft=soft,success=m['success'],valid=m['valid'],position_A=m['position_A'],orientation_deg=m['orientation_deg'],clashes_per100=m['clashes_per100'],adjacent_min_A=m['adjacent_min_A'],adjacent_max_A=m['adjacent_max_A'],future_geometry=[a[i] for a in future],future_logprob=[a[i] for a in lps],future_KL=[a[i] for a in kls]))
 return result
@torch.no_grad()
def generated(e,t):
 tt=copy.deepcopy(t);x=v['s'].xprefix(t);tt['t']=len(x)-4;tt['native_prefix']=x[:-4].cpu().tolist();tt['target']=x[-1].cpu().tolist();tt['orientation']=v['s'].frame(x).cpu().tolist();z=x[:-4];gs=[]
 for j in range(4):
  mu,ls,*_=e.dist(tt,z[None]);g=actions(mu,ls,eps(ns('generated_prefix',t['task_id'],j),1))[0];gs.append(g.cpu().tolist());z=torch.cat([z,v['s'].next_ca(z,g)[None]])
 out=copy.deepcopy(t);out.update(native_prefix=z.cpu().tolist(),condition='GENERATED',prefix_generation=dict(seed_convention='ns(generated_prefix,taskid,j)',geometry=gs,source_native_prefix_length=len(x)-4,intermediate_goal_position=tt['target'],intermediate_goal_orientation=tt['orientation'],policy_SHA=v['PSHA']));return out
def preflight():
 check();e=BatchEngine();t=read(R/'manifests/TASKS.json')['FIT'][0];x=v['s'].xprefix(t)
 with torch.no_grad():
  serial=e.e.distribution(t,x);batch=e.dist(t,x[None].expand(16,-1,-1));errors=[float((a-b[0]).abs().max()) for a,b in zip(serial[:4],batch[:4])];assert max(errors)<1e-4,errors
  g=actions(batch[0][:1],batch[1][:1],eps(ns('preflight'),1))[0];nb=next_batch(x[None],g[None])[0];assert float((nb[-1]-v['s'].next_ca(x,g)).abs().max())<1e-10
  start=time.monotonic();rr=suffix_batch(e,t,x,g,0,'PROFILE',list(range(16)));elapsed=time.monotonic()-start
  q=rr[0];y=nb.clone()
  for gg in q['future_geometry']:y=torch.cat([y,v['s'].next_ca(y,torch.tensor(gg,device='cuda',dtype=torch.float64))[None]])
  a=v['s'].metric(t,y,0);assert abs(.995**3*v['reward'](a)[0]-q['return_task'])<1e-9;assert bool(a['valid'])==bool(q['valid'])
 estimate=88*8*6*elapsed;assert estimate<CFG['max_estimated_collection_seconds'],estimate;e.verify();save('PREFLIGHT.json',dict(PASS=True,serial_batched_FP32_distribution_max_errors=errors,batch=16,profile_suffixes=16,profile_seconds=elapsed,estimated_collection_seconds=estimate,actor_fingerprint=e.fp,J1_tensor_SHA=e.e.prior.identity,peak_GPU=torch.cuda.max_memory_allocated(),actor_updates=0,geometry_and_CPU_GPU_reward_parity=True))
 print('PREFLIGHT',elapsed,estimate,flush=True)
def collect_panel(panel):
 check();assert read(R/'PREFLIGHT.json')['PASS'];e=BatchEngine();tasks=read(R/'manifests/TASKS.json');aa=anchors();pool=tasks[panel] if panel!='GENERATED' else [t for t in tasks['CONFIRM'] if t['task_id'] in tasks['SHIFT_IDS']];start=time.monotonic();n=0
 if panel in ['CONFIRM','GENERATED']:assert (R/'SELECTION_SEAL.json').exists()
 for ti,original in enumerate(pool):
  budget();path=R/f'results/{panel}_{ti:03}.json'
  if path.exists():
   assert sha(path)==read(path.with_suffix('.sha.json'))['sha256'];continue
  t=generated(e,original) if panel=='GENERATED' else copy.deepcopy(original);t.setdefault('condition','NATIVE');anchor=aa[t['task_id']]
  if panel=='GENERATED':
   anchor=dict(geometry=[]);z=v['s'].xprefix(t)
   for j in range(4):
    with torch.no_grad():mu,ls,*_=e.dist(t,z[None]);g=actions(mu,ls,eps(ns('shift_anchor',t['task_id'],j),1))[0];anchor['geometry'].append(g.cpu().tolist());z=torch.cat([z,v['s'].next_ca(z,g)[None]])
  records=[]
  for k in range(4):
   x=state(t,anchor,k)
   with torch.no_grad():mu,ls,rm,rl,_=e.dist(t,x[None]);noise=torch.stack([eps(ns('fixed_action',t['task_id'],t['condition'],k,a),1)[0] for a in range(8)]);gg=actions(mu,ls,noise)
   conditional_KL=float((rl-ls+.5*((ls.exp()/rl.exp()).square()+((mu-rm)/rl.exp()).square())-.5).mean())
   for a,g in enumerate(gg):
    budget();banks={}
    for bank in ['EST','ASSESS']:
     banks[bank]=suffix_batch(e,t,x,g,k,bank,list(range(1 if k==3 else 16)))
    nx=s['append'](x.cpu().numpy(),g.cpu().numpy());assert np.isfinite(nx).all();feat=s['features'](t,x.cpu().numpy(),g.cpu().numpy(),k).tolist();records.append(dict(task_id=t['task_id'],identity=t['identity'],group=t['protected_group_id'],panel=panel,stage=k,remaining_after_action=3-k,action=a,action_seed=ns('fixed_action',t['task_id'],t['condition'],k,a),geometry=g.cpu().tolist(),latent=(mu.double()[0]+ls.double().exp()[0]*noise[a]).cpu().tolist(),behavior_logprob=float(v['log_prob'](mu.double()[0],ls.double()[0],g,torch.ones(3,dtype=torch.bool,device='cuda'))),conditional_KL=conditional_KL,features=feat,banks=banks))
  save(str(path.relative_to(R)),dict(task=t,anchor_geometry=anchor['geometry'],records=records));save(str(path.with_suffix('.sha.json').relative_to(R)),dict(sha256=sha(path),complete=True));n+=len(records);save('LIVE_STATUS.json',dict(stage=panel,completed_task=ti+1,tasks=len(pool),state_action_labels=n,pid=os.getpid(),free_bytes=shutil.disk_usage(R).free));print('COLLECT',panel,ti+1,len(pool),flush=True)
 e.verify();save(f'{panel}_COMPLETED.json',dict(complete=True,tasks=len(pool),state_action_pairs=len(pool)*32,elapsed_seconds=time.monotonic()-start,network_calls=dict(e.calls),peak_GPU=torch.cuda.max_memory_allocated(),actor_fingerprint=e.fp,J1_tensor_SHA=e.e.prior.identity,actor_updates=0))
def records(panel):return [r for p in sorted((R/'results').glob(panel+'_[0-9][0-9][0-9].json')) for r in read(p)['records']]
def mean(r,bank='EST',field='return_task'):return float(np.mean([q[field] for q in r['banks'][bank]]))
def group_states(rows):
 out=collections.defaultdict(list)
 for r in rows:out[(r['task_id'],r['stage'])].append(r)
 return [sorted(a,key=lambda r:r['action']) for a in out.values()]
def contrast(a,b,bank='EST'):
 d=np.array([q['return_task'] for q in b['banks'][bank]])-np.array([q['return_task'] for q in a['banks'][bank]]);m=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else 0;return m,se
def support(a,b,bank='EST'):
 m,se=contrast(a,b,bank);return abs(m)>=.1 and abs(m)>1.96*se
def simple_models(rows,checkpoint=None):
 model=torch.load(B/'results/simple_models.pt',map_location='cpu',weights_only=False);X=np.array([r['features'] for r in rows]);xx=(X-model['mean'])/model['sd'];m=s['MLP'](58);m.load_state_dict(checkpoint or model['MLP']);m.eval()
 with torch.no_grad():p=m(torch.tensor(xx,dtype=torch.float32)).numpy()
 return p
def heuristic_raw(r):
 f=r['features'];good=not (f[35] or f[36]);return float(math.exp(-.5*(f[13]**2+(f[14]*180/35)**2))) if good else -1.
def fit_models():
 check();torch.set_num_threads(1);torch.manual_seed(87272);fit=records('FIT');dev=records('DEV');assert len(fit)==1024 and len(dev)==384;states=[rr for rr in group_states(fit) if rr[0]['stage']<3];flags=[any(support(rr[i],rr[j]) for i in range(8) for j in range(i)) for rr in states];ids={rr[0]['identity'] for rr,flag in zip(states,flags) if flag};admit=np.mean(flags)>=.25 and len(ids)>=12
 old=torch.load(B/'results/simple_models.pt',map_location='cpu',weights_only=False);cal=[]
 for k in range(4):
  subset=[r for r in fit if r['stage']==k];x=np.array([heuristic_raw(r) for r in subset]);y=np.array([mean(r) for r in subset]);slope=max(0,float(np.cov(x,y,ddof=0)[0,1]/max(np.var(x),1e-12)));cal.append([float(y.mean()-slope*x.mean()),slope])
 gate=dict(admitted=bool(admit),informative_states=sum(flags),states=len(states),proteins=len(ids),fraction=float(np.mean(flags)),scope='FIT EST labels only; normalapprox nominal pairwise evidence, not groundtruth');save('FIT_IDENTIFIABILITY_GATE.json',gate);save('HEURISTIC_CALIBRATION.json',dict(stage_intercept_slope=cal,fit_only=True,selection='raw joint residual with immediate clash/spacing rejection; no future outcomes'))
 curves=[];best=None;bestval=-float('inf');path=R/'checkpoints/small_recalibration.pt'
 if admit:
  m=s['MLP'](58);m.load_state_dict(old['MLP']);opt=torch.optim.Adam(m.parameters(),lr=.0003);X=torch.tensor((np.array([r['features'] for r in fit])-old['mean'])/old['sd'],dtype=torch.float32);y=torch.tensor([mean(r) for r in fit],dtype=torch.float32);rng=np.random.default_rng(87272);start=0;beststate=None
  if path.exists():
   st=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(st['model']);opt.load_state_dict(st['optimizer']);set_rng(st['rng']);rng.bit_generator.state=st['sampler_rng'];start=st['update'];bestval=st['best_value'];beststate=st['best_model'];curves=st['curves']
  else:
   initial=copy.deepcopy(m.state_dict());initialopt=copy.deepcopy(opt.state_dict());initialrng=copy.deepcopy(rng.bit_generator.state);indices=rng.choice(len(X),64);opt.zero_grad();loss=(m(X[indices])-y[indices]).square().mean();loss.backward();opt.step();sig=digest(dict(model=m.state_dict(),optimizer=opt.state_dict()));m.load_state_dict(initial);opt.load_state_dict(initialopt);rng.bit_generator.state=initialrng;indices=rng.choice(len(X),64);opt.zero_grad();loss=(m(X[indices])-y[indices]).square().mean();loss.backward();opt.step();assert sig==digest(dict(model=m.state_dict(),optimizer=opt.state_dict()));m.load_state_dict(initial);opt.load_state_dict(initialopt);rng.bit_generator.state=initialrng;save('MLP_REPLAY_TEST.json',dict(PASS=True,scope='sameprocess exact deterministic CPU next model+optimizer update; two engineering executions reset',actor_updates=0))
  for u in range(start,1000):
   budget();ii=rng.choice(len(X),64);opt.zero_grad();loss=(m(X[ii])-y[ii]).square().mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
   if (u+1) in [100,250,500,1000]:
    p=simple_models(dev,m.state_dict());selected=[mean(rr[int(np.argmax([p[dev.index(r)] for r in rr]))]) for rr in group_states(dev) if rr[0]['stage']<3];value=float(np.mean(selected));curves.append(dict(update=u+1,DEV_EST_selected_return=value,train_loss=float(loss),gradient_norm=float(norm)))
    if value>bestval:bestval=value;beststate=copy.deepcopy(m.state_dict())
    st=dict(model=m.state_dict(),optimizer=opt.state_dict(),update=u+1,rng=rng_state(),sampler_rng=rng.bit_generator.state,best_model=beststate,best_value=bestval,curves=curves,protocol=read(R/'PROTOCOL.json'),source=source(),replay_SHA={str(p):sha(p) for panel in ['FIT','DEV'] for p in (R/'results').glob(panel+'_[0-9][0-9][0-9].json')});rec=atomic_checkpoint(path,st,POLICY);save('checkpoints/INDEX.json',rec)
  best=beststate
 save('SMALL_MODEL_CALIBRATION.json',dict(executed=bool(admit),updates=1000 if admit else 0,curves=curves,selected_DEV_value=bestval if admit else None,parameters=5889,actor_updates=0))
 predictions={'MLP':simple_models(dev)}
 if best is not None:predictions['E']=simple_models(dev,best)
 values={name:float(np.mean([mean(rr[int(np.argmax([p[dev.index(r)] for r in rr]))]) for rr in group_states(dev) if rr[0]['stage']<3])) for name,p in predictions.items()};primary=max(values,key=values.get);save('SELECTION_SEAL.json',dict(primary=primary,DEV_EST_values=values,UTC=time.time(),CONFIRM_not_collected=True,ASSESS_never_used_for_fitting_or_selection=True,gate=gate))
 print('SMALL_MODEL',gate,primary,values,flush=True)
def predict():
 check();e=BatchEngine();ck=torch.load(B/'checkpoints/critic_0250_stop.pt',map_location='cpu',weights_only=False);qs=[s['LowQ'](s['PROFILES']['research']['low_q']).cuda().eval().requires_grad_(False) for _ in range(2)]
 for q,w in zip(qs,ck['critics']):q.load_state_dict(w,strict=True)
 assert digest(ck['actor'])==e.fp;del ck
 recal=torch.load(R/'checkpoints/small_recalibration.pt',map_location='cpu',weights_only=False)['best_model'] if read(R/'SMALL_MODEL_CALIBRATION.json')['executed'] else None
 out={}
 for panel in ['FIT','DEV','CONFIRM','GENERATED']:
  rr=records(panel);pred=dict(MLP=simple_models(rr).tolist());pred['E']=simple_models(rr,recal).tolist() if recal else None;values={};tasks={read(p)['task']['task_id']:read(p) for p in (R/'results').glob(panel+'_[0-9][0-9][0-9].json')}
  for rows in group_states(rr):
   budget();item=tasks[rows[0]['task_id']];t=item['task'];x=state(t,item['anchor_geometry'],rows[0]['stage']);batch=e.batch(t,x[None].expand(8,-1,-1));a=torch.full((8,),AA.index(t['sequence'][len(x)]),device='cuda');g=torch.tensor([r['geometry'] for r in rows],device='cuda',dtype=torch.float64);mask=torch.ones(8,3,dtype=torch.bool,device='cuda')
   with torch.no_grad():qvalues=torch.stack([q(batch,a,g,mask) for q in qs],-1).float().cpu().tolist()
   for r,z in zip(rows,qvalues):values[(r['task_id'],r['stage'],r['action'])]=z
  for j in range(2):pred['critic'+str(j+1)]=[values[(r['task_id'],r['stage'],r['action'])][j] for r in rr]
  out[panel]=dict(keys=[[r['task_id'],r['stage'],r['action']] for r in rr],predictions=pred);print('PREDICT',panel,len(rr),flush=True)
 e.verify();save('PREDICTIONS.json',out);save('PREDICTION_COMPLETED.json',dict(complete=True,network_calls=dict(e.calls),peak_GPU=torch.cuda.max_memory_allocated(),actor_updates=0))
def campaign():
 lock=(R/'runtime/campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 stages=['preflight','collect_FIT','collect_DEV','fit_models','collect_CONFIRM','collect_GENERATED','predict','analyze']
 done={'preflight':'PREFLIGHT.json','collect_FIT':'FIT_COMPLETED.json','collect_DEV':'DEV_COMPLETED.json','fit_models':'SELECTION_SEAL.json','collect_CONFIRM':'CONFIRM_COMPLETED.json','collect_GENERATED':'GENERATED_COMPLETED.json','predict':'PREDICTION_COMPLETED.json','analyze':'PROVENANCE_SHA256.json'}
 for stage in stages:
  if (R/done[stage]).exists():continue
  gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not gpu.strip(),gpu
  with (R/f'logs/{stage}_{time.time_ns()}.log').open('x') as f:
   child=subprocess.Popen([sys.executable,'-B',str(R/'source/scripts/e017_v72/study.py'),stage],stdout=f,stderr=subprocess.STDOUT);save('EXECUTION.json',dict(status='RUNNING',stage=stage,pid=child.pid,supervisor_pid=os.getpid(),automatic_restart=False,UTC=time.time()));rc=child.wait()
  if rc:save('EXECUTION.json',dict(status='STOPPED',failed_stage=stage,exit_code=rc,automatic_restart=False));raise RuntimeError((stage,rc))
 save('EXECUTION.json',dict(status='COMPLETED',automatic_restart=False,actor_updates=0,UTC=time.time()));save('CAMPAIGN_COMPLETED.json',dict(complete=True,actor_updates=0,UTC=time.time()))
if __name__=='__main__':
 signal.signal(signal.SIGTERM,handle);signal.signal(signal.SIGINT,handle);cmd=sys.argv[1]
 if cmd.startswith('collect_'):collect_panel(cmd[8:])
 elif cmd=='analyze':runpy.run_path(str(R/'source/scripts/e017_v72/analyze.py'),run_name='__main__')
 else:globals()[cmd]()
