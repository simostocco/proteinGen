"""Frozen H4 planning with exact terminal constraints; never optimizes a network."""
import collections,copy,fcntl,hashlib,json,math,os,random,resource,runpy,shutil,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
R=Path('/mnt/d/Simone/proteinGen_data/e017_v8_constrained_planning_20261023_v1');BASE=R.parent/'e017_v72_counterfactual_ranking_20261023_v1';V6=R.parent/'e017_v6_frozen_parent_candidates_20261023_v1';V7=R.parent/'e017_v7_amortized_lowlevel_20261023_v1'
b=runpy.run_path(str(R/'source/scripts/e017_v72/study.py'));v=b['v'];s=v['s'];AA=b['AA'];sha=b['sha'];digest=b['digest'];read=b['read']
from protein_joint_sac.crisp_v3.recovery import atomic_json,StoragePolicy
CFG=dict(schema='E017_V8_v1',seed=8817,H=4,TRAIN_identities=8,VALIDATION_identities=32,goals_per_identity=2,conditions=['NATIVE','GENERATED'],K=[8,16,32,64],neural_precision='FP32',geometry_precision='FP64',temperature=1.,batch=8,
 baseline='V6 validity-first selector: max(position/8,orientation/35), original Huber+5sin2 cost, index; iid parent H4 pool nested8/16/32/64; cache identical initial-state distribution once',
 beam='width2, at most4child prefixes/stage (K8 uses2); q=K/4terminal lookaheads perstage; terminalstage qactions; sampled completion max exactvalid potential; retain all evaluated terminals; diverse second prefix if visible-tail RMSD>=.1A; no future pose observed',
 prune='irreversible spacing/nonfinite/degenerate; existing clashes*100/finalN>2; finalRg lowerbound>originalhigh; necessary contour bound to goal with original8Atolerance. Never currentRg-low/high or current-denominator clash heuristic',
 CEM='initial8parent trajectories reused fromA; population8, elites2valid-only, blend.7, diagonal absolute-action latent Gaussian, std floor.05*mean initial conditional actor std; total8/16/32/64candidates; no clamp/projection; no valid elite=>retain previous search law',
 IK='trace variant of original V5 IK equations; two fixed starts; K/2 evaluated iterates/start, hence K/2-1Adam steps/start; lr.08 clip10 original loss/parameterization; no neural gradients; objective and exact assessment share FK',
 prior_audit='allCEM candidates and all selectedbeam/IK paths: parent conditional logdensity, latent standardized radius, parent-toJ1 conditionalKL; CEM unselected Gaussian-generation logq/logp MC diagnostic. Audit cost separate and also added to endtoend cost; selected search-law KL not equated to proposal KL',
 primary='each planner vsA atK64: point full-success gain>=.10 and paired protein98.333333%lower>0 in BOTH native and generated; family3 planners, intersection-union acrossconditions. Otherwise exploratory or no established improvement; never neural policy promotion.',
 compute='equal maximum terminal candidate caps, not equalFLOPs; include every lookahead/discard/gradient FK and exact eval; baseline comparisons additionally at nearest non-exceeding actual transition and actor-example counts; runtime reported separately, no unmeasured equal-cost claims',
 no_training=True,TEST=False,reserve_bytes=2*2**30,max_GPU_bytes=6*2**30,max_estimated_seconds=7200)
POLICY=StoragePolicy(min_headroom_bytes=2*2**30,near_term_growth_bytes=128*2**20,checkpoint_hint_bytes=8*2**20);STOP=False
def save(n,x):atomic_json(R/n,x)
def ns(*x):return int(hashlib.sha256(('E017V8|8817|'+ '|'.join(map(str,x))).encode()).hexdigest()[:15],16)
def source():return subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip()
def budget():
 POLICY.require(R,8*2**20)
 if STOP:raise InterruptedError('cooperative complete-block stop')
 if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()>6*2**30:raise RuntimeError('GPU6GiBgate')
def handle(*_):
 global STOP
 STOP=True
def clean(t):
 out=copy.deepcopy(t)
 for k in ['native_actions','hidden_native_future','witness_actions','witness_valid','witness_max_error_A']:out.pop(k,None)
 return out
def prepare():
 from protein_joint_sac.data.scaled_supervised import Data,canonical,internal
 from protein_joint_sac.data.accepted_supervised import CoordinateReader
 from protein_joint_sac.crisp_v3.execution import set_loader
 torch.set_num_threads(1);set_loader('arrow_scalar_lru4');data=Data(False);reader=CoordinateReader(data.proposal.external);excluded=set();exsources={}
 for root in sorted(R.parent.glob('e017*')):
  if root==R:continue
  for p in sorted((root/'manifests').glob('*.json')):
   if not any(z in p.stem.upper() for z in ['VALIDATION','DEV','CONFIRM','AUDIT']):continue
   d=read(p);ts=d.get('tasks',[]) if isinstance(d,dict) else [];used=set()
   for t in ts:
    h=t.get('identity');g=t.get('protected_group_id',t.get('group'))
    if h in data.proposal.sequences and data.proposal.sequences[h]['split']=='VALIDATION':used.add(g or data.proposal.sequences[h]['protected_group_id'])
   if used:excluded.update(used);exsources[str(p)]=dict(SHA256=sha(p),groups=len(used))
 candidates=sorted(data.jointids['VALIDATION'],key=lambda h:ns('newVAL',h));tasks=[];witnesses=[];reject=[];groups=set(excluded);start=time.monotonic()
 for h in candidates:
  if len(tasks)==64:break
  g=data.proposal.sequences[h]['protected_group_id']
  if g in groups or len(data.seqs[h])<24:continue
  o=min(data.owners[h]);row=data.rows[o]
  try:
   assert row['split']=='VALIDATION' and row['sequence_hash']==h;xx=canonical(reader.read(row));gg=internal(xx);seq=data.seqs[h];assert len(xx)==len(seq);x=torch.tensor(xx[:24],dtype=torch.float64)
   t=dict(identity=h,protected_group_id=g,split='VALIDATION',observation=o,source_sha=row['source_sha'],entry_id=row['entry_id'],coordinate_path=row['coordinate_path'],coordinate_row=row['coordinate_row'],N=len(seq),t=20,H=4,sequence=seq,native_prefix=xx[:20].tolist(),target=xx[23].tolist(),orientation=s.frame(x).tolist(),variant='native',task_id=h+'_20_4_native',goal_provenance='observed native endpoint/frame, valid deposited witness',action_masks=[[True]*3]*4)
   assert s.metric(t,x,0)['success'];native=gg[20:24].copy();rebuilt=s.fk(t,torch.tensor(native,dtype=torch.float64));assert float((rebuilt-x).abs().max())<1e-8
   alt=copy.deepcopy(t);a=native.copy();a[0,2]=(a[0,2]+.8+math.pi)%(2*math.pi)-math.pi;ax=s.fk(t,torch.tensor(a,dtype=torch.float64));alt.update(target=ax[-1].tolist(),orientation=s.frame(ax).tolist(),variant='kinematic_alt',task_id=h+'_20_4_alt',goal_provenance='synthetic kinematic control, native first torsion+.8rad; not experimental structure')
   if not s.metric(alt,ax,0)['success']:raise ValueError('paired synthetic witness invalid')
   tasks.extend([t,alt]);groups.add(g);witnesses.extend([dict(task_id=t['task_id'],actions=native.tolist()),dict(task_id=alt['task_id'],actions=a.tolist())]);print('PREPARED',len(tasks)//2,32,flush=True)
  except (AssertionError,ValueError,FloatingPointError) as ex:reject.append(dict(identity=h,observation=o,reason=repr(ex)))
 assert len(tasks)==64;trainraw=read(V7/'manifests/TRAIN.json')['tasks'];chosen=sorted(trainraw,key=lambda t:ns('TRAIN',t['identity']))[:8];train=[]
 for original in chosen:
  t=clean(original);t['goal_provenance']='existing TRAIN native witness';train.append(t);a=np.array(original['native_actions'],dtype=np.float64);a[0,2]=(a[0,2]+.8+math.pi)%(2*math.pi)-math.pi;x=s.fk(t,torch.tensor(a));alt=copy.deepcopy(t);alt.update(target=x[-1].tolist(),orientation=s.frame(x).tolist(),variant='kinematic_alt',task_id=t['identity']+'_20_4_alt',goal_provenance='TRAIN synthetic kinematic goal')
  if s.metric(alt,x,0)['success']:train.append(alt)
 assert not {t['protected_group_id'] for t in train}&{t['protected_group_id'] for t in tasks};save('manifests/TRAIN.json',dict(tasks=train));save('manifests/VALIDATION.json',dict(tasks=tasks));save('manifests/WITNESS_PROVENANCE.json',dict(records=witnesses,workers_must_not_read=True));save('manifests/REJECTIONS.json',reject);save('TASK_CENSUS.json',dict(TRAIN_tasks=len(train),TRAIN_identities=8,VALIDATION_tasks=64,VALIDATION_identities=32,excluded_previous_E017_groups=len(excluded),excluded_manifest_sources=exsources,eligible_VALIDATION_joint_population=len(data.jointids['VALIDATION']),previous_context='new for E017 planning-method selection; pretrained J1 and dataset history are not claimed to be globally unexposed',TEST_accessed=False,seconds=time.monotonic()-start))
 paths=[BASE/'FINAL_E017_V72_SCIENTIFIC_REPORT.md',V6/'FINAL_E017_V6_SCIENTIFIC_REPORT.md',V7/'FINAL_E017_V7_SCIENTIFIC_REPORT.md',v['PARENT'],R.parent/'e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt',R.parent/'pdb_supervised_split_50id80cov_20261016_v1/PROPOSAL_MANIFEST.json',Path(s.__file__),Path(s.v.__file__),V6/'source/scripts/e017_v6/study.py',V6/'source/scripts/e017_v6/analyze.py',R.parent/'e017_crisp_v3_20261021_v1/configs/NATIVE_CALIBRATION.json']+list((R/'manifests').glob('*.json'))
 assert sha(v['PARENT'])==v['PSHA'];save('INPUT_AUDIT.json',dict(input_SHA256={str(p):sha(p) for p in paths},actor_SHA256=v['PSHA'],J1_SHA256='881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412',actor_parameters=17438114,J1_parameters=12351746,D_free_bytes=shutil.disk_usage(R).free));save('DESIGN.json',CFG)
def seal():
 assert not (R/'PROTOCOL.json').exists();inputs=read(R/'INPUT_AUDIT.json')['input_SHA256'];inputs.update({str(p):sha(p) for p in (R/'source/scripts/e017_v8').glob('*.py')})
 for p in [V7/'checkpoints/B_00025.pt',V7/'checkpoints/C_00000_stop.pt',R.parent/'e017_v71_expected_return_calibration_20261023_v1/checkpoints/critic_0250_stop.pt',Path(s.v.PARENT)]:inputs[str(p)]=sha(p)
 save('PROTOCOL.json',dict(config=CFG,source=source(),input_SHA256=inputs,UTC=time.time(),freeze_before_prospective_planner_results=True))
def check():
 p=read(R/'PROTOCOL.json');assert p['source']==source() and p['config']==CFG
 for f,h in p['input_SHA256'].items():assert sha(f)==h,f
def epsilon(seed,n):return torch.randn(n,3,dtype=torch.float64,generator=torch.Generator().manual_seed(seed)).cuda()
class Engine(b['BatchEngine']):
 def __init__(self):super().__init__();self.count=collections.Counter();self.actor_seconds=0.;self.GPU_seconds=0.
 @torch.no_grad()
 def dist(self,t,x):
  torch.cuda.synchronize();beg=time.perf_counter();first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True);first.record();z=super().dist(t,x);last.record();last.synchronize();self.GPU_seconds+=first.elapsed_time(last)/1000;self.actor_seconds+=time.perf_counter()-beg;self.count['actor_examples']+=len(x);self.count['actor_batches']+=1;return z
def fk(t,actions,device='cpu'):
 x=torch.tensor(t['native_prefix'],dtype=torch.float64,device=device)
 for g in actions:x=torch.cat([x,s.next_ca(x,g.to(device))[None]])
 return x
def prune(t,x,goal=True):
 a=x.detach().cpu().numpy();N=t['t']+4;d=np.linalg.norm(np.diff(a,axis=0),axis=-1);reasons=[]
 if not np.isfinite(a).all():return ['nonfinite']
 if d.min()<s.v.LIMITS['adjacent_low_A'] or d.max()>s.v.LIMITS['adjacent_high_A']:reasons.append('irreversible_spacing')
 if (np.linalg.norm(np.cross(np.diff(a,axis=0)[1:],np.diff(a,axis=0)[:-1]),axis=-1)<=1e-8).any():reasons.append('irreversible_degenerate_frame')
 ix=np.arange(len(a));clashes=int(((np.linalg.norm(a[:,None]-a[None,:],axis=-1)<2)&(ix[:,None]-ix[None,:]>=3)).sum())
 if clashes*100/N>s.v.LIMITS['clashes_per100_max']:reasons.append('irreversible_nonlocal_clashes')
 rg_lower=np.sqrt(np.square(a-a.mean(0)).sum()/N)/N**(1/3)
 if rg_lower>s.v.LIMITS['Rg_normalized_high']:reasons.append('irreversible_Rg_upper_bound')
 if goal and np.linalg.norm(a[-1]-t['target'])>(N-len(a))*s.v.LIMITS['adjacent_high_A']+8:reasons.append('necessary_goal_contour_bound')
 return reasons
def evaluate(t,g,index,coordinates=None):
 a=torch.tensor(g,dtype=torch.float64);x=fk(t,a) if coordinates is None else coordinates.detach().cpu();m=s.metric(t,x,0);delta=np.abs((x[-1].numpy()-np.array(t['target']))/3.8);cost=float(np.where(delta<1,.5*delta**2,delta-.5).sum()+5*math.sin(math.radians(m['orientation_deg'])/2)**2);fail=[];d=torch.linalg.vector_norm(x[1:]-x[:-1],dim=-1)
 if d.min()<s.v.LIMITS['adjacent_low_A'] or d.max()>s.v.LIMITS['adjacent_high_A']:fail.append('spacing')
 if m['clashes_per100']>s.v.LIMITS['clashes_per100_max']:fail.append('nonlocal_clashes')
 if not s.v.LIMITS['Rg_normalized_low']<=m['Rg_normalized']<=s.v.LIMITS['Rg_normalized_high']:fail.append('Rg')
 return dict(index=index,geometry=a.tolist(),latent=s.inverse_latent(a).tolist(),valid=m['valid'],success=m['success'],translation_hit=m['translation_hit'],rotation_hit=m['rotation_hit'],position_A=m['position_A'],orientation_deg=m['orientation_deg'],goal_score=max(m['position_A']/8,m['orientation_deg']/35),original_goal_cost=cost,clashes_per100=m['clashes_per100'],Rg_normalized=m['Rg_normalized'],validity_failures=fail)
def select(rows):
 valid=[r for r in rows if r['valid']];best=min(valid,key=lambda r:(r['goal_score'],r['original_goal_cost'],r['index'])) if valid else None;assert bool(best and best['success'])==any(r['success'] for r in rows);return best
def stamp(e,start,c0,transitions,terminals,**extra):return dict(wall_seconds=time.perf_counter()-start,cpu_process_seconds=time.process_time()-c0[4],actor_examples=e.count['actor_examples']-c0[0],actor_batches=e.count['actor_batches']-c0[1],neural_GPU_seconds=e.GPU_seconds-c0[3],synchronized_neural_call_wall_seconds=e.actor_seconds-c0[2],simulated_transitions=transitions,terminal_evaluations=terminals,**extra)
def counters(e):return e.count['actor_examples'],e.count['actor_batches'],e.actor_seconds,e.GPU_seconds,time.process_time()
@torch.no_grad()
def complete(e,t,prefix,gs,seed):
 x=prefix.clone();gg=copy.deepcopy(gs);noise=epsilon(seed,4)
 for k in range(len(gs),4):
  mu,ls,*_=e.dist(t,x[None]);g=b['actions'](mu,ls,noise[k:k+1])[0];gg.append(g.cpu().tolist());x=torch.cat([x,s.next_ca(x,g)[None]])
 return gg,x
@torch.no_grad()
def baseline(e,t,K=64):
 start=time.perf_counter();c0=counters(e);rows=[];snap={};x0=s.xprefix(t);mu0,ls0,*_=e.dist(t,x0[None]);trans=0
 # Eight-candidate chunks give every snapshot a measured standalone-like
 # warm-start latency while sharing only the identical initial distribution.
 for offset in range(0,K,8):
  seeds=[ns('A',t['task_id'],t.get('condition','NATIVE'),i) for i in range(offset,offset+8)];noise=torch.stack([epsilon(seed,4) for seed in seeds]);x=x0[None].expand(8,-1,-1);gs=[];us=[];means=[];logstd=[]
  for k in range(4):
   mu,ls=(mu0.expand(8,-1),ls0.expand(8,-1)) if k==0 else e.dist(t,x)[:2];u=mu.double()+ls.double().exp()*noise[:,k];g=s.physical(u);x=b['next_batch'](x,g);gs.append(g.cpu());us.append(u.cpu());means.append(mu.cpu());logstd.append(ls.cpu());trans+=8
  for j in range(8):
   g=[a[j].tolist() for a in gs];r=evaluate(t,g,len(rows),coordinates=x[j]);r.update(seed=seeds[j],conditional_mu=[a[j].tolist() for a in means],conditional_ls=[a[j].tolist() for a in logstd]);rows.append(r)
  if len(rows) in CFG['K']:snap[str(len(rows))]=dict(selected=select(rows)['index'] if select(rows) else None,cost=stamp(e,start,c0,trans,len(rows)))
 return dict(candidates=rows,snapshots=snap)
@torch.no_grad()
def beam(e,t,K,pruning=True):
 start=time.perf_counter();c0=counters(e);trans=0;rows=[];rejections=[];partial=[dict(geometry=[],path='root',x=s.xprefix(t))];q=K//4
 if pruning and prune(t,partial[0]['x']):return dict(candidates=[],selected=None,cost=stamp(e,start,c0,0,0,unused_terminal_budget=K),rejections=[dict(stage=-1,path='root',reasons=prune(t,partial[0]['x']))])
 for stage in range(4):
  if not partial:break
  count=q if stage==3 else min(q,4);children=[];dists=[e.dist(t,p['x'][None])[:2] for p in partial]
  if stage==3:
   # Every complete extension is an evaluated terminal candidate, even if
   # invalid or outside the target. No free terminal pruning/refill.
   for attempt in range(q):
    pi=attempt%len(partial);p=partial[pi];mu,ls=dists[pi];seed=ns('beam_action',t['task_id'],t.get('condition','NATIVE'),stage,p['path'],attempt);g=b['actions'](mu,ls,epsilon(seed,1))[0];x=torch.cat([p['x'],s.next_ca(p['x'],g)[None]]);trans+=1;r=evaluate(t,p['geometry']+[g.cpu().tolist()],len(rows),coordinates=x);r.update(stage=3,child_path=p['path']+'/'+str(attempt),completion_seed=seed);rows.append(r)
   break
  for attempt in range(count*2):
   if len(children)>=count:break
   pi=attempt%len(partial);p=partial[pi];mu,ls=dists[pi];seed=ns('beam_action',t['task_id'],t.get('condition','NATIVE'),stage,p['path'],attempt);g=b['actions'](mu,ls,epsilon(seed,1))[0];x=torch.cat([p['x'],s.next_ca(p['x'],g)[None]]);trans+=1;reason=prune(t,x) if pruning else []
   if reason:rejections.append(dict(stage=stage,path=p['path'],seed=seed,geometry=g.cpu().tolist(),reasons=reason));continue
   children.append(dict(geometry=p['geometry']+[g.cpu().tolist()],path=p['path']+'/'+str(attempt),x=x,results=[]))
  if not children:break
  for j in range(q):
   child=children[j%len(children)];rep=j//len(children);seed=ns('beam_complete',t['task_id'],t.get('condition','NATIVE'),stage,child['path'],rep);g,y=complete(e,t,child['x'],child['geometry'],seed) if stage<3 else (child['geometry'],child['x']);trans+=3-stage;r=evaluate(t,g,len(rows),coordinates=y);r.update(stage=stage,child_path=child['path'],completion_seed=seed);rows.append(r);child['results'].append(r)
  if stage==3:break
  def key(p):
   winner=select(p['results']);fallback=min(p['results'],key=lambda r:(r['goal_score'],r['index']));return (winner is None,(winner or fallback)['goal_score'],p['path'])
  children=[p for p in children if p['results']];children.sort(key=key);partial=children[:1]
  if len(children)>1:
   first=children[0];other=next((p for p in children[1:] if float(torch.sqrt(((p['x'][t['t']:]-first['x'][t['t']:])**2).sum(-1).mean()))>=.1),children[1]);partial.append(other)
 selected=select(rows);return dict(candidates=rows,selected=selected['index'] if selected else None,cost=stamp(e,start,c0,trans,len(rows),unused_terminal_budget=K-len(rows)),rejections=rejections)
def cem(e,t,A):
 start=time.perf_counter();c0=counters(e);rows=copy.deepcopy(A['candidates'][:8]);snap={'8':dict(selected=select(rows)['index'] if select(rows) else None,cost=copy.deepcopy(A['snapshots']['8']['cost']))};u=np.array([r['latent'] for r in rows]);mu=u.mean(0);sd=u.std(0).clip(1e-6);floor=.05*np.exp(np.array([r['conditional_ls'] for r in rows])).mean(0);trans=32
 for gen in range(1,8):
  eligible=sorted([r for r in rows[-8:] if r['valid']],key=lambda r:(r['goal_score'],r['original_goal_cost'],r['index']))[:2]
  if eligible:
   elite=np.array([r['latent'] for r in eligible]);mu=.7*elite.mean(0)+.3*mu;sd=np.maximum(.7*elite.std(0)+.3*sd,floor)
  generator=np.random.default_rng(ns('CEM',t['task_id'],t.get('condition','NATIVE'),gen));draws=mu+sd*generator.standard_normal((8,4,3))
  for z in draws:
   g=s.physical(torch.tensor(z,dtype=torch.float64));r=evaluate(t,g.tolist(),len(rows));r.update(generation=gen,search_logq_latent=float((-.5*((z-mu)/sd)**2-np.log(sd)-.5*math.log(2*math.pi)).sum()),search_mean=mu.tolist(),search_sd=sd.tolist());rows.append(r);trans+=4
  if len(rows) in CFG['K']:
   cost=stamp(e,start,c0,trans,len(rows))
   for field in ['wall_seconds','cpu_process_seconds','neural_GPU_seconds','synchronized_neural_call_wall_seconds','actor_examples','actor_batches']:cost[field]+=A['snapshots']['8']['cost'][field]
   snap[str(len(rows))]=dict(selected=select(rows)['index'] if select(rows) else None,cost=cost)
 return dict(candidates=rows,snapshots=snap,initial8_reused_from_A=True)
def ik(t,K):
 start=time.perf_counter();cpu_start=time.process_time();rows=[];losses=[];grad=0;trans=0
 for starti in range(2):
  z=torch.zeros(4,3,dtype=torch.float64,requires_grad=True)
  with torch.no_grad():z[:,1]=.4;z[:,2]=.5 if starti==0 else -.5
  opt=torch.optim.Adam([z],lr=.08)
  for j in range(K//2):
   opt.zero_grad();g=torch.stack([s.v.LIMITS['adjacent_low_A']+.01+(s.v.LIMITS['adjacent_high_A']-s.v.LIMITS['adjacent_low_A']-.02)*z[:,0].sigmoid(),.03+(math.pi-.06)*z[:,1].sigmoid(),math.pi*z[:,2].tanh()],-1);x=fk(t,g);trans+=4;pos=F.huber_loss((x[-1]-x.new_tensor(t['target']))/3.8,torch.zeros(3,dtype=torch.float64),reduction='sum');rot=(3-(s.frame(x)*x.new_tensor(t['orientation'])).sum())/4;clash=s.Engine.local(None,x,t);loss=pos+s.CONFIG['IK_orientation_weight']*rot+2*clash+((g[:,0]-3.8)/.1).square().mean()+.001*z.square().mean()
   # Exact validity and gradient objective share the already computed FK.
   r=evaluate(t,g.detach().tolist(),len(rows),coordinates=x);r.update(start=starti,iterate=j);rows.append(r);losses.append(float(loss.detach()))
   if j+1<K//2:loss.backward();torch.nn.utils.clip_grad_norm_([z],10,error_if_nonfinite=True);opt.step();grad+=1
 selected=select(rows);return dict(candidates=rows,selected=selected['index'] if selected else None,cost=dict(wall_seconds=time.perf_counter()-start,cpu_process_seconds=time.process_time()-cpu_start,actor_examples=0,actor_batches=0,neural_GPU_seconds=0.,synchronized_neural_call_wall_seconds=0.,simulated_transitions=trans,terminal_evaluations=len(rows),gradient_objectives=K,latent_gradient_steps=grad,extra_differentiable_FK_transitions=0),loss_trace=losses)
@torch.no_grad()
def audit_paths(e,t,rows):
 start=time.perf_counter();c0=counters(e)
 for offset in range(0,len(rows),8):
  rr=rows[offset:offset+8];gs=torch.tensor([r['geometry'] for r in rr],device='cuda',dtype=torch.float64);x=s.xprefix(t)[None].expand(len(rr),-1,-1);logp=torch.zeros(len(rr),device='cuda',dtype=torch.float64);kl=torch.zeros_like(logp);zz=[]
  for k in range(4):
   mu,ls,rm,rl,_=e.dist(t,x);u=s.inverse_latent(gs[:,k]);z=(u-mu.double())/ls.double().exp();zz.append(z.cpu().numpy());logp+=(-.5*z.square()-ls.double()-.5*math.log(2*math.pi)).sum(-1);kl+=(rl-ls+.5*((ls.exp()/rl.exp()).square()+((mu-rm)/rl.exp()).square())-.5).mean(-1)/4;x=b['next_batch'](x,gs[:,k])
  for j,r in enumerate(rr):
   z=np.array([a[j] for a in zz]);value=dict(parent_logp_latent=float(logp[j]),mean_standardized_squared_latent=float(np.mean(z*z)),max_abs_standardized_latent=float(abs(z).max()),conditional_proposal_KL=float(kl[j]))
   r['prior_audit']=value
 return stamp(e,start,c0,4*len(rows),0,audited_paths=len(rows),scope='diagnostic overhead, separate from search')
def prior_audit(e,t,algorithms):
 total=collections.Counter()
 for name,result in algorithms.items():
  if name=='CEM':
   cumulative=collections.Counter()
   for offset in range(0,64,8):
    c=audit_paths(e,t,result['candidates'][offset:offset+8]);cumulative.update({k:z for k,z in c.items() if isinstance(z,(int,float))})
    if str(offset+8) in result['snapshots']:result['snapshots'][str(offset+8)]['prior_diagnostic_cost']=dict(cumulative)
   total.update(cumulative)
  elif name=='A':
   for snap in result['snapshots'].values():
    rr=[result['candidates'][snap['selected']]] if snap['selected'] is not None else [];c=audit_paths(e,t,rr);snap['prior_diagnostic_cost']=c;total.update({k:z for k,z in c.items() if isinstance(z,(int,float))})
  else:
   rr=[result['candidates'][result['selected']]] if result['selected'] is not None else [];c=audit_paths(e,t,rr);result['prior_diagnostic_cost']=c;total.update({k:z for k,z in c.items() if isinstance(z,(int,float))})
 return dict(total)
@torch.no_grad()
def generated(e,t):
 tt=copy.deepcopy(t);x=s.xprefix(t);tt['t']=len(x)-4;tt['native_prefix']=x[:-4].cpu().tolist();tt['target']=x[-1].cpu().tolist();tt['orientation']=s.frame(x).cpu().tolist();z=x[:-4];gs=[]
 for j in range(4):
  mu,ls,*_=e.dist(tt,z[None]);g=b['actions'](mu,ls,epsilon(ns('GENERATED_PREFIX',t['identity'],j),1))[0];gs.append(g.cpu().tolist());z=torch.cat([z,s.next_ca(z,g)[None]])
 out=copy.deepcopy(t);out.update(native_prefix=z.cpu().tolist(),condition='GENERATED',prefix_generation=dict(geometry=gs,initial_native_length=len(x)-4,intermediate_goal_position=tt['target'],intermediate_goal_orientation=tt['orientation'],seed_identity=t['identity']));return out
def solve(e,t):
 result={};result['A']=baseline(e,t);result['CEM']=cem(e,t,result['A'])
 for K in CFG['K']:budget();result['BEAM_'+str(K)]=beam(e,t,K);result['IK_'+str(K)]=ik(t,K)
 result['PRIOR_AUDIT']=prior_audit(e,t,result);return result
def preflight():
 check();e=Engine();torch.set_num_threads(1);t=read(R/'manifests/TRAIN.json')['tasks'][0];start=time.perf_counter();x=s.xprefix(t)
 with torch.no_grad():
  a=e.e.distribution(t,x);bb=e.dist(t,x[None].expand(8,-1,-1));errors=[float((u-z[0]).abs().max()) for u,z in zip(a[:4],bb[:4])];assert max(errors)<1e-4
 aa=baseline(e,t,8);r=aa['candidates'][0];noise=epsilon(r['seed'],4);xx=x.clone();serial=[]
 with torch.no_grad():
  for k in range(4):mu,ls,*_=e.e.distribution(t,xx);g=b['actions'](mu[None],ls[None],noise[k:k+1])[0];serial.append(g.cpu().tolist());xx=torch.cat([xx,s.next_ca(xx,g)[None]])
 assert np.max(abs(np.array(serial)-np.array(r['geometry'])))<1e-5;assert s.metric(t,xx,0)['success']==r['success']
 iktrace=ik(t,8);legacy=s.IK(t,iterations=3,starts=2)
 for i in range(2):assert np.max(abs(np.array(iktrace['candidates'][i*4+3]['geometry'])-np.array(legacy['attempts'][i]['actions'])))<1e-9
 bb=beam(e,t,8);assert len(bb['candidates'])<=8;assert all(not r['success'] or r['valid'] for r in bb['candidates']);pg=generated(e,t);assert pg['target']==t['target'];assert not {'native_actions','hidden_native_future','witness_actions'}&set(t)
 # All randomly completed full chains containing a pruned prefix must fail
 # terminal success. This supplements the algebraic proof, not replaces it.
 tested=0
 for r in aa['candidates']:
  xx=s.xprefix(t)
  for g in r['geometry']:
   xx=torch.cat([xx,s.next_ca(xx,torch.tensor(g,device='cuda',dtype=torch.float64))[None]])
   if prune(t,xx):assert not r['success'];tested+=1
 e.verify();save('PREFLIGHT.json',dict(PASS=True,serial_batched_FP32_errors=errors,V6_mathematical_law_sampling_and_selector=True,historical_BF16_numerics_not_claimed_byte_identical=True,legacy_IK_equation_and_endpoint_parity=True,IK_legacy_iterations=3,prune_crosschecks=tested,hidden_native_geometry_absent=True,actor_fingerprint=e.fp,J1_tensor_SHA=e.e.prior.identity,seconds=time.perf_counter()-start,peak_GPU=torch.cuda.max_memory_allocated()))
def run_panel(panel):
 check();assert read(R/'PREFLIGHT.json')['PASS'];e=Engine();torch.set_num_threads(1);tasks=read(R/('manifests/'+('TRAIN' if panel=='TRAIN' else 'VALIDATION')+'.json'))['tasks'];start=time.perf_counter();cpu_start=time.process_time();done=0
 if panel=='TRAIN':tasks=tasks[:4]
 prefix_cache={}
 for i,raw in enumerate(tasks):
  for condition in (['NATIVE'] if panel=='TRAIN' else CFG['conditions']):
   budget();path=R/f'results/{panel}_{i:03}_{condition}.json'
   if path.exists():
    assert sha(path)==read(path.with_suffix('.sha.json'))['sha256']
    if condition=='GENERATED':prefix_cache[raw['identity']]=read(path)['task']
    done+=1;continue
   prefix_cost_start=time.perf_counter();prefix_c0=counters(e)
   if condition=='GENERATED':
    if raw['identity'] not in prefix_cache:prefix_cache[raw['identity']]=generated(e,raw)
    t=copy.deepcopy(raw);t['native_prefix']=prefix_cache[raw['identity']]['native_prefix'];t['prefix_generation']=prefix_cache[raw['identity']]['prefix_generation']
   else:t=copy.deepcopy(raw)
   prefix_cost=stamp(e,prefix_cost_start,prefix_c0,4 if e.count['actor_examples']>prefix_c0[0] else 0,0)
   t['condition']=condition;result=solve(e,t);save(str(path.relative_to(R)),dict(task=t,prefix_generation_cost=prefix_cost,algorithms=result,root_irreparable_or_unreachable=prune(t,s.xprefix(t)),source=source(),protocol_SHA256=sha(R/'PROTOCOL.json')));save(str(path.with_suffix('.sha.json').relative_to(R)),dict(sha256=sha(path),complete=True));done+=1;save('LIVE_STATUS.json',dict(stage=panel,completed_condition_tasks=done,total_condition_tasks=len(tasks)*(1 if panel=='TRAIN' else 2),pid=os.getpid(),D_free_bytes=shutil.disk_usage(R).free));print('PLANNED',panel,done,len(tasks)*(1 if panel=='TRAIN' else 2),flush=True)
 e.verify();save(panel+'_COMPLETED.json',dict(complete=True,condition_tasks=done,elapsed_seconds=time.perf_counter()-start,cpu_process_seconds=time.process_time()-cpu_start,actor_examples=e.count['actor_examples'],actor_batches=e.count['actor_batches'],neural_GPU_seconds=e.GPU_seconds,synchronized_neural_call_wall_seconds=e.actor_seconds,peak_GPU=torch.cuda.max_memory_allocated(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,actor_fingerprint=e.fp,J1_tensor_SHA=e.e.prior.identity,no_neural_updates=True))
def campaign():
 lock=(R/'runtime/campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for stage,marker in [('preflight','PREFLIGHT.json'),('run_TRAIN','TRAIN_COMPLETED.json'),('run_VALIDATION','VALIDATION_COMPLETED.json'),('analyze','PROVENANCE_SHA256.json')]:
  if (R/marker).exists():continue
  if stage=='run_VALIDATION':
   measured=read(R/'TRAIN_COMPLETED.json');forecast=measured['elapsed_seconds']/measured['condition_tasks']*128;size=sum(p.stat().st_size for p in (R/'results').glob('TRAIN_[0-9][0-9][0-9]_NATIVE.json'))/measured['condition_tasks']*128;save('VALIDATION_RESOURCE_ADMISSION.json',dict(measured_TRAIN_seconds=measured['elapsed_seconds'],TRAIN_condition_tasks=measured['condition_tasks'],forecast_validation_seconds=forecast,forecast_result_bytes=size,maximum_seconds=CFG['max_estimated_seconds'],D_free_bytes=shutil.disk_usage(R).free));assert forecast<=CFG['max_estimated_seconds'];POLICY.require(R,int(size)+8*2**20)
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
  with (R/f'logs/{stage}_{time.time_ns()}.log').open('x') as f:
   child=subprocess.Popen([sys.executable,'-B',str(R/'source/scripts/e017_v8/study.py'),stage],stdout=f,stderr=subprocess.STDOUT);save('EXECUTION.json',dict(status='RUNNING',stage=stage,pid=child.pid,supervisor_pid=os.getpid(),automatic_restart=False,UTC=time.time()));rc=child.wait()
  if rc:save('EXECUTION.json',dict(status='STOPPED',failed_stage=stage,exit_code=rc,automatic_restart=False));raise RuntimeError((stage,rc))
 save('EXECUTION.json',dict(status='COMPLETED',automatic_restart=False,UTC=time.time()));save('CAMPAIGN_COMPLETED.json',dict(complete=True,no_neural_updates=True,UTC=time.time()))
if __name__=='__main__':
 signal.signal(signal.SIGTERM,handle);signal.signal(signal.SIGINT,handle);cmd=sys.argv[1]
 if cmd.startswith('run_'):run_panel(cmd[4:])
 elif cmd=='analyze':runpy.run_path(str(R/'source/scripts/e017_v8/analyze.py'),run_name='__main__')
 else:globals()[cmd]()
