"""Bounded admission tests. ALL optimizer.step methods are forbidden here."""
from pathlib import Path
import ast,copy,io,json,math,os,random,resource,shutil,subprocess,time
from collections import Counter
import numpy as np
import torch
from protein_joint_sac.crisp_v3.data import Population,CoverageSampler
from protein_joint_sac.crisp_v3.admission import *
from protein_joint_sac.crisp_v3.networks import System
from protein_joint_sac.crisp_v3.prior import FrozenPrior,SHA,file_sha,tensor_sha
from protein_joint_sac.crisp_v3.optimization import Learner
from protein_joint_sac.crisp_v3.recovery import StoragePolicy,RollingRecovery,serialized_size,state_digest
from protein_joint_sac.crisp_v3.storage import Replay,payload,restore,rng_state,set_rng
from protein_joint_sac.crisp_v3.training import seal_config,authorization
R=Path(__file__).resolve().parents[3];start=time.monotonic();config=json.load(open(R/'configs/SEALED_RESEARCH_INITIALIZATION.json'));configsha=seal_config(config)
# Engineering controls must never invoke a model optimizer.
def forbidden(*a,**k):raise AssertionError('Optimizer updates are forbidden during admission preflight')
for cls in [torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD]:cls.step=forbidden
for p in [Path(__file__),R/'source/src/protein_joint_sac/crisp_v3/admission.py',R/'source/src/protein_joint_sac/crisp_v3/training.py']:ast.parse(p.read_text())
try:authorization(None,config);raise AssertionError('missing authorization accepted')
except PermissionError:pass
bad=copy.deepcopy(config);bad['updates']=10001
try:seal_config(bad);raise AssertionError('ceiling not enforced')
except ValueError:pass
p=Population();panel=json.load(open(config['validation_panel']));train=json.load(open(config['matched_TRAIN_panel']));checks=verify_panel(p,panel,'VALIDATION');checks['cells']={'/'.join(k):v for k,v in checks['cells'].items()}
wrong=copy.deepcopy(panel);wrong['records'][0]['split']='TRAIN'
try:verify_panel(p,wrong,'VALIDATION');raise AssertionError('forged split accepted')
except ValueError:pass
# Metadata-only full 10k AA sampling simulation: no coordinate loading or model work.
sampler=CoverageSampler(p,config['seed']);examples=[];coverage={};milestones=set(config['validation_milestones']);counts=Counter();seen_positions=set();repeat_targets=0;sampling_start=time.monotonic()
for u in range(10000):
 for category,h,t,rng in sampler.draw(64):
  counts[category]+=1;key=(h,t)
  if key in seen_positions:repeat_targets+=1
  seen_positions.add(key)
 if u+1 in milestones:coverage[u+1]=dict(categories=dict(counts),distinct_identities={k:len(v) for k,v in sampler.seen.items()},unique_identity_position_targets=len(seen_positions),repeated_identity_position_targets=repeat_targets)
 # No validation labels or geometry are sampled here.
sampling_seconds=time.monotonic()-sampling_start
# This sampler simulation is explicitly discarded; future training always starts seed1703 fresh.
del seen_positions,sampler
limits=json.load(open(config['native_limits']));sampler=CoverageSampler(p,1703);geom=CoverageSampler(p,1704);goals=InitGoals(p,1705,limits)
records=sampler.draw(320);assert Counter(v[0] for v in records)==dict(complete=128,partial=96,aa_only=96)
ss=copy.deepcopy(sampler.state_dict());draw1=[(c,h,t) for c,h,t,r in sampler.draw(20)];sampler.load_state_dict(ss);assert draw1==[(c,h,t) for c,h,t,r in sampler.draw(20)]
gtasks=[goals.draw() for _ in range(6)];gs=copy.deepcopy(goals.state_dict());task1=goals.draw();goals.load_state_dict(gs);assert task1==goals.draw();assert all(t['split']=='TRAIN' for t in gtasks);assert len(goals.seen)==7
# Validate masks and prefix ownership on three real examples, before GPU inference.
mask_counts={}
for cat in CATEGORIES:
 r=next(r for r in panel['records'] if r['category']==cat and r['length_band']=='64-96');pp,aa,g,mask=load_decision(p,r,r['positions'][1],'cpu');assert pp['t'][0]==r['positions'][1];assert pp['ids'].shape[-1]==r['positions'][1]
 if cat=='aa_only':assert not pp['valid'].any() and not mask.any()
 mask_counts[cat]=mask.sum(0).tolist()
# Serialize the exact research network and prospective ACTOR-only Adam schema to a counting stream.
torch.manual_seed(1703);system=System('research');learner=Learner(system,**config['optimizer']);scope=configure_initialization(system,learner,config['schedule']);assert scope['active_parameters']==17438114 and list(learner.optimizers)==['actor'];assert system.counts()['total_trainable']==63389056
for o in learner.optimizers.values():
 for group in o.param_groups:
  for q in group['params']:o.state[q]={'step':torch.tensor(0.),'exp_avg':torch.zeros_like(q),'exp_avg_sq':torch.zeros_like(q)}
state=payload(system,learner,dict(AA=sampler,geometry=geom,goals=goals),config,None,dict(updates=0,monitor=None));actual=serialized_size(state);sampler_bytes=serialized_size(state['samplers'])
# Conservative exact all-identity coverage scenario uses actual accepted polymer lengths.
full=copy.deepcopy(state['samplers'])
for name in ['AA','geometry']:
 s=full[name];ids=[h for cat,v in s['orders'].items() if name=='AA' or cat=='complete' for h in v];s['position']={h:list(range(len(p.base.seqs[h]))) for h in ids};s['cycles']={h:0 for h in ids};s['seen']={k:set(v) if name=='AA' or k=='complete' else set() for k,v in s['orders'].items()}
full['goals']['seen']=set(p.groups['TRAIN']['complete']);state['samplers']=full;full_bytes=serialized_size(state)
counts_all=system.counts();del state,full,learner
# Research GPU forward only, with no gradient creation or updates.
assert torch.cuda.is_available();torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats();device='cuda';system=system.to(device);prior=FrozenPrior(device);before=prior.verify([]);frozen_before=tensor_sha(system);measurements=[]
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 for hi,label in [(96,'64-96'),(128,'97-128'),(256,'129-256'),(384,'257-384'),(500,'385-500')]:
  r=next(r for r in panel['records'] if r['category']=='complete' and r['length_band']==label);t=r['N']-1;pp,aa,g,mask=load_decision(p,r,t,device);gt,gv=goal_tokens(native_task(p,r['identity'],r['observation'],'P2','VALIDATION'),device)
  elapsed=[]
  for j in range(3):
   torch.cuda.synchronize();a=time.monotonic();b=prepare(pp,prior(pp),gt,gv);loss,terms=actor_loss(system,b,aa,g,mask,config['stream_weights']['goal'],True);torch.cuda.synchronize();elapsed.append(time.monotonic()-a);assert torch.isfinite(loss)
  measurements.append(dict(N=r['N'],prefix=t,seconds=elapsed,median_warm_forward_seconds=float(np.median(elapsed[1:])),valid_components=mask.sum().item()))
 # True geometry-free input, placeholder invariance, token visibility and no geometry loss.
 r=next(r for r in panel['records'] if r['category']=='aa_only' and r['length_band']=='64-96');pp,aa,g,mask=load_decision(p,r,30,device,True);zero=torch.zeros(1,1,36,device=device);vv=torch.zeros(1,1,device=device,dtype=torch.bool);b=prepare(pp,prior(pp),zero,vv);l,mu,ls=system.models['actor'](b);assert torch.isfinite(l).all();assert not pp['valid'].any();assert torch.equal(l,b['reference']['logits'])
 pp2=copy.deepcopy(pp);pp2['x'].fill_(12345.);bb=prepare(pp2,prior(pp2),zero,vv);ll,_,_=system.models['actor'](bb);assert torch.equal(l,ll)
 loss,terms=actor_loss(system,b,aa,g,mask,config['stream_weights']['AA']);assert float(terms['geometry'])==0. and float(terms['kl_geometry'])==0.
 # Bounded full validation runner exercises native + missing geometry + goals; NOT a quality panel evaluation.
 small=copy.deepcopy(panel);small['records']=[next(r for r in panel['records'] if r['category']==cat and r['length_band']=='64-96') for cat in CATEGORIES];selected={r['identity'] for r in small['records']};small['tasks']=[t for t in panel['tasks'] if t['identity'] in selected];small['generation']=[]
 for row in small['records']:row['positions']=row['positions'][:1]
 for row in small['tasks']:row['positions']=row['positions'][:1]
 fixture_eval=evaluate(system,prior,p,small,device,config)
assert frozen_before==tensor_sha(system);prior.verify([]);assert all(q.grad is None for q in system.parameters()) and all(q.grad is None for q in prior.model.parameters())
GPU=dict(forward_measurements=measurements,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),J1_bitwise_unchanged=True,all_system_tensors_unchanged=True,all_gradients_absent=True,geometry_free_finite=True,missing_coordinate_placeholder_invariance=True,AA_only_geometry_loss_zero=True,fixture_native_and_goal_decisions=len(fixture_eval['records'])+len(fixture_eval['tasks']))

with open(R/'results/FORWARD_AND_STORAGE_PREFLIGHT.json','w') as f:json.dump(dict(GPU=GPU,scope=scope,counts=counts_all,actual_checkpoint_bytes=actual,full_coverage_checkpoint_bytes=full_bytes,metadata_coverage=coverage,sampling_seconds=sampling_seconds,optimizer_updates=0),f,indent=2)
del system;torch.cuda.empty_cache()
# Complete-state atomic readback on a SMALL engineering system, not a scientific checkpoint.
small_system=System('development').cuda();small_learner=Learner(small_system);configure_initialization(small_system,small_learner,config['schedule']);smallconfig={**config,'profile':'development','fixture_only':True};samplers=dict(AA=sampler,geometry=geom,goals=goals)
for o in small_learner.optimizers.values():
 for group in o.param_groups:
  for q in group['params']:o.state[q]={'step':torch.tensor(0.),'exp_avg':torch.zeros_like(q),'exp_avg_sq':torch.zeros_like(q)}
fixture=R/'results/RECOVERY_FIXTURE_ONLY';manager=RollingRecovery(fixture,StoragePolicy(**config['storage_policy']),create=True);replay=Replay(fixture/'replay.sqlite');before_rng=rng_state();counters=dict(updates=0,fixture_only=True);saved=manager.publish(replay,lambda snap:payload(small_system,small_learner,samplers,smallconfig,snap,counters),0);path=manager.current()['latest']['checkpoint'];loaded=torch.load(path,map_location='cpu',weights_only=False);snapshot=loaded['replay_snapshot'];assert file_sha(snapshot['path'])==snapshot['sha256']
expected=tensor_sha(small_system);expected_sample=copy.deepcopy(sampler.state_dict());expected_goal=copy.deepcopy(goals.state_dict());random.random();np.random.rand();torch.rand(2,device='cuda');sampler.draw(3);goals.draw()
restore(path,small_system,small_learner,samplers,smallconfig);assert expected==tensor_sha(small_system);assert state_digest(sampler.state_dict())==state_digest(expected_sample);assert state_digest(goals.state_dict())==state_digest(expected_goal);assert state_digest(rng_state())==state_digest(loaded['rng']);assert list(small_learner.optimizers)==['actor'];replay.close();prior.verify(small_learner.optimizers.values())
# Test prospective stop monitor and paired-bootstrap mechanics on synthetic measurements only.
def ev(aa=2.,goal=1.):return dict(summary=dict(native=dict(aa=aa,distance=0.,angle=0.,torsion=0.),geometry_free=dict(aa=aa),goal=dict(loss=goal)))
monitor=Monitor(config['early_stop']);assert not monitor.observe(0,ev(),ev(),[])['stop'];assert not monitor.observe(250,ev(),ev(2.1),[])['stop'];assert monitor.observe(1000,ev(),ev(2.1),[])['stop'];mon2=Monitor(config['early_stop'],monitor.state);assert mon2.state==monitor.state
pi=paired_intervals(fixture_eval,fixture_eval);assert all(v['delta_NLL']==0 and v['identity_bootstrap_95pct']==[0.,0.] for v in pi.values())
# Verify immutable source bases and original published artifacts/checkpoints.
old=R.parent/'e017_crisp_v3_20261021_v1/storage_policy_20261021_v1';manifest=json.load(open(old/'PROVENANCE_SHA256.json'));protected=[]
for rel,sha in manifest['new_source_files'].items():assert file_sha(old/'source'/rel)==sha;protected.append(str(old/'source'/rel))
for rel,sha in manifest['new_artifacts'].items():assert file_sha(old/rel)==sha;protected.append(str(old/rel))
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=old/'source',text=True).strip()=='4374c4a766a088da4daf0b475a26357408876a8c'
assert file_sha(prior.path if hasattr(prior,'path') else R.parent/'e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt')==SHA
free=shutil.disk_usage(R).free;headroom=2*2**30;peak=3*full_bytes+64*2**20+headroom
result=dict(schema='crisp_zero_update_admission_preflight_v1',config_sha256=configsha,model_counts=counts_all,initialization_scope=scope,actor_only_optimizer=True,no_critic_or_planner_optimizer=True,full_scope_actor_Adam_checkpoint_bytes=actual,full_identity_coverage_actor_only_checkpoint_bytes=full_bytes,sampler_current_bytes=sampler_bytes,projected_latest_best_staging_bytes=3*full_bytes,conservative_admission_bytes=peak,free_bytes=free,storage_feasible=free>=peak,metadata_sampler_simulation=dict(optimizer_updates=0,seconds=sampling_seconds,milestones=coverage,simulation_discarded=True),panel_checks=checks,TRAIN_goal_sampler_resume=True,AA_sampler_resume=True,complete_fixture_checkpoint=dict(path=path,sha256=file_sha(path),bytes=Path(path).stat().st_size,all_networks_targets_actor_optimizer_scheduler_samplers_and_RNG_exact=True,replay_snapshot_sha256=snapshot['sha256'],synthetic_fixture_not_training_progress=True),GPU=GPU,prospective_stop_tests=True,bootstrap_tests=True,protected_storage_release_files_verified=len(protected),source_base_commit='4374c4a766a088da4daf0b475a26357408876a8c',optimizer_updates=0,J1_updates=0,full_validation_scoring_performed=False,free_generation_performed=False,seconds=time.monotonic()-start,host_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
with open(R/'results/BOUNDED_ADMISSION_PREFLIGHT.json','x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:result[k] for k in ['initialization_scope','full_scope_actor_Adam_checkpoint_bytes','full_identity_coverage_actor_only_checkpoint_bytes','free_bytes','storage_feasible','GPU','seconds','host_peak_RSS_bytes']},indent=2),flush=True)
