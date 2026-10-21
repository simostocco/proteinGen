"""Cloned-state complete-update benchmark; no scientific campaign progress."""
from pathlib import Path
import copy,json,random,time,math,os,resource,statistics,subprocess,contextlib,cProfile,pstats,io,ast
from collections import defaultdict
from unittest.mock import patch
import numpy as np
import torch
from protein_joint_sac.crisp_v3 import admission as A,optimization as O,pipeline as P,networks as N
from protein_joint_sac.crisp_v3.data import Population,CoverageSampler
from protein_joint_sac.crisp_v3.prior import FrozenPrior,tensor_sha,file_sha
from protein_joint_sac.crisp_v3.storage import restore,payload,rng_state,set_rng
from protein_joint_sac.crisp_v3.execution import set_loader
from protein_joint_sac.crisp_v3.recovery import state_digest,StoragePolicy,RollingRecovery
R=Path(__file__).resolve().parents[3]
# Do not overlap GPU optimization with the original worker, even if it is stopped.
try:
 st=Path('/proc/178643/stat').read_text().rsplit(') ',1)[1].split()[0]
 if st!='Z':raise RuntimeError('Original worker still exists; do not benchmark')
except FileNotFoundError:pass
receipt=json.load(open(R/'results/VERIFIED_OPERATOR_PAUSE_EXIT.json'));parent_path=receipt['checkpoint'];assert file_sha(parent_path)==receipt['checkpoint_sha256'];parent=torch.load(parent_path,map_location='cpu',weights_only=False);config=parent['config'];assert parent['counters']['updates']==250
assert not parent['counters'].get('scientific_stop',{}).get('stop')
torch.use_deterministic_algorithms(True);torch.manual_seed(1703);prior=FrozenPrior('cuda');system=N.System('research').cuda();learner=O.Learner(system,**config['optimizer']);A.configure_initialization(system,learner,config['schedule']);pop=Population();samplers=dict(AA=CoverageSampler(pop),geometry=CoverageSampler(pop),goals=A.InitGoals(pop,1705,json.load(open(config['native_limits']))))
load_seconds=0
# Cache ordinary immutable raw source hashes across trials (startup hashes do not belong to steady throughput).
def reset():
 global load_seconds
 t=time.monotonic();restore(parent_path,system,learner,samplers,config);system.train();
 for k,m in system.models.items():
  if k!='actor':m.eval()
 pop.cache.clear();pop.reader.group.cache_clear();pop.reader.small.cache_clear();torch.cuda.synchronize();load_seconds+=time.monotonic()-t
reset();inactive={k:tensor_sha(m) for k,m in system.models.items() if k!='actor'}|{'targets':tensor_sha(system.targets)}
class Measurements:
 def __init__(self):self.wall=defaultdict(float);self.calls=defaultdict(int);self.prefixes=[];self.current_band='unknown';self.band_times=defaultdict(lambda:defaultdict(float));self.band_calls=defaultdict(lambda:defaultdict(int))
 def wrapper(self,name,fn):
  def run(*a,**k):
   try:
    prefix=a[1]['prefix'] if name=='actor_forward_and_loss' else a[1] if name=='frozen_J1_inference' else a[0] if name=='feature_preprocessing' else None
    if prefix is not None:self.current_band=A.band(max(20,prefix['ids'].shape[1]))
   except (KeyError,IndexError,TypeError):pass
   t=time.monotonic()
   try:return fn(*a,**k)
   finally:
    elapsed=time.monotonic()-t;self.wall[name]+=elapsed;self.calls[name]+=1
    self.band_times[self.current_band][name]+=elapsed;self.band_calls[self.current_band][name]+=1
  return run
 def contexts(self):
  stack=contextlib.ExitStack()
  stack.enter_context(patch.object(pop,'coordinates',self.wrapper('coordinate_read_decode',pop.coordinates)))
  stack.enter_context(patch.object(pop,'batch',self.wrapper('data_batch_total',pop.batch)))
  stack.enter_context(patch.object(A,'prepare',self.wrapper('feature_preprocessing',A.prepare)))
  stack.enter_context(patch.object(P,'prepare',self.wrapper('feature_preprocessing',P.prepare)))
  stack.enter_context(patch.object(FrozenPrior,'__call__',self.wrapper('frozen_J1_inference',FrozenPrior.__call__)))
  stack.enter_context(patch.object(A,'actor_loss',self.wrapper('actor_forward_and_loss',A.actor_loss)))
  stack.enter_context(patch.object(torch.Tensor,'backward',self.wrapper('actor_backward',torch.Tensor.backward)))
  stack.enter_context(patch.object(O,'finish',self.wrapper('clip_optimizer_zero_grad',O.finish)))
  native_goal=P.native_goal_batch
  def goal_observe(*args,**kwargs):
   t0=time.monotonic();out=native_goal(*args,**kwargs);self.wall['goal_load_reference_features_total']+=time.monotonic()-t0;self.calls['goal_load_reference_features_total']+=1;self.prefixes.append(dict(category='goal',N=int(out[0]['prefix']['N'][0]),t=int(out[0]['prefix']['t'][0])));return out
  stack.enter_context(patch.object(P,'native_goal_batch',goal_observe))
  decision=pop.decision
  def observe(*args,**kwargs):
   value=decision(*args,**kwargs);self.prefixes.append(dict(category=args[0],N=len(pop.base.seqs[args[1]]),t=args[2]));return value
  stack.enter_context(patch.object(pop,'decision',observe));return stack
 def report(self):
  by=defaultdict(lambda:dict(decisions=0,prefix_residues=0))
  for p in self.prefixes:
   q=by[A.band(p['N'])];q['decisions']+=1;q['prefix_residues']+=p['t']
  return dict(inclusive_wall_seconds=dict(self.wall),calls=dict(self.calls),length_bands=dict(by),prefix_residues=sum(p['t'] for p in self.prefixes),per_padded_prefix_band=dict(self.band_times),per_padded_prefix_band_calls=dict(self.band_calls))

def update(micro,measure=False,capture=False):
 c={**config,'microbatch':micro};captured={};original_finish=O.finish
 def finish(opt,parameters,clip=1.):
  ps=list(parameters)
  if capture:captured['raw_gradient']=torch.cat([p.grad.flatten().float() if p.grad is not None else torch.zeros_like(p).flatten().float() for p in ps]).cpu()
  return original_finish(opt,ps,clip)
 t=time.monotonic();torch.cuda.reset_peak_memory_stats();M=Measurements()
 with M.contexts() if measure else contextlib.nullcontext():
  with patch.object(O,'finish',finish) if capture else contextlib.nullcontext():
   with torch.autocast('cuda',dtype=torch.bfloat16):out=A.initialization_update(system,learner,prior,pop,samplers['AA'],samplers['geometry'],samplers['goals'],c)
 torch.cuda.synchronize();seconds=time.monotonic()-t
 prior.verify(learner.optimizers.values());assert inactive==({k:tensor_sha(m) for k,m in system.models.items() if k!='actor'}|{'targets':tensor_sha(system.targets)})
 assert all(math.isfinite(float(out[k])) for k in ['loss','gradient_norm']);assert learner.steps['actor']>250
 if capture:
  captured['parameters']=torch.cat([p.detach().flatten() for p in system.models['actor'].parameters()]).cpu();captured['sampler_digest']=state_digest({k:v.state_dict() for k,v in samplers.items()});captured['rng_digest']=state_digest(rng_state());captured['optimizer_digest']=state_digest(learner.optimizers['actor'].state_dict());captured['scheduler_digest']=state_digest(learner.schedulers['actor'].state_dict())
 return dict(microbatch=micro,seconds=seconds,loss=out['loss'],terms=out['terms'],gradient_groups=out['gradient_groups'],gradient_norm=out['gradient_norm'],AA_categories=out['AA_categories'],valid_geometry_components=out['valid_geometry_components'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),host_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,measurement=M.report() if measure else None),captured

# Verify the revised reduction is EXACTLY the sealed per-decision objective.
oldsource=Path('/mnt/d/Simone/proteinGen_data/e017_crisp_v3_training_admission_20261021_v1/source/src/protein_joint_sac/crisp_v3/admission.py').read_text();node=next(n for n in ast.parse(oldsource).body if isinstance(n,ast.FunctionDef) and n.name=='actor_loss');namespace=A.__dict__.copy();exec(ast.get_source_segment(oldsource,node),namespace);old_loss=namespace['actor_loss']
reset()
with patch.object(A,'actor_loss',old_loss):old_row,old_capture=update(1,capture=True)
reset();new_row,new_capture=update(1,capture=True)
sealed_one_equivalence=dict(loss_exact=old_row['loss']==new_row['loss'],raw_gradients_exact=torch.equal(old_capture['raw_gradient'],new_capture['raw_gradient']),parameters_exact=torch.equal(old_capture['parameters'],new_capture['parameters']),samplers_exact=old_capture['sampler_digest']==new_capture['sampler_digest'])
assert all(sealed_one_equivalence.values()),sealed_one_equivalence
del old_capture,new_capture
with open(R/'results/SEALED_MICROBATCH1_EQUIVALENCE.json','x') as f:json.dump(sealed_one_equivalence,f,indent=2)
# First numerical trials from EXACT same restored full state, including observation RNG.
rows=[];captures={}
for micro in [1,2,4]:
 reset();row,cap=update(micro,measure=True,capture=True);rows.append(row);captures[micro]=cap
with open(R/'results/INITIAL_NUMERICAL_TRIALS.json','x') as f:json.dump(dict(trials=rows),f,indent=2)
base=rows[0];comparisons=[]
for i,m in enumerate([2,4],1):
 a=captures[1]['raw_gradient'];b=captures[m]['raw_gradient'];rel=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a));cos=float(F.cosine_similarity(a,b,dim=0)) if False else float(torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)));pd=float((captures[1]['parameters']-captures[m]['parameters']).abs().max());ld=abs(base['loss']-rows[i]['loss']);same=captures[1]['sampler_digest']==captures[m]['sampler_digest'];rngsame=captures[1]['rng_digest']==captures[m]['rng_digest'];passed=rel<=.03 and cos>=.999 and pd<=1e-5 and ld<=max(.003,.001*abs(base['loss'])) and same and rngsame
 comparisons.append(dict(microbatch=m,raw_gradient_relative_L2=rel,raw_gradient_cosine=cos,maximum_parameter_difference=pd,loss_absolute_difference=ld,sampler_state_exact=same,RNG_state_exact=rngsame,pass_numerical=passed))
# Repeat the same selected configuration exactly; recovery equivalence is stricter than cross-microbatch agreement.
reset();again,cap=update(4,capture=True);restart=dict(microbatch=4,parameters_bitwise_equal=torch.equal(captures[4]['parameters'],cap['parameters']),raw_gradients_bitwise_equal=torch.equal(captures[4]['raw_gradient'],cap['raw_gradient']),optimizer_state_exact=captures[4]['optimizer_digest']==cap['optimizer_digest'],scheduler_state_exact=captures[4]['scheduler_digest']==cap['scheduler_digest'],samplers_exact=captures[4]['sampler_digest']==cap['sampler_digest'],RNG_exact=captures[4]['rng_digest']==cap['rng_digest'])
with open(R/'results/NUMERICAL_AND_RESTART_TESTS.json','x') as f:json.dump(dict(comparisons=comparisons,restart=restart,tolerances=dict(gradient_relative_L2=.03,gradient_cosine_min=.999,parameter_max_abs=1e-5,loss_max_absolute='.003 or .001*baseline absolute, whichever larger'),interpretation='Cross-microbatch BF16 ordering tolerance; within microbatch restart must be exact. No benchmark update is unique scientific training progress.'),f,indent=2)
del captures,cap;torch.cuda.empty_cache()
# Fair counterbalanced complete-step throughput with fresh coordinate caches, same source checks.
timings=[]
for order in [[1,2,4],[4,2,1]]:
 for micro in order:
  reset()
  with patch.object(A,'actor_loss',old_loss) if micro==1 else contextlib.nullcontext():row,_=update(micro,measure=True)
  timings.append(row)
  with open(R/'results/COMPLETE_STEP_BENCHMARK_PARTIAL.json','w') as f:json.dump(dict(trials=timings),f,indent=2)
summary={str(m):dict(median_seconds=statistics.median([v['seconds'] for v in timings if v['microbatch']==m]),trials=[v for v in timings if v['microbatch']==m]) for m in [1,2,4]}
for m in summary:summary[m]['speedup_fraction']=1-summary[m]['median_seconds']/summary['1']['median_seconds']
# The CPU profile motivates a bounded lazy Arrow scalar-access comparison.
set_loader('arrow_scalar_lru4');reset();lazy_row,lazy_cap=update(1,capture=True)
# Recreate the eager baseline on the same snapshot for strict representation equivalence.
set_loader('eager');reset();eager_row,eager_cap=update(1,capture=True)
loader_equivalence=dict(raw_gradients_exact=torch.equal(lazy_cap['raw_gradient'],eager_cap['raw_gradient']),parameters_exact=torch.equal(lazy_cap['parameters'],eager_cap['parameters']),samplers_exact=lazy_cap['sampler_digest']==eager_cap['sampler_digest'],RNG_exact=lazy_cap['rng_digest']==eager_cap['rng_digest'],loss_exact=lazy_row['loss']==eager_row['loss'])
assert all(loader_equivalence.values()),loader_equivalence
del lazy_cap,eager_cap
loader_timings=[]
for order in [[1,4],[4,1]]:
 for micro in order:
  set_loader('arrow_scalar_lru4');reset();row,_=update(micro,measure=True);loader_timings.append(row)
loader_summary={str(m):dict(median_seconds=statistics.median([v['seconds'] for v in loader_timings if v['microbatch']==m]),trials=[v for v in loader_timings if v['microbatch']==m]) for m in [1,4]}
for m in loader_summary:loader_summary[m]['speedup_fraction_vs_original']=1-loader_summary[m]['median_seconds']/summary['1']['median_seconds']
with open(R/'results/LAZY_LOADER_COMPARISON.json','x') as f:json.dump(dict(equivalence=loader_equivalence,summary=loader_summary,bounded_rowgroup_cache=4,no_persistent_coordinate_copy=True),f,indent=2)
set_loader('eager')
# Optional profiler is one additional cloned baseline step; excluded from promotion timing.
reset();profile_error=None
try:
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False) as prof:update(1)
 (R/'results/TORCH_PROFILE_TABLE.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=40))
 stats=[dict(key=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,cpu_total_us=e.cpu_time_total,self_device_us=getattr(e,'self_device_time_total',None)) for e in prof.key_averages()]
 with open(R/'results/TORCH_PROFILE_EVENTS.json','x') as f:json.dump(stats,f,indent=2)
except Exception as e:profile_error=str(e)
result=dict(parent_checkpoint=parent_path,parent_sha256=receipt['checkpoint_sha256'],base_update=250,effective_AA_batch=64,geometry_decisions=32,goal_decisions=8,microbatch_scope='primary AA stream only; geometry and goal streams remain one each',summary=summary,lazy_loader_summary=loader_summary,loader_equivalence=loader_equivalence,comparisons=comparisons,restart=restart,profiler_error=profile_error,benchmark_updates_not_campaign_progress=len(rows)+1+len(timings)+9,sealed_microbatch1_equivalence=sealed_one_equivalence,J1_and_inactive_unchanged=True,restore_overhead_seconds_excluded=load_seconds)
with open(R/'results/COMPLETE_MICROBATCH_BENCHMARK.json','x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:result[k] for k in ['comparisons','restart','profiler_error']},indent=2));print(json.dumps({k:{n:v[n] for n in ['median_seconds','speedup_fraction']} for k,v in summary.items()},indent=2))
