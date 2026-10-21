from benchlib import *
import signal
choice=json.load(open(R/'results/EXECUTION_SELECTION.json'))['selected'];key=(choice['microbatch'],choice['geometry_microbatch'],choice['goal_microbatch']);X.QUALIFIED.setup(pop);original=X.initialization_update
flags=dict(geometry_microbatch=key[1],goal_microbatch=key[2],cpu_prefetch=True)
def selected(*args,**kw):
 c=copy.deepcopy(args[-1]);c['execution_protocol'].update(flags);return original(*args[:-1],c,**kw)
def run(capture=False,profile=False):
 prior.shared_prefix_features=True
 with patch.object(X,'initialization_update',selected):return step(key[0],True,capture,profile)
reset('qualified_ram_512MiB');baseline,base=run(True)
flags['geometry_cpu_prefetch']=True;reset('qualified_ram_512MiB');row,cap=run(True)
eq=dict(gradient_exact=torch.equal(base['gradient'],cap['gradient']),parameters_exact=torch.equal(base['parameters'],cap['parameters']),samplers_exact=base['samplers']==cap['samplers'],RNG_exact=base['RNG']==cap['RNG']);assert all(eq.values()),eq
(R/'results/GEOMETRY_PREFETCH_EQUIVALENCE.json').write_text(json.dumps(eq,indent=2));del cap
performance=[]
for loader in ['qualified_ram_512MiB','qualified_ram_1024MiB']:
 reset(loader)
 for i in range(6):
  row,_=run(False,profile=(i==0));row.update(loader=loader,consecutive_offset=i,cache_bytes=X.QUALIFIED.bytes,cache_hits=X.QUALIFIED.hits,cache_misses=X.QUALIFIED.misses);performance.append(row);print('FINAL_TUNE',json.dumps({k:v for k,v in row.items() if k not in ['terms','profile']}),flush=True)
  (R/'results/FINAL_CACHE_AND_PREFETCH_TIMING.json').write_text(json.dumps(dict(trials=performance),indent=2))
summary={loader:dict(mean_seconds=statistics.mean(r['seconds'] for r in performance if r['loader']==loader),steady_median_seconds=statistics.median(r['seconds'] for r in performance if r['loader']==loader and r['consecutive_offset']>=2)) for loader in ['qualified_ram_512MiB','qualified_ram_1024MiB']}
loader=min(summary,key=lambda k:summary[k]['mean_seconds']);choice.update(loader=loader,geometry_cpu_prefetch=True,mean_seconds=summary[loader]['mean_seconds'],steady_median_seconds=summary[loader]['steady_median_seconds'],compile_actor_rope=False)
# Compilation trial is limited to actor RoPE: J1 and physical likelihood code stay eager.
compile_result={};signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('bounded actor-RoPE compilation exceeded120s')))
reset(loader)
try:
 signal.alarm(120);t=time.monotonic();X.enable_compiled_rope(True);row,cap=run(True);compile_result['cold_seconds']=time.monotonic()-t;signal.alarm(0)
 reset(loader);X.enable_compiled_rope(False);eager,ec=run(True)
 a=ec['gradient'];b=cap['gradient'];rel=float((a-b).norm()/a.norm());cos=float(torch.dot(a,b)/(a.norm()*b.norm()));pd=float((ec['parameters']-cap['parameters']).abs().max());ld=abs(eager['loss']-row['loss']);passed=rel<=.03 and cos>=.999 and pd<=1e-5 and ld<=max(.003,.001*abs(eager['loss'])) and ec['samplers']==cap['samplers'] and ec['RNG']==cap['RNG']
 compile_result.update(gradient_relative_L2=rel,gradient_cosine=cos,parameter_max_abs=pd,loss_abs=ld,samplers_exact=ec['samplers']==cap['samplers'],RNG_exact=ec['RNG']==cap['RNG'],numeric_passed=passed)
 del cap,ec
 if passed:
  rows=[]
  for compiled in [False,True]:
   reset(loader);X.enable_compiled_rope(compiled)
   for i in range(4):
    r,_=run(False);rows.append(dict(compiled=compiled,offset=i,seconds=r['seconds']))
  compile_result['timings']=rows;means={v:statistics.mean(r['seconds'] for r in rows if r['compiled']==v) for v in [False,True]};compile_result['speedup_fraction']=1-means[True]/means[False]
  if compile_result['speedup_fraction']>=.05:choice.update(compile_actor_rope=True,mean_seconds=means[True])
except Exception as e:compile_result.update(error=repr(e),promoted=False)
finally:signal.alarm(0);X.enable_compiled_rope(False)
compile_result['promoted']=choice['compile_actor_rope'];(R/'results/COMPILED_EXECUTION_COMPARISON.json').write_text(json.dumps(compile_result,indent=2))
# Full next-write and near-growth policy remains inherited from scientific config.
(R/'results/FINAL_EXECUTION_SELECTION.json').write_text(json.dumps(dict(selected=choice,cache_comparison=summary,geometry_prefetch_equivalence=eq,compilation=compile_result),indent=2));print('FINAL_SELECTED',json.dumps(choice),flush=True)
prior.verify(learner.optimizers.values());assert inactive==({k:tensor_sha(m) for k,m in system.models.items() if k!='actor'}|{'targets':tensor_sha(system.targets)})
