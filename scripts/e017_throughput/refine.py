from benchlib import *
# Retain the failed large-batch evidence. No tolerance relaxation.
reset();baseline,base=step(4,False,True,True);trials=[baseline];comparisons=[]
for micro,gm,goal in [(8,1,1),(16,1,1),(4,2,1),(8,2,1),(4,1,4)]:
 reset();orig=X.initialization_update
 def selected(*args,**kw):
  c=copy.deepcopy(args[-1]);c['execution_protocol'].update(geometry_microbatch=gm,goal_microbatch=goal);return orig(*args[:-1],c,**kw)
 with patch.object(X,'initialization_update',selected):row,cap=step(micro,True,True,True)
 row.update(geometry_microbatch=gm,goal_microbatch=goal);a=base['gradient'];b=cap['gradient'];rel=float((a-b).norm()/a.norm());cos=float(torch.dot(a,b)/(a.norm()*b.norm()));pd=float((base['parameters']-cap['parameters']).abs().max());ld=abs(row['loss']-baseline['loss']);same=base['samplers']==cap['samplers'];rng=base['RNG']==cap['RNG'];passed=rel<=.03 and cos>=.999 and pd<=1e-5 and ld<=max(.003,.001*abs(baseline['loss'])) and same and rng
 comparisons.append(dict(microbatch=micro,geometry_microbatch=gm,goal_microbatch=goal,gradient_relative_L2=rel,gradient_cosine=cos,parameter_max_abs=pd,loss_abs=ld,sampler_exact=same,RNG_exact=rng,passed=passed));trials.append(row);print('REFINED',json.dumps(row),json.dumps(comparisons[-1]),flush=True)
 (R/'results/REFINED_NUMERICAL_BENCHMARK.json').write_text(json.dumps(dict(trials=trials,comparisons=comparisons),indent=2));del cap;torch.cuda.empty_cache()
# Loader-only numerical comparison must be BITWISE exact at unchanged B4/scalar auxiliaries.
X.QUALIFIED.setup(pop);reset('qualified_ram_512MiB');row,cap=step(4,False,True,True);eq=dict(loss_exact=row['loss']==baseline['loss'],gradient_exact=torch.equal(base['gradient'],cap['gradient']),parameters_exact=torch.equal(base['parameters'],cap['parameters']),samplers_exact=base['samplers']==cap['samplers'],RNG_exact=base['RNG']==cap['RNG']);assert all(eq.values()),eq
(R/'results/QUALIFIED_CACHE_EQUIVALENCE.json').write_text(json.dumps(eq,indent=2));del cap,base
# Counterbalanced steady steps, caches retained over NEW consecutive updates; startup separate.
perf=[]
for loader in ['eager','qualified_ram_512MiB','qualified_ram_512MiB','eager']:
 reset(loader)
 for i in range(4):
  row,_=step(4,False,profile=(i==0));row.update(loader=loader,consecutive_offset=i,cache_bytes=X.QUALIFIED.bytes,cache_hits=X.QUALIFIED.hits,cache_misses=X.QUALIFIED.misses);perf.append(row);print('CACHE_TIMING',json.dumps(row),flush=True)
  (R/'results/QUALIFIED_CACHE_THROUGHPUT.json').write_text(json.dumps(dict(trials=perf),indent=2))
prior.verify(learner.optimizers.values());assert inactive==({k:tensor_sha(m) for k,m in system.models.items() if k!='actor'}|{'targets':tensor_sha(system.targets)})
print('REFINE_DONE',flush=True)
