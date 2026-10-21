from benchlib import *
# Existing capture tolerances are locked. No scientific validation used here.
X.QUALIFIED.setup(pop)
original=X.initialization_update
selected_flags={}
def selected(*args,**kw):
 c=copy.deepcopy(args[-1]);c['execution_protocol'].update(selected_flags);return original(*args[:-1],c,**kw)
def run(micro,gm,goal,shared=False,prefetch=False,capture=True,profile=False):
 global selected_flags
 selected_flags=dict(geometry_microbatch=gm,goal_microbatch=goal,cpu_prefetch=prefetch)
 prior.shared_prefix_features=shared
 with patch.object(X,'initialization_update',selected):return step(micro,True,capture,profile)
def compare(a,b,row,baseline):
 g=a['gradient'];h=b['gradient'];rel=float((g-h).norm()/g.norm());cos=float(torch.dot(g,h)/(g.norm()*h.norm()));pd=float((a['parameters']-b['parameters']).abs().max());ld=abs(row['loss']-baseline['loss']);same=a['samplers']==b['samplers'];rng=a['RNG']==b['RNG']
 return dict(gradient_relative_L2=rel,gradient_cosine=cos,parameter_max_abs=pd,loss_abs=ld,sampler_exact=same,RNG_exact=rng,passed=rel<=.03 and cos>=.999 and pd<=1e-5 and ld<=max(.003,.001*abs(baseline['loss'])) and same and rng)
# Shared invariants must match BOTH reference outputs and actor inputs exactly.
reset('qualified_ram_512MiB');fixtures=[]
for cat in ['complete','partial','aa_only']:
 h=max(pop.groups['TRAIN'][cat],key=lambda h:len(pop.base.seqs[h]));length=len(pop.base.seqs[h]);records=[(cat,h,t,samplers['AA'].rng) for t in [0,1,2,3,length-1]];prefix,aa,g,mask,_=pop.batch(records,'cuda')
 prior.shared_prefix_features=False
 with torch.autocast('cuda',dtype=torch.bfloat16):
  ref=prior(prefix);bb=N.prepare(prefix,ref,torch.zeros(5,1,36,device='cuda'),torch.zeros(5,1,device='cuda',dtype=torch.bool))
  prior.shared_prefix_features=True;rf=prior(prefix);bf=N.prepare(prefix,rf,torch.zeros(5,1,36,device='cuda'),torch.zeros(5,1,device='cuda',dtype=torch.bool))
 def eq(a,b):
  if torch.is_tensor(a):return torch.equal(a,b)
  if isinstance(a,tuple):return all(eq(x,y) for x,y in zip(a,b,strict=True))
  if a is None:return b is None
  return False
 fixtures.append(dict(category=cat,N=length,t=[0,1,2,3,length-1],reference_exact=all(torch.equal(ref[k],rf[k]) for k in ref),actor_features_exact=eq(bb['features'],bf['features'])))
assert all(x['reference_exact'] and x['actor_features_exact'] for x in fixtures),fixtures
(R/'results/SHARED_PREFIX_FEATURE_EQUIVALENCE.json').write_text(json.dumps(fixtures,indent=2))
prior.shared_prefix_features=False;reset('qualified_ram_512MiB');baseline,base=step(4,False,True)
trials=[];caps={};comparisons=[]
for micro,gm,goal in [(8,1,4),(16,1,4),(32,1,4),(16,2,1)]:
 reset('qualified_ram_512MiB');row,cap=run(micro,gm,goal,True,False);c=compare(base,cap,row,baseline);row.update(geometry_microbatch=gm,goal_microbatch=goal,share_prefix_features=True,cpu_prefetch=False);trials.append(row);comparisons.append({**{k:row[k] for k in ['microbatch','geometry_microbatch','goal_microbatch']},**c});print('COMBINED',json.dumps(comparisons[-1]),flush=True)
 if c['passed']:caps[(micro,gm,goal)]=cap
 (R/'results/COMBINED_NUMERICAL_TESTS.json').write_text(json.dumps(dict(trials=trials,comparisons=comparisons),indent=2))
assert caps,'No combined execution passed; retain exact current B4/scalar with verified cache'
# Fixed safe candidates get unprofiled, consecutive-update timings under identical workload.
performance=[]
for micro,gm,goal in caps:
 for prefetch in [False,True]:
  reset('qualified_ram_512MiB')
  for i in range(4):
   row,_=run(micro,gm,goal,True,prefetch,False,profile=(i==0));row.update(geometry_microbatch=gm,goal_microbatch=goal,share_prefix_features=True,cpu_prefetch=prefetch,consecutive_offset=i,cache_bytes=X.QUALIFIED.bytes,cache_hits=X.QUALIFIED.hits,cache_misses=X.QUALIFIED.misses);performance.append(row);print('PERFORMANCE',json.dumps({k:v for k,v in row.items() if k not in ['terms','profile']}),flush=True)
   (R/'results/COMBINED_THROUGHPUT.json').write_text(json.dumps(dict(trials=performance),indent=2))
keys=list(caps);summaries=[]
for key in keys:
 for pf in [False,True]:
  rr=[r for r in performance if (r['microbatch'],r['geometry_microbatch'],r['goal_microbatch'])==key and r['cpu_prefetch']==pf]
  summaries.append(dict(microbatch=key[0],geometry_microbatch=key[1],goal_microbatch=key[2],share_prefix_features=True,cpu_prefetch=pf,loader='qualified_ram_512MiB',mean_seconds=statistics.mean(r['seconds'] for r in rr),steady_median_seconds=statistics.median(r['seconds'] for r in rr[1:]),peak_allocated=max(r['peak_allocated'] for r in rr)))
choice=min(summaries,key=lambda r:r['mean_seconds']);key=(choice['microbatch'],choice['geometry_microbatch'],choice['goal_microbatch'])
# Prefetch must not change a single gradient/model/sampler/RNG value at SAME batching.
reset('qualified_ram_512MiB');row,cap=run(*key,True,choice['cpu_prefetch']);old=caps[key];eq=dict(gradients_exact=torch.equal(old['gradient'],cap['gradient']),parameters_exact=torch.equal(old['parameters'],cap['parameters']),samplers_exact=old['samplers']==cap['samplers'],RNG_exact=old['RNG']==cap['RNG']);assert all(eq.values()),eq
(R/'results/EXECUTION_SELECTION.json').write_text(json.dumps(dict(selected=choice,candidates=summaries,prefetch_equivalence=eq,source_update=372,tolerances_unchanged=True),indent=2));print('SELECTED',json.dumps(choice),flush=True)
# Preserve only small fingerprints and report; large vectors remain RAM-only.
del caps,cap,base
prior.verify(learner.optimizers.values());assert inactive==({k:tensor_sha(m) for k,m in system.models.items() if k!='actor'}|{'targets':tensor_sha(system.targets)})
