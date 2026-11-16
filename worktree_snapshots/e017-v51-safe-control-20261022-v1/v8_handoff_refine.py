"""CPU-only clarifications and exact logical-count matching of existing pools."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
R=Path('/mnt/d/Simone/proteinGen_data/e017_v8_constrained_planning_20261023_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(n,x):
 p=R/n;q=p.with_name(p.name+'.part')
 with q.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def boot(v,g):
 v=np.array(v,float);g=np.array(g);ids=sorted(set(g));ss=np.array([v[g==x].sum() for x in ids]);nn=np.array([(g==x).sum() for x in ids]);draw=np.random.default_rng(18817).integers(0,len(ids),(20000,len(ids)));a=ss[draw].sum(1)/nn[draw].sum(1);return dict(estimate=float(v.mean()),CI95=np.quantile(a,[.025,.975]).tolist(),groups=len(ids),queries=len(v))
p=read(R/'PROVENANCE_SHA256.json');assert all(sha(R/n)==h for n,h in p['output_SHA256'].items());x=read(R/'MATCHED_COMPUTE_COMPARISONS.json');files=sorted(f for f in (R/'results').glob('VALIDATION_[0-9][0-9][0-9]_*.json') if not f.name.endswith('.sha.json'));data=[read(f) for f in files];fine={}
for condition in ['NATIVE','GENERATED']:
 ds=sorted([d for d in data if d['task']['condition']==condition],key=lambda d:d['task']['task_id']);fine[condition]={}
 for K in [8,16,32,64]:
  fine[condition][str(K)]={}
  for method in ['BEAM','CEM','IK']:
   rr=x['metrics'][condition][str(K)][method]['per_task'];out={}
   for axis in ['simulated_transitions','actor_examples']:
    pairs=[];groups=[];ns=[]
    for d,r in zip(ds,rr):
     budget=r['cost'][axis];n=min(64,max(0,int(budget//4))) if axis=='simulated_transitions' else min(64,max(0,int((budget-1)//3)))
     ns.append(n)
     if axis=='actor_examples' and n==0:continue
     base=any(a['success'] for a in d['algorithms']['A']['candidates'][:n]);pairs.append(float(r['success'])-float(base));groups.append(r['group'])
    out[axis]=dict(baseline_pool_prefixes=ns,paired_success_gain=boot(pairs,groups) if pairs else None,scope='exploratory exact logical count bound: baseline4ntransitions and1+3nactor-stateexamples; actual nonmultiples-of8batch runtime not measured; not identicalFLOPs; zeroactorbaseline excluded')
   fine[condition][str(K)][method]=out
save('FINE_LOGICAL_COMPUTE_MATCHING.json',fine)
clarification='''## Practical planning conclusion and limits

At cap 64, full valid success was Best-of-K 50/64 native and 17/64 generated; beam 48/64 and 16/64; CEM 51/64 and 15/64; IK 53/64 and 30/64. Every condition contains 32 protein groups with two goals each.

IK's generated-prefix gain is +20.31 percentage points, with familywise 98.333% interval [6.25, 35.94]. This is supported for that condition. Its native gain is +4.69 points, interval [-10.94, 21.88], and misses the declared 10-point minimum. Therefore no planner passes the joint two-condition promotion rule; that does not negate the generated-prefix improvement.

Measured warm planner latency at cap 64 was approximately 0.22 seconds/query for IK versus 1.05–1.07 for Best-of-K. CEM took 0.23–0.24 seconds; beam took 2.36–3.23 seconds. These exclude separately reported prior-audit overhead and model setup. Beam used fewer geometric transitions but more batch-1 neural forwards than the batched baseline; this implementation's runtime is not a universal algorithmic speed claim.

The practical recommendation is to retain classical IK and frozen Best-of-K as complementary H4 baselines. IK provides a useful, inexpensive model-based alternative, particularly for generated prefixes. It does not justify promoting a neural policy: about 47% of returned native IK solutions and 77% of generated-prefix IK solutions contain an action component beyond three parent standard deviations. A future prior-preserving planner would need an explicit, separately tested prior constraint. No such extension or RL training is launched here.

The current problem does not demonstrate that reinforcement learning is necessary. It is not completely solved either: two native and 31 generated-prefix queries remain unsolved by the union of cap-64 methods. Native witnesses prove feasibility of the former; root bounds distinguish proven obstructions from finite-search uncertainty in generated cases. Future high-level RL needs meaningful delayed or uncertain decisions rather than relabeling this deterministic H4 search.

Additional CPU-only analysis in FINE_LOGICAL_COMPUTE_MATCHING.json uses every available baseline prefix, rather than only 8/16/32/64, to match logical transition or actor-state-example counts without exceeding each planner's count. This is exploratory, preserves every primary outcome and gate, and does not claim measured latency or equal FLOPs for unprofiled batch sizes.

Effort-curve correction: A and CEM retain nested pools. Although IK is executed separately at each cap for timing, it extends the same two deterministic optimizer traces, so its evaluated geometry sets are nested as well. Only beam's changing allocation is not guaranteed to produce nested pools. The separate IK executions are not independent biological evidence.
'''
for name in ['FINAL_E017_V8_SCIENTIFIC_REPORT.md','NEXT_CRISP_RESEARCH_DECISION.md','PLANNING_METHODS_AND_VALIDITY.md']:
 path=R/name;value=path.read_text();value=value.replace('A/CEM pools are nested; beam and IK are separate bounded searches per cap, so their effort curves need not be monotone.','A/CEM pools are nested. IK reruns the same two deterministic traces at each cap and extends their evaluated sets; beam allocations are not nested.').replace('A and CEM retain nested candidate pools, so exact-valid selected success cannot decrease with cap. Beam and IK are separately bounded searches; their effort curves may fluctuate.','A, CEM and the fixed-start IK traces have nested evaluated sets. Beam allocations change across caps, so only its curve need not be monotone.');path.write_text(value+'\n\n'+clarification)
p['input_SHA256'][str(Path(__file__).resolve())]=sha(__file__);p['output_SHA256']={n:sha(R/n) for n in p['output_SHA256']};p['output_SHA256']['FINE_LOGICAL_COMPUTE_MATCHING.json']=sha(R/'FINE_LOGICAL_COMPUTE_MATCHING.json');p['handoff_refinement']=dict(UTC=time.time(),scope='clarify condition-specific IK evidence, nested fixed-start traces, and exploratory finer logical-count matching; no new planning or neural calls; no primary result/criterion change');save('PROVENANCE_SHA256.json',p)
v=read(R/'FINAL_VERIFICATION.json');v.update(provenance_SHA256=sha(R/'PROVENANCE_SHA256.json'),verified_input_files=len(p['input_SHA256']),verified_output_files=len(p['output_SHA256']),handoff_refinement_UTC=time.time());save('FINAL_VERIFICATION.json',v);assert all(sha(R/n)==h for n,h in p['output_SHA256'].items());print(json.dumps(dict(final_output_hashes_verified=True,beam64_native_fine_transition_match=fine['NATIVE']['64']['BEAM']['simulated_transitions']['paired_success_gain'])))
