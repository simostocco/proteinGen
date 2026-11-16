"""Final CPU-only V8 integrity, compute reconciliation and scientific handoff."""
from pathlib import Path
import collections,hashlib,json,os,runpy,shutil,subprocess,sys,time
import numpy as np
R=Path('/mnt/d/Simone/proteinGen_data/e017_v8_constrained_planning_20261023_v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(8*2**20),b''):h.update(a)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(n,x):
 p=R/n;q=p.with_name(p.name+'.part')
 with q.open('w') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def append(n,value):
 p=R/n;q=p.with_name(p.name+'.part')
 with q.open('w') as f:f.write(p.read_text()+'\n\n'+value+'\n');f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def active():
 out=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:comm=(p/'comm').read_text();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();state=(p/'stat').read_text().split(') ',1)[1][0]
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if state!='Z' and comm.startswith('python') and any(x in cmd.lower() for x in ['scripts/e017_','v8_watch.py','v72_watch.py','proteinmpnn','colabfold','e007','e015','e012']):out.append(dict(pid=int(p.name),command=cmd))
 return out
def boot(values,groups):
 v=np.array(values,float);g=np.array(groups);ids=sorted(set(g));s=np.array([v[g==x].sum() for x in ids]);n=np.array([(g==x).sum() for x in ids]);draw=np.random.default_rng(18817).integers(0,len(ids),(20000,len(ids)));x=s[draw].sum(1)/n[draw].sum(1);return dict(estimate=float(v.mean()),CI95=np.quantile(x,[.025,.975]).tolist(),groups=len(ids),queries=len(v))
def main():
 assert not (R/'FINAL_VERIFICATION.json').exists();assert read(R/'EXECUTION.json')['status']=='COMPLETED';assert not active(),active();p=read(R/'PROVENANCE_SHA256.json')
 for kind,items in [('in',p['input_SHA256']),('out',p['output_SHA256'])]:
  for name,h in items.items():assert sha(Path(name) if kind=='in' else R/name)==h,name
 st=runpy.run_path(str(R/'source/scripts/e017_v8/study.py'));metric=read(R/'MATCHED_COMPUTE_COMPARISONS.json');files=sorted(f for f in (R/'results').glob('VALIDATION_[0-9][0-9][0-9]_*.json') if not f.name.endswith('.sha.json'));data=[read(f) for f in files];assert len(data)==128;seeds={};counts=collections.Counter();root=collections.Counter();clean={};pruning={}
 mechanisms=[]
 for d in data:
  t=d['task'];al=d['algorithms'];key=t['task_id']+'|'+t['condition'];rows=[]
  for name,r in al.items():
   if name=='PRIOR_AUDIT':continue
   snap=r['snapshots']['64'] if name in ['A','CEM'] else r;cost=snap['cost'];counts['planner_actor_examples_standalone_sum']+=cost['actor_examples'];counts['planner_transitions_standalone_sum']+=cost['simulated_transitions']
   if name.startswith('BEAM_'):
    for z in r['candidates']:
     ids=z['child_path'].split('/')[1:];action_seeds=[st['ns']('beam_action',t['task_id'],t['condition'],j,'root'+('/'+'/'.join(ids[:j]) if j else ''),int(a)) for j,a in enumerate(ids)];rows.append(dict(method=name,candidate=z['index'],action_seeds=action_seeds,completion_seed=z['completion_seed']))
  # Standalone CEM includes A's first8; actual study reuses those calls/FK.
  counts['actual_planner_actor_examples']+=sum((r['snapshots']['64']['cost'] if name in ['A','CEM'] else r['cost'])['actor_examples'] for name,r in al.items() if name!='PRIOR_AUDIT')-al['A']['snapshots']['8']['cost']['actor_examples'];counts['actual_search_transitions']+=sum((r['snapshots']['64']['cost'] if name in ['A','CEM'] else r['cost'])['simulated_transitions'] for name,r in al.items() if name!='PRIOR_AUDIT')-32;counts['prior_audit_actor_examples']+=al['PRIOR_AUDIT']['actor_examples'];counts['prior_audit_transitions']+=al['PRIOR_AUDIT']['simulated_transitions'];counts['prefix_generation_actor_examples']+=d['prefix_generation_cost']['actor_examples'];counts['prefix_generation_transitions']+=d['prefix_generation_cost']['simulated_transitions'];root[(t['condition'],tuple(d['root_irreparable_or_unreachable']))]+=1
  seeds[key]=dict(identity=t['identity'],protected_group=t['protected_group_id'],A=[z['seed'] for z in al['A']['candidates']],CEM_generations={str(g):st['ns']('CEM',t['task_id'],t['condition'],g) for g in range(1,8)},beam=rows,generated_prefix=[st['ns']('GENERATED_PREFIX',t['identity'],j) for j in range(4)] if t['condition']=='GENERATED' else None,IK='two deterministic starts, no random stream')
  per={}
  for method in ['A','CEM','BEAM_64','IK_64']:
   pool=al[method]['candidates'];per[method]=dict(candidates=len(pool),valid=sum(r['valid'] for r in pool),full_success=sum(r['success'] for r in pool),translation_hits=sum(r['translation_hit'] for r in pool),orientation_hits=sum(r['rotation_hit'] for r in pool),invalid_goal_hits=sum(r['translation_hit'] and r['rotation_hit'] and not r['valid'] for r in pool),invalid_causes=dict(collections.Counter(x for r in pool for x in r['validity_failures'])))
  mechanisms.append(dict(task_id=t['task_id'],identity=t['identity'],group=t['protected_group_id'],condition=t['condition'],variant=t['variant'],root_proofs=d['root_irreparable_or_unreachable'],methods=per,native_goal_has_prepared_valid_witness=t['condition']=='NATIVE'))
 worker=read(R/'VALIDATION_COMPLETED.json');measured=counts['actual_planner_actor_examples']+counts['prior_audit_actor_examples']+counts['prefix_generation_actor_examples'];assert measured==worker['actor_examples'],(measured,worker['actor_examples']);counts['actual_total_actor_examples_reconciled']=measured
 counts['posthoc_diversity_reconstruction_transitions']=4*sum(v['candidate_count'] for kk in metric['metrics'].values() for mm in kk.values() for v in mm.values())
 for c in ['NATIVE','GENERATED']:
  q=metric['metrics'][c]['64'];base=q['A']['per_task'];keep=[i for i,r in enumerate(base) if not r['root_bounds']];groups=[base[i]['group'] for i in keep];clean[c]={}
  for name in ['A','BEAM','CEM','IK']:
   rr=q[name]['per_task'];clean[c][name]=dict(queries=len(keep),groups=len(set(groups)),full_success=boot([rr[i]['success'] for i in keep],groups) if keep else None,paired_gain_vs_A=boot([float(rr[i]['success'])-float(base[i]['success']) for i in keep],groups) if keep else None)
  beam=q['BEAM']['per_task'];pruning[c]=dict(root_stops=sum(bool(r['root_bounds']) for r in beam),partial_and_root_rejection_events=sum(r['prefix_rejections'] for r in beam),terminal_evaluations=q['BEAM']['terminal_evaluations'],maximum_terminal_cap=64*64,unused_terminal_capacity=64*64-q['BEAM']['terminal_evaluations'],rejection_causes=dict(collections.Counter(x for d in data if d['task']['condition']==c for a in d['algorithms']['BEAM_64']['rejections'] for x in a['reasons'])))
 save('SEED_MANIFEST.json',dict(seed=8817,rule='SHA256(E017V8|8817|joined inputs),first15hex; TorchCPUdoubleGaussian and NumPyPCG64 CEM',queries=seeds));save('POOL_FAILURE_MECHANISMS.json',mechanisms);save('COMPUTE_RECONCILIATION.json',dict(counts=dict(counts),measured_worker_actor_examples=worker['actor_examples'],exact_actor_example_reconciliation=True,nearest_resource_matching='primary raw terminal caps supplemented by actual transition/actor-count frontiers; CPU/GPU timers overlap and must not be added as exclusive durations',root_stratification=[dict(condition=k[0],reasons=list(k[1]),queries=n) for k,n in root.items()],no_root_bound_exploratory=clean,pruning=pruning))
 answers=['## Final measured scientific answers']
 for c in ['NATIVE','GENERATED']:
  q=metric['metrics'][c]['64'];answers.append(c+' cap64: '+', '.join(f"{m} {q[m]['full_success_count']}/64 successes, {q[m]['wall_seconds']/64:.4f} warm planner seconds/query" for m in ['A','BEAM','CEM','IK'])+'.')
 answers.append('Sequential beam versus Best-of64: '+str({c:metric['paired'][c]['64']['BEAM'] for c in ['NATIVE','GENERATED']})+'.')
 answers.append('Continuous CEM versus Best-of64: '+str({c:metric['paired'][c]['64']['CEM'] for c in ['NATIVE','GENERATED']})+'.')
 answers.append('Classical IK versus Best-of64: '+str({c:metric['paired'][c]['64']['IK'] for c in ['NATIVE','GENERATED']})+'.')
 answers.append('Early rejection: '+str(pruning)+'. Root stops avoid all downstream search for a provably obstructed query. Partial rejection can reallocate the fixed lookahead budget rather than reduce total work. There is no pruning-off wall-time ablation, so do not attribute the entire beam outcome or runtime difference to the filter alone.')
 answers.append('Validity uses the completed H4 fragment length t+4=24 for this panel. Deposited full-protein length remains neural metadata; it is not substituted into clash/Rg validity normalization. Original tolerances remain 8 A and 35 degrees. POOL_FAILURE_MECHANISMS.json separates goal misses, invalid goal hits and individual backbone failure criteria across the same evaluated pools.')
 answers.append('Effort scaling: A and CEM retain nested candidate pools, so exact-valid selected success cannot decrease with cap. Beam and IK are separately bounded searches; their effort curves may fluctuate. All discarded and optimized candidates count. Actual geometry transitions match A/CEM/IK at4K; beam can spend less through shared prefixes or early stops. IK gradient work remains additional CPU cost. Shared prior audits and reused initial CEM trajectories are separately reconciled in COMPUTE_RECONCILIATION.json.')
 for c in ['NATIVE','GENERATED']:
  prior=metric['prior'][c];answers.append(c+' geometric-prior diagnostics at64: '+', '.join(f"{m} selected any|z|>3 fraction={prior[m]['fraction_selected_with_any_abs_z_over3']}, mean max|z|={prior[m]['mean_max_abs_standardized_latent']}" for m in ['A','BEAM','CEM','IK'])+'.')
 answers.append('Prior interpretation: original actor-to-J1 conditional KL at visited states, the CEM unselected-generation q-to-parent Monte Carlo KL, and the final selected search law are different objects. Neither proposal KL nor pointwise parent density certifies selected-law KL compliance. IK is deterministic and has no continuous sampling density. Strong goal success outside typical actor actions is kinematic planning evidence, not a promoted autoregressive protein policy.')
 uns=metric['unsolved'];answers.append('Remaining unresolved queries by condition: '+str({c:sum(r['no_method_solved'] and r['condition']==c for r in uns) for c in ['NATIVE','GENERATED']})+'. Native witnesses establish feasibility of original-prefix goals; generated-prefix obstructions are distinguished from finite search failure. Exact query IDs remain in UNSOLVED_TASK_ANALYSIS.md/JSON. No unobserved future structure is treated as a planner input.')
 best={c:max(metric['metrics'][c]['64'],key=lambda m:metric['metrics'][c]['64'][m]['full_success_count']) for c in ['NATIVE','GENERATED']};answers.append('RL relevance: the strongest cap64success point estimates are '+str(best)+'. This static fully observed deterministic H4 task has an exact transition and terminal evaluator, so these results do not establish a need for reinforcement learning. The immediate methodological recommendation is to retain the measured model-based/Best-of-K frontier, including prior-deviation and generated-prefix limitations. A future RL objective should add prospectively meaningful delayed subgoal decisions or uncertain feedback, not relabel this deterministic search as RL. No future stage is launched.')
 if (R/'IMPLEMENTATION_CORRECTION_001.json').exists():answers.append('Engineering recovery: the first campaign stopped before any durable planning query on a panel-path concatenation error. The initial protocol/source/preflight and failed log are preserved. Path/timing instrumentation was corrected, preflight was rerun, and every scientific configuration/threshold remained frozen before all128validation queries. No production query was repeated.')
 if (R/'IMPLEMENTATION_CORRECTION_002.json').exists():answers.append('Analysis recovery: after all128queries completed, the initial CPU reader matched SHA sidecars as task records. Only the file-filter interface was corrected; every task-block hash, algorithm, statistic and scientific threshold remained unchanged. Collection and analysis source commits and the preserved protocol are in IMPLEMENTATION_CORRECTION_002.json. No planning or neural forward pass was repeated for this repair.')
 prose='\n\n'.join(answers);append('FINAL_E017_V8_SCIENTIFIC_REPORT.md',prose);append('NEXT_CRISP_RESEARCH_DECISION.md',prose)
 table=['## Candidate validity, collisions and diversity at cap 64','', '| Condition | Method | Terminal candidates | Valid candidates | Collision failures | Mean pair tail RMSD (A) |', '| --- | --- | --- | --- | --- | --- |']
 for c in ['NATIVE','GENERATED']:
  for m in ['A','BEAM','CEM','IK']:
   q=metric['metrics'][c]['64'][m];table.append('| '+ ' | '.join(map(str,[c,m,q['candidate_count'],q['candidate_valid_count'],q['candidate_collision_failures'],q['mean_pair_tail_RMSD_A']]))+' |')
 table.extend(['','Counts include failed complete candidates; beam partial rejection is separate. IK candidates are correlated optimizer iterates, not independent policy draws. CEM and A share their first eight candidates. Diversity is descriptive, not evidence of biological novelty.'])
 append('FINAL_E017_V8_SCIENTIFIC_REPORT.md','\n'.join(table));append('NATIVE_VS_GENERATED_PREFIX_RESULTS.md','\n'.join(table))
 # Actual code paths include inherited historical helpers. Hash them explicitly.
 paths={str(Path(__file__).resolve()),str(R/'source/scripts/e017_v8/study.py'),str(R/'source/scripts/e017_v8/analyze.py'),str(st['b']['__file__']),str(st['b']['s']['__file__']),str(st['v']['__file__'])};paths.update(str(f) for f in (R/'control').glob('*.py'))
 for module in list(sys.modules.values()):
  file=getattr(module,'__file__',None)
  if file and '/proteingen_data/' in str(file).lower() and str(file).endswith('.py') and Path(file).is_file():paths.add(str(Path(file).resolve()))
 for file in [Path(st['s'].__file__).with_name('frozen_science_config.json'),R/'IMPLEMENTATION_CORRECTION_001.json',R/'IMPLEMENTATION_CORRECTION_002.json',R/'PROTOCOL_INITIAL.json',R/'PROTOCOL_BEFORE_ANALYSIS_FIX.json',R/'PREFLIGHT_INITIAL.json',R/'VALIDATION_RESOURCE_ADMISSION.json']:
  if file.exists():paths.add(str(file))
 runtime={f:sha(f) for f in sorted(paths)};save('RUNTIME_SOURCE_AUDIT.json',dict(primary_source_commit=p['source_commit'],runtime_file_SHA256=runtime,historical_helpers='actual inheritedV72/V71/V7/V5/V4 source paths retained and hashed; no historical source modified'))
 res=read(R/'RESOURCE_AND_RECOVERY_REPORT.json');watch=[json.loads(x) for x in (R/'results/resource_watch.jsonl').read_text().splitlines()];res['watchdog']=dict(samples=len(watch),min_D_free_bytes=min(z['D_free_bytes'] for z in watch),min_available_RAM_bytes=min(z['available_RAM_bytes'] for z in watch),max_swap_used_bytes=max(z['swap_used_bytes'] for z in watch),resource_stop=any(z['resource_stop'] for z in watch));res['D_free_bytes']=shutil.disk_usage(R).free;res['software']=dict(Python=sys.version,Torch=st['torch'].__version__,NumPy=np.__version__,CUDA_build=st['torch'].version.cuda);res['compute_reconciliation']=dict(counts);res['workers']='all research workers exited; no automatic restart or further research stage';save('RESOURCE_AND_RECOVERY_REPORT.json',res)
 assert not subprocess.check_output(['git','-C',str(R/'source'),'status','--porcelain'],text=True).strip()
 assert not active(),active();gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not gpu.strip(),gpu;assert shutil.disk_usage(R).free>2*2**30;assert not list((R/'results').glob('*.part'))
 for f,h in p['input_SHA256'].items():assert sha(f)==h,f
 p['input_SHA256'].update(runtime);p['input_SHA256'][str(R/'results/resource_watch.jsonl')]=sha(R/'results/resource_watch.jsonl');p['output_SHA256']={f:sha(R/f) for f in p['output_SHA256']}
 for name in ['SEED_MANIFEST.json','COMPUTE_RECONCILIATION.json','RUNTIME_SOURCE_AUDIT.json','POOL_FAILURE_MECHANISMS.json']:p['output_SHA256'][name]=sha(R/name)
 p['final_review']=dict(UTC=time.time(),exact_actor_example_reconciliation=True,scope='CPU-only numerical handoff and resource/source/hash verification; no model evaluation or scientific gate changes');save('PROVENANCE_SHA256.json',p)
 verification=dict(PASS=True,UTC=time.time(),source_commit=p['source_commit'],verified_input_files=len(p['input_SHA256']),verified_output_files=len(p['output_SHA256']),provenance_SHA256=sha(R/'PROVENANCE_SHA256.json'),completed_validation_condition_queries=128,independent_VALIDATION_groups=32,active_research_workers=[],GPU_compute_processes=[],D_free_bytes=shutil.disk_usage(R).free,D_free_GiB=shutil.disk_usage(R).free/2**30,all_artifacts_on_D=True,actor_updates=0,J1_updates=0,critic_updates=0,SAC_updates=0,decisions=metric['decisions']);save('FINAL_VERIFICATION.json',verification);print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
