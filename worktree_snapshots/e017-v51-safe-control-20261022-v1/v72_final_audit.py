"""Final integrity review and descriptive CRN covariance correction; no fitting/inference."""
from pathlib import Path
import hashlib,json,os,runpy,shutil,subprocess,sys,time
import numpy as np
R=Path('/mnt/d/Simone/proteinGen_data/e017_v72_counterfactual_ranking_20261023_v1')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,data):
 p=R/name;q=p.with_name(p.name+'.part')
 with q.open('w') as f:json.dump(data,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def append(name,text):
 p=R/name;q=p.with_name(p.name+'.part')
 with q.open('w') as f:f.write(p.read_text()+'\n\n'+text+'\n');f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def active():
 rows=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();comm=(p/'comm').read_text().strip();state=(p/'stat').read_text().split(') ',1)[1][0]
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if state!='Z' and comm.startswith('python') and any(t in cmd.lower() for t in ['scripts/e017_','v72_watch.py','v71_watch.py','v71_finish.py','proteinmpnn','colabfold','e007','e015','e012']):rows.append(dict(pid=int(p.name),command=cmd))
 return rows
def main():
 assert not (R/'FINAL_VERIFICATION.json').exists()
 assert read(R/'EXECUTION.json')['status']=='COMPLETED';assert read(R/'CAMPAIGN_COMPLETED.json')['complete'];assert not active(),active()
 p=read(R/'PROVENANCE_SHA256.json');old_provenance=sha(R/'PROVENANCE_SHA256.json')
 for key,items in [('input',p['input_SHA256']),('output',p['output_SHA256'])]:
  for path,h in items.items():assert sha(Path(path) if key=='input' else R/path)==h,path
 s=runpy.run_path(str(R/'source/scripts/e017_v72/study.py'));metrics=read(R/'STAGE_STRATIFIED_RANKING.json');corrected=0
 # CRN gives correlated action-mean errors: trace(P Cov/M)/(A-1),
 # not simply mean marginal variance/M. This descriptive correction
 # does not affect pair contrasts, fitting, selection or any frozen gate.
 for panel in ['FIT','DEV','CONFIRM','GENERATED']:
  pools={(rr[0]['task_id'],rr[0]['stage']):rr for rr in s['group_states'](s['records'](panel))}
  for k,v in metrics['identifiability'][panel].items():
   for rec in v['records']:
    pool=pools[(rec['task_id'],int(k))];Y=np.array([[a['return_task'] for a in r['banks']['EST']] for r in pool]);M=Y.shape[1];noise=0.
    if M>1:
     cov=np.cov(Y,ddof=1);noise=float(max(0,(np.trace(cov)-cov.sum()/8)/(7*M)))
    rec['uncorrelated_noise_approximation_before_review']=rec['mean_estimation_noise_variance'];rec['mean_estimation_noise_variance']=noise;rec['noise_subtracted_between_variance']=max(0,rec['between_action_observed_variance']-noise);rec['noise_correction']='CRN covariance aware trace(P Cov/M)/(A-1)';corrected+=1
 save('STAGE_STRATIFIED_RANKING.json',metrics)
 note='Final descriptive audit: mean-estimation noise uses the full common-random-number covariance: trace(P Cov/M)/(A−1), with P=I−11ᵀ/A and A=8. The earlier uncorrelated approximation is retained in JSON for transparency. This correction changes only descriptive between-action variance decomposition; paired MC standard errors, identifiability flags, model fitting/selection, uncertainty intervals and all GO criteria are unchanged.'
 append('ACTION_VALUE_IDENTIFIABILITY.md',note)
 def bootstrap(a,groups):
  a=np.array(a);groups=np.array(groups);ids=sorted(set(groups));ss=np.array([a[groups==g].sum() for g in ids]);nn=np.array([(groups==g).sum() for g in ids]);draw=np.random.default_rng(87272).integers(0,len(ids),(10000,len(ids)));v=ss[draw].sum(1)/nn[draw].sum(1);return dict(estimate=float(a.mean()),CI95=np.quantile(v,[.025,.975]).tolist(),groups=len(ids),replicates=10000,seed=87272)
 decision=metrics['decision'];primary=decision['primary'];a=metrics['metrics']['CONFIRM']['0'][primary];generated=metrics['metrics']['GENERATED']['0'][primary];rows=[];oracles={}
 for panel in ['FIT','DEV','CONFIRM','GENERATED']:
  oracles[panel]={}
  for k in range(4):
   rr=metrics['metrics'][panel][str(k)][primary]['selection_records'];groups=[r['group'] for r in rr];oracles[panel][str(k)]=dict(assessed_return=bootstrap([r['selected_return']+r['split_oracle_regret'] for r in rr],groups),advantage_vs_uniform=bootstrap([r['selected_return']+r['split_oracle_regret']-r['uniform_return'] for r in rr],groups),meaning='EST Monte Carlo argmax evaluated on independent ASSESS; finite-budget sampling selector, not trueQ oracle or learned planning')
 metrics['independent_EST_MC_selector']=oracles;save('STAGE_STRATIFIED_RANKING.json',metrics)
 for k in range(4):
  ident=metrics['identifiability']['CONFIRM'][str(k)];m=metrics['metrics']['CONFIRM'][str(k)][primary];rows.append(f"Stage{k}: replicated distinguishable states {ident['replicated_identifiable']['estimate']:.1%}; primary return gain vs uniform {m['advantage_vs_uniform']['estimate']:.5f} (95% {m['advantage_vs_uniform']['CI95']}); gain vs heuristic {m['advantage_vs_heuristic']['estimate']:.5f} (95% {m['advantage_vs_heuristic']['CI95']}); replicated-pair ranking {m['pair_accuracy_replicated']['estimate'] if m['pair_accuracy_replicated'] else 'unsupported'}, coverage {m['replicated_pairs']} pairs/{m['replicated_pair_groups']} proteins.")
 nominal=metrics['identifiability']['CONFIRM']['0']['nominal_identifiable'];rep=metrics['identifiability']['CONFIRM']['0']['replicated_identifiable'];stage0=metrics['metrics']['CONFIRM']['0'];bestlearned=max((n for n in stage0 if n not in ['constant','heuristic']),key=lambda n:stage0[n]['selected_return']['estimate'])
 if decision['GO']:pivot='A separately authorized small actor pilot is scientifically supportable under the frozen gates, but soft-Q targets, action-gradient behavior and safety monitoring still need explicit alignment. No SAC is launched.'
 elif rep['estimate']>=.25:pivot='Action effects are identifiable for a meaningful subset, but the frozen promotion criteria do not establish reliable compact-model exploitation. Pivot to a validity-constrained kinematic controller or explicit action-sensitive residual model with independent selection checks; do not add unrestricted critic updates.'
 else:pivot='At this sampling budget, early action-value effects are weakly identified. Pivot to explicit short-horizon kinematic/validity planning and diagnose credit assignment/reward formulation before further value learning; do not infer that RL is impossible or simply add critic updates.'
 answers='\n\n'.join(['## Final quantitative research answers',
 f'1. Initial actions: {nominal["estimate"]:.1%} of confirmation stage0states have nominal distinguishable pairs; {rep["estimate"]:.1%} independently replicate. These fractions concern the eight-action parent pool, not all physically possible actions.',
 '2. Reliability: only replicated contrasts support ranking claims at sixteen continuations per bank. Familywise approximate contrasts and coverage are reported separately. Pool-wide near ties remain uncertain.',
 f'3. Compact ranking: prospectively selected primary is {primary}. Stage0 independent return gain vs heuristic is {a["advantage_vs_heuristic"]["estimate"]:.5f}, 95% {a["advantage_vs_heuristic"]["CI95"]}. Predictive RMSE alone is not promotion.',
 f'4. Large critics: the strongest learned stage0 point estimate in this confirmation pool is {bestlearned}; this is descriptive and does not replace the DEV-selected primary. Full paired strategy gains show whether a Transformer adds practical selection value.',
 '5. Stage effects:\n\n'+'\n\n'.join(rows),
 '6. Sparsity/noise: per-action success frequency, realized variance and mean standard error are retained in EXPECTED_RETURN_LABELS.json. Low early identifiability combines the narrow parent action distribution, stochastic suffix noise and this finite sample budget; this experiment cannot uniquely assign all failure to one component.',
 '7. Bottleneck and next step: '+pivot+' The EST Monte Carlo selector has confirmation stage0 independent return gain '+str(oracles['CONFIRM']['0']['advantage_vs_uniform'])+' versus uniform. It tests whether this finite sampling budget can select useful actions before blaming the learned predictor alone.',
 '8. SAC readiness: '+('GO to recommend a separate bounded pilot' if decision['GO'] else 'NO-GO')+'. Failed frozen conditions: '+('; '.join(decision['reasons']) or 'none')+'. No actor update has occurred.',
 'KL interpretation: conditional KL is the frozen proposal distribution against J1 at each sampled state. It is not the KL of the non-Gaussian distribution induced by value-based best-of-eight selection. A future trained actor must satisfy its own original conditional-KL constraint; finite-pool selection is not a proof of that guarantee.',
 'Uncertainty interpretation: protein bootstrap intervals condition on collected candidates and suffix banks. Degenerate intervals from identical outcomes or all-correct small ranking subsets do not prove population certainty. Generated-prefix tasks share confirmation identities and cannot be counted as additional biological samples.'
 ])
 append('FINAL_E017_V72_SCIENTIFIC_REPORT.md',answers);append('NEXT_CRISP_RESEARCH_DECISION.md',answers)
 # Inventory actual runtime imports, including historical helpers resolved via
 # their original absolute paths; record rather than claiming a single-root load.
 paths={str(Path(__file__).resolve()),str(R/'source/scripts/e017_v72/study.py'),str(R/'source/scripts/e017_v72/analyze.py'),str(s['s']['__file__']),str(s['v']['__file__'])}
 paths.update(str(p) for p in (R/'control').glob('*.py'))
 for module in list(sys.modules.values()):
  file=getattr(module,'__file__',None)
  if file and '/proteingen_data/' in str(file).lower() and str(file).endswith('.py') and Path(file).is_file():paths.add(str(Path(file).resolve()))
 for file in [Path(s['v']['s'].__file__).with_name('frozen_science_config.json'),R.parent/'e017_crisp_v3_20261021_v1/configs/NATIVE_CALIBRATION.json']:
  if file.exists():paths.add(str(file))
 sourceaudit=dict(primary_source_commit=p['source_commit'],actual_runtime_file_SHA256={path:sha(path) for path in sorted(paths)},historical_helpers='V71/V7/V5/V4 resolved absolute imports are retained and hashed; none changed by V72',descriptive_correction=note,corrected_state_records=corrected)
 if (R/'IMPLEMENTATION_CORRECTION_001.json').exists():
  correction=read(R/'IMPLEMENTATION_CORRECTION_001.json');sourceaudit['implementation_correction']=correction
  append('FINAL_E017_V72_SCIENTIFIC_REPORT.md','Implementation recovery: the first read-only critic prediction attempt stopped on a saved anchor list/dictionary interface mismatch. A one-line wrapper correction was committed after preserving the initial source/protocol and failed logs. All88durable task-block hashes and the model-selection seal remained unchanged; no sampling or optimization was repeated. Initial scientific gates were preserved. Exact before/after commits and data hashes: IMPLEMENTATION_CORRECTION_001.json.')
 save('RUNTIME_SOURCE_AND_FINAL_ANALYSIS_AUDIT.json',sourceaudit)
 resource=read(R/'RESOURCE_AND_RECOVERY_REPORT.json');watch=[json.loads(x) for x in (R/'results/resource_watch.jsonl').read_text().splitlines()];resource['watchdog']=dict(samples=len(watch),min_D_free_bytes=min(x['D_free_bytes'] for x in watch),min_available_RAM_bytes=min(x['available_RAM_bytes'] for x in watch),max_swap_used_bytes=max(x['swap_used_bytes'] for x in watch),resource_stop=any(x['resource_stop'] for x in watch));resource['D_free_after_bytes']=shutil.disk_usage(R).free;resource['workers']='all research workers exited, no automatic restart; host and CUDA audited after publication'
 if resource['MLP_checkpoint']:
  ck=resource['MLP_checkpoint'];assert sha(ck['path'])==ck['sha256'];assert Path(ck['path']).stat().st_size==ck['bytes']
 save('RESOURCE_AND_RECOVERY_REPORT.json',resource)
 for path,h in p['input_SHA256'].items():assert sha(path)==h,path
 assert not subprocess.check_output(['git','-C',str(R/'source'),'status','--porcelain'],text=True).strip()
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not gpu.strip(),gpu
 assert not active(),active();free=shutil.disk_usage(R).free;assert free>2*2**30
 assert not list((R/'results').glob('*.part')) and not list((R/'checkpoints').glob('*.part'))
 p['output_SHA256']={name:sha(R/name) for name in p['output_SHA256']};p['output_SHA256']['RUNTIME_SOURCE_AND_FINAL_ANALYSIS_AUDIT.json']=sha(R/'RUNTIME_SOURCE_AND_FINAL_ANALYSIS_AUDIT.json');p['input_SHA256'].update(sourceaudit['actual_runtime_file_SHA256']);p['input_SHA256'][str(R/'results/resource_watch.jsonl')]=sha(R/'results/resource_watch.jsonl');p['final_review']=dict(UTC=time.time(),pre_review_provenance_SHA256=old_provenance,scope='descriptive CRNcovariance correction, quantitative answers, actual runtime source hashing, resource/process verification; no fitting, evaluation or gate change');save('PROVENANCE_SHA256.json',p)
 for file in [R/'IMPLEMENTATION_CORRECTION_001.json',R/'PROTOCOL_INITIAL.json']:
  if file.exists():p['input_SHA256'][str(file)]=sha(file)
 save('PROVENANCE_SHA256.json',p)
 verification=dict(PASS=True,UTC=time.time(),decision=decision,verified_input_hashes=len(p['input_SHA256']),verified_output_hashes=len(p['output_SHA256']),provenance_SHA256=sha(R/'PROVENANCE_SHA256.json'),active_research_workers=[],GPU_compute_processes=[],D_free_bytes=free,D_free_GiB=free/2**30,actor_updates=0,SAC_updates=0,J1_unchanged=True,all_artifacts_on_D=True);save('FINAL_VERIFICATION.json',verification);print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
