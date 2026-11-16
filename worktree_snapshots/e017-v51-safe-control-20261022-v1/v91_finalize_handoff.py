"""Seal V9.1 metadata and verify idle state; never train or evaluate a network."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1')
def read(n):return json.loads((R/n).read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def save(n,obj):
 path=R/n;tmp=path.with_name(path.name+'.part')
 with tmp.open('w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def write(n,text):
 path=R/n;tmp=path.with_name(path.name+'.part')
 with tmp.open('w') as f:f.write(text);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
assert read('PRODUCTION_COMPLETED.json')['updates']==100;assert read('LIVE_STATUS.json')['status']=='STOPPED';assert read('FINAL_EVALUATION_COMPLETED.json')['status']=='COMPLETED';assert read('FINALIZATION_STATUS.json')['decision']=='NO-GO'
subprocess.run(['python3','-B',str(Path(__file__).resolve().parent/'v91_resource_sample.py')],check=True);snap=json.loads((R/'logs/RESOURCE_SAMPLES.jsonl').read_text().splitlines()[-1]);assert not snap['research_workers'] and not snap['CUDA_processes'];assert snap['D_free_bytes']>2*2**30
command="$m=Get-CimInstance Win32_OperatingSystem; [PSCustomObject]@{UTC=(Get-Date).ToUniversalTime().ToString('o');TotalVisibleMemorySizeKiB=$m.TotalVisibleMemorySize;FreePhysicalMemoryKiB=$m.FreePhysicalMemory;ResearchProcesses=@(Get-Process | Where-Object {$_.ProcessName -match 'python|colab|protein|ollama'} | Select-Object Id,ProcessName,WorkingSet64)} | ConvertTo-Json -Depth 4 -Compress"
windows=json.loads(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command',command],text=True).lstrip('\ufeff'));idle=dict(status='ALL_RESEARCH_WORKERS_STOPPED',UTC=snap['UTC'],GPU=snap['GPU'],CUDA_processes=[],research_workers=[],automatic_restart=False,supervisor=None,Windows=windows,E007_E015_E012_active=False,ProteinMPNN_ColabFold_active=False,J1_unchanged=True,all_artifacts_cache_temporary_files_on_D=True,D_free_bytes=shutil.disk_usage(R).free,minimum_headroom_bytes=2*2**30);save('FINAL_WORKER_STATUS.json',idle)
comparison=read('GOAL_CONDITIONING_COMPARISONS.json')
for stats in list(comparison['models'].values())+list(comparison['DEV_milestones'].values()):
 if 'sequence_repairs' in stats:stats['geometry_finite_grid_repairs']=stats.pop('sequence_repairs')
save('GOAL_CONDITIONING_COMPARISONS.json',comparison)
old=read('PROVENANCE_SHA256.json');old_analysis_commit=old['source_commit'];final_source=subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip();assert not subprocess.check_output(['git','-C',str(R/'source'),'status','--porcelain'],text=True).strip()
ck=read('CHECKPOINT_AND_RECOVERY_MANIFEST.json');ck.update(source_commit=final_source,numeric_analysis_execution_commit=old_analysis_commit,final_documentation_change='Only the geometry finite-grid repair metadata key was renamed after numeric analysis; results unchanged.',final_worker_status='FINAL_WORKER_STATUS.json',source_SHA256={p.name:sha(p) for p in (R/'source/scripts/e017_v91').glob('*.py')},recovery_test_scope='Exact restore and next on-policy update/model/Adam/scheduler/RNG match verified. Multi-update journal recovery is implemented with per-step digests; no additional optimizer replay was executed after the scientific stop. Partial publications fail closed.');save('CHECKPOINT_AND_RECOVERY_MANIFEST.json',ck)
resources=read('RESOURCE_AND_STORAGE_REPORT.json');resources.update(final_worker_status=idle,D_free_bytes=idle['D_free_bytes'],D_free_GiB=idle['D_free_bytes']/2**30,Windows=windows,all_model_outputs_preserved=True,deleted_checkpoints=0);seen=set();allocated=0
for path in R.rglob('*'):
 if path.is_file() and not path.is_symlink():
  s=path.stat();key=(s.st_dev,s.st_ino)
  if key not in seen:seen.add(key);allocated+=s.st_blocks*512
resources['experiment_allocated_bytes_including_worktree_inode_deduplicated']=allocated;save('RESOURCE_AND_STORAGE_REPORT.json',resources)
controls=comparison['controls'];table='| Model | Goal-on motif hits | Goal-off motif hits | Reversed-request motif hits | Ordered utility direction [95% CI] |\n|---|---:|---:|---:|---|\n'
for model,z in controls.items():
 v=z['ordered_utility_direction'];table+=f"| {model} | {z['goal_on_motif_hits']}/32 | {z['goal_off_motif_hits']}/32 | {z['order_swap_requested_motif_hits']}/32 | {v['estimate']:.6f} {v['CI95']} |\n"
write('GOAL_ON_OFF_AND_ORDER_SENSITIVITY.md','''# Instruction sensitivity versus property steering
All controls use32 independent goal identities and the original candidate0 random stream, with one candidate per goal. Goal-off removes motif/region information and feedback while retaining the fixed N64 horizon through the observed state. Order swap reverses only the requested motif, updates its semantic signature and keeps paired seeds. It does not supply native future data. Both ordered motifs may coexist in an8-residue region; direction metrics therefore quantify requested-versus-reverse preference, not mutually exclusive instruction classes.

'''+table+'''
Every complete-property success count was zero in these controls. Controlled engineering learning demonstrated correct ordered decisions on feasible TRAIN fixtures, but these full-generation controls do not establish reliable held-out property steering. C is repaired but untrained; D is the last-safe update50 descriptive model, not the selected promoted model. Exact differences and paired uncertainty are in GOAL_CONDITIONING_COMPARISONS.json.
''')
write('MODEL_AND_METRIC_DEFINITIONS.md','''# Matched model and metric definitions
A_J1 is frozen J1 alone. B_PREV9 uses the verified V3 selected10000 actor weights with the original V9 property adapter, preserving the historical comparison. C_FIXED uses exactly the same weights with the canonical repaired adapter and no RL. D_AFTER_RL is update50, the predeclared last-safe trained descriptive checkpoint. D_SELECTED retained update0, hence is identical to C by construction. V9+145 weights are not used. Goal-off C additionally measures the same pre-V9 actor without property inputs.

Full success requires original hard sequence/backbone/packing validity, ordered contiguous motif and all exposure/burial/contrast constraints. K1 has32 independent task outcomes. K8 reports32 task pools of8 candidates (256 attempts), selected using exact validity and joint property scoring; no failed sample is replaced. Motif counts alone are hard-instruction component outcomes, not complete-property success. Independent identity bootstrap retains all candidates for each sampled identity; exact marginal zero-event intervals remain nonzero. No comparisons to native-endpoint H4 studies are treated as matched baselines.

Conditional geometry KL is averaged per active geometric component at the same generated states, with AA-weighted conditional Gaussians; AA categorical KL is separate. Geometry finite-grid repair counts are not sequence edits. Structural diversity uses pairwise proper-rotation Kabsch C-alpha RMSD; sequence identity is an aligned bounded diversity diagnostic, not corpus-wide homology or novelty certification. All mean metrics and completeness/failure denominators are retained in JSON.
''')
packing=read('SCIENTIFIC_STOP_DECOMPOSITION.json');limit=read('REWARD_SPECIFICATION.json')['packing_limits']['64'][1]
write('VALIDITY_AND_FEASIBILITY_LIMITATIONS.md',f'''# Interpreting the stop and action bounds
At update100, all-hard validity lost4/64 candidates relative to C (−6.25pp), triggering the frozen5pp rule. Backbone validity lost3/64 (−4.6875pp) and did not independently cross that allowance; KL also did not fail. The paired8-identity hard-validity difference interval is{packing['paired_hard_validity_change_CI95']}. It is wide, so the observed stop does not establish population-wide degradation. It remains binding and was not loosened after outcomes.

One backbone-valid update100 candidate had long-range contact degree7.0, above the frozen native-window upper bound{limit}; its sequence passed. This is a benchmark packing violation, not proof of an experimentally impossible fold. The threshold came from isolated native64-residue windows and was not retuned to this candidate.

The original engineering fixture separately exposed a real local actor limitation: if the frozen reference top logit exceeds the requested AA by>=6, an actor residual bounded to(−3,+3) cannot make that AA greedy. Its stochastic probability remains positive. Prefix changes can alter the reference distribution, so this local bound does not prove that the de novo task or long-horizon corrections are impossible. The eligible fixture required a margin before learning and passed without changing action supports or success criteria.

No high-level policy, reachability predictor or SAC critic was trained. More updates or relaxed constraints are not justified by this pilot. A future independently authorized feasibility/correction diagnostic should distinguish instruction compliance, packing validity and regional exposure control under matched sampling budgets.
''')
# Provide compact examples from existing evidence; never generate replacements.
examples=[]
for policy in ['C_FIXED','D_AFTER_RL']:
 for path in sorted((R/'results').glob('CONFIRM_'+policy+'_[0-9][0-9][0-9].json')):
  for row in json.loads(path.read_text())['records']:
   if row['metrics']['hard_valid']:
    examples.append(dict(policy=policy,source=str(path.relative_to(R)),source_SHA256=sha(path),candidate_index=row['candidate_index'],seed=row['seed'],sequence=row['sequence'],goal_id=row['goal_id'],full_success=row['metrics']['success'],properties={k:row['metrics'][k] for k in ['exposed_mean','buried_mean','contrast','motif_utility','reward']},CA_reconstruction='Use original compact geometry actions and unchanged typed CA builder from an empty state; no native coordinates.'));break
  if sum(v['policy']==policy for v in examples)==2:break
save('GENERATED_CANDIDATE_EXAMPLES.json',dict(examples=examples,all_from_existing_frozen_evaluations=True,none_promoted=True,not_foldability_or_function_evidence=True))
main=R/'FINAL_E017_V91_SCIENTIFIC_REPORT.md';report=main.read_text();report=report.replace('Every confirmation row has32 independent goal/protein identities','Model definitions: A is frozen J1; B is the verified pre-V9 V3 actor under the legacy V9 adapter; C has the same weights under the repaired adapter; D is the last-safe update50 trained model. See MODEL_AND_METRIC_DEFINITIONS.md.\n\nEvery confirmation row has32 independent goal/protein identities');report=report.replace('Source commit'+old_analysis_commit,'Final source commit '+final_source+' (numeric analysis execution '+old_analysis_commit+')');report+='\nFinal integrity: all research workers exited, no CUDA compute owner remains, original J1/V3/V9 files retain verified hashes, and all rollout/checkpoint evidence is preserved. Current storage and host process evidence are in FINAL_WORKER_STATUS.json. See VALIDITY_AND_FEASIBILITY_LIMITATIONS.md for the packing-specific failure and broad paired uncertainty.\n';write(main.name,report)
status=read('FINALIZATION_STATUS.json');status.update(status='SEALED_NO_GO',workers_stopped=True,worker_idle_verification_pending=False,all_required_artifacts_present=True,representation_repair='PASS',feasible_controllability='PASS_23_updates',production_updates=100,production_trajectories=800,production_residue_decisions=51200,RL_promotion='NO_GO',high_level_training='NOT_AUTHORIZED_OR_RUN',automatic_restart=False);save('FINALIZATION_STATUS.json',status)
# Archive publisher controls in D: evidence; no scientific artifact is copied twice.
for name in ['v91_resource_sample.py','v91_finalize_handoff.py']:
 src=Path(__file__).resolve().parent/name;dest=R/'control'/name
 if not dest.exists():shutil.copyfile(src,dest)
changed={'GOAL_CONDITIONING_COMPARISONS.json','CHECKPOINT_AND_RECOVERY_MANIFEST.json','RESOURCE_AND_STORAGE_REPORT.json','FINAL_E017_V91_SCIENTIFIC_REPORT.md','FINALIZATION_STATUS.json','logs/RESOURCE_SAMPLES.jsonl'};mismatch=[]
for name,expected in old['artifact_SHA256'].items():
 if name in changed:continue
 if sha(R/name)!=expected:mismatch.append(name)
assert not mismatch,mismatch
required=['FINAL_E017_V91_SCIENTIFIC_REPORT.md','GOAL_ENCODING_DEFECT_ANALYSIS.md','ORDERED_GOAL_SCHEMA.md','GOAL_OBSERVABILITY_TESTS.md','CHECKPOINT_COMPATIBILITY.md','RL_PILOT_RESULTS.md','GOAL_CONDITIONING_COMPARISONS.json','CHECKPOINT_AND_RECOVERY_MANIFEST.json','RESOURCE_AND_STORAGE_REPORT.json','NEXT_CRISP_RESEARCH_DECISION.md','FINAL_WORKER_STATUS.json'];assert all((R/name).is_file() for name in required);assert not list((R/'checkpoints').rglob('*.part'));assert len(list((R/'rollouts').glob('RL_*.sha.json')))==100
manifest=old;manifest.update(source_commit=final_source,numeric_analysis_execution_commit=old_analysis_commit,source_SHA256={str(path.relative_to(R)):sha(path) for path in (R/'source/scripts/e017_v91').glob('*.py')},artifact_SHA256={str(path.relative_to(R)):sha(path) for path in R.rglob('*') if path.is_file() and path.name!='PROVENANCE_SHA256.json' and not path.name.endswith('.part') and 'runtime' not in path.parts and 'source' not in path.parts},seal_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),verified_mismatches=[],all_project_artifacts_on_D=True,final_decision='NO_GO',publication_control_sources={str(path.resolve()):sha(path) for path in Path(__file__).resolve().parent.glob('v91_*.py')});save('PROVENANCE_SHA256.json',manifest);assert shutil.disk_usage(R).free>2*2**30
print(json.dumps(dict(status='SEALED_NO_GO',observability_checks=1823,feasible_control_updates=23,RL_updates=100,trajectories=800,residue_decisions=51200,confirmation_goal_identities=32,K1_all_models='0/32',K8_all_models='0/32',hard_validity_stop='30/64 to26/64',D_free_bytes=shutil.disk_usage(R).free,workers_stopped=True,source_commit=final_source,latest_checkpoint=read('checkpoints/protein/RECOVERY_INDEX.json')['latest'])))
