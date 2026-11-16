"""Seal minimal V9 handoff metadata after all research workers have exited."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1')
def read(n):return json.loads((R/n).read_text())
def save(n,x):
 p=R/n;q=p.with_name(p.name+'.part')
 with q.open('w') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
 return h.hexdigest()
def write(n,s):
 p=R/n;q=p.with_name(p.name+'.part')
 with q.open('w') as f:f.write(s);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
assert read('SCIENTIFIC_INTEGRITY_STOP.json')['all_V9_actors_quarantined_from_promotion'];assert read('JOINT_COMPLETED.json')['updates']==145;assert read('JOINT_LIVE_STATUS.json')['status']=='STOPPED'
subprocess.run(['python3','-B',str(Path(__file__).resolve().parent/'v9_resource_sample.py')],check=True)
snap=json.loads((R/'logs/RESOURCE_SAMPLES.jsonl').read_text().splitlines()[-1]);assert not snap['CUDA_processes'] and not snap['research_workers'];assert snap['D_free_bytes']>2*2**30
ps_code="$m=Get-CimInstance Win32_OperatingSystem; [PSCustomObject]@{UTC=(Get-Date).ToUniversalTime().ToString('o');TotalVisibleMemorySizeKiB=$m.TotalVisibleMemorySize;FreePhysicalMemoryKiB=$m.FreePhysicalMemory;ResearchProcesses=@(Get-Process | Where-Object {$_.ProcessName -match 'python|colab|protein|ollama'} | Select-Object Id,ProcessName,WorkingSet64)} | ConvertTo-Json -Depth 4 -Compress"
win=json.loads(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command',ps_code],text=True).lstrip('\ufeff'))
software_code="import sys,json,torch,numpy,scipy,gemmi; print(json.dumps(dict(python=sys.version,torch=torch.__version__,compiled_CUDA=torch.version.cuda,numpy=numpy.__version__,scipy=scipy.__version__,gemmi=gemmi.__version__,production_neural_precision='FP32',geometry_and_density_precision='FP64',production_torch_deterministic=True,AMP=False,CPU_threads=1)))"
software=json.loads(subprocess.check_output([str(R/'run.sh'),'-c',software_code],text=True));save('SOFTWARE_AND_EXECUTION_CONFIGURATION.json',software)
state=dict(status='ALL_RESEARCH_WORKERS_STOPPED',UTC=snap['UTC'],GPU=snap['GPU'],CUDA_processes=[],research_workers=[],Windows=win,supervisor=None,automatic_restart=False,E007_E015_E012_active=False,ProteinMPNN_ColabFold_active=False,scientific_integrity_stop=True,further_training_authorized_by_this_study=False,all_project_artifacts_on_D=True,D_free_bytes=shutil.disk_usage(R).free,minimum_headroom_bytes=2*2**30);save('FINAL_WORKER_STATUS.json',state)
ck=read('CHECKPOINT_AND_RECOVERY_MANIFEST.json');ck['all_V9_models_quarantined_from_promotion']=True;ck['scientific_integrity_stop']=read('SCIENTIFIC_INTEGRITY_STOP.json');ck['checkpoint_stop_reason_preserved']='COOPERATIVE_PAUSE';ck['actual_training_implementation']=read('PRODUCTION_IMPLEMENTATION.json');ck['software']=software;save('CHECKPOINT_AND_RECOVERY_MANIFEST.json',ck)
resource=read('RESOURCE_AND_RUNTIME_REPORT.json');resource['final_worker_status']=state;resource['Windows_RAM_check']=win;resource['D_free_bytes']=state['D_free_bytes'];resource['D_free_GiB']=state['D_free_bytes']/2**30;resource['software']=software;save('RESOURCE_AND_RUNTIME_REPORT.json',resource)
# Publish useful candidate examples from existing frozen records only.
examples=[]
for f in sorted((R/'results').glob('TRAIN_B_V3_[0-9][0-9][0-9].json')):
 for r in json.loads(f.read_text())['records']:
  if r['metrics']['hard_valid']:
   examples.append(dict(source=str(f.relative_to(R)),candidate_index=r['candidate_index'],seed=r['seed'],sequence=r['sequence'],goal_id=r['goal_id'],full_success=r['metrics']['success'],reward=r['metrics']['reward'],exposed_mean=r['metrics']['exposed_mean'],buried_mean=r['metrics']['buried_mean'],motif_utility=r['metrics']['motif_utility'],geometry_reconstruction='Use the compact geometry actions in the referenced source with the unchanged CA builder; no native coordinates required.'))
   break
 if len(examples)==3:break
save('GENERATED_CANDIDATE_EXAMPLES.json',dict(examples=examples,all_from_existing_evidence=True,none_claimed_full_property_success=True,not_folding_or_function_validation=True))
trace=[json.loads(s) for s in (R/'logs/JOINT_TRAIN.jsonl').read_text().splitlines()];positive=[r for r in trace if r['full_success']];assert sum(r['full_success'] for r in trace)==1
save('ON_POLICY_TRAJECTORY_PROVENANCE_LIMITATION.json',dict(production_trajectories=1160,full_successes=1,positive_update_records=positive,per_rollout_actions_persisted=False,what_is_retained='Per-update aggregate evaluated rewards/successes/validity/gradients, exact goal order and candidate seed derivation, full recovery checkpoints including actor/moments/RNG, immutable source/config/reward/goal hashes.',exact_successful_sequence_not_directly_exported=True,reconstruction='Would require deterministic training replay from update100 through144, then replay the8 RL_145 candidates under the recovered pre-update145 actor. No such optimizer replay or generation was run after the integrity stop.',limitation='The successful candidate is an aggregate TRAIN event, not a published independently inspectable trajectory. It is not evidence of held-out improvement. Future runs must persist compact per-rollout actions, rewards, seeds and policy versions before optimizer publication.',frozen_baseline_and_milestone_candidate_records_complete=True))
write('GOAL_OBSERVABILITY_FAILURE.md','''# Confirmed ordered-motif encoding omission
The property adapter assigned both motif tokens the same RoPE position. A shared goal embedding plus self-attention is permutation-equivariant; the query output is invariant to swapping these goal tokens while context is unchanged. CPU forward checks on the preserved update145 actor gave EF/FE output differences no larger than5.96e-8 at the empty prefix and before the second exposed-region residue. Initial engineering tests missed this defect.

This is an early information omission, not proof that the whole task is unlearnable. After two region residues, motif-utility feedback can distinguish the orders: a controlled prefix changed that scalar from1 to0 and changed actor outputs. The defect creates an unintended information-gathering problem instead of directly conditioning all early decisions on the known ordered motif. It does not establish the cause of every observed failure.

The integrity concern triggered an existing cooperative pause while update145 was in progress. That update completed and its full checkpoint was atomically committed; no optimizer work was discarded. The CPU audit confirmed the concern after worker exit. The original checkpoint/stop record remains COOPERATIVE_PAUSE, and SCIENTIFIC_INTEGRITY_STOP.json records the reason separately. No original numerical/KL/validity gate was retroactively changed.

No adapter correction, additional training, factor ablation, confirmation generation, corrective-search campaign or high-level training followed. All V9 models are quarantined. The next implementation review should encode motif order losslessly using existing input capacity and require early-prefix forward and gradient sensitivity tests, while preserving the task, supports, architecture and safeguards. Any new study should begin from the immutable original parent under a newly sealed protocol rather than silently continuing this attempt.
''')
write('SCIENTIFIC_RESULT_AND_LIMITATIONS.md','''# Compact scientific handoff
The bounded joint pilot performed145 accepted actor updates:1,160 generated trajectories and74,240 residue decisions across32 TRAIN goal identities. One TRAIN rollout satisfied the complete computational task at update145. Because the policy changes during training,1/1,160 is an exposure aggregate, not a fixed-policy success estimate or evidence of learning. Its raw per-candidate actions were not directly exported; see ON_POLICY_TRAJECTORY_PROVENANCE_LIMITATION.json. No reconstruction involving optimizer updates was run after the stop.

The last fixed development evaluation was update100: K1=0/8 and K8=0/8, unchanged from the parent; backbone validity28/64 versus25/64, combined hard validity27/64 versus23/64. Those small-panel changes do not establish improved goal success. The current checkpoint145 has no new goal-success evaluation. Confirmation remains unexposed by V9 policy evaluation, and AA-only/geometry-only arms were blocked.

The scientific conclusion is NO-GO for promotion or hierarchical RL. This attempt is additionally confounded by a confirmed goal-adapter defect, so its outcome cannot be presented as a decisive failure of on-policy RL under correctly represented ordered goals. Correct and test goal observability before considering another bounded study. J1, original parent, reward thresholds, datasets and historical evidence remain unchanged.
''')
# Ensure the primary report makes the rare TRAIN event and provenance limitation visible.
p=R/'FINAL_E017_V9_SCIENTIFIC_REPORT.md';text=p.read_text();needle='## Integrity failure and interpretation';text=text.replace(needle,'One full computational success occurred among the1,160 changing-policy TRAIN rollouts at update145. This is not a fixed-policy or held-out improvement estimate. Its per-rollout actions were not directly exported; exact reconstruction would require training replay, which was not run after the stop. See ON_POLICY_TRAJECTORY_PROVENANCE_LIMITATION.json.\n\n'+needle);write(p.name,text)
final=read('FINALIZATION_STATUS.json');final.update(status='SEALED_NO_GO',workers_stopped=True,checkpoint_update=145,ordered_goal_encoding_integrity_stop=True,confirmation_unexposed=True,all_models_quarantined=True,automatic_restart=False,artifact_verification='all published checksums verified before seal');save('FINALIZATION_STATUS.json',final)
# Reuse previously checked checkpoint hashes; independently check them at this final publication.
manifest=read('PROVENANCE_SHA256.json');old=manifest['artifact_SHA256'];mismatches=[]
changed={'CHECKPOINT_AND_RECOVERY_MANIFEST.json','RESOURCE_AND_RUNTIME_REPORT.json','FINALIZATION_STATUS.json','FINAL_E017_V9_SCIENTIFIC_REPORT.md','logs/RESOURCE_SAMPLES.jsonl'}
for name,expected in old.items():
 if name in changed:continue
 path=R/name
 if not path.exists() or sha(path)!=expected:mismatches.append(name)
assert not mismatches,mismatches
manifest['artifact_SHA256']={str(path.relative_to(R)):sha(path) for path in R.rglob('*') if path.is_file() and path.name!='PROVENANCE_SHA256.json' and not path.name.endswith('.part') and 'runtime' not in path.parts and 'source' not in path.parts};manifest['seal_UTC']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());manifest['verification_mismatches']=[];manifest['all_V9_models_quarantined']=True;save('PROVENANCE_SHA256.json',manifest)
required=['FINAL_E017_V9_SCIENTIFIC_REPORT.md','PROPERTY_TASK_AND_GOAL_DISTRIBUTION.md','TRAIN_REFERENCE_AND_EXPOSURE_CALIBRATION.md','REWARD_VALIDITY_AND_EXPLOITATION_TESTS.md','FROZEN_DE_NOVO_BASELINES.md','LOW_LEVEL_RL_PILOT_AND_ABLATIONS.md','HIGH_LEVEL_MDP_AND_CORRECTION_ENVIRONMENT.md','CORRECTION_AND_SEARCH_BASELINE_COMPARISON.md','REACHABILITY_AND_COUNTERFACTUAL_INTERVENTION_DIAGNOSTICS.md','CHECKPOINT_AND_RECOVERY_MANIFEST.json','RESOURCE_AND_RUNTIME_REPORT.json','NEXT_HIERARCHICAL_RL_DECISION.md','PROVENANCE_SHA256.json','FINAL_WORKER_STATUS.json','GOAL_OBSERVABILITY_FAILURE.md']
assert all((R/name).is_file() for name in required);assert len(trace)==145 and [r['update'] for r in trace]==list(range(1,146));assert not list((R/'checkpoints').rglob('*.part'));assert not subprocess.check_output(['git','-C',str(R/'source'),'status','--porcelain'],text=True).strip();assert shutil.disk_usage(R).free>2*2**30
print(json.dumps(dict(status='SEALED_NO_GO',update=145,trajectories=1160,residue_decisions=74240,TRAIN_full_successes=1,DEV_K1='0/8',DEV_K8='0/8',D_free_bytes=shutil.disk_usage(R).free,checkpoint=read('checkpoints/JOINT/LATEST.json'),workers_stopped=True,checksums_verified=True)))
