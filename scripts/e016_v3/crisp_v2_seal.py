from crisp_v2_common import *
import subprocess,sys,importlib.metadata,time
required=['FINAL_E017_CRISP_V2_HANDOFF.md','AUDIT.json','TASK_DATASET_AND_GOAL_PROTOCOL.md','DIRECT_STEERING_ARCHITECTURE.md','HIERARCHICAL_MDP_AND_ACTION_SEMANTICS.md','MATHEMATICAL_REWARD_SPEC.md','FEASIBILITY_RESULTS.md','FEASIBILITY_COMPARISON.json','DIRECT_STEERING_TESTS.json','CPU_TASK_AND_LEAKAGE_TESTS.json','REWARD_AND_EXTENDED_CONTROL_TESTS.json','HIGH_LEVEL_INTERFACE_TESTS.json','COUNTERFACTUAL_AND_REACHABILITY_DIAGNOSTICS.json','SHORT_HORIZON_INTERVENTION_PROBE.json','CHECKPOINT_MANIFEST.json','FAILURES_LEAKAGE_AND_LIMITATIONS.md','NEXT_STAGE_RL_PLAN.md','DEPENDENCIES.json','RESOURCE_REPORT.json','EXPERIMENT_DECISIONS.md','CHANGELOG.md']
for n in required:assert (R/n).is_file()and(R/n).stat().st_size>0,n
for n in ['DIRECT_STEERING_TESTS.json','CPU_TASK_AND_LEAKAGE_TESTS.json','REWARD_AND_EXTENDED_CONTROL_TESTS.json','HIGH_LEVEL_INTERFACE_TESTS.json']:assert read(n)['all_pass'],n
f=read('FEASIBILITY_COMPARISON.json');assert sum(v['attempts']for v in f['summary'].values())==192;assert f['J1_frozen_after'];assert read('REWARD_AND_EXTENDED_CONTROL_TESTS.json')['J1_unchanged'];assert read('RESOURCE_REPORT.json')['J1_optimizer_updates']==0
checks=[]
for cp in read('AUDIT.json')['checkpoints']:
 actual=sha(cp['path']);assert actual==cp['sha256'];checks.append({**cp,'verification':'UNCHANGED'})
for name,expected in [('PROPOSAL_MANIFEST.json',read('AUDIT.json')['dataset_manifest_sha256']),('SEQUENCE_ASSIGNMENTS.parquet',read('AUDIT.json')['sequence_manifest_sha256'])]:assert sha(P/name)==expected;checks.append(dict(path=str(P/name),sha256=expected,verification='UNCHANGED'))
for r in read('CHECKPOINT_MANIFEST.json')['checkpoints']:assert sha(r['path'])==r['sha256']
# Reopen the full-state recovery and verify its immutable inputs/source bindings.
manifest=read('CHECKPOINT_MANIFEST.json');q=torch.load(manifest['latest_adapter_recovery'],map_location='cpu',weights_only=False);assert q['update']==80 and len(q['sample_indices'])==1280
for name,h in q['input_sha256'].items():assert sha(R/name)==h
for name,h in q['source_file_sha256'].items():assert sha(R/'source/scripts/e016_v3'/name)==h,name
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=R/'source',text=True)
# Exact installed dependencies without dependency resolution/download.
freeze=subprocess.check_output([sys.executable,'-m','pip','list','--format=json','--disable-pip-version-check'],text=True);write('INSTALLED_DEPENDENCIES.json',json.loads(freeze))
rr=read('RESOURCE_REPORT.json');rr['source_commit']=commit;rr['final_D_free_bytes']=shutil.disk_usage(R).free;rr['inference_workers_stopped']=True;rr['final_source_clean']=True;write('RESOURCE_REPORT.json',rr);guard()
# Explicit source roles for historical versus final correctness code.
write('SOURCE_EXECUTION_PROVENANCE.json',dict(source_commit=commit,branch='e017-crisp-v2-20261021-v1',base='b3af952083099520e3fbebdda97130ad05f1b098',training_checkpoint_base_commit=q['source_commit'],training_code_note='adapter fit occurred in isolated uncommitted implementation; checkpoint carries base revision plus exact final recovery source-file SHAs; code changes after fit affected diagnostics/control bookkeeping, not fitted adapter tensors',exact_comparison_core='results/FEASIBILITY_EXECUTED_CORE.py',exact_comparison_evaluator='results/FEASIBILITY_EXECUTED_EVALUATE.py',comparison_policy_no_reward_updates=True,original_failed_rewind_test='results/INITIAL_DIRECT_STEERING_TESTS.json',low_level_policy_unchanged_by_reward_ledger='REWARD_AND_EXTENDED_CONTROL_TESTS.json: matched direct-arm sequences PASS',terminal_clock_disclosure='comparison gamma^T diagnostic; finalized event reward gamma^(T-1), all comparison successes0',no_historical_E016_mutations=True))
files=[]
for p in R.rglob('*'):
 if not p.is_file()or any(x in p.relative_to(R).parts for x in ['source','runtime']):continue
 if p.name in ['PROVENANCE_SHA256.json','PROVENANCE_VERIFICATION.json']:continue
 files.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)))
code=[dict(path=str(p.relative_to(R/'source')),sha256=sha(p))for p in sorted((R/'source/scripts/e016_v3').glob('crisp_v2_*.py'))]
write('PROVENANCE_SHA256.json',dict(version='E017_CRISP_V2_bounded_feasibility1',source_commit=commit,files=files,code=code,protected_inputs=checks,required_outputs=required,no_J1_updates=True,no_RL_updates=True))
for p in files:assert sha(R/p['path'])==p['sha256']
write('PROVENANCE_VERIFICATION.json',dict(status='PASS',source_commit=commit,source_clean=True,evidence_files=len(files),source_scripts=len(code),required_outputs=len(required),all_correctness_gates_pass=True,adapter_unique_updates=80,J1_updates=0,RL_updates=0,feasibility_attempts=192,complete_task_successes=0,administrative_truncations=64,D_free_bytes=shutil.disk_usage(R).free,recovery_checkpoint=manifest['latest_adapter_recovery'],recovery_sha256=sha(manifest['latest_adapter_recovery']),stop_decision='STOP bounded feasibility; no prolonged SAC, no J1 fine-tuning, no learned intervention/reachability fitting'))
print('SEALED',commit,'files',len(files),'source',len(code),'DfreeGiB',shutil.disk_usage(R).free/2**30,'recoverySHA',sha(manifest['latest_adapter_recovery']),flush=True)
