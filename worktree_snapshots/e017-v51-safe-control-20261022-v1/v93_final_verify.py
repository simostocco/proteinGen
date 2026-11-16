from pathlib import Path
import ast,collections,datetime,hashlib,json,os,shutil,subprocess,time
r=Path('/mnt/d/Simone/proteinGen_data/e017_v93_regional_controllability_20261024_v1');v92=r.parent/'e017_v92_prefix_feasibility_20261024_v1';v91=r.parent/'e017_v91_ordered_goal_recovery_20261023_v1';read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def save(path,obj):
 tmp=path.with_name(path.name+'.part')
 with tmp.open('w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
required=['FINAL_E017_V93_SCIENTIFIC_REPORT.md','REGIONAL_PROPERTY_CONTROLLABILITY.md','FIXED_SEQUENCE_BASELINES.md','DELAYED_ACTION_EFFECTS.md','FILTERED_POLICY_DENSITY_AND_GRADIENTS.md','POLICY_GRADIENT_CORRECTNESS_TESTS.md','PROSPECTIVE_GOAL_PROTOCOL.md','SUBSTANTIAL_RL_TRAINING_PLAN.md','STORAGE_AND_RECOVERY_ADMISSION.md','NEXT_CRISP_RESEARCH_DECISION.md'];assert all((r/n).exists() for n in required)
assert read(r/'COMPLETED_train.json')['PASS'] and read(r/'COMPLETED_validation.json')['PASS'];assert read(r/'POLICY_GRADIENT_TESTS.json')['PASS'] and read(r/'PROTEIN_TRACE_BACKWARD_TEST.json')['PASS'] and read(r/'KL_GRADIENT_AND_SCHEMA_TESTS.json')['PASS'];assert not list(r.rglob('*.part'))
for p in (r/'source/scripts/e017_v93').glob('*.py'):ast.parse(p.read_text())
assert subprocess.check_output(['git','status','--porcelain'],cwd=r/'source',text=True)==''
# Verify every sealed historical V92 artifact/source, not any TEST dataset.
prior=read(v92/'PROVENANCE_SHA256.json');historical={}
for section in ['source_SHA256','artifact_SHA256']:
 for name,expected in prior[section].items():
  p=v92/name;assert sha(p)==expected,('historical V92 changed',str(p));historical[str(p)]=expected
for key in ['J1','parent']:
 rec=prior['external_model_references'][key];p=Path(rec['path']);assert sha(p)==rec['SHA256'];historical[str(p)]=rec['SHA256']
index=read(v91/'checkpoints/protein/RECOVERY_INDEX.json')
for u in ['0','50','100']:
 q=index['owned'][u];p=Path(q['path']);assert sha(p)==q['sha256'];historical[str(p)]=q['sha256']
prior91=read(v91/'PROVENANCE_SHA256.json')
for section in ['source_SHA256','artifact_SHA256']:
 for name,expected in prior91[section].items():
  if Path(name).name in ['FINAL_E017_V91_SCIENTIFIC_REPORT.md','REWARD_SPECIFICATION.json','goals.py','policy.py','core.py']:
   path=Path(name) if Path(name).is_absolute() else v91/name;assert sha(path)==expected;historical[str(path)]=expected
for name,expected in prior91['protected_V9_files'].items():assert sha(Path(name))==expected;historical[name]=expected
for p in (r/'results').glob('*.sha.json'):assert sha(p.with_name(p.name.replace('.sha.json','.json')))==read(p)['sha256']
# Authoritative TRAIN/VALIDATION reference membership and source bindings.
v9=r.parent/'e017_v9_property_steering_20261023_v1';train=read(v9/'manifests/TRAIN_REFERENCE.json')['records'];val=read(v9/'manifests/VALIDATION_REFERENCE.json')['records'];tr={q['identity']:q for q in train};va={q['identity']:q for q in val};dev=read(r/'manifests/TRAIN_TASKS.json')['tasks'];pros=read(r/'manifests/PROSPECTIVE_FIXED.json')['tasks'];pool=read(r/'manifests/FUTURE_FIXED_TRAIN_POOL.json')['tasks']
assert not {t['goal']['protected_group'] for t in pros}&{q['group'] for q in train}
assert not {t['goal']['protected_group'] for t in pool}&{t['goal']['protected_group'] for t in dev}
for t in pool:assert t['source_reference_identity'] in tr and t['fixed_sequence']==tr[t['source_reference_identity']]['sequence'] and t['source_reference']==str(v9/'manifests/TRAIN_REFERENCE.json')
for t in pros:assert t['source_reference_identity'] in va and t['fixed_sequence']==va[t['source_reference_identity']]['sequence']
# Reconstruct and replay each main baseline through unchanged CPU builder.
command=[str(r/'run.sh'),'-c',"import runpy,json;from pathlib import Path;r=Path('/mnt/d/Simone/proteinGen_data/e017_v93_regional_controllability_20261024_v1');d=runpy.run_path(str(r/'source/scripts/e017_v93/common.py'));n=0\nfor p in sorted((r/'results').glob('*_[ABC]_[0-9][0-9][0-9].json')):\n x=d['read'](p)\n for row in x['records']:\n  state=d['reconstruct'](row);assert d['c']['score'](row['sequence'],state.ca.numpy(),x['task']['goal'],d['SPEC'])==row['metrics'];n+=1\nd['save']('SCORE_REPLAY_VERIFICATION.json',dict(PASS=True,baseline_attempts_replayed=n,original_builder=True,CPU_only=True))\nprint('REPLAY',n)"]
subprocess.run(command,check=True)
# Additional categorical mask contract before publication release.
assert read(r/'CATEGORICAL_MASK_TEST.json')['PASS'];assert read(r/'SAMPLING_REPLAY_VERIFICATION.json')['PASS'];assert read(r/'COMPLETE_CHECKPOINT_SIZE_MEASUREMENT.json')['PASS']
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip();assert not gpu
lines=subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True).splitlines();active=[s for s in lines if 'python' in s and any(k in s for k in ['/scripts/e017_','v93_validation_launch.py','v93_monitor.py','e007','e015','e012','ProteinMPNN','colabfold'])];assert not active,active
windows=subprocess.run(['powershell.exe','-NoProfile','-Command',"$o=Get-CimInstance Win32_OperatingSystem; @{TotalRAMKiB=$o.TotalVisibleMemorySize;FreeRAMKiB=$o.FreePhysicalMemory;Processes=@(Get-Process | Where-Object {$_.ProcessName -match 'python|ProteinMPNN|colabfold|vmmem'} | Select-Object ProcessName,Id,WorkingSet64)} | ConvertTo-Json -Depth 4 -Compress"],capture_output=True,text=True,timeout=30)
try:host=json.loads(windows.stdout)
except ValueError:host=dict(stdout=windows.stdout,stderr=windows.stderr)
free=shutil.disk_usage(r).free;assert free>2*2**30;files=[p for p in r.rglob('*') if p.is_file() and 'source' not in p.relative_to(r).parts];allocation=sum(p.stat().st_blocks*512 for p in files);assert allocation<64*2**20,allocation
main=[read(p) for p in (r/'results').glob('*_[ABC]_[0-9][0-9][0-9].json')];mainrows=[z for g in main for z in g['records']];maincalls=collections.Counter()
for g in main:maincalls.update(g['calls'])
extra=read(r/'COMPLETED_train.json');counts=dict(main_generation_attempts=len(mainrows),main_complete=sum(q['metrics']['complete'] for q in mainrows),main_proposals=sum(len(q['proposals']) for q in mainrows),main_calls=dict(maincalls),remaining_TRAIN_execution=extra['counts'],remaining_TRAIN_calls=extra['calls'],matched_protein_gradient_replay=dict(J1_batches=64,actor_batches=64,J1_states=1024,actor_states=1024),failed_batch8_gradient_replay=dict(J1_batches=64,actor_batches=64,J1_states=512,actor_states=512),schema_failure_J1_batches=1,schema_failure_J1_states=8,sampling_replay=read(r/'SAMPLING_REPLAY_VERIFICATION.json'),engineering_toy_accepted_updates=250,engineering_toy_replay_executions=1,protein_optimizer_updates=0)
record=dict(PASS=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r/'source',text=True).strip(),historical_files_verified=len(historical),historical_checkpoints_and_artifacts_SHA256=historical,new_results_hashes_verified=True,baseline_scores_replayed=True,TRAIN_VALIDATION_group_exclusion_verified=True,new_protein_checkpoint_files=0,toy_checkpoint_bytes=(r/'results/TOY_RECOVERY.pt').stat().st_size,all_research_workers_exited=True,CUDA_processes=[],Windows=host,D_free_bytes=free,D_free_GiB=free/2**30,evidence_allocated_bytes=allocation,operational_reserve_bytes=2*2**30,counts=counts,peak_GPU_bytes=max(read(r/'COMPLETED_train.json')['peak_GPU_bytes'],read(r/'COMPLETED_validation.json')['peak_GPU_bytes']),all_project_artifacts_on_D=True)
save(r/'FINAL_RESOURCE_AND_INTEGRITY.json',record)
# Append the exact final idle/storage handoff before computing provenance.
p=r/'FINAL_E017_V93_SCIENTIFIC_REPORT.md';text=p.read_text();text+='\n\nFinal seal: all research workers exited; no CUDA compute process remains. Historical model/artifact hashes and all main baseline score replays passed. D: free '+str(free)+' bytes ('+f'{free/2**30:.3f}'+'GiB); compact evidence allocation '+str(allocation)+' bytes, excluding source worktree. No protein optimizer update or new protein model checkpoint. See FINAL_RESOURCE_AND_INTEGRITY.json.\n';p.write_text(text)
provenance={str(p.relative_to(r)):sha(p) for p in sorted(r.rglob('*')) if p.is_file() and 'source' not in p.relative_to(r).parts and p.name!='PROVENANCE_SHA256.json'};sources={str(p.relative_to(r)):sha(p) for p in sorted((r/'source/scripts/e017_v93').glob('*.py'))};save(r/'PROVENANCE_SHA256.json',dict(schema='E017_V93_FINAL_1',source_commit=record['source_commit'],source_SHA256=sources,artifact_SHA256=provenance,external_protected_SHA256=historical,required_deliverables=required+['PROVENANCE_SHA256.json'],all_required_present=True,no_self_hash=True))
print(json.dumps(dict(PASS=True,D_free_bytes=free,evidence_allocated_bytes=allocation,counts=counts,artifacts=len(provenance))),flush=True)
