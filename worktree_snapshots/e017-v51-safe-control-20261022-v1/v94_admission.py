"""Read-only production admission; publish blocked V94 evidence, no optimizer."""
from pathlib import Path
import collections,csv,datetime,hashlib,json,os,shutil,subprocess,time
BASE=Path('/mnt/d/Simone/proteinGen_data');V=BASE/'e017_v93_regional_controllability_20261024_v1';R=BASE/'e017_v94_fixed_sequence_geometry_rl_20261024_v1';E=Path('/mnt/e/Simone/proteinGen_data/e017_v94_fixed_sequence_geometry_rl_20261024_v1');read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def save(name,value):
 p=R/name;temp=p.with_name(p.name+'.part')
 with temp.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def md(name,text):(R/name).write_text(text.strip()+'\n')
assert not R.exists();assert shutil.disk_usage(BASE).free>2*2**30+4*2**20;R.mkdir();(R/'audit_source').mkdir();(R/'runtime').mkdir();shutil.copyfile(Path(__file__),R/'audit_source/v94_admission.py')
start=time.perf_counter();plan=read(V/'SUBSTANTIAL_CAMPAIGN_SPEC.json');prov=read(V/'PROVENANCE_SHA256.json');verified={}
# Verify the sealed V93 source and complete compact artifact set, no TEST reads.
for section in ['source_SHA256','artifact_SHA256']:
 for name,expected in prov[section].items():
  path=V/name;actual=sha(path);assert actual==expected,('V93 seal mismatch',str(path));verified[str(path)]=actual
source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=V/'source',text=True).strip();source_status=subprocess.check_output(['git','status','--porcelain'],cwd=V/'source',text=True);assert source_commit==prov['source_commit'] and not source_status
protected=prov['external_protected_SHA256'];parent=BASE/'e017_crisp_v3_throughput_20261021_v1/training_runs/run_d3e78a804534473884df42f0f34b7c41/checkpoints/checkpoint_00010000_7d8c5d05.pt';j1=BASE/'e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt';refs={}
for label,path,expected in [('J1',j1,plan['J1_SHA256']),('actor_initializer',parent,plan['parent_SHA256'])]:
 assert sha(path)==expected;refs[label]=dict(path=str(path),SHA256=expected,bytes=path.stat().st_size,verified_in_place=True,no_model_loaded_or_modified=True)
ref=BASE/'e017_v9_property_steering_20261023_v1/manifests/TRAIN_REFERENCE.json';train={r['identity']:r for r in read(ref)['records']};pool=read(V/'manifests/FUTURE_FIXED_TRAIN_POOL.json');dev=read(V/'manifests/TRAIN_TASKS.json')['tasks'];val=read(V/'manifests/PROSPECTIVE_FIXED.json')['tasks'];ids=set();groups=set();AA=set('ACDEFGHIKLMNPQRSTVWY');variants=collections.defaultdict(list)
for item in pool['tasks']:
 g=item['goal'];identity=item['source_reference_identity'];native=train[identity];seq=item['fixed_sequence'];a,z=g['exposed'];b,y=g['buried'];assert native['split']=='TRAIN';assert seq==native['sequence'] and len(seq)==64 and set(seq)<=AA;assert g['N']==64 and z-a==y-b==8 and abs(a-b)>=16;assert g['motif'] in seq[a:z] and len(g['motif'])==2;assert g['protected_group']==native['group'];assert item['source_reference']==str(ref);assert item['native_witness_success'] is True;ids.add(identity);groups.add(g['protected_group']);variants[identity].append(g['difficulty'])
assert len(ids)==len(groups)==434 and len(pool['tasks'])==868;assert all(sorted(v)==['relaxed','witness'] for v in variants.values());assert not groups&{x['goal']['protected_group'] for x in dev+val};assert len(dev)==16 and len({x['source_reference_identity'] for x in val})==16
reward=BASE/'e017_v91_ordered_goal_recovery_20261023_v1/REWARD_SPECIFICATION.json';assert sha(reward)=='989f5d83310b77a891b47e903ee5a36b96b956e7cd57cb2d8ef98e07df608f92';spec=read(reward)
# Windows actual volume identity/health, not a path-level free-space assumption.
ps="$o=Get-CimInstance Win32_OperatingSystem; @{RAM=@{TotalKiB=$o.TotalVisibleMemorySize;FreeKiB=$o.FreePhysicalMemory};Volumes=@(Get-Volume | Where-Object {$_.DriveLetter -in @('C','D','E')} | Select-Object DriveLetter,FileSystem,DriveType,HealthStatus,OperationalStatus,Size,SizeRemaining);Partitions=@(Get-Partition | Where-Object {$_.DriveLetter -in @('D','E')} | Select-Object DriveLetter,DiskNumber,PartitionNumber,Size);Disks=@(Get-Disk | Select-Object Number,FriendlyName,BusType,HealthStatus,OperationalStatus,Size,LargestFreeExtent);ResearchProcesses=@(Get-Process | Where-Object {$_.ProcessName -match 'python|ProteinMPNN|colabfold|vmmem'} | Select-Object ProcessName,Id,WorkingSet64)} | ConvertTo-Json -Depth 5 -Compress"
windows=json.loads(subprocess.check_output(['powershell.exe','-NoProfile','-Command',ps],text=True,timeout=40));vol={q['DriveLetter']:q for q in windows['Volumes']};disk={q['Number']:q for q in windows['Disks']};part={q['DriveLetter']:q for q in windows['Partitions']};d_free=shutil.disk_usage(BASE).free;e_free=shutil.disk_usage('/mnt/e').free;minimum=plan['minimum_free_bytes'];reserve=plan['operational_reserve_bytes'];checkpoint=plan['measured_checkpoint_bytes'];assert minimum==8*checkpoint+reserve+128*2**20+128*2**20+64*2**20+32*2**20;assert d_free<minimum;assert vol['E']['FileSystem']=='NTFS' and vol['E']['DriveType']=='Fixed' and vol['E']['HealthStatus']=='Healthy' and disk[part['E']['DiskNumber']]['BusType']=='NVMe';assert e_free>8*2**30
# Only small, already documented cache roots: no broad traversal, no deletions.
caches=[]
for name in ['e017_v93_regional_controllability_20261024_v1','e017_v92_prefix_feasibility_20261024_v1','e017_v91_ordered_goal_recovery_20261023_v1']:
 root=BASE/name/'runtime';files=[]
 for parentdir,dirs,names in os.walk(root,followlinks=False):
  dirs[:]=[n for n in dirs if not (Path(parentdir)/n).is_symlink()]
  for n in names:
   p=Path(parentdir)/n;stat=p.stat(follow_symlinks=False);files.append(dict(path=str(p),logical_bytes=stat.st_size,WSL_st_blocks_bytes=stat.st_blocks*512,SHA256=sha(p)));assert len(files)<=1000
 caches.append(dict(root=str(root),files=files,logical_bytes=sum(q['logical_bytes'] for q in files),WSL_st_blocks_bytes=sum(q['WSL_st_blocks_bytes'] for q in files),classification='only matplotlib font cache found; regenerable in principle, deletion not authorized or performed; WSL allocation is not an NTFS reclamation guarantee'))
cleanup=read(BASE/'e017_v51_safe_control_20261022_v1/recovery_storage_20261022/cleanup_plan.json');pastpaths=[Path(q.get('path',q.get('absolute_path',''))) for q in cleanup['files']];pastmissing=sum(not p.exists() for p in pastpaths if str(p)!='.')
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip();assert not gpu
processlines=subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True).splitlines();research=[line.strip() for line in processlines if 'python' in line and any(word in line for word in ['/scripts/e017_','v93_monitor.py','v93_validation_launch.py','ProteinMPNN','colabfold']) and 'powershell' not in line];assert not research,research
hardware=dict(GPU=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),CUDA_processes=[],WSL_RAM=subprocess.check_output(['free','-b'],text=True),Windows=windows,mounts=subprocess.check_output(['df','-B1'],text=True),research_processes=research)
allocation=sum(c['WSL_st_blocks_bytes'] for c in caches);deficit=minimum-d_free;prefer=8*2**30-d_free
record=dict(status='INFRASTRUCTURE_BLOCKED',stage='A2_STORAGE_ADMISSION',D_free_bytes=d_free,E_free_bytes=e_free,minimum_free_bytes=minimum,minimum_includes_operational_reserve=True,operational_reserve_bytes=reserve,D_deficit_bytes=deficit,D_preferred_8GiB_deficit_bytes=prefer,checkpoint_bytes=checkpoint,peak_checkpoint_images=8,D_capacity_GO=False,E_capacity_and_volume_health_GO=True,E_use_authorized=False,standing_user_constraint='All newly generated proteinGen artifacts must remain on D:',proposed_E_experiment_root=str(E),production_training_authorized_conditionally_by_current_request=True,storage_scope_exception_needed=True,artifacts_written_to_E=False,cache_candidates=caches,cache_WSL_blocks_total=allocation,past_wheel_cache_paths_already_absent=pastmissing,past_wheel_cache_paths=len(pastpaths),files_deleted=0,bytes_reclaimed=0,hardware=hardware,verified_model_references=refs,qualified_TRAIN_identities=434,qualified_TRAIN_protected_groups=434,qualified_TRAIN_task_variants=868,source_commit=source_commit,recovery_qualification='NOT_RUN_STORAGE_GATE',optimizer_updates=0,generation_attempts=0,complete_trajectories=0,elapsed_seconds=time.perf_counter()-start)
save('ADMISSION_AUDIT.json',record)
configuration=dict(plan,experiment='E017_V94',status='NOT_LAUNCHED',training_authorization='current user request authorizes5000 updates conditional on storage and recovery gates',historical_plan_authorized_field_retained_from_V93=plan['authorized'],launch_admitted=False,optimizer_updates=0,sampling_attempts=0,source_commit=source_commit,source_root=str(V/'source'),source_manifest_SHA256=sha(V/'PROVENANCE_SHA256.json'),TRAIN_task_manifest_SHA256=sha(V/'manifests/FUTURE_FIXED_TRAIN_POOL.json'),development_manifest_SHA256=sha(V/'manifests/TRAIN_TASKS.json'),prospective_manifest_SHA256=sha(V/'manifests/PROSPECTIVE_FIXED.json'),reward_SHA256=sha(reward),joint_AA_training=False,TEST=False)
save('TRAINING_CONFIGURATION_AND_SEEDS.json',configuration)
with (R/'RL_TRAINING_CURVES.csv').open('x',newline='') as f:csv.writer(f).writerow(['update','generation_attempts','complete_trajectories','accepted_geometric_actions','rejected_proposals','fallbacks','reward_mean','reward_SD','hard_valid','exposure_success','burial_success','full_success','proposal_KL','trace_KL','gradient_norm','learning_rate','elapsed_seconds','D_free_bytes'])
static=[dict(source=str(V/'source/scripts/e017_v93/campaign.py'),issue='checkpoint() repeatedly passes root/latest.pt to atomic_checkpoint(); that writer refuses an existing immutable destination. Unique checkpoint objects plus an atomic latest/best index must replace this release-candidate path before qualification.',category='static code finding, not an executed recovery failure'),dict(source=str(V/'source/scripts/e017_v93/campaign.py'),issue='resume loads the last checkpoint but does not replay/verify accepted UPDATES.jsonl entries newer than it. PENDING_ROLLOUT.json is overwritten each update; durable journal commitment and replay semantics remain unqualified.',category='static code finding, not an executed recovery failure')]
save('CHECKPOINT_RECOVERY_MANIFEST.json',dict(status='NOT_QUALIFIED_STORAGE_BLOCKED',new_checkpoints=[],latest_recovery=None,best_development=None,final_scientific_checkpoint=None,initializer=refs['actor_initializer'],J1=refs['J1'],source_commit=source_commit,checkpoint_size_receipt=str(V/'COMPLETE_CHECKPOINT_SIZE_MEASUREMENT.json'),checkpoint_size_receipt_SHA256=sha(V/'COMPLETE_CHECKPOINT_SIZE_MEASUREMENT.json'),required_qualification_cases=['uninterrupted versus resumed actor/Adam/scheduler/RNG/cursor/counters equality','crash after rollout publication before optimizer step','crash after accepted optimizer journal commit before rolling checkpoint','partial checkpoint write leaves previous verified index authoritative','hash corruption fails closed','latest/best/named retention with immutable images and atomic index','exact matching active behavior/replay batches16, correct abort/rejection traces'],static_findings=static,production_optimizer_steps=0,no_recovery_tests_executed=True))
md('PRODUCTION_STORAGE_ADMISSION.md',f'''# Production storage admission

**INFRASTRUCTURE BLOCKED — D: admission fails.**

|Item|Bytes|GiB|
|---|---:|---:|
|D: current free|{d_free:,}|{d_free/2**30:.6f}|
|Measured full recovery|{checkpoint:,}|{checkpoint/2**30:.6f}|
|8 checkpoint images|{8*checkpoint:,}|{8*checkpoint/2**30:.6f}|
|Operational reserve included in total|{reserve:,}|2.000000|
|Full campaign minimum|{minimum:,}|{minimum/2**30:.6f}|
|D: deficit to minimum|{deficit:,}|{deficit/2**30:.6f}|
|D: deficit to preferred8GiB|{prefer:,}|{prefer/2**30:.6f}|
|E: free|{e_free:,}|{e_free/2**30:.6f}|

The3.904GiB minimum ALREADY includes2GiB operational headroom. It also includes128MiB near growth,128MiB compact journals/telemetry,64MiB cache and32MiB reports. Do not add the reserve twice or treat merely exceeding2GiB as admission.

D: is healthy NTFS on SATA disk0, whose largest free unallocated extent is0. E: is healthy fixed NTFS on internal NVMe disk1 with{e_free/2**30:.2f}GiB genuinely free. It meets capacity/health criteria without cleanup. No E: directory or write probe was created because the standing user instruction requires all new proteinGen artifacts on D:. The current generic mounted-volume inspection request has not been treated as an implicit exception to that hard constraint.

Concrete preferred remediation: explicitly authorize new V9.4 artifacts/source/caches/temp/journals/checkpoints at `{E}`. Keep canonical models, datasets and historical evidence on D: read-only; reference hashes rather than copy them. Fork the release-candidate source into the new root, rebase all output guards and environment cache/temp paths to the explicitly authorized E: root, recheck free capacity, then fix/qualify production recovery before the already-authorized5000-update campaign. No new training permission is needed beyond the current conditional mission; only the volume exception is unresolved.

D:-only alternative: obtain at least{deficit:,} additional genuinely free bytes (prefer{prefer:,}) through a separately approved plan identifying exact affected paths, or add capacity. No automatic deletion, relocation, compression or dataset/model copying is authorized. Another directory on D: does not create free space; C:, WSL root and RAM disks remain excluded.

Bounded known-cache inventory found only3 matplotlib font-cache JSONs totaling{allocation:,} WSL allocation bytes, insufficient by orders of magnitude. These are not deleted; WSL st_blocks is not native NTFS reclaim proof. The previous54 wheel-cache files were already retired in the documented20261022 cleanup; current exact-path checks found{pastmissing}/{len(pastpaths)} absent. Do not count past reclaimed bytes a second time.

E010 neighbor exports are scientific per-decision evidence, not disposable cache; their18.22GiB opportunity remains subject to the separate preservation review. No protected checkpoint, dataset, manifest, source or historical evidence is a cleanup target.

No large fresh storage inventory or neural computation was run. Future new-run checkpoints may reference immutable inputs, use compact traces, bounded replay, same-run identical hardlinks when supported, and atomic publication; none of these savings justify silently reducing named-checkpoint retention or the approved reserve. Conservative admission assumes no hardlink savings.''')
md('OPTIMIZER_JOURNAL_RECOVERY_QUALIFICATION.md',f'''# Optimizer/journal recovery qualification

**NOT RUN — storage admission failed before optimizer creation.** V9.3's primitive/gradient/sampling/toy checks remain valid but do not constitute full production optimizer/journal qualification.

Static review identifies two release-candidate issues to repair after storage admission:

1. The driver reuses `latest.pt` with an immutable checkpoint writer that rejects an existing path. Use immutable update/hash filenames and atomically publish a validated latest/best/named index.
2. Resume does not implement replay/validation of accepted journal entries newer than the last checkpoint. The overwritten single pending-rollout file is insufficient as the journal design itself. Define durable per-update commit records, deterministic trace digests, optimizer/scheduler/cursor/RNG transaction boundaries and replay of post-checkpoint updates without duplicate commits or discarded accepted steps.

Required tests are recorded in CHECKPOINT_RECOVERY_MANIFEST.json. Compare uninterrupted versus controlled interrupted execution, all parameter/moment/scheduler/RNG/cursor/counter/config bindings, corrupt hashes, partially written staging, and abort/rejection replay at matching batch16. Perform no production launch until all pass. No synthetic release receipt or historical optimizer state is accepted as a substitute. These are infrastructure issues, not RL stability/performance findings.''')
md('TRAINING_STABILITY_AND_VALIDITY.md','''# Training stability and validity

NOT RUN:0 optimizer updates,0 generation attempts,0 complete trajectories. RL_TRAINING_CURVES.csv has a header only and no invented observations. No short performance pilot was substituted for the5000-update campaign.

Prospective plan remains V9.3:250/500/1000/2000/3500/5000 development monitoring, original numerical/step-KL/validity/diversity safety stops. K1=0 at an early milestone alone is not a stop. If training later destabilizes, classify RL STABILITY NO-GO; do not infer that adequate RL training cannot solve the task. Current outcome is INFRASTRUCTURE BLOCKED.''')
md('FILTERED_POLICY_DIAGNOSTICS.md',f'''# Filtered-policy contract audit

Source and sealed tests verified; no new rollout diagnostics executed. N64 canonical sequence supplied externally, with0 categorical score terms. Geometry conditions on the actual fixed AA. Active distance/angle/torsion masks begin at t>=1/2/3; transforms exp/pi-sigmoid/pi-tanh retain original support and finite-grid convention.

The controller allows at most8 proposals/state, accepts the first exact-prefix-valid extension and records deterministic incomplete failure if exhausted. OriginalN64 permits1 clash pair; a second violates2/100. Current Rg/contact averages are not treated as absorbing prefix failures. Original reward SHA256 `{sha(reward)}` is unchanged.

Correct stochastic score includes all reached raw Gaussian proposals, including rejections, with transformed Jacobians treated consistently. Prescribed AA is not an actor choice. Group16 leave-one-out baseline and trace-KL beta{plan['trace_KL_beta']} remain planned. Trace reference uses the identical filter/fallback contract; proposal KL is not mislabeled effective filtered-output KL. Unfiltered underlying actor and effective filtered controller must both be evaluated after an admitted substantial run. No current results are fabricated.''')
md('PROSPECTIVE_RL_EVALUATION.md','''# Prospective RL evaluation

NOT RUN: no trained model exists. Retain the frozen V9.3 fixed-sequence prospective32 goals/16 identities, two difficulty variants each, and independent TRAIN development procedure. No TEST access, adaptive target/reward changes or native target-coordinate inference.

Primary future comparison is trained filtered actor versus the immutable actor under the same finite filter. Require at least5pp K1 gain and positive paired95% lower interval, unchanged original hard-validity/KL/diversity safeguards and K8 not worse. Also compare unfiltered trained actor to unfiltered parent to isolate underlying proposal learning. Cluster shared goal variants/candidates by protein identity; report zero-event limitations. The V9.3 frozen21/256, K8=11/32, K1=0/32 are historical baselines, not V9.4 observations. Joint AA/geometry learning is outside this campaign.''')
md('RL_VS_SEARCH_COMPARISON.md','''# RL versus search

NOT RUN: V9.4 has no RL optimizer updates or generated trajectories. No RL advantage or failure is established.

Future A–G comparison retains J1, original actor, filtered parent, filtered parent BestofK, trained K1/K8, and a feasible frozen non-RL search control. Match and report actual actor-state calls, all accepted/rejected proposals, transitions, terminal evaluations and wall time. Exact terminal selection from a known candidate pool is inference search, not learned control. V9.3's4-TRAIN-task rewind result is contextual evidence, not a matched prospective V9.4 search baseline. Do not invent a replacement native-endpoint IK task.''')
md('NEXT_CRISP_RESEARCH_DECISION.md',f'''# Next CRISP decision

**INFRASTRUCTURE BLOCKED.** No RL learning/performance/stability verdict is possible; no optimizer update occurred. Environmental and likelihood primitive GO findings from V9.3 are preserved.

Preferred concrete next action: authorize E: as a scoped exception to the standing D:-only artifact rule, at `{E}`, with historical inputs read-only on D:. E: is healthy fixed NTFS/NVMe and has{e_free/2**30:.2f}GiB free. Then repair the identified immutable-checkpoint/journal issues, qualify full recovery with controlled interruptions, and execute the currently authorized5000×16 fixed-sequence geometry campaign when all gates pass.

If D:-only remains mandatory, free at least{deficit:,} bytes under explicit affected-path authorization, preferably restore8GiB free. No deletion proposal is approved by this audit. Do not repeat short protein performance pilots, add SAC/critics, or initiate joint/high-level RL. Do not classify a storage block as RL PERFORMANCE NO-GO.''')
md('FINAL_E017_V94_SCIENTIFIC_REPORT.md',f'''# E017 V9.4 production admission report

**Outcome: INFRASTRUCTURE BLOCKED at PhaseA2.** No production launch, optimizer creation/update, recovery qualification or new protein trajectory/evaluation occurred. The5000-update campaign remains conditionally authorized by the current mission; its storage and recovery gates are not satisfied.

D: free={d_free:,}bytes ({d_free/2**30:.3f}GiB), below the full{minimum:,}-byte ({minimum/2**30:.3f}GiB) requirement, which already includes2GiB operational reserve. E: is healthy fixed NTFS on an internal NVMe SSD with{e_free:,}bytes ({e_free/2**30:.3f}GiB) free. Its capacity is sufficient, but using it would require an explicit exception to the standing instruction that all newly generated proteinGen artifacts remain on D:. No redirection or E: write was attempted.

The compact V9.3 seal/source files and actual model-file hashes passed verification in place. Actor initializer is original V3 selected10000, SHA256 `{refs['actor_initializer']['SHA256']}`; permanently frozen J1 SHA256 `{refs['J1']['SHA256']}`. Source commit `{source_commit}`. No stopped V9.1 checkpoint was promoted or initialized.

The actual task manifest confirms434 eligible TRAIN identities/434 protected groups and868 witnessed fixed-sequence variants, with16 reserved TRAIN development identities and16 prospective VALIDATION identities excluded. Supplied AA sequences are canonical64 residues and already satisfy known motifs; geometric sampling is conditional on those actual AAs. Reward, supports, masks, finite filtering and full-trace likelihood rules remain unchanged. Verification uses the unchanged sealed qualification and current manifest/source bindings; no new native or generated evaluation was needed.

The full production recovery driver remains unqualified. Static review found reuse of immutable `latest.pt` and absent committed-journal replay after the last checkpoint; controlled crash/write/hash/restart tests must pass after storage admission. Existing primitive/toy/frozen-gradient checks do not release production.

Only tiny regenerable font-cache candidates were identified; they cannot bridge the storage deficit and were not deleted. Previously retired wheel caches are not new capacity. E010 exports and all historical evidence remain protected. No filesystem, partition, historical checkpoint, dataset, manifest or source was modified. All new audit files are on D: with more than2GiB reserve; curves contain no invented data.

The concrete E: plan and D:-only alternative are in PRODUCTION_STORAGE_ADMISSION.md. Scientific outcome is not RL PERFORMANCE NO-GO or RL STABILITY NO-GO: substantial protein RL has not run. No research worker or CUDA compute process is active. Stop here pending a storage decision; no high-level/joint AA/SAC/folding/TEST work is initiated.''')
save('INPUT_VERIFICATION.json',dict(PASS=True,V93_source_commit=source_commit,V93_source_clean=True,V93_sealed_files_verified=len(verified),verified_SHA256=verified,models=refs,TRAIN_manifest_SHA256=sha(V/'manifests/FUTURE_FIXED_TRAIN_POOL.json'),TRAIN_identities=434,TRAIN_groups=434,task_variants=868,development_and_validation_excluded=True,reward_SHA256=sha(reward),no_new_evaluation=True))
# Publish final free-space and idle receipt after all reports exist.
finalfree=shutil.disk_usage(R).free;assert finalfree>reserve;save('FINAL_IDLE_STATUS.json',dict(status='INFRASTRUCTURE_BLOCKED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),D_free_bytes=finalfree,E_free_bytes=e_free,CUDA_compute_processes=[],research_workers=[],optimizer_updates=0,generation_attempts=0,complete_trajectories=0,files_deleted=0,bytes_reclaimed=0,all_new_artifacts_on_D=True,no_E_writes=True,no_checkpoint_staging=True))
files={str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file() and p.name!='PROVENANCE_SHA256.json'};save('PROVENANCE_SHA256.json',dict(schema='E017_V94_ADMISSION_BLOCKED_1',source_commit=source_commit,artifact_SHA256=files,model_references=refs,status='INFRASTRUCTURE_BLOCKED',no_self_hash=True))
print(json.dumps(dict(status='INFRASTRUCTURE_BLOCKED',root=str(R),D_free_bytes=finalfree,E_free_bytes=e_free,required_bytes=minimum,D_deficit_bytes=minimum-finalfree,files=len(files),TRAIN_identities=434,optimizer_updates=0)),flush=True)
