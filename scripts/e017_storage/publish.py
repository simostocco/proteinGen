"""Publish storage-policy addendum; never rewrite original cc0915e evidence."""
from pathlib import Path
import json,hashlib,shutil,subprocess,time
E=Path(__file__).resolve().parents[3];R=E.parent;S=E/'source'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def put(n,x):
 with open(E/n,'x') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False))
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip();audit=json.load(open(E/'results/FULL_CHECKPOINT_AND_GROWTH_AUDIT.json'));tests=json.load(open(E/'results/ROLLING_STORAGE_TESTS.json'));resume=json.load(open(E/'results/EXACT_RESUME_AND_RETIREMENT_TESTS.json'));pause=json.load(open(E/'results/SAFE_ENGINE_PAUSE_TEST.json'));free=shutil.disk_usage(E).free;G=2**30;C=audit['actual_full_identity_coverage_checkpoint_bytes'];logs=32*2**20
# Verify protected evidence at its ORIGINAL paths, including restored cc0915e source.
old=json.load(open(R/'PROVENANCE_SHA256.json'));verified=0
for rel,h in old['artifacts'].items():assert sha(R/rel)==h,rel;verified+=1
for rel,h in old['source_files'].items():assert sha(R/'source'/rel)==h,rel;verified+=1
for p,h in old['protected_checkpoint_sha256'].items():assert sha(p)==h,p;verified+=1
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip()=='cc0915ef24b57e8c368e1b8bd43681ab8bbb98c4'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R/'source',text=True).strip()
# Objective/model configs can change only storage/logging/evaluation timing here.
allowed={'storage_policy','checkpoint_every','checkpoint_wall_seconds','log_every','validation_milestones','retain_all_checkpoints'};config_audits={}
for p in (E/'configs').glob('*.json'):
 c=json.load(open(p));before=json.load(open(R/'configs'/p.name));changes=[k for k in set(c)|set(before) if c.get(k)!=before.get(k)];assert set(changes)<=allowed,(p,changes);config_audits[p.name]=dict(changed_fields=changes,sha256=sha(p))
scenarios={}
for name,replay in [('supervised_unchanged_replay',0),('RL_with_1GiB_live_replay',G)]:
 cost=3*C+4*replay+logs;scenarios[name]=dict(three_checkpoint_peak_bytes=3*C,live_plus_three_snapshot_budget_bytes=4*replay,logs_and_small_reports_budget_bytes=logs,additional_peak_allocation_bytes=cost,operational_headroom_bytes=2*G,total_required_free_bytes=cost+2*G,projected_free_at_peak_bytes=free-cost,storage_feasible=free>=cost+2*G)
startup_hint_supervised=3*G+logs+2*G;startup_hint_rl=3*G+4*G+logs+2*G
readiness=dict(source_commit=commit,base_commit='cc0915ef24b57e8c368e1b8bd43681ab8bbb98c4',branch='e017-crisp-v3-dynamic-storage-20261021-v1',trainable_parameters=63389056,model_capacity_reduced=False,storage_ready=True,substantive_training_started=False,substantive_training_authorized=False,optimizer_updates_this_task=0,J1_updates_this_task=0,free_D_bytes=free,policy=dict(minimum_operational_headroom_bytes=2*G,warning_free_bytes=4*G,dynamic_next_write=True,near_term_growth_bytes_supervised=32*2**20,near_term_growth_bytes_RL=128*2**20,retained_roles=['latest_complete_recovery','best_validation_complete_recovery'],temporary_staging=True),actual_serialized_checkpoint_bytes=audit['actual_serialized_research_checkpoint_bytes'],full_coverage_checkpoint_bytes=C,scenarios=scenarios,configured_startup_admission_bytes=dict(supervised=startup_hint_supervised,RL=startup_hint_rl),configured_admission_passes=dict(supervised=free>=startup_hint_supervised,RL=free>=startup_hint_rl),largest_live_replay_bytes_for_measured_worst_window=int((free-3*C-logs-2*G)//4),replay_forecast_is_not_an_eviction_or_scientific_budget=True,protected_hashes_verified=verified,pending_scientific_gates=['Approve frozen validation panel/selection metric; validation_panel is currently null','Approve TRAIN-only objective/KL calibration and staged training schedule','Issue separate source/config-specific training authorization'])
put('TRAINING_READINESS.json',readiness)
put('PROPOSED_TRAINING_SCHEDULE.json',dict(execution_authorized=False,capacity=63389056,J1='permanently frozen',first_stage='goal-conditioned supervised initialization of independent RL actor',proposed_main_horizon_updates=10000,AA_effective_batch=64,complete_geometry_decisions=32,goal_demonstrations=8,AA_mixture=[.4,.3,.3],checkpoint_every_updates=100,checkpoint_wall_seconds=900,validation_milestones=[250,500,1000,2500,5000,7500,10000],validation_every_updates=500,review_points=[1000,2500,5000,10000],retention='latest plus best from identical frozen validation metric/panel, complete state for both; aliases reuse files',later_stages=['Critic return calibration on resolved generated episodes','Separately authorized low-level hybrid-SAC pilot, initially about1000 slots','High-level SAC only after direct steering/critic evidence','Reachability/planner conditioning only after calibration'],stop_or_review=['Numerical failure or frozen-J1 identity failure','Storage admission failure -> safe operational pause','Prospectively approved validation/task-quality futility gates'],runtime_estimate='Original bounded-profile extrapolation about8–12 seconds/supervised update plus I/O/validation: roughly22–34 hours/10000, not a measured campaign quote'))
put('TRAINING_READINESS_REPORT.md',f'''# E017 CRISP V3 dynamic-storage readiness

**Storage GO for the proposed full-capacity supervised initialization and the bounded-replay projection below. No substantive training has started or is authorized by this update.**

This addendum supersedes only the fixed10-GiB reserve/storage NO-GO in the cc0915e handoff. It does not alter historical scientific outcomes. Original source/evidence remain at their original paths and pass{verified} checksum checks. Updated source: branch `e017-crisp-v3-dynamic-storage-20261021-v1`, commit `{commit}`, worktree `{S}`. Capacity remains **63,389,056 trainable parameters**, plus all target networks and frozen J1.

## Actual size and peak-write audit

The research architecture and complete AdamW state schema were serialized to a counting stream, without training or retaining an unnecessary ~1-GB untrained checkpoint. PyTorch archives store tensor bytes uncompressed, so zero-initialized audit moments do not shrink the size. Data/sampler metadata come from verified full recovery states and actual TRAIN lengths.

| Item | Measured or planned size |
|---|---:|
| Full checkpoint, current sampler coverage |{audit['actual_serialized_research_checkpoint_bytes']:,} bytes ({audit['actual_serialized_research_checkpoint_bytes']/2**20:.1f} MiB)|
| Full checkpoint, all-identity position coverage scenario |{C:,} bytes ({C/2**20:.1f} MiB)|
| Full-coverage sampler payload |{audit['full_identity_coverage_sampler_bytes']/2**20:.1f} MiB|
| Latest + distinct best, persistent checkpoint tensors/state |{2*C/G:.2f} GiB|
| Latest + best + incoming atomic checkpoint, peak |{3*C/G:.2f} GiB|
| Measured compact-log projection,10000 updates |{audit['projected_10000_update_log_bytes']:,} bytes|
| Configured log/small-report allowance |32 MiB|
| Current D: free |{free/G:.2f} GiB|

Atomic publication requires space for the entire incoming checkpoint AND any changed replay snapshot while the old latest and best stay intact. It does not make a second .pt copy after staging: fsync/rename publishes the existing .part file. Exact serialization bytes are counted before writing; full readback fingerprints every state field, including trainable/target weights, optimizers, schedulers, samplers, counters, RNG and the replay binding. The1-GiB research startup hint covers the measured938.4-MiB full-coverage checkpoint; each actual publication is measured again.

## Rolling retention and replay

Each fresh run gets a unique immutable ownership receipt and one authoritative atomic `RECOVERY_INDEX.json` containing latest AND best pointers. A new complete checkpoint is checksum-verified and deserialized before index publication. Only then may verified superseded artifacts registered to that same run retire. Path confinement, no-symlink checks, link-count/content-hash checks and refusal to adopt nonempty historical runs protect other evidence. Compact RETIREMENTS.jsonl records identify retired own-run files and their verified replacement.

Latest and best may point to the same physical checkpoint. Best has full optimizer/sampler/RNG/replay recovery, not model-only weights. No best is fabricated before a finite score on a frozen validation panel exists. Selection currently uses identity-weighted VALIDATION AA NLL; its panel/metric identity must remain unchanged. This is a retention ranking, not automatic scientific promotion. A frozen validation panel/metric must be approved before training; the supplied validation_panel remains null.

Replay stays persistent in one live SQLite database with durable transactions and a persistent mutation revision. Unchanged replay reuses its immutable snapshot. Changed replay receives a checked snapshot, referenced jointly with the checkpoint; old snapshots retire only when neither latest nor best needs them. At worst there are one live replay plus three snapshots during replacement. Completed same-run resume restores into the SAME live replay path, avoiding abandoned full replay copies. Stage-parent files and all historical runs stay protected. No replay eviction, data truncation or scientific episode-budget constraint was introduced.

The fresh-run worst-window projection is3*C +4*R + small outputs +2-GiB headroom. With unchanged tiny supervised replay, required free space is{scenarios['supervised_unchanged_replay']['total_required_free_bytes']/G:.2f} GiB. With a1-GiB live replay and three distinct snapshots, it is{scenarios['RL_with_1GiB_live_replay']['total_required_free_bytes']/G:.2f} GiB, leaving approximately{scenarios['RL_with_1GiB_live_replay']['projected_free_at_peak_bytes']/G:.2f} GiB physically free at peak. The configured conservative startup checks use5.03/9.03 GiB respectively; both pass at present. The1-GiB replay value is a forecast, not a deletion limit. Actual replay growth, retained parent runs and unrelated D: usage are remeasured before later stages; this does not certify unlimited multi-stage history on the current volume.

## Dynamic admission and safe pause

The fixed10-GiB reserve is removed from active storage admission. Before updates, replay/log writes, replay restoration and checkpoint staging, the policy checks actual next-write requirements or a conservative live-size/observed-checkpoint forecast, near-term growth and **2 GiB operational headroom**. It warns when predicted free space approaches4 GiB. Default growth allowances are32 MiB for supervised and128 MiB for RL until the next checkpoint. SQLite backup size uses page_count*page_size; actual checkpoint serialization is measured. Full checkpoint publication is never replaced by a weights-only fallback.

Forecast pressure is detected before another update. The engine can save the coherent current full state with no further training-growth allowance, keep the2-GiB floor, publish PAUSED_STORAGE status and exit safely; CLI exit75 means pause/admission denial, not completion or automatic resume. ENOSPC/SQLITE_FULL preserves the last verified recovery. If unexpected external consumption prevents even an emergency write or interrupts an update, PAUSED_STORAGE explicitly names the last complete recovery and unsaved/incomplete progress; it does not pretend partial state is exact. Physical disk loss still recovers only from a completed durable checkpoint. One writer per run remains the execution contract.

No disposable full research checkpoint, full dataset copy, long checkpoint history, raw per-target validation export, new model download or historical cleanup was created. Compact per-update logs remain enabled. Caches remain on D:; none are proactively expanded or purged.

## Tests and unchanged science

Rolling latest/best, unchanged-snapshot reuse, changed-snapshot binding, injected publication failure, pending-write crash recovery, best promotion, strict ownership, complete-state readback and simulated low-space admission pass. A completed-run resume test blocks every optimizer entrypoint and verifies exact CPU/CUDA/Python/numpy RNG with no new replay directory. The injected pre-update pressure test saves the coherent current full state and returns PAUSED without optimizer calls. Reports are in [results](results/). This task made **zero optimizer updates and zero J1 updates**.

Model/distribution/reward/data definitions were not changed. J1 remains frozen; all historical E016/E017 checkpoints, original source files and certified input manifests are preserved. The new code lives in its own worktree so the old provenance manifest remains verifiable.

## Proposed schedule and launch

Propose the existing full63.39-M independent actor goal-conditioned initialization for up to10000 updates, AA batch64 with40/30/30 decisions, separate32 complete-geometry decisions and8 goal demonstrations. Review held-out direct-control/likelihood evidence at1000,2500 and5000 before continuing; define scientific stop/selection criteria and TRAIN-only coefficient calibration prospectively. Checkpoint before the first update, then every100 updates or900 seconds at an update boundary, and at validation/final milestones. Validation is planned at250/500/1000/2500 and then every500, including5000/7500/10000. The previous8–12-second update extrapolation implies roughly22–34 hours before extra I/O/validation; reprofile the approved real workload rather than treating this as a runtime guarantee.

Only after that evidence supports direct steering should separately authorized real-return critic calibration and an initial ~1000-slot low-SAC pilot begin. High SAC and reachability remain later independently gated phases. This policy update is not authorization to launch any of them.

Dry-run (no model updates):
```bash
bash {E}/run.sh {S}/scripts/e017_v3/train.py --config {E}/configs/research_supervised.json
```

A later launch adds `--execute --authorization-file <separately approved source/config-specific receipt> --output {E}/future_runs`. Use the new wrapper/worktree, not the preserved cc0915e wrapper. Select the authoritative index's latest checkpoint for `--resume`; same-run recovery is used when ownership and latest identity match. The best checkpoint can instead initialize an explicitly new lineage. No production receipt is provided. Full-size settings and science inputs remain unchanged; only storage/logging/evaluation timing fields differ.

See [machine-readable readiness](TRAINING_READINESS.json), [schedule](PROPOSED_TRAINING_SCHEDULE.json), [size/growth audit](results/FULL_CHECKPOINT_AND_GROWTH_AUDIT.json), [rolling tests](results/ROLLING_STORAGE_TESTS.json), [exact resume](results/EXACT_RESUME_AND_RETIREMENT_TESTS.json), [safe pause](results/SAFE_ENGINE_PAUSE_TEST.json), and [SHA provenance](PROVENANCE_SHA256.json).
''')
put('CHANGELOG.md',f'''# Storage policy v2

- Replaced the fixed10-GiB gate with dynamic actual-write admission,2-GiB operational headroom and4-GiB warning.
- Added full-state verified rolling latest/best bundles, shared unchanged replay snapshots, run-owned retirement receipts and same-run replay restoration.
- Added safe boundary pause, capacity-error handling, status/exit codes,100-update/900-second checkpoints and compact validation output.
- Measured873.4–938.4 MiB full research checkpoint serialization; retained63,389,056 trainable parameters.
- Passed storage/recovery tests without optimizer updates; source `{commit}`. Original cc0915e evidence unchanged.
''')
# Keep the addendum compact; persist only source/config/report checksums and its
# authoritative test recovery bundles. Old protected files are referenced by hash.
new_artifacts={}
for directory in ['results','configs','fixture_run','completed_resume_fixture','safe_pause_fixture']:
 for p in sorted((E/directory).rglob('*')):
  if p.is_file() and not p.name.endswith(('-wal','-shm','.part')):new_artifacts[str(p.relative_to(E))]=sha(p)
for name in ['TRAINING_READINESS.json','TRAINING_READINESS_REPORT.md','PROPOSED_TRAINING_SCHEDULE.json','CHANGELOG.md','run.sh']:new_artifacts[name]=sha(E/name)
changed=subprocess.check_output(['git','diff','--name-only','cc0915e','HEAD'],cwd=S,text=True).splitlines();source={n:sha(S/n) for n in changed if (S/n).is_file()}
put('PROVENANCE_SHA256.json',dict(source_commit=commit,base_commit='cc0915ef24b57e8c368e1b8bd43681ab8bbb98c4',source_branch=readiness['branch'],new_source_files=source,new_artifacts=new_artifacts,old_manifest_sha256=sha(R/'PROVENANCE_SHA256.json'),old_protected_hashes_verified=verified,old_checkpoints=old['protected_checkpoint_sha256'],launch_config_changes=config_audits,optimizer_updates=0,J1_updates=0))
# New operational pointer; does not replace any historical report or manifest.
p=R/'LATEST_STORAGE_POLICY.json'
with open(p,'x') as f:json.dump(dict(policy='dynamic_storage_v2',source_commit=commit,source=str(S),report=str(E/'TRAINING_READINESS_REPORT.md'),configs=str(E/'configs'),storage_ready=True,training_launched=False),f,indent=2)
print(json.dumps(dict(source_commit=commit,D_free_GiB=shutil.disk_usage(E).free/G,storage_ready=True,protected_hashes_verified=verified,substantive_training_started=False),indent=2),flush=True)
