"""Compact status/publication derived from durable production commitments."""
import datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94_fixed_sequence_geometry_rl_20261024_v1');Q=R/'recovery_qualification_20261025_v1';C=R/'uniref_cleanup_20261024_v1';P=R/'production'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 p=R/name;t=p.with_name(p.name+'.part')
 with t.open('w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def md(name,body):(R/name).write_text(body.strip()+'\n')
def publish():
 UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(); free=shutil.disk_usage(R).free
 clean=read(C/'POST_CLEANUP_VERIFICATION.json');cfg=read(R/'TRAINING_CONFIGURATION_AND_SEEDS.json')
 release=read(Q/'PRODUCTION_RECOVERY_RELEASE.json');assert release['PASS']
 commits=sorted((P/'commits').glob('u*.json')) if P.exists() else []
 count=dict(optimizer_updates=len(commits),generation_attempts=0,complete_trajectories=0,accepted_actions=0,proposals=0,rejected=0,aborts=0)
 seconds=0.; metrics=[]
 for i,p in enumerate(commits,1):
  x=read(p);l=x['logical'];s=l['stats'];assert l['update']==i
  count['generation_attempts']+=s['attempts'];count['complete_trajectories']+=s['complete']
  for k in ['accepted_actions','proposals','rejected','aborts']:count[k]+=s[k]
  seconds+=x['seconds'];metrics.append(l)
 index=read(P/'RECOVERY_INDEX.json') if (P/'RECOVERY_INDEX.json').exists() else None
 if index:
  cp=P/index['latest']['relative'];assert sha(cp)==index['latest']['SHA256']
  index['latest']['absolute_path']=str(cp)
 terminal=read(R/'WORKER_EXIT.json') if (R/'WORKER_EXIT.json').exists() else None
 comparison=read(R/'FINAL_PROSPECTIVE_COMPARISON.json') if (R/'FINAL_PROSPECTIVE_COMPARISON.json').exists() else None
 stop=read(P/'STOP_RECORD.json') if (P/'STOP_RECORD.json').exists() else None
 state='RUNNING_NOT_FINAL' if (R/'LAUNCH_RECEIPT.json').exists() and not terminal else 'ADMITTED_AWAITING_LAUNCH'
 if terminal:state=comparison['outcome'] if comparison else ('RL_STABILITY_NO_GO' if stop else 'INFRASTRUCTURE_BLOCKED')
 launch=read(R/'LAUNCH_RECEIPT.json') if (R/'LAUNCH_RECEIPT.json').exists() else None
 rate=len(commits)/seconds if seconds else None
 record=dict(status=state,UTC=UTC,counts=count,latest_recovery=index,qualification_PASS=True,source_commit=release['source_commit'],
  initializer_SHA256=cfg['parent_SHA256'],J1_SHA256=cfg['J1_SHA256'],D_free_bytes=free,
  cleanup_before_D_free_bytes=clean['D_before'],cleanup_after_D_free_bytes=clean['D_after'],physical_bytes_recovered=clean['actual_D_free_gain'],
  exact_deleted_files=clean['deleted'],all_new_artifacts_on_D=True,no_E_migration=True,TEST=False,AA_fixed=True,
  launch=launch,terminal=terminal,performance=comparison,training_update_seconds_sum=seconds,
  measured_updates_per_second=rate,remaining_training_compute_seconds=(5000-len(commits))/rate if rate else None,
  timing_limitation='sum committed update timings excludes initialization, development evaluation and checkpoint overhead; not a wall-clock completion guarantee')
 save('CURRENT_CAMPAIGN_HANDOFF.json',record)
 save('CHECKPOINT_RECOVERY_MANIFEST.json',dict(status=state,latest_recovery=index,
  journal_committed_updates=len(commits),journal_paths='production/commits/uNNNNNN.json',proposal_traces='production/rollouts/uNNNNNN.json',
  committed_updates_newer_than_checkpoint=len(commits)-(index['latest']['update'] if index else 0),
  qualification_release=str(Q/'RELEASE.json'),qualification_release_SHA256=sha(Q/'RELEASE.json'),
  real_writer_and_corruption_release=str(Q/'PRODUCTION_RECOVERY_RELEASE.json'),supplement_release_SHA256=sha(Q/'PRODUCTION_RECOVERY_RELEASE.json'),
  qualified_full_checkpoint_bytes=read(Q/'RELEASE.json')['checkpoint_bytes'],source_commit=release['source_commit'],
  production_launch=launch,no_automatic_restarts=True,counts=count,
  resume_command=f'{R}/run.sh {R}/source/scripts/e017_v94/production.py production',
  resume_caveat='Never automatically resume a scientific/stability stop. If a crash interrupts a development callback exactly at a milestone, reconcile that callback before continuation; driver fails closed at this boundary.'))
 md('PRODUCTION_STORAGE_ADMISSION.md',f'''# D-only production storage admission

PASS. Cleanup before={clean['D_before']:,} bytes, immediately after={clean['D_after']:,} bytes; measured physical gain={clean['actual_D_free_gain']:,} bytes. Current free={free:,} bytes ({free/2**30:.3f} GiB).

Exactly one verified allowlisted UniRef payload was removed; see uniref_cleanup_20261024_v1/REPORT.md and POST_CLEANUP_VERIFICATION.json. All 11,566 remaining UniRef files and E012 evidence were verified unchanged. Remaining UniRef unique allocation={clean['remaining_UniRef_unique_allocated_bytes']:,} bytes, primarily protected reservoir SQLite/WAL/SHM, key arrays, certificates/manifests and diagnostic evidence. No further cleanup is needed or authorized by this procedure. Original downloads/final FASTA/512 shards were already absent. Exact historical training-dataset replay is unavailable; canonical reservoir source snapshots and hash references are retained.

Original 4,191,962,264-byte floor already included the 2 GiB reserve. Revised conservative prequalification admission={cfg['minimum_free_bytes']:,} bytes, including23×224MiB full-state images/staging,2GiB complete compact traces,512MiB journal/cache/reports/near growth and2GiB reserve. This passed before qualification (11,008,770,048 bytes free). Qualified full checkpoint={read(Q/'RELEASE.json')['checkpoint_bytes']:,} bytes. Existing verified qualification images count toward that peak budget, not an additional future allocation. Dynamic admission remains mandatory before each group and checkpoint.

All source, logs, checkpoints, traces, caches, temporary data and reports stay on D:. No E:/C:/WSL-root/RAM staging, protected deletion or dataset copy.''')
 md('OPTIMIZER_JOURNAL_RECOVERY_QUALIFICATION.md',f'''# Full production recovery qualification

PASS at source `{release['source_commit']}`. Four isolated sequential cases used the same full17,438,114-parameter actor, fresh complete AdamW moments, scheduler, group16 protein sampler and two accepted engineering updates per case. Uninterrupted, crash after rollout, crash after durable optimizer commitment and interrupted publication matched complete semantic model/optimizer/scheduler/RNG/cursor/counter state and subsequent Torch CPU/CUDA, NumPy and Python draws. Repeated atomic publications passed. This is10 physical engineering optimizer executions including required journal replay; zero production updates are inferred from these tests.

An additional real209MB Torch checkpoint write was interrupted after10,886,528 bytes. The incomplete archive was rejected, the authoritative full checkpoint/index remained unchanged, and the actual production restore hash guard rejected an intentionally corrupt new fixture. All failed-write evidence is retained. Exact FP32/FP64 deterministic execution is qualified only for the documented same hardware/software/batch16 configuration.

The new driver atomically switches between two run-owned rolling slots. Named/best files retain immutable inodes; replacement never mutates a linked inode. Each update's full compact proposal trace is durable before optimization; semantic optimizer commitments are durable after acceptance. Restart verifies hashes/config/source and replays committed updates newer than the checkpoint exactly. No accepted work is silently discarded. Uncommitted unsafe work is excluded from authoritative recovery. No automatic restart after a scientific/stability stop.

Limit: interruption exactly during a development callback is fail-closed and requires explicit callback reconciliation; do not bypass the boundary guard. All qualification receipts, logs and hashes are in recovery_qualification_20261025_v1/.''')
 md('TRAINING_STABILITY_AND_VALIDITY.md',f'''# Training stability and validity

Current status: **{state}**. Durable optimizer updates={count['optimizer_updates']}; generation attempts={count['generation_attempts']}; complete trajectories={count['complete_trajectories']}. Complete attempts and finite aborts are counted separately. Accepted actions={count['accepted_actions']}, proposals={count['proposals']}, rejections={count['rejected']}, aborts={count['aborts']}.

Declared budget remains5000×16, with original reward, identity-uniform434-identity/868-goal TRAIN schedule, fixed supplied64-residue sequences and frozen J1. No critic/SAC/AA-policy/high-level update. AdamW peak1e-6,100-step warmup,flat4000,cosine to0.1peak; clip5; trace-KL beta0.009499081422170746; FP32 networks/FP64 densities; no AMP.

Stops: nonfinite/corrupt state, update KL>.02, original active-geometry KL>1,>5pp reserved-TRAIN hard/backbone validity loss,>20% structural-diversity loss, unsafe disk/VRAM. Early low success is not a performance stop. Reserved-TRAIN diagnostics at0/250/500/1000/2000/3500/5000 measure both filtered and underlying unfiltered actor. No prospective VALIDATION fitting or tuning. Curves derive from immutable accepted-update commitments; no engineering executions enter production counts.''')
 md('FILTERED_POLICY_DIAGNOSTICS.md',f'''# Executed stochastic controller

Finite8 validity-rejection proposals per step, deterministic abort if none accepted; actual fixed AA conditions every geometry action. Full raw-latent proposal traces include accepted AND rejected draws. Inactive geometric decisions have zero score; categorical AA scores are absent because AA is externally supplied. Full-trace RLOO with group16 and immutable J1 reference regularization is unchanged from V9.3. Transformed-density Jacobians cancel only in the documented same-transform reference ratios.

Proposal/trace KL is not the KL of the accepted-output filtered distribution. Filtering changes the executed law; no original actor likelihood is assigned to filtered outputs. The underlying unfiltered actor is assessed separately. Current production accounting: {json.dumps(count)}. Retained compact trace files permit exact reconstruction from source references and actions; full redundant backbones are not exported.''')
 if comparison:
  scientific=f"Outcome **{comparison['outcome']}**; selected update{comparison['selected_update']}. Matched filtered K1 difference={comparison['K1_paired']['delta']:.6f}, identity-clustered95%CI={comparison['K1_paired']['CI95']}; K8 difference={comparison['K8_paired']['delta']:.6f}. Exact summaries in FINAL_PROSPECTIVE_COMPARISON.json."
 else:scientific='Final prospective comparison is pending. No RL learning/performance conclusion is supported yet.'
 md('PROSPECTIVE_RL_EVALUATION.md',f'''# Frozen prospective comparison

{scientific}

Frozen32 goals/16 independent VALIDATION identities, witness/relaxed targets, no native-coordinate inference input, TEST sealed. Only reserved TRAIN controls select checkpoints. Primary model is the best eligible development checkpoint; latest recovery and final5000 state remain distinct. Matched actor/J1 frozen baselines are reused only after checking task, source, seed and initial-state fingerprints. K1/K8, full valid success, target error, exposure/burial, hard validity, diversity and proposal costs are reported. Paired bootstrap clusters by16 protected identities (10,000 draws,seed19319); two goals per identity are not independent biological samples. Coarse C-alpha proxy success is not foldability/function evidence.''')
 md('RL_VS_SEARCH_COMPARISON.md',f'''# RL and search

{scientific}

Frozen finite8 controller and exact terminal Best-of8 are the primary matched inference controls. A bounded frozen-controller suffix32 rewind with15 completions plus its initial candidate is a non-RL search control; all discarded proposals/transitions/terminal evaluations are charged. It uses generated prefixes only. Historical TRAIN suffix search is context, not matched validation evidence. Fixed-sequence geometry is the sole trained stage; there is no joint sequence-design or high-level RL claim.''')
 nexttext='Continue only the already declared bounded campaign under its controller; no additional study, thresholds or training extension.' if not terminal else ('Review the independently supported fixed-geometry results before separately authorizing any joint AA/geometry stage.' if comparison and comparison['outcome']=='RL_LEARNING_GO' else 'Review the preserved failure/comparison evidence; no automatic continuation or higher-level campaign.')
 md('NEXT_CRISP_RESEARCH_DECISION.md',f'# Current research decision\n\nStatus **{state}**. {nexttext}\n\n{scientific}\n\nStorage reclamation is verified and production recovery is qualified. No training-success claim follows from either result.')
 md('FINAL_E017_V94_SCIENTIFIC_REPORT.md',f'''# E017 V9.4 — {'campaign status; final science pending' if not terminal else 'scientific/recovery handoff'}

**{state}** as of{UTC}. {scientific}

UniRef cleanup: exactly `{clean['deleted'][0]}` removed; SHA25606d3a3b160bffd1123198404feaf38578edef188d887eeca4e708fa7c22b2b47. Native allocated/free-space gain8,390,508,544 bytes, D:2,647,822,336→11,038,330,880 bytes immediately across deletion. Current D:free={free:,} bytes. All remaining11,566 UniRef files, E012 checkpoints including1M milestone, optimizer/recovery evidence, all SQLite/WAL/SHM, scientific reports/hashes/manifests and E016/E017/PDB inputs were retained unchanged. Original final FASTA/shards/downloads were already absent; exact old dataset replay is unavailable. No migration or unrelated cleanup.

Full actor/Adam/journal/RNG crash/restart qualification PASS; real-write interruption and corruption guards PASS. Source `{release['source_commit']}`, strict parent SHA256 `{cfg['parent_SHA256']}`, permanently frozen J1 SHA256 `{cfg['J1_SHA256']}`. Actual manifest434 TRAIN identities/groups,868 witnessed variants;16 reserved TRAIN development identities excluded;32 prospective goals/16 VALIDATION identities; TEST unopened.

Verified production counts: **{count['optimizer_updates']} updates, {count['generation_attempts']} attempts, {count['complete_trajectories']} complete trajectories**. Counts exclude10 engineering optimizer executions/replays. Budget5000updates/80000 attempts remains unchanged; no additional campaign. Latest recovery/checkpoint SHA and launch PID/command are in CHECKPOINT_RECOVERY_MANIFEST.json and LAUNCH_RECEIPT.json. Per-update immutable commitments and traces preserve accepted work after the latest50-update checkpoint. No adaptive reward/threshold/schedule change.

The checkpoint file is209,390,739 bytes under this exact schema. Revised conservative admission10,234,101,760 bytes included the2GiB reserve and retained qualification evidence; it passed before qualification. Qualification already-allocated images are not double-counted as future writes. All new scientific/cache/tmp/checkpoint files remain on D:.

{nexttext}

This geometry-only study supplies fixed amino-acid sequences and known property requests, never native target coordinates. No SAC/critic/high-level/reachability/TEST/folding work. No biological functionality or thermodynamic conclusion follows from this simulator.''')
 save('FINAL_IDLE_STATUS.json',dict(status=state,UTC=UTC,production_counts=count,all_new_artifacts_on_D=True,worker_exit=terminal,
  research_idle=bool(terminal),GPU_released=terminal.get('CUDA_compute_processes_after')==[] if terminal else False))
 names=['FINAL_E017_V94_SCIENTIFIC_REPORT.md','PRODUCTION_STORAGE_ADMISSION.md','OPTIMIZER_JOURNAL_RECOVERY_QUALIFICATION.md','TRAINING_CONFIGURATION_AND_SEEDS.json','TRAINING_STABILITY_AND_VALIDITY.md','FILTERED_POLICY_DIAGNOSTICS.md','PROSPECTIVE_RL_EVALUATION.md','RL_VS_SEARCH_COMPARISON.md','CHECKPOINT_RECOVERY_MANIFEST.json','NEXT_CRISP_RESEARCH_DECISION.md','CURRENT_CAMPAIGN_HANDOFF.json']
 save('PROVENANCE_SHA256.json',dict(schema='E017_V94_ACTIVE_OR_FINAL_SNAPSHOT_2',status=state,UTC=UTC,source_commit=release['source_commit'],
  artifact_SHA256={n:sha(R/n) for n in names},auxiliary_source_SHA256={str(p.relative_to(R)):sha(p) for p in (R/'audit_source').glob('v94_*.py')},
  cleanup_verification_SHA256=sha(C/'POST_CLEANUP_VERIFICATION.json'),recovery_release_SHA256=sha(Q/'PRODUCTION_RECOVERY_RELEASE.json'),
  mutable_streams_excluded=['production/rollouts','production/commits','production/diagnostics','production.log','RL_TRAINING_CURVES.csv'],
  initial_storage_blocked_evidence_retained='initial_storage_blocked_20261024',self_hash_excluded=True))
 print(json.dumps(record),flush=True)
if __name__=='__main__':publish()
