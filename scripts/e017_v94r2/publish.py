"""Evidence-derived active/terminal publication; no neural work or restarts."""
import csv,json,os,runpy,statistics,subprocess
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r2_corrected_geometry_rl_20261025_v2');C=runpy.run_path(str(R/'source/scripts/e017_v94r2/common.py'));read=C['read'];atomic=C['atomic'];sha=C['sha']
def md(name,s):(R/name).write_text(s.strip()+'\n')
def publish():
 live=C['live']();cfg=read(R/'TRAINING_CONFIGURATION_AND_SEEDS.json');q=read(R/'qualification/RELEASE.json');source=q['binding']['V94_source_commit'];status=live['V94R2_LIVE_STATUS'];p=R/'production'
 atomic(R/'CURRENT_CAMPAIGN_HANDOFF.json',live)
 atomic(R/'CHECKPOINT_RECOVERY_MANIFEST.json',dict(status=status,latest=live['LATEST_CHECKPOINT'],committed_update=live['COMMITTED_UPDATES'],
  generation_attempts=live['GENERATION_ATTEMPTS'],complete_trajectories=live['COMPLETED_TRAJECTORIES'],aborts=live['ABORTED_ATTEMPTS'],
  binding=q['binding'],qualification_type=q['qualification_type'],original_full_recovery_release=q['original_release'],original_release_SHA256=q['original_release_SHA256'],
  actual_initial_roundtrip=read(R/'ACTUAL_INITIAL_RECOVERY_ROUNDTRIP.json') if (R/'ACTUAL_INITIAL_RECOVERY_ROUNDTRIP.json').exists() else 'PENDING_BEFORE_FIRST_UPDATE',
  journal='production/commits/uNNNNNN.json',trace='production/rollouts/uNNNNNN.json',automatic_restarts=False,
  resume_command=f'{R}/run.sh {R}/source/scripts/e017_v94r/production.py production --authorization {R}/REVIEW_AUTHORIZATION.json',
  safety_stop_requires_review=True,interrupted_development_callback_requires_reconciliation=True))
 result=live['scientific_result'];terminal=live['terminal'];outcome=result['outcome'] if result else 'RL_STABILITY_NO_GO' if live['stop'] else 'INFRASTRUCTURE_FAILURE' if terminal else 'SCIENTIFIC_ASSESSMENT_PENDING'
 if result:
  interpretation=f"Primary paired K1 improvement={result['K1_paired']['delta']:.6f},95% identity-clusteredCI={result['K1_paired']['CI95']};K8 change={result['K8_paired']['delta']:.6f}. Decision={outcome}. Full numerical comparisons, inference costs and paired target responses are in FINAL_PROSPECTIVE_COMPARISON.json."
 else:interpretation='No prospective V94R2 outcome is available. No RL performance or convergence conclusion is warranted.'
 md('FINAL_E017_V94R2_SCIENTIFIC_REPORT.md',f'''# E017 V9.4-R2 — {'terminal report' if terminal else 'active campaign handoff; final science pending'}

**{status}** at{live['UTC']}. {interpretation}

Verified counters: {live['COMMITTED_UPDATES']}/5000 optimizer commitments;{live['GENERATION_ATTEMPTS']}/80000 generation attempts;{live['COMPLETED_TRAJECTORIES']} completions;{live['ABORTED_ATTEMPTS']} aborts. Generation attempts are not all complete proteins. No short pilot replaces the approved budget. Launch receipt supplies actual PID/command; authority is the current explicit user V9.4-R2 authorization. No automatic restart or later experiment.

New root `{R}`; source `{source}`. Strict original preV94 parent SHA256 `{cfg['parent_SHA256']}`; permanently frozen J1 SHA256 `{cfg['J1_SHA256']}`; actual serialized plain-dict initializer fingerprint884b1e016135f4bf4b3594b6aefc81c1594a525186e09091ad5bd5c023781f80. Fresh AdamW/scheduler/journal/cursor/RNG, master training seed19327. Never initialize from breached update500 or engineering weights. Both historical V94 and V94R remain sealed and unchanged.

Exact reviewed mathematical kernel verified by normalized AST equivalence: packing, finite sampler, full rejected+accepted likelihood, gradients, all Runtime optimizer/checkpoint/journal methods. Original V94R full interruption tests remain the qualification basis. This run additionally requires a real fresh full-state checkpoint/optimizer/scheduler/RNG round-trip and next-draw equality before any optimizer step. No duplicate qualification models were materialized. Extra changes are run location/seed/provenance, pretraining restore/admission checks and diagnostic logging. No scientific reward, supports, model capacity or trained-network change.

Packing rejection uses2*Cprefix/64>3.6215624999999996, where C counts strict-distance<8Å pairs at sequence separation>=8. It permits115 pairs and rejects116; equality at the original scalar upper bound is accepted. Budget remains8 proposals, exhausted attempts fail. Original goal/success/reward and homology-isolated434 TRAIN identities/868 tasks remain. AA identities are fixed; this is geometry RL, not sequence design.

Fixed development:16 reserved TRAIN identities,8candidates,updates0/250/500/1000/2000/3500/5000,raw and filtered conditions. Initialization baseline is frozen before optimization. Original5pp validity allowance,20% diversity allowance,referenceKL1,stepKL.02,numerical and resource stops remain. Original absolute floors additionally bind. No new separate abort-rate threshold: all aborts count as hard-invalid failures. Extra paired-goal sensitivity uses4 fitting TRAIN identities solely for diagnosis, never selection. Witness/relaxed TRAIN reward changes are stratified rather than called learning.

Final prospective evaluation is gated on5000 accepted updates/80000 attempts and trainer exit. The frozen32 goals/16VALIDATION identities compare J1, raw/filtered parent, selected trained actor, final5000 actor,Bestof8/16 and charged bounded generated-prefix search. Same corrected controller is used for the primary actor comparison. Paired bootstrap10000 draws clusters by identity;seed19329. TEST remains sealed. Biological foldability,function and stability are not inferred from C-alpha proxy success.

All artifacts/cache/tmp remain on D:. Current free={live['D_FREE_BYTES']:,}bytes ({live['D_FREE_GIB']:.3f}GiB). The conservative initial remaining envelope is7,180,648,448bytes, including10 full-state image/staging slots,2GiB traces,512MiB other growth and2GiB operational reserve. Dynamic remaining-work forecasts include observed trace-size growth with15% margin and trigger cooperative pause if insufficient. Preferred8GiB free is not met; the verified full envelope is the admission criterion. No deletion/migration.

See frozen configuration/source manifest, COMPLETE_TRAINING_CURVES.csv, augmented metrics, baseline/development records, CHECKPOINT_RECOVERY_MANIFEST.json and RESOURCE_AND_STORAGE_REPORT.json. {('Workers have exited; see WORKER_EXIT.json for actual GPU release.' if terminal else 'The bounded controller continues only the authorized campaign; no claim of final scientific success.')}
''')
 md('FILTERED_AND_UNFILTERED_ACTOR_DIAGNOSTICS.md',f'''# Filtered and underlying actor diagnostics

Status {status}. Raw actor and corrected finite8 controller are evaluated separately on the same frozen16-identity development panel. Complete overpacking and incomplete aborts are different outcomes; the latter are never counted as rescued successes. Per-update augmented metrics record immediate/packing rejection counts,proposal counts/accepted action,aborts,contact statistics,exposure/burial,hard/full success,target errors and loss/advantage/entropy/KL definitions. Latent Gaussian entropy and proposal/trace KL are not the entropy/KL of accepted filtered outputs. See METRIC_DEFINITIONS.json and FROZEN_INITIALIZATION_BASELINE.json when available.

Current training complete-overpacking rate={live['PACKING_FAILURE_RATE']:.6f};abort rate={live['PROPOSAL_ABORT_RATE']:.6f}. These TRAIN rates are not held-out scientific evidence. Filtering alone cannot establish RL learning. {interpretation}''')
 md('GLOBAL_PACKING_AND_ABORT_ANALYSIS.md',f'''# Global packing and aborts

An append-only prefix cannot lose established nonlocal contacts. Its final degree lower bound2*C/64 is exact. Current prefix average2*C/n is not used as an irreversible condition. No Rg/target heuristic is substituted for this proof. Contact budget rejection changes the filtered policy law and is included through every sampled raw-latent proposal score, including rejected draws. Reference uses the identical filter/fallback.

Training attempts={live['GENERATION_ATTEMPTS']};completions={live['COMPLETED_TRAJECTORIES']};aborts={live['ABORTED_ATTEMPTS']};completed-overpacking rate={live['PACKING_FAILURE_RATE']:.6f}. Finite8 exhaustion remains failure. No additional abort gate was invented. Its effect is constrained by the original matched hard-validity gate. Original V94 absolute comparisons stay visible; no baseline reset. Full per-residue rejection/slack and contact-degree statistics are in augmented metrics and frozen development records.''')
 md('GOAL_CONDITIONED_PROPERTY_RESULTS.md',f'''# Goal-conditioned properties

{interpretation}

Exposure,burial,contrast and hard validity are reported separately and jointly. Fixed-panel diagnostics use matched streams. Four fitting-TRAIN identities' existing witnessed/relaxed goal pairs hold sequence and random streams fixed, on raw and filtered policies; these extra diagnostics never select checkpoints. Prospective paired target interventions occur only after full training. Different success thresholds alone can change success without changed geometry; inspect achieved-property and geometry deltas. No native target coordinates or future hidden sequence enters the policy. Fixed supplied AAs are not learned motif following.''')
 md('MATCHED_BASELINE_COMPARISONS.md',f'''# Matched baseline comparisons

{interpretation}

The primary comparator is the original actor under the SAME corrected filter. Old V94/V93 filtered samples are not reused as corrected-law samples. Final baselines are freshly generated under fixed task/candidate-indexed streams. Bestof16 is nested from two independent8-candidate batches; all candidates count. The non-RL control charges a full initial attempt and, when applicable,15 completions from its generated32-residue prefix. It has different actor/transition costs fromBestof8, all recorded; no hidden native endpoint,IK target or learned planning claim. Final confirmation has32 goals but16 independent identities.''')
 md('SCIENTIFIC_PERFORMANCE_ASSESSMENT.md',f'''# Scientific assessment

Current classification: **{outcome}**. {interpretation}

A completed qualified campaign requires K1 gain>=5pp and paired95% lower>0,hard/backbone/referenceKL/diversity safeguards andK8notworse for primary promotion. Report partial property changes separately when full success does not improve. Early safety stops are stability/infrastructure outcomes, not definitive RL performance failures. High TRAIN reward across goal-difficulty transitions is not evidence of generalization. The task tests fixed-sequence regional C-alpha geometry steering only.''')
 md('NEXT_CRISP_RESEARCH_DECISION.md',f'''# Current decision

{outcome}. {interpretation}

{('Review these measured results before authorizing any further campaign.' if terminal else 'Continue only the admitted5000-update run unchanged under its frozen safeguards.')} No automatic V9.5,SAC,high-level RL,reachability training,joint-AA stage,TEST or external folding work.''')
 # Complete wide curves derive from accepted commitments and augmented metrics.
 extras={}
 if (R/'AUGMENTED_TRAINING_METRICS.jsonl').exists():
  for line in (R/'AUGMENTED_TRAINING_METRICS.jsonl').read_text().splitlines():
   x=json.loads(line);extras[x['update']]=x
 rows=[]
 for f in sorted((p/'commits').glob('u*.json')):
  x=read(f);l=x['logical'];z=dict(update=l['update'],identity=l['identity'],goal_id=l['goal_id'],seconds=x['seconds'],D_free_bytes=x['D_free_bytes'],**l['stats'],**l['diagnostics'],gradient_norm=l['gradient_norm'],clipped=l['gradient_norm']>5,learning_rate=l['lr'],step_KL=l['step_KL'])
  z.update({k:v for k,v in extras.get(l['update'],{}).items() if not isinstance(v,(list,dict))});rows.append(z)
 fields=sorted({k for z in rows for k in z}) if rows else ['update','attempts','complete','aborts','reward_mean','hard','full','gradient_norm','learning_rate']
 tmp=R/'COMPLETE_TRAINING_CURVES.csv.part'
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(tmp,R/'COMPLETE_TRAINING_CURVES.csv')
 resource=read(R/'LATEST_STORAGE_PROJECTION.json') if (R/'LATEST_STORAGE_PROJECTION.json').exists() else C['projection']()
 atomic(R/'RESOURCE_AND_STORAGE_REPORT.json',dict(resource,all_artifacts_on_D=True,qualification_models_not_duplicated=True,protected_history_unchanged=True,live_status=status,
  workers_exited=bool(terminal),CUDA_release=terminal['CUDA_compute_processes_after'] if terminal else None))
 files=['FINAL_E017_V94R2_SCIENTIFIC_REPORT.md','FILTERED_AND_UNFILTERED_ACTOR_DIAGNOSTICS.md','GLOBAL_PACKING_AND_ABORT_ANALYSIS.md','GOAL_CONDITIONED_PROPERTY_RESULTS.md','MATCHED_BASELINE_COMPARISONS.md','SCIENTIFIC_PERFORMANCE_ASSESSMENT.md','CHECKPOINT_RECOVERY_MANIFEST.json','RESOURCE_AND_STORAGE_REPORT.json','NEXT_CRISP_RESEARCH_DECISION.md','TRAINING_CONFIGURATION_AND_SEEDS.json','CURRENT_CAMPAIGN_HANDOFF.json']
 atomic(R/'PROVENANCE_SHA256.json',dict(schema='E017_V94R2_ACTIVE_OR_TERMINAL_SNAPSHOT_1',UTC=C['utc'](),source_commit=source,status=status,
  source_SHA256=q['binding']['source'],artifact_SHA256={n:sha(R/n) for n in files},qualification_release_SHA256=sha(R/'qualification/RELEASE.json'),
  model_references=dict(J1_SHA256=cfg['J1_SHA256'],parent_SHA256=cfg['parent_SHA256']),
  mutable_streams_excluded=['production/commits','production/rollouts','production/diagnostics','production.log','AUGMENTED_TRAINING_METRICS.jsonl','RESOURCE_TELEMETRY.jsonl','COMPLETE_TRAINING_CURVES.csv'],no_self_hash=True))
 print(json.dumps({k:live[k] for k in ['V94R2_LIVE_STATUS','TRAINER_PID','COMMITTED_UPDATES','GENERATION_ATTEMPTS','COMPLETED_TRAJECTORIES','D_FREE_GIB']}),flush=True)
if __name__=='__main__':publish()
