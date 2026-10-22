"""Final offline scientific analysis of sealed V5. No model inference or optimizer calls."""
import os,pathlib,json,hashlib,shutil,datetime,collections,subprocess,math
os.environ['CUDA_VISIBLE_DEVICES']=''
R=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v5_goal_observability_20261022_v1');os.environ['MPLCONFIGDIR']=str(R/'runtime/cache/matplotlib');os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
import sys;sys.path.insert(0,str(R/'source/scripts/e017_v5'));import study as s

def read(n):return json.load(open(R/n))
def save(n,x):s.save(n,x)
def write(n,text):(R/n).write_text(text)
def tab(h,rr):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,x))+' |' for x in rr])
def f(x):return 'N/A' if x is None else f'{x:.4f}'

def main():
 status=read('LIVE_STATUS.json');assert status['phase']=='STUDY_COMPLETE';seal=read('SEALED_CONFIG.json');cen=read('TASK_CENSUS.json');eng=read('results/ENGINEERING_CHECKS.json');cal=read('TRAIN_LOSS_CALIBRATION.json');gates=read('results/CURRICULUM_GATES.json');latest=gates['latest'];teacher=read('manifests/IK_TEACHERS.json');now=datetime.datetime.now(datetime.timezone.utc).isoformat();comparisons={};native={};shift={};byH={};byshift={}
 for H in [2,4,8]:
  byH[str(H)]={};byshift[str(H)]={}
  for arm in ['A','B','C','D','E']:
   u=0 if arm=='A' else latest[arm];a=read(f'results/{arm}_H{H}_{u:05d}_CONFIRM_native.json');b=read(f'results/{arm}_H{H}_{u:05d}_CONFIRM_shifted.json');byH[str(H)][arm]=a;byshift[str(H)][arm]=b
 for first,second,label in [('B','C','observability_C_minus_B'),('D','C','full_feedback_C_minus_D'),('C','E','IK_guidance_E_minus_C')]:
  aa=read(f'results/{first}_H2_00250_CONFIRM_native_matched_H2.json');bb=read(f'results/{second}_H2_00250_CONFIRM_native_matched_H2.json');sa=read(f'results/{first}_H2_00250_CONFIRM_shifted_matched_H2.json');sb=read(f'results/{second}_H2_00250_CONFIRM_shifted_matched_H2.json');comparisons[label]=dict(common_updates=250,H=2,native={key:s.paired(aa['records'],bb['records'],key) for key in ['position_A','orientation_deg','success','valid']},shifted={key:s.paired(sa['records'],sb['records'],key) for key in ['position_A','orientation_deg','success','valid']},direction_first=aa['direction'],direction_second=bb['direction'],note='Matched updates and tasks. E has additional teacher likelihood compute, explicitly not an equal-compute comparison.')
 save('MATCHED_ARM_COMPARISONS.json',dict(comparisons=comparisons,latest_native={H:{a:v['summary'] for a,v in arms.items()} for H,arms in byH.items()},latest_shifted={H:{a:v['summary'] for a,v in arms.items()} for H,arms in byshift.items()},sample_unit='protected protein group; variants and repeated seeds clustered; 4000 paired bootstrap draws; zero-event degeneracy does not certify population zero'))
 usage={};checkpoint_manifest={};import torch
 from protein_joint_sac.crisp_v3.recovery import state_digest
 for arm in ['B','C','D','E']:
  logs=[json.loads(line) for line in (R/f'results/{arm}_updates.jsonl').read_text().splitlines() if line.strip()];idx=read(f'checkpoints/{arm}/INDEX.json');checkpoint_manifest[arm]={}
  for role in ['latest','best']:
   rec=idx[role]
   if rec is None:continue
   assert s.sha(rec['path'])==rec['sha256'];st=torch.load(rec['path'],map_location='cpu',weights_only=False);assert state_digest(st)==rec['state_digest'] and st['config_hash']==seal['config_hash'];assert st['parent_SHA']==read('SETUP.json')['parent_SHA'];assert all(s.sha(s.path(split))==st['task_hashes'][split] for split in ['TRAIN','DEV','CONFIRM']);assert s.sha(R/'manifests/IK_TEACHERS.json')==st['IK_SHA'];checkpoint_manifest[arm][role]=dict(**rec,recovery_fields=list(st),verified=True,J1_tensor_SHA=st['J1_tensor_SHA']);del st
  stage_done=[read(f'results/{arm}_H{H}_FINISHED.json') for H in [2,4,8] if (R/f'results/{arm}_H{H}_FINISHED.json').exists()];usage[arm]=dict(updates=len(logs),stage_updates={str(H):sum(x['H']==H for x in logs) for H in [2,4,8]},distinct_TRAIN_identities=len({h for x in logs for h in x['identities']}),task_presentations=4*len(logs),generated_geometry_actions=sum(4*x['H'] for x in logs),native_BC_decisions=4*len(logs),teacher_decisions_extra=sum(4*x['H'] for x in logs) if arm=='E' else 0,variant_presentations=dict(collections.Counter(v for x in logs for v in x.get('variants',[]))),optimizer_seconds=sum(x['seconds'] for x in logs),mean_update_seconds=float(np.mean([x['seconds'] for x in logs])),last50_update_seconds=float(np.mean([x['seconds'] for x in logs[-50:]])),mean_preclip_norm=float(np.mean([x['gradient_norm'] for x in logs])),clipping_frequency=float(np.mean([x['gradient_norm']>s.CONFIG['clip'] for x in logs])),peak_allocated=max(x['peak_allocated'] for x in stage_done),peak_reserved=max(x['peak_reserved'] for x in stage_done))
 assert s.sha(read('SETUP.json')['parent'])==read('SETUP.json')['parent_SHA'];assert s.sha(read('SETUP.json')['J1'])==read('SETUP.json')['J1_SHA']
 save('CHECKPOINT_AND_RECOVERY_MANIFEST.json',dict(source=seal['source_commit'],arms=checkpoint_manifest,parent=read('SETUP.json')['parent'],parent_SHA=read('SETUP.json')['parent_SHA'],J1=read('SETUP.json')['J1'],J1_file_SHA=read('SETUP.json')['J1_SHA'],J1_frozen_integrity_verified=True,fresh_GPU_resume=read('results/FRESH_GPU_RESTART.json'),evaluation_only_stage_snapshots='EVAL_H*.pt contain actor weights only and are never offered as complete recovery checkpoints; retained only for matched scientific comparisons',no_other_network_optimization=True))
 telemetry=[json.loads(x) for x in (R/'results/RESOURCE_TIMELINE.jsonl').read_text().splitlines() if x.strip()];usage['telemetry']=dict(samples=len(telemetry),peak_sampled_GPU_utilization=max(float(x['GPU'].split(',')[0]) for x in telemetry),peak_sampled_GPU_memory_MiB=max(float(x['GPU'].split(',')[1]) for x in telemetry),peak_worker_RAM_bytes=max((w['RAM'] for x in telemetry for w in x['workers']),default=0));usage['D_free_bytes']=shutil.disk_usage(R).free;usage['total_main_updates']=sum(usage[a]['updates'] for a in ['B','C','D','E']);usage['engineering_updates']=eng['engineering_optimizer_executions']+read('results/FRESH_GPU_RESTART.json')['engineering_optimizer_executions'];usage['IK_teacher_seconds']=teacher['seconds'];save('RESOURCE_AND_RUNTIME_REPORT.json',usage)
 # Learning plots: DEV curves; no confirmation-driven selection.
 fig,axes=plt.subplots(2,3,figsize=(15,8));axes=axes.ravel()
 for arm,color in [('B','tab:gray'),('C','tab:blue'),('D','tab:orange'),('E','tab:green')]:
  entries=[]
  for p in sorted((R/'results').glob(f'{arm}_H*_DEV_native.json')):
   z=json.load(open(p));entries.append(z)
  entries.sort(key=lambda x:x['update']);xs=[x['update'] for x in entries]
  for ax,key in [(axes[0],'position_A'),(axes[1],'orientation_deg')]:ax.plot(xs,[z['summary'][key] for z in entries],marker='o',label=arm,color=color)
  axes[2].plot(xs,[z['summary']['success']/z['summary']['attempts'] for z in entries],marker='o',label=arm,color=color);axes[3].plot(xs,[z['summary']['valid']/z['summary']['attempts'] for z in entries],marker='o',label=arm,color=color);axes[4].plot(xs,[z['summary']['KL'] for z in entries],marker='o',label=arm,color=color)
  logs=[json.loads(x) for x in (R/f'results/{arm}_updates.jsonl').read_text().splitlines()];axes[5].plot([x['update'] for x in logs],[np.mean([r['terms']['goal'] for r in logs[max(0,i-24):i+1]]) for i,x in enumerate(logs)],label=arm,color=color)
 for ax,title in zip(axes,['DEV position Å ↓','DEV orientation degrees ↓','DEV full valid success ↑','DEV backbone validity ↑','Geometry KL/component','TRAIN guide rolling25 ↓']):ax.set_title(title);ax.set_xlabel('Optimizer updates (horizon changes marked in JSON)');ax.grid(alpha=.2);ax.legend()
 fig.suptitle('V5 fixed-AA geometric control; DEV up to24 protected identities, paired native/alternate goals x2 seeds\nTRAIN sampled identities; raw points, identity-cluster uncertainty in MATCHED_ARM_COMPARISONS.json',fontsize=10);fig.tight_layout(rect=[0,0,1,.90]);fig.savefig(R/'TRAIN_VALIDATION_LEARNING_CURVES.png',dpi=145);plt.close(fig)
 def resulttable(H,condition='native'):
  pop=byH[str(H)] if condition=='native' else byshift[str(H)];rows=[]
  for arm,z in pop.items():
   q=z['summary'];rows.append([arm,z['update'],q['attempts'],q['identities'],f(q['position_A']),f(q['median_position_A']),f(q['orientation_deg']),str(q['success'])+'/'+str(q['attempts']),str(q['valid'])+'/'+str(q['attempts']),f(q['KL']),f(z['direction']['rate'])])
  return tab(['Arm','Updates','Attempts','Identities','Mean Å ↓','Median Å ↓','Mean degrees ↓','Full success','Validity','KL','Directional rate'],rows)
 census_text=json.dumps(cen,indent=2);write('DATASET_AND_TASK_CENSUS.md',f'# Dataset/task census\n\n{census_text}\n\nQualified PDB complete observations only. The existing 118,219 sequence population is not all geometrically eligible. One observation per canonical identity avoids deposit overweighting. Source SHA, coordinate row, sequence, native witnesses, original protected groups and partition are retained. Prefixes20/28 permit proteins beyond the old64–96length band. Each horizon has independently checked native and optional kinematic alternate witnesses. Synthetic alternatives are not experimentally verified folds. Original V4 validation groups are excluded from both new VALIDATION panels. Metadata potential-window counts are not an exhaustive coordinate-validity census of all21,475identities. All selected tasks were coordinate-verified. TEST labels/outcomes were never used.\n')
 accepted=collections.Counter();attempted=collections.Counter();reasons=collections.Counter()
 tm={t['task_id']:t for t in s.tasks('TRAIN')}
 for r in teacher['records']:
  H=tm[r['task_id']]['H'];attempted[H]+=1;accepted[H]+=int(r['accepted'])
  for a in r['attempts']:reasons.update(a.get('exclusion_codes',[]))
 ikstats={}
 for H in [2,4,8]:
  ikstats[str(H)]={}
  for condition in ['native','shifted']:
   q=read(f'results/F_IK_H{H}_{condition}.json');ikstats[str(H)][condition]=dict(task_attempts=q['task_attempts'],accepted=q['accepted'],solver_starts=sum(len(r['attempts']) for r in q['records']),seconds=sum(r['seconds'] for r in q['records']))
 save('IK_SOLVER_SUMMARY.json',dict(TRAIN_attempted_by_H=dict(attempted),TRAIN_accepted_by_H=dict(accepted),overlapping_failed_start_codes=dict(reasons),CONFIRM=ikstats))
 write('IK_TEACHER_DATASET_AND_VALIDITY_REPORT.md',f'''# IK teacher and direct baseline

TRAIN tasks attempted per H: {dict(attempted)}. Tasks with at least one valid teacher: {dict(accepted)}. Total teacher-generation wall seconds {teacher['seconds']:.3f}. Failed-start reason frequencies (overlapping): {dict(reasons)}.

Two bounded starts, 64 Adam steps each, target+visible prefix only, no native intermediate coordinates. Bond distances are parameterized strictly inside original spacing bounds; virtual angles/torsions inside production support. Objective uses normalized robust position, frame rotation, local clashes, a 3.8Å bond preference and a small latent anchor. The goal acceptance criteria and original frame/clash/Rg validity remain unchanged. Every accepted trajectory passes all criteria; failures and full action/tail records are retained. All accepted starts are available for stochastic teacher-label sampling, not declared unique inverse solutions. Native supervision and synthetic IK labels remain separate.

E adds one teacher trajectory per ordinary task presentation, weighted {s.CONFIG['weights']['IK']:.8g}, calibrated using TRAIN-only gradients. Additional teacher action likelihood evaluations and compute are explicitly counted. This is not an equal-compute comparison to C. The direct F baseline has up to2 optimization starts per task; actor summaries are individual stochastic attempts. Compare best-of2/any-success per task explicitly before comparing rates. A point-action optimizer has no finite stochastic policy KL.

Confirmation solver summary:\n{json.dumps(ikstats,indent=2)}\n''')
 write('GOAL_RELATIVE_REPRESENTATION.md',f'''# Explicit goal-relative state

The frame implementation stores basis vectors as rows. Thus `basis=frame(x)` equals `R_tip.T` in the column-basis notation. Context uses `basis @ (target-tip)` and `basis @ goal_frame.T`; the latter is the full9-entry relative rotation matrix. This representation is continuous at pi and avoids axis-log ambiguity; it is redundant but well-defined. Rigid-transform invariance was tested (max difference {eng['rigid_transform_context_maxdiff']}).

The64-dimensional context includes local translation/3.8Å, relative rotation9, target distance/3.8Å, remaining actions/8, normalized contour distance, necessary-reachability violation, last-bond length, local-spacing indicator, validity flags and fraction of horizon remaining. Unused dimensions are exactly zero. It is recomputed from generated state at every step. C/D/E goal tokens also use the local target pose. B uses original absolute canonical goal tokens and zero query context. Existing trainable query/goal projections encode these features; no actor capacity expansion is needed. J1 inputs/distributions/parameters are unchanged.

Only current prefix and sparse goal are used. Current AA is fixed as a controlled input; future suffix AAs and hidden coordinates do not enter policy evaluations. The descriptor API is domain-independent kinematics, but no robotics training or empirical transfer claim is made.\n''')
 write('MULTISTEP_GRADIENT_ABLATION.md',f'''# Matched feedback-gradient ablation

C and D have identical generated-prefix forward semantics, losses, inputs and matched random draws. D detaches only generated coordinates supplied to subsequent policy/J1 evaluations. It does NOT detach the geometric reconstruction chain: terminal gradients still reach early actions directly through forward kinematics. Both routes had bitwise-identical forward trajectories before updates; gradient differences were nonzero ({eng['route_gradient_maxdiff']}). All action means had nonzero terminal gradients in both routes. This isolates feedback-derivative contribution, not a stricter cut of all temporal credit.

Full-route action-gradient norms: {eng['full_mu_gradient_norms']}. Feedback-detached: {eng['detached_mu_gradient_norms']}. Finite differences: {eng['terminal_FK_FD']}. Input gradients through J1 were retained without parameter gradients. Main C/D escalation is coupled to preserve matched budgets.

Independent H2 confirmation comparison:\n{json.dumps(comparisons['full_feedback_C_minus_D'],indent=2)}\n''')
 write('GRADIENT_SCALING_AND_LOSS_AUDIT.md',f'''# TRAIN-only scaling and loss audit

`L = Huber((endpoint-target)/3.8, delta=1, sum xyz) + {s.CONFIG['orientation_weight']} * sin²(theta/2) + 0.1*local + 0.1*native_BC + 0.01*geometry_KL`, plus E-only IK NLL weight {s.CONFIG['weights']['IK']:.8g}. BC/NLL are per valid component; KL averaged across valid components and steps. The local term preserves original spacing/clash penalties. Raw Angstrom/degree metrics remain separate. The robust position derivative does not decay to zero for large coordinate errors as V4's bounded rational term does, but that is a mathematical property, not proof it improves learning. Rotation sin² remains bounded and can attenuate near pi; this limitation is retained explicitly.

Calibration and clipping shadow comparison, all TRAIN-only:\n{json.dumps(cal,indent=2)}\n
Actor output channel gradient norms (distance, angle, torsion, then their log-std): {eng['geometry_output_row_gradient_norms_d_angle_torsion_ls']}. Main logged encoder/geometry/residue group norms and clipping frequencies:\n{json.dumps({a:{k:u[k] for k in ['mean_preclip_norm','clipping_frequency']} for a,u in usage.items() if a in ['B','C','D','E']},indent=2)}\n
No longitudinal gradient cosine-conflict measurement or causal ablation of clipping alone is claimed. The initial shadow tests compare unchanged clip1 against5 on the same state; the sealed choice is shared across main arms. No VALIDATION-based tuning.\n''')
 write('HORIZON_CURRICULUM_RESULTS.md',f'# Curriculum results\n\n{json.dumps(gates,indent=2)}\n\nEscalation requires at least10%DEV mean-position gain with pairedCI supporting improvement, at least5percentage-points full-success improvement, validity within5points of A, directional rate>=0.60, plus at least5%shifted-prefix mean-position improvement and preserved validity. All criteria were sealed before baseline outcomes. Unpassed arms do not consume later allocations; C/D progress only together. No16/32training or RL. This staged study does not include a separately trained equal-budget direct-H8 arm, so it cannot by itself isolate curriculum causality (H4).\n')
 write('NATIVE_VS_GENERATED_PREFIX_EVALUATION.md','# Independent confirmation: native vs generated-prefix inputs\n\n'+ '\n\n'.join(f'## H={H}, native\n\n{resulttable(H)}\n\n## H={H}, shifted\n\n{resulttable(H,"shifted")}' for H in [2,4,8])+'\n\nShifted prefixes use two frozen-A-generated actions, identical across arms/goals. Original targets remain; feasible native witnesses do not certify shifted feasibility. Necessary reachability bounds and prefix-spacing flags are stored. Direct IK success supplies a posteriori feasibility evidence, not a replacement of unconditional denominators. Synthetic alternate goals have valid witnessed continuations; they are not experimentally observed alternate folds.\n')
 # All recommendations remain scoped; do not certify SAC from engineering readiness.
 primary=comparisons['observability_C_minus_B'];supported=primary['native']['position_A']['CI95'][1]<0 and primary['native']['success']['CI95'][0]>0 and primary['shifted']['position_A']['CI95'][1]<0;curriculum8=any((R/f'results/{a}_H8_FINISHED.json').exists() for a in ['B','C','D','E']);verdict='Limited positive geometric-control evidence; separately approve replication before RL.' if supported else 'NO-GO for automatic longer-horizon training or SAC: prospective reliable-control evidence insufficient.'
 decision=dict(observability_supported_with_success_and_shifted_position=supported,any_arm_reached_H8=curriculum8,longer_horizon_curriculum='Separate authorization and supported H8 confirmation required; no automatic GO',short_horizon_SAC='NO-GO for production; engineering is separate from demonstrated reward-rich valid on-policy control',high_level_corrective_planning='NO-GO: not tested or trained',no_automatic_next_training=True);save('SCIENTIFIC_GO_NO_GO.json',decision)
 write('NEXT_RL_STAGE_RECOMMENDATION.md',f'''# Research decision

{verdict}

{json.dumps(decision,indent=2)}

Use matched H2 comparisons for H1/H2/H3 attribution. Do not equate native-error reduction with full success or transfer. If an arm improves only native prefixes, repeat on independently feasible generated-prefix goals before SAC. If IK remains superior, retain it as a measured ceiling/diagnostic and assess amortized latency plus training cost rather than claiming neural superiority. No guarantee of biological stability follows from C-alpha validity.

The next smallest test should target the first failed gate, with a prospective new protocol: direct local-pose control on short feasible goals, paired goal-direction outcomes and independent protected groups. Only robust positive held-out valid trajectories warrant a separately authorized low-level SAC pilot with frozen J1/reference KL, then potential high-level reachability learning. None is executed here.
''')
 final=f'''# E017 V5 — Goal observability and IK-informed geometric control

{verdict}

Source `{seal['source_commit']}` on `{seal['branch']}`. Common initialization is original V3 actor SHA `{read('SETUP.json')['parent_SHA']}`. J1 file SHA `{read('SETUP.json')['J1_SHA']}` remained bitwise frozen; all critics, planner, reachability and SAC optimizers inactive. Actor architecture {eng['actor_parameters']:,} parameters, {eng['trainable_parameters']:,} enabled for fixed-AA geometry/shared learning. Explicit pose uses existing64-context and goal projections, adding no parameters.

## Engineering failure and correction before admission

The first task builder used prefix16 with H2, producing an18-residue fragment rejected by the unchanged BuildState minimumN20 contract. An outer exception discarded the entire candidate, causing repeated coordinate reads instead of admitting tasks. No model optimization or prospective validation outcome was executed. That preparer was interrupted with SIGINT after preserving PID, I/O counters and exact diagnostic traces. Historical TRAIN witness probes isolated the bug; corrected prefixes20/28 passed all six H2/H4/H8 witness checks before the new preparation. No action support or validity threshold was relaxed. PREPARATION_ATTEMPT_1_STOP.json, DATA_ADMISSION_PROBE.json, DATA_ADMISSION_PROBE_CORRECTED.json and versioned pipeline-failure records preserve the negative engineering evidence. The read-only source audit found639coordinatefiles totaling3.021GiB, so the large logical-read count was amplification from rejected repeated reads, not a new database download. This was an implementation error, not evidence against the scientific hypothesis.

## Actual data, training and prospective rules

{tab(['Panel','Identities','Observations','Tasks','Tasks by horizon'],[[p,v['selected_identities'],v['selected_observations'],v['tasks'],v['tasks_by_H']] for p,v in cen['panels'].items()])}

Task selection, native/synthetic labels, exclusions, source mappings, protected groups and original partitions are in DATASET_AND_TASK_CENSUS.md and immutable manifests. TRAIN uses identity-first shuffled coverage, then task position and a75/25native/alternate preference with native fallback when a valid alternate is absent. Actual variant counts, rather than assumed quotas, are reported below. All related tasks stay in their original partition. Development and independent confirmation groups are separate and exclude original V4 validation groups. TEST outcomes were never accessed.

Maximum authorized stage increments were250/500/1250, not a target to consume irrespective of learning. Progression and selection were frozen before prospective VALIDATION scores. Development uses24identities per horizon; confirmation uses32 per horizon, with paired goals where valid and two stochastic seeds. The larger prepared64identity DEV/CONFIRM inventories include additional unscored examples. Confirmation was accessed only after training/escalation stopped, and never fitted or tuned on. Complete recovery and evaluation-only stage snapshots preserve equal-update comparisons.

Actual main updates/exposure/resources:\n{json.dumps(usage,indent=2)}\n
## Independent matched H2 comparison

{resulttable(2)}

Final-arm latest checkpoints can have unequal update counts if gates differ. For causal comparisons use the matched+250H2 confirmation snapshots and MATCHED_ARM_COMPARISONS.json:\n{json.dumps({k:{'native':v['native'],'shifted':v['shifted']} for k,v in comparisons.items()},indent=2)}\n
## Higher-horizon evaluation

{resulttable(4)}

{resulttable(8)}

Evaluating H4/H8 does not imply training reached those horizons. Stage-specific completed-update evidence and all failed gates are retained in HORIZON_CURRICULUM_RESULTS.md. Native versus generated-prefix and same-prefix alternate-goal directional tests are separate. See NATIVE_VS_GENERATED_PREFIX_EVALUATION.md for all exact denominators, errors and KL. Original V4 tasks/results remain a distinct exposed historical diagnostic; no past criteria or scores were rewritten.

## IK teachers and direct optimization

TRAIN valid teachers by H: {dict(accepted)} out of {dict(attempted)}. All failed solutions and overlapping exclusion codes retained. Accepted teachers pass target and unchanged spacing/frame/clash/Rg conditions; supported actions alone are not sufficient. E's extra IK likelihood compute is recorded, using TRAIN-only calibrated weight. Native and synthetic labels remain separate. F uses2starts×64steps per task, no hidden intermediate coordinates. Point-action optimization has no stochastic KL; its best-of2 per-task success must not be compared without qualification to individual actor attempts. Exact solver cost/recovery are in IK_SOLVER_SUMMARY.json and IK_TEACHER_DATASET_AND_VALIDITY_REPORT.md.

## Engineering, losses and limitations

Relative-frame rigid-transform and near-pi matrix tests passed. Future-coordinate/suffix leakage checks passed. C/D forward paths match bitwise before updates, while feedback derivative paths differ and direct FK credit remains. Finite differences and frozen-J1 input derivatives passed. Atomic recovery, scheduler/sampler/config references and fresh-GPU next-update exactness passed. All checkpoints and references were verified after completion. No J1 or non-actor updates occurred. Loss transformations and clipping were calibrated on TRAIN only, before freezing; no opportunistic outcome-based tuning.

This study changes data size, goal features and loss versus V4; comparisons to V4 are historical, not exclusive causal attribution. C versus B isolates the new relative-goal input package under otherwise matched new training. C versus D isolates policy-feedback derivatives, not all temporal credit. E versus C adds both synthetic IK labels and compute. Curriculum causality is not isolated without a direct-H8 training arm. Only one training seed is used; protected groups and paired bootstraps mitigate pseudoreplication but do not prove evolutionary independence or biological function. Fixed native AAs isolate geometry and do not demonstrate joint sequence-policy learning or protein foldability.

## Scientific questions

**Q1/Q2, observability/generalization:** See C-minus-B protected-group intervals. Explicit pose is a tested input package, not an assumed missing observable; positive point estimates alone are insufficient. Independent shifted-prefix/full-success evidence is required.

**Q3, temporal gradient value:** See C-minus-D intervals. The engineering derivative difference is established; only supported held-out differences establish learning value beyond matched exposure.

**Q4, IK amortization:** See E-minus-C intervals and extra teacher cost. Valid teachers are experimentally distinct from synthetic kinematic labels. Failure to beat C is evidence against sufficient amortization at this budget, not against IK correctness.

**Q5, curriculum:** Actual horizon admission is recorded. Failure at H2 stops escalation; later inference diagnostics do not establish trained H8 control. No direct-H8 training arm means H4 curriculum benefit remains unisolated.

**Q6, shifted prefixes:** All unconditional results are retained; altered targets can become infeasible. Do not drop failures or replace them with easier samples.

**Q7, IK gap:** Direct optimization results and matched proposal counts expose the gap. Neural amortization must account for training cost and retained validity, not endpoint error alone.

**Q8, gradients:** Main clipping frequencies and component norms are measured. Robust position avoids the former large-error derivative attenuation; rotation loss can still attenuate at pi. No unmeasured causal attribution to loss scaling or clipping.

**Q9, SAC:** {decision['short_horizon_SAC']}. No RL was executed.

**Q10, transfer interfaces:** Goal-relative pose, remaining horizon, masks, action supports and state→goal descriptors are portable kinematic interfaces. No robotics experiment or empirical transfer is claimed.

## Recovery and stop

CHECKPOINT_AND_RECOVERY_MANIFEST.json gives exact latest/best paths, file SHA256, semantic state verification and external full-system bindings. Only superseded checkpoints created by this V5 run were retired after verified replacement; historical artifacts remained unchanged. Dynamic2GiBheadroom,4GiBwarning and staging checks remained active. Current freeD:{usage['D_free_bytes']/2**30:.3f}GiB. All workers have finished; no16/32training, full-protein campaign, SAC, high-levelRL, AlphaFold, dataset acquisition or TEST evaluation.

Decision: {json.dumps(decision)}. Await review; no automatic next-stage training.
'''
 write('FINAL_E017_V5_SCIENTIFIC_REPORT.md',final)
 write('EXPERIMENT_DECISIONS.md',f'# V5 sealed decisions\n\nSource{seal["source_commit"]}; task/config hashes sealed before prospective validation. TRAIN-only calibration retained. All progression failures recorded. {verdict}\n\nNo automatic extension, RL, or future architecture change.\n');write('CHANGELOG.md',f'# V5\n\n{now}: completed bounded engineering and scientific campaign; final evidence and hashes published.\n')
 files={}
 for p in R.rglob('*'):
  if p.is_file() and 'runtime' not in p.parts and 'source' not in p.parts and p.name!='PROVENANCE_SHA256.json':files[str(p.relative_to(R))]=dict(bytes=p.stat().st_size,sha256=s.sha(p))
 sources={str(p.relative_to(R/'source')):s.sha(p) for p in (R/'source/scripts/e017_v5').glob('*') if p.is_file()};save('PROVENANCE_SHA256.json',dict(UTC=now,source=seal['source_commit'],files=files,source_files=sources,J1_SHA=read('SETUP.json')['J1_SHA'],parent_SHA=read('SETUP.json')['parent_SHA'],no_TEST=True,no_RL=True));print('REPORT_PUBLISHED',str(R/'FINAL_E017_V5_SCIENTIFIC_REPORT.md'),flush=True)

if __name__=='__main__':main()
