"""Publish the safety review; no neural imports, evaluation, or worker launch."""
from pathlib import Path
import datetime,hashlib,json,math,shutil,statistics,subprocess
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r_packing_safety_20261025_v1');O=R.parent/'e017_v94_fixed_sequence_geometry_rl_20261024_v1'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def md(n,s):(R/n).write_text(s.strip()+'\n')
def save(n,v):(R/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def summary(rows):
 return dict(N=len(rows),complete=sum(x['metrics']['complete'] for x in rows),hard=sum(x['metrics']['hard_valid'] for x in rows),full=sum(x['metrics']['success'] for x in rows),overpacked=sum(x['metrics']['complete'] and not x['metrics']['packing_valid'] for x in rows),aborts=sum(x['termination'] is not None for x in rows))
def main():
 hist=read(R/'HISTORICAL_HASH_AND_CHECKPOINT_AUDIT.json');fail=read(R/'MATCHED_TRAJECTORY_FAILURES.json')['records'];reward=read(R/'REWARD_ADVANTAGE_AND_SCALE_AUDIT.json');grad=read(R/'HISTORICAL_GRADIENT_REPLAY.json');cpu=read(R/'FILTER_AND_GRADIENT_CPU_TESTS.json');gpu=read(R/'GPU_ENGINEERING_COMPLETE.json');qual=read(R/'qualification/RELEASE.json');resource=read(R/'RESOURCE_AND_STORAGE_REPORT.json');cfg=read(R/'TRAINING_CONFIGURATION_AND_SEEDS.json');cf=read(R/'AFFECTED_TRAIN_COUNTERFACTUALS.json');controls=read(R/'TRAIN_CONTROLLER_CONTROLS.json')['records'];branch=read(R/'FILTER_BRANCH_AND_GOAL_DETAILS.json')['records']
 assert all(x['PASS'] for x in [hist,grad,cpu,gpu,qual]);assert not (R/'production').exists(),'review must not launch production'
 decision='READY_FOR_SUBSTANTIAL_RL_RESTART' if resource['storage_GO'] else 'INFRASTRUCTURE_BLOCKED'
 source=qual['binding']['V94_source_commit'];assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip()==source
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=R/'source',text=True).strip()
 for name,h in qual['binding']['source'].items():assert sha(R/'source'/name)==h
 assert sha(R/'TRAINING_CONFIGURATION_AND_SEEDS.json')==qual['binding']['configuration_SHA256']
 # Historical sealed reports/source remain unchanged after all engineering.
 oldprov=read(O/'PROVENANCE_SHA256.json')
 for n,h in oldprov['artifact_SHA256'].items():assert sha(O/n)==h,n
 for n,h in oldprov['auxiliary_source_SHA256'].items():assert sha(O/n)==h,n
 idx=read(O/'production/RECOVERY_INDEX.json')['latest'];assert sha(O/'production'/idx['relative'])==idx['SHA256']
 assert read(O/'production/STOP_RECORD.json')['update']==500
 fitold=summary([x for g in controls for x in g['old_records']]);fitnew=summary([x for g in controls for x in g['new_records']]);aold=summary([x for g in cf['groups'] for x in g['old_records']]);anew=summary([x for g in cf['groups'] for x in g['new_records']])
 rows=[]
 for x in fail:
  a=x['original'];b=x['update500'];ident=x['goal']['goal_id'].split('_')[2];ci=x['candidate_index'];first=b['first_proven_overbudget_residue_1based'];tag='new' if x['new_failure'] else 'existing'
  rows.append(f"|{ident}/{ci} ({tag})|{a['metrics']['geometry']['longrange_contact_degree']:.5f}|{b['metrics']['geometry']['longrange_contact_degree']:.5f}|{first or 'none'}|{x['classification']}|")
 table='\n'.join(rows)
 goals=[]
 for x in fail:
  if not x['new_failure']:continue
  g=x['goal'];a=x['original']['metrics'];b=x['update500']['metrics'];f=lambda v:'not complete' if v is None else f'{v:.6f}'
  goals.append(f"|{g['goal_id'].split('_')[2]}/{x['candidate_index']}|{g['exposed']} / {g['buried']}|{g['exposure_min']} / {g['burial_max']}|{f(a.get('buried_mean'))} → {f(b.get('buried_mean'))}|")
 parent=cfg['parent_SHA256'];j1=cfg['J1_SHA256'];fresh=next(x for x in hist['checkpoints'] if 'development_best_00000.pt' in x['path'])
 md('GLOBAL_PACKING_FAILURE_ANALYSIS.md',f'''# Matched packing failure analysis

The stopped V94 arm retains500 committed updates,8000attempts,7958complete trajectories and42aborts. Its fixed reserved-TRAIN panel fell124/128→117/128 hard-valid; the frozen allowed decline was5pp, actual5.46875pp. No threshold was changed. No new VALIDATION/TEST result was generated.

|Goal hash / candidate index0|Initial degree|Update500 degree|First irrecoverable residue1|Classification|
|---|---:|---:|---:|---|
{table}

Degrees on aborted34/63-residue prefixes use the actual prefix denominator and must NOT be compared as completed N64 packing outcomes. Nine completed packing violations pass the original backbone, local-spacing, frame, clash and Rg tests. They exceed a TRAIN-calibrated99th-percentile global contact-density threshold; this is not proof of a physical steric defect. The remaining two failures are finite8 proposal exhaustion, not proven geometric unreachability. An incomplete score's backbone=False can mean incomplete, rather than a newly broken local backbone.

All nine completed failures cross the exact final-N contact bound at residues52–64; the five newly overpacked completions first cross at62,64,64,61,62. The two aborts have no such crossing. The old filter accepted these packing-crossing actions because it checked supports/local geometry/clashes but ignored the already computed final packing lower bound. A high CURRENT prefix average alone is not irreversible: appending residues changes its denominator.

MATCHED_TRAJECTORY_FAILURES.json contains every contributing nonlocal contact pair, additions/removals against initialization, per-residue contact leaders, complete prefix contact curves, exact goals, protected identity, candidate/seed and immutable source-file references. Pair indices are0-based; displayed residue positions are1-based. These are posthoc reconstructions from generated actions, not hidden future inputs to the controller.

|New failure|Exposed / buried windows0-based half-open|Exposure minimum / burial maximum|Observed burial mean initial→500|
|---|---|---|---|
{chr(10).join(goals)}

Burial improves in all five newly overpacked complete candidates, but three already met burial at initialization; two still miss burial at500. No additional full success results. Within witnessed and relaxed TRAIN goals, packing degree correlates negatively with burial proxy: r=-0.4933(n6910) and-0.4597(n1048) among complete trajectories. This supports a contact/burial tension, not a causal claim that the optimizer deliberately learned overpacking.

FILTER_BRANCH_AND_GOAL_DETAILS.json identifies first discrete accepted-proposal changes at residues18,51,64,50,40,38,60 for the seven new failures. Small policy changes can cross hard rejection boundaries and produce different subsequent random draws/geometry despite very small per-update KL. These acceptance-path changes often precede final packing failure; they are not themselves proofs of doomed prefixes.

Reward audit over all8000 attempts:286 overpacked completions and45 other invalid attempts all received reward0, all with negative augmented group-relative advantages. Full successes514 received mean reward0.95517 and mean advantage+0.61488; overpacked mean advantage=-0.30114. There is no observed positive-reward or positive-advantage teacher bug for overpacked trajectories. Valid burial is encouraged, while global invalidity is penalized only through a delayed zero reward. Credit assignment/variance and acceptance-boundary amplification remain measured limitations, not proven loss implementation defects.''')
 md('GRADIENT_CLIPPING_AND_LOSS_SCALE_AUDIT.md',f'''# Gradient clipping and loss scale

No aggregation defect was found. The objective averages16 trajectories exactly once and sums all stochastic proposal log-densities across the trajectory, including rejected proposals. It is not a per-token loss. A completed64-residue accepted trace has186 active scalar geometry decisions:0 at residue1,1 at2,2 at3, then3 each. Rejections add decisions. Historical observed group means span179.625–194.4375 active dimensions per attempt, mean189.3565, including finite aborts. Group1 has191.0625 dimensions per attempt. The original rough189 receipt was retained and its metadata corrected explicitly.

For L=Σlogπ(raw proposals), ell=L−Lref, a=R−βell−leave-one-out(R−βell), the implemented surrogate is mean[-aL+βell], β={cfg['trace_KL_beta']}. Group baselines exclude the same trajectory; no same-group whitening, token division, hidden accumulation multiplier or AA categorical term. Gradients accumulate64 residue-node graphs followed by one Adam step per16-attempt group. Fixed AAs condition the geometry distribution and carry zero categorical score.

Exact historical update1 replay, no optimizer step:

|Quantity|Value|
|---|---:|
|Original and reproduced combined norm|{grad['combined_norm']:.12f}|
|Reward component norm|{grad['reward_component_norm']:.9f}|
|Reference-KL component norm|{grad['reference_KL_component_norm']:.9f}|
|Component cosine|{grad['component_cosine']:.9f}|
|Maximum component-sum FP32 discrepancy|{grad['max_component_sum_error']:.9g}|
|Maximum sampled-score replay discrepancy|0|
|Clip multiplier at norm5|{grad['global_clip_factor']:.9f}|

Reference regularization opposes part of the reward gradient rather than dominating its norm. Clipping497/500 updates is a strong, frequent rescaling of mathematically correct long-trajectory gradients, not evidence by itself of numerical explosion. Historical mean norm32.3264,max163.1063; all finite. Mean augmented-advantage variance={statistics.mean(x['advantage_variance'] for x in reward['groups']):.9f}. The clip5 choice is documented as a conservative fixed safeguard; no empirical norm-calibration rationale was found. It is retained unchanged to isolate the packing-filter intervention, not raised opportunistically.

KL metrics are different:

* geometry_KL: KL(actor Gaussian||frozen J1 Gaussian), divided by active dimension, averaged on visited accepted states. Development500≈0.01021 nats/dimension; limit1.
* step_KL: KL(pre-update actor||post-update actor), summed active dimensions, maximum over fixed probe states. Historical maximum≈2.0117e-6 nats; limit0.02.
* trace_log_ratio: Monte Carlo full rejected+accepted proposal-trace log-ratio against J1 under the SAME deterministic filter/abort. Historical mean≈2.07927 nats/trajectory. This is not per-dimension KL or accepted-output KL.

The corrected gradients.py adds per-update policy-gradient surrogate value, augmented-advantage variance, reward variance and underlying latent Gaussian entropy per active dimension. Existing norms, clip incidence, both KL definitions and full-trace ratios remain. Latent Gaussian entropy is NOT effective filtered-output entropy. Component norms were measured in the engineering replay; expensive separate component backwards are not added to every production update. Reward/success, invalid/overpacked/aborted counts and raw-versus-filtered development results remain separate.''')
 md('FILTERED_POLICY_CORRECTNESS.md',f'''# Corrected finite-filter stochastic policy

One algorithmic change only: finite8 acceptance additionally rejects a generated extension when2*Cprefix/64 exceeds the unchanged upper packing degree3.6215624999999996. C counts undirected strict-distance<8Å contacts with sequence separation>=8. At N64,115 pairs are allowed;116 are forbidden. Existing coordinates are immutable under suffix append, so Cprefix<=Cfinal. This is an exact necessary condition for the original final metric, not a heuristic and not a guarantee of eventual success. Rg, exposure targets and current-prefix mean density are not used as unjustified irreversible constraints.

The original local/support/clash tests and finite8 deterministic abort remain unchanged. A bounded Cprefix/115 risk fraction and remaining pair slack are diagnostic heuristics only: they are not reachability predictions, rewards or extra rejection rules. No packing penalty, temperature change, support projection, new network input or capacity change is introduced.

The complete raw proposal trace includes every accepted AND rejected Normal draw and the finite-abort branch. Its density is the product of the sampled base densities at generated states; deterministic acceptance/fallback has no parameter-dependent likelihood term when conditioned on those raw samples. Fixed AAs have zero categorical score. Distance exp(u), angle πsigmoid(u), torsion πtanh(u), masks and transformed Jacobians remain unchanged. Reference traces use the same new controller. Full-trace likelihood ratios cancel the same transforms; original proposal KL must not be labeled effective filtered-output KL.

CPU qualification: {cpu['checks']} checks;156 hard-valid fitting-TRAIN trajectories retained every prefix; randomized append-monotonicity and integer-bound tests; exact finite-tree normalization; full-trace policy-gradient maximum error{cpu['exact_full_trace_gradient_error']:.3g}; accepted-only negative control error{cpu['accepted_only_negative_control_error']:.6f}; transformed Jacobian and masked Normal densities; fixed-AA zero gradient. Gaussian finite-rejection Monte Carlo gradient{cpu['Gaussian_MC_gradient']:.6f}±{cpu['Gaussian_MC_SE']:.6f}SE versus exact{cpu['Gaussian_exact_gradient']:.6f}. No future coordinates or VALIDATION/TEST labels are inputs.

Fresh full-actor batch16 optimizer/journal replay also passes under the corrected law. This mathematically qualifies the controller, not protein foldability or eventual RL performance. The initial missing-helper NameError was caught before any optimizer step, retained in ENGINEERING_ATTEMPT_01_FAILURE.json and fixed without changing the geometry transform; no failed scientific gate was reclassified.''')
 md('CORRECTIVE_INTERVENTION_EVIDENCE.md',f'''# Minimal corrective intervention evidence

The justified change is exact final-N contact-budget rejection. Keep the original reward, scientific thresholds, actor capacity, J1 freeze, clip5, optimizer, schedule, group16, finite8 budget and434-identity task population. No additional heuristic penalty is justified by this audit.

|Frozen-policy TRAIN comparison|Original hard/total|Corrected hard/total|Original→corrected full success|Overpacked complete|Aborts|
|---|---:|---:|---:|---:|---:|
|Original initializer,4 fitting identities|{fitold['hard']}/{fitold['N']}|{fitnew['hard']}/{fitnew['N']}|{fitold['full']}→{fitnew['full']}|{fitold['overpacked']}→{fitnew['overpacked']}|{fitold['aborts']}→{fitnew['aborts']}|
|Frozen update500,6 affected reserved-TRAIN identities|{aold['hard']}/{aold['N']}|{anew['hard']}/{anew['N']}|{aold['full']}→{anew['full']}|{aold['overpacked']}→{anew['overpacked']}|{aold['aborts']}→{anew['aborts']}|

Tasks/prefixes/goals and candidate-indexed streams are matched. The affected subset is selected because it failed historically, so it is mechanistic evidence, not an unbiased performance estimate. The fitting controls have only4 identities. No neural weights were changed in these comparisons; no generalization or RL-performance confidence claim follows.

At the first provably doomed action for the five newly overpacked candidates, valid alternatives among8 single-prefix proposals numbered5,3,7,0,0. First valid alternatives for the first three yielded24/24 hard-valid sampled suffix completions, zero full property successes. The other two have no acceptable proposal in the unchanged finite budget: the exact filter must abort rather than pretend to rescue them. Locally feasible alternatives are not guarantees of full-goal reachability. Single-prefix forwards may differ in floating-point rounding from the original batch8; cached behavior replay itself was separately exact.

Completed overpacking is eliminated by construction, but some failures become explicit aborts. The correction does not solve the two original proposal-exhaustion cases or the regional-property learning problem. Raw actor quality must be tracked separately so deterministic filtering cannot be promoted as neural learning.

Measured historical reward does not reward invalid geometry: all286 overpacked completions have reward0 and negative advantages. Burial/contact correlation and acceptance-branch amplification support the need for constraint-consistent execution; they do not establish an erroneous loss or a need for a new reward. Increasing clip norm, adding networks or simply resuming update500 is unsupported.

The mathematical and engineering prerequisites for reviewing ONE substantial corrected run are met. Do not insert another100–250-update scientific performance pilot. Existing immediate stability gates still apply.''')
 md('PRODUCTION_RECOVERY_QUALIFICATION.md',f'''# Corrected production recovery qualification

PASS at source `{source}`. The actual full17,438,114-parameter independent actor, fresh AdamW moments, scheduler, complete RNG states, identity order/next goal, counters, config/data/reference/source hashes and durable raw proposal journal were used. No J1 parameter/buffer update occurred.

Reference execution:2 engineering updates. Recovery execution sequentially interrupted after durable rollout/before optimizer, after accepted commitment/before checkpoint, and inside a real Torch archive write. The latter left a truncated archive and unchanged authoritative index; partial load was rejected. Restarts restored and replayed accepted work, then reached update2. Complete semantic actor/Adam/scheduler/RNG/cursor/counter state and subsequent CPU/CUDA Torch,NumPy andPython draws matched the uninterrupted run exactly.

There were6 physical engineering optimizer executions including replay, not6 production updates. Each branch has32 logical attempts,{qual['recovery']['counters']['complete']} completions and{qual['recovery']['counters']['aborts']} abort; the same two TRAIN identities/streams recur across branches and are not independent biological evidence. Full production updates=0. Engineering weights are isolated and will NOT initialize the proposed campaign.

Actual corrupt-hash rejection and repeated atomic publication passed. Checkpoint bytes={qual['checkpoint_bytes']:,}. Four full images are retained across two isolated dual-slot fixtures, plus the failed-write archive and all fault logs. The two new slots may replace only their own inactive rolling entries; historical V94 files are never retirement targets. All sampled traces/commitments remain durable. No uncommitted update enters the authoritative recovery state.

Exact replay is qualified for the documented RTX5060, FP32 networks/FP64 raw scores, deterministic CUDA, matching active batch16, no-AMP configuration. Cross-device/library bit equivalence is not claimed. Source/config changes require requalification.

After a genuine stop, no automatic restart is allowed. As in the original core, interruption exactly during a reserved-TRAIN development callback is fail-closed: restore/replay then reconcile the same frozen callback before continuation. This is an explicit recovery operation, not permission to bypass a scientific stop. Corrected stop publication updates LIVE_STATUS and refreshes the curves before returning, addressing the stale-status/CSV issue.

The unreviewed-production-entry test fails BEFORE runtime creation. No production authorization receipt exists and no R94R production directory was created. Full launch requires your separate review. See qualification/RELEASE.json, PROTOCOL.json and all retained logs.''')
 md('SUBSTANTIAL_RL_RESTART_PLAN.md',f'''# Substantial fixed-sequence geometry RL restart plan — REVIEW REQUIRED

Decision: **{decision}**. Current authorization covers engineering only; do not launch the full campaign from this review. This is a new controller-version arm, not seamless continuation at V94 update501.

Initialize from the verified ORIGINAL pre-V94 actor (`{parent}`), with the same repaired goal adapter as V94 initialization, state fingerprint `{fresh['actor_fingerprint']}`. Reference archive: `{fresh['path']}`, SHA256 `{fresh['SHA256']}`. Load model weights only, fresh Adam/scheduler/RNG/cursor. The implementation loads the common original initializer directly; it does not copy the historical archive or restore its optimizer. No stable250 checkpoint remains; the only selected earlier durable point is0. The safety-breached500 weights and all engineering weights are excluded from initialization.

Target5000 accepted optimizer updates,16 generation attempts/update=80000attempts; completions, aborts, accepted residues, rejected draws and actual environment transitions counted separately. Same434 qualified TRAIN identities/groups and868 existing witnessed/relaxed goals. Keep the original deterministic identity shuffle/variant schedule seed19317, rather than changing the data to hide failure. Report witness/relaxed metrics separately; transitions between difficulty blocks must not be interpreted as learning.

One algorithmic intervention: exact append-only final contact budget in finite8 acceptance. Original reward, success definitions, geometric supports and masks unchanged; AA externally fixed. Full rejected+accepted trace score with per-goal RLOO, β={cfg['trace_KL_beta']}, fresh AdamW lr1e-6,betas(.9,.999),eps1e-8,wd0; warmup100,flat through4000,cosine to0.1peak at5000; clip5; FP32NN/FP64scores,noAMP. Architecture unchanged and J1 permanently frozen (`{j1}`). No SAC,critic,high-level,reacher,AA learning or horizon expansion.

Recovery every50 updates; immutable named500/1000/2000/3500/5000; latest and best distinct. Retain best0 and250 if selected; later best milestones use their already named files. Do not copy or mutate linked inodes. Per-update raw-trace publication precedes optimization; semantic commitment follows accepted step. All source/config/reference/task/reward hashes and optimizer/scheduler/RNG/cursor/counters reside in recovery. STOP or failure is durable and never triggers automatic restart.

Reserved16 TRAIN development identities stay excluded from fitting. Evaluate at0/250/500/1000/2000/3500/5000 with the original candidate streams and both filtered and raw actor conditions. Keep all original gates: >5pp hard/backbone validity decline, >20% structural diversity decline, original active-geometry KL>1, step KL>.02, nonfinite/corrupt recovery, unsafe space/VRAM. The original hard/backbone/diversity floors0.91875/0.95/12.44911875 remain binding even if a new controller baseline is worse; stronger new-baseline floors are allowed, never weaker ones. No early performance stop merely because K1 stays zero. Major stability failures remain stability results, not proof RL cannot solve the task.

Track per update reward distribution, loss/advantage variance, norms and clipping incidence, raw Gaussian entropy, separately defined reference/step/trace KL, completion, hard validity, overpacking, accepted actions/proposal rejections/finite aborts, exposures and burial on frozen panels, target error and simultaneous success. Record packing slack/risk for diagnosis only. Structural diversity and reward exploitation checks stay independent. Sequence diversity is a property of the supplied fixed sequences, not learned AA design. Use compact telemetry and per-goal breakdowns; do not silently equate proxy reward with task success.

Goal-conditioning sensitivity: predeclare paired target changes from the EXISTING witnessed/relaxed goal pairs with the same fixed sequence and candidate streams; no native-coordinate target. TRAIN diagnostics are monitoring only, never additional selection labels. Final sensitivity uses the frozen32-goal/16-identity prospective panel, comparing requested target changes with achieved exposure/burial and geometry changes. Do not tune on confirmation outcomes.

Final comparison only after5000 updates or a separately reported genuine stop. Primary model=best eligible development checkpoint, final5000 evaluated descriptively when different. Evaluate J1, original actor raw, corrected frozen actor filtered/Bestof8, trained actor raw/filtered K1/K8 and a bounded frozen non-RL search comparator with all rejected proposals/transitions/forward calls/time charged. Freshly evaluate or strictly verify reuse: old V94/V93 FILTERED baseline outputs are NOT samples from the new controller. Use the same corrected filter for primary parent-versus-trained comparisons. Historical V94 stops remain contextual evidence, not approved models.

Primary promotion unchanged: >=5pp K1 gain with paired95% lower>0 clustered by16 independent identities, original validity/KL/diversity safeguards and K8 not worse. Both exposure/burial, continuous errors, aborts, raw actor quality and resource cost must be shown. No TEST access or biological foldability/function claim. Conclude RL LEARNING GO, properly completed RL PERFORMANCE NO-GO, genuine RL STABILITY NO-GO, or INFRASTRUCTURE BLOCKED without conflating them.

Future launch command after separate authorization: `run.sh source/scripts/e017_v94r/production.py production --authorization REVIEW_AUTHORIZATION.json`. The review receipt must have explicit_review_authorized=true and configuration_SHA256=`{qual['binding']['configuration_SHA256']}`. No such receipt was created. The qualified source/config must still match qualification/RELEASE.json; verify zero competing CUDA work and current storage again.

Measured historical update processing≈15.76s/group, total commitment≈16.46s including some overhead. Corrected filtering adds checks, so forecast from corrected qualification/production timings rather than promise this historical rate. The initial expectation is on the order of a day, not an unrestricted campaign. Complete all planned updates when stable; preserve safety stops. No automatic subsequent experiment.''')
 metricdefs=dict(geometry_KL=dict(reference='frozen J1 Gaussian',direction='actor||J1',normalization='per active scalar dimension',aggregation='mean visited accepted states',limit=1),step_KL=dict(reference='same actor before optimizer',direction='old||new',normalization='sum active dimensions',aggregation='max fixed probe states',limit=.02),trace_log_ratio=dict(reference='J1 under identical finite8 packing filter',normalization='sum all rejected+accepted sampled active dimensions per attempt',aggregation='MC mean attempts',is_effective_output_KL=False),latent_Gaussian_entropy=dict(normalization='per active dimension at visited underlying-policy states',is_filtered_output_entropy=False))
 save('METRIC_DEFINITIONS.json',metricdefs)
 md('CONFIGURATION_FIELD_INTERPRETATION.md','''# Configuration field precedence

The immutable training JSON retains historical V93 metadata fields for provenance. `peak_distinct_checkpoint_images=8`, `core_checkpoint_bytes` and the older `measured_checkpoint_bytes` are legacy estimates, not the V94R storage admission rule. Authoritative V94R fields are `conservative_peak_full_images=14`, `qualification_checkpoint_images_included=4`, `production_peak_full_images=10`, `checkpoint_hint_bytes=234881024` and the recalculated `minimum_free_bytes=8120172544` before qualification. qualification/RELEASE.json gives the actual full recovery size209392403 bytes. RESOURCE_AND_STORAGE_REPORT.json gives the remaining production envelope7180648448 bytes after qualification storage is already allocated. Both envelopes include the2GiB operational reserve. No model, scientific reward or training hyperparameter was altered to obtain these accounting bounds.''')
 md('FINAL_E017_V94R_SAFETY_REVIEW.md',f'''# E017 V9.4-R packing safety review

**{decision} — ready for review/authorization, NOT launched.** Source `{source}`. Historical V94 remains unchanged and stopped at500; its original gate is not relaxed. No RL effectiveness conclusion follows from that early stability stop.

1. Hard validity deteriorated through five newly overpacked completions and two finite-filter aborts. Four other packing failures already existed. Nine complete packing failures still satisfy original local/backbone/clash/Rg rules: their failure is a frozen statistical global contact limit, not proof of a steric catastrophe. Accepted-proposal branch changes amplify small policy changes into different trajectories.
2. No positive-reward or advantage bug was found: all286 overpacked TRAIN completions have reward0 and negative advantages. Higher contact density correlates with lower burial proxy, and all five newly overpacked completions became more buried, but this does not prove an optimizer incentive to exceed the cap. The delayed invalidity signal is weak credit assignment, not missing hard reward gating.
3. Gradient clipping is frequent but mathematically consistent. Historical norm51.94750213623047 was reproduced exactly; reward and KL components57.7378/7.7481 combine correctly. Mean16 trajectories, full proposal-score sums, all rejected draws, proper masks, no hidden extra factor. Keep clip5; add explicit telemetry, not an opportunistic larger threshold.
4. Minimal correction: reject only extensions whose current contact count already makes final-N packing impossible (116pairs for64 residues). The old prefix audit computed this bound but never enforced it. Do not prune on a heuristic current density or change any scientific tolerance. Full trace likelihood remains correct for the new deterministic controller.
5. Bounded fitting-TRAIN control hard validity62/64→63/64; affected reserved-TRAIN40/48→43/48. Completed overpacking2→0 and6→0; some failures become aborts. Full successes unchanged1/64 and3/48. Three of five first-crossing interventions admit alternative actions and24/24 sampled suffixes are hard-valid, with zero full property successes. This is safety-mechanism evidence, not neural learning/generalization.
6. CPU tests{cpu['checks']}PASS, exact normalized finite-tree gradient and Gaussian score checks PASS. Full actor/Adam/scheduler/RNG/journal recovery across rollout/commit/real-write interruption and corruption PASS. Six physical engineering optimizer executions including replay; production updates0. J1 and the frozen policies used in the counterfactual audit remain unchanged. An initial missing helper binding was caught before optimizer creation, fixed and retained as failed engineering evidence.
7. Use original pre-V94 initializer with repaired goal adapter and FRESH optimizer/schedule/data cursor. Do not resume update500 or reuse engineering weights. The changed acceptance law defines a new versioned arm; the original objective and scientifically frozen dataset/goals remain.
8. D: current admission snapshot={resource['D_free_bytes']:,}bytes ({resource['D_free_bytes']/2**30:.3f}GiB); conservative remaining production envelope={resource['minimum_for_future_production_including_reserve']:,}bytes ({resource['minimum_for_future_production_including_reserve']/2**30:.3f}GiB), including2GiB reserve. Preferred8GiB is not met, but the proven remaining envelope fits. Recheck immediately before any launch; do not delete evidence or migrate storage.

All engineering workers exited; CUDA compute owners empty. No VALIDATION/TEST inference, SAC, high-level RL, J1 update, new production campaign or destructive cleanup occurred. Check GLOBAL_PACKING_FAILURE_ANALYSIS.md and its exact-contact JSON records, GRADIENT_CLIPPING_AND_LOSS_SCALE_AUDIT.md, FILTERED_POLICY_CORRECTNESS.md, CORRECTIVE_INTERVENTION_EVIDENCE.md and PRODUCTION_RECOVERY_QUALIFICATION.md. SUBSTANTIAL_RL_RESTART_PLAN.md specifies the single5000-update/80000-attempt restart for separate authorization.

Remaining scientific uncertainty: filtering may increase aborts, raw proposal quality may not improve, and regional-property steering is still unproven. Existing gates must stop a new run if those conditions cause a genuine breach. These uncertainties require the planned substantial controlled run, not another open-ended series of tiny performance pilots.''')
 # Publish a closed record after verifying the original stop and new sources.
 status=dict(decision=decision,production_launched=False,new_production_updates=0,engineering_optimizer_executions=6,CUDA_resources_released=True,
   original_V94_unchanged=True,original_stop_checkpoint_SHA256=idx['SHA256'],source_commit=source,configuration_SHA256=sha(R/'TRAINING_CONFIGURATION_AND_SEEDS.json'),
   qualification_release_SHA256=sha(R/'qualification/RELEASE.json'),all_new_artifacts_on_D=True,TEST=False,prospective_evaluation=False)
 save('FINAL_STATUS.json',status)
 resource['D_free_bytes']=shutil.disk_usage(R).free;resource['headroom_above_envelope']=resource['D_free_bytes']-resource['minimum_for_future_production_including_reserve'];resource['storage_GO']=resource['headroom_above_envelope']>=0;save('RESOURCE_AND_STORAGE_REPORT.json',resource)
 checkpoints={str(p.relative_to(R)):sha(p) for p in (R/'qualification').glob('*/checkpoints/recovery_*.pt')}
 partials={str(p.relative_to(R)):sha(p) for p in (R/'qualification').rglob('*.part') if p.is_file()}
 artifacts={str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file() and 'source' not in p.relative_to(R).parts and 'runtime' not in p.relative_to(R).parts and p.suffix!='.pt' and p.name!='PROVENANCE_SHA256.json' and '.part' not in p.name}
 save('PROVENANCE_SHA256.json',dict(schema='E017_V94R_SAFETY_REVIEW_1',decision=decision,source_commit=source,source_SHA256=qual['binding']['source'],
    artifact_SHA256=artifacts,qualification_checkpoint_SHA256=checkpoints,failed_write_evidence_SHA256=partials,protected_historical_hash_audit_SHA256=sha(R/'HISTORICAL_HASH_AND_CHECKPOINT_AUDIT.json'),
    external_protected_references=dict(J1_SHA256=j1,parent_SHA256=parent,V94_stop_SHA256=idx['SHA256'],TRAIN_manifest=qual['binding']['TRAIN_SHA256'],reward=qual['binding']['reward_SHA256']),
    no_self_hash=True,no_model_checkpoint_duplicates_created_for_reporting=True))
 print(json.dumps(status),flush=True)
if __name__=='__main__':main()
