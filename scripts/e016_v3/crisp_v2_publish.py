from crisp_v2_common import *
import subprocess,sys,platform,resource,collections
import scipy,pandas,pyarrow,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[3];audit=read('AUDIT.json');tasks=read('tasks/PUBLIC_TASKS.json');tests=read('DIRECT_STEERING_TESTS.json');cpu=read('CPU_TASK_AND_LEAKAGE_TESTS.json');extended=read('REWARD_AND_EXTENDED_CONTROL_TESTS.json');fit=read('results/INITIAL_ADAPTER_TRAINING_CURVE.json');verify=read('ADAPTER_TRAINING_CURVE.json');f=read('FEASIBILITY_COMPARISON.json');cf=read('COUNTERFACTUAL_AND_REACHABILITY_DIAGNOSTICS.json');cfg=read('configs/FROZEN_PROTOCOL.json');summary=f['summary'];limits=read('configs/NATIVE_CALIBRATION.json');guard()
deps=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,torch_CUDA=torch.version.cuda,cudnn=torch.backends.cudnn.version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pandas.__version__,pyarrow=pyarrow.__version__,matplotlib=matplotlib.__version__,platform=platform.platform(),GPU='NVIDIA RTX5060 8GiB',environment_changes='none',external_network_queries=0)
write('DEPENDENCIES.json',deps)
fig,axes=plt.subplots(1,3,figsize=(13,4.8));names=list(summary);xx=np.arange(len(names));short=['Frozen','Supervised\njoint','AA only','Geometry\nonly','Joint +\nrewind','Frozen +\nrewind']
for ax in axes:ax.set_xticks(xx,short,rotation=35,ha='right')
axes[0].bar(xx,[summary[n]['completed']/summary[n]['attempts']for n in names],label='completed');axes[0].bar(xx,[summary[n]['administratively_truncated']/summary[n]['attempts']for n in names],bottom=[summary[n]['completed']/summary[n]['attempts']for n in names],label='safety truncation',alpha=.45);axes[0].set_ylabel('Attempt fraction');axes[0].set_title('Completion / censoring');axes[0].legend(fontsize=8)
axes[1].bar(xx,[summary[n]['motif_satisfaction']if summary[n]['completed']else np.nan for n in names]);axes[1].set_ylabel('Both target residues satisfied (fraction)');axes[1].set_title('Completed episodes only; higher better')
axes[2].bar(xx,[summary[n]['endpoint_error_mean_A']if summary[n]['endpoint_error_mean_A']is not None else np.nan for n in names]);axes[2].axhline(8,ls='--',color='gray',label='target tolerance');axes[2].set_ylabel('Mean endpoint error (A; lower better)');axes[2].set_title('Completed episodes only');axes[2].legend(fontsize=8)
fig.suptitle('E017 V2 bounded feasibility: 32 attempts per arm, 8 VALIDATION parents');fig.text(.5,.005,'Two P1/P2tasks and two matched seeds per parent. Bars are descriptive point estimates; parent-bootstrap success intervals are in FEASIBILITY_COMPARISON.json.\nAdministrative truncation is censoring, not biological failure. No complete-task successes; no causal-performance advantage claimed.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.12,1,.94))
for ext in ['png','pdf']:fig.savefig(R/f'FEASIBILITY_COMPARISON.{ext}',bbox_inches='tight')
plt.close(fig)
def md(n,s):
 import re
 words='TRAIN|VALIDATION|cached|complete|canonical|designs|sequences|unique|studies|tasks|updates|parameters|parents|attempts|successes|failures|residues|one|both|components|positions|groups|nats|bytes|calls|training|actor|examples|recycles|threads|targets|decisions|individual|local|stable|random|geometric|native|constrained|primitive|parameter|canonical|proposal|tuple|way|input|rewards|residue|optimizer'
 s=re.sub(r'(\d)('+words+r')\b',r'\1 \2',s);s=re.sub(r'\b(first|all|uses|leaving|excluding|at|with|of|total|than|above|below|mean|median|versus|length|case|these|only)(?=\d)',r'\1 ',s);s=re.sub(r',(?=[A-Za-z])',', ',s)
 (R/n).write_text(s)

md('TASK_DATASET_AND_GOAL_PROTOCOL.md',f'''# P1/P2 task protocol

No earlier E017 artifact was found in the inspected D: experiment roots. These are explicitly defined V2 tasks, not assumed historical definitions.

Select 16 TRAIN and 8 VALIDATION complete canonical identities of length 64–96 in deterministic SHA order, seed tag E017v2. One qualified observation per identity; distinct declared protected groups across roles. Existing accepted PDB v3 split is unchanged. Coordinate files, observation IDs, row/model mappings and original source hashes are retained in tasks/SOURCE_PROVENANCE.json. No TEST evaluation or retired prefilter reconstruction.

Both tasks start from three publicly supplied native residues/CA anchors. They are conditional construction tests, not unconditional generation. Targets and coordinates are expressed relative to the first-three-anchor right-handed frame. The policy must generate all remaining residues and coordinates. The hidden full chain is only used for TRAIN teacher forcing and posthoc native witnesses; it is not a policy input.

- P1: two specified canonical residues at zero-based floor(N/3), floor(2N/3), plus final CA target within 8A.
- P2: P1 plus midpoint CA target within 8A and final local-frame orientation within 45degrees, using the SO(3) geodesic angle. The target frame is right-handed. Other motif strings can be expanded into constrained-residue maps; this feasibility panel uses two single-residue constraints.

Targets are all explicit task information. Only initial anchors, requested constraints, current generated prefix and prior intervention results are observable. Position features include N and t/N; sparse target residues are visible only as declared constraints, not the full native suffix. No future native coordinates enter J1 features. Goal context and likelihood labels are separate.

TRAIN-only geometric calibration (all unusual selected observations retained):

{json.dumps(limits,indent=2)}

All 48 native witnesses satisfy the frozen task/validity rules. Replaying native distance/angle/signed-torsion actions reproduces the mapped native CA trace to max error {cpu['max_native_builder_replay_error_A']:.3g}A. This proves existence of a solution under the chosen seed and action semantics, not that the learned policy can reach it reliably. Split protections do not certify removal of all remote homology.
''')
md('DIRECT_STEERING_ARCHITECTURE.md','''# Output adapters

J1 remains exactly 12,351,746 parameters: 8 bidirectional prefix Transformer blocks, width352, 8 heads, RoPE, unchanged25-channel geometric pair interface and internal-coordinate builder. Its parameters and buffers are frozen; no causal conversion or KV cache is introduced. A single frozen model supplies both actor reference distributions and pretrained context; no checkpoint copy is needed.

New adapter parameters: **53,530**, entirely separate from J1.

- AA: Linear394→64, SiLU, Linear64→20:26,580 parameters.
- Geometry: Linear414→64, SiLU, Linear64→6:26,950 parameters.

Inputs: detached J1 prefix representation352; goal/state context42; geometry additionally receives selected-AA one-hot20. The42-vector contains normalized N/progress, P1/P2 indicators, endpoint residual in the current local frame, relative desired frame, currently constrained-AA one-hot, distance in sequence to next constraint, constraint-presence flag, selected subgoal residual and branch-context summary. Physical displacement is divided by3.8*remaining residues; this scales only new goal inputs, not any original J1 feature. Branch summary saturates after10 in the feature, but does not restrict interventions; full branch history remains available to the high-level interface.

AA logits: l'=l+2*tanh(f_A). Geometry latent mean: mu'=mu+[0.05,0.35,0.50]*tanh(f_G[:3]); logstd'=clip(logstd+0.2*tanh(f_G[3:]),−5,2). Geometry remains conditioned on the just-selected AA. All final adapter layers initialize to zero, yielding exact baseline equality and zero KL. Hidden layers receive gradients after the zero readouts begin learning.

AA is categorical. Distance=exp(u_d), angle=pi*sigmoid(u_a), signed torsion=pi*tanh(u_tau), unchanged Jacobians and right-handed construction. Sampling retains corrected FP64 finite-grid support repair; latent and represented-action densities are logged separately. Finite-grid endpoint repair is an inference convention, not a discrete probability mass or a new trainable density. Differentiable tests use unsaturated invertible samples and fail rather than silently assigning an invalid continuous gradient.

Exact joint KL: KL(AAadapt||AAref)+sum_a pi_adapt(a)*KL(N_adapt,a||N_ref,a). The latter equals physical-space KL under the shared bijection, on valid components. Tasks start at t=3, so all3components are valid; empty-prefix adapters/partial-frame goal features are outside this initial interface. Geometry KL is normalized by3 for regularization, AA per decision. AA/G modules have separate gradients/clipping, and J1 shared gradients are absent. No old J1 optimizer state is attached to adapter parameters.

The supervised pilot uses16 TRAIN parents,252 cached prefix/task examples and80 AdamW updates, effective batch16. Half-targeted sampling is implemented by the frozen feature pool containing both constrained positions plus random positions; actual minibatches are uniform over this pool. Thus the pool does **not** realize an exact50/50 decision quota, despite the initial proposal text. Actual exposure is logged; no exact-quota claim is made. AA CE plus0.10 local geometry NLL and0.01 each KL term. TRAIN gradient calibration gave AA1.1404/G1.2894 before weighting; chosen geometry coefficient is the capped proposal0.1. Separate adapter gradient clipping norm5; LR3e−4, weight decay0.01. No geometry/sequence weights or objective were retuned from VALIDATION.

The 80-update state includes adapter weights, AdamW, sampler/RNG, exposure/history, config, reference path/hash and frozen tensor hash. The auxiliary masked-reconstruction head from E016 is not loaded. Supervised-fit base revision is recorded in the checkpoint; final reproducibility source is pinned by the E017 manifest, with executed-source differences disclosed.
''')
md('HIERARCHICAL_MDP_AND_ACTION_SEMANTICS.md','''# Hierarchical semi-MDP

State: public goal, N, current generated sequence/CA/valid masks, latest prefix checkpoints, low-level branch-context state, and irreversible intervention results. The frozen J1 reference sees only the prefix. A domain-independent typed HighAction/ControlledWorld protocol uses strings/mappings; ProteinWorld is the protein boundary. No robotics controller is implemented.

CONTINUE selects endpoint or midpoint subgoal context and generates one or more residues. Each residue first samples categorical AA, then the AA-conditioned three-component continuous geometry distribution. Bidirectional attention is recomputed inside the exact generated prefix. Every residue consumes one primitive clock tick; a k-step CONTINUE option has duration k.

REWIND restores an earlier valid checkpoint on the current branch, including sequence, CA, masks, retained prefix action history, random generator and low-level context. It consumes one tick. The irreversible intervention audit log and global unique branch serial are not erased. No arbitrary internal segment edits are claimed.

REGENERATE follows REWIND, starts a new reproducible stream seed=episode_seed+1,000,003*branch_serial, records its context and saves a new checkpoint for the suffix. It consumes one tick; subsequent CONTINUE generates the suffix. Literal replay without REGENERATE uses restored RNG/context and reproduces the suffix exactly. Checkpoints are latest-branch prefix snapshots; arbitrary historical-branch editing is not provided.

STOP consumes one tick and terminates. Success is granted only for a complete task satisfying all declared goals and validity rules; incomplete STOP is a failure. Administrative safety truncation is not STOP and not a biological failure.

No correction budget enters the objective. The feasibility harness has a3*N primitive-event and30s walltime safeguard, plus a1,200s GPU-worker cap. These prevent uncontrolled execution only. Truncated episodes are censored, require appropriate nonterminal bootstrap in later RL, and are reported with success lower/upper bounds. They must not be silently used as negative biological training examples.

The comparison high-level managers are fixed heuristics, not learned policies. They rewind on observed missed residues/waypoints, completed-goal failure, or an exact-endpoint-center contour trigger. The original source named this last trigger necessary reachability; strictly, it is stronger than necessary for reaching the allowed8A target ball. Its actual baseline behavior is retained and documented, not claimed as a reachability proof. The separate necessary_position_reachability function includes target tolerance and is explicitly tested. Neither test certifies orientation/contact-constrained long-horizon reachability.
''')
md('MATHEMATICAL_REWARD_SPEC.md','''# Reward and regularization

Binary terminal success Y=1 iff the complete chain has all constrained residues, endpoint and (P2) midpoint/orientation targets within tolerance, finite coordinates, native-calibrated adjacent spacing, clash/Rg bounds, and no degenerate local frames. Otherwise Y=0 at an explicit STOP. There is no AF confidence or predictor reward, proposal-count penalty, runtime penalty, retry budget or hard optimization constraint on corrections.

Primitive events have indices t=0,...,T−1. Terminal STOP reward is Y, discounted return gamma^(T−1)*Y with gamma=0.995. The early comparison receipt used gamma^T as a descriptive terminal number; all192 successes are zero, so this did not affect results. The reward ledger and positive native-terminal unit test use the explicit event-index convention. This bookkeeping difference is disclosed rather than used to claim an improvement.

Optional potential: Phi(s)=−min(1, ||target_end−current_end||/(3.8*max(1,N−t))) for nonterminal states, Phi=0 for absorbing terminal states. Shaping r_shape=0.1*(gamma*Phi(next)−Phi(current)) is logged for every residue, REWIND, REGENERATE and STOP, including discarded suffix decisions. Its discounted sum telescopes to −0.1*Phi(initial)+0.1*gamma^T*Phi(terminal). The full reward-event ledger was added after the feasibility comparison and separately tested without changing the measured policy actions; the comparison itself used binary terminal outcomes only. Optional terminal proximity guidance is disabled initially.

Low-level regularization is a separate optimization term: beta_A times the discounted mean AA KL per proposed AA decision, plus beta_G times the discounted mean conditional geometry KL per valid geometry component. beta_A=beta_G=0.01 is a TRAIN-calibrated feasibility starting value, not a production RL optimum. Include discarded suffix proposals in these averages; do not compute KL only on the accepted final chain. Normalize by the corresponding discounted valid-decision counts so target length does not trivially scale the regularizer. The reference is immutable J1+7,500. There is no pretrained high-level correction policy to pretend is a high-level KL reference.

For option rewards r_0...r_(k−1), Bellman target=sum_i gamma^i*r_i + gamma^k*V(next), with bootstrap zero only on true termination. Administrative truncation remains nonterminal. This differs from the older E016 gamma=1 helper; option_target implements and tests the CRISP duration-aware target. Exact20-way AA expectation/score-function estimators are valid; backpropagation through integer sampling is not.

Validity, sequence plausibility, diversity, proposals and actual compute remain separate reports. Meeting these geometric targets does not establish thermodynamic stability, foldability or function.
''')
table='| Arm | Attempts | Completed | Safety truncations | Successes | Both residues, completed | Endpoint error A, completed | J1 calls | Seconds |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for arm,v in summary.items():
 end='undefined'if v['endpoint_error_mean_A']is None else f"{v['endpoint_error_mean_A']:.2f}";mot='undefined'if not v['completed']else f"{v['motif_satisfaction']:.3f}";table+=f"| {arm} | {v['attempts']} | {v['completed']} | {v['administratively_truncated']} | {v['successes']} | {mot} | {end} | {v['J1_calls']} | {v['seconds']:.1f} |\n"
md('FEASIBILITY_RESULTS.md','# Bounded feasibility comparison\n\n'+table+'\nEight VALIDATION parents, both task kinds and two matched streams each,32attempts per arm. All original attempts/failures/truncations are retained. AA-only and geometry-only gate a single joint-supervised adapter; they are not independently trained lineages. Supervised goal joint is the learned low-level adapter with continuation only; joint_subgoal_rewind adds a fixed subgoal/correction heuristic. Frozen rewind uses the same correction heuristic without adapters. No high-level or low-level RL update occurred.\n\nNo full-task successes were observed. Parent-cluster intervals and censoring bounds are in FEASIBILITY_COMPARISON.json. A zero observed lower-bound success rate does not prove the censored managers have zero eventual success; nor does successful software execution establish planning gains. Conditional geometry/sequence metrics of completed arms must not be compared to missing completed outputs from censored arms.\n\n![Feasibility](FEASIBILITY_COMPARISON.png)\n')
md('FAILURES_LEAKAGE_AND_LIMITATIONS.md','''# Failures and interpretation

- Original rewind replay failed because irreversible intervention-count features changed despite restoring RNG/prefix. Fixed low-level context restoration while retaining high-level audit history. Original failed test retained; no extra fitting required.
- CPU diagnostics initially failed JSON publication on numpy.bool_; converted diagnostic results to Python bool and reran. No policy/data change.
- Complete reward-event and branch-context bookkeeping added after the comparison; exact comparison core archived under results/. The additional tests verify literal branched replay, recovery, shaping telescoping and unchanged sampled actions for all four direct arms. They do not claim a fresh192episode comparison under a different trained policy.
- The fixed manager's center-contour trigger is conservative and ignores the endpoint-ball tolerance; it is not a necessary reachability classifier. The separately tested necessary bound includes tolerance. Full task outcomes still use the original frozen tolerance; no reward safeguard was relaxed.
- The initial protocol proposed50%motif positions. The implemented cached pool contains both motifs plus random positions, then draws minibatches uniformly. Exposure is measured, not asserted to be50/50. No sampler claim is used to explain gains; there are no demonstrated full-task gains.
- Three native anchor residues/coordinates are supplied as explicit conditioning. No claim of unconstrained de novo generation. Public goals legitimately reveal sparse native target information; full native suffix remains hidden from policy input. TRAIN teacher-forcing targets are labels, not state features. Native future perturbation and public-schema tests pass.
- No TRAIN/VALIDATION protected-group overlap in the selected parents. Existing split certification does not imply universal evolutionary independence. No TEST metrics.
- Zero complete task successes after80adapter updates. Rewind loops often hit the administrative safeguard; censored outcomes are not biological failures. More training is not automatically justified, and no prolonged RL may follow from this foundation result.
- Counterfactual diagnostics show local output/action sensitivity to goal changes; no learned causal-influence or reachability representation was trained. Kinematic bounds and native witnesses are not long-horizon policy reachability guarantees.
- All J1 parameters/buffers remain bitwise fixed. New adapter parameters can still alter AA-conditioned generated geometry indirectly. No AlphaFold, external MSA or structural evaluator was used.
''')
md('NEXT_STAGE_RL_PLAN.md','''# Next-stage plan: pending review, not executed

Engineering gate: task/native mapping, support/log-probabilities, output steering, independent gradients, exact GPU restart, hierarchical rollback and reward bookkeeping pass. Scientific control gate: not passed; the small adapter and fixed managers show no complete task successes. Do not launch prolonged SAC or full J1 fine-tuning now.

First authorized follow-up should establish a reliable goal-conditioned continuation baseline on TRAIN, retaining this exact benchmark as a frozen evaluation. Use a separately prospective curriculum with one residue or one positional target before the combined P1/P2 constraints; do not retroactively relabel current failures or relax its targets. Keep J1 frozen. Measure held-out per-constraint response and off-native-prefix behavior; local goal sensitivity alone is insufficient. Set a small explicit engineering walltime/update authorization for that future task, not a correction-count term in its objective.

Only after a viable direct-steering baseline: train adapters using goal-conditioned episode policy gradients or the existing hybrid SAC foundation, with duration-aware gamma0.995 targets, normalized AA/G reference KL and optional potential shaping. Manager choices remain CONTINUE/subgoal, REWIND, REGENERATE, STOP. Actor replay must include branch-context state, RNG identity, all generated proposals and actual per-action physical log densities. Treat truncation as censoring/nonterminal; never invent rewards for missing outcomes. Geometry gradients must not attenuate AA updates through an ordinary global clip. New manager learning requires an explicit separate policy implementation/calibration; current managers are heuristics.

Eventual ablation sequence: frozen J1; supervised goal conditioning only; AA/G/joint low-level RL; correction without learned intervention models; intervention representation added; reachability representation added; both combined. Match tasks, initialization, seeds and evaluation statistics. Report compute/proposals without inserting a scientific computation budget into the objective. Repeatedly improved critic values are not task-success evidence. Use at least three independent future training seeds and parent-cluster uncertainty once sample sizes support it.

Dependencies are pinned in DEPENDENCIES.json and the source SHA manifest. Existing Torch/CUDA, NumPy/SciPy, pandas/pyarrow, matplotlib and the local protein_joint_sac source suffice. No new external models, services, libraries or robotics stack are required. Existing E016 SAC synthetic results remain historical foundations, not real task-return certification. Counterfactual intervention and reachability model training is postponed.

This handoff authorizes no further execution. Review the weak control outcomes before selecting the next bounded experiment.
''')
# Parent-level paired secondary metrics; successes at the boundary need a separate caution.
rows=read('FEASIBILITY_EPISODES.json');rng=np.random.default_rng(170025);secondary={}
for arm in ['supervised_goal_joint','AA_only','geometry_only']:
 secondary[arm]={}
 for key in ['position_error_A','valid','motifs']:
  dif=[]
  for identity in sorted({r['identity']for r in rows}):
   def vals(label):return [float(r['assessment'][key])for r in rows if r['identity']==identity and r['arm']==label and r['status']=='COMPLETED']
   a,b=vals(arm),vals('frozen_J1');dif.append(np.mean(a)-np.mean(b))
  boot=np.mean(rng.choice(dif,(4000,len(dif))),axis=1);secondary[arm][key]=dict(mean_delta=float(np.mean(dif)),parent_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),parents=len(dif))
write('PARENT_PAIRED_SECONDARY_METRICS.json',dict(results=secondary,boundary_caution='zero-success empirical bootstrap is degenerate, not proof of zero future success; only8parent clusters; protected groups do not prove evolutionary independence'))
sourcecommit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip();resources=dict(initial_D_free_bytes=12303400960,final_D_free_bytes=shutil.disk_usage(R).free,reserve_bytes=10*2**30,adapter_unique_updates=80,discarded_restart_update_calls=4,J1_optimizer_updates=0,manager_RL_updates=0,critic_updates=0,fit_and_initial_tests_seconds=fit['seconds'],verify_only_tests_seconds=verify['seconds'],CPU_checks_seconds=cpu['seconds'],comparison_seconds=f['seconds'],extended_checks_seconds=extended['seconds'],interface_checks_seconds=read('HIGH_LEVEL_INTERFACE_TESTS.json')['seconds'],recovery_and_probe_seconds=read('SHORT_HORIZON_INTERVENTION_PROBE.json')['seconds'],comparison_J1_calls=sum(v['J1_calls']for v in summary.values()),comparison_attempts=sum(v['attempts']for v in summary.values()),comparison_rewinds=sum(v['rewinds']for v in summary.values()),GPU_peak_allocated_bytes=max(fit['GPU_peak_allocated_bytes'],f['GPU_peak_allocated_bytes']),comparison_max_RSS_bytes=f['host_max_RSS_bytes'],new_external_inference_calls=0,historical_deletions=0,one_GPU_worker_at_a_time=True,source_base_revision='b3af952083099520e3fbebdda97130ad05f1b098',source_commit=sourcecommit,phase_cap_seconds=1200)
write('RESOURCE_REPORT.json',resources)
md('EXPERIMENT_DECISIONS.md','''# E017 CRISP V2 decisions

2026-10-21: user selects CRISP, hierarchical goal-conditioned steering/correction; no hard scientific computation/correction budget. Retain discounted terminal success, bounded native-calibrated geometry guidance and normalized KL-to-reference.

Create isolated E017 worktree from E016 R1e b3af952; leave E010 main checkout and all E016 evidence unchanged. Verify J1+7,500, accepted+500 and original J1-FULL hashes; preserve historical checkpoint qualification.

Define explicit seed-conditioned P1/P2 tasks, freeze TRAIN/VALIDATION roles and tolerances before validation. No prior E017 artifact found; do not imply earlier definitions were verified. Use native witnesses and independently replay internal actions.

Freeze J1 and fit only53,530 output-adapter parameters for80updates. Separate AA/G controls and normalized KL; no full J1, manager, critic or RL fitting. Initial zero effect, trained-only gradient calibration, precise sampler exposure and recovery state recorded.

Preserve initial rewind failure and fix context restoration. Record JSON serialization failure. Add/test complete reward ledger without changing direct-arm actions. Preserve exact comparison implementation and disclose conservative manager trigger and proposed-versus-realized supervision mixture.

Stop after192attempt feasibility comparison and bounded correctness checks. No complete-task successes; engineering readiness does not establish scientific planning benefit or justify prolonged RL. Learned intervention/reachability models and robotics remain future work.
''')
md('CHANGELOG.md','''# E017 CRISP V2 changelog

Added compact verified task construction; domain-independent high-level action protocol; frozen-J1 AA/geometry output adapters; exact joint KL and density/support tests; deterministic prefix rollback/branch regeneration and recovery; discounted event reward/potential shaping; supervised adapter-only pilot; six-arm matched feasibility comparison; counterfactual/reachability diagnostics; handoff, plots and provenance. No E016 artifact, pretrained parameter, split or normalization changed.
''')
md('FINAL_E017_CRISP_V2_HANDOFF.md',f'''# E017 CRISP V2 scientific handoff

**Working steering/correction foundation; no demonstrated complete-task improvement. Stop before full RL.**

Verified frozen J1+7,500 and two historical comparator checkpoint hashes. The unrelated main repository checkout remains untouched; isolated branch/source and final SHA manifest are below. No previous E016 data or evidence modified.

## Implemented

- 48 P1/P2 task instances from16TRAIN and8VALIDATION complete PDB parents, length64–96. Sparse sequence constraints, relative positions/orientations and TRAIN-calibrated validity; first3native anchors explicitly supplied. All48native solutions replay successfully with max CA error{cpu['max_native_builder_replay_error_A']:.3g}A.
-53,530 goal-conditioned adapter parameters over frozen12,351,746parameter J1. Independent AA and selected-AA-conditioned distance/angle/signed-torsion steering. No architecture, builder, input normalization or bidirectional-prefix semantics changes.
- Domain-independent typed CONTINUE/subgoal, REWIND, REGENERATE and STOP interface, with protein-specific state adapter. Exact RNG/context/prefix recovery; immutable intervention history and branch semantics. No causal KV cache or arbitrary internal edits.
- Binary task success, gamma0.995 terminal preference, bounded potential shaping and separately normalized AA/G KL. No correction-budget objective.3*N/30s administrative episode safeguards are censoring, not scientific failure labels.

## Measured feasibility

{table}

One supervised adapter lineage:80unique updates,4discarded GPU restart calls, zero J1/RL/critic updates. Fit uses{fit['feature_examples']} cached TRAIN prefix-task examples;{fit['TRAIN_identities']} identities and{fit['unique_identity_position_pairs']} unique identity-position pairs actually visited. The tiny task panel is intentionally not the full118,219identity E016 population.

Counterfactual same-noise checks: AA categorical decisions changed in{sum(v['same_draw_AA_changes_of512']for v in cf['rows'])} of{512*len(cf['rows'])} local sampled comparisons; geometry actions also changed. Goal perturbations alone change AA-logit vector L2 by only about0.009–0.010 and geometry means by about0.001; much of the adapter effect is therefore not demonstrably strong goal control. The two-state matched-noise short-horizon torsion intervention probe is in SHORT_HORIZON_INTERVENTION_PROBE.json. These establish local steering/sensitivity, not long-horizon success. Complete-task successes are0across all192attempts. Censored managers have unresolved eventual outcomes; no lower-bound success advantage is demonstrated. Per-parent uncertainty and detailed outputs are in FEASIBILITY_COMPARISON.json.

## Correctness and limitations

All direct-steering, GPU interrupted-restart, native-action replay, support/Jacobian/KL, typed-interface, hidden-future, split-isolation and extended reward/control tests pass. Frozen J1 tensors/buffers remain bitwise equal. The initial failed rewind-context and CPU serialization checks are preserved with repairs.

The comparison ablates AA/G gates on one supervised joint adapter; it does not isolate the effect of independent RL training. Managers are fixed heuristics and often loop until the administrative safeguard. No learned intervention/reachability model exists yet. The exact-center contour trigger is a conservative heuristic, not a correct necessary test for the tolerated target ball; a separate tolerance-aware necessary bound is tested. No complete task criterion was relaxed.

A full event reward ledger and branched-context restoration were finalized after the comparison; exact executed source is archived and matched sampled-action checks pass for the four direct arms. Primitive clock and early terminal-discount receipt conventions are explicitly distinguished. The original proposed50%motif sampling quota was not realized by the uniform cached-pool implementation; actual exposures are retained and no50/50claim is made.

## Resources and recommendation

Comparison walltime{f['seconds']:.1f}s;{resources['comparison_J1_calls']:,} actual J1 prefix calls; peak allocated GPU memory{resources['GPU_peak_allocated_bytes']/2**20:.1f}MiB; comparison maximum host RSS{resources['comparison_max_RSS_bytes']/2**30:.2f}GiB. D: free{resources['final_D_free_bytes']/2**30:.3f}GiB with10GiB reserve protected. No external model calls, downloads, deletion, robotics or prolonged optimization.

**Engineering GO for a bounded next baseline study; NO-GO for prolonged CRISP/SAC training from these results.** Establish held-out direct goal control before attributing benefits to hierarchical correction, intervention modeling or reachability representations. No automatic continuation.

Detailed handoff:

- [Repository/checkpoint audit](AUDIT.json)
- [Tasks and goal protocol](TASK_DATASET_AND_GOAL_PROTOCOL.md)
- [Steering architecture](DIRECT_STEERING_ARCHITECTURE.md)
- [Hierarchical MDP](HIERARCHICAL_MDP_AND_ACTION_SEMANTICS.md)
- [Mathematical reward](MATHEMATICAL_REWARD_SPEC.md)
- [Feasibility results](FEASIBILITY_RESULTS.md)
- [Failure/leakage limitations](FAILURES_LEAKAGE_AND_LIMITATIONS.md)
- [Next-stage plan](NEXT_STAGE_RL_PLAN.md)
- [Full recovery/checkpoints](CHECKPOINT_MANIFEST.json), [dependencies](DEPENDENCIES.json), [resources](RESOURCE_REPORT.json), [SHA provenance](PROVENANCE_SHA256.json)

![Feasibility](FEASIBILITY_COMPARISON.png)
''')
print('HANDOFF PUBLISHED',summary,flush=True)
