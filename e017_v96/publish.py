"""Publish evidence-derived review package, never start training."""
import datetime, math
from common96 import *


def doc(name, text):
    with (ROOT/name).open('x') as f:f.write(text.strip()+'\n')


def main():
    signal=read(ROOT/'REWARD_SIGNAL_EVIDENCE.json');opt=read(ROOT/'OPTIMIZATION_EVIDENCE.json')
    obs=read(ROOT/'OBSERVABILITY_EVIDENCE.json');swap=read(ROOT/'GOAL_SWAP_EVIDENCE.json')
    early=read(ROOT/'EARLY_ACTION_EVIDENCE.json');grad=read(ROOT/'COMPONENT_GRADIENT_EVIDENCE.json')
    tests=read(ROOT/'qualification/PROTEIN_TESTS.json');release=read(ROOT/'qualification/RELEASE.json')
    panel=read(ROOT/'VALIDATION_PANEL_MANIFEST.json');config=read(ROOT/'TRAINING_CONFIGURATION_AND_SEEDS.json')
    assert tests['PASS'] and read(ROOT/'qualification/SYNTHETIC_TESTS.json')['PASS']
    for name,expected in read(ROOT/'PRE_DIAGNOSTIC_PLAN.json')['source96_SHA256'].items():
        actual=sha(ROOT/'source96'/name)
        if actual!=expected:
            change=read(ROOT/'qualification/METADATA_COUNT_FIX.json')['source_changes'][name]
            assert change['before']==expected and change['after']==actual and change['metadata_only']
    ready=bool(release['PASS'])
    binding=read(ROOT/'HISTORICAL_BINDINGS.json')
    for model in binding['models'].values():assert sha(model['path'])==model['SHA256']
    p=read(V95/'PROVENANCE_SHA256.json')
    for name in ['FINAL_E017_V95_VALIDATION_REPORT.md','GOAL_CONDITIONING_ANALYSIS.md','EXPANDED_VALIDATION_RESULTS.json']:
        assert sha(V95/name)==p['files'][name]['SHA256']
    doc('BURIAL_GOAL_OBSERVABILITY.md',f'''# Burial goal observability

Canonical V9.1 serialization includes burial interval and threshold on the original exact 1e-4 grid. public_batch context columns 2,6,7 carry burial maximum and interval endpoints/N; 13,14,27,33,35 carry prefix burial observations, observed fraction, final exposure upper bound, knownness and residual. Query projection concatenates this context with frozen J1 features. Seven Transformer blocks feed residue-conditioned geometric mu/log-scale heads. Burial is not dropped or aliased.

Representation asymmetry: exposure/motif instructions have explicit goal tokens anchored at the exposed region through RoPE. Burial has only global query scalars, with no corresponding regional token. This is a demonstrated architectural asymmetry, not proof of a causal defect or information-theoretic unobservability.

{len(obs['probes'])} same-prefix probes at lengths 0,8,32,48 compare original and update5000 actors on two TRAIN identities. All burial observations differ; usable burial input gradients observed: {obs['any_burial_input_gradient']}. Exact context differences, mu/log-scale changes and input Jacobian norms are in OBSERVABILITY_EVIDENCE.json. Targets, sequence and motif are fixed except the indicated region. The canonical public observation uses generated prefixes only, never native future coordinates.

The proposed single change adds one explicit burial token using the existing goal projection and attention blocks. It adds zero parameters and does not delete the existing burial context or motif tokens.
''')
    lines=['# Reward and advantage audit','','Original reward is unchanged: hard-valid gate times [0.5 full-success + 0.1 each exposure, burial, contrast, motif and compatibility utility]. Abort reward is zero. Fixed sequence makes motif achievement largely a validity/completion term; it is not an AA learning signal.','',f"Historical audit sampled {signal['attempts']} trajectories from 64 equally spaced original training updates, selected by index rather than outcomes. Total reward–burial satisfaction correlation: {signal['reward_burial_satisfaction_correlation']:.6f}; burial satisfaction: {signal['burial_satisfaction_rate']:.6f}; hard validity: {signal['hard_valid_rate']:.6f}.",'','| Component | Mean weighted reward | SD | Nonzero rate | Correlation with total |','|---|---:|---:|---:|---:|']
    for j,name in enumerate(signal['component_names']):lines.append(f"| {name} | {signal['component_mean'][j]:.8f} | {signal['component_SD'][j]:.8f} | {signal['component_nonzero_rate'][j]:.6f} | {signal['component_reward_correlations'][j]} |")
    lines += ['', 'Per-goal group-relative advantages are leave-one-out augmented terminal returns: A_i=(R_i-beta log(P_i/Q_i)) - mean_{j!=i}(R_j-beta log(P_j/Q_j)). There is no same-group whitening or advantage clipping. The same trajectory coefficient multiplies every sampled proposal score, including rejected proposals. This estimator is unbiased under independent trajectory draws; it does not identify which early decision created a late regional contact.', '',f"Clipping occurred on {opt['clipped']}/{opt['updates']} updates ({100*opt['clip_fraction']:.2f}%). Gradient-norm quantiles [min,Q1,median,Q3,p95,p99,max]: {opt['gradient_norm_quantiles']}. Median clipping scale: {opt['clip_scale_quantiles'][2]:.6f}. This is measured optimization behavior, not causal proof of the burial failure. Adam's adaptive moments prevent interpreting it as a simple fixed effective learning-rate multiplier.", '',f"Reference trace-KL beta: {opt['beta']}; trace-KL quantiles: {opt['trace_KL_quantiles']}. Latent Gaussian entropy quantiles: {opt['entropy_quantiles']}. Negative differential entropy is allowed; this is not filtered structure entropy. No entropy bonus or KL coefficient is changed.", '', 'Component score-gradient norms, stage head-gradient norms and pairwise gradient cosines are measured on the original first on-policy TRAIN group in COMPONENT_GRADIENT_EVIDENCE.json. That one-group measurement can reveal competing directions, but does not establish the dominant cause across 5000 updates. No old off-policy protein trajectory is used for new optimization.']
    doc('REWARD_AND_ADVANTAGE_SIGNAL_AUDIT.md','\n'.join(lines))
    doc('LONG_HORIZON_CREDIT_ASSIGNMENT.md',f'''# Long-horizon credit assignment

There are 63 stochastic residue nodes and 186 active geometric components before additional rejected proposals; the first residue is gauge-fixed. Proposal trace scores are summed, then averaged over 16 trajectories. Finite rejection adds score terms, including the final rejected proposals on exhausted trajectories.

The original terminal reward gives all nodes the same trajectory-level advantage. Regional burial depends on nonlocal future contacts: current exposure is an upper bound that may decrease as later residues arrive. Early geometric decisions can therefore influence much later burial, while also competing with exposure and global packing. This demonstrates delayed credit structure, not a diagnosed estimator bias.

Current-R3 matched continuation evidence: {len(early['branches'])} branch/status records at generated-prefix lengths 8 and 32, two locally valid action alternatives where available, eight shared suffix seeds. EARLY_ACTION_EVIDENCE.json and immutable suffix traces report final burial, hard validity and full success. No native prefix was used. Missing/unsuccessful branches do not prove mathematical infeasibility; local validity is not a final feasibility certificate.

The proposed experiment changes goal representation only. It deliberately does not simultaneously change the baseline, reward, clipping, learning rate, KL, task mix or filters. A prefix-conditioned control variate is a separate future hypothesis if representation-focused training fails; it is not combined into this arm.
''')
    rows=['# Same-motif goal swaps and controllability','','Two fitting-TRAIN identities were selected deterministically before model outcomes, requiring native witnesses for base, alternative burial and alternative exposure. Sequence, motif string, numerical thresholds, contrast and validity requirements are identical; only one eight-residue regional instruction changes. Alternative exposure is admitted only where the same motif occurs and the native witness passes the unchanged scoring rule. Native coordinates are used for admission only.','','| Update | Group | Mode | Target | S | Valid pairs /8 |','|---|---|---|---|---:|---:|']
    for v in swap['records']:rows.append(f"| {v['update']} | {v['group']} | {v['mode']} | {v['target']} | {v['S']:.8f} | {v['valid_pairs']} |")
    rows+=['','These are bounded TRAIN diagnostics, not a powered scientific improvement test. No bootstrap significance claim is made from two groups. Historical V9.5 exposure responses changed both exposed region and motif and therefore cannot alone establish motif-independent exposure control. The new frozen VALIDATION diagnostics preserve motif and thresholds for both interventions.','', 'EARLY_ACTION_EVIDENCE.json provides matched suffix outcomes and branch validity. Feasible native witnesses demonstrate existence within the geometric criteria, not that the prior gives substantial probability to every alternative or that any output is experimentally foldable.']
    doc('GOAL_SWAP_CONTROLLABILITY.md','\n'.join(rows))
    doc('PROPOSED_RL_ALGORITHM_CHANGE.md',f'''# Proposed single intervention: explicit burial-region token

The independent actor receives one additional deterministic goal token through its existing 36-to-448 goal projection and seven-layer Transformer. Existing motif tokens and global context are unchanged. No parameters are added. New token fields: role=-1, interval width/N, start/N, end/N, burial maximum, contrast minimum; positional anchor is interval midpoint/N in field32; validity bits33 and35 are one. AA motif fields12:32 are zero. Field0 separates its role from nonnegative motif indices. The finite interval/threshold tuple is distinguishable in the float32 declared domain.

This removes an architectural asymmetry by giving the requested burial region a direct positional attention path, analogous to the existing exposed-region motif anchors. Evidence supports a usable but weak/asymmetric original path; the token's benefit remains a hypothesis for the production experiment.

The conditional policy changes from pi_theta(u|s,g) to pi_theta(u|s,augment(g)). The deterministic observation map is independent of theta. Original raw Gaussian proposal densities, masks and deterministic physical-action transform remain exact. For full trace tau, J(theta)=E[R(tau)-beta log(P_theta(tau)/Q(tau))]; gradient is E[(R-beta log(P/Q)-beta) grad log P]. Independent RLOO subtraction remains an action-independent baseline. Every accepted and rejected proposal is included. No new Jacobian, importance ratio, hindsight relabeling or successful-trajectory weighting is introduced. Reference Q remains the permanently frozen J1 geometry policy under the same deterministic controller. No potential shaping or objective-preservation assumptions are needed because reward and objective are unchanged.

The same actor forward pre-hook is installed for sampling and replay and required by checkpoint source binding. It is never installed on J1 or the immutable parent. Token-disabled execution is the original policy, not a newly trained control. Implementation: source96/algorithm.py.

Initializer: exact update5000 actor tensors, fresh AdamW and original schedule. This is a continuation-improvement question. Comparisons include original-encoding update5000, augmented-token update0 with identical weights, and augmented final. The augmented update0 control separates immediate input-map effects from learned change. A single continuation arm does not causally isolate the token from the benefit of extra RL; a matched old-method continuation would be needed for that stronger claim and is not authorized here.
''')
    doc('GRADIENT_AND_LIKELIHOOD_TESTS.md',f'''# Gradient, likelihood and recovery qualification

Synthetic exact enumeration verifies finite rejected-proposal trace gradients, KL derivative and independent RLOO. An accepted-only shortcut fails that test as expected. Gaussian score finite differences and inactive masks pass. Token tests preserve motif/context, encode distinct burial requests and verify the positional anchor without adding parameters. Missed-frontier callback ordering is tested before later journal steps.

Actual full-size TRAIN qualification: {tests['unique_engineering_updates']} unique updates, {tests['physical_optimizer_calls']} optimizer calls including one exact committed-journal replay; {tests['new_training_attempts']} new training attempts. Sampling and gradient replay use the same augmented actor path. Replay maximum errors: {tests['first_logprob_maxerror']}, {tests['second_logprob_maxerror']}; rejected proposals included: {tests['rejected_proposals_replayed']}. Maximum step-KL probes: {tests['first_step_KL']}, {tests['second_step_KL']}. AA score gradients are zero, J1 and parent unchanged, actor parameter count {tests['parameter_count']} (added parameters0).

Full actor/Adam moments/scheduler/CPU-CUDA-NumPy-Python RNG restore and exact post-commit replay pass. An intentionally interrupted actual full-state CPU checkpoint writer exits75 after8MiB; the old committed recovery index remains unchanged and restores exactly. Partial evidence is preserved and never adopted as a complete checkpoint. Incorrect token/source binding is rejected. Future production callbacks reconcile at restored frontiers and between journal updates.

Reserved TRAIN baseline:256 attempts, separate from fitting gradients; original absolute initial safety admission passes: {tests['initial_safety_pass']}. Count correction and raw legality-flag labeling are documented in qualification/METADATA_COUNT_FIX.json; all trajectories, optimizer steps and model tensors were preserved. The token-input Jacobian probe used empty-prefix head outputs; the gauge-fixed first-step score is masked. Actual full-trajectory replay additionally backpropagates the stochastic nodes. Full evidence: qualification/SYNTHETIC_TESTS.json, PROTEIN_TESTS.json and AUGMENTED_INITIAL_TRAIN_BASELINE.json. Engineering correctness does not establish scientific performance improvement. Production updates:0.
''')
    doc('SUBSTANTIAL_TRAINING_PROTOCOL.md',f'''# Substantial training protocol — prepared, NOT launched

One independent fixed-sequence geometry actor, exact R3 update5000 initializer plus explicit burial token; fresh AdamW. Permanently frozen J1 and immutable original parent. 5000 additional updates x16 on-policy attempts =80000 fitting attempts. Same434 TRAIN identities,868 witness/relaxed tasks and identity-cycle task selection; seed19629, fixed before diagnostic outcomes. No validation examples/metrics enter optimization. Historical VALIDATION findings informed this question, not per-example training targets.

Unchanged peak lr1e-6, warmup100, flat through4000, cosine decay to0.1peak at5000; AdamW betas.9/.999, eps1e-8, weight_decay0; norm clip5; beta {config['trace_KL_beta']}; same terminal rewards and independent RLOO; full raw-proposal trace likelihoods; same finite8 filter, packing rejection and one-step shield. No new filter, AA learning, SAC, critic or high-level policy.

Baseline and monitoring use only the original reserved16 TRAIN identities/16 goals,8 candidates, raw and filtered:256 attempts per checkpoint at0,250,500,1000,2000,3500,5000. This is1792 monitoring attempts, separate from80000 fitting attempts. Baseline and monitor results never select validation identities. Named and dual rolling checkpoints retain original cadence. Exact state/optimizer/RNG and per-update immutable trace+commit recovery are retained. source96/campaign.py adds explicit missed-callback reconciliation; no automatic restart.

Safety: original initial hard validity>=.91875, backbone>=.95, diversity>=12.44911875, proposalKL<=1. During monitoring retain <=5pp hard/backbone deterioration with original absolute floors, >=80% diversity versus stronger original/new baseline, proposalKL<=1, per-stepKL<=.02, finite gradients and storage reserve2GiB. Abort failures remain in denominators. Numerical, binding, journal or storage failures stop closed. One heavy CUDA workload only; durable stdout/stderr, immutable commits and verified terminal exit required.

All artifacts remain under this new D root. Conservative storage: original <=10 distinct full checkpoint images (~2.1GiB), <=2GiB proposal traces plus existing2GiB reserve and other-report margin; require at least8GiB free before launch and maintain measured sidecar projection. Engineering partial files are evidence and are not reused in production. Estimated GPU work is comparable to the prior5000-update campaign; measured prior timing and future runtime replace forecasts. Do not promise benefit or power from scale alone.

Correctness release PASS: {release['correctness_qualified']}. Initial safety PASS: {release['initial_safety_pass']}. Ready for review: {ready}. NO production authorization receipt exists. The only production entrypoint requires --authorization with explicit_review_authorized=true plus exact configuration and all source96 hashes. It cannot be started by running preparation, diagnostics or publication. Do not create that receipt automatically. Future command after review: ./run.sh source96/campaign.py --authorization <reviewed-receipt.json>.
''')
    doc('VALIDATION_PROTOCOL.md',f'''# Frozen development VALIDATION protocol

{panel['identities']} identities/{panel['independent_groups']} independent protected groups, one selected structural observation per group;1024 primary witness/relaxed goals. Deterministic outcome-independent hash selection excludes all768 V9.5 evaluated identities and known prior E017 prospective/reference-calibration identities. Prior E016 model-selection exposure remains allowed. This is development VALIDATION, not untouched confirmation. Training groups are disjoint. TEST is never read.

Freeze VALIDATION_PANEL_MANIFEST.json before training. {panel['diagnostic_groups']} groups support both same-motif, same-threshold regional alternatives: {panel['diagnostic_goals']} separately generated diagnostic goals (base plus two alternatives per group). Native witnesses establish original computational success; coordinates are never policy inputs. Primary witness/relaxed targets remain unchanged. Separate diagnostic triples use shared native-supported thresholds on the original0.05 grid and original witness/relaxed slack, recorded explicitly; these may be looser than primary targets and cannot substitute for primary performance. The diagnostic base may use a common motif different from the original primary motif; that base is generated separately, and the identical motif is held fixed across all three diagnostic goals. All changes are explicit in the manifest. Eight candidate seeds per primary goal; base and alternate diagnostic goals use their declared common diagnostic stream, separate from primary streams. Diagnostic outcomes never enter primary denominators. No adaptive enlargement, threshold change, seed replacement, or best-of8 masquerading as K1.

Evaluate original J1 plus original-encoding R3 update5000, augmented V96 update0, and V96 final5000, raw and filtered for each actor. Total fixed budget: 7*1024*8 + 6*{panel['diagnostic_goals']}*8 attempts. Final5000 is primary; a TRAIN-selected checkpoint may be descriptive and must not replace it. No VALIDATION inference has been run in this preparation.

Primary comparison: final V96 filtered K1 minus original-encoding R3 update5000 filtered K1 on this exact new panel. Promotion retains >=5pp gain and strictly positive two-sided95% paired-group bootstrap lower bound, K8 nondecrease and original validity/diversity/KL safeguards. Also report final versus augmented update0 to distinguish learning from immediate token effects.

Average the two primary goals within group, bootstrap512 protected groups with10000 draws, seed19329. Six secondary families retain Bonferroni .05/6 (99.1667% intervals): raw/filtered K8 and raw/filtered exposure/burial directional responsiveness. Diagnostic estimates resample the protected diagnostic groups. Invalid pairs contribute0; valid-only results are descriptive. Burial learning requires positive final response and final-minus-augmented-initial bounds, with final point response>=.01. Report comparisons against original encoding separately. Exposure preservation requires a supported positive raw response>=.01 and no point decrease versus original update5000 response; report paired uncertainty, do not label this point rule a formal noninferiority test. No post-result gate substitution.

Report raw/filtered K1/K8, full success, backbone/hard validity, finite-proposal exhaustion, packing/clashes, target error including abort penalty, complete-only diversity, Gaussian proposal KL (not effective filtered-policy KL), costs and identity-level paired records. Report the primary interval even on NO-GO.512 groups do not guarantee80% power for a5pp effect or the intersection of all promotion gates; no uncomputed power claim is made. Historical32-goal and V9.5 outcomes are not pooled.
''')
    doc('FINAL_E017_V96_SCIENTIFIC_DIAGNOSIS.md',f'''# Final E017 V9.6 scientific diagnosis and prepared experiment

Production training has NOT started. Historical results/checkpoints are unchanged and verified. Engineering-only updates:2 unique plus one exact recovery replay. Correctness qualified:{tests['PASS']}; initial safety:{tests['initial_safety_pass']}; ready for substantial training review:{ready}.

Demonstrated: burial instructions survive serialization, public prefix context and the Transformer query; same-motif regional swaps give distinguishable observations and usable gradients. Exposure/motif tokens are position-anchored while burial has no corresponding token. Original clipping frequency is99.8% (4990/5000). The original RLOO terminal-return estimator assigns a common coefficient to early and late full-trace scores. Reward components, per-group signal and bounded current-controller continuation evidence are published separately.

Not demonstrated: a unique causal root cause for failed burial steering. The evidence supports a combination of representational asymmetry and delayed multiobjective credit/optimization limitations, not a missing burial input, proven estimator bias, mathematical infeasibility, or a new geometric correctness defect. The bounded same-motif controls avoid the historical exposure/motif confound, but two TRAIN identities cannot establish population-level exposure or burial learning.

Choose ONE intervention: an explicit burial-region token using the existing Transformer projection, with no added parameters. Reward, advantages, clipping, task distribution, KL, sampler and deterministic filters stay unchanged. Its prospective benefit is a hypothesis. A simultaneous credit-estimator change would confound this test and is deferred.

The prepared arm continues from the exact5000 actor, fresh optimizer, another5000 updates/80000 fitting attempts after separate review. Original-encoding5000 and augmented update0 are distinct frozen comparators. This is a continuation experiment, not an independent fresh-start replication or definitive token-versus-extra-training causal comparison. The new development panel excludes known prior E017 evaluated identities; E016 exposure remains permitted. No VALIDATION forward passes and no TEST reads occurred.

Hierarchy: reliable burial/exposure primitives could later support choosing regional subgoals, selecting generated-prefix correction points, suffix regeneration and learned future-feasibility estimates. Current evidence does not qualify a general high-level policy. No high-level RL or AA training is started. C-alpha success is not experimental folding stability.

TEST_SEAL=UNKNOWN historically; no observed breach and no TEST content accessed here. Source and checkpoint bindings, actual tests, limited diagnostic denominators and explicit remaining causal uncertainty accompany this report. Production remains gated on a future review receipt.
''')
    write('FINAL_STATUS.json',dict(BURIAL_CONTROL_ROOT_CAUSE='CAUSAL_ROOT_UNRESOLVED; REPRESENTATIONAL_ASYMMETRY_AND_DELAYED_CREDIT_SUPPORTED_AS_MECHANISMS',GOAL_ENCODING_STATUS='OBSERVABLE_WITH_USABLE_GRADIENTS; BURIAL_LACKS_REGIONAL_TOKEN',CREDIT_ASSIGNMENT_STATUS='CORRECT_FULL_TRACE_RLOO; DELAYED_TERMINAL_CREDIT_AND_FREQUENT_CLIPPING; VARIANCE_LIMITATION_UNPROVEN',PROPOSED_METHOD='EXPLICIT_BURIAL_REGION_TOKEN_WITH_EXISTING_TRANSFORMER',IMPLEMENTATION_QUALIFIED=tests['PASS'],READY_FOR_SUBSTANTIAL_RL_TRAINING='YES' if ready else 'NO',PRODUCTION_AUTHORIZED=False,TEST_SEAL='UNKNOWN',TRAINING_UPDATES='ENGINEERING_ONLY',unique_engineering_updates=2,physical_optimizer_calls=3,production_updates=0,VALIDATION_forward_passes=0))
    files={str(p.relative_to(ROOT)):dict(SHA256=sha(p),bytes=p.stat().st_size) for p in sorted(ROOT.rglob('*')) if p.is_file() and 'runtime' not in p.parts and p.name!='PROVENANCE_SHA256.json'}
    write('PROVENANCE_SHA256.json',dict(files=files,self_excluded=True,runtime_excluded=True,historical_bindings_SHA256=sha(ROOT/'HISTORICAL_BINDINGS.json'),TRAINING_UPDATES='ENGINEERING_ONLY',production_started=False))
    assert all(sha(ROOT/name)==v['SHA256'] for name,v in files.items())
    print('V96_REVIEW_PACKAGE_PUBLISHED',len(files),'files; READY',ready,'PRODUCTION_UPDATES0',flush=True)


if __name__=='__main__':main()
