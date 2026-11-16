from common_review import *
import numpy as np,shutil,subprocess,collections,math
from scipy.stats import binom,beta
x=read(ROOT/'EXISTING_RECORD_ANALYSIS.json');probe=read(ROOT/'TOKEN_RECORDED_PREFIX_PROBES.json');grad=read(ROOT/'TRAIN_GRADIENT_COMPONENT_PROBES.json');change=read(ROOT/'CHECKPOINT_POLICY_CHANGE.json');assert read(ROOT/'ENGINEERING_PROBE_COMPLETED.json')['PASS']
old=read(PAIR/'PAIRED_SCIENTIFIC_RESULTS.json');records=probe['records']
def q(v):return [float(z) for z in np.quantile(v,[0,.5,1])]
def doc(name,s):write(name,s+'\n')
def j(v):return json.dumps(v,indent=2)
aggregate={k:dict(active_KL_min_median_max=q([r[k]['selected_active_KL'] for r in records]),AA_logit_max_delta=max(r[k]['AA_logits_max_delta'] for r in records)) for k in ['token_effect','masked_token_effect','visible_zero_token_effect','visible_role_only_effect']}
aggregate['first_layer_query_attention_new_token_min_median_max']=q([v for r in records for v in r['new_token_query_attention_first_layer']]);aggregate['last_layer_query_RMS_min_median_max']=q([r['layers'][-1]['query_RMS'] for r in records]);aggregate['last_layer_prefix_RMS_min_median_max']=q([r['layers'][-1]['prefix_RMS'] for r in records]);write('PROBE_SUMMARY.json',aggregate)
# Prospective calculations, no new safety decisions for old studies.
alpha=.05/(2*2*7);n=4096;k=np.arange(n+1);L=np.r_[0,beta.ppf(alpha/2,k[1:],n-k[1:]+1)];U=np.r_[beta.ppf(1-alpha/2,k[:-1]+1,n-k[:-1]),1];accept=int(k[L>=.95][0]);fail=int(k[U<.95][-1]);operating=[]
for p in [.99,.98,.97,.966,.96,.95,.94,.93]:operating.append(dict(true_validity=p,probability_admit_or_continue=float(binom.sf(accept-1,n,p)),probability_definite_failure=float(binom.cdf(fail,n,p)),probability_pause_inconclusive=float(binom.cdf(accept-1,n,p)-binom.cdf(fail,n,p))))
write('PROSPECTIVE_SAFETY_OPERATING_CHARACTERISTICS.json',dict(n=n,alpha_per_two_sided_interval=alpha,continue_min_successes=accept,fail_max_successes=fail,otherwise_pause=True,operating=operating,assumptions='independently randomly draw identity and goal from prespecified reserved TRAIN finite population with replacement; exactly one fresh candidate per draw; draws independent conditional on frozen checkpoint and past; common population and matched task/seed schedules across arms; not8 correlated candidates counted as independent'))
refs='Sources: [NIST exact binomial intervals](https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm); [Howard et al. confidence sequences](https://arxiv.org/abs/1810.08240); [Adam](https://arxiv.org/abs/1412.6980). These support statistical/optimization principles, not causal claims about ProteinGen.'
doc('BURIAL_TOKEN_INITIALIZATION_AUDIT.md',f'''# Burial token initialization audit

Information availability, initial perturbation and learnability are separate. The original actor already receives burial start/end and thresholds in context. Prior gradient evidence confirms a usable path. The V9.6 token adds an anchored path, but its success at learning useful burial control was never established.

Exact path: algorithm.augment appends a visible36-feature token with role=-1, width, start/N, end/N, burial maximum, contrast, midpoint field32, and flags33/35. It leaves original motif tokens/context unchanged. The existing biased Linear36->448 projects the new token. Prefix+G motif+query becomes prefix+(G+1)+query. Masks remain correct: the new key is true, prefix padding is masked, and query remains the final row. RoPE uses explicit positions, so old prefix/motif/query positions do not shift when the array grows. The new anchor is the burial midpoint. There is no accidental final-token selection or AA-mask defect.

Each of7 pre-normalized bidirectional Transformer blocks normalizes per token, then computes Q/K/V and softmax over all valid keys, followed by output/residual and normalized feedforward/residual paths. The extra key changes the softmax denominator even if its value were zero; zero input also projects to nonzero bias. The LayerNorm does not average across sequence length; its downstream changes arise from attention/residual inputs. Prefix and motif representations change as well as query. Final LayerNorm(query) drives both AA and all20 conditional geometry heads. Fixed sequence suppresses categorical learning, not the internal AA-logit response. J1 has no new token.

Existing VALIDATION initial filteredK8 falls540->458/1024 (-8.0078125pp), rawK8 308->239 (-6.73828125pp); filteredK1 94->79 (-1.46484375pp). These are pre-learning effects. Tiny one-step distribution shifts can compound over64 residues and alter discrete finite rejection branches. This is a mechanism, not a quantified attribution of the total yield loss to a particular head.

New engineering inference used16 previously recorded TRAIN prefixes,2 identities x2 candidates x4 prefix positions8/24/40/56. No candidates or weights were generated/trained. Results:

{j(aggregate)}

Masking the token is a function-preserving inference control in these fixtures, but removes its learning/information path. A visible all-zero or role-only token is not neutral. Parameter-count equality never implied function equality. No production integration was changed.

Proposed strategies: (1) permanently masked token preserves outputs but cannot solve conditioning; (2) zero-input/zero-value token alone does not preserve attention normalization; (3) zero-initialized output residual gate on the difference of augmented and original geometry predictions preserves the policy algebraically and has a usable gate derivative. The third is the sole recommended future treatment, subject to exact likelihood, sampling/replay and recovery qualification. It adds one explicitly reported scalar, uses shared Transformer weights, and is not yet implemented/qualified. No evidence that it improves RL performance is claimed.
''')
lines=['| Arm/update | Valid /128 | iid exact95% CI | paired identity change CI | abort prefixes |','|---|---:|---|---|---|']
for arm,ds in x['development'].items():
 for u,v in ds.items():lines.append(f"| {arm}/{u} | {v['valid']}/128 | {v['iid_CP95']} | {v['initial_transitions']['paired_identity_CI95'] if v['initial_transitions'] else 'baseline'} | {v['abort_prefix_positions']} |")
doc('TRAINING_SAFETY_MONITOR_STATISTICS.md',f'''# Safety monitoring reliability and prospective design

Historical stop decisions remain correct under their frozen rules. A500=120/128<.95; B250=121/128<.95. Their separate hard-validity deterioration, diversity and KL limits passed. Neither can be restarted or reclassified as safe by this review.

{chr(10).join(lines)}

These new retrospective intervals are uncertainty diagnostics, not replacement historical analyses. iid intervals ignore grouping; identity bootstrap uses16 groups and seed310196,10000 draws, resampling whole8-candidate clusters. With16 groups both intervals and ICC estimates are unstable; negative method-of-moments ICC is not evidence of negative physical dependence. Reused identities/seeds make repeated comparisons paired. An exactly unchanged deterministic policy reproduces the same panel, so repeated evaluations are not independent fresh binomials. Changed-policy outputs may diverge through rejection branches. The observed paired declines do not isolate a population causal degradation.

A500 transitions:7 previously valid failures,2 recovered,118 stayed valid,1 stayed invalid. B250:4 new failures,3 recovered,118 stayed valid,3 stayed invalid. A failures span6 identities and B6 identities; no single identity explains either stop. Every monitoring goal is witness-class, while final VALIDATION has two primary variants per identity. It is not a representative estimate of all TRAIN/VALIDATION goal difficulty classes.

At A500 all64 terminal proposals of8 aborts violate the exact final contact budget, some also clash. B250 has53 contact-budget and18 clash reason flags across56 terminal proposals; reasons overlap. Aborted positions are late, but sampled-proposal exhaustion does not prove no feasible action existed. One-step feasibility rejects certified dead successors; UNKNOWN is not a reachability certificate. Prefix-validity filters do not guarantee eventual property satisfaction.

Illustrative iid false-point-stop probability at true validity .97 is9.1519%; at.965 is16.2914%; at.95 is45.9286%. Detection at.94 is65.2873%, at.93 is79.9420%. A six-independent-look extrapolation at.97 gives43.7793%, but is NOT an estimate of this fixed-seed evolving-policy campaign. Full curves are in MONITOR_DESIGN_CALCULATIONS.json. Published larger-panel filtered validity A:7899->7933/8192; B:7916->7899/8192, all above95%. That is evidence against a simple broad collapse claim, not evidence that the historical point-rule stops should be bypassed. TRAIN and VALIDATION distributions differ.

Prospective common-arm measurement protocol: inventory eligible TRAIN homology groups, reserve a representative minimum128-group development pool, exclude it from new fitting, retain original task/threshold distributions, and report any common reallocation explicitly. If availability is insufficient, admission is blocked. Freeze the population, strata, task distribution, seed arrays and independent measurement RNG before training. It is a finite development population, not a claim of new independent identities on every draw or untouched replication.

At each of7 looks (0,250,500,1000,2000,3500,5000) draw4096 identity/goal pairs independently with replacement from that frozen distribution; generate ONE fresh candidate per draw. Independent random identity choice and candidate seed yield independent Bernoulli observations conditional on the checkpoint and past, despite repeated identities. This independence does not apply to the old fixed8-per-identity panel, nor does it permit pretending4096 unique homology groups exist. Pair task/seed arrays across arms for comparability; use fresh arrays across looks. No dependence on monitor outcomes may modify goals or hyperparameters.

Use two-sided Clopper-Pearson intervals at alpha=.05/(2arms*2validity metrics*7looks)={alpha}, a union-bound family guarantee across these predeclared intervals. Continue/admit only if backbone lower bound>=.95 AND all original hard/backbone preservation, KL, diversity, numerical and resource gates pass. Definite failure if backbone upper<.95; if the interval straddles.95, PAUSE with no more optimizer updates. Do not lower the floor or automatically resample until it passes. No extra looks without prespecified alpha spending. Original hard-preservation point gate and original absolute hard floor remain binding, with the same population and matched baseline. Keep legacy128 panel only as descriptive sentinel, not an undeclared second safety test.

At N4096 the backbone decision is continue if successes>={accept}, fail if successes<={fail}, otherwise pause. Full prospective operating characteristics are in PROSPECTIVE_SAFETY_OPERATING_CHARACTERISTICS.json. Planned safety cost57344 attempts across2arms*7looks, plus unchanged property/diversity diagnostics and final scientific evaluation. This cost must be admitted rather than hidden. A conservative whole-identity Hoeffding alternative is documented, but is less efficient; the selected future protocol requires validated independent draws. More general time-uniform procedures are possible but are not implemented here.

{refs}
''')
gsummary={arm:{k:v for k,v in a.items() if k not in ['groups','abort_positions']} for arm,a in x['training'].items()}
doc('GRADIENT_CLIPPING_AND_EFFECTIVE_UPDATE.md',f'''# Gradient clipping and effective parameter changes

The source sums all stochastic latent geometry log-probabilities over every sampled accepted and rejected proposal, then averages16 trajectories. AA categorical score terms are absent. There is no hidden average over trajectory length, same-group whitening, advantage clipping, or extra optimizer step. A common terminal augmented-return RLOO coefficient acts on all positions. That is the registered score estimator, not by itself a bug; dividing each trajectory by its varying length would change its weighting/objective.

Reward R, trace ratio L and beta yield coefficient -RLOO(R)+beta*RLOO(L)+beta for the negative objective. AdamW weight_decay0, beta1=.9 beta2=.999 eps1e-8, original100-step warmup and1e-6 peak are unchanged. The journal's lr is the NEXT scheduler LR; first actual applied LR is1e-8, not the journal's2e-8. Clipping computes a single global preclip norm and scales by min(1,5/(norm+1e-6)), preserving the gradient direction. High frequency is real, not proof of a wrong implementation.

Recorded summaries (all committed updates):

{j(gsummary)}

New exact first-group backward decompositions, with ZERO optimizer construction/calls:

{j(grad)}

All components use actual recorded on-policy TRAIN traces and include rejected draws. Only2 groups were decomposed, one per encoding, so component cosines cannot identify the campaign-wide dominant cause. Cosines involving zero or numerical-zero components (success in both first groups, motif in B) are undefined for interpretation; displayed clamped numerical values must be ignored. Stage norms are contributions to the same total gradient, not a causal credit estimate. Zero first-group success gradient reflects no successful candidates in that group, not permanently missing success reward.

Applied-parameter calculations use existing checkpoint tensors and stored Adam moments only. Parameters with step0 (notably fixed-AA heads) have zero update; division by zero is not a training fault. We reconstructed the last applied Adam displacement from bias-corrected moments and compared existing final weights with certified initial weights. No displacement was applied. See CHECKPOINT_POLICY_CHANGE.json for per-prefix geometry changes and cumulative L2 changes.

{j({arm:{k:v for k,v in s.items() if k!='common_TRAIN_prefix_changes_from_own_initialization'} for arm,s in change.items()})}

The analytic first fresh-Adam direction is -lr*g_clip/(abs(g_clip)+eps); its cosine with negative raw gradient differs from1 because of coordinatewise adaptation. Directional derivatives on an existing TRAIN prefix predict nonzero geometry response without stepping the model. The final-vs-initial TRAIN distributions and journal step KL show nonzero policy movement, but do not establish useful steering. Single-step KL probes cover selected recorded prefixes, not a global trust region or a guarantee of stable end-to-end validity. A Gaussian proposal KL is not the KL of the accepted filtered controller.

Adam often cancels a common gradient scale in moment ratios; varying clipping scales alter momentum history and relative weighting across updates. The median clipping factor is NOT an effective-LR multiplier. Negative latent differential entropy is valid and not a measurement of structural diversity. No evidence justifies increasing LR, removing clipping, or changing beta during this audit.

{refs}
''')
doc('BURIAL_CREDIT_ASSIGNMENT_DIAGNOSIS.md',f'''# Burial learning: demonstrated mechanisms and unresolved causes

Verified: original burial coordinates/thresholds are observable in context and carry gradients; V9.6 adds an anchored but initially disruptive attention path. New token effects are not isolated to burial. Burial reward is nonzero and informative in many groups: A{ x['training']['A']['informative_burial_groups'] }/500 and B{ x['training']['B']['informative_burial_groups'] }/250 have both satisfied/unsatisfied candidates. Total reward–burial satisfaction correlations are A{x['training']['A']['reward_burial_corr']:.6f}, B{x['training']['B']['reward_burial_corr']:.6f}. Therefore complete reward blindness is contradicted. Reward component correlations are descriptive and not gradient signal-to-noise ratios.

Full reward = hard*[.5 success +.1 each exposure,burial,contrast,motif,compatibility utility]. Abort reward0 and the fixed-sequence motif term largely distinguish completion/validity. Exponential residual utilities can be extremely small when targets are missed widely. Exposure and burial satisfaction correlate mildly negatively in pooled data (A{x['training']['A']['exposure_burial_corr']:.6f},B{x['training']['B']['exposure_burial_corr']:.6f}), but pooled correlations do not prove contradictory goals. Native-compatible task witnesses establish some feasible configurations, not accessibility under this frozen J1 proposal support.

There are roughly186 active latent dimensions for a completed64-residue trajectory, plus rejected-proposal terms. The SAME terminal coefficient multiplies early and late scores. Final regional burial depends on contacts with future residues, while early geometric decisions can constrain later packing. This establishes a delayed-credit mechanism and high-variance opportunity, not proof it dominates representation/exploration. Exact stage/component probes provide local directions; two groups cannot measure full estimator variance or causal credit.

A had500 updates/8000 attempts, B250/4000. Their first250 matched schedules are witness goals; A's extra66 updates after the first434-identity pass introduce relaxed variants. TRAIN reward trend comparisons across this switch are confounded by task difficulty. Historical training end points are unequal; inadequate full-budget exposure remains an unresolved limitation rather than proof more RL would work. Existing final validation raw burial B response is negative both before/after, and difference-in-changes includes0. Filtered yield is not proof of learned conditioning.

Unresolved: feasible controllable burial support under J1; the relative importance of long-horizon score variance, reward saturation, entropy/exploration, optimizer history and representation. No new candidate search or native-future-coordinate input was used. No new geometric filter is justified: existing contact and one-step feasibility checks acted as designed. C-alpha geometric acceptance does not establish experimental folding stability.

The narrow actionable diagnosis is an avoidable initial representation perturbation plus insufficiently precise historical safety measurement. No unique burial-learning root cause has been proven. Correct initialization before altering reward/optimizer/credit methods, and evaluate substantial equal budgets only after prospective safety admission.
''')
doc('PROPOSED_NEXT_RL_INTERVENTION.md',r'''# One primary intervention: function-preserving residual burial conditioning

PROPOSAL ONLY. No production implementation, new checkpoint or optimizer was created. Exact engineering admission is outstanding.

Let F0(theta,s,g) be the original encoding actor's geometry Gaussian parameters (mu0,ls0), and F1(theta,s,g) the same shared-weight actor with the qualified burial token. Introduce ONE scalar a initialized exactly0 and gamma=tanh(a). Define mu=mu0+gamma*(mu1-mu0), ls=clip(ls0+gamma*(ls1-ls0),-5,2). Keep the original AA output and fixed AA decisions. No separate network, reward, optimizer formula, RLOO estimator, filter, temperature or proposal budget changes. Report the +1 parameter honestly; the V9.6 zero-added-parameter claim does not apply to this new proposal.

At a=0, outputs equal F0 algebraically for every state/goal. Do not implement a Python branch returning F0: that would disconnect the gate derivative. Evaluate both deterministic branches, share the same theta, and combine continuously. dF/dtheta at0 equals dF0/dtheta; dF/da at0 equals F1-F0 before clamp, which can be nonzero. The original geometry branch therefore keeps its gradient while the gate can learn whether the new information/path is useful. Extra-token-only gradient routes initially cancel in dF/dtheta, a cold-start limitation that must be tested. This preserves initial policy, not a promise of effective regional information learning. A single global gate may be too restrictive, but do not add region-dependent gates or capacity without separate evidence.

tanh permits learning a negative coefficient; negative gamma is an explicit permitted control, not silently clipped. Logstd remains bounded; branch means remain bounded by the original residual architecture, so the combination is finite and bounded, although not guaranteed to stay inside the original single-branch residual envelope. Existing action transform and physical filtering remain unchanged. This behavior must pass existing likelihood and safety admission; do not redefine limits to admit it.

A visible zero embedding or zero value is rejected as a function-preservation strategy because it still changes attention normalization. A fully masked token is a useful equality control but removes the information path. Output-difference gating avoids that denominator issue without altering Transformer internals. Compute is approximately2 actor forwards per distribution/replay, with shared parameters and one J1 reference forward; qualify real VRAM/time rather than asserting it is free.

Required qualification: bitwise or declared-tolerance equality of original/combined initial distributions, same-seed finite-filter trajectories and AA masks; nonzero correct gate gradient including clamp boundaries; finite-difference or exact synthetic score-estimator tests; accepted/rejected full-trace likelihood; one-gate parameter counting; zero J1/parent gradients; full optimizer/journal/RNG/atomic-write recovery including gate; config/source bindings; bounded TRAIN-only admission. These checks may include separately authorized engineering learning tests. None is satisfied by algebra alone. No production launch now.

Why this intervention: it addresses a directly measured initial perturbation while preserving the original RL objective. Current evidence does not justify changing clipping, reward, baseline or learning rate at the same time. If a gate cannot carry useful goal-specific gradients during qualification, stop and report that blocker, without switching methods through a sequence of favorable short pilots.
''')
doc('SUBSTANTIAL_TRAINING_DESIGN.md',f'''# Prospective matched substantial-training design — not admitted or launched

Question: does function-preserving burial conditioning improve learned regional control beyond matched original-encoding continuation? Control is exact original update5000 actor; treatment has the same actor tensors plus scalar gate0 as specified in PROPOSED_NEXT_RL_INTERVENTION.md. Both receive fresh original AdamW/scheduler and matched tasks/seeds. Original J1 and certified parent remain frozen. No previous A500/B250 endpoint is a training initializer. Planned5000 updates*16attempts=80000 per arm, sequential GPU execution. One seed pair remains conditional evidence, not algorithmic robustness; no secondary campaign starts automatically.

All scientific training settings remain original: fixed sequences, task thresholds, hard validity, reward, full finite8 trace likelihood, RLOO, beta, clip5, warmup/LR and deterministic filters. The sole treatment difference is the gated burial representation. A new common TRAIN split and safety measurement are measurement/admission design changes shared by BOTH arms, explicitly distinct from the treatment. Reuse of historical VALIDATION is development assessment, not untouched confirmation. No VALIDATION reward, target, checkpoint selection or tuning enters fitting. TEST/AA/high-level RL/AlphaFold remain excluded.

Before any campaign freeze actual reserved TRAIN identities, goal strata, training population, schedules, initializer/J1 hashes, gate architecture, source, optimizer/RNG plans, reporting/recovery rules, and adequate D: storage. Future representative safety panel details and costs are in TRAINING_SAFETY_MONITOR_STATISTICS.md. Do not assume128 new eligible identities are available until inventory/qualification. Both arms must pass baseline interval lower-bound>=.95 and unchanged safeguards before either trains. Inconclusive means no updates, not permission to proceed. Any common TRAIN population reallocation is disclosed; no task weakening or favorable identity selection based on outcomes.

Primary endpoint: raw matched regional burial S on the same type of sequence/motif-controlled diagnostic triples as V9.6, with meaningfulB-A>=.01, positive95% lower bound, difference-in-changes>=.01 with positive lower bound, positive treatment-own change, and meaningful final treatment response>=.01 with positive lower bound. Preserve invalid-pair0 rule. Freeze eligible80-or-more independent diagnostic homology groups BEFORE outcomes, not by selecting supported effects; no new validation manifest was created in this audit. Use frozen10000 group bootstrap seed19329. Retain512-group/1024-primary-goal development assessment if predeclared reuse is accepted; otherwise freeze eligibility and counts prospectively. Do not call any unfrozen panel ready.

Secondary8 endpoints(raw/filtered K1,K8,burialS,exposureS)*4contrasts with original Bonferroni.05/32. K1 practical promotion against original starting policy remains>=5pp with positive95% lower bound and original K8/validity/KL/diversity safeguards. Same evaluation panel and candidate-indexed seeds across initial/control/treatment final policies; paired identity bootstrap. Evaluate actual update0 equivalence BEFORE learning, then own changes and difference-in-changes. If equal initial outputs fail qualification, no production admission. Preserve exposure response, report signed effects and uncertainty; do not infer preservation solely from nonsignificant decline.

Final5000 endpoints are primary; safety-stopped partial checkpoints are reported separately. Equal-budget endpoints require both arms5000/80000. There is no promise to override safety for full exposure, and no adaptive hyperparameter restart. Short qualification does not establish scientific performance. Generation attempts, total proposals, development/safety/evaluation costs, J1/actor forward counts, raw versus filtered validity and diversity all reported independently.

READY_FOR_SUBSTANTIAL_RL_TRAINING=NO. Outstanding: implementation and likelihood/gradient/recovery qualification, actual manifests and inventories, prospective safety qualification, resource forecast, reviewed production admission and explicit launch authorization. No new production training is authorized by this document. A validated low-level goal-responsive primitive could later support subgoal selection, feasibility estimation and suffix correction, but current evidence does not justify joint sequence control or hierarchical RL.
''')
conclusion=dict(TOKEN_INITIALIZATION_ROOT_CAUSE='VISIBLE_TOKEN_PERTURBS_GLOBAL_BIDIRECTIONAL_ATTENTION_AND_RESIDUAL_STREAMS',SAFETY_PANEL_RELIABILITY='LOW_PRECISION_NEAR_95_PERCENT; HISTORICAL_STOPS_REMAIN_VALID',GRADIENT_OPTIMIZATION_STATUS='EXACT_SCORE_REPLAY_AND_DIRECTION_PRESERVING_CLIPPING; NONZERO_ADAM_POLICY_CHANGES; EFFICACY_UNPROVEN',BURIAL_LEARNING_BOTTLENECK='MULTIFACTORIAL_UNRESOLVED; INITIAL_PERTURBATION_AND_DELAYED_CREDIT_MECHANISM_VERIFIED',PROPOSED_INTERVENTION='ZERO_INITIALIZED_SHARED_WEIGHT_GEOMETRY_OUTPUT_DIFFERENCE_GATE',READY_FOR_SUBSTANTIAL_RL_TRAINING='NO',TEST_SEAL='UNKNOWN_HISTORICAL_ACCESS; NO_TEST_CONTENT_ACCESSED',PRODUCTION_TRAINING_UPDATES=0)
doc('FINAL_E017_V96_STABILITY_REVIEW.md',f'''# Final E017 V9.6 stability review

{j(conclusion)}

The original conclusions remain unchanged: A stopped500, B250; no supported burial-specific improvement; no practical K1 promotion; no equal-budget token advantage. The controller completed frozen validation and publication. Stop records are immutable evidence. This review adds retrospective stability analyses and explicitly bounded TRAIN-only inference/backward probes, not replacement scientific tests.

Demonstrated initialization cause: extra visible token changes attention normalization, prefix/motif/query residuals and action distributions even with identical actor tensors. The query index, masks, positional encoding and fixed-AA score semantics are correct. Zero-visible-token ablation still changes policy; masked token reproduces old outputs in the16 measured contexts. A burial-information gradient path existed before token insertion, so missing information is not a sufficient diagnosis.

Demonstrated stop mechanism: finite8 proposal exhaustion, mostly late contact-budget failures, made sampled backbone-validity estimates cross95%. A7new failures/2recoveries; B4new/3recoveries. This is a valid registered stop, not proof of systematic population collapse or mathematical infeasibility. The small16-witness-goal panel has substantial uncertainty and misses goal strata. The final512-identity assessment stays above95% but cannot cancel prior stops.

Demonstrated optimization: original sum-over-trace RLOO and global clipping are correct on reconstructed TRAIN groups; strong clipping scales preserve direction, while Adam changes coordinate weighting. Parameter and action distributions do move; useful burial steering does not follow from that fact. Very frequent clipping, delayed consequences and sparse complete success remain concerns, not identified unique root causes. Two first-group decompositions cannot establish campaign-wide gradient SNR or optimal step size.

Recommend one new treatment: mathematically function-preserving shared-weight output-difference gate, initially0, with original RL unchanged. It is a proposal, not a shipped or trained fix. Use a prospectively qualified representative safety design for both future arms, retaining95% and pausing unresolved evidence. No automatic campaign. All8 requested reports plus machine-readable measurements are in this review directory. Details and limitations are explicit in the dedicated reports.

{refs}
''')
# Verify historical frozen inputs, journal evidence and published package were not modified.
files=read(PAIR/'PROVENANCE_SHA256.json')['files'];bad=[]
for path,v in files.items():
 if sha(path)!=v['SHA256']:bad.append(path)
assert not bad,bad
frozen=read(PAIR/'FROZEN_SOURCE_AND_INPUTS.json')
for group in ['files','historical_models']:
 for path,h in frozen[group].items():assert sha(path)==h,path
inputs={str(PAIR/'PROVENANCE_SHA256.json'):sha(PAIR/'PROVENANCE_SHA256.json'),str(PAIR/'FROZEN_SOURCE_AND_INPUTS.json'):sha(PAIR/'FROZEN_SOURCE_AND_INPUTS.json')}
for arm,root in ARMS.items():
 for rel in ['production/STOP_RECORD.json','production/RECOVERY_INDEX.json','TRAINING_OUTCOME.json','TRAINING_CONFIGURATION_AND_SEEDS.json']:inputs[str(root/rel)]=sha(root/rel)
output={str(p.relative_to(ROOT)):dict(SHA256=sha(p),bytes=p.stat().st_size) for p in ROOT.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(ROOT).parts}
write('RESOURCE_AND_PROVENANCE_REPORT.json',dict(UTC=utc(),historical_publication_files_reverified=len(files),historical_mismatches=bad,frozen_bindings_reverified=True,input_manifests=inputs,review_files=output,self_excluded=True,resources=dict(D_free_bytes=shutil.disk_usage(ROOT).free,GPU_processes_after_probes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout,probe=read(ROOT/'ENGINEERING_PROBE_COMPLETED.json')),scope=dict(new_candidate_generation_attempts=0,new_optimizer_calls=0,production_updates=0,new_neural_parameters_persisted=0,historical_checkpoint_writes=0,TEST_content_accessed=False),conclusions=conclusion,engineering_execution_notes=['ENGINEERING_EXECUTION_NOTE.json','ENGINEERING_EXECUTION_NOTE_2.json']))
print('STABILITY_REVIEW_PUBLISHED',j(conclusion),flush=True)
