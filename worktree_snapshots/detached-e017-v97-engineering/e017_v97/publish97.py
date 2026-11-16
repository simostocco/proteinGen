from core97 import *
from safety97 import *
import shutil,subprocess,numpy as np
verify();cpu=read(ROOT/'CPU_AND_SAFETY_QUALIFICATION.json');eq=read(ROOT/'INITIAL_EQUIVALENCE.json');gr=read(ROOT/'GATE_GRADIENT_TESTS.json');cost=read(ROOT/'COMPUTE_COSTS.json');ra=read(ROOT/'RECOVERY_A.json');rb=read(ROOT/'RECOVERY_B.json');assert all(x['PASS'] for x in [cpu,eq,gr,ra,rb])
def doc(n,s):save(n,s+'\n')
def j(x):return json.dumps(x,indent=2)
# Actual trace sizes + conservative growth; do not reduce budgets.
trace_sizes=[p.stat().st_size for arm in ['A','B'] for p in (ROOT/f'qualification/recovery_{arm}/rollouts').glob('*.json')];per=max(trace_sizes);checkpoint=max(ra['checkpoint_bytes'],rb['checkpoint_bytes']);free=shutil.disk_usage(ROOT).free
forecast=20*checkpoint+2*5000*per*1.5+57344*per/16*1.5+89088*per/16*1.5+2*2**30
storage=free>forecast+2*checkpoint+2*2**30
monitor_seconds=cost['estimated4096_monitor_seconds'];estimated=dict(A_training_seconds=ra['first_update_seconds']*5000,B_training_seconds=rb['first_update_seconds']*5000,method='one full engineering group per arm; coarse linear extrapolation, not a runtime guarantee',safety_seconds_per_B_look=monitor_seconds,B_safety_seven_looks_seconds=7*monitor_seconds,full_pair_safety_attempts=57344,full_pair_fit_attempts=160000,final_validation_attempts=89088,checkpoint_bytes=checkpoint,available_D_bytes=free,forecast_required_bytes=forecast,storage_admission=storage)
save('RESOURCE_AND_RUNTIME_ESTIMATE.json',dict(UTC=now(),actual=cost,engineering_A=ra,engineering_B=rb,estimate=estimated,qualification_new_generation_attempts=176,unique_engineering_optimizer_updates=2,physical_engineering_optimizer_calls=4,production_optimizer_updates=0,estimated_validation_seconds='historical V96 full89088 took60690s; gated forward overhead adds cost, measure before production',single_GPU=True))
doc('FUNCTION_PRESERVING_GATE_IMPLEMENTATION.md',f'''# Function-preserving gate implementation

Executable source: {WT/'e017_v97/gate97.py'}. One scalar Parameter burial_gate a=0. gamma=tanh(a/0.00450095) in(-1,1). The constant is fixed prospectively as the sum of the original5000 scheduled learning rates; it defines the new scalar's units, without a separate optimizer or changed LR. Nominal unit-normalized Adam steps can then reach a useful gate range within the planned budget, though actual motion is not guaranteed. The original bound actor forward is called on the original batch and again on the qualified V9.6 augmented batch, sharing every Transformer/projection/head weight. Return original AA logits, mu0+gamma*(mu1-mu0), clamp(ls0+gamma*(ls1-ls0),-5,2). This is one Gaussian parameterization, NOT a mixture density or interpolation of samples. No extra Transformer/network/critic is created. Counts:17438114 original,17438115 gated. Parent and J1 are unchanged.

At a=0 finite outputs equal original by arithmetic identity; gradients are not short-circuited. doutput/da=(alternative-original)/0.00450095 at0 before clamp; doutput/dtheta=doriginal/dtheta at0. Nonzero gamma permits shared gradients through the augmented path. A signed gate avoids a zero-boundary dead derivative. Negative gamma permits bounded extrapolation of residual means; it need not stay in the old one-branch residual envelope. Logstd support remains[-5,2], existing latent-to-physical transform/support/filter remains unchanged. Safety admission remains mandatory. Learned a is serialized in actor state_dict with independent optimizer moments. Install before strict loading a gated checkpoint; do not install V9.6 direct-token hook as well.

An extra original forward is intentionally retained. Prefix/goal/query representations differ across branches after the first attention layer, so reusing their hidden states would change the specified function. One J1 reference forward can be shared because the batch reference is unchanged. No unsafe computation-sharing optimization was introduced.
''')
doc('INITIAL_POLICY_EQUIVALENCE_TESTS.md',f'''# Initial policy equivalence

PASS. {len(eq['fixtures'])} forward/likelihood/transformed-action fixtures across4 TRAIN goals, lengths32/64/96 and early/middle prefixes. N32/N96 are explicitly synthetic robustness fixtures derived solely from recorded TRAIN prefixes; the scientific task and complete generation tests remain N64. Original and gated outputs are torch.equal at gate0 for AA logits and all20 conditional geometry means/logstds. Inactive early-residue masks and transformed samples agree. No tolerance relaxation was required. Finite deterministic fixtures plus the algebraic identity support the initialization contract, not equivalence after learning.

Eight complete raw/filtered goal-mode tests each compare8 same-seed candidates under original versus gated execution:64 paired trajectories,128 total new attempts. Sequence, every latent/physical proposal, acceptance/rejection reason, packing-prefix evidence, abort, log-probability, reward, validity and KL fields agree exactly. Elapsed-time fields are excluded. The same frozen controller and reward implementations are called; no native future coordinates enter public batches. Full records are in INITIAL_EQUIVALENCE.json.

The16 randomized batch1 monitor candidates are a bounded collector/recovery exercise only. A full4096-candidate safety-admission look has NOT been performed. Initial policy equality is not evidence that the new monitoring population passes admission.
''')
doc('BURIAL_GRADIENT_FLOW_TESTS.md',f'''# Gate and shared-branch gradients

{j(gr)}

A nonzero gate gradient is measured from the actual burial-only terminal utility RLOO coefficient on an existing16-trajectory TRAIN on-policy trace. This is a decomposition of the unchanged reward, not a new reward or optimizer objective. Full accepted/rejected proposal scores are used. At gate0 score replay matches the original trace within the frozen5e-5 tolerance. Direct mean-parameter gate finite differences at raw0,+.2*scale,-.2*scale use fixed measured branch outputs with FP64 gate arithmetic (epsilon=.0001*scale; tolerance max(1e-5,abs(gradient)*1e-4)). A prior FP32 summed-output finite difference failed because of cancellation and its diagnostic is retained. Logstd clamp boundary conventions are tested explicitly on CPU; the map has framework-chosen subgradients at kinks, not a unique symmetric derivative and shared goal-projection gradients are finite/nonzero. At active gate the augmented burial-token input has a nonzero gradient. These facts qualify a learning pathway, not successful protein steering.

Engineering recovery performs one actual update per arm plus one exact replay per arm:2 unique updates,4 physical optimizer calls,32 new fitting attempts. This includes the scalar in original AdamW/RLOO/clipping semantics. None is a production or scientific learning campaign.
''')
doc('FILTERED_POLICY_LIKELIHOOD_TESTS.md',f'''# Likelihood and filtering qualification

The gated actor outputs one set of latent Gaussian parameters. Original normal log density, active-coordinate masks, physical transform/Jacobian, full finite8 accepted/rejected trace and deterministic abort semantics remain unchanged. There is no branch-probability term or log-sum-exp mixture likelihood and no duplicate score term for the two deterministic forwards. The policy gradient differentiates the composed distribution parameters once.

CPU enumeration confirms exact finite-rejection gate gradient and RLOO against the enumerated objective, including reference KL-integrand derivative. Error:{cpu['RLOO_error']}. Analytic transform log-Jacobian agrees with autograd; inactive coordinate Normal scores are masked. Actual GPU full-trace replay error A:{ra['likelihood_replay_maxerror']}, B:{rb['likelihood_replay_maxerror']}. Original AA-head categorical terms remain0 because sequence is externally fixed. At gate0 all full raw and filtered proposal records match exactly. Packing/one-step feasibility code and budget8 were not modified. J1 verifies before/after each full engineering update.

Accepted-only likelihoods, treating deterministic filter acceptance as another stochastic action, or adding two branch log-probabilities would be wrong; none occurs here. Gaussian reference KL is not the KL of the effective accepted-output controller.
''')
doc('SAFETY_MONITOR_STATISTICAL_DESIGN.md',f'''# Frozen prospective safety-monitor design

Population:128 qualified TRAIN homology groups,256 unchanged witness/relaxed tasks, deterministically hash-selected without outcomes from the existing434-group fitting pool. Remaining306 groups/612 tasks fit future arms. The old16 reserved TRAIN goals remain separate descriptive/property/KL/diversity sentinels. This common partition change applies to both arms and is disclosed; historical populations/results remain untouched. Source tasks retain the original qualified TRAIN/witness bindings, sequences and thresholds. Monitoring estimates the uniform finite-population average probability of hard/backbone-valid generation under a frozen actor and unchanged finite8 controller.

At each look0,250,500,1000,2000,3500,5000:4096 identity/goal draws independently with replacement from256 tasks, ONE new candidate per draw, batch size1. Identity and candidate RNG streams are separate from fitting, candidate seeds unique across looks, task/seed schedules matched across arms. Repeating an identity does not create stochastic dependence if its draws and candidate randomness are independent; conditional on the fixed finite task probabilities, the unconditional outcomes are iid Bernoulli with mean p. This is not4096 unique independent homology groups. No cached candidate reuse, same-cluster shared random shock, outcome-adaptive sampling, batch-dependent semantics or policy changes during a look are permitted. Model fingerprint and source/checkpoint binding are checked. Unknown shared randomness invalidates binomial inference and requires INCONCLUSIVE/infrastructure pause, not blind use of the interval.

Use exact two-sided Clopper-Pearson intervals with alpha=.05/(2arms*2metrics*7looks)={ALPHA}. Bonferroni controls family noncoverage across28 intervals without requiring independence across arms or looks; within-look conditional iid assumptions remain necessary. New fresh look RNG keeps validity conditional on past stopping decisions. This finite TRAIN estimand is not a guarantee for every individual identity, every possible protein, or an untouched population.

PASS: backbone lower bound>=.95, hard lower bound>=original hard floor/preservation bound, original KL<=1 and diversity floors/preservation, numerical and resource safeguards all pass. BREACH: a required upper bound is below its floor, or another deterministic original safeguard fails. Otherwise INCONCLUSIVE. Missing/uncommitted/corrupted evidence is incomplete/infrastructure pause. Aborts count as backbone/hard failures. Initial hard floor=.91875; later bound=max(initial_hard,.96875)-.05. Initial diversity>=12.44911875; later>=.8*max(initial_diversity,15.561398437372143). Original per-stepKL<=.02 and all other original runtime guards remain active.

Backbone-only N4096 boundaries:>=3934 successes supports PASS;<=3845 supports BREACH;3846..3933 is INCONCLUSIVE. Full decisions also enforce hard/KL/diversity. No optimizer update is permitted after any non-PASS. An incomplete interrupted look resumes only its exact missing frozen suffix on the same preserved checkpoint. Complete INCONCLUSIVE results receive NO additional samples or training under this protocol. Additional collection would require a separately reviewed prospective error-controlled amendment, never repeated draws until favorable. Previous V9.6 stops remain binding and are not retrospectively overridden.

Durable implementation: safety97.Monitor commits each full candidate result then a SHA256 commitment with arm/look/checkpoint/source/schedule/model bindings. Restart verifies a contiguous prefix, rejects missing/corrupt committed records and wrong bindings, adopts only a valid complete orphan, and preserves partial uncommitted evidence. Frontier JSON is a convenience, not authority. The collector uses the qualified sampler, same reward/filters, and a frozen actor fingerprint. No production actor updates happen inside it.

References: [NIST exact binomial intervals](https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm); [Howard et al. time-uniform inference](https://arxiv.org/abs/1810.08240). This design uses a finite-look union bound, not an unimplemented confidence sequence.
''')
doc('SAFETY_MONITOR_SIMULATION_RESULTS.md',f'''# Safety simulation and recovery results

{j(cpu)}

Seed1973197;20,000 iid replicates per probability. PASS below95% is the unsafe continuation event; INCONCLUSIVE is a pause and must not be counted as a missed breach that permits fitting. At exactly95%, most looks correctly remain unresolved. Exact operating probabilities accompany Monte Carlo estimates; rare zero event estimates are not proof of zero probability. A heterogeneous deterministic-identity fixture randomizes the identity separately for every candidate and satisfies the contract. The eight-correlated-candidates-per-group negative control deliberately violates it and shows why the same binomial intervals must not be used for arbitrary clustered evaluations.

Repeated-look simulation covers2arms*7looks with hard/backbone equal for that synthetic fixture. Bonferroni does not need inter-look independence, but this simulation does not validate all possible adapted policies or hidden shared random effects. A stable fixed-seed historical128 panel is not modeled as independent fresh binomials. Historical V9.6 stop records and published validation conclusions are preserved. The measured real16-candidate collector run tests execution only and cannot certify the4096-candidate population admission or actual false-stop rate.
''')
doc('PRODUCTION_RECOVERY_QUALIFICATION.md',f'''# Production-path engineering recovery

The original qualified Runtime, sampler and gradient implementation are reused. New root/manifest/source bindings and the gate installer are explicit infrastructure adapters; original scientific algorithms are unchanged. Separate A/B engineering processes load the exact certified5000 actor with fresh original AdamW/scheduler, one heavy GPU worker at a time.

A:{j(ra)}

B:{j(rb)}

Full actor (including gate), Adam moments, scheduler, CPU/CUDA/NumPy/Python RNG, task order, next identity/goal/seeds, counters and source bindings round-trip exactly. An interruption after a durable update commitment but before checkpoint publication restores the prior checkpoint and replays the already committed trace exactly, without new candidates. Actual full Torch checkpoint writes are interrupted by child exit75 after8MiB; the previous index stays authoritative and restores identically. Corrupted copied checkpoint SHA is rejected by the real restore path, and wrong source bindings are rejected. Corruption/partial fixtures are retained under qualification. Monitor missing/duplicate/gap/orphan tests are separate in CPU_AND_SAFETY_QUALIFICATION.json.

No production process has launched. Engineering update counts are disclosed, not described as scientific RL improvement. Before a future campaign, both arms still require full initialization safety admission and an explicit production authorization receipt bound to these sources/configs. A missing receipt, non-PASS safety status or mismatched fingerprint blocks updates. Complete INCONCLUSIVE must remain paused rather than bypassing the gate.
''')
doc('SUBSTANTIAL_PAIRED_TRAINING_PROTOCOL.md',f'''# Frozen future paired training protocol — authorization required

ArmA: original actor/encoding,17438114 parameters. ArmB: same actor tensors plus scalar gate0,17438115 parameters. Both initialize certified R3 milestone05000 SHA256 d4078113b2103da025245be76be2f9e703fb673a7b6c96144f13649001d9d216; frozen J1 SHA256 881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412. Never use V9.6 A500/B250 as full-budget comparator or initializer.

Each targets5000 additional updates x16attempts=80000. Fitting306 qualified TRAIN groups/612 unchanged goals,128 groups/256 tasks reserved for the common randomized safety monitor. Both schedules are frozen in TRAINING_SCHEDULE.json, seed19731, and MONITOR_SCHEDULE.json. The new common partition is a measurement-design change, not a treatment difference. Original goal distributions/thresholds, fixed sequences, reward, full-trace RLOO, AdamW fresh moments/scheduler, LR, clip5, KL beta, finite8 packing/feasibility filters, checkpoint schedule and original safeguards are unchanged. RNG is matched at indexed candidate/residue/rejection boundaries; differing acceptance paths may consume different indexed draws. Independent arm checkpoints, journals, RNG and logs.

Only treatment difference: explicit function-preserving burial residual gate. No new critic, Transformer, reward term, entropy bonus, AA learning, SAC or high-level policy. One training seed pair permits model-conditional inference, not algorithmic robustness. ArmA thenB sequential GPU execution with release verified; no concurrent heavy CUDA. No safety-stopped arm may be retried with changed settings. Infrastructure failure preserves the exact committed frontier; use qualified restoration only, never fresh schedule substitution.

Both arms require full4096-draw initialization safety PASS, source/model/storage checks and authorization before either production arm starts. This engineering package did not run full initialization admission. Look callbacks are0,250,500,1000,2000,3500,5000, with no optimizer continuation while safety is incomplete, BREACH or INCONCLUSIVE. Other original KL/nonfinite/storage/diversity guards stay active. The old16 TRAIN sentinel supplies descriptive K1/K8/raw/filtered property curves and original KL/diversity gates; it does not add an undeclared128-candidate absolute-validity test.

Monitor60s; durable reports1800s; immediate evaluation/pause/recovery events. Track reward by identity/difficulty, raw/filtered validity/K1/K8, exposure/burial, full success, property error, proposal exhaustion/rejections, KL, gradient norm/clipping, entropy, diversity and compute. Do not infer learning from changing TRAIN goal difficulty. Checkpoints dual-slot plus named/best as originally; final5000 is scientific primary, not a favorable selected checkpoint. Partial/asymmetric budgets are published as such with no full-budget causal claim.

Safety cost57344 attempts across both arms, additional to160000 fitting attempts and89088 final assessment attempts. Resource forecast and measured throughput are in RESOURCE_AND_RUNTIME_ESTIMATE.json. Do not lower training budgets to offset the two-forward cost. Gate0 equivalence does not certify a positive safety margin on the new finite TRAIN population. No production training is authorized or launched by this file.
''')
doc('VALIDATION_COMPARISON_PROTOCOL.md',f'''# Frozen scientific assessment protocol

Use the byte-identical qualified V9.6 VALIDATION_PANEL_MANIFEST.json (512 independent groups,1024 primary goals;80 matched diagnostic groups/240 separately generated diagnostic goals). Original candidate seeds, thresholds, primary success criteria and diagnostic invalid-pair-zero conventions remain unchanged. Diagnostics may have looser native-supported targets; never combine them with primary denominators. This repeatedly used VALIDATION is development assessment, not untouched replication or sealed TEST. Only TRAIN is used for engineering/fitting/safety.

Primary: raw regional burial responsiveness S(B5000)-S(A5000), point>=.01 and paired95% lower>0. Also require difference in changes >=.01 and positive95% lower, B-own change positive lower, and Bfinal S>=.01 with positive lower. Paired identity/homology bootstrap10000 draws seed19329 across80 diagnostic groups; these are intersection requirements, not jointly simultaneous95% intervals. Sequence/motif/thresholds fixed within each goal-swap triple; only requested regional instruction differs. Initial policies are identical by qualified gate0 contract; evaluate/verify baseline explicitly, do not infer treatment effect from final yields alone.

Secondary8 families(raw/filtered K1,K8,burialS,exposureS)*4contrasts(Achange,Bchange,B-A,difference-in-changes), Bonferroni.05/32,99.84375% intervals. Primary K1/K8 denominators1024 goals with8 candidates each8192 attempts per actor/mode; K1 is candidate0, K8 any of8, aborts failures. Original K1 promotion against certified starting policy requires>=5pp, paired95% lower>0, K8 preservation and original validity/KL/diversity contract. Report raw learning separately from deterministic filtering, and exposure preservation with effect intervals; no nonsignificance-as-noninferiority claim. Report property-target error, simultaneous satisfaction, structural diversity, reference proposal KL and costs.

Final evaluation includes J1, A0raw/filter,B0raw/filter,Afinalraw/filter,Bfinalraw/filter:89088 total attempts under the existing schedule. No validation inference was launched in this qualification task. No prior stopped policy substitutes for equal-budget5000 endpoints. If either stops early, evaluate preserved partial endpoints only under disclosed asymmetric exposure. No TEST/native-future coordinates/AlphaFold/AA/high-level RL, and no automatic subsequent research stage.
''')
ready=bool(storage and cpu['PASS'] and eq['PASS'] and gr['PASS'] and ra['PASS'] and rb['PASS'])
status=dict(FUNCTION_PRESERVATION='PASS',GATE_GRADIENT='PASS',POLICY_LIKELIHOOD='PASS',SAFETY_STATISTICAL_DESIGN='QUALIFIED',SAFETY_MONITOR_RELIABILITY='QUALIFIED_UNDER_FROZEN_CONDITIONAL_IID_DESIGN; INCONCLUSIVE_PAUSES',RECOVERY_QUALIFICATION='PASS',INITIAL_POLICY_IDENTICAL='YES',ADDITIONAL_PARAMETERS=1,ESTIMATED_TRAINING_RUNTIME=estimated,STORAGE_ADMISSION='PASS' if storage else 'FAIL',ENGINEERING_READY=ready,READY_FOR_SUBSTANTIAL_RL='NO',BLOCKER='Full4096-draw initialization admission and explicit production authorization remain pending; no campaign launched',PRODUCTION_UPDATES=0,TEST_SEAL='UNKNOWN_HISTORICAL_ACCESS; NO_TEST_CONTENT_ACCESSED')
save('QUALIFICATION_STATUS.json',status)
doc('FINAL_E017_V97_ENGINEERING_REPORT.md',f'''# Final V9.7 engineering qualification

{j(status)}

The gate correction, policy/gradient/recovery pathways and fixed-look monitor design qualify under documented assumptions. All historical scientific conclusions remain unchanged. Full4096-candidate initialization safety admission is intentionally outstanding; the16 real monitor candidates exercise collection only. ENGINEERING_READY distinguishes implementation qualification from permission to start production. Both future arms still need actual safety PASS and reviewed authorization. No production process/optimizer schedule is launched.

New generation:128 equivalence attempts +16 randomized single-candidate monitor attempts +32 fresh fitting engineering attempts =176. Actual engineering optimizer calls4, comprising1 unique update per arm plus exact replay; production updates0. Synthetic different-length fixtures are robustness checks, not a changed task. All workers exit after their bounded checks. No TEST/VALIDATION candidate generation or J1 update occurred. Eight reports, training/validation protocols, resource forecast, machine-readable evidence and SHA256 provenance are included.
''')
# Verify entire prior paired publication and stability review without modifying either.
prior=read(PAIR/'PROVENANCE_SHA256.json')['files']
for p,v in prior.items():assert sha(p)==v['SHA256'],p
review=ROOT.parent/'e017_v96_stability_review_20261031_v1';rr=read(review/'RESOURCE_AND_PROVENANCE_REPORT.json')
for p,v in rr['review_files'].items():assert sha(review/p)==v['SHA256'],p
verify();files={str(p):dict(SHA256=sha(p),bytes=p.stat().st_size) for base in [ROOT,WT/'e017_v97'] for p in base.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(base).parts and p.name!='PROVENANCE_SHA256.json'}
save('PROVENANCE_SHA256.json',dict(files=files,self_excluded=True,runtime_excluded=True,historical_pair_files_verified=len(prior),historical_review_files_verified=len(rr['review_files']),source_binding_SHA256=sha(ROOT/'SOURCE_BINDING.json')))
for p,v in files.items():assert sha(p)==v['SHA256']
save('runtime/PUBLICATION_VERIFIED.json',dict(PASS=True,UTC=now(),files=len(files),production_updates=0,GPU_processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout))
print('V97_PUBLICATION_VERIFIED',len(files),ready,flush=True)
