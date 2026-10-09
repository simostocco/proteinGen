# E014 unconditional coordinate-v scale-up v1

Base: frozen E013 commit `dd421bde2a6b3a9b5eb549f72b874ae2ca875bac`.
Branch: `e014_unconditional_coordinate_v_scaleup_v1`.
Worktree: `/mnt/d/Simone/proteinGen/.e014_worktree`.
Generated artifacts: `/mnt/d/Simone/proteinGen/e014_artifacts` exclusively.

## Motivation and interpretation boundary

E013 was a bounded pilot: 980,794 parameters, 400 matched updates, 20 training
identities and 17 homology-clean validation identities. Its checkpoint improved
aggregate v-MSE about 7%, but local-distance MSE worsened 35.6%, torsion MSE
improved only 2.1%, and timestep-490 reconstruction collapse occurred in 10/17
versus 2/17 controls. These historical results remain frozen. They establish that
the **400-update checkpoint** failed its structural gates, not that the
representation or local branch is incapable. E014 tests capacity, data coverage
and optimization duration together; it cannot separately attribute those factors.

No architectural conclusion is permitted before **two complete TRAIN epochs**.
Poor early geometry is an outcome to track, not an early-stop condition.
No de novo generation or biochemical claim follows from reconstruction metrics.

## Unchanged representation and inference contract

The model remains **p(X|N)**. Inference accepts only current noisy C-alpha
coordinates `[B,N,3]`, timestep `[B]`, requested length `[B]` and a chain-order
mask `[B,N]`. Polymer order and current geometry provide all derived features.
No amino-acid sequence, target, target RMSD, teacher, SLSQP, safety shield,
self-conditioning or external correction enters inference.

E014 subclasses E013 without editing its source. The subclass only replaces
ModuleLists/local projections to configure widths and depths; it inherits the
entire frozen forward computation. Default E014-small initialization/state and
forward output match E013 exactly with a fixed seed and 980,794 parameters.
The scaled model retains time width 64, length width 16, five position scalars,
85 raw node scalars, 22 edge scalars, 32 local features, 32-dimensional RoPE q/k,
zero-initialized vectors, k=24 current-coordinate spatial neighbors, polymer
offsets +/-1/2/3, and coarse stride 8. The two local residual blocks are unchanged
in number; local hidden width scales with capacity. Channel scalar/vector updates,
frame construction, chirality features, reliability and the `(1-t/499)^2` local
envelope retain their E013 definitions.

RBF: 16 overlapping Gaussians, centers 0..5 normalized units, gamma equal to
inverse squared center spacing. Frozen scale is **12.22820347644835 Å**. No
validation/test normalization fitting occurs. Normalization SHA256:
`1be9ef05c82110861482eb46e61ef433045fa9678b9cdfffd921f0defebb03cd`.

## Capacity ladder and selection (completed before scientific training)

| Candidate | Scalars | Vectors | Sparse blocks | Global blocks | Local width | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| MEDIUM |192|48|5|2|192|2,378,155|
| LARGE |256|64|6|3|256|5,119,069|
| XL, conditional |320|80|6|3|320|7,867,597|

**LARGE selected**: the smallest predeclared member in 5M–10M that passes finite
gradients, symmetry and N=500 memory comfort. XL's parameter count was checked,
but its CUDA preflight was unnecessary because LARGE met the target. No
validation quality informed selection. E007's unchanged architecture has
7,586,505 parameters; E014 is in the requested broad capacity range, with fewer
parameters than E007. This capacity difference must be retained in interpretation.

LARGE module counts: time 8,320; length 336; input 22,016; sparse blocks 3,019,014;
global blocks 1,509,507; scalar broadcast 65,792; vector broadcast 4,096; global
head 82,240; local input 8,448; local residuals 394,240; local head 771; gate 4,289.

Scalar/vector states are `[B,N,256]` and `[B,N,64,3]`. Coarse states are
`[B,ceil(N/8),256]`, `[B,ceil(N/8),64,3]`; M=63 at N=500. Local hidden state
is `[B,N,256]`. Final centered coordinate-v remains `[B,N,3]`.

Sparse message complexity is O((k+6)N), coarse communication O((N/8)^2), and
local features/messages O(N). The inherited kNN discovery still uses dense cdist
and stable sorting: O(N² log N) time and O(N²) storage. It is not a fully linear
algorithm. Variable-length proteins are processed separately inside the inherited
forward method; physical batching saves optimizer overhead but does not vectorize
the residue message computation.

## Verified full population and frozen panels

TRAIN is derived from the hash-verified authoritative accepted-coordinate
identity-30 cluster assignments, not a random-draw panel: **230,132 identities**.
The canonical sorted TRAIN sample-ID SHA256 is
`d3f2ee345d0e4e9a27d0136aad1eca9dfe87821a3a82d90aa4c3194bd928a3f9`, matching
the frozen normalization's fitted identity hash. Definition: SHA256 of compact,
sorted-key JSON encoding of the sorted sample-ID string list. Train manifest hash:
`28500b3b23a2d1ffab8ce5e5e17271fcd2e9514c4b9708e483af5f86e3cd476d`.

| TRAIN length stratum | Identities |
|---|---:|
|20–64|21,390|
|65–128|66,547|
|129–256|90,077|
|257–384|38,489|
|385–500|13,629|

Clean validation contains 21,583 identities under the frozen conservative
30%-identity/80%-coverage exclusion. Train and clean validation have no shared
identity, homology cluster or PDB entry. From validation, rank
`SHA256(e014-panel-v1|sample_id)` within each stratum. Exclude the historical
panel; select first 100 per stratum for development (**500**), then next 400 for
prospective final validation (**2,000**). The panels are disjoint and metric-blind.
No test is accessed. Development identity hash:
`3a36bccfee801b618674594762ed2e6e394220ca21db94260ccb5becf7328dcf`;
prospective identity hash:
`373cb7c0b2f0f7b64fdd5d47d742cf7f07b8b0ddbc0da8683032375cd8dec25b`.
Exact selections and all source/manifest hashes are in `population/`.

A D-resident read-only packed float32 coordinate cache contains 232,649 unique
identities (TRAIN plus both panels and 17 historical validation identities),
42,769,187 residues, and 247 source-shard hashes. The cache has no sequence
column. Every accessed source shard was hash-verified; accepted coordinates,
masks, continuity and lengths were checked before caching. Full TRAIN loader
indexing uses only the TRAIN manifest, never the validation rows sharing the
physical cache. Packed-coordinate SHA256:
`c1b7e2f52db9194587becf35dd1639a0b464c93eae755795d1d5a876dd11f723`.

## Complete epochs, optimizer and objective

An epoch exposes every TRAIN identity exactly once. PCG64(seed+epoch*1000003)
shuffles indices within length buckets; then shuffles the resulting batches.
Batch sizes are `[4,2,1,1,1]` for the five strata. No identity resampling or length
weighting is introduced. All schedules are persisted on D and regenerated/hash
checked on resume. Epoch 1 has **180,817 optimizer updates**, no duplicates or
missing identities, min/max scheduled exposure both one. Schedule hash:
`9a32c939108db08ae8ad11599ce6c62a8f9edae31a63be5dac519b002cd5f2c1`.
Completed epochs assert min=max=epoch over all identities and persist exposure
counts/hash. Pending updates remain provisional until checkpoint commitment.

Reuse unchanged E007 CoordinateVPDiffusion(500):
`X_t=alpha_t X0+sigma_t epsilon`, centered Gaussian epsilon;
`v*=alpha_t epsilon-sigma_t X0`;
`X0_hat=alpha_t X_t-sigma_t v_hat`. The sole objective is equal-protein masked
coordinate-component MSE, normalized by 3N per protein then averaged over batch.
AdamW: lr=2e-4, weight decay=0.01, default betas/eps matching E013's PyTorch stack,
gradient clip=1.0. Uniform integer timestep sampling and all diffusion semantics
are unchanged. No LR schedule, warmup, EMA, geometry objective or AMP. TF32 is off.
Per-identity/epoch deterministic RNG streams make corruptions reproducible across
batching and a future comparator without changing the uniform sampling law.

Train for **at least two, at most four** complete epochs. At an epoch >=2, plateau
stopping requires primary development v-MSE relative improvement <1% for two
consecutive completed epochs. Initialization is the epoch-zero reference. Poor
local geometry/radius/torsion cannot stop training. Technical hard stops are
nonfinite/exploding loss or gradients, data/provenance/checkpoint corruption,
GPU/runtime failure, symmetry regression, D free space below 20 GiB, or sequence
contention. A hard-stop result before epoch 2 has no architectural interpretation.

## Atomic checkpoint and exact resume contract

Permanent verified snapshots at initialization and 25/50/75/100% of each epoch.
Two rolling recovery slots also limit at-risk provisional work to about 15
minutes. Save model, optimizer, all RNG states, epoch, next schedule position,
next batch, schedule hash, per-identity exposure accounting, losses, curves,
pending evaluation boundary, contention state, config/population/cache/source/
normalization hashes and git SHA. Write `.partial` on D, fsync, reload and compare
the entire payload (including optimizer/RNG tensors), hash, then atomic rename.
Publish LATEST.json atomically; never replace the currently referenced rolling
slot before committing its successor. A failure-injection test preserves the
previous committed checkpoint if pointer publication fails.

One advisory owner lock refuses concurrent trainers/resumes. Resume verifies all
contracts, exact next batch and exact expected exposure vector. Records beyond
LATEST are explicitly provisional and rollback is recorded in RESUME_<attempt>.
json; committed optimizer updates are never silently replayed. A pending metric
boundary is completed before resumed optimization. Clean SIGINT/SIGTERM and
contention pauses save a verified checkpoint at the current completed update.

GPU atomics retain E013's numerical behavior; exact next identities/corruptions
and serialized optimizer/RNG state are guaranteed, not cross-hardware bitwise
floating-point trajectories. CPU optimizer/RNG continuation tests are bitwise.

## UniRef50 coexistence

UniRef50 production files are read-only inputs for contention monitoring. Never
stop, restart, renice or modify the sequence job. OMP/MKL/OpenBLAS/NumExpr threads
are one; loader workers zero. All scratch/cache/temp paths are explicitly on D.
Read recent native completed-batch candidates/sec from the heartbeat, not repeated
pending-batch reports. Baseline is the median of the last five completed batches.
The first **600 seconds of actual full-population training** is a contention
probe and counts as legitimate epoch exposure. Require at least three completed
sequence batches during the probe. A single noisy batch cannot trigger pause.
When the active phase is protected verification, the screening-rate field is
stale: instead use completed audit population deltas and native archived
`command.rss.txt` wall times. These sequence files are read-only. An unsupported
active phase or phase transition pauses E014 rather than mixing rate definitions.
Pause E014 when three consecutive completed-batch throughputs are below both
90% of the baseline median and baseline minus twice its median absolute deviation.
An inconclusive probe or stale nonterminal sequence heartbeat also pauses E014
with a checkpoint. This is a conservative priority guard, not proof of causation
for every later sequence slowdown. Explicit terminal sequence completion removes
the stale-heartbeat check. The native driver terminates with
`frozen_corpus_ready`; this state is explicitly recognized. A persistent
supervisor waits through unsupported finalization phases and resumes only when
the sequence corpus is ready. It never restarts a numerical/provenance/runtime
hard failure. The sequence job is never interrupted.

The initial probe ran from `e7fb20e6a65f504e16206bd80c245d57fd27457a`.
A narrowly audited checkpoint handoff recognizes the native terminal phase:
the entire runner diff must be exactly one added terminal-guard line. Preserve
the original probe checkpoint and write a new, fully verified checkpoint with
the replacement runner hash. Model, optimizer, RNG, schedule, losses and
per-identity exposure state remain unchanged. Standard arbitrary source/config/
data mismatch refusal remains active. The handoff record is
`e014/TERMINAL_GUARD_HANDOFF.json`; its existence establishes execution.

Executed: 604.02-second probe, 2,198 optimizer updates and 2,802 TRAIN exposures.
Before/during verification medians: 994.13/1,074.92 candidates/s (ratio 1.0813);
no persistent material degradation. Handoff execution preserved all original
scientific payload fields. The first resumed update was 2,199, identity
`7mfn_AAA`; no committed optimizer update was replayed. Live proof:
`e014/LIVE_RESUME_VERIFICATION.json`. Persistent supervisor/child resumed from
training source commit `f86b1f997d147b50fe1445b2939a63accf4d88ea`.
The original E013 evidence and original E014 probe checkpoint are preserved.
Measured probe TRAIN exposure throughput was 4.64 proteins/s: about 13.8 hours
per full epoch if sustained, excluding evaluation and sequence-priority pauses.

## Evaluation boundaries and metrics

Freeze paired noises at t=5/50/250/400/490 using hash(namespace|identity|t),
unchanged across checkpoints. Development at initialization and every quarter
boundary (including the required 0.25/0.5/1/2/3/4 exposures reached). Historical
20/17 panel at initialization, epoch 1 and final uses E013's exact split-specific
index noise seeds `1300000+index*500+t`; it never selects checkpoints. Prospective
2,000-identity final validation is disjoint from development; its initialization
comparison is evaluated only after training, with the frozen initialization model.

Record equal-protein v-MSE; unaligned X0 component MSE; proper Kabsch RMSD;
offset-1/2/3 distance MSE and mean; adjacent mean/RMSE/quantiles and [3.4,4.2] Å
valid fraction/count; pseudo-torsion sine/cosine MSE and wrapped angular MAE;
normalized signed-triple MSE, inversion counts/rates with assessable denominators;
Rg predicted/reference and ratio quantiles; exact E013 collapse (Rg_hat<0.25Rg_ref)
with count/denominator; clashes at <3 Å excluding separations <=3; long-range
contacts <8 Å at separations >=12, with pair denominators; finite status.
All metrics retain noise, length and length×noise strata. Rates never omit counts.
Continuous Rg-ratio distributions prevent a binary collapse gate hiding scale.

Read-only temporary hooks capture global vector output and effective gated local
output; center each so their sum matches prediction. Report both norms, their
ratio rho and cosine, raw sigmoid gate mean/median, reliability mean/median and
frame assessability count/fraction. Hooks are absent during training and change
no output. Per-update total raw gradient norm/clipping, and bounded 500-update
parameter/gradient/optimizer-step norms by trunk/local/heads, are logged without
tensor dumps. Boundary summaries include parameter norms and clipping fraction.

## Predeclared final interpretations and sampler authorization

Only after >=2 complete epochs: A: substantial v and structural improvement;
B: v improves but structural discrepancy persists; C: v does not improve;
D: high-noise behavior becomes acceptable but low-noise refinement remains poor;
E: low-noise geometry improves but high-noise collapse remains. Quantitative
operationalization is frozen in config before training: >=10% improvement versus
paired E014 initialization, low-noise adjacent valid fraction >=90%, t490 collapse
<=5%, and t490 non-neighbor clash fraction <=1%. These are operational feasibility
criteria, not biochemical validation. E013/E007 historical context is non-matched.
Only case A authorizes a later sampler-exposure study; even then this runner does
not automatically run reverse trajectories. All other cases stop after reporting.

## E007 comparator and resource estimates

An adapter and separate entrypoint prepare the unchanged E007 architecture with
the same full data, schedules, identity noise streams, optimizer and panels.
Primary E014 must complete first. A post-primary resource assessment is required
before launching it. E007 CUDA preflight passed numerics but N=500 reserved about
6.76 GiB, above the declared comfortable 5 GiB envelope. Thus its matched full
run is prepared, **not executed**. Historical E007 comparisons remain non-matched.

Cold/noisy preflight estimates (excluding loader/evaluation) were E014 LARGE
28.64 hours/epoch, 57.28–114.55 hours for 2–4 epochs; E007 13.58 hours/epoch,
27.16–54.31 hours for 2–4 epochs. These are conservative extrapolations from one
backward per representative length under CPU contention, not promised completion
times. The live contention probe provides measured full-pipeline throughput.

## Implementation and verification status

Implemented: new capacity subclass, full-population manifests/packed cache,
complete-epoch schedule, resumable trainer, atomic checkpoints/owner lock,
paired evaluation, component/gradient telemetry, contention guard, matched E007
adapter/entrypoint, final reporting. E013 source/results and E007/E010 sources
are unchanged. No final experiment interpretation is available yet.

CUDA capacity preflight passed: proper rotation/translation tolerance 5e-5,
maximum LARGE absolute error 3.58e-7, relative L2 3.53e-7; parity-sensitive
reflection gaps 3.14..8.75. N=500 LARGE: 1,163,598,336 allocated bytes,
1,369,440,256 reserved bytes; forward 32.44 ms; forward/backward 108.92 ms.
The selected physical batches `[4,2,1,1,1]` also passed CUDA forward/backward
and AdamW steps. N=500 with optimizer state allocated 1,204,750,336 bytes and
reserved 1,421,869,056 bytes; warmed full-update latency was 361.07 ms under
CPU contention. All 19 frozen E013 artifact hashes were verified. There is no
TRAIN/clean-validation identity, homology-cluster or PDB-entry overlap.
Focused suite: **23 passed, 1 deselected**; the subsequent active-phase monitor
scope passed **4 tests**, including the new verification-timing regression.
The operational/reporting scope passed **3 tests**, including refusal of any
unrelated handoff source change and effective-gate agreement with live hooks.
The deselected historical cache-env
test would select E013 runtime directories; equivalent E014 D-path rejection
tests passed. The focused unit scope includes E013 regression, N=20/500,
padding/degeneracy/parity, full-epoch coverage, mid-epoch exposure/optimizer/RNG
resume, hash refusals, atomic failure injection, deterministic panels/noises,
D-only paths, metric denominators and contention persistence.

Preflight commands executed from this worktree with PYTHONDONTWRITEBYTECODE=1
and D TMPDIR: `scripts/preflight_e014_scaleup.py --stage population|capacity|cache`
and `scripts/audit_e014_execution_boundary.py`.
Training entrypoints (execution status below):
`scripts/run_e014_scaleup.py --probe-only` for a checkpointed initial probe;
`scripts/run_e014_scaleup.py --resume` to continue the exact full run after a
passing probe. Both primary and matched control finish with final reports and
factual documentation updates; no sampler is launched. Live records are
`e014/HEARTBEAT.json`, `COMMITTED.json`, `CURVES.json`, `TRAINING.jsonl`,
`checkpoints/LATEST.json` and `evaluation/`. No claims about folds/stability/function.

`scripts/launch_e014_background.py` starts persistent guarded continuation after
a passing probe. `SUPERVISOR.json` records training/waiting/hard-stop status.
Post-training `scripts/report_e014_effective_gates.py` evaluates only the frozen
checkpoint time/gate modules on paired current noisy coordinates, adding the
exact mean/median of `sigmoid_gate * reliability * noise_envelope`. The main
evaluator already reports learned sigmoid gates and effective vector norms.
This supplemental pass never changes the model, optimizer or training objective.

Next scientific experiment is **not preselected as a local-branch ablation**:
choose it from the adequately trained A–E result. Before that boundary, the next
action is continuing E014's complete-epoch training subject to coexistence.

## Execution status (automatically refreshed)

Updated unix timestamp: 1791527466.235. Phase: **training**.

TRAIN population: 230132; sample-ID hash `d3f2ee345d0e4e9a27d0136aad1eca9dfe87821a3a82d90aa4c3194bd928a3f9`. Selected model: 5,119,069 parameters, {'scalar_channels': 256, 'vector_channels': 64, 'sparse_blocks': 6, 'global_blocks': 3, 'local_hidden': 256}.

Optimizer updates: 2980; protein exposures: 3791; exposure min/max: 0/1.

**The required minimum two complete TRAIN epochs has not yet completed. No architectural conclusion is available.** E013’s v-MSE/geometry divergence remains an open question. No sampler exposure or de novo sampling is authorized at this stage.

## Execution throughput optimization

2026-10-09: audited and benchmarked; **not deployed**. Five-minute live baseline: 4.576 proteins/s, 3.593 updates/s, GPU utilization mean 11.55%, approximately 2.68 GiB total device memory. CPU dispatch of thousands of small kernels per update is the leading bottleneck. Rolling checkpoints occur every 900 seconds in addition to quarter-epoch/guard/signal checkpoints.

The original coexistence guard saved `contention_pause_1791527258268763959_8113.pt` (SHA256 `66e0712279172ff60dc01360090329db7e599f1aa82209824bd61060bbfa8d1c`) at update 8,113 / 10,262 exposures, then paused for uncertain sequence heartbeat. The unchanged supervisor waits for sequence CPU availability. No switchover/rollback occurred.

An inactive execution-cache candidate met output/loss/gradient limits but missed the frozen 5e-7 parameter limit across five actual paired TRAIN batches. Original-versus-original also missed it, so required tight equivalence is unresolved; cache-specific harm is not established. UniRef throughput subsequently decreased substantially, preventing deployment pending coexistence clearance. Larger batches fit memory but change effective optimizer batches because accumulation is one. No production speedup is claimed. Model, objective, precision, optimizer, corruption, grouping and schedule remain unchanged.

See [full profiling, checkpoint and memory report](/mnt/d/Simone/proteinGen/e014_artifacts/throughput_optimization/REPORT.md) and adjacent bounded JSON evidence. Next engineering step: alternating warmed static-only cache trials after sequence availability, with an explicitly justified CUDA-repeatability acceptance contract, exact optimizer-state/resume audit and sustained throughput window.

## Explicit concurrent-work resume policy

2026-10-09: the user explicitly permits simultaneous UniRef/MMseqs and DATA2-D diagnostics and accepts performance contention. `scripts/launch_e014_background.py --allow-concurrent-sequence` now launches the existing `scripts/run_e014_scaleup.py --resume --allow-concurrent-sequence` in the original persistent D-logged background method. This flag changes only sequence contention/phase/heartbeat pause policy; monitoring stays read-only. Scientific/numerical, disk, checkpoint and runtime failures remain fatal. The config, CPU threads=1, loader workers=0, batches, optimizer, diffusion, data, panels, schedule and 900-second/quarter-epoch checkpoint cadence remain frozen. No cache/AMP/compile acceleration is enabled.

The minimal runner patch must reverse byte-for-byte to frozen source SHA256 `3063e842f6cd93f0fdcc8a86d1bcd70b0598b51386e01c8dad0da06dd4608601`; any additional scientific source change refuses resume. The checkpoint scientific source contract retains that exact audited identity; actual execution-runner/helper hashes and explicit concurrency policy are separately recorded in `e014/EXECUTION_POLICY.json` and normal newly written checkpoint metadata. Existing checkpoints are not rewritten or synthesized. Regression tests cover exact frozen-source recovery, rejection of optimizer/unrelated edits, and preservation of degradation telemetry.

Prelaunch selection: `contention_pause_1791527258268763959_8113.pt`, full SHA256 `66e0712279172ff60dc01360090329db7e599f1aa82209824bd61060bbfa8d1c`, epoch 1 / position and update 8,113 / exposure 10,262. Exact next batch is index 168675 (`8g8q_A`). D-resident resume telemetry and the postlaunch rolling-checkpoint verification are stored under `e014_artifacts/concurrent_resume/`; these operational counters are not scientific adjudication.

## Interim 50%-epoch observation — non-adjudicating

2026-10-09: read-only CPU analysis reused the frozen paired development evaluations at initialization, 25% and nominal 50%. The active trainer continued throughout; no new model/GPU evaluation was required. Nominal 50% is the first completed optimizer batch crossing 115,066: update/cursor 90,413, actual 115,067 exposures (+1), previous batch 115,065. Checkpoint SHA256 `2e3eb451b4581931c3eefdf79d3c974a289ed5e61489a72310faec673e51cd29`. Frozen panel: 500 homology-clean validation IDs (100 per length bin), hash `3a36bccfee801b618674594762ed2e6e394220ca21db94260ccb5becf7328dcf` in manifest order; no prospective/test evaluation.

Interim descriptive pattern **C**, without adjudication: aggregate v-MSE 0.900314 → 0.432746 → 0.413250 (25→50 improvement 4.50%); local-distance MSE 34.3504 → 8.97413 → 9.73401 Å² (25→50 worsening 8.47%). Low-noise local geometry improves; mid/high-noise local geometry worsens. t=490 collapse 500/500 → 264/500 → 500/500, mean Rg ratio 0.0370 → 0.2452 → 0.1123. Overall torsion/triple-product diagnostics improve modestly, with t=490 inversion near 50%. The effective t=490 local branch is small; its relative norm rises mainly as the global component shrinks, and no same-timestep noncollapsed 50% controls exist. This does not identify a causal architecture component.

Training cumulative v-loss declines 0.357303 → 0.343201 and development v-MSE improves in all five length strata. No scientific gate, final architectural conclusion, plateau stop, sampler exposure, de novo generation, matched control or retuning was performed. The minimum two complete TRAIN epochs remains pending.

Post-evaluation snapshot: 140,989 exposures (61.26% epoch 1), update 110,840; recent 67.2s rate 2.635 proteins/s. Active-training estimates: 75% 3.33 h; epoch 1 9.40 h; epoch 2 33.66 h. These short-window estimates vary with workload and exclude future evaluation overhead.

See [full interim report](/mnt/d/Simone/proteinGen/e014_artifacts/e014/interim_50pct/REPORT.md), [metric tables](/mnt/d/Simone/proteinGen/e014_artifacts/e014/interim_50pct/metrics.json), [all 500 collapse cases](/mnt/d/Simone/proteinGen/e014_artifacts/e014/interim_50pct/collapse_cases.csv), and [training curves](/mnt/d/Simone/proteinGen/e014_artifacts/e014/interim_50pct/training_curve.png). All outputs are D-resident; prior evidence is unchanged.

## 50%-epoch reverse-field diagnostic — non-adjudicating

2026-10-09: the user separately authorized this bounded observational diagnostic while training and corrected E012 screening continued untouched. The frozen nominal 50% checkpoint, SHA256 `2e3eb451b4581931c3eefdf79d3c974a289ed5e61489a72310faec673e51cd29`, remained unchanged: epoch 1, update/cursor 90,413, exposures 115,067. The same 500-ID development panel/noise contract was used (100 per length stratum; hash `3a36bccfee801b618674594762ed2e6e394220ca21db94260ccb5becf7328dcf`). The metric-blind rollout subset had 100 IDs / 20 per stratum, hash `c3e94b415bd7a13269bbb7f3a26372764cbba33b91b16589c7f6906a563020a0`.

The unchanged `CoordinateVPDiffusion` helper passed true-v reverse identity on all 2,500 conditions: maximum absolute error 9.54e-7 normalized units, predeclared tolerance 5e-6. Frozen per-example v-MSE replay differed by at most 4.77e-7. One-step equal-protein next-state MSE at t=5/50/250/400/490: 5.27877e-6 / 2.83596e-6 / 1.88904e-6 / 3.49516e-6 / 7.45080e-6; no-op: 1.15512e-5 / 9.64003e-6 / 8.54668e-6 / 7.74631e-6 / 7.57232e-6. Gains: 54.30% / 70.58% / 77.90% / 54.88% / 1.60%. Extreme-noise step cosine/norm ratio: 0.08617/0.09923; v cosine/norm ratio: 0.08551/0.09952. Continuous direct X0 radius ratio correlates with v norm calibration (Pearson/Spearman 0.98242/0.98223); association does not establish component causality.

All 5,100 short-rollout states were finite. After 10/25/50 reverse steps, normalized coordinate-component MSE to the paired original-epsilon path grew 0.00075640 / 0.00486720 / 0.02045440; exposure gaps were 0.00074923 / 0.00486091 / 0.02044953. Noisy-state mean Rg ratios remained 1.00067 / 1.00155 / 1.00242. Thus direct X0 collapse (500/500, mean ratio 0.11231) did not imply short noisy-trajectory radius collapse, but neither did small absolute one-step error establish good incremental orientation/calibration. Descriptive **R4 + R5 on this paired-reference test**, not R1 and not a final architectural conclusion. The paired path is not the unique valid generative trajectory; conditional-mean denoising at near-pure noise cannot necessarily identify an individual clean reference. These diagnostics do not prove that the marginal unconditional field is wrong.

Execution took ~29 minutes including reporting recovery, with one CPU thread and CUDA FP32 (no AMP/compile/TF32). Peak reserved memory was 356 MiB for one-step inference and 310 MiB for rollout continuation. The first one-step CSV was preserved after a boolean-quantile summary error; reporting-only repair and hash-verified diagnostic continuation did not repeat or alter training. Initial-state transition metadata is clarified in a separate indexed CSV, preserving raw output. Nine focused tests and lint pass. Training advanced from update/exposure 118,280/150,415 to 123,040/156,463, phase training. No model/training/diffusion implementation was modified; only new diagnostic/report code and this documentation were added.

The required two complete TRAIN epochs remain pending. Existing gates/stopping rules are unchanged; no matched E007, full de novo generation or additional sampler authorization was performed. See [diagnostic report](/mnt/d/Simone/proteinGen/e014_artifacts/e014/reverse_field_50pct_v1/REPORT.md), [one-step metrics](/mnt/d/Simone/proteinGen/e014_artifacts/e014/reverse_field_50pct_v1/metrics.json), [rollout summaries](/mnt/d/Simone/proteinGen/e014_artifacts/e014/reverse_field_50pct_v1/rollout_summary.json), and [trajectory curves](/mnt/d/Simone/proteinGen/e014_artifacts/e014/reverse_field_50pct_v1/rollout_curve.png). All generated artifacts remained on D:; historical evidence was preserved.

## 50%-epoch marginal reverse-distribution diagnostic — non-adjudicating

2026-10-09: separately authorized observational population-level evaluation of the same frozen checkpoint, SHA256 `2e3eb451b4581931c3eefdf79d3c974a289ed5e61489a72310faec673e51cd29`, epoch 1 / update and cursor 90,413 / exposures 115,067. The 500-ID development panel (100 per stratum), hash `3a36bccfee801b618674594762ed2e6e394220ca21db94260ccb5becf7328dcf`, and prior 100-ID subset (20 per stratum), hash `c3e94b415bd7a13269bbb7f3a26372764cbba33b91b16589c7f6906a563020a0`, were preserved. Model and checkpoint were verified unchanged before/after inference; training and corrected E012 screening continued untouched.

Thirty-four per-protein rotation/translation-invariant descriptors cover radius, covariance spectrum/anisotropy, pair-distance and chain-separation statistics, and residue norms. Independent source/reference-A/reference-B noise namespaces prevent paired-target scoring. Reference-only median/IQR standardization is fitted per length stratum; pooled results use common columns after stratum normalization. Mean/covariance differences, energy V-statistic, Gaussian-kernel MMD², 64-projection sliced Wasserstein, radius W1/KS and covariance-spectrum W1 are calibrated against independent ref-ref and no-op populations. Each of 84 conditions has 500 deterministic protein-unit bootstraps; preprocessing is held fixed, uncertainty is conditional, and comparisons are exploratory without multiplicity correction. Gains are undefined when no-op does not exceed the reference floor.

One-step energy ref-ref / no-op / model at t=5/50/250/400/490: `0.001367/0.152606/0.007502`, `0.002164/0.016073/0.003019`, `0.003822/0.004978/0.003999`, `0.004360/0.004805/0.004378`, `0.004230/0.003886/0.003855`. Low-noise t=5/50 energy gains are 95.94%/93.85%, MMD² gains 95.98%/93.49%, and sliced-Wasserstein gains 87.79%/88.68%, with model-minus-no-op intervals excluding zero. All five t=490 primary intervals include zero: model-minus-no-op energy interval `[-0.00023137, 0.00019659]`. Thus high-noise marginal improvement is **unresolved**, not established by near-floor distances. At t=5 all five strata improve; t=50 two improve and three are unresolved; all strata at t=250/400/490 are unresolved. Each one-step stratum contains 100 proteins.

The t=490 one-step mean-Rg model/reference ratio is 0.999713. The previous direct-X0 collapse therefore does **not** imply collapse of the noisy marginal. Teacher q490 rollouts after 10/25/50 steps have energy distances 0.027689/0.027178/0.043624 and mean-Rg ratios 1.002677/1.001751/1.009310. Independent reference floors are 0.027423/0.022228/0.025313. Separate centered-Gaussian initialization at t=499 yields prefix energy distances at t=490/480/465/440 of 0.023874/0.023133/0.025766/0.041545 and mean-Rg ratios 0.999580/1.001910/1.000817/1.007816. Neither branch reaches t=0. At t=440 Gaussian mean covariance trace is 1.01535 times reference; adjacent/i+2/i+3 distance-mean ratios are 0.99732/1.00889/1.00867. Descriptor differences grow modestly by t=440 without noisy-scale collapse.

All pooled teacher/Gaussian versus held-initial comparisons are unresolved. All eight pooled Gaussian-minus-teacher metric intervals include zero at all four endpoints; at t=440 energy difference is −0.002079, interval `[-0.044862, 0.054404]`. No additional Gaussian-initialization penalty is resolved, but intervals are broad and each rollout stratum has only 20 proteins. Interpretation: **M1-like low-noise descriptor improvement; M4-like preserved noisy scale with residual shape-statistic differences**. High-noise M1 and M2/M3/M5 are not established. These descriptors omit chirality/topology and do not establish the full coordinate law, clean backbone quality or final architecture success. Small high-noise marginal differences and already-near-Gaussian reference states limit power; nonsignificance is not equivalence. Deterministic sampler discretization and model error are not isolated by this observational test.

The unchanged diffusion helper passed true-v identity on 125 bounded conditions: maximum absolute error 7.15e-7 normalized units, tolerance 5e-6. Final reference-order permutation discrepancy across all conditions is at most 1.78e-15 against unchanged 1e-10 tolerance. The initial rollout CPU analysis correctly refused a 1.93e-10 permutation discrepancy from cancellation in a norm/matmul distance formula; only new diagnostic distance code was corrected to direct squared differences, and analysis was recomputed from preserved descriptors without rerunning inference or changing tolerances. Twenty-two focused marginal/reverse tests and lint/format checks pass. No historical model/training/data implementation changed.

Inference took 1,431.97 seconds, with one CPU thread and unchanged CUDA FP32 (no AMP/compile), peak allocated/reserved 133.1/356.0 MiB. Final CPU analysis took 123.93 seconds including recomputation of one-step comparisons; earlier analysis is archived. Training advanced from update/exposures 129,880/165,194 to 133,840/170,248 during inference and continued afterward. Cumulative throughput changed from 5.98 to 5.86 proteins/s; this is not an isolated causal contention measurement. Both active workloads remained untouched; bounded contention was accepted. No scientific gate, objective, optimizer, model, schedule, training input, stopping rule, matched E007 control or full de novo sampling changed. The minimum two complete TRAIN epochs remains pending: **not final E014 adjudication**.

See [full marginal report](/mnt/d/Simone/proteinGen/e014_artifacts/e014/marginal_reverse_50pct_v1/REPORT.md), [all distribution metrics](/mnt/d/Simone/proteinGen/e014_artifacts/e014/marginal_reverse_50pct_v1/metrics.json), [bootstrap intervals](/mnt/d/Simone/proteinGen/e014_artifacts/e014/marginal_reverse_50pct_v1/bootstrap_intervals.json), and [prefix figure](/mnt/d/Simone/proteinGen/e014_artifacts/e014/marginal_reverse_50pct_v1/marginal_prefix.png). Every generated artifact/cache/temp is D-resident; all earlier evidence is preserved. Next action: continue unchanged E014 to its planned training boundary, then repeat marginal diagnostics alongside denoising/geometry metrics, increasing independent-reference replication if extreme-noise differences remain below the finite-sample floor.

## 75%-epoch high-noise-resolution diagnostic — non-adjudicating

2026-10-09 implementation/preflight boundary: frozen `epoch_1_fraction_0.75.pt` is loadable and internally consistent, SHA256 `e72230c40ae38f6503a1b19785a54bd7162ba6858a2ad8deb4de90a9ed36bdf4`; epoch 1 / update and deterministic schedule cursor 135,676 / exposures 172,599 exactly (zero boundary offset), UTC timestamp 2026-10-09 16:55:49. All initialization/25/50/75 checkpoint model/config/TRAIN/normalization/source contracts and schedule-prefix exposure accounting pass. No live model is used.

The user authorized this bounded read-only diagnostic separately. Protocol is frozen before checkpoint metrics under `e014_artifacts/e014/high_noise_75pct_v1/PROTOCOL.json`, with an additional pre-metric geometry/replay policy. Same 500-ID development panel, independent marginal-noise families, 34 invariant descriptors, reference-only stratum preprocessing, metric implementations, 500 protein-unit bootstraps and 100-ID Gaussian subset/seeds as before. Paired field/clean geometry uses the unchanged historical `e014-paired-validation-v1` family; marginal evaluation uses the unchanged independent `e014-marginal-*` families. All checkpoint comparisons stay within each protocol, never mixing noise-family results.

Predeclared descriptive marginal front: energy model-minus-no-op 95% bootstrap interval upper bound <0 and either MMD² or sliced-Wasserstein upper bound <0, with mean next-state Rg and covariance-trace ratios each within [0.8,1.2]. Highest resolved sampled timestep and contiguous sampled-prefix front are both reported, flagging nonmonotone/isolated crossings. This is not an E014 scientific gate; all multiple comparisons are exploratory and unadjusted. Direct clean-estimate onset (median Rg ratio <0.5 or historical collapsed fraction >50%) is kept separate from actual noisy-state scale (population Rg/trace ratios <0.8). Local geometry front uses paired-protein bootstrap local-MSE differences, reported both against initialization and incrementally between adjacent checkpoints.

Implemented grids: 24 paired field/clean-geometry timesteps; 18 marginal timesteps spanning 350–498 plus5/50/250; Gaussian499→400 prefixes at eight endpoints for50% and75%. Reference descriptors are computed once and reused across checkpoints, preventing checkpoint-specific preprocessing. Tests: 32 focused high-noise/marginal/reverse checks passed. Inference started with one CPU thread, FP32/no AMP/compile, a bounded2GiB GPU-reserved ceiling and all output/temp/cache on D:. Active E014 PID2413455 and corrected E012 PID2627419 continue unchanged; contention is allowed. Analysis and interpretation are pending at this implementation boundary. No scientific gate or final E014 judgment is authorized before the required two complete TRAIN epochs.
