# E015 integrated multiscale parity self-conditioned coordinate diffusion

Experiment: `E015_integrated_multiscale_parity_selfconditioned_v1`. All improvements are integrated, so causal attribution among individual components is limited. Scientific success requires joint denoising, geometry, chirality and reverse-transport improvement; training loss alone is insufficient.

The latest clean E014 tip is `053ec32dd256349a6ff8504a0f796167f5c1c727`. E015 starts from its clean evidence descendant `7e0ab0804deb06b8344d51659a7acb079dee3123`, including the frozen comparison records. E014, E007 and E012 branches, source and historical artifacts are unchanged.

Worktree: `/mnt/d/Simone/proteinGen/.e015_worktree`.
Branch: `e015_integrated_multiscale_parity_selfconditioned_v1`.
Artifact root: `/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1`.

## Scientific baseline and inference information

The model remains p(X|N). Inference receives only current noisy coordinates, time/logSNR, requested length, chain order/connectivity, valid masks, current geometry and an optional previous model-generated detached clean estimate. Its API has no clean target, sequence, target RMSD, teacher action, optimization solver or target-derived gradient input. Known clean coordinates enter training corruption and losses only.

Coordinates are centered and divided by **12.22820347644835 Angstrom**. The imported historical 500-step VP cosine schedule and coordinate-v identities are unchanged: Xt=alpha X0+sigma epsilon, v*=alpha epsilon-sigma X0, X0hat=alpha Xt-sigma vhat. Gaussian noise is centered over valid residues.

TRAIN: **230,132 identities**, population hash `d3f2ee345d0e4e9a27d0136aad1eca9dfe87821a3a82d90aa4c3194bd928a3f9`. Development: frozen 500 identities, hash `3a36bccfee801b618674594762ed2e6e394220ca21db94260ccb5becf7328dcf`. Gaussian subset: frozen 100 identities, hash `c3e94b415bd7a13269bbb7f3a26372764cbba33b91b16589c7f6906a563020a0`. Runtime verifies normalization, manifests, coordinate cache and population identity hashes. Prospective manifest checksum verification does not inspect its rows or outcomes. The 2,000-identity holdout remains unopened for scientific evaluation.

## Capacity and hierarchy

Selected LARGE: **33,128,139 parameters**. Scalar/polar/pseudoscalar/axial widths are **384/96/48/48** at all scales. Fine/medium/coarse blocks: **8/4/4**. Local hidden width: **384**. A validation-blind ladder compares LARGE MLP expansion ratios 1 and 2; the smallest eligible candidate is selected by parameter count and worst-path feasibility. Exact counts by module are recorded in `capacity_ladder.json`. No validation performance selects capacity or physical batches.

At N=500, valid-node counts are 500 → 125 → 32. H shapes are [500,384], [125,384], [32,384]; V shapes append [96,3], P append [48], A append [48,3]. Padding never enters per-protein graph construction or pooling.

Fine edges are the deterministically sorted, deduplicated union of chain offsets ±1/±2/±3, 24 current-coordinate neighbors and 16 self-conditioned neighbors. Provenance flags remain distinct. Scalar edge features include two 16-channel distance RBFs, signed normalized chain separation, offset flags and availability. Vector messages receive both current and generated-context relative directions. There are no raw Cartesian components in scalar MLPs.

For contiguous group G, each state and coordinate is pooled as sum(valid child values)/number(valid children), including partial final groups. This is normalized masked scalar pooling and an equivariant scalar-weighted vector mean. Four fine nodes form a medium node; four medium nodes form a coarse node. Medium uses segment-neighbor/kNN messages and scalar RoPE attention on its edges. Coarse uses dense distance-biased scalar RoPE attention and typed vector/parity messages. RoPE chain positions reset independently for every protein in a packed physical batch.

Top-down child state is child + sigmoid(W[Hchild,Hparent]) × typed_projection(parent). Parent index is floor(child_index/4); scalar gates are even, P projections have no bias and vector projections act only on channel indices. Residual skips retain fine and medium state. Packed message passing uses disjoint graph indices; a test proves agreement with independent per-protein execution.

Message work is O(N(k_current+k_self+6)) per fine block and O((N/16)^2) per coarse block. Neighbor selection uses dense pairwise distances and stable sorting, so selection costs O(N²) memory/O(N² log N) work. Clash loss and descriptor construction also use pairwise geometry; the model is not advertised as wholly linear-complexity.

## Approved symmetry contract

The **trunk is O(3)-typed**, while the **final generative field is SE(3)-equivariant only**. Reflections are intentionally not treated as a protein symmetry, enabling preference for biological handedness from symmetric Gaussian noise.

For orthogonal Q: H'=H, V'=QV, P'=det(Q)P, A'=det(Q)QA. Trunk scalar inputs are even invariants; odd channels use bias-free linear maps and odd activations. Polar×polar cross products produce A; polar·axial produces P; P×A produces polar messages. Invariant norms, scalar gates, pooling and typed broadcast preserve these identities.

The output is vpolar + gchiral(vchiral+vlocal), then valid-residue centering. vpolar consists of H-weighted V and P-weighted A. vchiral consists of H-weighted A and P-weighted V. The oriented right-handed local frame is part of the chiral branch. Disabling the chiral head removes both chiral and local terms and recovers full reflection equivariance. Chiral projections start at nonzero Gaussian scale 1e-5, so gradient flow is not permanently blocked.

Tests cover O(3) trunk identities, proper rotations/translation for raw/EMA/context paths, enabled reflection gap, disabled reflection symmetry, nonzero chiral gradients, packed-batch independence and 500→125→32 pooling. Achiral/degenerate inputs are not required to exhibit a reflection gap.

## TRAIN-only calibration and SNR sampling

A deterministic SHA256-ranked TRAIN panel contains 20 identities per length stratum. Fixed grid: 0,5,20,50,100,150,200,250,300,350,400,450,490,498. Thresholds are mean local-distance correlation >0.25, centered circular torsion correlation >0.15, chirality sign agreement >0.60. No threshold is relaxed. Distance aggregates per-protein Pearson correlations over offsets 1/2/3. Gates use the lowest qualifying sampled logSNR, transition width 1.0.

Frozen midpoints: distance/clash **5.124029636383057**, torsion/chirality **3.51250958442688**. The distance gate activates at higher SNR than torsion in this panel; this measured result is retained. Clash shares the distance midpoint and the historical 3 Angstrom, separation>3 definition. Trunk/head parity gate uses the chirality midpoint, frame reliability and assessability: sigmoid((logSNR-midpoint)/width+2(reliability-1)) × reliability. It is smooth and near zero at extreme noise, with trainable output projections rather than a hard terminal cutoff.

Eight deterministic logSNR-rank bins cover all 500 times. Production draws one bin uniformly per physical microbatch, then a time uniformly within it for each protein. Drawn and actual post-rollout time histograms are both persisted. Sharing a bin permits direct gradient-norm telemetry by SNR regime. This choice is frozen before validation.

V weights, low-SNR bin first: **[1.61609745,1.33995152,0.94299793,0.79083985,0.82778746,0.77518260,0.82616740,0.88097572]**. They derive from TRAIN-only FP32 gradient norms by inverse square root. Normalization is solved under clipping so all weights stay in [0.5,2.0] and their bin-uniform mean equals one.

Structural gradient targets are 15% distance, 10% torsion, 10% chirality, 5% clash relative to weighted v gradients in active regions. Frozen coefficients: **distance 17.083404031072213; torsion 0.9274404147247816; chirality 1.3079999663494615; clash 100.0**. Clash reached the coefficient cap, so its achieved gradient fraction may fall below 5%. The first two short examples had zero clash gradient; the final fixed calibration samples two identities from each of all five TRAIN strata, retaining zero-gradient examples in averages. This was a calibration identifiability correction, with no validation inspection. Raw evidence, panel hashes and exact weights are in `calibration/`.

## Self-conditioning, objectives and optimization

With probability 0.5, a no-gradient preliminary pass of the same raw model reconstructs a detached estimate. Otherwise zeros plus an explicit absence mask are used. At Gaussian inference, the first step has no context; subsequent steps carry the previous generated clean estimate. Paired one-step evaluation uses one model-only preliminary pass, documented as an extra inference cost relative to E014.

Structural losses apply to reconstructed X0hat, never directly to Xt. Distances use equal-protein mean offset-1/2/3 squared errors; torsion uses masked sine/cosine MSE; chirality uses Huber normalized triple-product error plus 0.1×softplus(-qhat qtarget/0.1); clashes penalize only nonneighbor distances below the frozen threshold. Each has its own fixed sigmoid SNR gate. Optional global log-scale regularization is implemented behind a disabled flag; actual E014 noisy-state scale collapse was not observed.

AdamW: LR **0.0002**, weight decay **0.01**, no scheduler, clipping **1.0** after gradient combination. Microbatches accumulate both objective gradients with equal-protein weighting across the entire optimizer update. If their dot product is negative, the structural gradient is projected against v in FP32, then combined. Missing gradients use the available objective. No silent summed-loss fallback exists; only an explicit config disable changes the combination. Distributed execution is refused until global reduction/exposure semantics are certified, preventing an unsafe rank-local extension.

After **115,066 exposures**, 10% of physical batches use 1–3 deterministic EMA reverse steps from a forward-noised TRAIN state. Implied epsilon and the consistent v target are constructed using the known TRAIN clean target; no target is supplied to the EMA inference call. Forward-state and induced-state losses, steps and SNR are separately logged. Rollout training is enabled but deliberately inactive before its half-epoch warmup.

EMA is FP32 with decay **0.9999**, initialized from raw weights and updated after every optimizer step. It serves validation, bounded prefixes and model-induced states. Both raw and EMA weights are checkpointed and compared at development boundaries.

## Precision, memory and batch semantics

Master parameters, AdamW states, EMA, geometry accumulation and PCGrad projections remain FP32. BF16 is selected when supported; FP16 with GradScaler is the fallback. Fine/medium/coarse blocks use non-reentrant activation checkpointing. Dedicated self-conditioning, rollout, timestep and noise RNGs are serialized. Deterministic algorithms and the cuBLAS workspace contract are enabled before certification.

Each isolated batch probe includes real AdamW/EMA state, self-conditioning preliminary inference, all three rollout steps, geometry, dual gradient accumulation/projection, clipping and optimizer update. A second, independent model-generated context stresses less-overlapping current/context neighbor graphs; this is a memory-only probe and is never used by production training. Probes expand and binary-search integer batch sizes, then repeat the selected size three times. OOM/failing child processes are isolated and preserved as logs.

Selection requires peak reserved ≤90% detected device memory and ≥**768 MiB** free at estimated peak; hard production reserve cap is 92%. The 85–90% reserved target is subordinate to safety. Display/other allocations and CUDA context memory can make that target infeasible on the detected 8,151 MiB device. No artificial allocation is used to inflate occupancy. Final per-bucket measurements and selected sizes are in `batch_autotuning.json`.

Production accumulates full safe physical microbatches until at least **4,096 valid residues** per update, splitting at frozen exposure boundaries. This is a minimum target, not an exact batch size: a large physical microbatch or the final accumulation can overshoot. Actual protein/residue denominators and batch shapes are logged; equal-protein objective weighting uses the actual update population. No validation-driven adjustment of this budget or learning rate is allowed.

## Durability, storage and coexistence

Every epoch is a deterministic shuffled complete pass over the unchanged TRAIN population. Every identity is seen exactly once. Checkpoints contain schedule, cursor, per-identity exposure counts, raw/EMA/optimizer/scaler, all RNGs, calibration/config/source/normalization/population hashes. Resume reconstructs the schedule and proves exposure equality. Writes use D-resident partial files, fsync, full reload/recursive tensor verification, SHA256 and atomic replacement. Alternating rolling slots are bounded; fraction/epoch boundaries are durable. Resuming at a saved boundary completes pending evaluation before advancing.

HEARTBEAT, STATE and COMMITTED record phase, epoch/update/exposures, identity counts, loss groups, branch/gate telemetry, PCGrad conflict/clipping/norms, timestep distributions, context/rollout fractions, throughput, GPU allocation/reservation and disk availability. SIGTERM/SIGINT only ask E015 to stop at a committed update. A global E015 GPU-worker lease and supervisor lock reject duplicates.

All cache/temp paths—including TMPDIR/TMP/TEMP, XDG, Matplotlib, Torch, HF, Numba, CUDA and Triton—resolve inside the declared D artifact root. The guard rejects parent/symlink escapes. CPU threads are at most two, loader workers zero. The E012 original reserve of **56,883,428,966 bytes** is protected, rather than its lower temporary screening floor. The provisional E015 cap is **16 GiB**, covering model/EMA/Adam checkpoints, verification partials, bounded rolling state, logs and scratch. Admission fails closed if the remaining budget would breach the reserve. E012's retained 32 GB final-sort requirement is explicitly acknowledged.

E007's last inspected state was paused by its E012 contention guard, not normally completed. E015 does not alter it. The supervisor waits for no active E007 trainer and available GPU resources, rechecks ownership between stages and stops E015 if E007 becomes active. It never stops/signals/renices E007 or E012. A paused E007 is not recorded as a successful completion.

## Gates, evaluation and launch

Mandatory gates: CPU tests/lint; N=500 full path; TRAIN-only calibration; five-bucket repeated autotuning; full-model CUDA raw/EMA/context symmetry; serialized exact model/EMA/optimizer/RNG replay; D audit; 100-update and 1,000-update TRAIN-only smokes. Smoke models are discarded; production starts from the frozen initialization and exposure zero. Failed gates stop the supervisor without fallback or duplicate production.

Production evaluates initialization, quarter epochs, epoch1/2 and later reached boundaries. Raw/EMA core times are 5/50/250/400/490; EMA epoch diagnostics include dense high-noise field, independent 34-descriptor marginal references and Gaussian prefixes ending at 490/480/470/460/450/440/420/400/350. Direct-X0 collapse remains distinct from actual noisy-state Rg/covariance collapse. No path runs prefixes below350.

Formal comparisons use E014 epoch2 (SHA256 `61a3acfe1024c9a6adcb21bfd864b3d0980e978d2fb90cbb02d9f1d095b9d902`) and contextual newest (SHA256 `935337d559f87b079a0bc9bb8bce6ecf90e971840d59c4b6163f036697c0f4e7`). Paired protein bootstrap uses 5,000 stratified draws for v/local/torsion/chirality and Gaussian energy at400/350. Marginal diagnostics use the existing reference-only standardization and bootstrap, with equal protein weighting and the reference-v-reference floor. Length-stratified raw results remain available. Hundreds of secondary metrics are descriptive, not independent uncorrected discovery claims.

Two complete production epochs are mandatory. Initial maximum is four. After epoch2, continuation requires >1% joint v/local improvement over the preceding epoch, no >5% aggregate structural regression and no noisy-state collapse. A full-sampler recommendation additionally requires material local improvement, improved below-chance mid-noise chirality, improved torsion/transport, stable prefixes and no severe length regression. The recommendation never launches t=0 sampling; owner authorization remains necessary. Prospective validation is disabled.

Exact numerical replay is separate from scientific effect sizes. E014's historical descriptor envelope **8.962951e-6** remains uncertified against **5e-6**. E015 retains that tolerance and records its own envelope, without relaxing the old finding.

Persistent command (from the E015 worktree):

```bash
PYTHONDONTWRITEBYTECODE=1 /home/simostocco/miniforge3/envs/proteingen/bin/python -B scripts/supervise_e015_training.py
```

After an interrupted existing production run, use the same command with `--resume`. The supervisor automatically completes mandatory remaining stages before production. Actual PID, phase, batch measurements, first durable checkpoint and launch commits are recorded in D-resident `LAUNCH.json`, `SUPERVISOR.json`, `COMMITTED.json`, `HEARTBEAT.json` and the launch record below. A prepared config or running supervisor alone is not described as production training.
