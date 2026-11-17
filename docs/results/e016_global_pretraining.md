# Final J1 global pretraining report

**VERIFIED: all 30,000 additional real-PDB optimizer updates executed. The new checkpoint improved held-out AA/local likelihood modestly, but failed the frozen final generation safeguard and was NOT promoted. The original J1-FULL remains the selected fallback.**

Training implementation commit 2695487; branch e016-j1-global-pretraining-20261017-v1. Source and all scientific inputs/outputs are sealed separately by PROVENANCE_SHA256.json. Historical datasets, priors and checkpoints remain unchanged.

## What ran

One existing J1, with 12,351,746 trainable parameters: width 352,8 blocks,8 heads(head dimension 44),FFN 4, dropout 0, dense full-prefix geometric attention with sequence RoPE. Chemistry stays disabled; the approved 14-feature registry and action system are unchanged. FP32 weights, BF16 neural matrix operations, FP32 geometry-head projection, FP64 coordinate invariants/density/builder operations.

AdamW moments restored from the 30000-total-update parent. New 250-update warmup, peak LR 5e-5, cosine decay to 5e-6, weight decay 0.01, clipping 1. Effective batch 64, live 16, 4 accumulation passes. Ordinary decisions: 56 complete joint plus 8 partial AA-only, identity-first then observation/position sampling. Partial geometric inputs retain authoritative masks and positions; available partial geometry is intentionally not a training target in this AA-only branch. Missing coordinates were not fabricated or reclassified.

Base loss remains AA CE plus per-decision mean of valid conditional distance/angle/signed-torsion NLL. The periodic auxiliary objective is exactly
L(u)=L_base+I[u mod 16=0]*ramp(u)*(0.13868601683008452*L_Rg+0.04997356779049633*L_clash).
Ramp reaches 1 at 1000 updates. Effective average post-ramp coefficients per ordinary update are 0.0086678761 and 0.0031233480. No compensating 16x importance factor was used.

Rg is Huber(delta 0.5) of the log generated/native Rg ratio at identical prefix endpoints/count. Repulsion is squared penetration below 2 Å for sequence separation>=3 and at least one generated endpoint, summed per generated residue. TRAIN 512-identity calibration had 9033260 eligible nonlocal pairs, none below 2 Å. Native future coordinates were loss targets only. Rollouts introduced true AAs one at a time after each AA prediction, sampled actual transformed-Gaussian geometry through reparameterization, and fed attached generated coordinates back into subsequent encoders. They are not fully free-running joint samples.

Exactly 1920000 ordinary decisions: 1,680,000 complete, 240,000 partial AA-only, including 15968 targets at missing-CA positions. There were 1875 global updates and 59367 differentiable generated residues (horizons 17–32); no full 500-step differentiable rollout was trained. Two separate, discarded engineering recovery proofs executed 34 optimizer updates. Main lineage now totals 60000 updates.

## Sequence evidence

Both frozen evaluation panels have 2947 distinct identities and 23576 fixed decisions, 8 per identity using the same first-sorted-observation/rounded-linspace protocol and matched length strata. This includes BOS once per identity; it is not whole-chain residue-weighted perplexity. TRAIN and VALIDATION identities do not overlap. TEST was not evaluated. All fits use TRAIN only; smoothing is 0.5 per AA/transition cell, with BOS represented only as a Markov context. Exact probabilities, both identity-first/residue-weighted variants and full-PDB statistical controls are in SEQUENCE_BASELINES.json.

| Predictor | TRAIN PPL | VALIDATION PPL |
|---|---:|---:|
| Uniform | 20.000000 | 20.000000 |
| TRAIN identity-first unigram | 18.670699 | 18.905275 |
| TRAIN identity-first Markov | 17.560356 | 17.135997 |
| Original J1-FULL | 10.906462 | 15.865763 |
| Trained J1 +30k | 9.203208 | 15.733081 |


Latest VALIDATION AA NLL2.755765591 versus 2.764163531 originally: delta−0.008397941 nats, paired identity-bootstrap 95%interval[-0.012649295158226338, -0.00393298425351013]. This is about 0.84%lower perplexity. Intervals are conditional on validation-based checkpoint selection and do not remove family dependence or establish universal homology independence. The accepted practical split still has its documented residual similarities/context overlap.

TRAIN–VALIDATION NLL gap widened 0.374808→0.536213. Most new fitting occurred on TRAIN; this is not evidence that the generalization gap was solved. VALIDATION top 1 accuracy 15.838%→16.182%; top 3 accuracy 33.810%→34.467%. Bits/residue are 3.975729 on VALIDATION and 3.202137 on TRAIN. These are native-prefix-geometry-conditioned AA metrics, not masked-language-model perplexities or unconditional sequence-only scores.

## Local and global geometry

VALIDATION geometry NLL per decision improved−0.213709→−0.253267. Distance/angle/torsion NLL changed from[-1.917618,-0.173647,1.358548]to[-1.951040,-0.198334,1.281031], with 20629 valid targets per component. Density units remain A/radian; negative continuous NLL is legitimate.

The actual trained checkpoint produced 160/160 finite complete outputs,32 independent seeds at each length, all 32 sequences unique within every length. Baseline samples were reused only for the identical original checkpoint/seed/protocol. The actual last-trained policy is reported even though it was rejected.

| Length | Original Rg A | Trained Rg A | Original clashes/100 res | Trained clashes/100 res |
|---:|---:|---:|---:|---:|
| 64 | 18.521 | 17.775 | 1.9531 | 2.7832 |
| 128 | 25.976 | 25.201 | 2.3438 | 1.9531 |
| 256 | 37.315 | 37.241 | 2.9541 | 3.1128 |
| 384 | 49.931 | 47.699 | 2.6204 | 3.1087 |
| 500 | 54.858 | 52.484 | 3.2063 | 3.5875 |


All five mean Rg point estimates decreased slightly, but every paired 95%interval for native-normalized Rg-error change includes 0. There is no demonstrated broad compactness improvement; long chains remain substantially more extended than length-matched native references. There is no evidence of gross chain collapse.

Clash means increased at 4 of 5 lengths and decreased at 128; all paired clash-change intervals also include 0. At64 residues,2.783203 clashes/100 res exceeds the frozen point-estimate tolerance 1.25*1.953125+0.25=2.691406. The paired clash-change interval[-0.048828,1.806641]is uncertain, but no post hoc tolerance relaxation or reseeding was used. The candidate failed this declared safeguard; other final checks passed. More compactness alone was not accepted as success.

Long-contact fractions rose at 64/128/256/384 and fell slightly at 500. At500,the trained fraction 0.003107 remains below the native TRAIN/VALIDATION reference distributions (about 0.0062). CA spacing, virtual-angle/circular-torsion summaries, degeneracy, composition and normalized shape diversity remain in NATIVE_VS_GENERATED_GLOBAL_GEOMETRY.json. These are CA proxies, not all-atom chemistry, experimental folding, free energy or function.

## Verification and interpretation

35 distinct focused CPU tests passed:29 mask/geometry/controller tests plus 3 assessment and 3 recovery-launcher tests. One initial pytest descriptor-capture failure occurred before collection on D:; the preserved retry with capture disabled passed. Tests cover native reconstruction, rotation/translation invariance, masked/future independence, zero-auxiliary base parity, finite-difference builder gradients and eager/recomputed parity.

Actual full-model GPU checks passed through rollout horizon 128. Global gradients reached geometry heads and Transformer weights. BF16/FP32 relative gradient error was 0.0052503 at horizon 4, max coordinate difference 0.030157A; eager/recomputed gradients matched exactly. Real interrupted/resumed global-update comparison had model error 0, loss error 0, optimizer maximum difference 7.28e-12, with identical sampler/scheduler/RNG/rollout ordering.

A separate 10-prefix TRAIN diagnostic found local geometry/distance gradient L2 about 239–246 versus AA9–11 before clipping; this small, explicitly frozen panel is diagnostic, not a population estimate or a reason to silently change the base objective. Sparse short rollouts and native-prefix/true-AA conditioning remain major limits on full free-generation credit assignment. Raw first/last 100 global TRAIN losses are not matched panels and do not demonstrate a monotonic Rg-loss reduction.

## Resources

Main elapsed 15744.21 seconds(4.373 hours), including milestone evaluation/checkpoints; timed updates 13478.50 seconds, including 4804.13 global-loss seconds. Throughput was about 142.45 ordinary decisions/s over timed updates and 121.95/s over the complete main run. Peak allocated/reserved CUDA memory 3.473/3.607 GiB; host RSS high-water 6.821 GiB; sampled device-wide GPU peak 5345 MiB. No swap, OOM, nonfinite optimizer step or disk-reserve breach was observed. Source/runtime/environment hashes and measured storage are in RESOURCE_REPORT.json and PROVENANCE_SHA256.json.

## Direct answers to the 15 scientific questions

1. Uniform perplexity: exactly 20.
2. TRAIN-derived identity-first unigram: TRAIN 18.670699, VALIDATION 18.905275; exact probabilities/smoothing and residue-weighted alternatives are preserved.
3. First-order Markov: TRAIN 17.560356, VALIDATION 17.135997.
4. Starting J1: TRAIN 10.906462, VALIDATION 15.865763.
5. Gap: substantial 0.374808 nats initially; it widened to 0.536213.
6. Best and final trained checkpoint by AA NLL is +30,000: TRAIN 9.203208, VALIDATION 15.733081. The quality-selected fallback remains the original 10.906462/15.865763 model.
7. Latest AA gains beyond unigram/Markov are 0.183675/0.085416 nats on VALIDATION; incremental gain beyond the original J1 is 0.008398 nats.
8. YES: nonzero verified pathwise gradients reach generated CA, geometry heads and upstream Transformer parameters.
9. NOT ESTABLISHED: Rg point estimates improved modestly without collapse, but length-specific uncertainty includes no change and chains remain too extended.
10. NO consistent reduction: clashes rose at 4 of 5 lengths; long contacts improved at 4 of 5 but 500 remains deficient.
11. YES for native-prefix likelihood: all three component NLL point estimates improved; free-running global quality did not become validated.
12. NO defensible simultaneous AA/global improvement was promoted: likelihood improved, final generation safeguard failed.
13. 30,000 additional main updates, 1,920,000 ordinary decisions, 59,367 rollout residues; 34 separate engineering updates.
14. Retained trained checkpoint: /mnt/d/Simone/proteinGen_data/e016_j1_global_pretraining_20261017_v1/runs/J1/checkpoint_30000.pt, SHA256 c37d2bd7eb9136f12bc4d9ea4128403fc72a6a323937bccfbd68a582c1fd78b1. Selected fallback: /mnt/d/Simone/proteinGen_data/e016_partial_supervision_20261016_v1/runs/J1-FULL/checkpoint_20000.pt, SHA256 9e9215d3720ffc73edb0c3e382db25332f91cb26b0c8d38d01668e383b1e3810.
15. Remaining limits: growing generalization gap, short/true-AA-conditioned rollout mismatch, unresolved long-range organization/clashes, sparse auxiliary strength, CA-only proxies, practical split limitations, and no calibrated oracle or experimental validation. No RL readiness or folding success is established.

Data sampling and loss changed together; the observed likelihood gain cannot be attributed exclusively to Rg. The full authorized experiment is complete; scientific improvement across both objectives was not achieved. No TEST, J2/S1/S2 training, SAC, AlphaFold or E015 work occurred.

Figures: [TRAIN/VALIDATION learning curves](figures/TRAIN_VALIDATION_PERPLEXITY.png), [global geometry point estimates](figures/GLOBAL_GEOMETRY_COMPARISON.png).
