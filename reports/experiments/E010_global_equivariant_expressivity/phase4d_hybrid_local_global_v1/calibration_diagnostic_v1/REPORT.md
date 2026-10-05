# Phase 4D frozen local-branch calibration v1

CPU diagnostic completed 2026-10-05 from preparation commit `7745c3b45c856251af16c83d22995d479329cfd0` on `e010-phase4d-hybrid-local-global`. **No training, optimizer step, checkpoint continuation, development tuning or CUDA.**

A small safe aggregate direction exists, but its improvement is modest. The registered alpha=1e−4 passes all eight diagnostic criteria with 0.62% mean-local improvement. At alpha=1e−3 the gain is 6.43%, but chirality worsens. At alpha=1e−2, i+1, Cartesian quality, chirality and displacement bounds fail. These results do not establish substantial safe improvement or authorize the 500-update run.

## Frozen protocol

`protocol.json` was written before executing the sweep. It pins A=`L_local`, B=`L_local+L_guard`, C=`L_local+L_guard+0.01 L_disp`; guard beta=1; delta_cart={0,0.005,0.01}; alpha={0,1e−4,1e−3,1e−2}. No additional coefficients or scales were evaluated. Direction d is the raw, unnormalized negative total local-parameter gradient at exact zero initialization. No optimizer state is used.

All nine initial variant/tolerance directions were verified bitwise equal: guard and displacement derivatives are zero, including the PyTorch ReLU derivative at the delta=0 boundary. Therefore all variants visit the same four physical points. This is an initialization-direction diagnostic, not evidence that variants remain equivalent after a later update. Each alpha uses an independent local parameter copy from theta0; displacement is not cumulative.

The guard is rectified after the equal-example Cartesian mean over the fixed 60 examples. Streaming gradients use exact equal-example 1/60 weights; guard gradients are formed from the aggregate activation indicator, not independently rectified microbatch guards. Tests compare this calculation to joint-batch autograd.

The unchanged panel, source checkpoint and model-state pins are checked before/after. Global Pg is computed once under no_grad, then detached. The checkpoint SHA256 is `f5211cbc1be5092175ce15b9761a4efba761d242310287cd6b1e04df6a6744ef`. All global parameters remain frozen without gradients; global and original local states are unchanged. Every pinned preparation/historical/cache file is unchanged.

## Initial gradients

Every variant/tolerance: total/local norm **31.03347442**, Cartesian norm **1.84735985**, cosine **-0.84779932**. Guard/displacement norms are zero. Predicted d-direction derivatives: local **-963.07653476 Å²/alpha**, Cartesian **48.60433039 Å²/alpha**. Only the zero-initialized final head has nonzero initial gradients; input and all four blocks have zero initial gradients. Per-module component norms are recorded for every point.

## Aggregate finite results

Mean-local RMSE is the preparation convention: equal-example mean of the arithmetic mean of offset RMSEs. Offset MSE is equal-example mean. `results.json` additionally reports sqrt(mean MSE) explicitly, avoiding confusion with mean per-example RMSE. Raw Cartesian includes historical xyz-component MSE and 1e−5 source-residual regularization. Aligned RMSD uses proper rotations only. Units: MSE/objectives Å²; RMSD/displacement Å. Displacement RMS here is sqrt(equal-example mean squared vector displacement), consistent with L_disp; its maximum is over all residues. No correction gate is present.

| Alpha | Mean-local RMSE Å | Improvement | Raw Cartesian Δ Å² (%) | Aligned RMSD Δ % | Chirality inversions/assessable | Displacement RMS/max Å | Aggregate pass |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | 2.774440 | 0.0000% | +0.000000 (+0.0000%) | +0.0000% | 5493/13029 | 0.000000/0.000000 | False |
| 0.0001 | 2.757308 | 0.6175% | +0.005105 (+0.0146%) | -0.0096% | 5491/13029 | 0.027007/0.032627 | True |
| 0.001 | 2.595988 | 6.4320% | +0.072917 (+0.2079%) | -0.0051% | 5540/13029 | 0.270070/0.326274 | False |
| 0.01 | 2.417093 | 12.8800% | +2.917332 (+8.3167%) | +8.3128% | 5631/13029 | 2.700702/3.262739 | False |

| Alpha | i+1 MSE / mean RMSE | i+2 MSE / mean RMSE | i+3 MSE / mean RMSE |
| --- | --- | --- | --- |
| 0 | 3.877305 / 1.895241 | 9.757564 / 2.874420 | 16.233933 / 3.553660 |
| 0.0001 | 3.779027 / 1.870959 | 9.662810 / 2.858596 | 16.136827 / 3.542368 |
| 0.001 | 2.916705 / 1.642294 | 8.790142 / 2.710031 | 15.210147 / 3.435638 |
| 0.01 | 5.398804 / 2.274073 | 5.031884 / 2.217105 | 8.275470 / 2.760100 |

All points are finite. The Pg frames used to produce corrections remain fixed: 13,089 eligible, zero degenerate interior frames. Output-geometry frame counts also remain 13,089/0. Chirality assessability is unchanged at 13,029 windows, with zero lost/gained assessability; common-window comparisons give the same inversion differences. The 2-inversion aggregate gain at alpha=1e−4 includes condition-specific worsening, so it is not a uniform orientation improvement or a claim of chirality resolution.

## Guard activation

| delta_cart | First sampled alpha | Sampled bracket | Local improvement at first activation | Raw Cartesian cost Å² (%) | RMS displacement Å |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | 0.0001 | [0.0, 0.0001] | 0.6175% | 0.005105 (0.0146%) | 0.027007 |
| 0.005 | 0.01 | [0.001, 0.01] | 12.8800% | 2.917332 (8.3167%) | 2.700702 |
| 0.01 | 0.01 | [0.001, 0.01] | 12.8800% | 2.917332 (8.3167%) | 2.700702 |

These are first activations on the registered grid, not exact crossing locations. The direction is head-only, so its correction is linear in alpha and raw Cartesian cost is quadratic. Analytical boundary estimates (no extra model evaluations) are:

| delta_cart | Estimated boundary alpha | Estimated RMS displacement Å |
| --- | ---: | ---: |
| 0 | 0 | 0.000000 |
| 0.005 | 0.0018662678 | 0.504023 |
| 0.01 | 0.0029281386 | 0.790803 |

The boundary estimates use the initial Cartesian directional derivative and the quadratic coefficient inferred from the registered displacement. Activation is strictly above the boundary; local/chirality metrics at these unsampled boundaries are unknown. With delta=0.005 or 0.01, the guard remains inactive at alpha=1e−3 even though chirality already worsens. A Cartesian guard does not control orientation.

## Displacement regularizer

| Alpha | L_disp Å² | rho L_disp Å² | ||g_disp|| | ||rho g_disp|| / ||g_local|| |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.00000000 | 0.00000000 | 0.000000 | 0.0000% |
| 0.0001 | 0.00072938 | 0.00000729 | 0.497500 | 0.0159% |
| 0.001 | 0.07293794 | 0.00072938 | 4.977687 | 0.1563% |
| 0.01 | 7.29379327 | 0.07293793 | 52.391111 | 2.0777% |

L_disp grows quadratically along this head-only path. Its weighted gradient is tiny at the bounded scales (0.0159% and 0.1563% of local-gradient norm) and reaches only 2.08% at the overshooting point. The C penalty changes the fixed-direction derivative by approximately +0.146, +1.459 and +14.588 at the three nonzero scales. It cannot alter the initial direction, and these small penalty gradients do not demonstrate a meaningful hard bound on correction magnitude. No later trajectory was simulated and rho was not retuned.

## Conditions

| Alpha | Condition | Local improvement % | Raw Cartesian Δ % | Aligned RMSD Δ % | Inversion Δ | RMS/max displacement Å |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0.0001 | 50 | 1.0714 | -0.4795 | -0.2486 | +2 | 0.027344/0.031486 |
| 0.0001 | 250 | 0.4411 | +0.0420 | +0.0208 | +1 | 0.026659/0.032254 |
| 0.0001 | 450 | 0.5825 | +0.0196 | +0.0133 | -5 | 0.027014/0.032627 |
| 0.001 | 50 | 10.6933 | -3.5226 | -1.8455 | +18 | 0.273438/0.314861 |
| 0.001 | 250 | 4.7998 | +0.5571 | +0.2818 | +25 | 0.266594/0.322538 |
| 0.001 | 450 | 6.0878 | +0.2209 | +0.1478 | +4 | 0.270136/0.326274 |
| 0.01 | 50 | -49.6805 | +92.0087 | +39.6631 | +90 | 2.734376/3.148607 |
| 0.01 | 250 | 16.0266 | +19.2834 | +9.7437 | +15 | 2.665936/3.225378 |
| 0.01 | 450 | 31.4025 | +4.6992 | +2.9352 | +33 | 2.701361/3.262739 |

All conditions improve locally at alpha=1e−4 and 1e−3. At alpha=1e−3, chirality worsens in all three conditions. At alpha=1e−2, condition 50 locally regresses by about 49.68%, despite aggregate local improvement.

## Length strata

| Alpha | Stratum | Local improvement % | Raw Cartesian Δ % | Aligned RMSD Δ % | Inversion Δ | RMS/max displacement Å |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0.0001 | 20-64 | 0.6007 | +0.0122 | -0.0125 | +0 | 0.026018/0.031876 |
| 0.0001 | 65-128 | 0.5897 | +0.0052 | -0.0097 | +0 | 0.026981/0.032299 |
| 0.0001 | 129-256 | 0.6366 | +0.0228 | -0.0083 | +4 | 0.027159/0.032199 |
| 0.0001 | 257-384 | 0.6450 | +0.0169 | -0.0102 | -2 | 0.027522/0.032272 |
| 0.0001 | 385-500 | 0.6189 | +0.0132 | -0.0081 | -4 | 0.027330/0.032627 |
| 0.001 | 20-64 | 6.3410 | +0.2087 | -0.0077 | +3 | 0.260183/0.318760 |
| 0.001 | 65-128 | 6.1960 | +0.1402 | +0.0042 | +5 | 0.269813/0.322994 |
| 0.001 | 129-256 | 6.5812 | +0.3025 | +0.0110 | +5 | 0.271588/0.321987 |
| 0.001 | 257-384 | 6.6855 | +0.2183 | -0.0216 | +30 | 0.275218/0.322719 |
| 0.001 | 385-500 | 6.3800 | +0.1758 | -0.0080 | +4 | 0.273295/0.326274 |
| 0.01 | 20-64 | 19.7760 | +10.7345 | +10.8416 | +5 | 2.601829/3.187600 |
| 0.01 | 65-128 | 20.4168 | +10.2455 | +9.0553 | +13 | 2.698127/3.229937 |
| 0.01 | 129-256 | 8.4494 | +10.4403 | +8.7178 | +28 | 2.715884/3.219867 |
| 0.01 | 257-384 | 9.1812 | +7.1371 | +7.3231 | +86 | 2.752181/3.227191 |
| 0.01 | 385-500 | 5.5621 | +6.1093 | +6.6878 | +6 | 2.732955/3.262739 |

All five strata improve locally at both small nonzero scales. At alpha=1e−3 every stratum has increased inversion counts. At alpha=1e−2 all strata violate the 1% aligned-RMSD bound; aggregate local improvement still hides offset/condition harm. Four identities per stratum cannot establish a population length law.

## Records and validation

- `results.json`: all nine initial gradient audits; four shared physical points with per-example, condition and stratum metrics; all 36 variant/tolerance/alpha objective records and explicit point references; sampled activation brackets; protected-file/state hashes.
- `interpretation.json`: stratified changes, analytical guard estimates and regularizer directional effects.
- `validation.json`: focused CPU tests, lint and provenance integrity checks.

The runnable diagnostic refuses to overwrite its results. Reproduction requires a new versioned output location, preserving the registered protocol. Use the existing environment with CUDA_VISIBLE_DEVICES empty and two CPU threads. No environment packages were changed.

**Exactly one recommended next action:** review this diagnostic and freeze an explicit tiny-overfit acceptance contract before deciding whether to authorize its GPU execution.
