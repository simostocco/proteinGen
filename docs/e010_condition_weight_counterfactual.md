# E010 completed condition-weight counterfactuals

Recorded 2026-10-05 against audited baseline
`058ac3e747d9dc28d1a3e9d0f449627de24cf2be`. These are frozen-state diagnostics,
not training experiments or authorization to deploy a structure generator.

## Fixed protocol and reproducibility

The only scientific variable was the condition weighting used to construct the
training gradient. Conditions 50/250/450 are frozen E007 corruption timesteps,
not protein lengths. The refiner receives coordinates and masks, not a timestep.

- Checkpoint: `reports/experiments/E010_global_equivariant_expressivity/phase4b_real_denoiser_v1/phase4b_training_v1.final/latest.pt`.
- SHA256: `f5211cbc1be5092175ce15b9761a4efba761d242310287cd6b1e04df6a6744ef`.
- Training panel: `1bf9_A` (41), `1chp_D` (103), `1czz_A` (187), `1fhf_A` (304),
  `1k6m_A` (432), each at all three conditions.
- Development panel: `2elx_A`, `2gmg_A`, `3pdd_A`, `1og6_C`, `1w26_A`, each at
  all three conditions; never used to construct update directions.
- Preserved sample-manifest SHA256:
  `91d3d6b9295329517c5d5fe411a31da00c9458b2dac8570fb50827756229e6ba`.
- Exact cached input/target/noise records and hashes reused; CPU float32, two
  threads, seed 41046, deterministic evaluation mode.

The unchanged per-protein objective is masked Cartesian component MSE plus
`1e-5` masked residual-component MSE, in Å², without alignment. The diagnostic
local objectives are endpoint-masked distance-error MSE at offsets 1/2/3 and
their arithmetic mean. Local evaluation retains equal protein/condition weights
for every candidate; it is not reweighted to favor a candidate.

For each candidate, reconstruct virtual AdamW step 1093 from the same saved
step-1092 moments: LR 0.0003, betas (0.9, 0.999), epsilon 1e-8, weight decay 0,
global gradient clip norm 5. No optimizer step or persistent moment update occurs.
Cartesian directional derivatives always evaluate the original equal-condition
objective. Positive local derivatives mean harm; negative derivatives mean
improvement. Derivative units are Å² per displacement scale alpha.

## CONTROL and CF1 — W2, partial improvement

CONTROL weights are `(1/3, 1/3, 1/3)`; CF1 weights are `(5/12, 5/12, 1/6)`.
CF1 retains 89.00% of Cartesian descent, reduces mean-local predicted harm
37.62%, reduces harm per Cartesian gain 29.91%, and reduces development local
harm 33.61%. Aggregate local harm and all three aggregate offset derivatives
remain positive. Condition-50 training mean-local harm falls from +0.024932
to +0.003151 but is not eliminated. Alpha=0.01 finite displacement confirms the
partial improvement. Classification: **W2 — partial improvement**.

## CF2 — X4 under the pre-registered retention gate

CF2 weights are `(11/24, 11/24, 1/12)`; exact fractions sum to one. This is the
second and final new weighting. CONTROL and CF1 reproduced their earlier
direction statistics within established tolerances, and their frozen/finite
per-example metrics reproduced exactly, before CF2 interpretation.

| Metric | CONTROL | CF1 | CF2 |
|---|---:|---:|---:|
| Pre-clipping gradient norm | 8.334528 | 5.521452 | 4.481148 |
| Clip coefficient | 0.599914 | 0.905559 | 1.000000 |
| Post-clipping gradient norm | 4.999999 | 4.999999 | 4.481148 |
| Adam displacement norm | 0.234484 | 0.228349 | 0.222103 |
| Cartesian derivative | −0.260010 | −0.231406 | −0.172280 |
| Cartesian retention | 100% | 89.00% | **66.26%** |
| Training i+1 derivative | +0.095763 | +0.056232 | −0.009151 |
| Training i+2 derivative | +0.138581 | +0.083560 | −0.017002 |
| Training i+3 derivative | +0.173878 | +0.114852 | −0.003238 |
| Training mean-local derivative | +0.136074 | +0.084881 | −0.009797 |
| Local harm / Cartesian gain | 0.523341 | 0.366808 | Non-harmful |
| Development Cartesian derivative | −0.175045 | −0.150457 | −0.124363 |
| Development mean-local derivative | +0.137008 | +0.090960 | +0.001568 |
| Condition-50 training mean-local | +0.024932 | +0.003151 | −0.026110 |
| Condition-50 development mean-local | +0.057538 | +0.040981 | +0.010211 |

CF2 improves every aggregate training offset but retains only 66.26% of
Cartesian descent, below the pre-registered 70% gate. Development mean-local
MSE harm is reduced 98.28% versus CF1 but remains positive; development i+1/i+3
and all condition-50 local offsets remain harmful. Training condition 250 also
retains mean-local harm (+0.020108).

CF2 signed condition-gradient projection fractions are
`w_c <g_c, g_mix> / ||g_mix||²`:

| Condition | CF2 projection contribution |
|---|---:|
| 50 | +3.70% |
| 250 | 71.13% |
| 450 | 25.17% |

Condition 450 no longer dominates; condition 250 becomes dominant. These are
signed vector projection contributions, not fractions of loss or raw norms.

## Finite validation and module interpretation

Independent model copies received exactly `theta0 + 0.01 * Delta_theta`.
No cumulative steps, alpha 0.1/1 evaluations, retraining or noise regeneration
were performed in these weight counterfactuals.

| Panel | CF2 Cartesian Δ, Å² | CF2 local MSE Δ, Å² | Mean-local RMSE Δ, Å |
|---|---:|---:|---:|
| Training | −0.001719451 | −0.000098165 | −0.000036847 |
| Development | −0.001245213 | +0.000014856 | −0.000003786 |

Observed/predicted ratios are 0.99805 and 1.00199 for training Cartesian/local
MSE; 1.00128 and 0.94766 for development. Development MSE harm and mean-RMSE
improvement coexist because averaging square roots weights examples differently
from averaging MSEs. The primary decision metric remains MSE. Evaluator local
RMSE agreement is within 4.41e-7 Å.

`blocks.0` remains the largest harmful Adam contributor (+0.130086).
`blocks.3` becomes more protective (−0.147252). `blocks.4` changes its Adam
contribution to improvement (−0.008744), while retaining strongly negative raw
interaction (cosine −0.974781). Aggregate benefit combines a block-level sign
change and stronger protective cancellation; underlying raw objective conflict
persists. CF2 is unclipped, unlike CONTROL/CF1. Sign reversals cannot be explained
by uniform update shrinkage alone, but clipping is part of the actual experiment.

## Pre-registered gates and decision

| Gate | Result |
|---|---|
| A: Cartesian descent | PASS |
| B: Cartesian retention >=70% | **FAIL: 66.26%** |
| C: training aggregate local harm eliminated | PASS training; FAIL preferred development confirmation |
| D: all training local offsets <=0 | PASS |
| E: condition-50 local harm eliminated | PASS training; FAIL preferred development confirmation |
| F: development tradeoff improves versus CF1 | PASS |

Classification: **X4 — excessive Cartesian sacrifice under the pre-registered
retention gate**. This classification does not deny the genuine geometry benefit
or attribute it solely to step shrinkage.

**Condition weighting materially changes the tradeoff but is not a sufficiently
efficient and transferable solution. Weight-only tuning is now closed. CF2
should not be trained.** No further weights, including `(23/48, 23/48, 1/24)`,
are authorized by this record.

The panels have one identity per length stratum and cannot establish a length
law. Development is independent of the five diagnostic training identities,
not a proven homology-independent end-to-end test. No actual retraining outcome
or final structure-generator validity is established.

## Next question and preserved evidence

The next intervention is
`L_total = L_cart + lambda * mean(L_local_i+1, L_local_i+2, L_local_i+3)`, with
historical equal condition weights `(1/3, 1/3, 1/3)` restored. First pre-register
coefficient choices for a frozen-state diagnostic; do not retrain yet. Use the
validated endpoint-masked distance errors rather than an unrestricted O(N²)
loss. Chirality-aware representation remains a separate future architectural
track; reweighting does not remove E010's reflection-inclusive symmetry.

Raw scripts, tensor-level results, sample/noise manifests, checkpoint hashes and
checksums remain in external audit storage under the dated
`condition_weight_counterfactual/` and `condition_weight_counterfactual_v2/`
directories. They are intentionally not committed or runtime dependencies.
This note records their established results, not a standalone executable
reproduction bundle. See the [Phase 4B closeout](e010_phase4b_forensic_closeout.md)
and [baseline consolidation](recovery/baseline_consolidation_20261004.md).
