# E010 Phase 4D v3 bounded recurrent local capacity experiment

Preparation lineage: Phase 4C base `215fb5b3eb127f690b382adae1c1607979169a85`,
Phase 4D architecture `7745c3b45c856251af16c83d22995d479329cfd0`, frozen v1 diagnostic
`5eaa96c30d7b18ef712aeb5a84c412ba8db5ab97`, and objective-v2 diagnostic
`05073f4b5c9eefcc10de5d3c3440b12834036714`. V2 is committed and pushed; the remote
SHA was verified before implementation. All these scientific records stay
unchanged. The old one-shot 500-update contract remains closed.

This new user-authorized experiment asks whether recomputing local geometry
between bounded corrections improves local/Cartesian/orientation behavior, and
whether local capacity limits the result. Capacity is the sole variable across
arms. There is no held-out evaluation or automatic follow-on pilot.

## Fixed architecture and source

E010 Large stays completely frozen, 12,844,352 parameters, original Phase 4B
update-1092 checkpoint SHA256
`f5211cbc1be5092175ce15b9761a4efba761d242310287cd6b1e04df6a6744ef`.
Frozen Pg is computed on CPU under no_grad. E010 does not participate in the
training process or optimizer; the training arms consume immutable cached Pg.
All historical/cache/checkpoint input hashes are checked before and after.

| Arm | Width | Local residual blocks | Parameters |
| --- | ---: | ---: | ---: |
| S | 128 | 4 | 927,875 |
| M | 192 | 6 | 3,115,587 |
| L | 256 | 8 | 7,369,987 |

Large naturally exceeds the suggested 7.0M upper target; its requested 256/8
architecture is retained without changing features or other components. All
arms use the same original 41-channel representation, exact oriented-frame
convention/epsilon/eligibility, six sequence edges at ±1/±2/±3, LayerNorm, SiLU,
and zero-initialized three-component local head. Small has exactly the historical
local-branch parameter topology. There are no amino-acid/annotation/E011/E012
features, global attention, added hidden dimensions, or per-step parameters.

P0=Pg. Exactly K=4 applications of one shared local network produce P1..P4.
Every application recomputes frames and all features from its current Pt.
Backpropagation traverses the entire recurrence, including frame/feature geometry;
there is no detach between steps. Only Pg is detached. All five states equal Pg
exactly at initialization for every arm.

The correction is the fixed radial map
`u_b = u / sqrt(1 + ||u||² / 0.04²)`, then `delta = F @ u_b`.
Its mathematical norm is strictly less than 0.04 Å. A one-ULP inward radial
rounding safeguard handles float32 saturation rounding, without a new scientific
coefficient. Invalid frames/endpoints retain exactly zero correction. For four
steps, triangle inequalities bound per-residue displacement by 0.16 Å and any
pair-distance change by 0.32 Å. Numerical checks use <=0.160001 and <=0.320002 Å;
per-step Cartesian telemetry uses <=0.040001 Å while bounded local vectors are
strictly below the represented 0.04 value. Rotation/translation behavior and
reflection parity retain the reviewed conventions.

## Fixed supervision and panel

Only P4 is supervised:
`L = L_local(P4,X) + 16.8 L_cart(P4,X) + 2 L_chiral(P4,X)`.
No guard, displacement penalty, PCGrad, adaptive coefficient or intermediate
supervision is used. Local/Cartesian equal-example reductions preserve prior
semantics, including the historical 1e-5 source-residual stabilizer in raw Cartesian.
Chirality uses the v2 normalized signed triple product and **pooled frozen eligible
quartet mean** over exactly 13,029 Pg/target-assessable quartets. Targets and loss
masks stay detached and cannot be dropped by prediction degeneracy.

The same deterministic 20 training identities, four per length stratum, with
conditions 50/250/450 supply all 60 examples. No redraw or outcome filtering.
The immutable NPZ stores Pg, original E010 coordinate input (needed for historical
Cartesian semantics), target, mask, lengths and identity/condition/stratum metadata.
The versioned manifest pins all bytes, source model/panel provenance and exact
roundtrip parity with all 60 live CPU E010 predictions. The cache is only used
for this panel. Coordinates and checkpoints remain untracked.

## Matched optimization and schedule

The common contract is frozen in `configs/e010_phase4d_recurrent_capacity_v3.yaml`.
Each arm initializes from the same namespace and seed 41047, set immediately
before local-model construction. Different widths/depths consume different tensor
shapes, but no arm-specific seed is selected. The final head is always zero.

AdamW: LR 3e-4, betas=(0.9,0.999), epsilon=1e-8, weight decay=0,
clip global norm at 5, constant LR, float32 parameters/coordinates, no AMP or TF32.
This preserves the previously proposed Phase 4D optimizer settings, with all
previously implicit Adam defaults explicitly frozen. There is no augmentation.
Full-panel batch=60, sorted by sample ID then condition 50/250/450, repeats every
update identically for S/M/L. Activation checkpointing of local blocks is enabled
for every arm and tested equivalent to eager gradients. It affects memory only.

Every feasible arm runs **500 successful optimizer updates**, without scientific
early stopping. Evaluate at 0/25/50/100/250/500. Resume restores exact local and
Adam state, successful-update cursor, RNG, historical boundary records and
clipping/gradient statistics; cache/config pins must match. Checkpoint writes are
atomic. No global optimizer/checkpoint state is continued.

## Frozen gates and selection

At update 500 each arm independently must pass all seven:

1. Mean-local RMSE improvement >=5% relative to Pg.
2. All three offset RMSEs strictly improve.
3. Every condition's mean-local RMSE strictly improves.
4. Aggregate aligned RMSD degradation <=1%; raw Cartesian is separate telemetry.
5. Continuous chirality loss non-worsening, inversion count non-increasing,
   and exactly preserved assessability, with no lost original windows.
6. All outputs finite, no collapse (historical radius-of-gyration <1e-3 Å criterion),
   original frame eligibility preserved at every recurrent state, step/cumulative
   corrections within the architectural bounds.
7. All five strata improve mean-local RMSE; every stratum's aligned RMSD
   degradation <=1%.

Mean-local is the historical equal-example arithmetic mean of offset RMSEs;
MSE and sqrt(mean MSE) are also retained explicitly in records. Assessability
comparisons report common windows and losses/gains, preventing denominator hiding.
The capacity selection rule is **smallest passing S, then M, then L**, after all
feasible arms finish. Numerical superiority cannot override this rule. Resource
infeasibility is explicitly recorded; if an available smaller arm passes it may
be selected, otherwise an incomplete capacity curve is CAP6.

CAP1/2/3 denote S/M/L smallest passing; CAP4 denotes no passes but monotonically
better mean-local gains with capacity, by >0.1 percentage point at each rung;
CAP5 denotes no material monotonic benefit; CAP6 denotes incomplete scientific
capacity evidence due to infrastructure/resources. All failure gates and other
metrics accompany this descriptive classification; monotonic mean-local gain
alone does not establish improved chirality or safe repair.

## Telemetry and execution safeguards

Every boundary reports P0..P4 with offset/mean-local RMSE, raw Cartesian,
aligned RMSD, continuous chirality loss, inversion/assessability, per-step and
cumulative RMS/max corrections, frame eligibility/degeneracy, finite/collapse
status, identity rows, condition groups and five strata. Gradient interactions
at 0 and 500 report all three objective norms, pairwise dots/cosines and module
contributions. They are descriptive, never used to choose model size.

Secondary arm records include runtime, CUDA allocated/reserved peaks, every
pre-clipping gradient norm, clipping fraction and local-gain percentage per
million parameters. Parameter efficiency does not alter smallest-passing selection.

GPU ownership was checked through NVML compute processes and Windows Python
processes; neither reported a competing E011/E012 job. Before training, each arm
ran the same batch-60 length-500 K=4 forward/backward smoke with zero optimizer
steps, finite outputs/gradients and unchanged model/optimizer state. Reserved
memory must stay below 80% of total GPU capacity. No K/length reductions, open-ended
scaling or environment changes are allowed. All three registered variants fit
the RTX 5060; detailed allocation/latency is in `cuda_preflight.json`.

Operational nonfinite/OOM/bound failures stop the affected execution and must be
classified; they are not scientific early stopping. Historical Phase 4B/4C and
v1/v2 records, main working files and E011/E012 remain untouched.

Implementation/preparation and execution result records are committed separately
on the existing branch and pushed; no merge to main. Run commands use the existing
proteingen environment, CPU prepare then CUDA preflight/train; the contract,
cache and preflight hashes are checked. Held-out training/pilot: **not authorized**.
