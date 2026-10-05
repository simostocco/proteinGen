# E010 Phase 4D hybrid global/local preparation

Prepared 2026-10-05 on `e010-phase4d-hybrid-local-global` at
`/home/simostocco/proteinGen-hybrid-local-global`, base
`215fb5b3eb127f690b382adae1c1607979169a85`. **Preparation only; training is not
launched or authorized by this record.** CPU execution, two threads, CUDA hidden.
Treat CUDA as reserved for E011; actual GPU ownership was not probed. No environment
changes. Neither `/mnt/d/Simone/proteinGen` nor the sequence-context worktree was
modified. Historical tracked Phase 4B/4C files are unchanged.

Phase 4C motivates parameter separation and an oriented local representation:
its auxiliary trajectory regressed partially after update 273, raw gradient
conflict persisted, and module behavior depended on protective/harmful
cancellation. This is a new architecture experiment, not an extension of its
fixed-lambda continuation.

## Frozen global provenance

E010 Large is the unchanged `GlobalEquivariantResidual`: width 416, six blocks,
eight heads, 64 vector channels, max length 500, radial sigma 2 Å,
**12,844,352 parameters**. Its original Phase 4B update-1092 source is
`/home/simostocco/proteinGen/reports/experiments/E010_global_equivariant_expressivity/phase4b_real_denoiser_v1/phase4b_training_v1.final/latest.pt`.
SHA256 `f5211cbc1be5092175ce15b9761a4efba761d242310287cd6b1e04df6a6744ef`.
The new CPU loader checks file bytes before trusted deserialization and strictly
loads the `model` mapping into the historical architecture. The preparation
record includes its tensor-state hash, hash convention, schema, global update,
cache manifest and journal prefix. No Phase 4C endpoint, optimizer, scheduler,
scaler or RNG state initializes this experiment. The local branch gets seed
41047; its zero head preserves exact global predictions regardless of hidden
weights.

All E010 parameters have `requires_grad=False`. E010 remains in evaluation mode,
even when the hybrid is put in training mode. Global inference runs under
`no_grad`; Pg and local features are detached. Only local parameters train.
Empty batch members bypass historical E010 (which rejects empty proteins).
All-padding/zero-length inputs are safe in the representation and hybrid forward;
the historical objective deliberately rejects empty proteins.

## Oriented frame and eligibility

Coordinates are in Å, masks are right-padded contiguous C-alpha chains. Holes
are rejected in v1. For residue i, let a = x_i − x_(i−1), b = x_(i+1) − x_i.
The columns of F are:

- e1 = b / ||b||;
- v = a − dot(a,e1)e1; e2 = v / ||v||;
- e3 = cross(e1,e2).

The central triple must be valid; both bond lengths must exceed **1e−6 Å**;
||v||/||a|| must exceed **1e−6** (dimensionless). Comparisons are strict.
Invalid frames are zero tensors. There are no fallback axes. Endpoints,
padding, missing central neighbors, coincident bonds and near-collinear turns
receive exactly zero correction. `interior` counts central valid triples;
`degenerate = interior & ~eligible`. Thus endpoints/padding are not counted as
degenerate. Features for every correction-ineligible residue are zero.
Normalization denominators are clamped only for safe arithmetic, never to make
an invalid frame eligible. Tests verify orthonormality and determinant +1 on
assessable frames, including correct zero-mask behavior.

## Features and parity

The 41 channels, in exact order, are:

| Indices | Meaning |
| --- | --- |
| 0–5 | neighbor distances at offsets −3, −2, −1, +1, +2, +3, Å |
| 6–11 | corresponding endpoint availability bits |
| 12–29 | corresponding Fᵀ(x_(i+k)−x_i), xyz in Å |
| 30 | cos angle between forward bonds a,b (internal bond-angle cosine is its negative) |
| 31–33 | i/(L−1) with denominator ≥1, i/500, (L−1−i)/500 |
| 34–35 | signed pseudo-torsion sin(phi), cos(phi) |
| 36 | normalized signed triple product |
| 37–40 | torsion, frame, bond-angle, triple-product assessability bits |

Unavailable neighbor features are zero and retain explicit availability bits.
There are no amino-acid, E007/E011 learned features or biological annotations.
The historical denoiser coordinate input is reused only as the existing E010
input; the local branch consumes geometry computed from Pg.

For consecutive bonds a,b,c = (i−1→i, i→i+1, i+1→i+2),
n1 = normalize(a×b), n2 = normalize(b×c).
`sin(phi) = dot(cross(n1,n2), normalize(b))`,
`cos(phi) = dot(n1,n2)`. Both adjacent normalized cross-product magnitudes must
exceed 1e−6, with all bonds longer than 1e−6 Å. The pseudoscalar is
`dot(cross(a,b),c)/(||a|| ||b|| ||c||)`; it needs four valid residues and three
nonzero bonds, and may validly be zero for planar geometry. Signed channels are
zero when unassessable, with separate assessability bits.

For any orthogonal Q with det Q=−1, F(QX)=QF(X)diag(1,1,−1).
Local z, sin(phi), and the triple product reverse sign; local x/y,
cos(phi), distances, angles, metadata and masks remain unchanged. Proper
rotations leave all feature channels invariant and rotate the frame. Translation
leaves features/frames/delta invariant and translates output. The signed features
preserve reflection parity; this does **not** establish complete chirality
resolution or require reflection-equivariant learned output.

## Local network and output

**927,875 trainable parameters**: 41→128 input projection; four local residual
blocks, each with six offset-specific bias-free 128→128 message maps, mean
aggregation over eligible neighbors, LayerNorm, a 128→512→128 SiLU feed-forward
network, and another LayerNorm. Hidden values are zeroed after normalization
at ineligible residues. Four blocks can propagate information up to 12 sequence
positions, but each interaction edge is exclusively ±1/±2/±3. No all-to-all
attention or geometry U-Net.

The 128→3 final head has exactly zero weight and bias. It predicts u in the
local frame. `delta = F @ u`, `P = Pg + delta`. No gate. Invalid corrections are
exactly zero. At initialization every tested hybrid prediction equals Pg bit
for bit, including all 60 historical panel examples. Only the head receives
nonzero gradients at the first backward; upstream local parameters begin
receiving gradients after the head moves. That is expected for zero-init heads.
Nonzero-head tests prevent equivariance from being tested only vacuously.

## Proposed objective and initialization scale audit

`L = L_local + beta * relu(L_cart(P,X) − (1+delta_cart)L_cart(Pg,X)) + rho * L_disp`.

`L_local` is the arithmetic mean of offset-1/2/3 distance-error MSEs. Each
uses endpoint masks, normalizes independently per protein, then averages
proteins. Empty offset sets contribute zero using the established clamped
normalizer. `L_cart` preserves historical raw xyz-component MSE plus 1e−5
residual-component MSE relative to the original E010 input; no alignment.
Both P and Pg use the same source and target; the baseline scalar is detached.
`L_disp` is equal-protein mean squared vector displacement (sum xyz, Å²).
Chirality is telemetry only.

One small candidate for review is **beta=1, rho=0.01, delta_cart=0.01**. The guard
penalizes excess over a 1% raw-Cartesian allowance; the regularizer is deliberately
small. These values were proposed before reading panel metrics. They are not
tuned against development results and remain a proposal, not a validated
non-inferiority guarantee. A soft penalty does not enforce a hard constraint.

The training-only CPU initialization audit sums local-parameter gradients with
exact equal-example 1/60 weights, without an optimizer or parameter update:

| Term | Gradient norm |
| --- | ---: |
| Local objective | 31.033474422962794 |
| Raw Cartesian objective (diagnostic only) | 1.8473598486461973 |
| Guard | 0 |
| Displacement | 0 |

Local/Cartesian gradient cosine is −0.8477993190081703. Parameter separation
prevents changes to E010; it does not imply the local branch has no objective
conflict. Guard and displacement are inactive at zero correction, so their
later scale is **not calibrated** by this audit. Active guard derivative and
baseline-detachment semantics are tested independently. No finite optimization
step was taken.

## Deterministic tiny panel and later protocol

The versioned `tiny_panel.json` freezes 20 training identities and 60 exact
identity-condition examples. The training-manifest SHA256 is
`69696551bbfbe00c0b7140a6e1b6e3c79de0f050dba25269ccf3068dd9f56cd0`.
Eligibility requires cached **training** examples at all three conditions; this
metadata filter avoids generating new denoiser inputs. Within each stratum,
rank `sha256(e010_phase4d_hybrid_local_global_v1|train|sample_id)`, breaking ties
by sample ID, and take four. Select minimum `(seed, schedule_index)` per
identity-condition from the historical cache. No development metrics or local
performance enter selection. Records pin shard bytes and individual coordinate
tensor hashes without storing coordinates in Git.

| Stratum | Identities |
| --- | --- |
| 20–64 | 2mdw_A, 1t8o_B, 9k3q_1, 2jo5_A |
| 65–128 | 5x1e_A, 9v0p_F, 9cfg_H, 8fmw_K |
| 129–256 | 1mhp_B, 9q1s_DD, 8cmd_A, 5fds_A |
| 257–384 | 8cqx_A, 1qqn_A, 9pbc_A, 6szs_z |
| 385–500 | 5avn_A, 6tzk_A, 3qv9_A, 9m6h_B |

Later proposal: 500 AdamW updates, LR 3e−4, no weight decay, gradient clipping
5, boundaries 0/50/100/250/500, all 60 examples with equal weights per update.
E010 stays frozen throughout. Precompute/verify detached Pg once; reuse exactly
the pinned examples and targets. The **guard applies after the effective-batch
Cartesian reduction**. Averaging independently rectified microbatch guards is
a different objective and is prohibited. Accumulation must preserve this
ordering, or evaluate the local branch on the full padded panel.

Before GPU execution, review and freeze these proposed gates relative to
boundary zero: ≥20% improvement in mean-local RMSE (arithmetic mean of three
RMSEs), every offset improves, aligned RMSD degradation ≤1%, every condition
locally non-harmful, aggregate chirality inversion rate non-worsening with
assessability non-decreasing, all outputs finite/no collapse, displacement
RMS ≤1 Å and maximum ≤3 Å. The latter bounds are proposals for explicit review.
Evaluate every boundary; update 500 is the predeclared terminal endpoint.
Do not pick a boundary using held-out metrics or silently change the gates.
These tiny training-panel outcomes would establish overfit capability only.

## Evaluation, validation and reproduction

Evaluation prepares proper-rotation-aligned coordinate RMSD, historical raw
Cartesian objective, offset RMSEs and mean-local RMSE, jointly assessable
nonplanar signed-triple chirality inversion, frame eligible/degenerate/interior
counts, RMS/max displacement, finite status and radius-of-gyration collapse
(<1e−3 Å). Nonfinite valid coordinates are flagged before SVD. Report pooled
counts and equal-example metrics overall, by condition and by length stratum.
Chirality rate denominators must accompany every comparison. Inapplicable
strata or empty assessability cannot be treated as passing evidence.

Distance-matrix diversity is available for matched-length predictions. The
record measures it across the three conditions per identity; cross-identity
pairwise distance comparison needs matching lengths and is inapplicable for
this panel's distinct lengths. Later evaluation must compare applicable
same-group diversity to frozen Pg and review a numerical collapse ratio before
execution; raw diversity alone is not a pass threshold.

The preparation script has no optimizer or training mode CLI. Reproduce with
the existing environment (do not install dependencies):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src \
/home/simostocco/miniforge3/envs/proteingen/bin/python scripts/prepare_e010_phase4d.py
```

Run `tests/test_e010_phase4d_hybrid.py` on CPU for frame/feature symmetry,
reflection parity, orthonormality/handedness, endpoint/degenerate/padding masks,
known synthetic torsion and triple products, zero-init parity against actual
Phase 4B weights, global gradient isolation/local gradients, nonzero output
equivariance, short/empty-chain safety, parameter counts, objective guard,
nonfinite telemetry and deterministic selection. Float64 nonzero-output
rotation tests use 1e−12 tolerances; actual checkpoint parity is exact.

The full CPU suite ran with 1,519 passed, 13 skipped, 134 failed. All 134 failing
test IDs reproduce on the untouched base commit, chiefly because historical
untracked inputs are absent in this worktree. The versioned validation record
contains the exact IDs and focused final results. No historical artifacts were
copied into or modified in this worktree to make these tests pass.

The remaining execution blockers are review/freeze of the candidate objective,
gates (including a diversity-collapse threshold), and GPU allocation. Exactly
one recommended next action: **review and freeze the Phase 4D tiny-overfit
contract before scheduling its separate GPU run.**
