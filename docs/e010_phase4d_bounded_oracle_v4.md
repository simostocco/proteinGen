# E010 Phase 4D v4 bounded correction oracle

This is independent per-example coordinate optimization, not neural training.
Source is the exact immutable v3 frozen panel and result commit
`43864b38001c2ee0118bc8b7530b547cba4ca5f1`. Historical evidence stays unchanged.
The source E010 checkpoint SHA256 remains
`f5211cbc1be5092175ce15b9761a4efba761d242310287cd6b1e04df6a6744ef`.
E010 is neither constructed nor included in autograd/optimization.

## Pre-registration

Each example has a separate zero-initialized leaf tensor z[4,1,N,3], with no
neural network, weights, cross-protein sharing or random initialization. Step t
rebuilds the reviewed local frame from Pt and maps u=.04*z[t] through the
same smooth radial bound u/sqrt(1+||u||²/.04²). A one-ULP inward rounding safeguard
matches v3. Ineligible frames and endpoints receive zero correction. All four
steps are differentiated; frames are not detached. K=4, s_max=.04 Å, cumulative
bound .160001 Å and pair-distance bound .320002 Å remain fixed.

Local-frame variables match the recurrent refiner's geometric semantics while
removing its learned capacity and optimizer constraints. Dimensionless z improves
conditioning without changing reachable corrections. Only geometry is needed;
no learned feature processing or neural parameters occur.

The final-state objective uses **the exact v3 implementation**: beta=16.8,
gamma=2, no added loss or regularizer. Each independent example minimizes its
contribution to the full-panel objective: local/60 + 16.8*Cartesian/60 +
2*chiral_sum/13029. This preserves equal-example local/Cartesian and pooled
frozen-quartet chirality weighting; replacing it by a per-protein chiral mean
would change the scientific objective. Eligibility is frozen from Pg/target.

Deterministic CPU L-BFGS, LR=1, history=20, strong-Wolfe line search, one outer
iteration per call, maximum 1000 iterations. No clipping, restarts, setting
sweeps or panel-outcome tuning. Internal gradient/change termination is disabled
in favor of explicit bookkeeping. Corrections start at exact zero. No seed or
RNG is used. Single CPU thread, float64 correction/frame arithmetic; historical
Cartesian prediction.float() is deliberately retained, so that term still has
float32 rounding. Targets/cache values are exact casts of original float32 data.

Every 10 iterations check convergence, requiring 10 consecutive iterations with
relative objective change <=1e-8 (denominator max(1,|previous contribution|)) and
maximum coordinate change <=1e-6 Å, plus projected-gradient residual <=.001.
The residual perturbs each bounded local vector, including downstream frame
derivatives, projects onto its .04 Å ball and divides by the radius. Its gradient
trial scale gives the largest gradient one radius; gradients <=1e-7 in absolute
contribution units are treated as numerically stationary. This absolute floor
handles inherited float32 Cartesian arithmetic. Radial saturation alone cannot
pass the projected residual. Exhausting budget is **not** convergence.

A synthetic geometry preflight checks this contract before panel execution.
No panel outcomes are used to choose settings. Configuration and implementation,
all tracked v3 records, historical source pins and frozen cache are hashed before
execution and verified afterward. Per-example records are immutable; resuming
only skips already completed records under matching pins.

## Metrics, safety and interpretation

Use the exact historical/v3 local, Cartesian, aligned RMSD, signed-chirality and
aggregation functions. Record baseline, P0–P4, condition/stratum/overall metrics,
correction RMS/max, frame eligibility, assessability, optimizer iterations,
closure evaluations, projected residual and stability history. Distances and
RMS corrections are Å; MSE is Å²; normalized chiral loss is dimensionless.

Safe means all offsets non-harmful, raw Cartesian within a declared 1e-6 relative
numerical tolerance, aligned RMSD regression <=1%, continuous chiral loss within
1e-6 relative numerical tolerance, inversion count non-increasing, original
assessability preserved, frames preserved and finite outputs. The tiny tolerances
are numerical, not outcome-selected. Neural comparisons are descriptive only.

This nonconvex fixed-objective solution is a practical attainable correction,
**not a proof of maximum local gain over all Cartesian/chirality-safe corrections**.
To avoid attributing an objective trade-off to the bound, also compute a rigorous
optimistic geometry envelope: every pair's absolute distance residual can shrink
by at most the sum of its residues' cumulative displacement radii (.16 Å per
interior residue, zero endpoints). Clamp the resulting residual lower bound at
zero and compute offset/mean-local RMSE. Relaxing pair coupling and safety makes
this an upper bound on achievable local improvement, never an optimized loss.

Classification precedence: O5 if any example fails numerical convergence; O1 if
condition 450 safely improves >=5%; O3 if it achieves >=5% but violates safety
(observed trade-off, not proof that unsafe corrections are the only possibility);
O4 if both controls safely exceed 5% while 450 does not; O2 only if the optimistic
geometry envelope itself rules out 5% at 450. Otherwise O5: the permitted oracle
cannot distinguish bound insufficiency from objective limitation. A converged
fixed-objective gain below 5% alone cannot establish that no safe direction exists.

No changed K/s_max/beta/gamma, neural architecture/training, PCGrad, development
tuning or automatic downstream experiment. CPU execution cannot contend for
E012 CUDA ownership. Python environment stays unchanged.
