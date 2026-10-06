# E010 Phase 4D v5 Cartesian constrained-oracle cross-check

Source result: `f9b42c36459f17c9f7fb42c20496b7c7b58a4ea6`. V4 and every earlier
artifact remain immutable. This is independent coordinate optimization, with
no neural/global model constructed, no CUDA, no development data, and no
change to the Python environment.

## Sole scientific change

Replace free moving-local-frame variables by free Cartesian variables. At each
step v=.04*z (the same dimensionless optimizer scale as v4) and
delta=v/sqrt(1+||v||²/.04²). Apply the same one-ULP inward rounding safeguard
and current eligibility mask, then Pt+1=Pt+delta. Initialize all z exactly zero.
K=4, s_max=.04 Å, beta=16.8 and gamma=2 remain unchanged. Final-state supervision
only, with no regularizer, guard, clipping, intermediate losses or fallback.

Eligibility is recomputed from Pt using the exact reviewed geometry routine.
Its frame/features are not consumed by correction variables and do not enter
the differentiable Cartesian trajectory. Endpoints/invalid frames get exactly
zero correction. Metric evaluation still uses the historical frame/chirality
functions. Per-step and cumulative limits are .04/.16 Å; pair-distance limit
is .32 Å, with the same historical numerical tolerances.

For an eligible residue, F is orthonormal, so ||F u||=||u||. Both free
representations span the same per-step correction ball at fixed geometry.
The optimizer sees different moving-coordinate derivatives. This does not make
the complete optimizations mathematically identical when nonlinear recurrence
or eligibility changes are involved.

## Exactly preserved solver, objective and data

Use the historical v4 solve function's **same code object**, with an isolated
copy of its globals binding only trajectory to the Cartesian implementation.
Never mutate the historical module globals. Initialization, L-BFGS construction,
closure, stopping and bookkeeping are literally shared. The same projected
gradient routine receives bounded Cartesian vectors instead of bounded local
vectors (the compatibility key name is unchanged, its coordinate basis changes).

L-BFGS LR=1, history=20, strong-Wolfe search, max_iter=1 per outer call,
max_eval=25, at most 1000 outer iterations. No restarts, Adam fallback or tuning.
Historical internal gradient/change termination remains disabled. Check every
10 iterations: ten stable iterations, relative objective change <=1e-8,
coordinate change <=1e-6 Å, projected residual <=.001, absolute gradient floor
1e-7. All values exactly match v4. Retain float64 arithmetic and the historical
Cartesian prediction.float() cast. Each of four CPU workers uses one thread.
No RNG or initialization from v4 results.

Use the exact immutable 20-identity/60-example Pg/source/target/mask cache.
Each independent contribution remains local/60 + 16.8*Cartesian/60 +
2*chiral_sum/13029, preserving v3/v4 pooled frozen-quartet reductions. No
condition weighting or loss-eligibility change. Reuse v4 metric/aggregation and
safety code verbatim. Preserve all initial/final metrics, P0–P4, temporary
assessability changes, convergence history, objective changes and evaluations.
Line-search failure counts are unavailable from the unchanged v4 solver; record
that explicitly. There are zero explicit optimizer restarts.

## Frozen interpretation and comparison

Strong convergence is predeclared as >=48/60 overall (80%) and >=18/20 at
condition 450 (90%), a substantial increase over v4's 26/60 and 10/20. Otherwise
classify CART-O5, irrespective of correction saturation or failure to reach 5%.
With strong coverage: CART-O1 if 450 safely gains >=5%; CART-O3 if that gain
violates safety (observed trade-off, not proof of necessity); CART-O4 if both
controls safely exceed 5% while 450 does not; CART-O2 if 450 is safe but below
5%. CART-O2 supports combined bound/objective limitation, not bound alone.
Safety thresholds are identical to v4 and cannot be outcome-adjusted.

Compare convergence by condition, stratum and overall, with all four paired
transitions. For converged-both examples retain paired objective, local gain,
raw Cartesian, chiral loss, inversion and cumulative correction differences.
For newly converged examples report aggregate condition results versus matched
v4 outcomes. Predeclared equivalence uses objective relative tolerance 1e-6,
local RMSE/correction RMS <=1e-5 Å, raw Cartesian <=1e-5 Å², chiral loss <=1e-5,
and exactly equal inversion counts. These are descriptive, never optimizer knobs.
A lower weighted objective alone does not prove maximum safe local repair.

## Execution and validation

Before panel execution, hash config, implementation, protocol, historical v4
source/config and records, v3 source/config and frozen cache. Check them again
after execution. Run focused CPU tests for bounds, eligibility, no neural/global
parameters, exact solver/objective/metrics, protected inputs and deterministic
reproduction. Repeat **all 60 examples from zero** with identical execution
settings; compare every non-timing scalar, metric, state and convergence history
exactly. Commit/push only after tests and reproduction succeed.

Create new records under cartesian_oracle_v5 only. No historical overwrites,
larger correction budgets, coefficient sweeps, optimizer variants, neural
training or automatic follow-on experiment.
