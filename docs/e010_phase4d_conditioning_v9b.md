# E010 Phase 4D V9B: fixed-state strict-constraint conditioning audit

Historical V9 commit: `244ac4d1ba4e8375cca30e4172fd44e51c9d0a7f`.
No scientific panel, CUDA, neural network, new geometry, or changed tolerance.

The mandatory cases are the existing length-12 and length-32 strict synthetic
preflights. No intermediate-size strict case was previously recorded; none is
created or selected based on outcomes. V9 did not persist final correction arrays
or coordinate hashes. The unchanged historical solver is replayed only to recover
missing states and terminal solver telemetry. Every recorded non-timing field,
including all multipliers, constraints, metrics, and iteration history must match
exactly after JSON canonicalization. A mismatch stops recovery. New state and
input hashes freeze the recovered arrays; they cannot retroactively establish a
historical coordinate hash. Binary synthetic states are untracked. Scientific
cache inputs and E010 are never loaded.

## Frozen analysis contract

See `configs/e010_phase4d_conditioning_v9b.yaml` and the report preregistration.
Raw local loss has units Å². Normalize it by the fixed Pg local loss L0 and use
physical coordinates x=delta/(0.04 Å). The dimensionless stationarity residual is
0.04 times the raw correction-space residual divided by L0. This scale does not
depend on the final Lagrangian gradient. Also report raw residuals in Å and raw
complementarity in Å². Constraint normalization is exactly V9's baseline scaling;
primal violations are additionally checked against the exact evaluator.

Active inequalities have normalized slack at most 1e-5; correction balls use the
historical relative boundary tolerance 1e-4. Both are analysis masks, not relaxed
feasibility. Fit nonnegative active multipliers by linear least squares and
compare with the original barrier multipliers. SVD rank uses
max(matrix dimensions)*float64 epsilon*largest singular value. Analyze original,
row-normalized, and radial-chain Jacobians separately. Duplicate directions are
identified by absolute row cosine above 1-1e-8.

The tangent-cone projection is obtained through nonnegative least-squares dual
reconstruction. It is a fixed-state calculation, not coordinate optimization.
Evaluate independent, noniterated shadow steps of 1e-6, 1e-5, and 1e-4 Å after
scaling the direction's maximum residue norm to one and projecting each ball
inward. Recompute eligibility and all original constraints. A material decrease
means at least 1e-6 in normalized local loss (one part per million). Strict extra
inequalities remain exactly nonpositive, global safety tolerance remains 1e-8,
and no new inversions or lost assessability are accepted.

Float64 directional checks use three deterministic normalized directions and
central differences at 1e-6, 1e-5, 1e-4 Å. Require one passing scale per direction,
with absolute tolerance 1e-8 and relative tolerance 1e-5. Validate objective,
selected active constraint rows, and the complete fixed-multiplier Lagrangian.

## Installed SciPy interpretation

SciPy 1.18.1 `minimize_trustregion_constr.py`, `update_state_sqp`, constructs
`lagrangian_grad = grad(f) + sum(J.T @ multipliers)` and reports its infinity
norm as `optimality`. V9 optimizes z=v/0.04 with f=Llocal/L0 and baseline-normalized
constraints. Thus this number is a dimensionless derivative with respect to z,
not physical ball-constrained stationarity. `constr_violation` is the maximum
positive bound violation of these normalized constraints. `tr_radius` is the
optimizer/slack-space trust radius. The logarithmic-barrier coefficient and
barrier-subproblem tolerance are `barrier_parameter` and `barrier_tolerance`;
they are not physical geometric tolerances. `cg_stop_cond` describes the last
inner CG stop: 0 not evaluated, 1 iteration limit, 2 trust-region boundary,
3 negative curvature, 4 tolerance satisfied.

Interior-point status 1 requires both optimality and constraint violation below
gtol. Status 2 requires trust radius below xtol AND barrier parameter below
barrier_tol. Thus status-2 success does not certify gtol. V9 correctly retained
its separate 1e-12 test. Installed source hashes and terminal fields accompany
this audit. Source references are local to the existing environment; it is not
modified.

## Limits and prospective termination

No historical result is rewritten. A physical stopping proposal is considered
only after derivatives, feasibility, complementarity and feasible-shadow steps
are checked. Thresholds must follow float64 derivative resolution and a frozen
physical sensitivity budget, not the desire to accept length 32. No proposed
contract is applied to the scientific panel here.

Even a converged weighted-sum optimum is different from maximum safe local
repair. This audit evaluates numerical stationarity only and cannot classify
scientific local feasibility or correction-budget adequacy.

A candidate future contract can use a dimension-independent sensitivity rule.
Let M be the number of eligible step/residue correction vectors, epsilon_F=1e-6
(the frozen material relative-loss threshold), and h=1e-4 Å (the largest frozen
shadow displacement per vector). Cauchy–Schwarz gives a first-order loss change
bounded by ||r_x||_2 sqrt(M) h/0.04, where r_x is the dimensionless physical
stationarity residual. A candidate sufficient sensitivity threshold is therefore
||r_x||_2 <= epsilon_F/(sqrt(M)*h/0.04). Derivative-validation uncertainty must be
at least ten times smaller than this threshold; otherwise the certificate is
numerically unresolved. Keep original exact chirality/assessability feasibility,
original global normalized feasibility <=1e-8, dual negativity <=1e-8,
complementarity <=1e-6, exact correction bounds, and require no material feasible
shadow decrease at any frozen step. This is a prospective, local certificate,
not a global optimality proof. It is not applied to any panel example.
