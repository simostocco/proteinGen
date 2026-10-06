# E010 Phase 4D v7 float64 local-feasibility oracle

Historical v6 commit `36113e708194d280ead04af4359e6a8d599a6955` and all earlier
artifacts remain unchanged. Use the same immutable 20 identities, 60 cached
Pg/target/source/mask examples, three conditions and five length strata.
Independent zero initialization for each arm/example; no warm starts or RNG.
No neural models, CUDA, environment modification, held-out selection or bound sweep.

## Geometry, objectives and constraints

Use physical Cartesian v=.04*z and the historical smooth radial map, K=4,
radius .04 Å and historical one-ULP inward safeguard. Recompute current frame
eligibility at each step. No frame features enter the free variables. Invalid
frames/endpoints have zero correction. Net displacement and path length cannot
exceed .16 Å per residue (numerical tolerance .160001); pair-distance change
cannot exceed .320002 Å. All coordinates, variables, derivatives, losses,
linear algebra and stopping calculations are CPU float64. Reject other dtypes.

V7 evaluates the same radial map as v/sqrt(1+sum(v²)/s²), avoiding the historical
hypot(norm(v)) expression's undefined intermediate norm Hessian at zero. The
inward safeguard likewise clamps squared norm before sqrt. Zero parity,
nonzero-map agreement and analytic zero-state Hessian are tested. An initial
execution stopped on this implementation issue before any example result;
its source/config/trace are preserved in preflight_attempt_01. The implementation
was corrected without changing any solver setting, objective, constraint or
feasible correction set. All60 zero-state Hessian preflights now precede execution.

The next execution saved all120 coordinate states but encountered a NumPy-bool
JSON serialization error. That attempt is preserved in preflight_attempt_02.
Only output conversion was corrected. Every regenerated variable/coordinate
must match those archived states exactly; no mathematical or solver setting
changed and no approximate replacement is accepted.

Both arms minimize the same endpoint-masked mean of offset-1/2/3 distance MSEs,
divided by its positive frozen baseline as a constant numerical normalization.
There is no weighted Cartesian/chiral term and no beta/gamma in the objective.
Historical coefficients 16.8/2 are recorded only as context. Arm A has no safety
inequalities. Arm B imposes proper-Kabsch aligned RMSD <=1.01 times baseline
and continuous signed chirality MSE <=baseline. Frozen quartet eligibility and
the signed convention are unchanged. Normalize each inequality by its positive
baseline limit; refuse a zero normalization rather than silently substituting.
Raw Cartesian telemetry retains the historical source-residual term 1e-5 with
the float32 cast removed. It is never an optimization term or safety substitute.

Proper row-vector Kabsch centers both structures, computes U,S,Vh of Pc.T@Xc,
and applies U*diag(1,1,det(U@Vh))*Vh. RMSD is the residual Frobenius norm/sqrt(N).
Validate the float64 gradient on synthetic geometry and every fixed Pg example
with the three deterministic directions/epsilons in the config. Any failed
direction stops execution. Binary inversion count is final telemetry/gating,
never a differentiable constraint. Every Arm-B example must preserve baseline
assessability and not exceed its own baseline inversion count to pass the final
scientific gate; also report aggregates and condition-specific changes.

## Solver and convergence, frozen before panel results

One SciPy 1.18.1 `trust-constr` contract for both arms/all examples, with exact
autograd objective/constraint gradients and matrix-free Hessian-vector products.
This avoids a dense Hessian for up to 6000 correction variables. The optimizer's
internal merit/barrier handling enforces inequalities; it is not a beta/gamma
scientific weighted objective. No restarts, per-example changes or fallback.
Max iterations 1000, gtol1e-12, xtol1e-12, barrier_tol1e-12, initial radius1 and
constraint penalty1. Initial barrier parameter/tolerance1e-7. Synthetic active
constraint validation showed default .1 could stop with a boundary error ~4e-4;
the smaller barrier was selected for numerical correctness before panel outcomes.
The strict parameter-space tolerance reflects V6's strongly suppressed radial
gradients and is selected before any panel optimization. All remaining options,
tolerances and solver identity are pinned in the config.

Record solver success separately from audit convergence. Audit convergence
requires solver success, normalized Lagrangian optimality <=1e-12, normalized
constraint excess <=1e-8, dual feasibility <=1e-8, complementarity <=1e-6 and
correction-space normalized ball KKT and projected mapping <=.001. Constraints
are active within1e-5. Correction balls are near-active within1e-4 relative.
Use solver multipliers for the two normalized safety inequalities, and minimize
ball-stationarity residuals over nonnegative radial multipliers. Interior
residual is the full correction-space Lagrangian gradient. These checks prevent
vanishing radial parameter gradients alone from establishing convergence.

Keep objective/constraint history every10 iterations and final diagnostics.
No result is called constraint-feasible beyond the pinned tolerance. Arm A is
an achievable local-only oracle candidate, not a proposed neural loss or a
certified global upper bound. Both optimizations remain nonconvex: strong local
convergence does not prove global maximum repair. A failed inversion gate is
evidence that the solver found no gate-passing candidate, not a mathematical
proof that every feasible correction would worsen binary handedness.

## Frozen decisions

Strong numerical coverage requires >=48/60 and >=18/20 at condition450 for each
arm. If either arm lacks it or Kabsch validation fails, FEAS-E takes precedence.
With strong coverage, Arm-A primary gain <5% gives FEAS-C (evidence within this
deterministic oracle, with the nonconvex limitation above). Arm-A gain >=5% and
Arm-B gain >=5%, with every Arm-B inequality and inversion/assessability gate
passing, gives FEAS-A. Arm-A >=5% but no gate-passing Arm-B primary5% candidate
gives FEAS-B; report active constraints and any gate failure separately. Reserve
FEAS-D for mixed conditions not covered by these rules. Do not change decisions
after results. Do not infer budget insufficiency from failed convergence.

## Records and reproduction

Hash all tracked historical sources/results and protected checkpoint/data inputs;
pin V7 config/implementation before optimization. Record all120 arm/example
results with baseline, P0-P4, correction RMS/max, path length, assessability,
frames, solver histories, constraint residuals/multipliers and coordinate digests.
Persist local-only final variables outside Git for future numerical audit.
Repeat all120 runs from zero and require exact non-timing result parity before
commit/push. Reports are new under local_feasibility_v7 only; no checkpoints or
datasets are committed. No high-K/high-s experiment runs automatically.

Solver API references: [trust-constr](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html)
and [NonlinearConstraint](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NonlinearConstraint.html).
