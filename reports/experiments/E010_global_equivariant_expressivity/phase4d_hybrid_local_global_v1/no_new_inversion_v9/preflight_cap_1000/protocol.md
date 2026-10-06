# E010 Phase 4D V9: strict no-new-inversion oracle

Historical V8 commit e293857953b064439645f1c2ae36fdd67b36fe2b remains immutable.
The only scientific intervention is endpoint chirality constraints. Keep K=8,
radius.04 Å, CPU float64, the exact frozen60-example panel, zero initialization,
local-only objective normalized by its frozen baseline, and the two V8 safety
inequalities. Do not warm start or introduce beta/gamma. No neural network or
E010 model is optimized; the immutable frozen-global cache supplies Pg.

## Exact evaluator and inequalities

Historical signed_status uses the local_representation pseudoscalar at central
residue i, quartet(i-1,i,i+1,i+2). With a=x_i-x_(i-1), b=x_(i+1)-x_i,
c=x_(i+2)-x_(i+1), q=dot(cross(a,b),c)/(||a||||b||||c||), with historical EPS³
clamp. Assessability requires all three lengths>EPS, central frame projected
ratio ||a-dot(a,e1)e1||/||a||>EPS, and |q_pred|,|q_target|>EPS, EPS=1e-6.
An inversion is assessable q_pred*q_target<0. Masks and contiguous quartets
are unchanged; the central frame condition is essential and not replaced by
an unrelated torsion or angle convention.

Let tau=nextafter(float64(EPS),+infinity). This is the smallest representable
float64 value satisfying the strict historical >EPS predicate; it is not a
chosen geometric margin. All13,029 panel quartets are assessable at Pg.
Freeze correct/inverted labels using the evaluator, and target signs s_j.
For baseline-correct quartets only impose s_j*q_j>=tau. For baseline-inverted
quartets impose q_j²>=tau² only, to preserve assessability without restricting
orientation sign. Also impose the historical central-frame ratio>=tau on all
baseline-assessable quartets and bond-length>=tau on their participating bonds.
Do not add a second-plane torsion-angle condition: it is absent from the evaluator.
The unsigned square inequality permits both signs. An inverted quartet may
stay inverted or become correct; final feasibility is not a restriction on
its orientation throughout optimizer iterations. keep_feasible is not enabled.

Normalize new inequalities by their positive baseline absolute q, squared q,
frame ratio, and bond length respectively. This constant scaling does not alter
the feasible set. The V8 aligned and continuous-chiral ratios remain identical.
Crucially accept each new constraint only if its computed residual is <=0;
the historical1e-8 tolerance applies to the two pre-existing inequalities only.
No feasibility slack may admit equality at EPS or erase assessability. Independently
verify the historical discrete evaluator at final output on every example.

Proof in float64: the signed floor makes each formerly correct quartet retain
its target sign with |q|>EPS. The unsigned floor on inverted quartets guarantees
|q|>EPS with either sign. The bond and frame floors ensure historical geometric
assessability. Targets/masks are fixed. Since all panel quartets were assessable,
there are no excluded quartets that could become newly assessed inversions.
Thus no previously correct quartet can invert and inversion count cannot exceed
its baseline. Equality at tau remains assessable; equality at EPS is rejected.
The inequalities are differentiable on their positive-length, nondegenerate
feasible domain. Their squared unsigned q is smooth even at a sign crossing.
Nondifferentiable collapsed geometry is infeasible, as in the historical losses.

## Validation and numerical contract before panel results

Verify baseline feasibility, exact evaluator quantities, synthetic sign crossing,
reflection, unconstrained repair of baseline-inverted quartets, geometric collapse
rejection, constraint Jacobians/HVPs against float64 finite differences, proper
Kabsch/continuous parity and zero/init/bounds. Validate each of60 actual baselines
without optimizing it: three fixed normalized directional Jacobian checks,
zero-state Hessian checks, and existing aligned-gradient validations.

Retain SciPy1.18.1 trust-constr and every V8 numerical tolerance/iteration cap,
including maxiter1000, gtol/xtol/barrier_tol1e-12, initial barrier1e-7, four CPU
workers/one thread each. Sparse Jacobian storage is the sole numerical linear
algebra change necessitated by the hundreds of endpoint inequalities. Jacobians
are exact: coordinate-space derivatives mapped through block-local radial
Jacobians. Validate against full autograd and finite differences. Hessian-vector
products remain exact autograd; no approximation, per-condition tuning or retries.
Synthetic lengths12 and32 use the same predeclared geometry/targets as V8
(2*i,sin(i),cos(i)) plus(.7*sin(1.7*i),.8*cos(1.2*i),.6*sin(2.3*i)).
Both must pass the full solver/physical convergence gate before panel execution.
If the preflight fails, preserve it and stop; no scientific panel is permitted
under an unvalidated numerical contract. No revised cap is currently proposed.

Retain correction-space ball KKT/projection thresholds.001, complementarity1e-6,
dual feasibility1e-8, and strong coverage>=48/60 and>=18/20 at condition450.
All scientific results remain infeasible unless new constraints and discrete
assessability/no-new-inversion predicates pass exactly, even if SciPy terminates.
Weighted Lagrangian diagnostics use solver multipliers only, never a changed
scientific weighted objective. Nonconvex stationarity is not global optimality.

## Reporting and frozen classification

Record every P0-P8 metric, quartet activity and signed margins, new/repaired
inversions, assessment counts, correction step/net/path RMS/max, cosine and
saturation telemetry, solver residuals/multipliers and histories. Aggregate overall,
by conditions and historical length strata, comparing only against V8 Arm B.
No development metrics or panel selection. Repeat all60 solves from zero with
exact non-timing record equality before commit. Pin all historical and protected
input bytes; variable/coordinate caches remain outside Git.

Missing strong numerical coverage gives INV-D. With strong coverage, any new
inversion/per-example count increase gives INV-C. Any other unpassed safety gate
or infeasibility gives INV-D. Otherwise condition450 gain>=5% gives INV-A;
<5% gives INV-B. Every protein must pass, with no exceptions. Do not infer a
geometric radius problem merely from a stricter chirality constraint or a stalled
solver. No K/radius sweep, neural training, CUDA or follow-on experiment occurs.
