# E010 Phase 4D V11 — direct correction-space strict-chirality solver

This changes only the correction parameterization of the V10 scientific problem.
Use the identical immutable 20-identity / 60-example training panel, frozen E010
Phase 4B update-1092 outputs, targets and masks. CPU float64, deterministic, one
thread per worker, four workers. No neural model is evaluated or trained.

Eight Cartesian recurrent steps share no learned parameters: delta = 0.04 z Å.
Explicit inequalities 1 - ||z||² >= 0 represent the intended CLOSED 0.04 Å ball.
The old radial map asymptotically represented its interior; admitting the exact
boundary does not increase the physical radius. The linear mapping Jacobian is
0.04 I, condition number 1. Eligibility is recomputed with the unchanged historical
rule at every step; ineligible physical corrections are exactly zero. All z start
at zero. Loose path/net bound is 0.32 Å; pair-distance change bound is 0.64 Å.

Final-state local-only loss is divided by the fixed baseline local loss solely for
the historical numerical normalization. There are no weighted Cartesian/chiral
terms. Proper-Kabsch aligned RMSD <= 1.01 baseline and continuous chiral loss <=
baseline remain unchanged. Import V9 QuartetConstraints directly: baseline-correct
s*q >= nextafter(1e-6,+inf); baseline-inverted q² >= tau²; exact historical bond and
frame assessability requirements. No new inversions per example, assessability
preserved. No feature, margin, ownership or target changes.

Use historical trust-constr options and exact objective/scientific-constraint HVPs.
Ball Jacobian is -2z, weighted Hessian -2 diag(repeat(mu,3)) in the lower-bound
convention. SciPy lower-bound multipliers are negative; report their negation as
nonnegative c<=0 ball multipliers. The user explicitly requires maxiter=1000,
superseding V10's 2000. No other trust radius, barrier, tolerance or settings change.
No retries or outcome-based reruns. Serialize states before optional telemetry.

Load the committed V9B configuration and conclusion limits through V10 setup.
Reuse V10 physical screen and certificate, including correction-space multiplier
reconstruction, fixed finite-difference directions/epsilons, projected cone and
fixed shadow steps. All interior certificates are byte-identical for matching
physical inputs/multipliers. Only exact-boundary primal bookkeeping replaces the
old representational '<' radius check with the intended '<=' bound. No tolerance
is relaxed; strict quartet feasibility and global normalized 1e-8 remain unchanged.
Raw trust-constr optimality is telemetry. Full certification requires ALL 60
physical certificates, plus >=5% condition-450 gain and every scientific gate.

Before optimization, validate all 60 baseline feasibility/zero parity and fixed
interior equivalence, all scientific constraints, objective and ball directional
derivatives at phases [0,.7,1.3], epsilons [1e-6,1e-5,1e-4] Å, historical mixed
error 1e-8 + 1e-5*abs(expected). Run historical synthetic length-12/32 from zero;
repeat length-12 to verify determinism. Valid preflights automatically authorize
the panel even if their physical convergence is incomplete; implementation or
integrity failure blocks execution. Never tune from preflight/scientific metrics.

Freeze DIRECT-B material numerical improvement before results: at least six more
passing examples and/or six fewer material-shadow examples (ten percent of panel).
DIRECT-A requires all 60 pass and all scientific gates; DIRECT-C all 60 pass but
condition-450 <5%; DIRECT-D no material numerical improvement; DIRECT-E execution
or implementation failure. Nonconverged endpoint metrics remain descriptive.
Compare matched V10 endpoints, optional converged-both coordinates/constraints.
Reconstruct all metrics and certificates independently from hashed saved z/delta
without reoptimization. Historical tracked artifacts and input hashes are pinned.
Private NPZ states are retained locally and excluded from commits.

Commands: run_e010_direct_correction_v11.py register, preflight, run, reproduce;
report_e010_direct_correction_v11.py publishes the versioned results. No follow-on
K/radius/solver or neural experiment launches automatically.
