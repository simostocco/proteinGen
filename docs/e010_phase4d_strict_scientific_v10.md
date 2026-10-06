# E010 Phase 4D V10 strict scientific panel

Only termination bookkeeping changes from V9. Use its exact Oracle,
QuartetConstraints, trust-constr wrapper, exact Hessian products, solver settings
including cap2000, and zero initialization. K8/.04 Å, 60 historical examples,
local-only objective and all strict safety constraints are unchanged. No CUDA,
network training, warm starts, outcome-based reruns or tolerance changes.

Load the committed V9B config and conclusion directly. Import its physical KKT,
active-system, cone projection, radial-ball projection and finite-difference
implementations without editing them. The certificate uses normalized physical
stationarity L2 <= material/(sqrt(M)*maximum_shadow_step/S), literally the V9B
formula. Load complementarity1e-6, dual negativity1e-8, exact extra feasibility,
global safety1e-8, all epsilon/direction sets and material1e-6 from those artifacts.

Evaluate cheap exact Lagrangian-vector products at the inherited callback cadence
(1 and each10 iterations). This screens candidate points only. Before accepting,
materialize the full physical Jacobian and execute the complete V9B certificate.
A passing certificate stops via the SciPy callback. Raw SciPy optimality and status
are telemetry, never acceptance gates. At any SciPy termination, including the
unchanged cap, compute the full physical certificate. Never retry a failed solve.

Derivative uncertainty uses S times the best fixed-epsilon maximum objective/Lagrangian absolute
central-difference error for each deterministic direction. Require all existing
mixed absolute/relative derivative tests and uncertainty <= stationarity_limit/10.
In accordance with the committed V9B requirement to cover certificate directions,
also validate the projected feasible-gradient and physical Lagrangian-gradient
unit directions using exactly the same epsilons and mixed tolerances. Constraint Jacobians separately retain the frozen mixed tolerance, since an
unweighted constraint derivative has different scaling from stationarity. This
bookkeeping rule is fixed before panel outcomes; no per-example settings are tuned.

Independent shadow perturbations use the unchanged V9B cone and radial projection,
max-vector step normalization, three fixed step sizes and one-ppm materiality.
Recompute all original constraints, discrete evaluator, and eligibility. Report
infeasible shadows rather than interpreting them as descent. No material feasible
shadow may lower the objective. All final steps must remain <.04 Å, with cumulative
path/net bounds .32 Å (historical numerical reporting tolerance .320001).

Save complete final variables, all states and multipliers with hashes immediately
on result production. Reproduction rebuilds trajectories from the saved variables,
recomputes every metric, independently recomputes full physical certificates and
checks exact canonical JSON agreement; it does not rerun optimization.

Before execution, verify old source hashes, the immutable cache and source-input
pins, all60 baselines exactly feasible, all13029 signed quantities bitwise, the
pinned unchanged historical180+180 panel derivative checks, and focused CPU tests.
New tests cover the certificate on frozen recovered V9B states and synthetic cases.

Classification: all60 full physical contracts and safety gates plus condition450
>=5% give V10-A. All60 passing but <5% give V10-B. Converged geometry with
per-example safety failure gives V10-C when aggregate >=5%. Any unresolved
numerical certificate failure gives V10-D. No failed example is filtered from
metrics. Report all trajectories, length/condition groups, strict margins,
inversion transitions, constraint activity and numerical telemetry.
