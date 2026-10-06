# E010 V12 fixed-state trust-constr / physical-KKT audit

Analysis only on the e010-phase4d-hybrid-local-global branch. Load all 60 original
V11 final NPZ files against the committed state-file, variable and P0–P8 coordinate
hashes at 4a82eff5169298022683f47cd345af5ceae02a2c. Recompute exact metrics, final
objective, scientific/ball constraints and frozen physical certificates. Solver
status/iteration metadata are verified against committed records; they cannot be
inferred anew from coordinates. No missing state may be regenerated. Pin historical
tracked artifacts, all state files and protected input hashes before execution.

No nonlinear/scientific optimization, model training or CUDA. Guard every historical
trust-constr/solver entry point and torch optimizer steps with exceptions. The only
solve is the expressly authorized fixed-state tangent-cone diagnostic linear QP:
min_h 0.5||h + S*g||² subject to A*h <= 0, via the unchanged deterministic V9B NNLS
dual (same iteration formula, no new settings). Here g is the derivative of
L_local/L_local(Pg) with respect to physical delta in Angstrom; A uses the exact
V9B active-set convention. Scaling the cone projection by its maximum per-vector
norm gives an approximate linear-loss minimizer for the frozen BLOCK trust radius.
It is not claimed to globally solve the block-radius linear program. Predicted
decrease and primal/dual/complementarity diagnostics are reported. No nonlinear
step is iterated or chained.

Use identical V11/V9B shadows: 1e-6,1e-5,1e-4 Angstrom, historical inward ball
projection, global residual <=1e-8, exact extra constraints, zero new inversions,
assessability preservation, exact historical material normalized decrease >=1e-6.
Recomputed direction, projected norm and shadow labels must match the historical
certificate exactly. The raw MSE decrease is additionally recorded in Angstrom².
Physical stationarity/complementarity/dual thresholds are unchanged. V11 remains
DIRECT-D regardless of V12 interpretation.

Report the frozen group counts before aggregate interpretation: A4 physically
converged, B55 nonconverged with material shadows, C1 nonconverged without shadows.
Conditions A/B/C: 50=2/18/0, 250=1/19/0, 450=1/18/1. Stratum20–64=4/7/1;
all other strata=0/12/0. C is index14, 2mdw_A at450. No example is excluded.

Active Jacobian/rank/near-dependence use the committed V9B functions/tolerances.
Report both frozen normalized singular values and physical derivatives (divide
all rows by S=.04); condition/rank unchanged. Row-unit conditioning is a companion
diagnostic, not a constraint rescaling applied to the scientific problem. Exactly
empty active sets have rank0 and no condition number; do not invent infinity.
Distinguish the frozen active-ball band (relative tolerance1e-4) from descriptive
99%-radius saturation. Radial outward descent means a NEGATIVE g·unit(delta);
inward descent a POSITIVE dot. Tangential projection removes that dot component.
Report local-objective and solver-multiplier Lagrangian decompositions separately,
active/interior contributions, every active ball and step/residue positions.

Parse only saved telemetry. Histories are sampled every10 iterations, not every
iteration. Window (last_iteration-W,last_iteration] uses available snapshots;
report their actual span. Fit ordinary scalar slopes, with no convergence-time
extrapolation. Physical screen histories are saved, but full certificates and
multiplier vectors are only final. Constraint penalty, internal slack states and
actual accepted-step norms are unavailable. Never invent them. Compare final and
sampled length32 V9/V9B records without any solver replay.

Installed SciPy optimality is the infinity norm of the optimizer-z Lagrangian
gradient; reconstruct it including exact quadratic-ball multipliers. Barrier-stage
termination uses an augmented slack-space criterion. Implied true-slack centering
nu*(1-||z||²)-barrier is a computed proxy, NOT saved internal slack telemetry.
S*trust_radius is only a nominal physical global displacement upper bound, not an
observed coordinate step. Configured terminal barrier_tol is1e-12; physical
convergence can occur above it, so a large barrier/tolerance ratio alone is not
proof of failure. Multiplier time evolution is not recoverable; saved scalar
dual/complementarity histories are proxies only.

Descriptive evidence flags do not recertify states: normalized objective decrease
>=frozen1e-6 in the last100 available-iteration window; stationarity relative drop
>=1%; direction tangential squared-norm fraction>=50%; and frozen/99%-band ball
activity. Report these separately, plus overlap/unclear cases. Infer NUM-A..F only
after all fixed analyses reproduce. Percentages expressing evidence signatures
must not be called causal proof. No automatic2000-iteration or different-solver
experiment. One next experiment may be recommended after interpretation.

Private endpoint arrays/Jacobians/directions are not committed. Only versioned
analysis code, provenance, compact per-case reports and aggregate records are
published after independent reproducibility checks.
