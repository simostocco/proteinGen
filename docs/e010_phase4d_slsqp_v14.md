# E010 V14 direct-space SLSQP benchmark

This solver-only intervention retains V11/V13's exact closed-ball K=8,
s_max=0.04 Å scientific problem and the committed V9B physical certificate.
All 60 panel identities, conditions, cached inputs, targets, eligibility,
proper-Kabsch, continuous/signed/assessability constraints and telemetry remain
historical. The objective is the same Pg-normalized local-only objective.

Use installed SciPy 1.18.1 SLSQP without changing the environment. Freeze
maxiter=2000, ftol=1e-12, zero initialization, CPU float64, one arithmetic thread,
and sequential panel execution. Two vector-valued inequality dictionaries use
analytic float64 Jacobians: negated historical scientific c<=0 constraints and
historical 1-||z||²>=0 balls. There is no added numerical constraint scaling.
SLSQP's internal dense BFGS/QP treatment replaces trust-constr's exact HVPs;
no Hessian/HVP is supplied. SLSQP success is not physical certification.

Before execution, validate objective and every constraint family against the
same historical directional finite differences, and match all 60 historical
baseline/derivative records. Run historical synthetic lengths 12 and 32 from
zero, then exactly one longest-length resource/plumbing smoke using unchanged
`recover_e010_conditioning_v9b.synthetic(500)`. No preflight outcome changes any
solver setting.

The longest smoke first calculates installed SLSQP's exact mandatory array
allocation. With n variables and m inequalities (no equalities), its float64
workspace contains n(n+1)/2 + 3mn + 9m + 8n² + 35n + 28 elements, in addition
to the m-by-n dense constraint normals. Other solver arrays, dense Jacobian
temporaries, PyTorch and OS memory are extra. Reserve max(2 GiB, 25% of available
RAM); swap does not count. If required arrays exceed the remaining budget,
validate length-500 derivative plumbing but refuse the dense allocation, stop
before the scientific panel, and report SQP-E. A 600-second longest-smoke wall
budget is also frozen. This operational gate changes no variable, correction
radius, constraint, objective or physical certificate threshold.

Completed optimizer variables/status are serialized before certificate or
optional telemetry processing. Empty active sets are legal; absent solver
fields are null. Independently reconstruct physical multipliers with historical
active-system extraction and NNLS cone projection. Feed them to the unchanged
V11/V9B physical certificate, including exact shadow scales/materiality.
Save and reproduce all coordinates and certificates, including failed solves.
Do not rerun panel examples; private arrays remain untracked.

If resource/derivative preflight passes, execute all 60 once from zero. SQP-A
requires all 60 physical passes, condition-450 gain >=5% and all strict safety.
SQP-B requires both >=30 passes and <=25 material-shadow cases but incomplete
certification. SQP-C otherwise denotes little numerical improvement in a valid
run. SQP-D requires 60 physical passes with condition-450 gain <5%. SQP-E is
resource failure; SQP-F is integrity/implementation failure. No scientific
conclusion follows from resource failure or nonconverged endpoints.

Commands use the existing proteingen Python with OMP/MKL/OpenBLAS threads=1 and
PYTHONPATH=src:.; register, preflight, guarded run/reproduce, then report. No CUDA,
neural training, new package, different solver, increased K/radius or automatic
follow-on is authorized.
