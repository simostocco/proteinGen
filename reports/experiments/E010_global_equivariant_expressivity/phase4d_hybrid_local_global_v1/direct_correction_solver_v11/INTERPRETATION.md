# V11 adjudication and practical limits

**DIRECT-D — no material numerical benefit in the preregistered execution.**
All 60 cases ran exactly once from zero, CPU float64. All final trajectories,
metrics, physical certificates and convergence labels were independently
reproduced. Aggregate JSON/Markdown reproduced byte-for-byte. Historical artifacts
and protected inputs remain unchanged. Scientific oracle execution consumes cached
Pg without running or updating E010. No CUDA or neural training occurred.

The linear optimizer-to-correction Jacobian is 0.04 I, condition number 1; no radial
map is called. Ball/scientific derivatives passed validation. Exact ball Hessians
are included. All final corrections are strictly inside 0.04 Å, so no exact-boundary
certificate bookkeeping exception was used in the panel.

Physical convergence is **4/60**, versus historical V10 **12/60**. Passing cases:
2jo5_A conditions 50/250/450, and 9k3q_1 condition 50. The other 56 reached the
1,000-iteration cap (SciPy status 0). Stationarity fails in 56, material feasible
shadow descent in 55, dual feasibility in 49, complementarity in 24. All primal and
derivative gates pass. Material-shadow cases increased from V10's 35 to 55;
median physical stationarity increased from approximately 0.003226 to 0.004852.
This does not certify the complete historical feasibility target.

**Budget caveat:** the user explicitly required V11 maximum 1000 iterations;
historical V10 used 2000. All other numerical options were preserved. This is the
requested V11-versus-recorded-V10 comparison, not an equal-cap causal estimate.
The result does not prove that the radial map was irrelevant, or that the direct
problem would never converge. No larger-budget replay was run.

Descriptive condition gains remain almost identical: V11 19.1552%, 8.3586%,
8.4865%, versus V10 19.1349%, 8.3924%, 8.4651% for 50/250/450. Condition-450
offset gains are 11.9351%, 9.0521%, 6.6298%; aligned RMSD changes +0.1368%.
All continuous and strict endpoint constraints pass. Inversions 5493 -> 5492:
one repaired, zero new, zero per-example failures. Assessability 13029 -> 13029,
with no temporary loss. No collapse or degenerate eligible frames.

Active physical balls: 1336; active baseline-correct signed inequalities: zero;
unsigned assessability: 16; continuous chirality: 13; aligned RMSD: zero. No
bond/frame assessability inequalities are active. Minimum signed/absolute-q margin
above the evaluator epsilon is 2.069503e-6; minimum frame margin 2.082985e-4;
minimum bond margin 0.171980 Å. Activity uses unchanged V9B tolerances.

Step RMS is approximately 0.039280 Å throughout P1–P8. Net/path RMS approximately
0.314241 Å, maximum 0.319999889 Å. Maximum step 0.0399999861 Å. Consecutive-step
mean cosines exceed 0.99999989 in every condition, with no negative cosines.
Trajectories continue almost coherently; they do not oscillate. Full P0–P8 metrics,
all five length strata, constraints and shadow effects are in result.json/B records.

The four converged-both solutions have Cartesian coordinate RMS differences
0.000103–0.000301 Å from V10, with the same inversion counts. Nonconvex solutions
and active sets need not be exactly identical.

Preflight: length-12 direct passes in 20 iterations (deterministic repeat exact),
historical radial passes in 39. Length-32 direct remains nonconverged at 1000,
stationarity 0.004522 and dual negativity 5.8425e-7; historical recovered radial
state is physically converged after 1807. No preflight outcome changed settings.

Validation: 26 focused tests pass; broader E010 suite 241 pass/14 inherited
FileNotFoundError failures for unavailable Phase4A artifacts. Historical test bytes
are unchanged. Ruff passes. Private NPZ solver states are excluded from Git.

Exactly one recommended next experiment: a fixed-state trust-constr barrier and
physical-KKT conditioning audit of the saved nonconverged V11 endpoints, preserving
the same feasible set. No follow-on experiment has been launched.
