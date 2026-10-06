# V7 interpretation and numerical limitations

The frozen decision rule gives **FEAS-C**. Arm A converged on all 60 examples,
including all 20 condition-450 examples. Its condition-450 mean-local gain is
4.236904%, below the unchanged 5% threshold. Arm B converged on 59/60 examples
(20/20, 20/20, 19/20 at conditions 50/250/450), exceeding the preregistered
strong-coverage requirements. The sole nonconverged result is index 32,
6tzk_A at condition 450: 1,000 iterations, correction-space normalized ball KKT
5.04301e-5. It remains nonconverged under the original strict stopping rule.
No retries or changed settings were used.

This is direct evidence that the present geometric budget misses the high-noise
repair target in this converged local-only diagnostic. These are nonconvex
local-stationarity results, not a certificate of the global maximum achievable
repair. The conclusion is stronger than an inference from the old weighted
objective or from bound saturation alone.

Historical scalarization was also limiting: at condition 450, removing the
weighted auxiliary objectives increases local repair from about 0.06635% to
4.23690%. Explicit continuous safety constraints retain 4.22705%, a cost of
only 0.00986 percentage points relative to Arm A. This is still below 5%.
Both arms improve all three offsets in all three conditions.

Arm B satisfies both continuous inequalities on every example within the
frozen normalized 1e-8 tolerance. The Cartesian inequality is active on 0/60
examples; the chirality inequality is active on 42/60 (3/20, 20/20, 19/20 by
condition). Assessability is preserved. Aggregate inversion changes are
-33/-72/-52 by condition, but nine examples increase their own inversion
counts (4/2/3 by condition). Therefore Arm B does **not** pass every binary
scientific gate. Continuous chirality safety cannot guarantee binary safety;
no binary metric entered optimization.

All eligible corrections are near the radial boundary under the fixed
relative 1e-4 definition. Net and path RMS are about 0.158325 Å; maximum path
and net displacement remain below 0.16 Å. Individual steps remain below
0.04 Å. Frame eligibility is preserved and no frame degeneracy or numerical
collapse appears. Final chirality assessability is preserved on every example.
Both arms temporarily lose one assessable quartet at P3 for index 54,
9q1s_DD/condition 50; it is restored at P4. Full intermediate states,
condition/length metrics, histories and residuals are recorded in the JSON.

Two implementation/output issues are preserved transparently. The first
attempt stopped before an example result because the historical norm-based
radial expression had an undefined intermediate zero-state Hessian. The same
mathematical map was rewritten using squared norm. The second attempt completed
all 120 numerical solutions but failed JSON serialization of a NumPy boolean.
Only its output conversion changed. Every regenerated variable and coordinate
state is required to match that archived computation exactly. Config bytes are
identical across both attempts and the final execution; no result-driven
numerical tuning occurred. Neither correction modifies any historical source
or result.

Validation evidence is in `aligned_gradient_preflight.json`,
`reproduction_complete.json`, and `post_execution_validation.json`. The full
120-example/arm execution is independently replayed from zero with exact
non-timing record equality before commit. Focused CPU tests pass 118/118; two
expected SciPy warnings arise from the deliberately infeasible synthetic case.
Source and protected input hashes are checked before and after execution.
Coordinate/variable caches remain outside Git.

Recommended next experiment: pre-register the same two-arm float64 oracle
with **K=8 and s_max=0.04 Å**, keeping data, safety constraints and numerical
settings fixed, to test a longer geometric horizon. Do not launch it as part
of V7.
