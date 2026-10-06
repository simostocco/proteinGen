# V8 interpretation and validation

Classification: **K8-D under the retained strict per-example binary inversion gate**.
Both arms converge60/60 under the unchanged V7 stopping/KKT criteria. Each
condition converges20/20 in each arm. All60 Arm B continuous safety inequalities
pass, final chirality assessability is preserved, and condition450 gains8.571851%.
The condition450 Delta_K is +4.344802 percentage points for Arm B (+4.363727 for
Arm A). Doubling K alone crosses the desired5% continuous safe-repair target.

The strict binary gate still fails on nine Arm B examples: five at condition50,
two at250, two at450. Aggregate inversion changes are -37/-158/-105 (overall
-300). Therefore the failure is **binary-gate incompleteness**, not an inability
of the continuous constrained arm to reach5%. K8-D must not be read as evidence
that the aligned/continuous-chirality constraints prohibit5% repair. The explicit
interpretation policy was frozen before scientific execution and retains V7's
per-example gate. No looser aggregate-only acceptance was silently substituted.
No causal claim of unavoidable binary/geometry trade-off follows from these
solutions; a failed gate is not a proof that no gate-passing solution exists.

The aligned safety constraint is active0/60 and continuous chirality43/60
(4/20,20/20,19/20). Continuous constraint cost at450 is only.028780 percentage
points compared with Arm A. The remaining issue is handedness gate handling,
not insufficient mobility for the5% threshold. **Increasing s_max is not justified
by this diagnostic for that target.** No radius or additional K has been tested.

All eight steps saturate >=90/95/99% of.04 Å on100% of eligible residues,
matching V7's four-step saturation. Consecutive correction cosines equal1 within
floating-point roundoff at every adjacent step and condition; no oscillation is
observed. Net/path RMS are.316650 Å, maximum<.320 Å, and individual steps<.040 Å.
Net equals path because the free Cartesian corrections are collinear across
steps. Shared final-state supervision, symmetric zero initialization and unchanged
eligibility permit identical free-variable updates across steps. Thus the oracle
shows an expanded reachable displacement budget, not evidence that a learned
model performs progressively different geometry-aware repairs.

All states remain finite without collapse. Eligibility is preserved at every
state with zero degenerate frames; no transient or final chirality-assessability
losses occur. Architectural loose bounds .32 Å net/path and .64 Å pair-distance
change remain worst-case guarantees, not the reported expected movement.

The synthetic preflight retained a strictly nonconverged xtol case with small
physical KKT residual; no cap/tolerance changed. All scientific solves satisfy
the full historical criterion. Numerical validation checks180 baseline aligned
finite-difference directions,60 zero-state objective/constraint HVPs, and123
focused CPU tests (two expected inherited infeasible-case warnings). All120
scientific arm/example runs are replayed from zero with exact non-timing record
parity before commit. Protected historical/input hashes and saved float64
coordinate/correction hashes are checked after execution. State caches stay
outside Git. No CUDA, neural training, E010 mutation, held-out data or environment
change occurs.

Recommended next experiment: pre-register a K=8/s_max=.04 Å local-feasibility
oracle with explicit no-new-inversion signed-quartet constraints for quartets
correct at Pg, retaining the continuous safety constraints, to test whether
>=5% repair survives a guaranteed per-example binary inversion gate. This
changes safety handling in a separate experiment; it is not launched in V8.

## Arm B per-example inversion telemetry

| Index | Identity | Condition | Baseline | Final | Change | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1mhp_B | 50 | 46 | 44 | -2 | pass |
| 1 | 1mhp_B | 250 | 86 | 80 | -6 | pass |
| 2 | 1mhp_B | 450 | 101 | 100 | -1 | pass |
| 3 | 1qqn_A | 50 | 97 | 90 | -7 | pass |
| 4 | 1qqn_A | 250 | 190 | 182 | -8 | pass |
| 5 | 1qqn_A | 450 | 180 | 167 | -13 | pass |
| 6 | 1t8o_B | 50 | 24 | 25 | +1 | FAIL |
| 7 | 1t8o_B | 250 | 22 | 25 | +3 | FAIL |
| 8 | 1t8o_B | 450 | 19 | 18 | -1 | pass |
| 9 | 2jo5_A | 50 | 1 | 1 | +0 | pass |
| 10 | 2jo5_A | 250 | 8 | 9 | +1 | FAIL |
| 11 | 2jo5_A | 450 | 9 | 10 | +1 | FAIL |
| 12 | 2mdw_A | 50 | 8 | 8 | +0 | pass |
| 13 | 2mdw_A | 250 | 10 | 8 | -2 | pass |
| 14 | 2mdw_A | 450 | 10 | 8 | -2 | pass |
| 15 | 3qv9_A | 50 | 145 | 145 | +0 | pass |
| 16 | 3qv9_A | 250 | 245 | 232 | -13 | pass |
| 17 | 3qv9_A | 450 | 256 | 245 | -11 | pass |
| 18 | 5avn_A | 50 | 89 | 82 | -7 | pass |
| 19 | 5avn_A | 250 | 178 | 172 | -6 | pass |
| 20 | 5avn_A | 450 | 184 | 176 | -8 | pass |
| 21 | 5fds_A | 50 | 28 | 31 | +3 | FAIL |
| 22 | 5fds_A | 250 | 68 | 61 | -7 | pass |
| 23 | 5fds_A | 450 | 55 | 47 | -8 | pass |
| 24 | 5x1e_A | 50 | 22 | 22 | +0 | pass |
| 25 | 5x1e_A | 250 | 53 | 49 | -4 | pass |
| 26 | 5x1e_A | 450 | 46 | 42 | -4 | pass |
| 27 | 6szs_z | 50 | 73 | 69 | -4 | pass |
| 28 | 6szs_z | 250 | 185 | 175 | -10 | pass |
| 29 | 6szs_z | 450 | 185 | 178 | -7 | pass |
| 30 | 6tzk_A | 50 | 165 | 156 | -9 | pass |
| 31 | 6tzk_A | 250 | 216 | 201 | -15 | pass |
| 32 | 6tzk_A | 450 | 230 | 224 | -6 | pass |
| 33 | 8cmd_A | 50 | 60 | 60 | +0 | pass |
| 34 | 8cmd_A | 250 | 87 | 79 | -8 | pass |
| 35 | 8cmd_A | 450 | 85 | 76 | -9 | pass |
| 36 | 8cqx_A | 50 | 70 | 73 | +3 | FAIL |
| 37 | 8cqx_A | 250 | 143 | 134 | -9 | pass |
| 38 | 8cqx_A | 450 | 144 | 147 | +3 | FAIL |
| 39 | 8fmw_K | 50 | 34 | 37 | +3 | FAIL |
| 40 | 8fmw_K | 250 | 60 | 51 | -9 | pass |
| 41 | 8fmw_K | 450 | 49 | 48 | -1 | pass |
| 42 | 9cfg_H | 50 | 49 | 42 | -7 | pass |
| 43 | 9cfg_H | 250 | 57 | 49 | -8 | pass |
| 44 | 9cfg_H | 450 | 60 | 58 | -2 | pass |
| 45 | 9k3q_1 | 50 | 3 | 3 | +0 | pass |
| 46 | 9k3q_1 | 250 | 16 | 16 | +0 | pass |
| 47 | 9k3q_1 | 450 | 21 | 17 | -4 | pass |
| 48 | 9m6h_B | 50 | 84 | 82 | -2 | pass |
| 49 | 9m6h_B | 250 | 200 | 178 | -22 | pass |
| 50 | 9m6h_B | 450 | 208 | 198 | -10 | pass |
| 51 | 9pbc_A | 50 | 91 | 92 | +1 | FAIL |
| 52 | 9pbc_A | 250 | 155 | 135 | -20 | pass |
| 53 | 9pbc_A | 450 | 156 | 146 | -10 | pass |
| 54 | 9q1s_DD | 50 | 59 | 55 | -4 | pass |
| 55 | 9q1s_DD | 250 | 101 | 90 | -11 | pass |
| 56 | 9q1s_DD | 450 | 103 | 98 | -5 | pass |
| 57 | 9v0p_F | 50 | 48 | 42 | -6 | pass |
| 58 | 9v0p_F | 250 | 53 | 49 | -4 | pass |
| 59 | 9v0p_F | 450 | 63 | 56 | -7 | pass |
