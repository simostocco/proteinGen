# E010 Phase 4D v6 precision and correction-space KKT audit

Classification: **PREC-C**. Historical v5 CART-O5 remains unchanged.

## Interpretation and complete reproduction

The fixed-state audit and optional precision replay both reproduced exactly:
22/22 audit records, and 16/16 fresh-zero replay states, objectives, metrics,
histories and stopping results (excluding runtime). The focused CPU suite
passed 103 tests; Ruff passed. All protected historical/input hashes match.
No historical result was rewritten, and no E010/neural model was constructed.

The historical Cartesian prediction.float() cast quantizes objective values
used by strong-Wolfe search. At the same frozen states, directions, epsilons
and tolerances, historical arithmetic fails 75/96 directional comparisons;
pure float64 passes 132/132. Six of the 396 individual float64 epsilon checks
fail, but every direction has a passing scale under the frozen mixed error
tolerance. The large maximum relative error occurs for tiny derivatives;
maximum absolute error is 4.21e-9. This supports arithmetic precision as the
numerical bottleneck, without diagnosing a defect in the mathematical gradient.

All 16 float64 replays converge: condition 50 6/6, 250 6/6, 450 4/4.
Iterations min/median/max are 70/155/280. Maximum historical projected residual
after replay is 7.54e-5; maximum correction-space normalized KKT residual is
1.52e-4, below the frozen .001 threshold. Historical fixed stalled-state KKT
maxima range from .002024 to 1.0; controls range from .000142 to .000919.
The value 1.0 reflects interior corrections under the strict near-boundary
tolerance, not an inference that all of their gradient is feasible at the
boundary. No predeclared shadow step produced a material objective decrease.

The radial map suppresses parameter gradients in both groups. Pooled median
condition numbers are 1.95e6 for stalls and 1.29e7 for controls; median radial
eigenvalues are 3.66e-10 and 2.14e-11. Thus controls are more ill-conditioned,
not less. Near-boundary saturation alone does not identify the stall mechanism.
Per-variable, step and condition distributions remain available in JSON.

Scientific response changes little. Substituting the 16 precision-replayed
stalls into the 44 unchanged historical outputs gives local gains of
5.067276%, .511176% and .066354% at conditions 50/250/450. This is an assembled
panel, not a uniform 60-example float64 rerun. Local/Cartesian/aligned metric
changes remain below the predeclared scientific relative materiality threshold.
Two continuous chiral differences cross that threshold: example 11 improves
by .10576%, while example 42 worsens by .04318%. Example 28 gains one inversion
relative to its historical oracle output; aggregate assessability is unchanged.
These differences are retained in precision_replay_metric_comparison.json.

Condition 450 therefore remains around .066% under the fixed weighted objective.
Improved numerical convergence does not demonstrate the maximum achievable
local repair under safety constraints and does not establish bound insufficiency.

Recommended next experiment: pre-register a float64 constrained local-feasibility
oracle retaining K=4 and s_max=.04 Å, minimizing local error subject to explicit
Cartesian/chirality safety limits. No follow-on experiment was launched.

All sixteen historical stalls and six pre-registered strongly-converged controls were recovered by the exact historical solver with mixed historical arithmetic. Final/state hashes, metrics, convergence status, iterations, closure evaluations and complete recorded history matched exactly. Recovery is the explicitly authorized state-reconstruction exception, not a replacement scientific result. Recovered tensors remain outside Git.

Fixed scientific settings: K=4, s_max=.04 Å, beta=16.8, gamma=2, original panel/Pg/targets/masks, current geometric correction eligibility and frozen chiral quartets. No neural model or CUDA. Numerical tolerances, examples, shadow scales and classification rules were frozen before recovery/audit.

For delta=a*v, J=a*I-a³*v*v^T/s²: radial eigenvalue a³, tangential a, condition number a^-2. Physical variables are v=.04*z. The correction-space gradient is taken before the radial Jacobian, and the analytic chain is tested against autograd.

At active balls, lambda=max(0,-g·delta/(2||delta||²)); residual g+2*lambda*delta retains tangential and feasible inward descent. Interior residual is g. Near-boundary tolerance is 1e-6 relative; normalized KKT threshold .001. This is distinct from the historical v-space gradient and projected finite-step mapping.

Maximum absolute/relative objective precision differences: 7.7720216e-08/6.74552657e-08. Material objective cases: []. Maximum relative gradient difference 0.00140082149; minimum cosine 0.999999095651. Material gradient cases [9].

Finite-difference maximum/median relative error: 0.354198671/5.82802916e-07. All individual scales/errors are retained; validation requires at least one passing epsilon per fixed direction/space under mixed absolute+relative tolerances. Failed directions: 0; failed analytic chain checks: 0.

Near-KKT stalled examples: []. Material single shadow-step decreases: []. Shadow perturbations are not iterated and keep the original correction balls; eligibility changes invalidate comparisons.

## Per-example audit

| Index | Identity | Condition | Group | g_v norm | g_delta norm | Normalized KKT max | Active fraction | Median kappa | Median radial eigenvalue | Relative gradient precision difference | Shadow material |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 1qqn_A | 250 | stall | 3.63012276e-08 | 0.140375024 | 0.00212397228 | 1 | 2646944.91 | 2.32045517e-10 | 0.000143145732 | False |
| 11 | 2jo5_A | 450 | stall | 3.87934431e-06 | 0.654321681 | 1 | 0 | 307651.508 | 5.15239568e-09 | 4.00803003e-06 | False |
| 18 | 5avn_A | 50 | stall | 1.64103959e-08 | 0.0470551715 | 0.00332668379 | 1 | 2777923.64 | 2.15982988e-10 | 0.000366038919 | False |
| 19 | 5avn_A | 250 | stall | 3.14339003e-08 | 0.112200783 | 0.0867418198 | 0.979220779 | 1166512.29 | 7.93717625e-10 | 0.000295176619 | False |
| 21 | 5fds_A | 50 | stall | 6.91079117e-08 | 0.0748551669 | 0.0367878838 | 0.992248062 | 1021411.9 | 9.68720815e-10 | 0.000124375239 | False |
| 27 | 6szs_z | 50 | stall | 2.11874031e-08 | 0.0455285288 | 0.00345905742 | 0.997222222 | 2871302.15 | 2.05430816e-10 | 0.000350598457 | False |
| 28 | 6szs_z | 250 | stall | 6.89712657e-08 | 0.122316201 | 0.00351500762 | 1 | 1708394.55 | 4.47726123e-10 | 0.000136252359 | False |
| 30 | 6tzk_A | 50 | stall | 3.46443566e-08 | 0.0401047447 | 0.0200752658 | 0.993318486 | 1581760.67 | 5.02676801e-10 | 0.000234648969 | False |
| 34 | 8cmd_A | 250 | stall | 4.59045375e-08 | 0.182148224 | 0.00212432651 | 1 | 4983774.35 | 8.98798718e-11 | 0.00010523011 | False |
| 35 | 8cmd_A | 450 | stall | 5.37204329e-08 | 0.44906976 | 0.0020681786 | 1 | 13431808.5 | 2.03141318e-11 | 3.92839911e-05 | False |
| 36 | 8cqx_A | 50 | stall | 1.49491344e-08 | 0.0530801802 | 0.00202403711 | 1 | 2514568.57 | 2.50709842e-10 | 0.000366484993 | False |
| 42 | 9cfg_H | 50 | stall | 1.18904927e-06 | 0.0702657729 | 1 | 0 | 92204.0355 | 3.57170109e-08 | 2.55524476e-05 | False |
| 49 | 9m6h_B | 250 | stall | 1.14984251e-07 | 0.129746801 | 0.0255700504 | 0.997493734 | 1313927.01 | 6.63961868e-10 | 0.000103601189 | False |
| 50 | 9m6h_B | 450 | stall | 3.10074159e-07 | 0.466829415 | 0.00606138493 | 1 | 1862140.73 | 3.93533085e-10 | 1.99345462e-05 | False |
| 53 | 9pbc_A | 450 | stall | 1.22266418e-07 | 0.393502528 | 0.00327007441 | 1 | 4244690.93 | 1.1434861e-10 | 3.65438215e-05 | False |
| 58 | 9v0p_F | 250 | stall | 1.15983068e-07 | 0.263511326 | 0.00236952939 | 1 | 4935402.35 | 9.05028111e-11 | 3.58184252e-05 | False |
| 2 | 1mhp_B | 450 | control | 6.29062868e-09 | 0.351253101 | 0.000240327325 | 1 | 16897093.7 | 1.43747074e-11 | 0.000257107939 | False |
| 6 | 1t8o_B | 50 | control | 3.26438327e-08 | 0.101098481 | 0.000918612084 | 1 | 1294816.74 | 6.68913432e-10 | 0.000305888621 | False |
| 8 | 1t8o_B | 450 | control | 3.97460638e-08 | 0.531554925 | 0.000684228428 | 1 | 12443307.7 | 2.24806087e-11 | 6.33907395e-05 | False |
| 9 | 2jo5_A | 50 | control | 2.86332821e-09 | 0.169669685 | 0.00014193077 | 1 | 15775149.6 | 1.57957786e-11 | 0.00140082149 | False |
| 10 | 2jo5_A | 250 | control | 6.96076525e-08 | 0.399925855 | 0.000431119181 | 1 | 1531794.71 | 5.0618524e-10 | 7.29948576e-05 | False |
| 25 | 5x1e_A | 250 | control | 1.54229937e-08 | 0.240915839 | 0.000669260984 | 1 | 11362426.3 | 2.61059821e-11 | 0.00018869097 | False |

## Group/step distributions

Each row is one recurrent step, with pooled eligible-residue statistics. Full quantiles for all requested quantities, per-variable columns (including fixed endpoints), metric differences and shadow results are retained in JSON.

| Group | Step | Median kappa | Max kappa | Median radial eigenvalue | Median tangential eigenvalue | Median saturation | Max KKT residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stalled | 0 | 1952165.36 | 76717198.4 | 3.66397796e-10 | 0.000715568066 | 0.999999743874 | 0.136497818 |
| stalled | 1 | 1952165.36 | 76717198.4 | 3.66397796e-10 | 0.000715568066 | 0.999999743874 | 0.136497818 |
| stalled | 2 | 1952165.36 | 76717198.4 | 3.66397796e-10 | 0.000715568066 | 0.999999743874 | 0.136497818 |
| stalled | 3 | 1952165.36 | 76717198.4 | 3.66397796e-10 | 0.000715568066 | 0.999999743874 | 0.136497818 |
| control | 0 | 12948513.1 | 127061068 | 2.1396218e-11 | 0.000277616754 | 0.999999961386 | 4.40468567e-05 |
| control | 1 | 12948513.1 | 127061068 | 2.1396218e-11 | 0.000277616754 | 0.999999961386 | 4.40468567e-05 |
| control | 2 | 12948513.1 | 127061068 | 2.1396218e-11 | 0.000277616754 | 0.999999961386 | 4.40468567e-05 |
| control | 3 | 12948513.1 | 127061068 | 2.1396218e-11 | 0.000277616754 | 0.999999961386 | 4.40468567e-05 |
| stalled_condition_50 | 0 | 2220117.16 | 59854110.9 | 3.02029793e-10 | 0.000670939347 | 0.999999774787 | 0.00622950437 |
| stalled_condition_50 | 1 | 2220117.16 | 59854110.9 | 3.02029793e-10 | 0.000670939347 | 0.999999774787 | 0.00622950437 |
| stalled_condition_50 | 2 | 2220117.16 | 59854110.9 | 3.02029793e-10 | 0.000670939347 | 0.999999774787 | 0.00622950437 |
| stalled_condition_50 | 3 | 2220117.16 | 59854110.9 | 3.02029793e-10 | 0.000670939347 | 0.999999774787 | 0.00622950437 |
| stalled_condition_250 | 0 | 1554018.06 | 63849196.5 | 5.16197514e-10 | 0.000802180258 | 0.999999678253 | 0.000517724681 |
| stalled_condition_250 | 1 | 1554018.06 | 63849196.5 | 5.16197514e-10 | 0.000802180258 | 0.999999678253 | 0.000517724681 |
| stalled_condition_250 | 2 | 1554018.06 | 63849196.5 | 5.16197514e-10 | 0.000802180258 | 0.999999678253 | 0.000517724681 |
| stalled_condition_250 | 3 | 1554018.06 | 63849196.5 | 5.16197514e-10 | 0.000802180258 | 0.999999678253 | 0.000517724681 |
| stalled_condition_450 | 0 | 3678288.47 | 76717198.4 | 1.41752694e-10 | 0.000521407299 | 0.999999864067 | 0.136497818 |
| stalled_condition_450 | 1 | 3678288.47 | 76717198.4 | 1.41752694e-10 | 0.000521407299 | 0.999999864067 | 0.136497818 |
| stalled_condition_450 | 2 | 3678288.47 | 76717198.4 | 1.41752694e-10 | 0.000521407299 | 0.999999864067 | 0.136497818 |
| stalled_condition_450 | 3 | 3678288.47 | 76717198.4 | 1.41752694e-10 | 0.000521407299 | 0.999999864067 | 0.136497818 |
| control_condition_50 | 0 | 1349516.74 | 32658133.5 | 6.36333991e-10 | 0.00086012527 | 0.999999629497 | 1.18550291e-05 |
| control_condition_50 | 1 | 1349516.74 | 32658133.5 | 6.36333991e-10 | 0.00086012527 | 0.999999629497 | 1.18550291e-05 |
| control_condition_50 | 2 | 1349516.74 | 32658133.5 | 6.36333991e-10 | 0.00086012527 | 0.999999629497 | 1.18550291e-05 |
| control_condition_50 | 3 | 1349516.74 | 32658133.5 | 6.36333991e-10 | 0.00086012527 | 0.999999629497 | 1.18550291e-05 |
| control_condition_250 | 0 | 10809326.9 | 62404372.7 | 2.79863054e-11 | 0.000303609383 | 0.999999953744 | 3.66373194e-05 |
| control_condition_250 | 1 | 10809326.9 | 62404372.7 | 2.79863054e-11 | 0.000303609383 | 0.999999953744 | 3.66373194e-05 |
| control_condition_250 | 2 | 10809326.9 | 62404372.7 | 2.79863054e-11 | 0.000303609383 | 0.999999953744 | 3.66373194e-05 |
| control_condition_250 | 3 | 10809326.9 | 62404372.7 | 2.79863054e-11 | 0.000303609383 | 0.999999953744 | 3.66373194e-05 |
| control_condition_450 | 0 | 16439432.7 | 127061068 | 1.49747333e-11 | 0.000246482656 | 0.999999969585 | 4.40468567e-05 |
| control_condition_450 | 1 | 16439432.7 | 127061068 | 1.49747333e-11 | 0.000246482656 | 0.999999969585 | 4.40468567e-05 |
| control_condition_450 | 2 | 16439432.7 | 127061068 | 1.49747333e-11 | 0.000246482656 | 0.999999969585 | 4.40468567e-05 |
| control_condition_450 | 3 | 16439432.7 | 127061068 | 1.49747333e-11 | 0.000246482656 | 0.999999969585 | 4.40468567e-05 |

## Optional precision replay

{
  "by_condition": {
    "250": {
      "converged": 6,
      "examples": 6
    },
    "450": {
      "converged": 4,
      "examples": 4
    },
    "50": {
      "converged": 6,
      "examples": 6
    }
  },
  "converged": 16,
  "cuda_used": false,
  "examples": 16,
  "neural_training_launched": false,
  "protected_hashes_checked": true
}

## Scientific limit

Even a fully converged minimum of L_local+16.8*L_cart+2*L_chiral does not determine maximum local repair subject to Cartesian/chirality safety. Weighted-sum optimization and constrained local feasibility are different questions. Condition-450 historical gain remains 0.066314%; this audit cannot alone establish correction-budget insufficiency.
