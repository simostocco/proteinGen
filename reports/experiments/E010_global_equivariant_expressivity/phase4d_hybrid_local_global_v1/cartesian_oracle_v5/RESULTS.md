# E010 Phase 4D v5 Cartesian oracle cross-check

**Classification: CART-O5.** Cartesian convergence: 44/60 versus v4 26/60. All 60 fresh-zero repeats reproduce every non-timing metric, coordinate digest, objective/convergence history and stopping result exactly. No neural training or CUDA use.

Historical v4 commit `f9b42c36459f17c9f7fb42c20496b7c7b58a4ea6` is unchanged. Only the free correction coordinate basis changed. K=4, s_max=.04 Å, beta=16.8, gamma=2, original 60-example panel/targets/Pg/masks, final-state loss reductions, signed chirality and eligibility remain fixed. The exact same v4 solver code object is bound to a Cartesian trajectory through an isolated globals copy; historical module globals are untouched.

Physical v=.04*z uses the same dimensionless variable scale as v4. Corrections are v/sqrt(1+||v||²/.04²), with the same inward rounding safeguard and current geometric eligibility mask. Cartesian variables consume no frame features. Frames are computed for the identical eligibility check and telemetry. ||F u||=||u|| at an orthonormal eligible frame, so the per-step correction ball is unchanged; moving-basis/recurrent derivatives and eligibility dependencies prevent claiming identical complete optimization problems.

Optimizer exactly preserved: deterministic CPU L-BFGS LR=1, history=20, strong-Wolfe search, up to 1000 outer iterations, zero initialization, no RNG, same precision and all convergence tolerances. Four one-thread workers. No fallback or explicit restarts. Failure counts are not exposed by the unchanged v4 solver; closure evaluation counts are available per example.

Iterations min/median/max: 50/60/1000. Full-panel objective: 600.11135623 → 586.85567737. Runtime summed across examples: 2524.810 seconds; this is not parallel wall time.

## Interpretation

The Cartesian parameterization improves convergence from 26/60 to 44/60, but
falls below the predeclared 48/60 overall and 18/20 condition-450 thresholds
(condition 450 reaches 16/20). All 16 remaining examples exhaust the unchanged
1000-iteration budget; their final projected residuals range from 0.00105918
to 0.01778262 (median 0.00184916). Classification remains **CART-O5 — STILL
INCONCLUSIVE**. Correction saturation and the 0.06631% condition-450 gain do
not establish bound insufficiency or maximum achievable safe repair.

Condition conclusions barely change relative to v4 despite improved numerical
coverage. Among 20 converged-both examples, 14 satisfy all metric-equivalence
tolerances and none has a materially lower v5 objective. Among 24 newly
converged examples, 14 are equivalent and one has a materially lower objective;
the matched condition conclusions remain unchanged. Cartesian variables resolve
some conditioning problems, but the experiment does not isolate the cause of
the remaining stalls or separate correction-budget from objective limitations.

All outputs are finite, no collapse occurs, and frame eligibility is preserved
at every recorded state. The maximum step correction is 0.0399999999441 Å and
cumulative maximum is 0.1599999997763 Å. Final chirality assessability is
preserved, with a transient loss of one assessable quartet at an intermediate
state; this telemetry is retained rather than hidden by final-state aggregation.

Recommended next experiment: pre-register a numerical loss/gradient precision
audit of the 16 remaining Cartesian-oracle stalls, retaining the correction
budget and objective coefficients. The inherited Cartesian float32 cast is
an audit target, not an established explanation. No follow-on was executed.

## Convergence coverage and paired transitions

| Group | Examples | V4 converged | V5 converged | Non→conv | Conv→non | Both conv | Both non |
| --- | --- | --- | --- | --- | --- | --- | --- |
| condition_250 | 20 | 8 | 14 | 8 | 2 | 6 | 4 |
| condition_450 | 20 | 10 | 16 | 8 | 2 | 8 | 2 |
| condition_50 | 20 | 8 | 14 | 8 | 2 | 6 | 4 |
| overall | 60 | 26 | 44 | 24 | 6 | 20 | 10 |
| stratum_129-256 | 12 | 5 | 9 | 6 | 2 | 3 | 1 |
| stratum_20-64 | 12 | 9 | 11 | 3 | 1 | 8 | 0 |
| stratum_257-384 | 12 | 3 | 7 | 6 | 2 | 1 | 3 |
| stratum_385-500 | 12 | 1 | 7 | 6 | 0 | 1 | 5 |
| stratum_65-128 | 12 | 8 | 10 | 3 | 1 | 7 | 1 |

## Metric changes

| overall | Local gain % | Offset gains % (1/2/3) | Raw cart change % | Aligned change % | Chiral loss change % | Inversion change | Assessable final/base | Correction RMS/max Å | Eligible/degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 1.04380 | 1.00342/0.99393/1.10568 | -2.22361 | -1.51495 | -5.61238 | -311 | 13029/13029 | 0.158325/0.160000000 | 13089/0 |

| by_condition | Local gain % | Offset gains % (1/2/3) | Raw cart change % | Aligned change % | Chiral loss change % | Inversion change | Assessable final/base | Correction RMS/max Å | Eligible/degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250 | 0.51118 | 0.08243/0.39948/0.85884 | -4.06982 | -2.05248 | -4.87568 | -69 | 4343/4343 | 0.158325/0.159999999 | 4363/0 |
| 450 | 0.06631 | 0.06911/0.04825/0.07850 | -1.70703 | -0.58445 | -1.80934 | -27 | 4343/4343 | 0.158325/0.160000000 | 4363/0 |
| 50 | 5.06709 | 4.25165/5.00778/5.83845 | -11.70297 | -6.09930 | -18.24242 | -215 | 4343/4343 | 0.158325/0.159999999 | 4363/0 |

| by_stratum | Local gain % | Offset gains % (1/2/3) | Raw cart change % | Aligned change % | Chiral loss change % | Inversion change | Assessable final/base | Correction RMS/max Å | Eligible/degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 129-256 | 1.03373 | 0.86904/0.93911/1.19117 | -2.54707 | -1.61366 | -3.78662 | -45 | 2115/2115 | 0.159075/0.159999999 | 2127/0 |
| 20-64 | 1.01843 | 1.15708/0.98548/0.96262 | -2.54320 | -1.21819 | -3.98557 | -11 | 411/411 | 0.154822/0.159999999 | 423/0 |
| 257-384 | 1.02413 | 0.94596/0.98141/1.10025 | -1.99847 | -1.43613 | -5.94391 | -96 | 3999/3999 | 0.159518/0.159999999 | 4011/0 |
| 385-500 | 1.06781 | 1.06992/0.99872/1.12238 | -1.80908 | -1.46770 | -6.30819 | -125 | 5178/5178 | 0.159628/0.160000000 | 5190/0 |
| 65-128 | 1.07374 | 0.95290/1.05957/1.14373 | -2.81139 | -1.82485 | -5.06566 | -34 | 1326/1326 | 0.158583/0.160000000 | 1338/0 |

## Absolute metric values

| Group | State | Offset RMSE Å | Mean local Å | Raw Cartesian Å² | Aligned RMSD Å | Chiral loss | Inversions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | Pg | 1.895241/2.874420/3.553660 | 2.774440 | 35.077882 | 7.858572 | 0.423339 | 5493 |
| overall | v5 | 1.876224/2.845850/3.514368 | 2.745481 | 34.297887 | 7.739518 | 0.399580 | 5182 |
| 50 | Pg | 1.263806/1.442583/1.446908 | 1.384432 | 1.762931 | 2.245481 | 0.199807 | 1196 |
| 50 | v5 | 1.210074/1.370341/1.362431 | 1.314282 | 1.556616 | 2.108522 | 0.163358 | 981 |
| 250 | Pg | 1.976871/2.848015/3.353130 | 2.726006 | 15.549012 | 6.507821 | 0.504340 | 2133 |
| 250 | v5 | 1.975241/2.836638/3.324332 | 2.712071 | 14.916195 | 6.374249 | 0.479750 | 2064 |
| 450 | Pg | 2.445046/4.332661/5.860942 | 4.212883 | 87.921702 | 14.822414 | 0.565870 | 2164 |
| 450 | v5 | 2.443357/4.330570/5.856341 | 4.210089 | 86.420851 | 14.735784 | 0.555631 | 2137 |

| Condition | V5 local gain % | V4 local gain % | Neural S/M/L gain % | V5 aggregate safe |
| --- | --- | --- | --- | --- |
| 50 | 5.06709 | 5.06753 | 6.19601/5.80689/5.63495 | True |
| 250 | 0.51118 | 0.51102 | 0.46603/0.54757/0.53305 | True |
| 450 | 0.06631 | 0.06636 | 0.00693/0.00142/0.12796 | True |

## Paired converged-both and newly-converged results

converged_both: 20 examples; 14 meet all predeclared metric-equivalence tolerances; 0 have materially lower Cartesian-oracle objectives. Mean per-example objective-contribution difference: -1.2796154086336386e-08.

| Group | Condition | V4 local gain % | V5 local gain % | V4 raw Cartesian | V5 raw Cartesian | V4 chiral loss | V5 chiral loss | V4/V5 inversions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| converged_both | 250 | 0.42603 | 0.42614 | 13.971580 | 13.971581 | 0.446323 | 0.446320 | 250/250 |
| converged_both | 450 | 0.09463 | 0.09463 | 67.267623 | 67.267623 | 0.556786 | 0.556780 | 648/647 |
| converged_both | 50 | 5.46048 | 5.46064 | 1.355747 | 1.355747 | 0.189276 | 0.189278 | 99/99 |

newly_converged: 24 examples; 14 meet all predeclared metric-equivalence tolerances; 1 have materially lower Cartesian-oracle objectives. Mean per-example objective-contribution difference: -5.070905689958752e-07.

| Group | Condition | V4 local gain % | V5 local gain % | V4 raw Cartesian | V5 raw Cartesian | V4 chiral loss | V5 chiral loss | V4/V5 inversions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| newly_converged | 250 | 0.51216 | 0.51218 | 15.416929 | 15.416927 | 0.446576 | 0.446569 | 942/943 |
| newly_converged | 450 | 0.04109 | 0.04109 | 93.011935 | 93.011928 | 0.541344 | 0.541375 | 1031/1033 |
| newly_converged | 50 | 4.78420 | 4.78417 | 1.699275 | 1.699275 | 0.169807 | 0.169806 | 501/501 |

| Index | Identity | Condition | V4/V5 converged | Objective difference | Local gain difference pp | Raw Cartesian difference | Chiral loss difference | Inversion difference | Correction RMS v4/v5 Å | Equivalent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1mhp_B | 50 | False/True | 3.53943239e-08 | 0.00000608 | -1.34345026e-07 | 2.63400427e-06 | 0 | 0.15916447/0.15916444 | True |
| 1 | 1mhp_B | 250 | False/True | 4.57033673e-08 | 0.00012166 | 1.70947242e-06 | -8.86570678e-06 | 1 | 0.15916443/0.15916440 | False |
| 2 | 1mhp_B | 450 | True/True | -7.38317851e-10 | 0.00000703 | 3.22804325e-07 | -2.59530858e-06 | 0 | 0.15916448/0.15916448 | True |
| 3 | 1qqn_A | 50 | False/True | -1.13910216e-08 | 0.00000203 | 1.14313812e-06 | -5.77858569e-06 | 0 | 0.15957610/0.15957610 | True |
| 5 | 1qqn_A | 450 | True/True | 3.71808433e-08 | 0.00000070 | -6.5291033e-08 | 5.64608456e-07 | 0 | 0.15957616/0.15957615 | True |
| 6 | 1t8o_B | 50 | False/True | -5.39010836e-09 | -0.00028678 | -6.14793734e-07 | 2.05790366e-06 | 0 | 0.15721712/0.15721711 | True |
| 7 | 1t8o_B | 250 | True/True | 5.51172352e-09 | 0.00017966 | 6.73221621e-07 | 1.57243774e-05 | 0 | 0.15721714/0.15721716 | False |
| 8 | 1t8o_B | 450 | True/True | 1.53494675e-08 | -0.00004059 | -1.00667397e-06 | 1.18288258e-06 | 0 | 0.15721717/0.15721717 | True |
| 9 | 2jo5_A | 50 | True/True | 3.80154491e-09 | 0.00016853 | 3.04003104e-07 | -5.15852006e-07 | 0 | 0.15178930/0.15178932 | True |
| 10 | 2jo5_A | 250 | False/True | 2.01511341e-09 | -0.00011174 | -2.13583722e-06 | 3.75544714e-05 | 0 | 0.15178928/0.15178928 | False |
| 12 | 2mdw_A | 50 | True/True | 8.24415992e-08 | 0.00049468 | 1.24563978e-06 | 1.34570822e-05 | 0 | 0.15396001/0.15395963 | False |
| 13 | 2mdw_A | 250 | True/True | -1.89626759e-08 | 0.00042356 | 3.23579552e-06 | 6.65871758e-06 | 0 | 0.15395996/0.15396002 | False |
| 14 | 2mdw_A | 450 | True/True | -2.32939286e-08 | 0.00001780 | 5.66405355e-07 | -2.16875967e-06 | 0 | 0.15396006/0.15396006 | True |
| 15 | 3qv9_A | 50 | False/True | -2.03151063e-08 | -0.00007577 | 9.80452231e-07 | -4.632987e-06 | 0 | 0.15967896/0.15967899 | True |
| 16 | 3qv9_A | 250 | False/True | -6.85550852e-08 | -0.00003197 | -5.07569045e-07 | -2.09994134e-07 | 0 | 0.15967891/0.15967902 | True |
| 17 | 3qv9_A | 450 | False/True | -1.22863092e-07 | 0.00000171 | -4.38457424e-06 | 1.32128995e-05 | 0 | 0.15967900/0.15967901 | False |
| 20 | 5avn_A | 450 | True/True | 2.32483934e-08 | -0.00000943 | -7.15827895e-07 | 2.72188219e-06 | 0 | 0.15958602/0.15958602 | True |
| 22 | 5fds_A | 250 | False/True | 1.80766708e-08 | 0.00001584 | 2.12876795e-07 | -1.88385521e-06 | 0 | 0.15877391/0.15877389 | True |
| 23 | 5fds_A | 450 | True/True | -4.54633327e-08 | -0.00006545 | 3.01817578e-06 | -6.86270288e-05 | -1 | 0.15877390/0.15877392 | False |
| 24 | 5x1e_A | 50 | True/True | 4.75199752e-09 | -0.00000901 | -3.34285251e-07 | 5.37910322e-06 | 0 | 0.15853874/0.15853872 | True |
| 25 | 5x1e_A | 250 | True/True | -4.15036201e-08 | -0.00008854 | -1.09344011e-06 | 1.02883937e-06 | 0 | 0.15853873/0.15853877 | True |
| 26 | 5x1e_A | 450 | False/True | -3.84393495e-09 | -0.00003177 | -1.95304108e-06 | 1.96161709e-05 | 0 | 0.15853877/0.15853877 | False |
| 29 | 6szs_z | 450 | False/True | -7.96723683e-08 | 0.00001434 | 5.59059032e-07 | -3.30459246e-06 | 0 | 0.15955738/0.15955738 | True |
| 31 | 6tzk_A | 250 | False/True | -1.03030859e-07 | 0.00001081 | 6.32638394e-06 | -2.64418912e-05 | 0 | 0.15964474/0.15964482 | False |
| 32 | 6tzk_A | 450 | False/True | 1.68298442e-09 | -0.00003041 | -6.67370102e-07 | -4.6557884e-08 | 0 | 0.15964483/0.15964483 | True |
| 33 | 8cmd_A | 50 | False/True | -8.60703991e-09 | 0.00005818 | -1.91092548e-07 | 2.58701085e-06 | 0 | 0.15912325/0.15912324 | True |
| 37 | 8cqx_A | 250 | False/True | -5.99025917e-06 | 0.00019222 | -1.86988025e-05 | -2.89916916e-06 | 0 | 0.15946157/0.15946576 | False |
| 38 | 8cqx_A | 450 | False/True | -1.34320143e-07 | 0.00006744 | -4.85634368e-06 | 3.44681682e-05 | 0 | 0.15946574/0.15946577 | False |
| 39 | 8fmw_K | 50 | True/True | -7.01951086e-09 | -0.00006327 | 4.85432128e-08 | -3.07209986e-06 | 0 | 0.15862654/0.15862655 | True |
| 40 | 8fmw_K | 250 | True/True | 1.79002253e-08 | 0.00010886 | 9.4018692e-07 | -4.33313529e-08 | 0 | 0.15862656/0.15862657 | True |
| 41 | 8fmw_K | 450 | True/True | -2.06628783e-07 | -0.00001000 | -1.0275804e-06 | 7.40828501e-07 | 0 | 0.15862649/0.15862658 | True |
| 43 | 9cfg_H | 250 | False/True | -2.13339479e-09 | 0.00001690 | 3.14169736e-07 | -4.40950879e-08 | 0 | 0.15860258/0.15860256 | True |
| 44 | 9cfg_H | 450 | False/True | -3.64989816e-09 | 0.00002966 | -1.68790159e-07 | 1.42376981e-05 | 0 | 0.15860258/0.15860257 | False |
| 45 | 9k3q_1 | 50 | True/True | 7.54527781e-08 | 0.00035141 | 6.22520678e-07 | 1.12846805e-05 | 0 | 0.15632133/0.15632107 | False |
| 46 | 9k3q_1 | 250 | True/True | -1.27367648e-07 | 0.00000993 | 6.59387387e-07 | -4.86596229e-05 | 0 | 0.15632130/0.15632129 | False |
| 47 | 9k3q_1 | 450 | False/True | 8.43637658e-08 | -0.00012089 | -9.27244415e-07 | -1.80617361e-05 | 0 | 0.15632133/0.15632131 | False |
| 48 | 9m6h_B | 50 | False/True | -1.14411604e-08 | 0.00014750 | -1.88862138e-07 | 2.24186791e-06 | 0 | 0.15960048/0.15960046 | True |
| 51 | 9pbc_A | 50 | False/True | 1.95692577e-08 | -0.00007316 | -1.57093319e-07 | 1.48148027e-07 | 0 | 0.15947451/0.15947448 | True |
| 52 | 9pbc_A | 250 | False/True | 5.17599918e-09 | 0.00000276 | -2.14991953e-07 | 1.78411025e-06 | 0 | 0.15947454/0.15947451 | True |
| 54 | 9q1s_DD | 50 | False/True | 9.54874213e-09 | -0.00002831 | 1.01521785e-07 | -1.17272318e-06 | 0 | 0.15923985/0.15923982 | True |
| 55 | 9q1s_DD | 250 | True/True | 8.1768281e-10 | 0.00004431 | 9.97547295e-07 | -4.07603911e-06 | 0 | 0.15923983/0.15923987 | True |
| 56 | 9q1s_DD | 450 | False/True | -5.8262315e-06 | 0.00002274 | -4.46548201e-05 | 0.000225900218 | 2 | 0.15923927/0.15923989 | False |
| 57 | 9v0p_F | 50 | True/True | -1.76137852e-08 | -0.00000604 | -1.22481754e-07 | 9.10932005e-07 | 0 | 0.15856493/0.15856497 | True |
| 59 | 9v0p_F | 450 | True/True | -3.37877353e-08 | 0.00002836 | 1.76154806e-07 | -5.254227e-06 | 0 | 0.15856497/0.15856498 | True |

Metric equivalence does not certify identical final coordinates: v4 did not preserve coordinate digests. Comparisons refer to the evaluated objectives/metrics under the predeclared tolerances. Both-converged and newly-converged aggregate results retain matched baseline examples; no identities are redrawn or filtered for optimization.

## Per-example objectives and convergence

| Index | Identity | Condition | Stratum | Converged | Iterations | Initial objective contribution | Final contribution | Projected residual | Local gain % | Closure evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1mhp_B | 50 | 129-256 | True | 50 | 0.50426388 | 0.44379056 | 0.00087603 | 5.41612 | 215 |
| 1 | 1mhp_B | 250 | 129-256 | True | 50 | 4.21255097 | 4.03428487 | 0.00083746 | 0.76238 | 202 |
| 2 | 1mhp_B | 450 | 129-256 | True | 60 | 17.00803314 | 16.63827893 | 0.00013261 | -0.31152 | 287 |
| 3 | 1qqn_A | 50 | 257-384 | True | 50 | 0.59660933 | 0.52812987 | 0.00088620 | 4.55172 | 259 |
| 4 | 1qqn_A | 250 | 257-384 | False | 1000 | 5.29717871 | 5.09694963 | 0.00106199 | 0.46545 | 12581 |
| 5 | 1qqn_A | 450 | 257-384 | True | 50 | 29.40492358 | 28.91283314 | 0.00056391 | -0.09750 | 250 |
| 6 | 1t8o_B | 50 | 20-64 | True | 50 | 0.47127885 | 0.41398620 | 0.00049005 | 4.10763 | 234 |
| 7 | 1t8o_B | 250 | 20-64 | True | 50 | 5.20939857 | 5.02344246 | 0.00083698 | -0.05928 | 226 |
| 8 | 1t8o_B | 450 | 20-64 | True | 60 | 12.41004338 | 12.12226198 | 0.00037658 | 0.06663 | 291 |
| 9 | 2jo5_A | 50 | 20-64 | True | 60 | 0.54989826 | 0.49486655 | 0.00006279 | 5.35701 | 275 |
| 10 | 2jo5_A | 250 | 20-64 | True | 50 | 2.92219715 | 2.79576708 | 0.00023854 | 0.02300 | 230 |
| 11 | 2jo5_A | 450 | 20-64 | False | 1000 | 8.03440670 | 7.83177707 | 0.00438924 | 0.32689 | 6887 |
| 12 | 2mdw_A | 50 | 20-64 | True | 50 | 0.39356281 | 0.34429541 | 0.00078399 | 4.95284 | 271 |
| 13 | 2mdw_A | 250 | 20-64 | True | 50 | 3.16023944 | 3.01602789 | 0.00065339 | 0.28986 | 202 |
| 14 | 2mdw_A | 450 | 20-64 | True | 60 | 16.59025149 | 16.28029735 | 0.00011026 | 0.28162 | 321 |
| 15 | 3qv9_A | 50 | 385-500 | True | 60 | 0.69973629 | 0.62662590 | 0.00081265 | 4.46954 | 317 |
| 16 | 3qv9_A | 250 | 385-500 | True | 50 | 4.98379802 | 4.78382299 | 0.00089714 | 0.50860 | 221 |
| 17 | 3qv9_A | 450 | 385-500 | True | 50 | 25.80124098 | 25.35632753 | 0.00070266 | 0.00964 | 208 |
| 18 | 5avn_A | 50 | 385-500 | False | 1000 | 0.60343780 | 0.53379542 | 0.00186485 | 4.59646 | 13539 |
| 19 | 5avn_A | 250 | 385-500 | False | 1000 | 3.51227743 | 3.34715079 | 0.00275489 | 0.51389 | 7871 |
| 20 | 5avn_A | 450 | 385-500 | True | 60 | 29.33849647 | 28.86589869 | 0.00062422 | -0.02989 | 328 |
| 21 | 5fds_A | 50 | 129-256 | False | 1000 | 0.56419673 | 0.49862738 | 0.00233936 | 5.68267 | 12611 |
| 22 | 5fds_A | 250 | 129-256 | True | 50 | 5.52599024 | 5.32309943 | 0.00075106 | 0.18932 | 268 |
| 23 | 5fds_A | 450 | 129-256 | True | 60 | 15.10197451 | 14.75674559 | 0.00017243 | 0.26345 | 304 |
| 24 | 5x1e_A | 50 | 65-128 | True | 50 | 0.56206174 | 0.49817439 | 0.00058544 | 4.66632 | 206 |
| 25 | 5x1e_A | 250 | 65-128 | True | 60 | 4.68321734 | 4.50153220 | 0.00033463 | 0.38177 | 292 |
| 26 | 5x1e_A | 450 | 65-128 | True | 60 | 13.30492438 | 12.97887654 | 0.00030687 | 0.05349 | 276 |
| 27 | 6szs_z | 50 | 257-384 | False | 1000 | 0.53792917 | 0.47275535 | 0.00173336 | 4.98750 | 6836 |
| 28 | 6szs_z | 250 | 257-384 | False | 1000 | 3.90782701 | 3.73772206 | 0.00183347 | 0.39752 | 14527 |
| 29 | 6szs_z | 450 | 257-384 | True | 50 | 42.35175612 | 41.77521180 | 0.00075531 | -0.14433 | 236 |
| 30 | 6tzk_A | 50 | 385-500 | False | 1000 | 0.51120271 | 0.44878637 | 0.00290924 | 4.83855 | 13585 |
| 31 | 6tzk_A | 250 | 385-500 | True | 60 | 5.04485764 | 4.84960734 | 0.00031779 | 0.60862 | 327 |
| 32 | 6tzk_A | 450 | 385-500 | True | 60 | 38.70799607 | 38.14822743 | 0.00047653 | 0.01778 | 324 |
| 33 | 8cmd_A | 50 | 129-256 | True | 50 | 0.46723478 | 0.40896037 | 0.00061958 | 5.99943 | 217 |
| 34 | 8cmd_A | 250 | 129-256 | False | 1000 | 4.36166964 | 4.18155917 | 0.00121377 | 0.80065 | 7799 |
| 35 | 8cmd_A | 450 | 129-256 | False | 1000 | 26.34880796 | 25.89729071 | 0.00105918 | -0.07435 | 12555 |
| 36 | 8cqx_A | 50 | 257-384 | False | 1000 | 0.60178211 | 0.53367640 | 0.00118693 | 4.82668 | 8772 |
| 37 | 8cqx_A | 250 | 257-384 | True | 50 | 4.14780029 | 3.97124047 | 0.00082862 | 0.93838 | 203 |
| 38 | 8cqx_A | 450 | 257-384 | True | 60 | 30.13925092 | 29.64074976 | 0.00092360 | -0.08064 | 268 |
| 39 | 8fmw_K | 50 | 65-128 | True | 60 | 0.43774155 | 0.38076694 | 0.00083177 | 5.90336 | 293 |
| 40 | 8fmw_K | 250 | 65-128 | True | 50 | 4.25403182 | 4.08268696 | 0.00095891 | 0.72838 | 205 |
| 41 | 8fmw_K | 450 | 65-128 | True | 60 | 19.37046941 | 18.99273877 | 0.00028645 | 0.25270 | 302 |
| 42 | 9cfg_H | 50 | 65-128 | False | 1000 | 0.42868451 | 0.37041577 | 0.01778262 | 5.49910 | 6882 |
| 43 | 9cfg_H | 250 | 65-128 | True | 50 | 4.78018457 | 4.59084632 | 0.00065308 | 0.68614 | 227 |
| 44 | 9cfg_H | 450 | 65-128 | True | 50 | 13.87195288 | 13.55670782 | 0.00024277 | 0.27143 | 198 |
| 45 | 9k3q_1 | 50 | 20-64 | True | 50 | 0.34982574 | 0.30523423 | 0.00092977 | 6.91013 | 274 |
| 46 | 9k3q_1 | 250 | 20-64 | True | 50 | 3.74185900 | 3.59443583 | 0.00025554 | 0.76756 | 166 |
| 47 | 9k3q_1 | 450 | 20-64 | True | 50 | 27.15890495 | 26.74092150 | 0.00026514 | 0.04311 | 237 |
| 48 | 9m6h_B | 50 | 385-500 | True | 50 | 0.51692366 | 0.45302110 | 0.00043261 | 5.16926 | 216 |
| 49 | 9m6h_B | 250 | 385-500 | False | 1000 | 4.79952430 | 4.60645240 | 0.00389093 | 0.71763 | 11675 |
| 50 | 9m6h_B | 450 | 385-500 | False | 1000 | 60.66777004 | 59.99593892 | 0.00312077 | 0.02142 | 14523 |
| 51 | 9pbc_A | 50 | 257-384 | True | 50 | 0.64214913 | 0.57194555 | 0.00059081 | 4.49193 | 205 |
| 52 | 9pbc_A | 250 | 257-384 | True | 50 | 5.59245601 | 5.38702937 | 0.00068977 | 0.52972 | 248 |
| 53 | 9pbc_A | 450 | 257-384 | False | 1000 | 33.03195656 | 32.50715107 | 0.00169357 | -0.02959 | 12598 |
| 54 | 9q1s_DD | 50 | 129-256 | True | 50 | 0.72310109 | 0.64948978 | 0.00041539 | 4.44860 | 198 |
| 55 | 9q1s_DD | 250 | 129-256 | True | 60 | 4.22700601 | 4.04895190 | 0.00074840 | 0.46535 | 287 |
| 56 | 9q1s_DD | 450 | 129-256 | True | 60 | 23.21732884 | 22.80432286 | 0.00037740 | 0.10581 | 286 |
| 57 | 9v0p_F | 50 | 65-128 | True | 60 | 0.49687661 | 0.43692897 | 0.00018168 | 5.23852 | 302 |
| 58 | 9v0p_F | 250 | 65-128 | False | 1000 | 5.66124096 | 5.46335649 | 0.00129299 | 0.65914 | 11633 |
| 59 | 9v0p_F | 450 | 65-128 | True | 60 | 17.56706596 | 17.20288173 | 0.00048900 | 0.15229 | 281 |

## Recurrent state telemetry

| State | Offset RMSE Å | Mean local Å | Raw cart Å² | Aligned Å | Chiral loss | Inversions | Assessable | Step RMS/max Å | Cumulative RMS/max Å | Eligible/degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.895241/2.874420/3.553660 | 2.774440 | 35.077882 | 7.858572 | 0.423339 | 5493 | 13029 | 0.00000000/0.000000000000 | 0.00000000/0.000000000000 | 13089/0 |
| 1 | 1.890920/2.867667/3.544064 | 2.767550 | 34.881316 | 7.828889 | 0.417326 | 5394 | 13029 | 0.03958132/0.039999999944 | 0.03958132/0.039999999944 | 13089/0 |
| 2 | 1.886308/2.860648/3.534314 | 2.760423 | 34.685795 | 7.799145 | 0.411299 | 5307 | 13028 | 0.03958132/0.039999999944 | 0.07916264/0.079999999888 | 13089/0 |
| 3 | 1.881408/2.853374/3.524415 | 2.753066 | 34.491319 | 7.769351 | 0.405346 | 5274 | 13029 | 0.03958132/0.039999999944 | 0.11874396/0.119999999832 | 13089/0 |
| 4 | 1.876224/2.845850/3.514368 | 2.745481 | 34.297887 | 7.739518 | 0.399580 | 5182 | 13029 | 0.03958132/0.039999999944 | 0.15832528/0.159999999776 | 13089/0 |

Individual JSON records preserve all baseline/final metrics, state/step corrections, temporary eligibility and chirality-assessability changes, initial/final objective, history and coordinate digests. Paired comparisons and complete-repeat hashes are separately versioned. No coordinates/datasets/checkpoints are committed.

## Validation

92 focused CPU tests passed, including 12 v5 cases; Ruff checks passed. Exact v4 solver code, optimizer/config, convergence routine, objective implementation and metric function parity are tested. Bounds, eligibility, no neural/global parameters or mutation, zero parity, deterministic stopping/reproduction and input integrity are tested. Historical source/cache/result hashes were rechecked. The inherited full-suite failures remain separately documented; no complete-suite rerun was needed for this isolated diagnostic.

Full-panel result reproduction: `{"all_exact_non_timing_matches": true, "examples": 60, "fresh_zero_initialization": true, "protected_hashes_verified": true, "same_workers_threads_optimizer": true, "states_coordinates_metrics_histories_convergence_reproduced": true}`. No commit/push preceded tests and this reproduction. CUDA used: NO. Neural training launched: NO. Environment and historical E010/E011/E012 work remained unchanged. No downstream sweep or experiment was launched.
