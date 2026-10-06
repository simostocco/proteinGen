# E010 Phase 4D v4 bounded correction oracle

**Classification: O5.** Numerical convergence passed for 26/60 examples. No neural model was trained and CUDA was not used.

The maximum practical safe repair is **not established** by this run. Attained gains are 5.0675% / 0.5110% / 0.0664% for conditions 50/250/450; all three condition aggregates satisfy the fixed safety criteria. However, 34 examples exhaust the budget without meeting stationarity, with projected residual as high as 0.3067. The inherited float32 Cartesian cast and saturating local-variable conditioning are numerical limitations worth checking; this run does not isolate their contributions.

Removing neural capacity and shared learning did not produce large condition-450 repair in the attained solutions. That supports a follow-up numerical/feasibility check, not a definitive O2 or O3 conclusion. The optimistic bound still allows up to 7.0554% condition-450 improvement, so geometry alone has not ruled out 5%. A converged weighted-objective minimizer would also not automatically maximize local gain over the Cartesian/chirality-safe feasible set.

Recommended next experiment: **pre-register a Cartesian-variable constrained-oracle convergence cross-check with the same K, s_max, beta and gamma**. Do not change the neural architecture or correction budget on the basis of this inconclusive oracle.

Source result commit: `43864b38001c2ee0118bc8b7530b547cba4ca5f1`. Oracle preparation: `20c20dea94f258ea3032aadd2f31566f0ef6eebe`. The fixed implementation/config and all tracked v3 records stayed unchanged during panel execution.

The oracle uses four independent free local-frame correction fields per example. Frames are recomputed from the current coordinates, without detaching recurrence. K=4, s_max=0.04 Å; the smooth radial map and zero-correction semantics match v3. Independent per-example optimization preserves the exact full-panel objective contributions: local/60 + 16.8 Cartesian/60 + 2 chiral_sum/13029. No neural parameters or global forward are present.

Optimizer: deterministic L-BFGS, LR=1, history=20, strong-Wolfe search, maximum 1000 outer iterations; exact-zero initialization and no RNG. CPU float64 variables/frame arithmetic retain the historical float32 Cartesian prediction cast. Four independent workers each use one CPU thread; serial example 0 was retained before redistribution. No settings changed after panel execution began.

Convergence requires ten stable iterations (relative objective change ≤1e-8, coordinate change ≤1e-6 Å) and normalized projected-gradient residual ≤0.001, checked every ten iterations, with an absolute gradient floor 1e-7. Reaching 1000 iterations is not convergence. Full solver histories and closure evaluation counts are recorded per example.

Iterations min/median/max: 50/1000/1000. Sum of per-example runtime: 6772.10 seconds (worker concurrency means this is not wall time).

## Overall, conditions and lengths

| overall | Local gain % | Offset gains % 1/2/3 | Raw cart change % | Aligned change % | Chiral loss change % | Inversion change | Assessable final/base | Correction RMS/max Å | Eligible/degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 1.04384 | 1.00329/0.99408/1.10572 | -2.22359 | -1.51492 | -5.61425 | -313 | 13029/13029 | 0.158324/0.160000 | 13089/0 |

| by_condition | Local gain % | Offset gains % 1/2/3 | Raw cart change % | Aligned change % | Chiral loss change % | Inversion change | Assessable final/base | Correction RMS/max Å | Eligible/degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250 | 0.51102 | 0.08200/0.39947/0.85871 | -4.06976 | -2.05246 | -4.87777 | -70 | 4343/4343 | 0.158323/0.160000 | 4363/0 |
| 450 | 0.06636 | 0.06909/0.04836/0.07852 | -1.70703 | -0.58442 | -1.81273 | -28 | 4343/4343 | 0.158325/0.160000 | 4363/0 |
| 50 | 5.06753 | 4.25177/5.00838/5.83903 | -11.70277 | -6.09921 | -18.23943 | -215 | 4343/4343 | 0.158323/0.160000 | 4363/0 |

| by_stratum | Local gain % | Offset gains % 1/2/3 | Raw cart change % | Aligned change % | Chiral loss change % | Inversion change | Assessable final/base | Correction RMS/max Å | Eligible/degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 129-256 | 1.03369 | 0.86905/0.93910/1.19108 | -2.54706 | -1.61365 | -3.79135 | -47 | 2115/2115 | 0.159075/0.160000 | 2127/0 |
| 20-64 | 1.01848 | 1.15693/0.98566/0.96269 | -2.54319 | -1.21808 | -3.99471 | -11 | 411/411 | 0.154822/0.160000 | 423/0 |
| 257-384 | 1.02434 | 0.94587/0.98166/1.10057 | -1.99845 | -1.43609 | -5.94362 | -95 | 3999/3999 | 0.159515/0.160000 | 4011/0 |
| 385-500 | 1.06781 | 1.06983/0.99878/1.12239 | -1.80907 | -1.46769 | -6.30896 | -125 | 5178/5178 | 0.159627/0.160000 | 5190/0 |
| 65-128 | 1.07373 | 0.95259/1.05984/1.14364 | -2.81135 | -1.82485 | -5.07248 | -35 | 1326/1326 | 0.158580/0.160000 | 1338/0 |

## Absolute metrics

| Group | State | Offset RMSE Å | Mean local Å | Raw Cartesian Å² | Aligned RMSD Å | Chiral loss | Inversions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | baseline | 1.895241/2.874420/3.553660 | 2.774440 | 35.077882 | 7.858572 | 0.423339 | 5493 |
| overall | oracle | 1.876226/2.845846/3.514366 | 2.745479 | 34.297893 | 7.739521 | 0.399572 | 5180 |
| 50 | baseline | 1.263806/1.442583/1.446908 | 1.384432 | 1.762931 | 2.245481 | 0.199807 | 1196 |
| 50 | oracle | 1.210072/1.370333/1.362423 | 1.314276 | 1.556619 | 2.108524 | 0.163364 | 981 |
| 250 | baseline | 1.976871/2.848015/3.353130 | 2.726006 | 15.549012 | 6.507821 | 0.504340 | 2133 |
| 250 | oracle | 1.975250/2.836638/3.324337 | 2.712075 | 14.916205 | 6.374250 | 0.479739 | 2063 |
| 450 | baseline | 2.445046/4.332661/5.860942 | 4.212883 | 87.921702 | 14.822414 | 0.565870 | 2164 |
| 450 | oracle | 2.443357/4.330566/5.856340 | 4.210087 | 86.420855 | 14.735788 | 0.555612 | 2136 |

| State | Full-panel exact v3 objective |
| --- | --- |
| baseline | 600.11135623 |
| oracle | 586.85575259 |

## Descriptive recurrent-neural comparison

| Condition | Oracle attained local gain % | Converged examples | Safe aggregate | Optimistic geometric gain ceiling % | Neural S/M/L local gains % |
| --- | --- | --- | --- | --- | --- |
| 50 | 5.06753 | 8/20 | True | 18.00747 | 6.19601/5.80689/5.63495 |
| 250 | 0.51102 | 8/20 | True | 10.12021 | 0.46603/0.54757/0.53305 |
| 450 | 0.06636 | 10/20 | True | 7.05535 | 0.00693/0.00142/0.12796 |

Safe aggregate requires all offsets non-harmful, raw Cartesian ≤1e-6 relative numerical tolerance, aligned regression ≤1%, continuous chiral loss ≤1e-6 relative numerical tolerance, inversions non-increasing, original assessability/frame eligibility preserved and finite outputs. These are fixed diagnostic safety criteria, not outcome-selected coefficients.

The geometric ceiling relaxes coupling and Cartesian/chiral safety: each absolute pair-distance error can shrink by no more than the sum of endpoint displacement radii. It is an upper bound, not an attainable estimate. Since condition 450’s ceiling exceeds 5%, it cannot certify insufficient correction budget. A low weighted-objective solution alone also cannot certify the maximum local gain over all safe corrections. Reported oracle gains are attained results, not proven global maxima.

## Per-example baseline and oracle

| Index | Identity | Condition | Stratum | Local gain % | Oracle mean local Å | Cart change % | Aligned change % | Chiral change % | Inversion change | Assessable final/base | Correction RMS/max Å | Iterations | Converged | Projected residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1mhp_B | 50 | 129-256 | 5.41611 | 1.088899 | -12.05838 | -6.23307 | -12.18196 | -6 | 189/189 | 0.159164/0.160000 | 1000 | False | 0.00125105 |
| 1 | 1mhp_B | 250 | 129-256 | 0.76226 | 2.227270 | -4.29446 | -2.16919 | -0.97707 | -5 | 189/189 | 0.159164/0.160000 | 1000 | False | 0.00115944 |
| 2 | 1mhp_B | 450 | 129-256 | -0.31153 | 3.708794 | -2.21669 | -1.08132 | -0.95048 | -2 | 189/189 | 0.159164/0.160000 | 60 | True | 0.00028817 |
| 3 | 1qqn_A | 50 | 257-384 | 4.55172 | 1.517893 | -11.45098 | -5.89415 | -22.35406 | -21 | 375/375 | 0.159576/0.160000 | 1000 | False | 0.00116044 |
| 4 | 1qqn_A | 250 | 257-384 | 0.46546 | 2.876477 | -3.85400 | -1.93033 | -4.25996 | -2 | 375/375 | 0.159576/0.160000 | 1000 | False | 0.00377839 |
| 5 | 1qqn_A | 450 | 257-384 | -0.09750 | 3.892128 | -1.69213 | -0.81841 | -1.05836 | -3 | 375/375 | 0.159576/0.160000 | 60 | True | 0.00028484 |
| 6 | 1t8o_B | 50 | 20-64 | 4.10791 | 1.206970 | -12.39880 | -6.38316 | -12.78372 | -7 | 55/55 | 0.157217/0.160000 | 1000 | False | 0.00228858 |
| 7 | 1t8o_B | 250 | 20-64 | -0.05946 | 2.789383 | -3.66936 | -1.84060 | -1.64003 | -2 | 55/55 | 0.157217/0.160000 | 50 | True | 0.00097498 |
| 8 | 1t8o_B | 450 | 20-64 | 0.06667 | 4.331690 | -2.38145 | 0.57123 | -0.07284 | 0 | 55/55 | 0.157217/0.160000 | 60 | True | 0.00009289 |
| 9 | 2jo5_A | 50 | 20-64 | 5.35684 | 1.343803 | -9.97592 | -5.42653 | -19.15307 | -1 | 17/17 | 0.151789/0.160000 | 60 | True | 0.00028753 |
| 10 | 2jo5_A | 250 | 20-64 | 0.02311 | 3.274523 | -4.61312 | -2.17501 | -3.94331 | 3 | 17/17 | 0.151789/0.160000 | 1000 | False | 0.00113080 |
| 11 | 2jo5_A | 450 | 20-64 | 0.32786 | 3.821211 | -2.58343 | 0.19116 | -1.87793 | 0 | 17/17 | 0.151789/0.160000 | 60 | True | 0.00040123 |
| 12 | 2mdw_A | 50 | 20-64 | 4.95234 | 1.384016 | -12.77152 | -6.61231 | -17.14950 | -1 | 24/24 | 0.153960/0.160000 | 50 | True | 0.00017221 |
| 13 | 2mdw_A | 250 | 20-64 | 0.28944 | 2.679031 | -4.73004 | -2.49418 | 5.52980 | -1 | 24/24 | 0.153960/0.160000 | 50 | True | 0.00057082 |
| 14 | 2mdw_A | 450 | 20-64 | 0.28160 | 5.123182 | -1.91283 | 0.45457 | 1.93834 | 0 | 24/24 | 0.153960/0.160000 | 50 | True | 0.00036176 |
| 15 | 3qv9_A | 50 | 385-500 | 4.46961 | 1.490968 | -10.38376 | -5.32637 | -17.13861 | -22 | 496/496 | 0.159679/0.160000 | 1000 | False | 0.00217637 |
| 16 | 3qv9_A | 250 | 385-500 | 0.50864 | 2.691647 | -4.07030 | -2.06116 | -6.21573 | -9 | 496/496 | 0.159679/0.160000 | 1000 | False | 0.00297750 |
| 17 | 3qv9_A | 450 | 385-500 | 0.00963 | 3.815387 | -1.74060 | -0.86071 | -3.01452 | -4 | 496/496 | 0.159679/0.160000 | 1000 | False | 0.00255106 |
| 18 | 5avn_A | 50 | 385-500 | 4.59648 | 1.458865 | -11.52200 | -5.95537 | -20.22940 | -23 | 384/384 | 0.159586/0.160000 | 1000 | False | 0.00239375 |
| 19 | 5avn_A | 250 | 385-500 | 0.51390 | 2.426442 | -4.81738 | -2.42886 | -3.59203 | -2 | 384/384 | 0.159583/0.160000 | 1000 | False | 0.01634611 |
| 20 | 5avn_A | 450 | 385-500 | -0.02988 | 3.719358 | -1.62440 | -0.76817 | -2.37554 | -7 | 384/384 | 0.159586/0.160000 | 60 | True | 0.00027596 |
| 21 | 5fds_A | 50 | 129-256 | 5.68214 | 1.397635 | -11.60641 | -5.98622 | -18.20181 | -6 | 128/128 | 0.158774/0.160000 | 50 | True | 0.00065929 |
| 22 | 5fds_A | 250 | 129-256 | 0.18930 | 2.822537 | -3.75871 | -1.87290 | -1.37459 | 1 | 128/128 | 0.158774/0.160000 | 1000 | False | 0.00429167 |
| 23 | 5fds_A | 450 | 129-256 | 0.26351 | 4.724499 | -2.33569 | -0.22442 | -1.04672 | -4 | 128/128 | 0.158774/0.160000 | 60 | True | 0.00071286 |
| 24 | 5x1e_A | 50 | 65-128 | 4.66633 | 1.489410 | -11.46887 | -5.92336 | -20.16124 | -6 | 107/107 | 0.158539/0.160000 | 50 | True | 0.00088840 |
| 25 | 5x1e_A | 250 | 65-128 | 0.38185 | 2.817852 | -3.96898 | -1.95722 | -5.66831 | -6 | 107/107 | 0.158539/0.160000 | 50 | True | 0.00077740 |
| 26 | 5x1e_A | 450 | 65-128 | 0.05353 | 4.243468 | -2.51014 | -1.02265 | -0.23284 | 0 | 107/107 | 0.158539/0.160000 | 1000 | False | 0.00115961 |
| 27 | 6szs_z | 50 | 257-384 | 4.98723 | 1.362916 | -12.05987 | -6.22119 | -23.49480 | -17 | 359/359 | 0.159557/0.160000 | 1000 | False | 0.00276131 |
| 28 | 6szs_z | 250 | 257-384 | 0.39752 | 2.425781 | -4.43351 | -2.25219 | -5.43618 | -8 | 359/359 | 0.159557/0.160000 | 50 | True | 0.00048426 |
| 29 | 6szs_z | 450 | 257-384 | -0.14434 | 3.707827 | -1.37007 | -0.68730 | -2.45206 | -4 | 359/359 | 0.159557/0.160000 | 1000 | False | 0.00148962 |
| 30 | 6tzk_A | 50 | 385-500 | 4.83852 | 1.324766 | -12.25104 | -6.32581 | -18.33986 | -25 | 448/448 | 0.159645/0.160000 | 1000 | False | 0.00525338 |
| 31 | 6tzk_A | 250 | 385-500 | 0.60861 | 2.698435 | -3.91531 | -1.97575 | -8.01664 | -8 | 448/448 | 0.159645/0.160000 | 1000 | False | 0.00501347 |
| 32 | 6tzk_A | 450 | 385-500 | 0.01781 | 4.322050 | -1.45844 | -0.67877 | -2.78588 | -4 | 448/448 | 0.159645/0.160000 | 1000 | False | 0.00153558 |
| 33 | 8cmd_A | 50 | 129-256 | 5.99938 | 1.022715 | -12.48981 | -6.44663 | -13.41259 | -4 | 180/180 | 0.159123/0.160000 | 1000 | False | 0.00115357 |
| 34 | 8cmd_A | 250 | 129-256 | 0.80065 | 2.487699 | -4.19594 | -2.09546 | -2.35899 | -2 | 180/180 | 0.159123/0.160000 | 90 | True | 0.00092779 |
| 35 | 8cmd_A | 450 | 129-256 | -0.07440 | 4.346353 | -1.74074 | -0.52507 | 0.11633 | 1 | 180/180 | 0.159123/0.160000 | 1000 | False | 0.00162141 |
| 36 | 8cqx_A | 50 | 257-384 | 4.83188 | 1.481823 | -11.34777 | -5.84190 | -17.82052 | -11 | 297/297 | 0.159425/0.160000 | 1000 | False | 0.30668214 |
| 37 | 8cqx_A | 250 | 257-384 | 0.93819 | 2.771021 | -4.32110 | -2.19207 | -6.27350 | -12 | 297/297 | 0.159462/0.160000 | 1000 | False | 0.04870351 |
| 38 | 8cqx_A | 450 | 257-384 | -0.08071 | 4.173834 | -1.67319 | -0.21872 | -2.27115 | -1 | 297/297 | 0.159466/0.160000 | 1000 | False | 0.00277098 |
| 39 | 8fmw_K | 50 | 65-128 | 5.90343 | 1.236237 | -13.05945 | -6.76711 | -19.38377 | -2 | 114/114 | 0.158627/0.160000 | 50 | True | 0.00066287 |
| 40 | 8fmw_K | 250 | 65-128 | 0.72827 | 2.475697 | -4.08825 | -2.07304 | -4.91518 | 0 | 114/114 | 0.158627/0.160000 | 60 | True | 0.00074090 |
| 41 | 8fmw_K | 450 | 65-128 | 0.25271 | 4.962807 | -1.98579 | -0.95703 | -1.36632 | 1 | 114/114 | 0.158626/0.160000 | 70 | True | 0.00097352 |
| 42 | 9cfg_H | 50 | 65-128 | 5.50465 | 1.073569 | -13.72436 | -7.14654 | -14.09409 | -11 | 112/112 | 0.158603/0.160000 | 60 | True | 0.00064727 |
| 43 | 9cfg_H | 250 | 65-128 | 0.68613 | 2.617788 | -4.02987 | -2.02736 | -2.12423 | -3 | 112/112 | 0.158603/0.160000 | 1000 | False | 0.00227221 |
| 44 | 9cfg_H | 450 | 65-128 | 0.27140 | 4.801721 | -2.33004 | -1.16691 | -0.81623 | 1 | 112/112 | 0.158603/0.160000 | 1000 | False | 0.00109822 |
| 45 | 9k3q_1 | 50 | 20-64 | 6.90978 | 1.163015 | -12.68797 | -6.68882 | -14.14857 | 0 | 41/41 | 0.156321/0.160000 | 50 | True | 0.00033893 |
| 46 | 9k3q_1 | 250 | 20-64 | 0.76755 | 2.879635 | -4.03345 | -1.96294 | -6.18132 | 0 | 41/41 | 0.156321/0.160000 | 50 | True | 0.00091378 |
| 47 | 9k3q_1 | 450 | 20-64 | 0.04323 | 3.715974 | -1.55163 | -0.49432 | 0.11847 | -2 | 41/41 | 0.156321/0.160000 | 1000 | False | 0.00114510 |
| 48 | 9m6h_B | 50 | 385-500 | 5.16912 | 1.375551 | -12.36255 | -6.39701 | -19.71668 | -15 | 398/398 | 0.159600/0.160000 | 1000 | False | 0.00733756 |
| 49 | 9m6h_B | 250 | 385-500 | 0.71771 | 2.820964 | -4.08573 | -2.07851 | -5.60972 | -4 | 398/398 | 0.159600/0.160000 | 1000 | False | 0.00931480 |
| 50 | 9m6h_B | 450 | 385-500 | 0.02136 | 3.955470 | -1.11224 | -0.54887 | -1.50334 | -2 | 398/398 | 0.159600/0.160000 | 1000 | False | 0.00134159 |
| 51 | 9pbc_A | 50 | 257-384 | 4.49201 | 1.432588 | -10.95016 | -5.62726 | -16.74476 | -13 | 302/302 | 0.159475/0.160000 | 1000 | False | 0.00332130 |
| 52 | 9pbc_A | 250 | 257-384 | 0.52972 | 2.885735 | -3.73886 | -1.86500 | -4.28826 | -3 | 302/302 | 0.159475/0.160000 | 1000 | False | 0.00122324 |
| 53 | 9pbc_A | 450 | 257-384 | -0.02965 | 4.089306 | -1.60350 | -0.41275 | -2.90370 | 0 | 302/302 | 0.159475/0.160000 | 50 | True | 0.00086283 |
| 54 | 9q1s_DD | 50 | 129-256 | 4.44863 | 1.303317 | -10.16077 | -5.20716 | -18.51096 | -17 | 208/208 | 0.159240/0.160000 | 1000 | False | 0.00164467 |
| 55 | 9q1s_DD | 250 | 129-256 | 0.46531 | 2.548385 | -4.29412 | -2.11265 | -5.52277 | -3 | 208/208 | 0.159240/0.160000 | 50 | True | 0.00066908 |
| 56 | 9q1s_DD | 450 | 129-256 | 0.10579 | 3.792510 | -1.79797 | -0.78468 | -0.41496 | 0 | 208/208 | 0.159239/0.160000 | 1000 | False | 0.03655329 |
| 57 | 9v0p_F | 50 | 65-128 | 5.23853 | 1.130565 | -12.15382 | -6.29513 | -10.81787 | -7 | 109/109 | 0.158565/0.160000 | 60 | True | 0.00079543 |
| 58 | 9v0p_F | 250 | 65-128 | 0.65693 | 3.025199 | -3.56040 | -1.76800 | -3.00303 | -4 | 109/109 | 0.158524/0.160000 | 1000 | False | 0.21998961 |
| 59 | 9v0p_F | 450 | 65-128 | 0.15227 | 4.954180 | -2.12368 | -0.70221 | -1.73871 | 2 | 109/109 | 0.158565/0.160000 | 50 | True | 0.00088354 |

## Recurrent correction and frame telemetry

| State | Offset RMSE Å | Mean local Å | Raw cart Å² | Aligned Å | Chiral loss | Inversions | Assessable | Step RMS/max Å | Cumulative RMS/max Å | Eligible/degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.895241/2.874420/3.553660 | 2.774440 | 35.077882 | 7.858572 | 0.423339 | 5493 | 13029 | 0.000000/0.00000000 | 0.000000/0.00000000 | 13089/0 |
| 1 | 1.890921/2.867665/3.544062 | 2.767549 | 34.881319 | 7.828890 | 0.417326 | 5397 | 13029 | 0.039581/0.04000000 | 0.039581/0.04000000 | 13089/0 |
| 2 | 1.886309/2.860646/3.534313 | 2.760423 | 34.685799 | 7.799147 | 0.411298 | 5307 | 13028 | 0.039581/0.04000000 | 0.079162/0.08000000 | 13089/0 |
| 3 | 1.881410/2.853370/3.524414 | 2.753064 | 34.491325 | 7.769353 | 0.405342 | 5272 | 13029 | 0.039581/0.04000000 | 0.118743/0.12000000 | 13089/0 |
| 4 | 1.876226/2.845846/3.514366 | 2.745479 | 34.297893 | 7.739521 | 0.399572 | 5180 | 13029 | 0.039581/0.04000000 | 0.158324/0.16000000 | 13089/0 |

Every example JSON preserves full baseline/oracle metrics, per-step and cumulative corrections, P0–P4 assessability/frame counts and convergence history. Exact zero initialization and per-step/cumulative bounds are verified. No historical artifact was overwritten.

## Validation and limits

Focused CPU tests: 80 passed, including 10 v4 cases, plus Ruff checks. Tests cover parity, bounds, current-frame recomputation, leaf variables/no neural parameters, deterministic initialization/solution, independent examples, unchanged input tensors/global gradient isolation, exact v3 objective and panel reduction parity, convergence bookkeeping, metric units and geometric envelope. The previously documented 134 inherited full-suite failures are not rerun for this isolated diagnostic.

All protected v3 and historical input hashes and the immutable frozen cache were verified after execution. Source E010/global model state was not loaded or mutated; E011/E012 and Python environment were unchanged. CUDA used: NO. Neural training launched: NO. No downstream coefficient, bound, capacity or held-out experiment was launched.

Post-execution validation verified 32 pinned v3/oracle files, six v3 source/config pins, 57 historical/source inputs, exact iteration-zero metric parity and tight baseline parity against v3. Maximum observed step correction is 0.03999999999950916 Å; cumulative maximum is 0.15999999981937016 Å. All states are finite, without collapse, and preserve frame eligibility. Final chirality assessability is preserved for every example; one intermediate state (example 49, P2) temporarily loses one quartet. See `post_execution_validation.json`.
