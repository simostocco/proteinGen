# E002 Versus E001 Calibrated Ensemble Comparison

## Executive Conclusion

This selected-model comparison evaluates E002 relative to E001 under the calibrated paired generated-ensemble protocol. Positive improvement means the E001 value is lower than the E002 value for lower-is-better metrics; strict empirical real-like geometry and heuristic validity are reported separately.

## Protocol Compatibility

- Compatibility status: `True`
- Paired key: `['requested_length', 'sample_index', 'seed']`
- Paired sample count: `375`
- Counts by length: `{'64': 100, '128': 100, '256': 100, '384': 50, '500': 25}`
- Input hashes preserved: `True`

## Checkpoint And Training-History Caveat

Selected-model comparison: E001 epoch 4/global step 160166 versus E002 epoch 5/global step 169181. Selected epochs, optimizer steps, and training histories differ, so this is not a perfectly controlled causal auxiliary-loss ablation.

## Paired Primary Results

| metric | baseline_mean | candidate_mean | mean_improvement_baseline_minus_candidate | bootstrap_ci_low | bootstrap_ci_high | fraction_improved | paired_standardized_effect_size | sign_test_p_value_bh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| triangle_violation_fraction | 0.00192448 | 0.00219271 | -0.000268229 | -0.00056901 | 1.5625e-05 | 0.405333 | -0.0898152 | 0.618595 |
| negative_eigenvalue_mass_fraction | 0.0974289 | 0.0926963 | 0.00473265 | 0.00359787 | 0.00582859 | 0.688 | 0.379422 | 3.8209e-12 |
| rank3_residual_energy_fraction | 0.154451 | 0.147684 | 0.00676679 | 0.00427944 | 0.00915147 | 0.650667 | 0.282599 | 4.23828e-08 |
| classical_mds_stress | 0.075929 | 0.0721133 | 0.00381575 | 0.00172839 | 0.00587892 | 0.605333 | 0.187524 | 0.000227376 |
| adjacent_residue_distance_rmse | 0.252429 | 0.279273 | -0.0268437 | -0.0343004 | -0.0197966 | 0.354667 | -0.36315 | 1.17547e-07 |

Positive improvement means E002 has the lower value for these lower-is-better metrics.

## Validity Transitions

| group | requested_length | flag | baseline_fail_candidate_fail | baseline_fail_candidate_pass | baseline_pass_candidate_fail | baseline_pass_candidate_pass | discordant_candidate_better | discordant_baseline_better | mcnemar_exact_p_value | baseline_pass_fraction | candidate_pass_fraction | mcnemar_exact_p_value_bh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall |  | empirical_real_like_geometry_pass | 375 | 0 | 0 | 0 | 0 | 0 |  | 0 | 0 |  |
| overall |  | heuristic_edm_quality_pass | 312 | 9 | 15 | 39 | 9 | 15 | 0.307456 | 0.144 | 0.128 | 0.409942 |

Strict empirical real-like geometry pass fractions: E001 0, E002 0. Heuristic EDM-quality pass fraction changes from 0.144 to 0.128.

## Length-Dependent Effects

| requested_length | metric | baseline_mean | candidate_mean | mean_improvement_baseline_minus_candidate | bootstrap_ci_low | bootstrap_ci_high | fraction_improved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | triangle_violation_fraction | 0.00310547 | 0.00430176 | -0.00119629 | -0.00229004 | -0.000283081 | 0.31 |
| 64 | negative_eigenvalue_mass_fraction | 0.0486933 | 0.0519124 | -0.00321908 | -0.00521753 | -0.00128385 | 0.37 |
| 64 | rank3_residual_energy_fraction | 0.07898 | 0.0759439 | 0.00303612 | -0.000902084 | 0.00709834 | 0.59 |
| 64 | classical_mds_stress | 0.0395467 | 0.036432 | 0.00311468 | 0.000262963 | 0.00579348 | 0.69 |
| 64 | adjacent_residue_distance_rmse | 0.286774 | 0.339538 | -0.0527641 | -0.0719009 | -0.0338886 | 0.24 |
| 128 | triangle_violation_fraction | 0.00125488 | 0.00119141 | 6.34766e-05 | -0.000200317 | 0.00034668 | 0.41 |
| 128 | negative_eigenvalue_mass_fraction | 0.0831648 | 0.0745467 | 0.00861809 | 0.00672568 | 0.0105405 | 0.86 |
| 128 | rank3_residual_energy_fraction | 0.133561 | 0.125967 | 0.0075945 | 0.00271597 | 0.0125483 | 0.68 |
| 128 | classical_mds_stress | 0.0603264 | 0.0589699 | 0.00135654 | -0.00297035 | 0.00568094 | 0.54 |
| 128 | adjacent_residue_distance_rmse | 0.224176 | 0.230232 | -0.00605616 | -0.0182303 | 0.00554154 | 0.5 |
| 256 | triangle_violation_fraction | 0.0012793 | 0.00129395 | -1.46484e-05 | -0.000239258 | 0.0002052 | 0.44 |
| 256 | negative_eigenvalue_mass_fraction | 0.116703 | 0.111324 | 0.00537887 | 0.00256254 | 0.00805064 | 0.7 |
| 256 | rank3_residual_energy_fraction | 0.187899 | 0.180478 | 0.0074209 | 0.00182265 | 0.0130949 | 0.64 |
| 256 | classical_mds_stress | 0.0923384 | 0.088114 | 0.00422444 | -0.00082313 | 0.00915688 | 0.57 |
| 256 | adjacent_residue_distance_rmse | 0.241845 | 0.255939 | -0.0140943 | -0.0252096 | -0.00263643 | 0.43 |
| 384 | triangle_violation_fraction | 0.00170898 | 0.00161133 | 9.76563e-05 | -0.000429932 | 0.000664063 | 0.46 |
| 384 | negative_eigenvalue_mass_fraction | 0.149874 | 0.13833 | 0.0115441 | 0.00877531 | 0.0145341 | 0.9 |
| 384 | rank3_residual_energy_fraction | 0.234862 | 0.223251 | 0.0116108 | 0.00701061 | 0.0162983 | 0.72 |
| 384 | classical_mds_stress | 0.121802 | 0.114122 | 0.00768017 | 0.00293661 | 0.0127362 | 0.6 |
| 384 | adjacent_residue_distance_rmse | 0.25854 | 0.292697 | -0.0341574 | -0.0467022 | -0.0221773 | 0.22 |
| 500 | triangle_violation_fraction | 0.00289062 | 0.00251953 | 0.000371094 | -0.000371094 | 0.00113281 | 0.52 |
| 500 | negative_eigenvalue_mass_fraction | 0.167441 | 0.162651 | 0.00478994 | 0.00146343 | 0.00819272 | 0.8 |
| 500 | rank3_residual_energy_fraction | 0.24528 | 0.239206 | 0.00607424 | 0.000519015 | 0.0114279 | 0.68 |
| 500 | classical_mds_stress | 0.126484 | 0.119391 | 0.00709321 | 0.00176412 | 0.0126851 | 0.68 |
| 500 | adjacent_residue_distance_rmse | 0.258177 | 0.30086 | -0.0426828 | -0.0594318 | -0.0242852 | 0.2 |

## Local Versus Global Trade-Offs

H1 is supported for several global geometry metrics: overall improvements are 0.00473265 for negative eigenvalue mass, 0.00676679 for rank-3 residual energy, and 0.00381575 for classical MDS stress. Triangle violation improvement is -0.000268229. For N >= 256, mean improvements by metric are `{'adjacent_residue_distance_rmse': -0.030311480033040947, 'classical_mds_stress': 0.00633260793740921, 'negative_eigenvalue_mass_fraction': 0.007237642891355068, 'rank3_residual_energy_fraction': 0.008368636797839965, 'triangle_violation_fraction': 0.0001513671875}`.

## Distribution Matching

Distribution matching is evaluated as movement toward the corresponding real distribution, not as raw generated metric decrease.

| length | descriptor | baseline_generated_mean | candidate_generated_mean | real_mean | baseline_abs_standardized_mean_discrepancy | candidate_abs_standardized_mean_discrepancy | mean_discrepancy_improvement | baseline_wasserstein_distance | candidate_wasserstein_distance | wasserstein_improvement | baseline_ks_statistic | candidate_ks_statistic | ks_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | adjacent_residue_distance_mean | 3.71568 | 3.59314 | 3.80587 | 0.0901834 | 0.212733 | -0.122549 | 0.159567 | 0.224618 | -0.0650507 | 0.664375 | 0.84 | -0.175625 |
| 64 | contact_fraction_6A | 0.0887401 | 0.0896726 | 0.0817253 | 0.00701476 | 0.0079473 | -0.00093254 | 0.00974701 | 0.0113181 | -0.00157107 | 0.31625 | 0.334375 | -0.018125 |
| 64 | contact_fraction_8A | 0.127743 | 0.134911 | 0.127598 | 0.000145508 | 0.00731317 | -0.00716766 | 0.00333537 | 0.00730271 | -0.00396734 | 0.165625 | 0.1875 | -0.021875 |
| 64 | contact_fraction_10A | 0.201334 | 0.216825 | 0.204454 | 0.00311976 | 0.0123713 | -0.00925156 | 0.00672191 | 0.0124819 | -0.00575995 | 0.175 | 0.184375 | -0.009375 |
| 64 | long_range_contact_fraction | 0.018816 | 0.0267119 | 0.0264903 | 0.0076743 | 0.000221627 | 0.00745267 | 0.00758345 | 0.00463336 | 0.00295009 | 0.285625 | 0.14125 | 0.144375 |
| 64 | distance_mean | 18.7024 | 17.1594 | 19.3094 | 0.606948 | 2.14997 | -1.54302 | 1.31388 | 2.74031 | -1.42642 | 0.19125 | 0.255625 | -0.064375 |
| 64 | distance_std | 9.34219 | 8.35337 | 9.94323 | 0.601031 | 1.58986 | -0.988828 | 0.997611 | 2.10787 | -1.11026 | 0.161875 | 0.190625 | -0.02875 |
| 64 | radius_of_gyration | 14.6995 | 13.4057 | 15.2856 | 0.586095 | 1.87984 | -1.29375 | 1.16379 | 2.40992 | -1.24612 | 0.185625 | 0.22125 | -0.035625 |
| 64 | triangle_violation_fraction | 0.00310547 | 0.00430176 | 0 | 0.00310547 | 0.00430176 | -0.00119629 | 0.00310547 | 0.00430176 | -0.00119629 | 0.82 | 0.9 | -0.08 |
| 64 | triangle_violation_mean | 0.000859773 | 0.00125006 | 0 | 0.000859773 | 0.00125006 | -0.00039029 | 0.000859773 | 0.00125006 | -0.00039029 | 0.82 | 0.9 | -0.08 |
| 64 | negative_eigenvalue_mass_fraction | 0.0486933 | 0.0519124 | 2.64322e-07 | 0.048693 | 0.0519121 | -0.00321908 | 0.048693 | 0.0519121 | -0.00321908 | 1 | 1 | 0 |
| 64 | rank3_residual_energy_fraction | 0.07898 | 0.0759439 | 2.65188e-07 | 0.0789798 | 0.0759436 | 0.00303612 | 0.0789798 | 0.0759436 | 0.00303612 | 1 | 1 | 0 |
| 128 | adjacent_residue_distance_mean | 3.85081 | 3.87117 | 3.80791 | 0.0429059 | 0.0632589 | -0.0203531 | 0.0989855 | 0.109864 | -0.010878 | 0.465 | 0.5275 | -0.0625 |
| 128 | contact_fraction_6A | 0.0383083 | 0.0394882 | 0.0405543 | 0.00224594 | 0.00106607 | 0.00117987 | 0.00317779 | 0.0012302 | 0.00194759 | 0.3225 | 0.144375 | 0.178125 |
| 128 | contact_fraction_8A | 0.0686713 | 0.0674988 | 0.0707085 | 0.00203725 | 0.00320974 | -0.00117249 | 0.00310039 | 0.003817 | -0.000716603 | 0.258125 | 0.349375 | -0.09125 |
| 128 | contact_fraction_10A | 0.115787 | 0.112387 | 0.121088 | 0.00530058 | 0.00870117 | -0.00340059 | 0.00684435 | 0.00955082 | -0.00270647 | 0.24125 | 0.31 | -0.06875 |
| 128 | long_range_contact_fraction | 0.0186792 | 0.014084 | 0.0231119 | 0.00443272 | 0.00902792 | -0.0045952 | 0.00488865 | 0.00902099 | -0.00413234 | 0.286875 | 0.43875 | -0.151875 |
| 128 | distance_mean | 19.8526 | 20.8344 | 21.4557 | 1.60306 | 0.621314 | 0.981749 | 1.54584 | 0.832754 | 0.713088 | 0.19 | 0.119375 | 0.070625 |
| 128 | distance_std | 8.58245 | 9.46977 | 10.136 | 1.55353 | 0.66622 | 0.887313 | 1.46965 | 0.789702 | 0.679943 | 0.226875 | 0.095625 | 0.13125 |
| 128 | radius_of_gyration | 15.2567 | 16.1519 | 16.7716 | 1.51487 | 0.619708 | 0.895159 | 1.45722 | 0.782344 | 0.674878 | 0.194375 | 0.13375 | 0.060625 |

## Diversity

E002 moves the generated/real diversity ratio closer to 1 for 8/20 length-metric rows. Ratios below 1 are reported as reduced diversity, not by themselves as mode collapse.

| domain | requested_length | metric | baseline_generated_mean | candidate_generated_mean | real_mean | baseline_generated_over_real | candidate_generated_over_real | baseline_ratio_distance_from_one | candidate_ratio_distance_from_one | candidate_moves_closer_to_real_ratio | ratio_closeness_improvement | baseline_ci_low | baseline_ci_high | candidate_ci_low | candidate_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diversity | 64 | distance_map_rmse | 0.617019 | 0.544294 | 0.729441 | 0.84588 | 0.74618 | 0.15412 | 0.25382 | False | -0.0996999 | 0.60124 | 0.63402 | 0.534192 | 0.55624 |
| diversity | 64 | contact_hamming_distance | 0.0668522 | 0.0821205 | 0.0851186 | 0.785401 | 0.964778 | 0.214599 | 0.0352217 | True | 0.179378 | 0.064645 | 0.0689654 | 0.0797508 | 0.0846098 |
| diversity | 64 | contact_jaccard_distance | 0.629494 | 0.682824 | 0.759495 | 0.828833 | 0.89905 | 0.171167 | 0.10095 | True | 0.0702173 | 0.613858 | 0.642677 | 0.670492 | 0.695263 |
| diversity | 64 | descriptor_distance | 7.76369 | 4.52313 | 10.526 | 0.737574 | 0.429711 | 0.262426 | 0.570289 | False | -0.307863 | 7.36101 | 8.1459 | 4.29919 | 4.7886 |
| diversity | 128 | distance_map_rmse | 0.533875 | 0.576886 | 0.632262 | 0.844388 | 0.912416 | 0.155612 | 0.0875841 | True | 0.0680276 | 0.526168 | 0.543092 | 0.5649 | 0.590285 |
| diversity | 128 | contact_hamming_distance | 0.0553898 | 0.0458561 | 0.0650064 | 0.852066 | 0.705408 | 0.147934 | 0.294592 | False | -0.146658 | 0.0542496 | 0.0563325 | 0.0447075 | 0.0468553 |
| diversity | 128 | contact_jaccard_distance | 0.827251 | 0.744725 | 0.895233 | 0.924062 | 0.831878 | 0.0759381 | 0.168122 | False | -0.0921842 | 0.817148 | 0.835457 | 0.734491 | 0.754134 |
| diversity | 128 | descriptor_distance | 3.87202 | 5.57799 | 8.15839 | 0.474607 | 0.683712 | 0.525393 | 0.316288 | True | 0.209106 | 3.6226 | 4.12984 | 5.11655 | 5.96373 |
| diversity | 256 | distance_map_rmse | 0.516182 | 0.513706 | 0.548821 | 0.940528 | 0.936017 | 0.0594721 | 0.0639834 | False | -0.00451121 | 0.510548 | 0.52157 | 0.50801 | 0.519453 |
| diversity | 256 | contact_hamming_distance | 0.024658 | 0.0243048 | 0.0348865 | 0.706806 | 0.696682 | 0.293194 | 0.303318 | False | -0.0101239 | 0.0242985 | 0.0250092 | 0.0239127 | 0.0246795 |
| diversity | 256 | contact_jaccard_distance | 0.777718 | 0.780025 | 0.865048 | 0.899046 | 0.901713 | 0.100954 | 0.098287 | True | 0.00266721 | 0.771636 | 0.783834 | 0.77309 | 0.786709 |
| diversity | 256 | descriptor_distance | 5.03301 | 4.96988 | 5.41122 | 0.930105 | 0.918439 | 0.0698949 | 0.0815606 | False | -0.0116657 | 4.74055 | 5.34869 | 4.63121 | 5.26391 |
| diversity | 384 | distance_map_rmse | 0.465203 | 0.474574 | 0.553279 | 0.840811 | 0.857749 | 0.159189 | 0.142251 | True | 0.0169378 | 0.461485 | 0.469463 | 0.469689 | 0.479023 |
| diversity | 384 | contact_hamming_distance | 0.0151023 | 0.0143249 | 0.0254496 | 0.593421 | 0.562873 | 0.406579 | 0.437127 | False | -0.0305489 | 0.0148645 | 0.0153417 | 0.0141513 | 0.0144937 |
| diversity | 384 | contact_jaccard_distance | 0.717658 | 0.713874 | 0.884152 | 0.811691 | 0.807411 | 0.188309 | 0.192589 | False | -0.0042798 | 0.712072 | 0.723835 | 0.709881 | 0.717892 |
| diversity | 384 | descriptor_distance | 3.85302 | 4.26599 | 7.41238 | 0.519808 | 0.575522 | 0.480192 | 0.424478 | True | 0.0557139 | 3.6548 | 4.03638 | 4.0385 | 4.50208 |
| diversity | 500 | distance_map_rmse | 0.463941 | 0.469928 | 0.444954 | 1.04267 | 1.05613 | 0.042671 | 0.0561268 | False | -0.0134558 | 0.457129 | 0.471613 | 0.464197 | 0.476739 |
| diversity | 500 | contact_hamming_distance | 0.0111226 | 0.0108993 | 0.0176046 | 0.6318 | 0.619115 | 0.3682 | 0.380885 | False | -0.0126857 | 0.0107129 | 0.0115026 | 0.0106821 | 0.0110932 |
| diversity | 500 | contact_jaccard_distance | 0.713471 | 0.708246 | 0.805246 | 0.886028 | 0.879539 | 0.113972 | 0.120461 | False | -0.00648881 | 0.703192 | 0.724062 | 0.701406 | 0.714263 |
| diversity | 500 | descriptor_distance | 5.14891 | 4.63626 | 2.8018 | 1.83772 | 1.65474 | 0.837716 | 0.654744 | True | 0.182972 | 4.72802 | 5.63872 | 4.18011 | 5.08732 |

## Novelty

E002 moves calibrated novelty ratios closer to the real calibration baseline for 10/15 rows. Larger novelty distance is not interpreted as automatically better because excessive distance can indicate out-of-distribution invalidity.

| domain | requested_length | metric | baseline_generated_mean | candidate_generated_mean | real_mean | baseline_generated_over_real | candidate_generated_over_real | baseline_ratio_distance_from_one | candidate_ratio_distance_from_one | candidate_moves_closer_to_real_ratio | ratio_closeness_improvement | baseline_ci_low | baseline_ci_high | candidate_ci_low | candidate_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| novelty | 64 | descriptor_distance | 0.768522 | 0.600961 | 0.489536 | 1.5699 | 1.22761 | 0.569898 | 0.227613 | True | 0.342285 | 0.656177 | 0.903122 | 0.524365 | 0.689074 |
| novelty | 64 | refined_distance_map_rmse | 0.281715 | 0.296002 | 0.335591 | 0.839459 | 0.882031 | 0.160541 | 0.117969 | True | 0.0425721 | 0.26046 | 0.302979 | 0.273953 | 0.319823 |
| novelty | 64 | refined_contact_jaccard_distance | 0.609316 | 0.625481 | 0.664296 | 0.917235 | 0.941569 | 0.0827652 | 0.0584309 | True | 0.0243343 | 0.569042 | 0.648549 | 0.584582 | 0.665384 |
| novelty | 128 | descriptor_distance | 0.505287 | 0.548802 | 1.07454 | 0.470237 | 0.510734 | 0.529763 | 0.489266 | True | 0.0404968 | 0.433944 | 0.610651 | 0.458726 | 0.657243 |
| novelty | 128 | refined_distance_map_rmse | 0.354119 | 0.383342 | 0.4296 | 0.824299 | 0.892324 | 0.175701 | 0.107676 | True | 0.0680251 | 0.333284 | 0.373533 | 0.364797 | 0.400442 |
| novelty | 128 | refined_contact_jaccard_distance | 0.756862 | 0.755474 | 0.849519 | 0.890931 | 0.889297 | 0.109069 | 0.110703 | False | -0.00163382 | 0.72861 | 0.785721 | 0.727915 | 0.781205 |
| novelty | 256 | descriptor_distance | 0.791779 | 0.884634 | 0.782007 | 1.0125 | 1.13124 | 0.0124966 | 0.131237 | False | -0.11874 | 0.663178 | 0.933123 | 0.666788 | 1.21427 |
| novelty | 256 | refined_distance_map_rmse | 0.433476 | 0.437433 | 0.440717 | 0.983571 | 0.992549 | 0.0164294 | 0.00745129 | True | 0.00897806 | 0.425692 | 0.440676 | 0.429184 | 0.445062 |
| novelty | 256 | refined_contact_jaccard_distance | 0.838121 | 0.846042 | 0.849482 | 0.986626 | 0.995951 | 0.0133742 | 0.00404947 | True | 0.00932469 | 0.821531 | 0.8539 | 0.834789 | 0.859314 |
| novelty | 384 | descriptor_distance | 1.72197 | 1.91863 | 2.15237 | 0.800037 | 0.891403 | 0.199963 | 0.108597 | True | 0.0913666 | 1.42486 | 2.03112 | 1.6045 | 2.24169 |
| novelty | 384 | refined_distance_map_rmse | 0.427829 | 0.420919 | 0.479565 | 0.892119 | 0.877709 | 0.107881 | 0.122291 | False | -0.0144098 | 0.418158 | 0.439094 | 0.411454 | 0.431522 |
| novelty | 384 | refined_contact_jaccard_distance | 0.856873 | 0.870643 | 0.894379 | 0.958065 | 0.973462 | 0.0419346 | 0.0265382 | True | 0.0153964 | 0.838475 | 0.873999 | 0.853004 | 0.88864 |
| novelty | 500 | descriptor_distance | 3.2537 | 3.31049 | 2.02688 | 1.60527 | 1.63329 | 0.605272 | 0.633293 | False | -0.0280207 | 2.40454 | 4.19196 | 2.54851 | 4.12338 |
| novelty | 500 | refined_distance_map_rmse | 0.429311 | 0.427659 | 0.476448 | 0.901066 | 0.897598 | 0.0989339 | 0.102402 | False | -0.00346814 | 0.417767 | 0.442119 | 0.41432 | 0.442504 |
| novelty | 500 | refined_contact_jaccard_distance | 0.837303 | 0.839821 | 0.903575 | 0.926656 | 0.929443 | 0.0733439 | 0.0705575 | True | 0.00278638 | 0.819382 | 0.856151 | 0.825643 | 0.854026 |

## Hypothesis Evaluation

- H1: Supported for several global geometry diagnostics, particularly negative eigenvalue mass and rank-3 residual at large N.
- H2: Partially supported for heuristic validity, but not for strict empirical real-like geometry, which remains 0/375.
- H3: Supported as no evidence of diversity collapse; diversity ratios remain nonzero and are compared to real-control calibration.
- H4: Supported with caveats; novelty does not indicate exact duplication, but shifts must be read alongside persistent strict-geometry failure.

## Limitations

- Selected-model comparison: E001 epoch 4/global step 160166 versus E002 epoch 5/global step 169181. Selected epochs, optimizer steps, and training histories differ, so this is not a perfectly controlled causal auxiliary-loss ablation.
- The matched-step comparison had only two samples per length and remains exploratory screening.
- Existing calibrated outputs are reused; no ensemble evaluation was rerun.
- Approximate nearest-neighbour novelty remains approximate.
- Strict validity is all-zero, so percent-change summaries are intentionally avoided for that endpoint.

## Decision For Next Experiment

E002 should be interpreted against E001 using the paired metrics above. Any next experiment should be justified from the full pattern of global geometry, local geometry, validity, diversity, and novelty diagnostics rather than a single scalar.
