# E001 Versus E000 Calibrated Ensemble Comparison

## Executive Conclusion

This selected-model comparison finds that E001 improves several global geometry diagnostics relative to E000, especially negative eigenvalue mass and rank-3 residual energy, while both models still fail strict empirical real-like geometry for all 375 paired generated samples. E001 increases heuristic EDM-quality passes and remains diverse, but the result is not a perfect causal architecture ablation because the selected checkpoints differ in training history.

## Protocol Compatibility

- Compatibility status: `True`
- Paired key: `['requested_length', 'sample_index', 'seed']`
- Paired sample count: `375`
- Counts by length: `{'64': 100, '128': 100, '256': 100, '384': 50, '500': 25}`
- Input hashes preserved: `True`

## Checkpoint And Training-History Caveat

The complete ensembles compare selected models: E000 selected checkpoint epoch 8, and E001 selected checkpoint epoch 4 at global_step 160166. This is a selected-model comparison, not a perfectly controlled causal architecture ablation. The earlier matched-step, two-samples-per-length comparison should be treated as exploratory screening, not formal statistical evidence.

## Paired Primary Results

| metric | baseline_mean | candidate_mean | mean_improvement_baseline_minus_candidate | bootstrap_ci_low | bootstrap_ci_high | fraction_improved | paired_standardized_effect_size | sign_test_p_value_bh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| triangle_violation_fraction | 0.00564844 | 0.00192448 | 0.00372396 | 0.00304684 | 0.00444538 | 0.736 | 0.510207 | 9.64357e-15 |
| negative_eigenvalue_mass_fraction | 0.132981 | 0.0974289 | 0.0355518 | 0.0337958 | 0.0372658 | 0.952 | 1.40055 | 7.16759e-15 |
| rank3_residual_energy_fraction | 0.189246 | 0.154451 | 0.0347947 | 0.0312829 | 0.0384539 | 0.76 | 0.780951 | 7.16759e-15 |
| classical_mds_stress | 0.0949313 | 0.075929 | 0.0190023 | 0.0160464 | 0.0217924 | 0.698667 | 0.514353 | 2.84172e-14 |
| adjacent_residue_distance_rmse | 0.327114 | 0.252429 | 0.074685 | 0.0610191 | 0.0891523 | 0.672 | 0.43095 | 5.28333e-11 |

Positive improvement means E001 has the lower value for these lower-is-better metrics.

## Validity Transitions

| group | requested_length | flag | baseline_fail_candidate_fail | baseline_fail_candidate_pass | baseline_pass_candidate_fail | baseline_pass_candidate_pass | discordant_candidate_better | discordant_baseline_better | mcnemar_exact_p_value | baseline_pass_fraction | candidate_pass_fraction | mcnemar_exact_p_value_bh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall |  | empirical_real_like_geometry_pass | 375 | 0 | 0 | 0 | 0 | 0 |  | 0 | 0 |  |
| overall |  | heuristic_edm_quality_pass | 319 | 44 | 2 | 10 | 44 | 2 | 3.07523e-11 | 0.032 | 0.144 | 8.83347e-11 |

Strict empirical real-like geometry remains zero-pass: E000 pass fraction 0, E001 pass fraction 0. Heuristic EDM-quality pass fraction changes from 0.032 to 0.144.

## Length-Dependent Effects

| requested_length | metric | baseline_mean | candidate_mean | mean_improvement_baseline_minus_candidate | bootstrap_ci_low | bootstrap_ci_high | fraction_improved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | triangle_violation_fraction | 0.00949707 | 0.00310547 | 0.0063916 | 0.00399402 | 0.00901416 | 0.66 |
| 64 | negative_eigenvalue_mass_fraction | 0.0699479 | 0.0486933 | 0.0212546 | 0.0180909 | 0.0247344 | 0.91 |
| 64 | rank3_residual_energy_fraction | 0.0961428 | 0.07898 | 0.0171628 | 0.0099715 | 0.0243812 | 0.64 |
| 64 | classical_mds_stress | 0.0481718 | 0.0395467 | 0.00862511 | 0.00483496 | 0.0126087 | 0.66 |
| 64 | adjacent_residue_distance_rmse | 0.534919 | 0.286774 | 0.248145 | 0.203054 | 0.292542 | 0.87 |
| 128 | triangle_violation_fraction | 0.00153809 | 0.00125488 | 0.000283203 | -0.000175903 | 0.000717773 | 0.47 |
| 128 | negative_eigenvalue_mass_fraction | 0.100467 | 0.0831648 | 0.0173027 | 0.0145455 | 0.0203271 | 0.91 |
| 128 | rank3_residual_energy_fraction | 0.141592 | 0.133561 | 0.00803028 | 0.00148419 | 0.0143788 | 0.58 |
| 128 | classical_mds_stress | 0.0531255 | 0.0603264 | -0.00720094 | -0.0125957 | -0.00190297 | 0.48 |
| 128 | adjacent_residue_distance_rmse | 0.236607 | 0.224176 | 0.0124311 | -0.00751748 | 0.0340697 | 0.55 |
| 256 | triangle_violation_fraction | 0.00424805 | 0.0012793 | 0.00296875 | 0.00251953 | 0.00342773 | 0.89 |
| 256 | negative_eigenvalue_mass_fraction | 0.162386 | 0.116703 | 0.0456827 | 0.0420967 | 0.0491746 | 1 |
| 256 | rank3_residual_energy_fraction | 0.234042 | 0.187899 | 0.0461427 | 0.0384626 | 0.054123 | 0.89 |
| 256 | classical_mds_stress | 0.118785 | 0.0923384 | 0.0264465 | 0.0198863 | 0.0332664 | 0.77 |
| 256 | adjacent_residue_distance_rmse | 0.252599 | 0.241845 | 0.010754 | -0.00548978 | 0.0272717 | 0.63 |
| 384 | triangle_violation_fraction | 0.00772461 | 0.00170898 | 0.00601563 | 0.00517554 | 0.00695313 | 0.98 |
| 384 | negative_eigenvalue_mass_fraction | 0.21272 | 0.149874 | 0.0628456 | 0.0576788 | 0.0682593 | 1 |
| 384 | rank3_residual_energy_fraction | 0.303953 | 0.234862 | 0.0690909 | 0.0606089 | 0.0774521 | 0.98 |
| 384 | classical_mds_stress | 0.170377 | 0.121802 | 0.048575 | 0.0405176 | 0.056912 | 0.92 |
| 384 | adjacent_residue_distance_rmse | 0.277715 | 0.25854 | 0.0191755 | 0.000837748 | 0.0371921 | 0.68 |
| 500 | triangle_violation_fraction | 0.00814453 | 0.00289062 | 0.00525391 | 0.00427734 | 0.00623047 | 1 |
| 500 | negative_eigenvalue_mass_fraction | 0.238067 | 0.167441 | 0.0706262 | 0.0634891 | 0.077862 | 1 |
| 500 | rank3_residual_energy_fraction | 0.343675 | 0.24528 | 0.0983952 | 0.0862768 | 0.111267 | 1 |
| 500 | classical_mds_stress | 0.202886 | 0.126484 | 0.0764021 | 0.065933 | 0.0871818 | 1 |
| 500 | adjacent_residue_distance_rmse | 0.254781 | 0.258177 | -0.00339591 | -0.0199121 | 0.0110105 | 0.52 |

## Local Versus Global Trade-Offs

H1 is supported for several global geometry metrics: overall improvements are 0.0355518 for negative eigenvalue mass, 0.0347947 for rank-3 residual energy, and 0.0190023 for classical MDS stress. Triangle violation improvement is 0.00372396. For N >= 256, mean improvements by metric are `{'adjacent_residue_distance_rmse': 0.008844512931665644, 'classical_mds_stress': 0.05047452594572912, 'negative_eigenvalue_mass_fraction': 0.05971817560277212, 'rank3_residual_energy_fraction': 0.07120958951958913, 'triangle_violation_fraction': 0.00474609375}`.

## Distribution Matching

Distribution matching is evaluated as movement toward the corresponding real distribution, not as raw generated metric decrease.

| length | descriptor | baseline_generated_mean | candidate_generated_mean | real_mean | baseline_abs_standardized_mean_discrepancy | candidate_abs_standardized_mean_discrepancy | mean_discrepancy_improvement | baseline_wasserstein_distance | candidate_wasserstein_distance | wasserstein_improvement | baseline_ks_statistic | candidate_ks_statistic | ks_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | adjacent_residue_distance_mean | 3.2963 | 3.71568 | 3.80587 | 0.509563 | 0.0901834 | 0.41938 | 0.509481 | 0.159567 | 0.349914 | 1 | 0.664375 | 0.335625 |
| 64 | contact_fraction_6A | 0.0942609 | 0.0887401 | 0.0817253 | 0.0125356 | 0.00701476 | 0.00552083 | 0.0149504 | 0.00974701 | 0.00520337 | 0.45875 | 0.31625 | 0.1425 |
| 64 | contact_fraction_8A | 0.148497 | 0.127743 | 0.127598 | 0.0208995 | 0.000145508 | 0.020754 | 0.0208525 | 0.00333537 | 0.0175171 | 0.5025 | 0.165625 | 0.336875 |
| 64 | contact_fraction_10A | 0.248447 | 0.201334 | 0.204454 | 0.0439933 | 0.00311976 | 0.0408736 | 0.0439632 | 0.00672191 | 0.0372413 | 0.53 | 0.175 | 0.355 |
| 64 | long_range_contact_fraction | 0.045279 | 0.018816 | 0.0264903 | 0.0187887 | 0.0076743 | 0.0111144 | 0.0188796 | 0.00758345 | 0.0112961 | 0.318125 | 0.285625 | 0.0325 |
| 64 | distance_mean | 14.9264 | 18.7024 | 19.3094 | 4.38295 | 0.606948 | 3.77601 | 4.35728 | 1.31388 | 3.0434 | 0.465625 | 0.19125 | 0.274375 |
| 64 | distance_std | 6.6545 | 9.34219 | 9.94323 | 3.28873 | 0.601031 | 2.6877 | 3.26852 | 0.997611 | 2.27091 | 0.373125 | 0.161875 | 0.21125 |
| 64 | radius_of_gyration | 11.4809 | 14.6995 | 15.2856 | 3.80469 | 0.586095 | 3.21859 | 3.78043 | 1.16379 | 2.61664 | 0.465625 | 0.185625 | 0.28 |
| 64 | triangle_violation_fraction | 0.00949707 | 0.00310547 | 0 | 0.00949707 | 0.00310547 | 0.0063916 | 0.00949707 | 0.00310547 | 0.0063916 | 0.91 | 0.82 | 0.09 |
| 64 | triangle_violation_mean | 0.0041002 | 0.000859773 | 0 | 0.0041002 | 0.000859773 | 0.00324043 | 0.0041002 | 0.000859773 | 0.00324043 | 0.91 | 0.82 | 0.09 |
| 64 | negative_eigenvalue_mass_fraction | 0.0699479 | 0.0486933 | 2.64322e-07 | 0.0699476 | 0.048693 | 0.0212546 | 0.0699476 | 0.048693 | 0.0212546 | 1 | 1 | 0 |
| 64 | rank3_residual_energy_fraction | 0.0961428 | 0.07898 | 2.65188e-07 | 0.0961426 | 0.0789798 | 0.0171628 | 0.0961426 | 0.0789798 | 0.0171628 | 1 | 1 | 0 |
| 128 | adjacent_residue_distance_mean | 3.63401 | 3.85081 | 3.80791 | 0.173894 | 0.0429059 | 0.130988 | 0.173958 | 0.0989855 | 0.0749726 | 0.90875 | 0.465 | 0.44375 |
| 128 | contact_fraction_6A | 0.0369697 | 0.0383083 | 0.0405543 | 0.00358452 | 0.00224594 | 0.00133858 | 0.00432355 | 0.00317779 | 0.00114576 | 0.425 | 0.3225 | 0.1025 |
| 128 | contact_fraction_8A | 0.0739567 | 0.0686713 | 0.0707085 | 0.00324819 | 0.00203725 | 0.00121094 | 0.00320531 | 0.00310039 | 0.000104912 | 0.209375 | 0.258125 | -0.04875 |
| 128 | contact_fraction_10A | 0.131893 | 0.115787 | 0.121088 | 0.0108055 | 0.00530058 | 0.00550489 | 0.0107291 | 0.00684435 | 0.00388478 | 0.32375 | 0.24125 | 0.0825 |
| 128 | long_range_contact_fraction | 0.0291679 | 0.0186792 | 0.0231119 | 0.00605603 | 0.00443272 | 0.00162331 | 0.00607278 | 0.00488865 | 0.00118414 | 0.390625 | 0.286875 | 0.10375 |
| 128 | distance_mean | 17.8523 | 19.8526 | 21.4557 | 3.60339 | 1.60306 | 2.00033 | 3.50374 | 1.54584 | 1.95789 | 0.649375 | 0.19 | 0.459375 |
| 128 | distance_std | 7.05656 | 8.58245 | 10.136 | 3.07943 | 1.55353 | 1.52589 | 2.98538 | 1.46965 | 1.51573 | 0.596875 | 0.226875 | 0.37 |
| 128 | radius_of_gyration | 13.5251 | 15.2567 | 16.7716 | 3.24646 | 1.51487 | 1.73159 | 3.1516 | 1.45722 | 1.69437 | 0.6325 | 0.194375 | 0.438125 |

## Diversity

E001 moves the generated/real diversity ratio closer to 1 for 11/20 length-metric rows. Ratios below 1 are reported as reduced diversity, not by themselves as mode collapse.

| domain | requested_length | metric | baseline_generated_mean | candidate_generated_mean | real_mean | baseline_generated_over_real | candidate_generated_over_real | baseline_ratio_distance_from_one | candidate_ratio_distance_from_one | candidate_moves_closer_to_real_ratio | ratio_closeness_improvement | baseline_ci_low | baseline_ci_high | candidate_ci_low | candidate_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diversity | 64 | distance_map_rmse | 0.478693 | 0.617019 | 0.729441 | 0.656247 | 0.84588 | 0.343753 | 0.15412 | True | 0.189633 | 0.470772 | 0.487906 | 0.60124 | 0.63402 |
| diversity | 64 | contact_hamming_distance | 0.111071 | 0.0668522 | 0.0851186 | 1.3049 | 0.785401 | 0.304903 | 0.214599 | True | 0.0903036 | 0.108833 | 0.113421 | 0.064645 | 0.0689654 |
| diversity | 64 | contact_jaccard_distance | 0.76577 | 0.629494 | 0.759495 | 1.00826 | 0.828833 | 0.00826236 | 0.171167 | False | -0.162905 | 0.756263 | 0.776284 | 0.613858 | 0.642677 |
| diversity | 64 | descriptor_distance | 2.91983 | 7.76369 | 10.526 | 0.277392 | 0.737574 | 0.722608 | 0.262426 | True | 0.460182 | 2.80228 | 3.05654 | 7.36101 | 8.1459 |
| diversity | 128 | distance_map_rmse | 0.381609 | 0.533875 | 0.632262 | 0.603561 | 0.844388 | 0.396439 | 0.155612 | True | 0.240827 | 0.372249 | 0.391764 | 0.526168 | 0.543092 |
| diversity | 128 | contact_hamming_distance | 0.0597721 | 0.0553898 | 0.0650064 | 0.919481 | 0.852066 | 0.080519 | 0.147934 | False | -0.0674146 | 0.058286 | 0.061197 | 0.0542496 | 0.0563325 |
| diversity | 128 | contact_jaccard_distance | 0.79062 | 0.827251 | 0.895233 | 0.883145 | 0.924062 | 0.116855 | 0.0759381 | True | 0.0409173 | 0.776977 | 0.802671 | 0.817148 | 0.835457 |
| diversity | 128 | descriptor_distance | 1.25969 | 3.87202 | 8.15839 | 0.154404 | 0.474607 | 0.845596 | 0.525393 | True | 0.320202 | 1.18518 | 1.33181 | 3.6226 | 4.12984 |
| diversity | 256 | distance_map_rmse | 0.416249 | 0.516182 | 0.548821 | 0.758441 | 0.940528 | 0.241559 | 0.0594721 | True | 0.182087 | 0.413283 | 0.418901 | 0.510548 | 0.52157 |
| diversity | 256 | contact_hamming_distance | 0.0316379 | 0.024658 | 0.0348865 | 0.90688 | 0.706806 | 0.0931204 | 0.293194 | False | -0.200073 | 0.0311801 | 0.0320431 | 0.0242985 | 0.0250092 |
| diversity | 256 | contact_jaccard_distance | 0.800545 | 0.777718 | 0.865048 | 0.925434 | 0.899046 | 0.074566 | 0.100954 | False | -0.0263882 | 0.795612 | 0.804923 | 0.771636 | 0.783834 |
| diversity | 256 | descriptor_distance | 1.27574 | 5.03301 | 5.41122 | 0.235759 | 0.930105 | 0.764241 | 0.0698949 | True | 0.694346 | 1.21711 | 1.34888 | 4.74055 | 5.34869 |
| diversity | 384 | distance_map_rmse | 0.376415 | 0.465203 | 0.553279 | 0.680335 | 0.840811 | 0.319665 | 0.159189 | True | 0.160476 | 0.374604 | 0.378158 | 0.461485 | 0.469463 |
| diversity | 384 | contact_hamming_distance | 0.0299301 | 0.0151023 | 0.0254496 | 1.17606 | 0.593421 | 0.176056 | 0.406579 | False | -0.230523 | 0.029475 | 0.0303819 | 0.0148645 | 0.0153417 |
| diversity | 384 | contact_jaccard_distance | 0.870592 | 0.717658 | 0.884152 | 0.984663 | 0.811691 | 0.0153368 | 0.188309 | False | -0.172972 | 0.865649 | 0.875412 | 0.712072 | 0.723835 |
| diversity | 384 | descriptor_distance | 0.708479 | 3.85302 | 7.41238 | 0.0955805 | 0.519808 | 0.90442 | 0.480192 | True | 0.424228 | 0.686586 | 0.731919 | 3.6548 | 4.03638 |
| diversity | 500 | distance_map_rmse | 0.361624 | 0.463941 | 0.444954 | 0.812721 | 1.04267 | 0.187279 | 0.042671 | True | 0.144608 | 0.3582 | 0.364687 | 0.457129 | 0.471613 |
| diversity | 500 | contact_hamming_distance | 0.0238634 | 0.0111226 | 0.0176046 | 1.35552 | 0.6318 | 0.355518 | 0.3682 | False | -0.0126813 | 0.0230695 | 0.0245257 | 0.0107129 | 0.0115026 |
| diversity | 500 | contact_jaccard_distance | 0.866652 | 0.713471 | 0.805246 | 1.07626 | 0.886028 | 0.0762571 | 0.113972 | False | -0.0377148 | 0.857051 | 0.875434 | 0.703192 | 0.724062 |
| diversity | 500 | descriptor_distance | 0.760388 | 5.14891 | 2.8018 | 0.271393 | 1.83772 | 0.728607 | 0.837716 | False | -0.109108 | 0.700441 | 0.831203 | 4.72802 | 5.63872 |

## Novelty

E001 moves calibrated novelty ratios closer to the real calibration baseline for 9/15 rows. Larger novelty distance is not interpreted as automatically better because excessive distance can indicate out-of-distribution invalidity.

| domain | requested_length | metric | baseline_generated_mean | candidate_generated_mean | real_mean | baseline_generated_over_real | candidate_generated_over_real | baseline_ratio_distance_from_one | candidate_ratio_distance_from_one | candidate_moves_closer_to_real_ratio | ratio_closeness_improvement | baseline_ci_low | baseline_ci_high | candidate_ci_low | candidate_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| novelty | 64 | descriptor_distance | 0.761576 | 0.768522 | 0.489536 | 1.55571 | 1.5699 | 0.55571 | 0.569898 | False | -0.0141886 | 0.680765 | 0.850418 | 0.656177 | 0.903122 |
| novelty | 64 | refined_distance_map_rmse | 0.3113 | 0.281715 | 0.335591 | 0.927618 | 0.839459 | 0.0723818 | 0.160541 | False | -0.0881592 | 0.290755 | 0.332781 | 0.26046 | 0.302979 |
| novelty | 64 | refined_contact_jaccard_distance | 0.695431 | 0.609316 | 0.664296 | 1.04687 | 0.917235 | 0.0468692 | 0.0827652 | False | -0.035896 | 0.663053 | 0.72497 | 0.569042 | 0.648549 |
| novelty | 128 | descriptor_distance | 0.383097 | 0.505287 | 1.07454 | 0.356523 | 0.470237 | 0.643477 | 0.529763 | True | 0.113713 | 0.342504 | 0.420912 | 0.433944 | 0.610651 |
| novelty | 128 | refined_distance_map_rmse | 0.312741 | 0.354119 | 0.4296 | 0.727983 | 0.824299 | 0.272017 | 0.175701 | True | 0.0963159 | 0.281288 | 0.341426 | 0.333284 | 0.373533 |
| novelty | 128 | refined_contact_jaccard_distance | 0.726995 | 0.756862 | 0.849519 | 0.855773 | 0.890931 | 0.144227 | 0.109069 | True | 0.0351578 | 0.687192 | 0.769151 | 0.72861 | 0.785721 |
| novelty | 256 | descriptor_distance | 1.31886 | 0.791779 | 0.782007 | 1.6865 | 1.0125 | 0.686503 | 0.0124966 | True | 0.674007 | 1.20112 | 1.45896 | 0.663178 | 0.933123 |
| novelty | 256 | refined_distance_map_rmse | 0.392382 | 0.433476 | 0.440717 | 0.890328 | 0.983571 | 0.109672 | 0.0164294 | True | 0.0932423 | 0.38552 | 0.398441 | 0.425692 | 0.440676 |
| novelty | 256 | refined_contact_jaccard_distance | 0.856249 | 0.838121 | 0.849482 | 1.00797 | 0.986626 | 0.0079664 | 0.0133742 | False | -0.00540776 | 0.847331 | 0.865798 | 0.821531 | 0.8539 |
| novelty | 384 | descriptor_distance | 5.03042 | 1.72197 | 2.15237 | 2.33716 | 0.800037 | 1.33716 | 0.199963 | True | 1.1372 | 4.67575 | 5.34982 | 1.42486 | 2.03112 |
| novelty | 384 | refined_distance_map_rmse | 0.420654 | 0.427829 | 0.479565 | 0.877157 | 0.892119 | 0.122843 | 0.107881 | True | 0.0149617 | 0.417203 | 0.423554 | 0.418158 | 0.439094 |
| novelty | 384 | refined_contact_jaccard_distance | 0.904005 | 0.856873 | 0.894379 | 1.01076 | 0.958065 | 0.0107632 | 0.0419346 | False | -0.0311714 | 0.890452 | 0.919299 | 0.838475 | 0.873999 |
| novelty | 500 | descriptor_distance | 7.04433 | 3.2537 | 2.02688 | 3.47545 | 1.60527 | 2.47545 | 0.605272 | True | 1.87018 | 6.73536 | 7.38704 | 2.40454 | 4.19196 |
| novelty | 500 | refined_distance_map_rmse | 0.421406 | 0.429311 | 0.476448 | 0.884474 | 0.901066 | 0.115526 | 0.0989339 | True | 0.0165918 | 0.415825 | 0.427488 | 0.417767 | 0.442119 |
| novelty | 500 | refined_contact_jaccard_distance | 0.887526 | 0.837303 | 0.903575 | 0.982238 | 0.926656 | 0.0177617 | 0.0733439 | False | -0.0555822 | 0.870243 | 0.905064 | 0.819382 | 0.856151 |

## Hypothesis Evaluation

- H1: Supported for several global geometry diagnostics, particularly negative eigenvalue mass and rank-3 residual at large N.
- H2: Partially supported for heuristic validity, but not for strict empirical real-like geometry, which remains 0/375.
- H3: Supported as no evidence of diversity collapse; diversity ratios remain nonzero and are compared to real-control calibration.
- H4: Supported with caveats; novelty does not indicate exact duplication, but shifts must be read alongside persistent strict-geometry failure.

## Limitations

- This is selected-model evidence, not a perfectly controlled architecture-only ablation.
- The matched-step comparison had only two samples per length and remains exploratory screening.
- Existing calibrated outputs are reused; no ensemble evaluation was rerun.
- Approximate nearest-neighbour novelty remains approximate.
- Strict validity is all-zero, so percent-change summaries are intentionally avoided for that endpoint.

## Decision For Next Experiment

E001 provides enough execution and selected-model quality evidence to proceed, but it still does not reach the empirical 3D distance-matrix manifold. The next scientifically justified intervention is a bounded physical auxiliary-loss experiment under the same calibrated evaluation protocol, while retaining E000 and E001 as explicit baselines.
