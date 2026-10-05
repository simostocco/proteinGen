# E010 Phase 4C short matched continuation


## 1. Executive classification

C1 — SHORT-TRAINING SUCCESS
Primary endpoint: auxiliary vs matched Cartesian control after exactly 364 continuation updates per arm.

## 2. Contract and reproducibility

Execution Git SHA: a91d3432ec461f334ca5d9dd27fec9efd19b61d4; public reviewed baseline: 3b200ffb0389be87cc928c391117ce96bd8b5eb2.
Starting checkpoint SHA256: f5211cbc1be5092175ce15b9761a4efba761d242310287cd6b1e04df6a6744ef. Lambda: 0.067194076521841617.
Contract SHA256: 4994479030542d0bc9e0f0e78a1d16ea437fcc8fd29c93018189d8390309d956; schedule SHA256: f8944455b46394269a32328c35db0cb77e407f310af814cb7fb689a350cba7b0.
Training: 32760 cached examples per arm, 14668 identities; fixed historical first364 batches, five18-protein stratum microbatches. Realized conditions: {'50': 10886, '250': 11011, '450': 10863}.
Evaluation: all 320 existing fixed development identities, 960 condition records. Same arrays/order/seeds across arms; no regeneration. Optimizer: saved1092 AdamW ->1456; constant LR0.0003, clip5, float32, no AMP.

## 3. Matched update-zero verification

{"metrics_exactly_equal": true, "model_optimizer_exactly_equal": true, "schedule_sha256": "f8944455b46394269a32328c35db0cb77e407f310af814cb7fb689a350cba7b0"}

## 4. Training trajectories

| Arm | Boundary | Interval total | Cartesian | Local MSE | i+1 | i+2 | i+3 | Gradient norm | Clip |
|---|---|---|---|---|---|---|---|---|---|
| cartesian_control | 91 | 33.804883 | 33.804883 | 10.393505 | 4.3905559 | 10.362698 | 16.42726 | 4.3611574 | 0.95514421 |
| cartesian_control | 182 | 34.774999 | 34.774999 | 10.543903 | 4.5061583 | 10.504207 | 16.621343 | 4.7754112 | 0.92924838 |
| cartesian_control | 273 | 33.663138 | 33.663138 | 10.445327 | 4.4262013 | 10.39223 | 16.517549 | 4.2629705 | 0.96182439 |
| cartesian_control | 364 | 33.224615 | 33.224615 | 10.372119 | 4.4068823 | 10.345441 | 16.364032 | 4.3316096 | 0.9534501 |
| local_auxiliary | 91 | 34.436667 | 33.825607 | 9.0939543 | 3.5778032 | 8.8971745 | 14.806884 | 4.4324289 | 0.95320349 |
| local_auxiliary | 182 | 35.444418 | 34.837219 | 9.0364898 | 3.6015204 | 8.7990374 | 14.708911 | 4.6249409 | 0.93959863 |
| local_auxiliary | 273 | 34.292053 | 33.69155 | 8.9368467 | 3.5171382 | 8.6878106 | 14.60559 | 4.1004586 | 0.97064079 |
| local_auxiliary | 364 | 33.869425 | 33.277066 | 8.8156336 | 3.4736128 | 8.5894163 | 14.383871 | 4.3263921 | 0.952537 |

## 5. Update-364 primary comparison

{
  "classification": "C1",
  "condition_local_improvement": {
    "250": 0.08073359242184328,
    "450": 0.10986279606510488,
    "50": 0.14681024052829555
  },
  "gates": {
    "1": true,
    "2": true,
    "3": true,
    "4": true,
    "5": true,
    "6": true
  },
  "legacy_length_slope_not_increased": false,
  "paired_cartesian": {
    "bootstrap_ci95": [
      0.0020617896649675472,
      0.0042340245196461155
    ],
    "identity_count": 320,
    "mean_paired_percentage_improvement": 0.003142906113410962
  },
  "paired_mean_local": {
    "bootstrap_ci95": [
      0.10104349188902517,
      0.10928822867003798
    ],
    "identity_count": 320,
    "mean_paired_percentage_improvement": 0.10510443089350682
  },
  "stratum_cartesian_improvement": {
    "129-256": 0.004563990968987683,
    "20-64": 0.010151187661345214,
    "257-384": 0.0011095472778981511,
    "385-500": -0.005705670196256153,
    "65-128": 0.005735496523037305
  },
  "stratum_local_improvement": {
    "129-256": 0.12342315737045585,
    "20-64": 0.12812843748723374,
    "257-384": 0.06459474676908766,
    "385-500": 0.0759319987140958,
    "65-128": 0.1348023667808841
  }
}

## 6. Cartesian results

| Update | Control RMSD Å | Aux RMSD Å | Paired improvement | CI95 | Control Cartesian Å² | Aux Cartesian Å² |
|---|---|---|---|---|---|---|
| 0 | 7.768927 | 7.768927 | 0 | [0.0, 0.0] | 34.036246 | 34.036246 |
| 91 | 7.7757892 | 7.7474856 | 0.0039336861 | [0.0032198945713484218, 0.0046716981880869715] | 33.95677 | 34.040204 |
| 182 | 7.7513161 | 7.7337691 | 0.0023084647 | [0.0016936394277896314, 0.002949417290947594] | 33.951879 | 34.011904 |
| 273 | 7.7286857 | 7.719633 | 0.0012601085 | [0.00035538696237256187, 0.0021580496532080406] | 33.950812 | 33.974995 |
| 364 | 7.7584624 | 7.7387811 | 0.0031429061 | [0.0020617896649675472, 0.0042340245196461155] | 33.888416 | 33.954393 |

## 7. Local offset results

| Update | Arm | i+1 RMSE Å | i+2 RMSE Å | i+3 RMSE Å | Mean local Å |
|---|---|---|---|---|---|
| 0 | cartesian_control | 1.9046374 | 2.8607132 | 3.4770086 | 2.7474531 |
| 0 | local_auxiliary | 1.9046374 | 2.8607132 | 3.4770086 | 2.7474531 |
| 91 | cartesian_control | 2.0628475 | 2.9817519 | 3.5462886 | 2.8636293 |
| 91 | local_auxiliary | 1.8016444 | 2.6529193 | 3.2640509 | 2.5728715 |
| 182 | cartesian_control | 2.0133415 | 2.9610561 | 3.5599679 | 2.8447885 |
| 182 | local_auxiliary | 1.7320734 | 2.6465301 | 3.3042186 | 2.5609407 |
| 273 | cartesian_control | 1.9820643 | 2.902792 | 3.4796214 | 2.7881592 |
| 273 | local_auxiliary | 1.7447778 | 2.6175536 | 3.2330467 | 2.5317927 |
| 364 | cartesian_control | 2.1046183 | 3.085889 | 3.6672653 | 2.9525909 |
| 364 | local_auxiliary | 1.8226894 | 2.7263167 | 3.3614068 | 2.6368043 |

## 8. Condition-specific endpoint

| Condition | Arm | Cartesian RMSD Å | i+1 Å | i+2 Å | i+3 Å | Mean local Å | Chirality |
|---|---|---|---|---|---|---|---|
| 50 | cartesian_control | 2.2865784 | 1.4731779 | 1.6929834 | 1.6137645 | 1.5933086 | 0.28201115 |
| 50 | local_auxiliary | 2.2375523 | 1.2541197 | 1.4095365 | 1.4145276 | 1.3593946 | 0.27466678 |
| 250 | cartesian_control | 6.5165455 | 2.1650122 | 3.0358596 | 3.5170433 | 2.9059717 | 0.49563881 |
| 250 | local_auxiliary | 6.5237651 | 1.9384008 | 2.7884623 | 3.2872234 | 2.6713622 | 0.49809932 |
| 450 | cartesian_control | 14.472263 | 2.6756648 | 4.5288242 | 5.870988 | 4.3584923 | 0.49584801 |
| 450 | local_auxiliary | 14.455026 | 2.2755478 | 3.9809512 | 5.3824695 | 3.8796562 | 0.50023096 |

## 9. Length-stratum endpoint

| Stratum | Arm | Cartesian RMSD Å | i+1 Å | i+2 Å | i+3 Å | Mean local Å | Chirality |
|---|---|---|---|---|---|---|---|
| 129-256 | cartesian_control | 8.0454551 | 2.0995861 | 3.1177627 | 3.7901706 | 3.0025064 | 0.43173983 |
| 129-256 | local_auxiliary | 8.0087357 | 1.7871321 | 2.6992385 | 3.4094122 | 2.6319276 | 0.42782242 |
| 20-64 | cartesian_control | 6.0209308 | 2.3657204 | 3.3322159 | 3.7426299 | 3.1468554 | 0.42143962 |
| 20-64 | local_auxiliary | 5.9598112 | 1.9581154 | 2.8775439 | 3.3953018 | 2.7436537 | 0.41964556 |
| 257-384 | cartesian_control | 8.7231037 | 1.9310213 | 2.8814151 | 3.4651802 | 2.7592055 | 0.42389933 |
| 257-384 | local_auxiliary | 8.713425 | 1.7721277 | 2.6786139 | 3.2921845 | 2.5809754 | 0.42181688 |
| 385-500 | cartesian_control | 8.8186468 | 1.9245739 | 2.8716347 | 3.472623 | 2.7562772 | 0.42226325 |
| 385-500 | local_auxiliary | 8.8689631 | 1.7542803 | 2.6287254 | 3.2579569 | 2.5469875 | 0.42318233 |
| 65-128 | cartesian_control | 7.1841755 | 2.2021898 | 3.2264169 | 3.8657227 | 3.0981098 | 0.42315457 |
| 65-128 | local_auxiliary | 7.1429707 | 1.8417916 | 2.7474615 | 3.4521787 | 2.6804773 | 0.42919457 |

## 10. Chirality and safety

| Update | Arm | Chirality | Eligible tetrahedra | Finite rate | Length slope |
|---|---|---|---|---|---|
| 0 | cartesian_control | 0.42409782 | 203064 | 1 | 0.007076382 |
| 0 | local_auxiliary | 0.42409782 | 203064 | 1 | 0.007076382 |
| 91 | cartesian_control | 0.4286555 | 203064 | 1 | 0.0074697785 |
| 91 | local_auxiliary | 0.42580607 | 203064 | 1 | 0.0075639274 |
| 182 | cartesian_control | 0.42711604 | 203064 | 1 | 0.0072627917 |
| 182 | local_auxiliary | 0.42394405 | 203064 | 1 | 0.0072745428 |
| 273 | cartesian_control | 0.42769224 | 203064 | 1 | 0.007026925 |
| 273 | local_auxiliary | 0.42395242 | 203064 | 1 | 0.0070108588 |
| 364 | cartesian_control | 0.42449932 | 203064 | 1 | 0.0066760799 |
| 364 | local_auxiliary | 0.42433235 | 203064 | 1 | 0.0069444771 |

Exact pre-registered Gates1–6: {"1": true, "2": true, "3": true, "4": true, "5": true, "6": true}
The reflection-inclusive E(3) architecture is unchanged; the local auxiliary does not identify handedness.

## 11. Gradient-conflict trajectory

| Update | Arm | Raw cart/local cosine | Cartesian grad norm | Local grad norm | Total grad norm | Clip |
|---|---|---|---|---|---|---|
| 0 | cartesian_control | -0.69875418 | 8.3345281 | 62.018323 | 8.3345281 | 0.59991392 |
| 0 | local_auxiliary | -0.69875418 | 8.3345281 | 62.018323 | 6.188048 | 0.80800912 |
| 91 | cartesian_control | -0.31259083 | 5.8966393 | 37.149433 | 5.8966393 | 0.84794047 |
| 91 | local_auxiliary | -0.81340884 | 8.7254373 | 55.888549 | 6.0769782 | 0.82277721 |
| 182 | cartesian_control | -0.55634851 | 6.9963076 | 47.363368 | 6.9963076 | 0.71466258 |
| 182 | local_auxiliary | -0.79223976 | 9.650142 | 61.695802 | 6.8500507 | 0.7299215 |
| 273 | cartesian_control | -0.44427336 | 8.3688182 | 32.951643 | 8.3688182 | 0.59745585 |
| 273 | local_auxiliary | -0.63668577 | 7.181292 | 42.427065 | 5.7990384 | 0.86221177 |
| 364 | cartesian_control | 0.18751097 | 5.9152742 | 23.586284 | 5.9152742 | 0.84526922 |
| 364 | local_auxiliary | -0.57823346 | 6.3511347 | 50.784594 | 5.1882285 | 0.9637199 |

## 12. Module analysis

| Update | Arm | Module | Raw dot | Raw cosine | Virtual Adam local derivative |
|---|---|---|---|---|---|
| 0 | cartesian_control | blocks.0 | -107.22955 | -0.82393886 | 0.15183339 |
| 0 | cartesian_control | blocks.3 | 3.4497781 | 0.1772231 | -0.054997825 |
| 0 | cartesian_control | blocks.4 | -227.82693 | -0.99454617 | 0.039240546 |
| 0 | local_auxiliary | blocks.0 | -107.22955 | -0.82393886 | 0.12451323 |
| 0 | local_auxiliary | blocks.3 | 3.4497781 | 0.1772231 | -0.15606586 |
| 0 | local_auxiliary | blocks.4 | -227.82693 | -0.99454617 | -0.057685972 |
| 364 | cartesian_control | blocks.0 | -13.928905 | -0.46780764 | 0.053957509 |
| 364 | cartesian_control | blocks.3 | 13.413216 | 0.41269374 | -0.013628949 |
| 364 | cartesian_control | blocks.4 | 28.142775 | 0.83641813 | -0.046781624 |
| 364 | local_auxiliary | blocks.0 | -30.796424 | -0.72404189 | 0.082539171 |
| 364 | local_auxiliary | blocks.3 | 14.751502 | 0.46798862 | -0.17942588 |
| 364 | local_auxiliary | blocks.4 | -151.10787 | -0.98080764 | 0.013047438 |

Module audit directions use the fixed explanatory five-protein training panel and each arm’s current saved-history Adam state, not the entire training population.

## 13. Control-versus-auxiliary interpretation

Both arms differ only in the fixed local coefficient; all batches, cached corruptions, initialization, optimizer configuration, boundary selection and evaluation are matched. The primary causal comparison is ArmB vs ArmA at364; historical checkpoints are contextual only.

## 14. Limitations

One matched seed, 364 updates, replay of an existing cached schedule, and previously studied development data. Identity bootstrap is not homology-cluster bootstrap. This is not an independent production test or a complete generative sampling evaluation. Five-protein gradient telemetry is explanatory only. No outcome-based early stop or budget extension occurred.

## 15. Exactly one next experiment

One full matched Phase 4C adaptation: restart both arms from the same original Phase4B update1092 checkpoint and Adam state, replay the full 1092-update historical cached schedule, with lambda=0 versus the unchanged selected lambda=0.06719407652184162. Keep equal condition sampling, optimizer, precision and evaluation identical; pre-register the full contract and retain condition/stratum/chirality/module telemetry. Do not tune lambda or add an architectural intervention. This is a recommendation only; no full run was launched.

## Validation

Before launch: complete pytest 1638 passed,13 existing skips,0 failures; 11 CUDA-specific skipped tests separately passed on the training GPU; focused suite51 passed. Ruff lint/format,356-file Python syntax and Git whitespace checks passed. Verification confirms both arms changed all129 parameter tensors, all129 Adam states reached1456, exact initial model/optimizer/scheduler/scaler/all RNG equality, identical evaluation identities/order at every boundary, all ten checkpoint hashes, and exactly reproduced paired/bootstrap/adjudication results. No checkpoint, dataset or large raw diagnostic is tracked.

## Artifact safety

Historical source checkpoint unchanged: True
Checkpoints and large raw artifacts remain ignored under execution/. Full arm evaluation reports and telemetry are preserved there.

## Endpoint qualifications and mechanism

All three aggregate offsets improve. All conditions and all strata improve mean-local RMSE; the longest stratum has a 0.5706% aligned-Cartesian regression, below the frozen 2% margin. The legacy slope-not-increased gate FAILS (0.00667608 ->0.00694448 Angstrom/residue); this is separately reported and was not the new Phase4C Gate6. No threshold was changed after seeing results.
Aggregate chirality slightly improves, but condition250 and450 inversion rates worsen by about0.00246 and0.00438 respectively; handedness is not resolved. Aggregate chirality, rather than condition-specific chirality, was the pre-registered Gate4.
The unaligned Cartesian objective is slightly worse with auxiliary (33.888416 ->33.954393 Angstrom squared, about0.195%) even though the primary aligned metric improves. These are different frames/reductions; do not call all Cartesian quantities superior.
At endpoint, control panel raw cosine becomes+0.187511 while auxiliary remains-0.578233. The auxiliary block0 harm declines within its trajectory but remains positive and greater than control; block4 protective Adam contribution at0 becomes harmful again at364. Block3 supplies strong protective cancellation. Successful aggregate training does not establish permanent resolution in every module.
The benefit persisted versus matched control at91/182/273/364. Auxiliary local RMSE is also below its own initialization; control deteriorates locally. Neither arm is a validated final structure generator.
