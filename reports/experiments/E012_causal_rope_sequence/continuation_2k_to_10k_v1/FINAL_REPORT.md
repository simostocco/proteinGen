# E012 continuation — CSEQ-C / CONT-B

Final primary CE 2.894097, independent 2.904614. Primary Gates A–E: {"A": false, "B": false, "C": true, "D": true, "E": false}; independent confirmation: {"A": false, "B": false, "C": true, "D": true, "E": false}. CSEQ/continuation labels follow the frozen rules. Short-stratum point failures are distinguished from statistically established harm. The full diagnostics at 5000 and 10000 show whether prefix order and long context were retained; training CE decline is not a success criterion.

The additional budget strengthened prefix-order sensitivity but did not improve endpoint likelihood over historical 2k: normal CE changed by +0.006024 primary and +0.011598 independent. Gate B fails: primary bigram improvement is unestablished and independent CE is significantly worse than bigram. The 20–64 and 65–128 regimes have worse bucket-baseline point estimates on both panels; significance is established for primary 65–128 and both independent short regimes. The sealed “underfit” label refers to held-out likelihood gates and does not establish that more optimization or capacity would help.

Exactly 8,000 additional successful updates, global 10,000; no extension or biological generation. Historical 2k results remain immutable.

Preparation `f0a19187b2945f8f34e2ad30bd709fdeca9d5e5a`; source checkpoint SHA256 `ff666eccdb6d28d51f2d3d561242152512e2dddf74fc0f251429d55b886d0188`. Model 15,533,952 parameters, Adam moments/RNG/sampler restored. Contract `09af1c37a485a6b1746a49b65d49b41b09e1774d9ca85803a6f31178f1b40c28`.

## Likelihood trajectory

| Global update | Primary CE | Independent CE |
|---|---|---|
| 2000 | 2.888072 | 2.893015 |
| 3000 | 2.883708 | 2.889910 |
| 5000 | 2.880229 | 2.889330 |
| 7500 | 2.907171 | 2.917368 |
| 10000 | 2.894097 | 2.904614 |

## Frozen gates and paired identity intervals

10,000 bootstrap resamples; seed 12112, equal-protein primary likelihood. Empirical trigram is worse than unigram; beating trigram alone does not establish context learning.

| Population | A | B | C | D | E |
|---|---|---|---|---|---|
| primary | False | False | True | True | False |
| independent | False | False | True | True | False |

Independent confirmation: `{"A": false, "B": false, "C": true, "D": true, "E": false}`.

| Population | Comparison | Delta [95% CI] |
|---|---|---|
| primary | bigram | -0.000607 [-0.007258, +0.005791] |
| primary | bucket_unigram | -0.004946 [-0.011457, +0.001324] |
| primary | global_unigram | -0.005939 [-0.012681, +0.000520] |
| primary | last1 | -0.585870 [-0.607678, -0.564159] |
| primary | last8 | -0.113208 [-0.122391, -0.104157] |
| primary | shuffle | -0.064828 [-0.073943, -0.055581] |
| primary | trigram | -0.012083 [-0.018583, -0.005777] |
| independent | bigram | +0.008007 [+0.001914, +0.014204] |
| independent | bucket_unigram | +0.003300 [-0.002640, +0.009466] |
| independent | global_unigram | +0.002917 [-0.003220, +0.009237] |
| independent | last1 | -0.588343 [-0.610151, -0.566790] |
| independent | last8 | -0.113136 [-0.121596, -0.104603] |
| independent | shuffle | -0.051542 [-0.059759, -0.043220] |
| independent | trigram | -0.004949 [-0.010970, +0.001208] |

## Context windows, order sensitivity and length ablation

| Panel | Last1 | Last4 | Last8 | Last16 | Last32 | Last64 | Full | Shuffle |
|---|---|---|---|---|---|---|---|---|
| primary | 3.476981 | 3.043285 | 3.004320 | 2.973056 | 2.944110 | 2.914900 | 2.891111 | 2.955940 |
| independent | 3.495453 | 3.048241 | 3.020247 | 2.985249 | 2.954196 | 2.930495 | 2.907110 | 2.958652 |

primary: KL 0.202820, JS 0.045217, top1 change 0.741425, top3 set change 0.932404; neutral-length minus normal +0.029560 [+0.026040, +0.033060].


independent: KL 0.198310, JS 0.044353, top1 change 0.747803, top3 set change 0.933044; neutral-length minus normal +0.024109 [+0.020509, +0.027604].

## Length regimes and relative position

| Panel | Stratum | Normal−bucket | Normal−shuffle | Full−last8 |
|---|---|---|---|---|
| primary | 20–64 | +0.009434 [-0.013459, +0.031720] | -0.115113 [-0.143431, -0.086954] | -0.089415 [-0.110851, -0.068510] |
| primary | 65–128 | +0.015494 [+0.002571, +0.028545] | -0.081937 [-0.102099, -0.061252] | -0.135252 [-0.155978, -0.114282] |
| primary | 129–256 | -0.004242 [-0.023652, +0.010954] | -0.040066 [-0.065219, -0.018444] | -0.123668 [-0.150295, -0.100229] |
| primary | 257–384 | -0.018076 [-0.022962, -0.013011] | -0.046988 [-0.061448, -0.033261] | -0.108803 [-0.126688, -0.091733] |
| primary | 385–500 | -0.027428 [-0.031197, -0.023555] | -0.039933 [-0.049298, -0.030372] | -0.108883 [-0.122855, -0.094827] |
| independent | 20–64 | +0.036265 [+0.011928, +0.060614] | -0.101959 [-0.131458, -0.073280] | -0.105743 [-0.129015, -0.081544] |
| independent | 65–128 | +0.014805 [+0.001551, +0.028269] | -0.051522 [-0.070346, -0.032159] | -0.119925 [-0.141143, -0.098803] |
| independent | 129–256 | +0.003901 [-0.005050, +0.012185] | -0.037808 [-0.052291, -0.022810] | -0.122120 [-0.140473, -0.103430] |
| independent | 257–384 | -0.013119 [-0.019600, -0.005044] | -0.034856 [-0.046835, -0.023019] | -0.113462 [-0.129136, -0.097193] |
| independent | 385–500 | -0.025464 [-0.029682, -0.021475] | -0.031473 [-0.041216, -0.021786] | -0.104412 [-0.118637, -0.090043] |

Existing ten relative-position deciles are preserved and reported by length stratum at each normal evaluation boundary in trajectory.json. The 20–64 and 65–128 regimes are shown explicitly above; position telemetry is descriptive, not a gate. Update 3000 contains normal likelihood only, so it cannot adjudicate shuffle/window retention; update 5000 contains the complete fixed diagnostics in scientific_boundaries.json. No checkpoint selection occurred.

## Throughput, integrity and limitations

Frozen physical plan: `{"0": {"forward_backward_seconds": 0.06010669795796275, "partition": [64], "peak_allocated_bytes": 925710336, "peak_reserved_bytes": 968884224, "physical_proteins": 64}, "1": {"forward_backward_seconds": 0.07541347341611981, "partition": [64], "peak_allocated_bytes": 1547896832, "peak_reserved_bytes": 1572864000, "physical_proteins": 64}, "2": {"forward_backward_seconds": 0.15214944595936686, "partition": [32, 32], "peak_allocated_bytes": 1635281408, "peak_reserved_bytes": 1677721600, "physical_proteins": 32}, "3": {"forward_backward_seconds": 0.25122741248924285, "partition": [40, 24], "peak_allocated_bytes": 2728467456, "peak_reserved_bytes": 2780823552, "physical_proteins": 40}, "4": {"forward_backward_seconds": 0.3488014121539891, "partition": [40, 24], "peak_allocated_bytes": 3466172928, "peak_reserved_bytes": 3649044480, "physical_proteins": 40}}`. Effective optimizer batch remains 64, microbatch mean scaled by n/64. LR restarts linearly from 0 at update 2000 to .0003 at 2100, then cosine to zero at 10000. BF16 and architecture/dropouts unchanged.

Historical training throughput 2.0469 updates/sec; continuation 3.6065; measured speedup 1.762×. Evaluation excluded consistently. Peak allocated/reserved 2.122/3.277 GiB. Total exposures 640,000, residues 117,588,908, nominal passes 2.7617.

All 172 paired intervals independently reproduced; four new checkpoint hashes, source hash, Adam steps, scheduler, RNG fields and exact sampler-derived exposure sequence verified. Focused/sequence tests and preflight passed; see quality.json and resume_smoke.json. Protected baseline/panel/data/source hashes unchanged. E011 and structure worktrees receive no E012 writes; concurrent owner snapshots are in independent_verification.json.

One continuation trajectory and one warm restart; physical batching changes dropout draw assignment, so this is not bitwise replay of 8×8. Identity bootstrap omits cluster dependence; PDB cohort bias persists. This continuation does not independently replicate initialization or demonstrate biological generation quality. Longer training does not by itself prove context use.

## Exactly one recommended next experiment

Preregister a fixed-checkpoint, length-stratified TRAIN-versus-held-out generalization audit of the 10k model, including early-prefix errors, without further training.

## Context retention across fixed full diagnostic boundaries

| Panel | Global update | Normal−global | Normal−shuffle | Full−last8 | Full−last1 |
|---|---|---|---|---|---|
| primary | 2000 | -0.011963 [-0.017364, -0.006796] | -0.032555 [-0.038672, -0.026641] | -0.100602 [-0.108216, -0.093027] | -0.348581 [-0.361589, -0.335687] |
| primary | 5000 | -0.019806 [-0.025508, -0.014666] | -0.048076 [-0.055534, -0.040902] | -0.098464 [-0.106187, -0.090915] | -0.630901 [-0.650966, -0.611008] |
| primary | 10000 | -0.005939 [-0.012681, +0.000520] | -0.064828 [-0.073943, -0.055581] | -0.113208 [-0.122391, -0.104157] | -0.585870 [-0.607678, -0.564159] |
| independent | 2000 | -0.008682 [-0.013199, -0.004006] | -0.028323 [-0.034251, -0.022306] | -0.097153 [-0.104570, -0.089697] | -0.343472 [-0.356486, -0.330105] |
| independent | 5000 | -0.012367 [-0.017383, -0.007423] | -0.041225 [-0.047941, -0.034607] | -0.096283 [-0.103346, -0.089061] | -0.632800 [-0.652995, -0.612840] |
| independent | 10000 | +0.002917 [-0.003220, +0.009237] | -0.051542 [-0.059759, -0.043220] | -0.113136 [-0.121596, -0.104603] | -0.588343 [-0.610151, -0.566790] |

## Checkpoint hashes

Historical source SHA256 `ff666eccdb6d28d51f2d3d561242152512e2dddf74fc0f251429d55b886d0188`.

| Global update | SHA256 |
|---|---|
| 3000 | `8bb6ddd9fcb7bdab49f91d163390243cb8f3d45cffee7cd0c1b124378db80030` |
| 5000 | `06ef3eddc9560ee4d3160a8aeb280d14bd1112c334d19f023556dbf3e2b32df4` |
| 7500 | `533340bf81b01111401c2765cb0e0ed16c57429a36bf3dc8608b82d66057bd11` |
| 10000 | `23d4aad8327c491d31d22192e2b4f51bda8e68f13e1c213ed7925cfcafb7bf89` |

## Short-regime relative-position telemetry

Historical ten equal relative-position deciles are retained (0–10%, 10–20%, …, 90–100%). Values are equal-protein means within each decile and stratum.

primary endpoint:

| Length | CE deciles from N to C terminus |
|---|---|
| 20–64 | 3.1488, 3.0877, 2.9531, 2.8764, 2.8738, 2.8303, 2.8455, 2.8874, 2.8870, 2.8624 |
| 65–128 | 2.9737, 2.9740, 2.9487, 2.9114, 2.8918, 2.8963, 2.9155, 2.8849, 2.9074, 2.8693 |

independent endpoint:

| Length | CE deciles from N to C terminus |
|---|---|
| 20–64 | 3.2072, 3.0772, 2.9640, 2.8933, 2.9316, 2.9032, 2.9211, 2.8680, 2.8998, 2.9208 |
| 65–128 | 2.9621, 2.9774, 2.9413, 2.9341, 2.8986, 2.9321, 2.9187, 2.9079, 2.8918, 2.8397 |


The endpoint model-minus-bucketed-unigram deciles are also recorded in relative_position_baseline_comparison.json. They compare the same targets within fixed bins against unchanged TRAIN-only baseline probabilities; this is descriptive and changes no gate.

primary: short-stratum model-minus-bucket CE by decile:

| Length | Delta deciles |
|---|---|
| 20–64 | +0.1888, +0.1588, +0.0118, -0.0114, -0.0360, -0.0717, -0.0721, -0.0208, -0.0211, -0.0638 |
| 65–128 | +0.0629, +0.0746, +0.0442, +0.0129, -0.0033, -0.0006, +0.0055, -0.0189, +0.0131, -0.0401 |

independent: short-stratum model-minus-bucket CE by decile:

| Length | Delta deciles |
|---|---|
| 20–64 | +0.2355, +0.1467, +0.0441, -0.0193, +0.0052, -0.0185, -0.0227, -0.0283, -0.0145, +0.0130 |
| 65–128 | +0.0372, +0.0855, +0.0325, +0.0270, +0.0041, +0.0240, +0.0073, -0.0116, -0.0129, -0.0529 |


Initial preparation `d8956176933acd3eafe15b263e9d65ef22d32afa`; final pushed pre-execution preparation `f0a19187b2945f8f34e2ad30bd709fdeca9d5e5a`. The owner guard correction preceded update2001 and changed no scientific protocol.

GPU utilization median 41.0% over 229 recorded driver samples, including evaluations.

Early-bin telemetry shows the largest short-protein excess over the bucket baseline in the first 20–30% of the sequence. This concentration is descriptive; no positional gate or causal explanation is introduced. E011 and Phase4C match their starting snapshots. Phase4D independently advanced and is running its owner’s CPU diagnostics; E012 made no writes there. The E012 CUDA workload has completed (postflight 5% device utilization, 849MiB desktop usage).
