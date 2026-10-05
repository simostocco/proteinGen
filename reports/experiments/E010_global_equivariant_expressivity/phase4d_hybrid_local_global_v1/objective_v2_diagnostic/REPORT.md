# E010 Phase 4D objective-v2 diagnostic

Completed 2026-10-05 on CPU. Diagnostic v1 was preserved unchanged, committed and pushed as `5eaa96c30d7b18ef712aeb5a84c412ba8db5ab97`; local/remote SHA equality was verified before v2. The hybrid architecture is unchanged. Original Phase 4B update-1092 E010 is completely frozen. **No training, optimizer steps, CUDA, environment changes or development tuning. The existing 500-update contract is not authorized.**

A common first-order direction exists. Beta=16.8 and deterministically selected gamma=2 give negative derivatives for local, raw Cartesian and continuous chirality objectives. At 0.03 Å RMS, all three continuous objectives improve and aggregate inversions decrease by four. The 0.05 Å point passes the stated aggregate criteria but its continuous chirality loss rises slightly. All larger sampled corrections increase inversion counts. This supports only a narrow, modest aggregate benefit.

## Protocol and signed loss

`protocol.json` was written before gradient computation. Exactly one beta (16.8) and one analytically selected gamma were evaluated. The objective is `L_local + 16.8 L_cart + gamma L_chiral`; no ReLU guard or displacement penalty participates in its direction. Historical raw Cartesian and equal-example local reductions remain unchanged.

For quartet (i−1,i,i+1,i+2), a=x_i−x_(i−1), b=x_(i+1)−x_i, c=x_(i+2)−x_(i+1), and q=dot(cross(a,b),c)/(||a|| ||b|| ||c||). This is exactly the normalized signed triple-product convention of the existing evaluator on assessable quartets. Proper rotation and translation preserve q; reflection reverses it; opposed target/prediction signs indicate inversion. The denominator is clamped at (1e−6 Å)^3 only for finite arithmetic.

Freeze exactly the 13,029 jointly assessable, nonplanar Pg/target quartets identified by the existing evaluator at initialization, with its original frame/bond/|q| eligibility tests. Target q and masks are detached. Later predictions cannot drop quartets from the loss by becoming degenerate or planar. `L_chiral` is the **pooled eligible-quartet mean** of (q_pred−q_target)^2, dimensionless; local and Cartesian losses remain equal-example means in Å². Gamma=2 uses the current Å-coordinate convention. Empty eligible sets contribute zero to the helper; the diagnostic requires a nonempty panel denominator.

No target/development selection occurs. The existing 20 identities and 60 identity-condition examples are reused with their immutable pins. Source checkpoint SHA256: `f5211cbc1be5092175ce15b9761a4efba761d242310287cd6b1e04df6a6744ef`. Preparation, v1, Phase 4B/4C, cache and checkpoint hashes are verified unchanged after execution.

## Gradient calibration

| Component | Initial norm |
| --- | ---: |
| cartesian | 1.847359849 |
| chiral | 1.374366244 |
| local | 31.033474423 |

| Pair | Dot product | Cosine |
| --- | ---: | ---: |
| local/cartesian | -48.604330392 | -0.847799319 |
| local/chiral | -24.375533564 | -0.571506600 |
| cartesian/chiral | 1.263976647 | 0.497834592 |

Every initial component gradient is confined to the head; input and all four residual blocks have zero initial norms and dot contributions. Zero-module cosines are undefined, recorded as null. Complete per-module contributions and pairwise Gram matrices are in `gradient_audit.json`.

Writing d=−(g_local+16.8 g_cart+gamma g_chiral), strict descent requires each row below to be positive before negation:

| Objective | Intercept | Gamma slope |
| --- | ---: | ---: |
| cartesian | 8.729672311 | 1.263976647 |
| chiral | -3.140726920 | 1.888882574 |
| local | 146.523821750 | -24.375533564 |

The feasible interval is **1.662743340 < gamma < 6.011102131**. The pre-registered 1-2-5 decade rule selects **gamma=2**, the first simple value strictly inside the interval, with no finite-metric search. Without chirality, beta=16.8 descends local and Cartesian losses but predicts increased chirality loss. With gamma=2, predicted d-direction derivatives are:

| Objective | Predicted derivative per alpha |
| --- | ---: |
| cartesian | -11.257625642 |
| chiral | -0.637038245 |
| local | -97.772753644 |

## Independent correction-scale sweep

All initial directions are head-only, so corrections along this fixed path are linear in alpha. The exact directional output gives RMS slope **92.779890236 Å/alpha**; alpha is target RMS divided by that slope. No extra scale-search model evaluations or parameter steps are used. Every candidate is an independent copy of theta0. Actual RMS matches targets within approximately 1e−8 Å.

The raw-Cartesian tolerance was predeclared as 1e−6 relative (0.0001%), not selected from results; every aggregate nonzero point actually improves raw Cartesian. Mean-local/offset RMSE follows the preparation mean per-example convention, with MSE and sqrt(mean MSE) also recorded. Aligned RMSD uses proper rotations. Correction RMS is sqrt(equal-example mean squared vector displacement); maximum is over all residues.

| RMS target Å | Alpha | Local gain % | Raw Cartesian Δ % | Aligned RMSD Δ % | Continuous chiral loss | Inversions | Max correction Å | Aggregate pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.00 | 0 | 0.0000 | +0.00000 | +0.00000 | 0.423339066 | 5493 | 0.000000 | False |
| 0.03 | 0.000323345931 | 0.2104 | -0.00952 | -0.00947 | 0.423311508 | 5489 | 0.040998 | True |
| 0.05 | 0.000538909885 | 0.3618 | -0.01492 | -0.01456 | 0.423498320 | 5492 | 0.068330 | True |
| 0.10 | 0.00107781977 | 0.7774 | -0.02509 | -0.02294 | 0.424539835 | 5512 | 0.136661 | False |
| 0.15 | 0.00161672966 | 1.2427 | -0.03050 | -0.02514 | 0.426157792 | 5559 | 0.204991 | False |
| 0.20 | 0.00215563954 | 1.7541 | -0.03117 | -0.02112 | 0.428159361 | 5571 | 0.273322 | False |
| 0.27 | 0.00291011338 | 2.5407 | -0.02412 | -0.00506 | 0.431400461 | 5579 | 0.368985 | False |

| RMS Å | i+1 mean RMSE Å | i+2 mean RMSE Å | i+3 mean RMSE Å | Mean-local RMSE Å | Raw Cartesian Å² | Aligned RMSD Å |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 1.895241 | 2.874420 | 3.553660 | 2.774440 | 35.077881 | 7.858572 |
| 0.03 | 1.887802 | 2.868501 | 3.549507 | 2.768603 | 35.074542 | 7.857827 |
| 0.05 | 1.882374 | 2.864284 | 3.546552 | 2.764403 | 35.072648 | 7.857428 |
| 0.10 | 1.867271 | 2.852824 | 3.538523 | 2.752873 | 35.069081 | 7.856769 |
| 0.15 | 1.850161 | 2.840122 | 3.529601 | 2.739961 | 35.067181 | 7.856596 |
| 0.20 | 1.831258 | 2.826245 | 3.519819 | 2.725774 | 35.066947 | 7.856912 |
| 0.27 | 1.802143 | 2.804959 | 3.504748 | 2.703950 | 35.069421 | 7.858174 |

All points are finite, with unchanged 13,029 chirality-assessable windows and no lost/gained eligibility. Both input-Pg and corrected-output frames remain 13,089 eligible / zero degenerate interior frames. All local offsets improve at all nonzero points. The two sampled aggregate passing points are 0.03 and 0.05 Å; **only 0.03 Å also improves continuous chirality loss**. These samples do not prove every intervening scale passes, nor a uniform condition/length-stratum non-inferiority result. The continuous loss is smooth but does not enforce the binary inversion count.

## Conditions

| RMS Å | Condition | Local gain % | Raw Cartesian Δ % | Aligned RMSD Δ % | Continuous chirality Δ | Inversion Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.03 | 50 | 0.3866 | -0.16813 | -0.08860 | -0.00015075 | -7 |
| 0.03 | 250 | 0.1507 | -0.02115 | -0.01307 | +0.00073723 | +7 |
| 0.03 | 450 | 0.1911 | -0.00428 | +0.00409 | -0.00066915 | -4 |
| 0.05 | 50 | 0.6489 | -0.26186 | -0.13915 | -0.00020089 | -8 |
| 0.05 | 250 | 0.2610 | -0.03321 | -0.02073 | +0.00150691 | +4 |
| 0.05 | 450 | 0.3327 | -0.00673 | +0.00703 | -0.00082825 | +3 |
| 0.10 | 50 | 1.3203 | -0.43193 | -0.23565 | -0.00015585 | -10 |
| 0.10 | 250 | 0.5694 | -0.05626 | -0.03623 | +0.00428177 | +5 |
| 0.10 | 450 | 0.7335 | -0.01142 | +0.01512 | -0.00052361 | +24 |
| 0.15 | 50 | 2.0130 | -0.51021 | -0.28935 | +0.00012261 | +9 |
| 0.15 | 250 | 0.9230 | -0.06915 | -0.04650 | +0.00808385 | +24 |
| 0.15 | 450 | 1.1965 | -0.01405 | +0.02427 | +0.00024971 | +33 |
| 0.20 | 50 | 2.7259 | -0.49670 | -0.30018 | +0.00062196 | +12 |
| 0.20 | 250 | 1.3191 | -0.07187 | -0.05153 | +0.01272304 | +28 |
| 0.20 | 450 | 1.7162 | -0.01464 | +0.03450 | +0.00111589 | +38 |
| 0.27 | 50 | 3.7551 | -0.32357 | -0.24331 | +0.00166859 | +23 |
| 0.27 | 250 | 1.9407 | -0.05861 | -0.04976 | +0.02032919 | +32 |
| 0.27 | 450 | 2.5299 | -0.01202 | +0.05066 | +0.00218641 | +31 |

All conditions improve locally and in raw Cartesian at all sampled nonzero points. Condition 250 has increased inversions even at 0.03 Å (+7); condition 450 worsens at 0.05 Å (+3). At 0.03 Å, condition-50/450 gains offset the condition-250 orientation harm. Aggregate success therefore still includes cancellation between conditions.

## Length strata

| RMS Å | Stratum | Local gain % | Raw Cartesian Δ % | Aligned RMSD Δ % | Continuous chirality Δ | Inversion Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.03 | 20-64 | 0.2389 | -0.04665 | -0.01404 | +0.00224964 | -1 |
| 0.03 | 65-128 | 0.1874 | -0.02409 | -0.00977 | -0.00004987 | -1 |
| 0.03 | 129-256 | 0.2118 | -0.00924 | -0.01960 | -0.00047992 | +5 |
| 0.03 | 257-384 | 0.2100 | +0.00003 | -0.00508 | -0.00032600 | -8 |
| 0.03 | 385-500 | 0.2044 | +0.00573 | -0.00225 | +0.00021267 | +1 |
| 0.05 | 20-64 | 0.4113 | -0.07638 | -0.02216 | +0.00437737 | +4 |
| 0.05 | 65-128 | 0.3240 | -0.03881 | -0.01488 | +0.00021011 | -2 |
| 0.05 | 129-256 | 0.3634 | -0.01428 | -0.03135 | -0.00064095 | +12 |
| 0.05 | 257-384 | 0.3603 | +0.00080 | -0.00727 | -0.00033308 | -3 |
| 0.05 | 385-500 | 0.3507 | +0.01019 | -0.00268 | +0.00051850 | -12 |
| 0.10 | 20-64 | 0.8859 | -0.14589 | -0.03804 | +0.01075510 | +9 |
| 0.10 | 65-128 | 0.7042 | -0.07093 | -0.02276 | +0.00156628 | +0 |
| 0.10 | 129-256 | 0.7771 | -0.02302 | -0.05610 | -0.00058747 | +11 |
| 0.10 | 257-384 | 0.7705 | +0.00536 | -0.00858 | +0.00035669 | -1 |
| 0.10 | 385-500 | 0.7501 | +0.02366 | -0.00002 | +0.00173111 | +0 |
| 0.15 | 20-64 | 1.4183 | -0.20853 | -0.04758 | +0.01719305 | +14 |
| 0.15 | 65-128 | 1.1361 | -0.09637 | -0.02365 | +0.00354396 | +3 |
| 0.15 | 129-256 | 1.2372 | -0.02620 | -0.07422 | +0.00003143 | +19 |
| 0.15 | 257-384 | 1.2275 | +0.01370 | -0.00392 | +0.00182662 | +20 |
| 0.15 | 385-500 | 1.1952 | +0.04040 | +0.00797 | +0.00339675 | +10 |
| 0.20 | 20-64 | 2.0034 | -0.26430 | -0.05071 | +0.02291833 | +12 |
| 0.20 | 65-128 | 1.6155 | -0.11511 | -0.01753 | +0.00578737 | +6 |
| 0.20 | 129-256 | 1.7396 | -0.02385 | -0.08570 | +0.00113027 | +14 |
| 0.20 | 257-384 | 1.7285 | +0.02580 | +0.00671 | +0.00389155 | +34 |
| 0.20 | 385-500 | 1.6830 | +0.06040 | +0.02131 | +0.00536062 | +12 |
| 0.27 | 20-64 | 2.9028 | -0.33083 | -0.04422 | +0.02974090 | +11 |
| 0.27 | 65-128 | 2.3602 | -0.13011 | +0.00280 | +0.00901689 | +1 |
| 0.27 | 129-256 | 2.5077 | -0.01122 | -0.09056 | +0.00331357 | +9 |
| 0.27 | 257-384 | 2.4978 | +0.04909 | +0.03161 | +0.00751667 | +55 |
| 0.27 | 385-500 | 2.4324 | +0.09389 | +0.04894 | +0.00845590 | +10 |

Every stratum improves locally. The two longest strata have small raw-Cartesian increases even where the aggregate improves. Orientation changes are mixed; at 0.03 Å, 129–256 and 385–500 gain +5 and +1 inversions. Four identities per stratum do not establish a population law.

## Prepared displacement bound

The standalone utility is prepared but **not attached to the unchanged hybrid architecture**. For raw local vector u, define

`u_bounded = u / sqrt(1 + ||u||² / s_max²)`.

Its norm is strictly below s_max for finite u, it is smooth with identity Jacobian at zero, and preserves exact zero-head parity. As an isotropic radial map, it commutes with orthogonal changes of local coordinates. On eligible orthonormal frames, ||F u_bounded||=||u_bounded||; existing invalid-frame masking retains exact zero. Proper global rotations rotate F and Cartesian corrections, while translations have no effect. This does not impose reflection equivariance on the parity-sensitive learned network.

After the finite response, a conservative **s_max review range is 0.04–0.05 Å per residue**, around the observed 0.040998 Å maximum at the strongest 0.03 Å RMS aggregate point. No single s_max is selected or activated. This is a maximum-per-residue bound, not an RMS bound. Saturation changes the direction and response, so the bounded parameterization itself still requires a separate reviewed diagnostic; these unbounded samples cannot prove its safety.

## Validation and records

Focused/adjacent CPU tests cover beta arithmetic and real initialization descent, analytic gamma feasibility/empty-interval classification, signed-loss parity and evaluator agreement, target/frozen eligibility and no mask escape, proper rotation/translation, independent copies and RMS calibration, global gradient isolation, historical/v1 integrity, and smooth-bound parity, Jacobian, rotation and large-float safety. Final outcomes are in `validation.json`.

`protocol.json`, `gradient_audit.json`, `results.json`, and `interpretation.json` preserve complete results including per-example/condition/stratum metrics. The runnable script refuses to overwrite v2 outputs. Use the existing environment with CUDA hidden and two CPU threads; no Python environment changes are required.

**Exactly one recommended next action:** review the limited 0.03 Å aggregate result and the proposed bound before defining any replacement tiny-overfit contract. The old 500-update contract remains closed.
