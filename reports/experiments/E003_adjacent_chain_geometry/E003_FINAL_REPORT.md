# E003 Final Report

## Decision

E003 is finalized as an analysis/documentation-only experiment. It is not promoted
to the new baseline. E002 remains the current baseline.

E003 modestly improves the local adjacent-residue distance objective it was
designed to target, but significantly degrades two global geometry diagnostics:
negative eigenvalue mass and triangle-violation fraction. Rank-3 residual and
classical MDS stress are inconclusive under the paired bootstrap intervals.
Strict empirical real-like geometry remains zero for both E002 and E003.

The experiment is retained as useful evidence for a local-versus-global
objective trade-off: directly regularizing adjacent chain distances can improve
that local statistic while worsening global distance-matrix validity.

## Provenance

The comparison uses the calibrated paired generated-ensemble protocol in
`comparison_to_E002`.

| Model | Selected checkpoint | Stored epoch index | Optimizer step | SHA-256 |
| --- | --- | ---: | ---: | --- |
| E002 | `outputs/recovered_full_b2_v_axial_edm_e002/checkpoints/final_validation_selected.pt` | 4 | 169181 | `2b6b9967a5de6035accad7cc1c24e379743459531c641727221541fde9aa669f` |
| E003 | `outputs/recovered_full_b2_v_axial_edm_chain_e003/checkpoints/final_validation_selected.pt` | 4 | 169182 | `bdbaf25ee3b9d7d7a7cfe2ebe03a12b09b16261b1a8a3e95d28b2c48058ca101` |

Both are epoch-5 selections in user-facing numbering. The comparison is a
selected-model comparison, not a perfectly controlled causal ablation of one
training seed, because selected epochs, optimizer steps, and training histories
differ.

Bootstrap confidence intervals quantify generated-pair variability for this
paired 375-sample bank. They do not quantify training-seed uncertainty, so this
single-run comparison should not be generalized as a universal causal claim.

## Training Stability

E003 selected epoch 5 with validation loss `0.0183565374`, compared with E002
epoch-5 validation loss `0.0185558852`. E003 reached optimizer step `169182`.
Training recorded `63` AMP overflow events; the affected four-microbatch windows
contained `252` log rows. The maximum consecutive overflow count was `1`; no
non-finite losses were observed; accumulation state at the selected checkpoint
was `0`. Diffusion, stochastic EDM spectral, and adjacent median losses all
decreased across epochs.

## Overall Paired Effects

Positive improvement means E003 is lower/better than E002 for the lower-is-better
metric.

| Metric | E002 mean | E003 mean | Mean improvement | Bootstrap CI | Fraction improved | Decision |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Adjacent-residue distance RMSE | 0.2792727744 | 0.2708957802 | 0.0083769942 | [0.0007474087, 0.0161266259] | 0.570667 | Modest improvement |
| Negative eigenvalue mass fraction | 0.0926963031 | 0.0969014067 | -0.0042051035 | [-0.0050886455, -0.0033259991] | 0.309333 | Significant degradation |
| Triangle violation fraction | 0.0021927083 | 0.0026302083 | -0.0004375 | [-0.0006771159, -0.0001874674] | 0.333333 | Significant degradation |
| Rank-3 residual energy fraction | 0.1476842877 | 0.1484084815 | -0.0007241938 | [-0.0025555061, 0.0011522294] | 0.520000 | Inconclusive |
| Classical MDS stress | 0.0721132626 | 0.0708779061 | 0.0012353564 | [-0.0004186714, 0.0028830080] | 0.544000 | Inconclusive |

## Per-Length Primary Effects

| N | Metric | E002 mean | E003 mean | Mean improvement | Bootstrap CI | Fraction improved |
| ---: | --- | ---: | ---: | ---: | --- | ---: |
| 64 | Adjacent RMSE | 0.339538 | 0.407048 | -0.0675099 | [-0.0836315, -0.0518208] | 0.17 |
| 64 | Negative eigenvalue mass | 0.0519124 | 0.0534542 | -0.00154185 | [-0.00298515, -0.000195868] | 0.47 |
| 64 | Triangle violation fraction | 0.00430176 | 0.00516113 | -0.000859375 | [-0.00166516, -0.000024292] | 0.39 |
| 64 | Rank-3 residual | 0.0759439 | 0.0745395 | 0.00140441 | [-0.00195061, 0.0044613] | 0.62 |
| 64 | Classical MDS stress | 0.0364320 | 0.0348060 | 0.00162594 | [-0.000856594, 0.00401118] | 0.59 |
| 128 | Adjacent RMSE | 0.230232 | 0.229999 | 0.000233609 | [-0.0178085, 0.0182817] | 0.52 |
| 128 | Negative eigenvalue mass | 0.0745467 | 0.0788088 | -0.00426214 | [-0.005982, -0.00263338] | 0.32 |
| 128 | Triangle violation fraction | 0.00119141 | 0.00184082 | -0.000649414 | [-0.000922852, -0.000366089] | 0.20 |
| 128 | Rank-3 residual | 0.125967 | 0.122673 | 0.00329356 | [0.000105067, 0.00681568] | 0.63 |
| 128 | Classical MDS stress | 0.0589699 | 0.0539975 | 0.00497234 | [0.00165969, 0.0085922] | 0.63 |
| 256 | Adjacent RMSE | 0.255939 | 0.204507 | 0.0514323 | [0.0390651, 0.0649144] | 0.78 |
| 256 | Negative eigenvalue mass | 0.111324 | 0.114893 | -0.00356933 | [-0.00577845, -0.00127335] | 0.30 |
| 256 | Triangle violation fraction | 0.00129395 | 0.00135742 | -0.0000634766 | [-0.000361328, 0.000249146] | 0.40 |
| 256 | Rank-3 residual | 0.180478 | 0.182214 | -0.00173578 | [-0.00654216, 0.00289899] | 0.47 |
| 256 | Classical MDS stress | 0.0881140 | 0.0880312 | 0.0000827419 | [-0.004101, 0.00449018] | 0.54 |
| 384 | Adjacent RMSE | 0.292697 | 0.223185 | 0.0695117 | [0.0566891, 0.0835039] | 0.94 |
| 384 | Negative eigenvalue mass | 0.138330 | 0.146553 | -0.00822235 | [-0.0101125, -0.0064144] | 0.10 |
| 384 | Triangle violation fraction | 0.00161133 | 0.00169922 | -0.0000878906 | [-0.000488281, 0.0003125] | 0.36 |
| 384 | Rank-3 residual | 0.223251 | 0.229520 | -0.0062687 | [-0.0102395, -0.00187742] | 0.36 |
| 384 | Classical MDS stress | 0.114122 | 0.114744 | -0.00062142 | [-0.00439434, 0.00353399] | 0.44 |
| 500 | Adjacent RMSE | 0.300860 | 0.250853 | 0.0500072 | [0.0263785, 0.0717417] | 0.80 |
| 500 | Negative eigenvalue mass | 0.162651 | 0.171790 | -0.00913855 | [-0.0124815, -0.00583461] | 0.08 |
| 500 | Triangle violation fraction | 0.00251953 | 0.00261719 | -0.0000976563 | [-0.000722656, 0.000585937] | 0.32 |
| 500 | Rank-3 residual | 0.239206 | 0.249380 | -0.0101743 | [-0.014732, -0.00560942] | 0.20 |
| 500 | Classical MDS stress | 0.119391 | 0.126342 | -0.00695088 | [-0.0111347, -0.00285788] | 0.24 |

Adjacent RMSE improves at N=256, N=384, and N=500, is effectively unchanged at
N=128, and worsens at N=64. The global spectral degradation strengthens with
requested length for negative eigenvalue mass and is not offset by strict
validity improvements.

## Validity

Strict empirical validity is unchanged at zero:

- E002: `0/375`
- E003: `0/375`

Heuristic EDM-quality validity is essentially unchanged:

- E002: `0.128`
- E003: `0.1306667`
- E002-fail/E003-pass transitions: `9`
- E002-pass/E003-fail transitions: `8`
- exact McNemar p-value: `1.0`

By requested length, heuristic pass fractions are unchanged at N=64 (`0.43` to
`0.43`), improve slightly at N=128 (`0.04` to `0.05`), remain unchanged at N=256
(`0.01` to `0.01`), and remain zero at N=384 and N=500.

`generated_count=0` warnings for `empirical_real_like_geometry_pass` novelty
subgroups are expected because no generated sample passed strict empirical
validity. They are not missing-data warnings.

## Distribution Matching, Diversity, And Novelty

Distribution matching is mixed and does not support promotion. Across 60
length-descriptor rows, E003 moves closer to the matched real distribution in
25 rows by standardized mean discrepancy, 24 rows by Wasserstein distance, and
23 rows by KS statistic. By length, the counts for mean-discrepancy improvement
are N=64 `4/12`, N=128 `5/12`, N=256 `8/12`, N=384 `6/12`, and N=500 `2/12`.

Diversity does not collapse. E003 moves generated/real diversity ratios closer
to one in 13/20 length-metric rows, with per-length counts N=64 `3/4`, N=128
`2/4`, N=256 `2/4`, N=384 `2/4`, and N=500 `4/4`. Ratios below one are treated
as reduced diversity relative to real controls, not automatically as exact
duplication or collapse.

Novelty is mixed and approximate. E003 moves calibrated novelty ratios closer to
the real calibration baseline in 6/15 rows, with per-length counts N=64 `1/3`,
N=128 `1/3`, N=256 `1/3`, N=384 `2/3`, and N=500 `1/3`. Larger novelty
distances are not interpreted as automatically better because excessive distance
can also indicate out-of-distribution invalidity.

## Interpretation

The adjacent-chain auxiliary term successfully targets a local geometric
quantity in much of the length range, but it does not repair the global metric
structure needed for strict empirical validity. The significant degradation in
negative eigenvalue mass and triangle violations means E003 should not replace
E002 as the reference model.

The result is scientifically useful because it narrows the problem: a local
backbone-distance objective can conflict with global EDM consistency rather than
simply complementing it.

## Diagnostic Before E004

Before implementing E004, run an analysis-only gradient diagnostic that
stratifies diffusion, stochastic EDM, and adjacent-chain gradient norms and
pairwise cosine similarities by:

- diffusion timestep bins;
- requested-length bins;
- optionally the timestep-by-length interaction.

The purpose is to determine whether adjacent-chain gradients conflict with EDM
gradients primarily at high-noise timesteps, at long requested lengths, or
throughout training.

Conditional future directions:

- If conflict concentrates at high noise, evaluate timestep- or SNR-gated
  adjacent loss.
- If conflict concentrates at long lengths, evaluate length-dependent adjacent
  weighting.
- If conflict is widespread, abandon the adjacent loss or investigate a
  multi-objective gradient method.
- If no systematic conflict appears, evaluate training-seed variability before
  changing the objective.

Do not implement E004 from this report alone.
