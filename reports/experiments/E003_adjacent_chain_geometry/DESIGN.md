# E003 Adjacent Chain Geometry

## Status

Finalized as an analysis/documentation-only experiment. CUDA calibration,
worst-case stress, the full-data step-2000 preflight, full training, calibrated
evaluation, and the paired E002-versus-E003 comparison have completed. E003 is
not promoted to the new baseline; E002 remains the current baseline.

## Motivation

E002 added stochastic EDM spectral regularization to E001 and produced the
intended global effect in the 375-sample paired E001-versus-E002 comparison:

- negative eigenvalue mass improved by `0.00473265`;
- rank3 residual improved by `0.00676679`;
- classical MDS stress improved by `0.00381575`;
- strict empirical real-like geometry remained 0/375;
- adjacent-residue RMSE significantly worsened by `0.02684371`, with bootstrap
  CI `[0.01979658, 0.03430045]` in the degradation direction.

Adjacent RMSE worsened at every requested length:

| Requested length | Adjacent RMSE degradation |
| ---: | ---: |
| 64 | 0.0527641 |
| 128 | 0.00605616 |
| 256 | 0.0140943 |
| 384 | 0.0341574 |
| 500 | 0.0426828 |

Triangle changes were mixed across lengths and the overall bootstrap interval
included zero. E003 therefore adds only local adjacent-chain geometry. It does
not add triangle, contact, non-negativity, radius-of-gyration, projection, or
architectural changes.

## Controlled Intervention

E003 preserves E002 exactly except for a disabled-by-default adjacent-residue
auxiliary loss:

- E001/E002 architecture unchanged;
- full recovered train/validation manifests unchanged;
- normalization unchanged;
- batch size 2 and gradient accumulation 4 unchanged;
- mixed precision `float16` unchanged;
- v-prediction unchanged;
- cosine diffusion schedule unchanged;
- optimizer and checkpoint settings unchanged;
- E002 stochastic EDM spectral loss retained at weight `0.01`, 500-step warmup,
  subset size 64, one subset per sample, component weights 1.0, seed 2002.

The adjacent-chain term is computed from reconstructed `x0_hat`, using the
repository's existing v-prediction-to-x0 conversion and the same normalization
scale used by E002 for physical Angstrom conversion.

## Loss Definition

For each valid protein, E003 compares only upper adjacent-diagonal entries:

```text
D[i, i+1], i = 0,...,N-2
```

Each symmetric C-alpha neighbour pair is counted once. Padded residues and
invalid adjacent pairs from the biological pair mask are excluded. The target is
the actual clean training distance matrix in physical Angstrom units, not a
hardcoded 3.8 Angstrom value, because real examples may contain experimental
variation or chain discontinuities retained by preprocessing.

The raw adjacent loss is SmoothL1/Huber:

```text
0.5 * delta^2 / beta, if |delta| < beta
|delta| - 0.5 * beta, otherwise
```

E003 uses `adjacent_auxiliary_huber_beta_angstrom: 0.25`. This is a robust
physical scale: deviations below a quarter Angstrom are treated quadratically,
while larger local-chain errors enter linearly so early preflight gradients are
bounded. It intentionally regularizes to the empirical clean adjacent distances
rather than to a universal peptide-geometry constant.

## Configuration

New fields default to disabled or neutral values:

| Field | Default | Meaning |
| --- | ---: | --- |
| `adjacent_auxiliary_loss_enabled` | `false` | Turns on adjacent-chain regularization. |
| `adjacent_auxiliary_loss_weight` | `0.0` | Weight multiplying the raw adjacent SmoothL1 loss. |
| `adjacent_auxiliary_loss_warmup_steps` | `0` | Linear optimizer-step warmup. |
| `adjacent_auxiliary_huber_beta_angstrom` | `0.25` | SmoothL1 beta in physical Angstrom units. |

The E003 preflight config is
`configs/train_recovered_full_v_axial_edm_chain_e003.yaml`.

The definitive full-training config is
`configs/train_recovered_full_v_axial_edm_chain_e003_full.yaml`, with output
directory `outputs/recovered_full_b2_v_axial_edm_chain_e003`.

## CPU No-Update Calibration

Calibration used no optimizer updates, no checkpoint writes, and no sampling.
The profiler measured diffusion, EDM spectral, and adjacent-chain gradient norms
plus cosine similarities among those gradients.

Selected E002 checkpoint, general full-data batches:

| Adjacent weight | Median adjacent/diff ratio | p90 | Max | Median total auxiliary/diff ratio | p90 total | Max total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00005 | 0.0272672 | 0.0296125 | 0.0301988 | 0.0850590 | 0.101417 | 0.105507 |
| 0.00010 | 0.0545343 | 0.0592250 | 0.0603976 | 0.103706 | 0.117848 | 0.121383 |
| 0.00020 | 0.109069 | 0.118450 | 0.120795 | 0.148748 | 0.162198 | 0.165560 |
| 0.00030 | 0.163603 | 0.177675 | 0.181193 | 0.198491 | 0.213836 | 0.217673 |

Selected E002 checkpoint, N=495-500 batches:

| Adjacent weight | Median adjacent/diff ratio | p90 | Max | Median total auxiliary/diff ratio | p90 total | Max total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00005 | 0.0163844 | 0.0199491 | 0.0208403 | 0.0301798 | 0.0307907 | 0.0309434 |
| 0.00010 | 0.0327688 | 0.0398983 | 0.0416807 | 0.0419386 | 0.0436406 | 0.0440661 |
| 0.00020 | 0.0655375 | 0.0797966 | 0.0833613 | 0.0713153 | 0.0797489 | 0.0818573 |
| 0.00030 | 0.0983070 | 0.119696 | 0.125043 | 0.102756 | 0.118315 | 0.122204 |

Observed cosine similarities for the selected checkpoint at adjacent weight
`0.0001` did not show extreme antagonistic tails in this small CPU calibration:

- general batches: diffusion/EDM `0.12465` and `-0.05766`,
  diffusion/adjacent `0.19452` and `0.34115`, EDM/adjacent `0.64576` and
  `0.14867`;
- N=495-500 batches: diffusion/EDM `0.09229` and `0.06420`,
  diffusion/adjacent `0.24630` and `0.19003`, EDM/adjacent `-0.21176` and
  `0.33934`.

Provisional weight: `adjacent_auxiliary_loss_weight: 0.0001`. It lands in the
requested 5-10% median adjacent/diffusion band for the selected E002 checkpoint
on general CPU batches, keeps p90 total auxiliary/diffusion below 25%, and keeps
the N=495-500 total auxiliary ratio modest. The N=495-500 adjacent ratio is below
the nominal target, so CUDA calibration should check whether the long-chain
signal remains acceptable before the 2000-step preflight.

## CUDA Calibration And Preflight Evidence

Selected adjacent-chain settings:

- `adjacent_auxiliary_loss_weight: 0.0001`
- `adjacent_auxiliary_huber_beta_angstrom: 0.25`
- `adjacent_auxiliary_loss_warmup_steps: 500`

General CUDA profile at adjacent weight `0.0001`:

| Quantity | Median | p90 | Maximum |
| --- | ---: | ---: | ---: |
| Adjacent/diffusion gradient ratio | 0.0385003 | 0.0593762 | 0.0722183 |
| Combined auxiliary/diffusion gradient ratio | 0.0916737 | 0.128118 | 0.150298 |

N=495-500 CUDA profile:

| Quantity | Median | p90 | Maximum |
| --- | ---: | ---: | ---: |
| Adjacent/diffusion gradient ratio | 0.0514392 | 0.114064 | 0.176750 |
| Combined auxiliary/diffusion gradient ratio | 0.0877296 | 0.162382 | 0.235053 |

Median gradient cosines:

| Profile | diffusion/adjacent | EDM/adjacent | diffusion/EDM |
| --- | ---: | ---: | ---: |
| General CUDA | 0.334203 | 0.331063 | 0.201883 |
| N=495-500 CUDA | 0.312720 | 0.211347 | 0.211596 |

Negative tails were occasional and modest, with no systematic gradient
antagonism.

Worst-case CUDA stress:

- N=495-500;
- batch size 2;
- gradient accumulation 4;
- 10 steps plus resume to 11;
- peak allocated memory approximately 4.13 GiB;
- zero AMP overflows;
- zero skipped updates;
- successful checkpoint resume.

Full-data step-2000 preflight:

- 8008 JSONL records;
- two isolated AMP overflows;
- final AMP scale 16384;
- maximum consecutive overflows 1;
- no non-finite losses;
- successful resume to step 2001;
- frozen step-2000 SHA-256:
  `c1e63f84522f22aaeab1ee6627fa60906c46f55bc4ed811d872768e535635764`.

First-versus-last quartile medians:

| Quantity | First quartile median | Last quartile median |
| --- | ---: | ---: |
| Adjacent loss | 7.39698 | 1.88258 |
| Diffusion loss | 0.0568248 | 0.0244930 |
| EDM loss | 0.330965 | 0.0825215 |
| EDM negative component | 0.141549 | 0.0287379 |
| EDM rank3 component | 0.188910 | 0.0514521 |
| Weighted adjacent contribution | 0.000296195 | 0.000188258 |
| Weighted EDM contribution | 0.00128259 | 0.000825215 |

## Step-2000 Generated Screening

The paired step-2000 generated screening bank used 10 paired samples: two each
at N=64, 128, 256, 384, and 500, with the same sampling seeds for E002 and E003.

Observed effects:

- adjacent RMSE-to-3.8 Angstrom improved in 9/10 samples;
- mean adjacent RMSE-to-3.8 Angstrom improvement was `0.452242`;
- negative eigenvalue mass improved in 9/10 samples;
- mean negative-eigenvalue-mass improvement was `0.00192891`;
- rank3 residual improved in 9/10 samples;
- mean rank3 improvement was `0.00630272`;
- triangle behavior was mixed;
- negative-distance fraction was slightly improved overall.

Mean adjacent RMSE-to-3.8 Angstrom by requested length:

| Requested length | E002 | E003 |
| ---: | ---: | ---: |
| 64 | 3.57996 | 3.56029 |
| 128 | 2.71956 | 2.46473 |
| 256 | 2.19848 | 1.81046 |
| 384 | 2.24399 | 1.35034 |
| 500 | 2.26398 | 1.55894 |

This is a small screening bank and not final statistical evidence. The
definitive evaluation must use calibrated real controls, not only a fixed
3.8 Angstrom reference.

## Scientific Boundary

Adjacent-chain SmoothL1 regularization is established physical regularization,
not a novelty claim. The experiment's purpose is to test whether combining E002's
global spectral consistency with a narrow local backbone-distance constraint
resolves the specific E002 trade-off: improved global embeddability but degraded
adjacent-residue geometry.

E003 should remain causal and interpretable. Triangle, contact, non-negativity,
radius-of-gyration, projection, and architecture changes are deferred unless
E003 evidence justifies a new experiment.

## Final Calibrated Outcome

The final selected-model comparison is E002 epoch 5/global step `169181` versus
E003 epoch 5/global step `169182`. The selected E002 checkpoint SHA-256 is
`2b6b9967a5de6035accad7cc1c24e379743459531c641727221541fde9aa669f`; the
selected E003 checkpoint SHA-256 is
`bdbaf25ee3b9d7d7a7cfe2ebe03a12b09b16261b1a8a3e95d28b2c48058ca101`.

E003 training was stable: best validation epoch 5, validation loss
`0.0183565374`, `169182` optimizer steps, `63` isolated AMP overflow events,
maximum consecutive overflows `1`, no non-finite losses, and accumulation state
`0` at the selected checkpoint. Diffusion, stochastic EDM spectral, and adjacent
median losses decreased across epochs.

On the 375-sample calibrated paired comparison, E003 modestly improves
adjacent-residue distance RMSE by `0.0083769942` with bootstrap CI
`[0.0007474087, 0.0161266259]`. It significantly degrades negative eigenvalue
mass by `-0.0042051035` with CI `[-0.0050886455, -0.0033259991]` and triangle
violation fraction by `-0.0004375` with CI
`[-0.0006771159, -0.0001874674]`. Rank-3 residual and classical MDS stress are
inconclusive because their bootstrap intervals include zero.

Strict empirical real-like geometry remains `0/375` for both E002 and E003.
Heuristic EDM-quality validity is essentially unchanged: E002 `0.128`, E003
`0.1306667`, exact McNemar p-value `1.0`.

Distribution matching is mixed: E003 moves closer to real controls in `25/60`
descriptor-length rows by standardized mean discrepancy, `24/60` by
Wasserstein distance, and `23/60` by KS statistic. Diversity does not collapse:
generated/real diversity ratios move closer to one in `13/20` rows. Novelty is
mixed and approximate, with calibrated novelty ratios moving closer to the real
calibration baseline in `6/15` rows.

`generated_count=0` warnings for `empirical_real_like_geometry_pass` novelty
subgroups are expected because no generated samples passed strict empirical
validity. They are not missing-data warnings.

Bootstrap confidence intervals quantify variability across the generated pairs
in this comparison, not training-seed uncertainty. Since there is one training
run per configuration, E003 should be interpreted as evidence of a
local-versus-global objective trade-off, not as a universal causal result.

Before E004, run an analysis-only gradient diagnostic that stratifies diffusion,
EDM, and adjacent gradient norms and pairwise cosine similarities by diffusion
timestep bins, requested-length bins, and optionally their interaction. If
conflict concentrates at high noise, consider timestep- or SNR-gated adjacent
loss; if it concentrates at long lengths, consider length-dependent weighting;
if it is widespread, abandon the adjacent loss or investigate a multi-objective
gradient method; if no systematic conflict appears, evaluate training-seed
variability before changing the objective.
