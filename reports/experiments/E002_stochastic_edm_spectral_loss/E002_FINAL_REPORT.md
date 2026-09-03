# E002 Final Report: Stochastic EDM Spectral Auxiliary Loss

## Scope

E002 preserves the E001 symmetric axial-attention architecture and changes only
the training objective by adding the stochastic EDM spectral auxiliary loss.
This report records the completed training, calibrated evaluation, and
375-sample paired E001-versus-E002 comparison.

## Training

- Selected checkpoint: epoch 5/global step 169181
- Checkpoint SHA-256:
  `2b6b9967a5de6035accad7cc1c24e379743459531c641727221541fde9aa669f`
- Best validation loss: `0.0185558852`
- E001 best validation loss: `0.0184613932`
- AMP overflows: 64 isolated events
- Maximum consecutive overflows: 1
- Non-finite losses: none

Auxiliary-loss evolution:

| Quantity | First quartile median | Last quartile median |
| --- | ---: | ---: |
| Total auxiliary loss | 0.033532 | 0.014498 |
| Negative component | 0.007644 | 0.002655 |
| Rank3 component | 0.023429 | 0.010450 |

## CUDA Calibration And Preflight

Gradient calibration at auxiliary weight `0.01`:

| Batch set | Median aux/diff grad ratio | p90 | Maximum |
| --- | ---: | ---: | ---: |
| General full-data batches | 0.0553471 | 0.162501 | 0.183165 |
| Worst-case N=495-500 batches | 0.0166878 | 0.0431464 | 0.0627834 |

The lower N=500 signal is a limitation of using one fixed-size 64-residue
subset. E002 leaves that limitation visible for causal attribution.

Worst-case CUDA stress covered lengths 495-500, batch size 2, gradient
accumulation 4, 10 optimizer steps plus resume to step 11, peak allocated memory
approximately 4.13 GiB, zero AMP overflows, zero skipped updates, and successful
checkpoint resume.

The full-data preflight covered 2000 optimizer steps plus resume to step 2001
with auxiliary weight `0.01`, 500-step warmup, subset size 64, and one subset
per sample. Peak allocated memory was approximately 4.10 GiB, with 2 early AMP
overflows, final AMP scale 16384, final consecutive overflows 0, no non-finite
losses, and successful checkpoint serialization/resume.

## Paired E001-Versus-E002 Comparison

The corrected selected-model comparison is E001 epoch 4/global step 160166
versus E002 epoch 5/global step 169181. Selected epochs, optimizer steps, and
training histories differ, so this is not a perfectly controlled causal
auxiliary-loss ablation.

Overall paired effects:

| Metric | E001 mean | E002 mean | Mean improvement | 95% bootstrap CI | Improved pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Negative eigenvalue mass | 0.09742895 | 0.09269630 | 0.00473265 | [0.00359787, 0.00582859] | 68.8% |
| Rank3 residual | 0.15445108 | 0.14768429 | 0.00676679 | [0.00427944, 0.00915147] | 65.07% |
| Classical MDS stress | 0.07592901 | 0.07211326 | 0.00381575 | [0.00172839, 0.00587892] | 60.53% |
| Adjacent-residue RMSE | 0.25242906 | 0.27927277 | -0.02684371 | [-0.03430045, -0.01979658] | 35.47% |
| Triangle violations | 0.00192448 | 0.00219271 | -0.00026823 | [-0.00056901, 0.00001563] | 40.53% |

The triangle-violation interval includes zero, so this is not a conclusive
regression. The adjacent-residue RMSE interval is entirely negative, so E002 has
a statistically supported degradation in local adjacent-residue geometry.

## Validity

- Strict empirical real-like geometry remains 0/375 for both E001 and E002.
- Heuristic pass fraction changes from 14.4% to 12.8%.
- McNemar `p=0.30746`; the heuristic validity change is not statistically
  significant.

Per-length validity:

| Length | Heuristic E001 | Heuristic E002 | Strict E001 | Strict E002 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.53 | 0.43 | 0.0 | 0.0 |
| 128 | 0.01 | 0.04 | 0.0 | 0.0 |
| 256 | 0.0 | 0.01 | 0.0 | 0.0 |
| 384 | 0.0 | 0.0 | 0.0 | 0.0 |
| 500 | 0.0 | 0.0 | 0.0 | 0.0 |

## Per-Length Primary Effects

Positive improvement means E002 is lower/better than E001.

| Length | Negative eigenvalue mass | Rank3 residual | MDS stress | Triangle violations | Adjacent RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | -0.00321908 | 0.00303612 | 0.00311468 | -0.00119629 | -0.0527641 |
| 128 | 0.00861809 | 0.00759450 | 0.00135654 | 0.0000634766 | -0.00605616 |
| 256 | 0.00537887 | 0.00742090 | 0.00422444 | -0.0000146484 | -0.0140943 |
| 384 | 0.0115441 | 0.0116108 | 0.00768017 | 0.0000976563 | -0.0341574 |
| 500 | 0.00478994 | 0.00607424 | 0.00709321 | 0.000371094 | -0.0426828 |

## Distribution Matching

Per-length counts of 12 descriptors where E002 moved closer to the real
distribution:

| Length | Mean discrepancy improved | Wasserstein improved | KS improved |
| ---: | ---: | ---: | ---: |
| 64 | 2/12 | 2/12 | 1/12 |
| 128 | 7/12 | 7/12 | 6/12 |
| 256 | 4/12 | 6/12 | 3/12 |
| 384 | 7/12 | 7/12 | 3/12 |
| 500 | 7/12 | 7/12 | 2/12 |

## Diversity

Per-length generated/real diversity-ratio movement toward 1:

| Length | Metrics closer | Mean closeness improvement |
| ---: | ---: | ---: |
| 64 | 2/4 | -0.0394921 |
| 128 | 2/4 | 0.00957272 |
| 256 | 1/4 | -0.00590838 |
| 384 | 2/4 | 0.00945576 |
| 500 | 1/4 | 0.0375854 |

## Novelty

Per-length calibrated novelty-ratio movement toward the real calibration
baseline:

| Length | Metrics closer | Mean closeness improvement |
| ---: | ---: | ---: |
| 64 | 3/3 | 0.136397 |
| 128 | 2/3 | 0.0356293 |
| 256 | 2/3 | -0.0334791 |
| 384 | 2/3 | 0.0307844 |
| 500 | 1/3 | -0.00956749 |

## Descriptor Cache Reuse

E001 training descriptors were reused because they are model-independent and the
manifest, normalization, descriptor implementation, and evaluation protocol were
identical.

Descriptor SHA-256:
`ad738db4085c42ff44f91a62d023d26e3416f548faa2d2d17090ef6434f8f865`

## Interpretation

The spectral loss operates on squared distances, so it does not distinguish
positive and negative predicted distances. Soft PSD/rank penalties on sampled
submatrices do not guarantee global triangle inequalities.

E002 confirms that stochastic spectral EDM regularization improves global
low-dimensional embeddability, negative-spectrum consistency, and MDS stress. It
does not achieve strict empirical protein geometry and introduces a statistically
supported degradation in local adjacent-residue geometry. Retain E002 as a
positive partial experiment, not as the final model.

Do not claim that E002 produces physically valid proteins.
