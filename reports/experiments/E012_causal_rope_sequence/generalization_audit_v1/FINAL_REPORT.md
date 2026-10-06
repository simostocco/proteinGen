# E012 V3 — GEN-A: clear overfitting / data-limited

Matched TRAIN CE keeps improving while both held-out panels worsen from 5k to 10k. The expanding gap appears in all five length strata. This supports TRAIN-specific specialization and insufficient effective sequence diversity as the primary current bottleneck; it does not support increasing model capacity on the same corpus as the next intervention. Corpus-size causality is not proven by this retrospective audit.

GEN-A is preferred over GEN-D because the gap expands broadly across every length stratum, rather than being confined to one regime. GEN-B is inconsistent with continued matched TRAIN improvement. GEN-C is inconsistent with the large gap after the earliest causal positions. GEN-E is not supported: hashes, evaluator semantics, exact available historical batch reconstructions and independent statistics checks passed.

Evaluation-only matched TRAIN versus held-out audit. Historical CSEQ-C / CONT-B results and all four checkpoints remain immutable. No optimizer step, continuation, model change or biological generation was performed.

TRAIN panel: 2048 identities; strata 410/410/410/409/409; SHA256 `6924250770c80b0c0df1d56a37b54fec5a0709690d624609c90872877c81b427`. Primary and independent 2048-identity panels are unchanged. CUDA BF16 forward / float32 log-softmax, historical batch 16 evaluator, model.eval and no_grad for all panel metrics.

| Update | TRAIN CE | Primary CE | Independent CE | Primary−TRAIN | Independent−TRAIN |
|---:|---:|---:|---:|---:|---:|
| 2000 | 2.12040 | 2.88807 | 2.89302 | 0.76768 | 0.77262 |
| 5000 | 1.60035 | 2.88023 | 2.88933 | 1.27988 | 1.28898 |
| 7500 | 1.40828 | 2.90717 | 2.91737 | 1.49889 | 1.50909 |
| 10000 | 1.36162 | 2.89410 | 2.90461 | 1.53248 | 1.54300 |

From 5000 to 10000, paired identity-bootstrap CE changes (10000 resamples, seed 12112):

- train: -0.23874 [-0.25898, -0.21907] nats/token.
- primary: +0.01387 [+0.01090, +0.01698] nats/token.
- independent: +0.01528 [+0.01206, +0.01860] nats/token.

Original optimization `CE=...` is instantaneous, pre-update train-mode CE over 64 proteins, with active dropout. Reduction is the mean of valid-token means within each protein, then the 64-protein mean. Physical means are weighted n/64; gradient accumulation does not divide logged CE incorrectly. The sampler follows the natural TRAIN length distribution, not equal stratum quotas; no rolling/epoch averaging, MLM masking or special target filtering. The loss formula matches evaluation but raw logged values are not directly comparable to balanced fixed-panel eval-mode values. AMP cross-entropy kernels introduce a small measured numerical difference; the preflight found 0.0000775 nats/token, not an explanation for the observed gap.

| Checkpoint | Exact historical batch | Original log CE | Reconstructed train-mode CE | Same-batch eval CE | Train−eval |
|---:|---:|---:|---:|---:|---:|
| 2000 | 2001 | 2.20913 | 2.20913 | 2.11024 | 0.09889 |
| 5000 | 5001 | 1.86601 | 1.86601 | 1.77229 | 0.09372 |
| 7500 | 7501 | 1.54638 | 1.54638 | 1.45769 | 0.08869 |
| 10000 | 10000 | 1.25520 | 1.25893 | 1.13370 | 0.12522 |

At 2000/5000/7500, exact next-batch identities and saved incoming RNG are available. At 10000 the exact final batch is replayed from sampler7500; final LR=0 preserves the pre-step parameters, but original incoming dropout RNG is not available. A fresh train-mode draw from saved outgoing RNG is explicitly used; no claim of exact original dropout replay is made there.

Final equal-protein CE by length:

| Length | TRAIN | Primary | Independent | Primary gap | Independent gap |
|---|---:|---:|---:|---:|---:|
| 20–64 | 0.77802 | 2.92861 | 2.96171 | 2.15058 | 2.18369 |
| 65–128 | 0.80105 | 2.91768 | 2.92128 | 2.11664 | 2.12024 |
| 129–256 | 1.07869 | 2.89142 | 2.89562 | 1.81273 | 1.81693 |
| 257–384 | 1.71650 | 2.87271 | 2.88237 | 1.15621 | 1.16588 |
| 385–500 | 2.43731 | 2.85993 | 2.86192 | 0.42262 | 0.42461 |

Final context evidence, same historical shuffle/window protocol:

| Panel | Normal−shuffle (95%CI) | Full−last8 (95%CI) | Full−last1 (95%CI) | KL | JS | Top1 change | Top3 change |
|---|---|---|---|---:|---:|---:|---:|
| train | -1.68608 [-1.74436, -1.62761] | -1.56443 [-1.61692, -1.51001] | -2.18140 [-2.24486, -2.11846] | 1.68380 | 0.31894 | 81.778% | 96.048% |
| primary | -0.06483 [-0.07398, -0.05560] | -0.11321 [-0.12240, -0.10436] | -0.58587 [-0.60770, -0.56398] | 0.20282 | 0.04522 | 74.142% | 93.240% |
| independent | -0.05154 [-0.05998, -0.04296] | -0.11314 [-0.12166, -0.10467] | -0.58834 [-0.61021, -0.56636] | 0.19831 | 0.04435 | 74.780% | 93.304% |

The complete checkpoint/panel/stratum context curves are in results.json. Matched trajectory includes contexts at7500, computed in this audit; historical2k/5k/10k held-out evidence is reused, not relabeled.

| Panel | Context1 | 4 | 8 | 16 | 32 | 64 | Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 3.48835 | 3.02299 | 2.87138 | 2.57256 | 2.18062 | 1.79569 | 1.30695 |
| primary | 3.47698 | 3.04329 | 3.00432 | 2.97306 | 2.94411 | 2.91490 | 2.89111 |
| independent | 3.49545 | 3.04824 | 3.02025 | 2.98525 | 2.95420 | 2.93049 | 2.90711 |

Confidence and entropy use the unchanged full24-way predictive distribution; target-AA probability and twenty canonical predicted means are also reported. No canonical renormalization is performed; residual special-token mass is explicit. Target empirical frequencies are fixed across checkpoints.

| Update | Panel | Token-weighted CE | PPL(equal-protein) | Top1 | Top3 | Entropy | Max probability | Correct-token probability |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2000 | train | 2.46018 | 8.334 | 35.107% | 49.740% | 2.11199 | 0.34887 | 0.29368 |
| 2000 | primary | 2.87171 | 17.959 | 10.472% | 27.331% | 2.80631 | 0.12920 | 0.06547 |
| 2000 | independent | 2.87561 | 18.048 | 10.332% | 26.815% | 2.80847 | 0.12841 | 0.06481 |
| 5000 | train | 2.01993 | 4.955 | 51.420% | 63.203% | 1.60943 | 0.50606 | 0.46149 |
| 5000 | primary | 2.86831 | 17.818 | 10.874% | 27.788% | 2.78174 | 0.13943 | 0.06863 |
| 5000 | independent | 2.87297 | 17.981 | 10.632% | 27.387% | 2.78546 | 0.13836 | 0.06745 |
| 7500 | train | 1.83024 | 4.089 | 57.287% | 68.011% | 1.38785 | 0.57305 | 0.52690 |
| 7500 | primary | 2.88583 | 18.305 | 10.853% | 27.813% | 2.71832 | 0.15935 | 0.07152 |
| 7500 | independent | 2.89012 | 18.493 | 10.563% | 27.388% | 2.72327 | 0.15789 | 0.07012 |
| 10000 | train | 1.78209 | 3.902 | 58.630% | 69.075% | 1.35659 | 0.58223 | 0.54019 |
| 10000 | primary | 2.87674 | 18.067 | 11.066% | 28.018% | 2.73649 | 0.15239 | 0.07122 |
| 10000 | independent | 2.88207 | 18.258 | 10.710% | 27.511% | 2.74170 | 0.15087 | 0.06977 |

AA calibration (all checkpoints/panels): aa_calibration_metrics.csv. Complete token-weighted/equal-protein frequencies, NLL contributions and conditional per-class NLL are in results.json. Relative-position CE (all five strata, exact historical ten deciles) is in relative_position_metrics.csv; explicit short-stratum early-versus-later gaps are in short_position_diagnosis.json.

TRAIN exposure history:
- Update 10000: 2048/2048 panel identities seen, 5656 total exposures; histogram {'2': 488, '3': 1560}.
- Update 2000: 1151/2048 panel identities seen, 1151 total exposures; histogram {'0': 897, '1': 1151}.
- Update 5000: 2048/2048 panel identities seen, 2824 total exposures; histogram {'1': 1272, '2': 776}.
- Update 7500: 2048/2048 panel identities seen, 4247 total exposures; histogram {'2': 1897, '3': 151}.

TRAIN identity/exact-sequence counts: {'natural_training_stratum_exposures_through_10k': {'129–256': 250501, '20–64': 59286, '257–384': 107447, '385–500': 37929, '65–128': 184837}, 'train_panel_unique_exact_sequences': 1733, 'training_identities': 231743, 'training_unique_exact_sequences': 87930}


All five strata show substantial held-out-minus-TRAIN gaps at 10k, including the longest stratum (about 0.42 nats). The largest gaps are in the two shortest strata (about 2.12–2.18 nats). Short failures are not confined to early positions: see the exact ten-decile comparison in short_position_diagnosis.json and relative_position_metrics.csv. The complete requested four-checkpoint likelihood/context/gap table is matched_trajectory.csv.

From 5k to 10k, TRAIN entropy falls 1.6094→1.3566 and mean maximum probability rises 0.5061→0.5822. Primary held-out entropy falls 2.7817→2.7365 and maximum probability rises 0.1394→0.1524 while CE worsens. Independent held-out shows the same pattern. TRAIN conditional target NLL improves for all 20 AA classes; both held-out panels worsen for 12 classes, with the largest increases for C, W and P. Predicted aggregate AA frequencies get closer to held-out target frequencies (primary L1 0.04313→0.02034; independent 0.04463→0.02311), so a simple worsening of global marginals does not explain the likelihood regression. These AA observations are descriptive telemetry, not new gates.

The corpus has 87,930 exact sequences across 231,743 TRAIN identities; the fixed TRAIN panel has 1,733 exact sequences across 2,048 identities. The preregistered identity bootstrap is retained, and dependence among biological duplicates can make its intervals optimistic. A future ~1M nonredundant corpus would be about 11.4× the current exact-sequence diversity; this is a planning target, not data acquisition or proof that size alone fixes the problem.

Interpretation and limitations:

- Retrospective four-checkpoint audit; no checkpoint selected.
- TRAIN panel has2048 identities but1733 exact sequences; fullTRAIN has231743 identities but87930 exact sequences. Identity bootstrap can be optimistic for correlated biological duplicates.
- At 2000 897 panel identities were not yet exposed, and unseen identities can share TRAIN sequences/homology with exposed identities. Exposure history is explicit.
- Exact last10000 batch membership/weights reconstructed; original incoming dropout RNG unavailable.
- Frozen held-out context records at2000/5000/10000 reused under protected hashes and independently replayed fresh; all panels at7500 evaluated now.
- No intervention on corpus size, diversity, regularization or homology was performed; data-size causality unproven.
- Repository-wide tests have documented missing legacy artifact dependencies; focused sequence/audit tests pass.

Exactly one recommended next experiment: Run one controlled corpus-diversification experiment with approximately 1M deduplicated, cluster-balanced sequence-only proteins, retaining the 15.5M architecture and protected homology-excluded held-out panels with a matched-compute control.

Fresh full-context verification: fresh_context_replay_verification.json; all six historical held-out panel/checkpoint combinations freshly replayed with parsed metrics bitwise identical. Independent verification: independent_verification.json. Focused tests, Ruff/format/syntax/whitespace and the legacy full-suite failure inventory are in quality.json and full_suite_failure_recheck.json. All source/panel/baseline/checkpoint hashes were checked before and after inference; model-state fingerprints were unchanged. Other worktrees were read-only; Phase4D concurrently advanced through its own CPU-only work, so its independently owned commits/status changes are recorded rather than attributed to this audit.

No training or biological generation launched.
