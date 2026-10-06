# E012 causal RoPE pilot — CSEQ-C

Exactly 2,000 successful optimizer updates; no extension, structure conditioning or biological generation. Scientific interpretation follows the frozen primary gates and independent confirmation rules.

The model establishes ordered-prefix sensitivity and benefit from context beyond eight residues on both fixed panels (Gates C and D). It does not meet the required 0.05-nat improvement over either unigram baseline (Gate A), and the 20–64 and 65–128 strata have positive model-minus-bucketed-unigram point estimates (Gate E). Those short-stratum intervals include zero, so this is a preregistered point-estimate failure rather than established significant harm. Gate B passes on the primary panel; the independent bigram comparison has favorable direction but its interval includes zero. Under the sealed classifier the result is CSEQ-C, not CSEQ-A. A larger budget might be informative, but these results do not establish that more training will remedy the likelihood and stratum failures.

## Reproducibility and preflight

Base E011 `30b657be81cd7bd943918980812d68ecaa0778de`; preparation `76814fd071a20f1bcc1e381533bfe195381aa1ca`; execution start `d5b793ca60b6dcba0d9487bb879f08186d0b6169`. Contract SHA256 `1a56b8958bb9a81c7dc269bce7c3efa79f19fc6dc85498beb09cfb10f346e612`. The existing causal architecture is unchanged: 15,533,952 parameters, seed 12012, scratch initialization. Inputs: previous canonical residues, BOS/PAD, causal/key mask, RoPE and continuous requested length. All structural and metadata predictive features are excluded. AdamW .0003, betas .9/.95, decay .01, 100 warmup updates, bounded cosine, clip 1. Physical 8 × accumulation 8 = 64 equally weighted proteins; bfloat16, no scaler.

The audited E011 TRAIN/validation population (231,743/23,307) and exact disjoint panels (2,048/2,048) are reused. Raw 744-shard hashes, sequence-only projection and protected zero cross-split overlap checks passed; historical homology policy 30% identity/80% coverage. No dataset/checkpoint is committed. See data_integrity.json, quality.json and cuda_smoke.json.

## Likelihood trajectory

| Update | Primary CE | Independent CE |
| --- | --- | --- |
| 0 | 3.191277 | 3.193664 |
| 250 | 2.888059 | 2.891987 |
| 500 | 2.904731 | 2.905188 |
| 1000 | 2.913907 | 2.916776 |
| 1500 | 2.889636 | 2.894788 |
| 2000 | 2.888072 | 2.893015 |

## Final likelihood and statistical baselines

| Panel | Normal | Global unigram | Bucket unigram | Bigram | Trigram |
| --- | --- | --- | --- | --- | --- |
| primary | 2.888072 | 2.900035 | 2.899043 | 2.894704 | 2.906180 |
| independent | 2.893015 | 2.901697 | 2.901314 | 2.896606 | 2.909563 |

## Gates and paired identity bootstrap

10,000 resamples, seed 12112; delta is model minus baseline or full/normal prefix minus comparator. Units are nats/token. Intervals are 95% percentile CIs. Negative values favor the model.

| Panel | Comparison | Delta [95% CI] |
| --- | --- | --- |
| primary | bigram | -0.006632 [-0.011986, -0.001544] |
| primary | bucket_unigram | -0.010971 [-0.016115, -0.006150] |
| primary | global_unigram | -0.011963 [-0.017364, -0.006796] |
| primary | last1 | -0.348581 [-0.361589, -0.335687] |
| primary | last8 | -0.100602 [-0.108216, -0.093027] |
| primary | shuffle | -0.032555 [-0.038672, -0.026641] |
| primary | trigram | -0.018107 [-0.023427, -0.013021] |
| independent | bigram | -0.003591 [-0.008077, +0.001014] |
| independent | bucket_unigram | -0.008299 [-0.012679, -0.003840] |
| independent | global_unigram | -0.008682 [-0.013199, -0.004006] |
| independent | last1 | -0.343472 [-0.356486, -0.330105] |
| independent | last8 | -0.097153 [-0.104570, -0.089697] |
| independent | shuffle | -0.028323 [-0.034251, -0.022306] |
| independent | trigram | -0.016547 [-0.021194, -0.011889] |

| Panel | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| primary | False | True | True | True | False |
| independent | False | False | True | True | False |

Independent confirmation (distinct from requiring the primary effect thresholds twice): {"A": true, "B": true, "C": true, "D": true, "E": false}

## Prefix order and context windows

Up to 16 fixed interior prediction targets per protein; identical targets and prefix composition, no future residues. Complete diagnostics were preregistered for update 2000; likelihood was evaluated at every boundary.

| Panel | Last1 | Last4 | Last8 | Last16 | Last32 | Last64 | Full | Shuffle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary | 3.234268 | 3.003182 | 2.986289 | 2.965781 | 2.940011 | 2.908316 | 2.885687 | 2.918242 |
| independent | 3.238589 | 3.012763 | 2.992270 | 2.979852 | 2.954383 | 2.917191 | 2.895117 | 2.923440 |

| Panel | KL normal||shuffle | JS | Top1 change | Top3 set change |
| --- | --- | --- | --- | --- |
| primary | 0.108992 | 0.024958 | 0.647491 | 0.885162 |
| independent | 0.107906 | 0.024697 | 0.656677 | 0.885406 |

## Length ablation

Zero continuous length embedding output at evaluation; no retraining. Neutral minus normal full-likelihood CE:

| Panel | Delta [95% CI] |
| --- | --- |
| primary | +0.026575 [+0.023529, +0.029781] |
| independent | +0.024030 [+0.021344, +0.026821] |

## Five length strata

| Panel | Length | Normal−bucket | Normal-prefix−shuffle | Full−last8 |
| --- | --- | --- | --- | --- |
| primary | 20–64 | +0.007110 [-0.008431, +0.022873] | -0.032779 [-0.049466, -0.016413] | -0.053561 [-0.071974, -0.035244] |
| primary | 65–128 | +0.007408 [-0.002239, +0.017548] | -0.028319 [-0.042204, -0.013823] | -0.108919 [-0.127879, -0.089376] |
| primary | 129–256 | -0.016470 [-0.034037, -0.003962] | -0.042419 [-0.062590, -0.027154] | -0.130339 [-0.152801, -0.110533] |
| primary | 257–384 | -0.025284 [-0.030251, -0.021036] | -0.035248 [-0.044020, -0.026545] | -0.113794 [-0.127409, -0.100537] |
| primary | 385–500 | -0.027693 [-0.030279, -0.025127] | -0.023997 [-0.029745, -0.018229] | -0.096418 [-0.107882, -0.085103] |
| independent | 20–64 | +0.012283 [-0.004263, +0.029121] | -0.027938 [-0.046734, -0.008351] | -0.067056 [-0.085197, -0.048722] |
| independent | 65–128 | +0.003459 [-0.007196, +0.015006] | -0.032428 [-0.049584, -0.014615] | -0.102413 [-0.122293, -0.082525] |
| independent | 129–256 | -0.009784 [-0.017533, -0.003038] | -0.027826 [-0.039185, -0.016720] | -0.111659 [-0.127766, -0.095573] |
| independent | 257–384 | -0.023134 [-0.026431, -0.019933] | -0.030008 [-0.037397, -0.022606] | -0.106962 [-0.119459, -0.094994] |
| independent | 385–500 | -0.024393 [-0.026853, -0.021991] | -0.023405 [-0.029622, -0.017265] | -0.097700 [-0.110474, -0.085247] |

## Additional metrics

Token-weighted CE/perplexity, top-1/top-3 accuracy and relative-position deciles:

```json
{
  "primary": {
    "equal_protein_ce": 2.8880722931862692,
    "token_weighted_ce": 2.871712292642669,
    "equal_protein_perplexity": 17.95865718660889,
    "token_weighted_perplexity": 17.667243806496483,
    "equal_protein_top1": 0.104715325757752,
    "equal_protein_top3": 0.27331261726430967,
    "token_weighted_top1": 0.10501703789782313,
    "token_weighted_top3": 0.2745271337177021,
    "relative_position_ce_deciles": [
      2.9067066292627715,
      2.9234098156011896,
      2.9147377175348765,
      2.894973488007963,
      2.883750185925578,
      2.8697886861209554,
      2.875978796802883,
      2.870767838321626,
      2.8731317800629768,
      2.8646940517937765
    ]
  },
  "independent": {
    "equal_protein_ce": 2.8930154010886326,
    "token_weighted_ce": 2.8756093106110874,
    "equal_protein_perplexity": 18.04764853257311,
    "token_weighted_perplexity": 17.736227701498606,
    "equal_protein_top1": 0.10332041483252397,
    "equal_protein_top3": 0.2681510036227337,
    "token_weighted_top1": 0.10357595920526173,
    "token_weighted_top3": 0.27083347698886073,
    "relative_position_ce_deciles": [
      2.9146897914470173,
      2.926226409850642,
      2.9110757791786455,
      2.893215892436274,
      2.893217740485852,
      2.887832943946705,
      2.885798671748489,
      2.8819708144292235,
      2.8723661313415505,
      2.8592534643248655
    ]
  }
}
```

## Verification and safety

All checkpoint hashes, optimizer/scheduler/sampler counters and update-0 initialization independently verified. All gate intervals and trajectory means reproduced. E011 remains immutable S1-E. Structural worktree snapshots are reported honestly in postflight_verification.json; any owner changes are distinguished from E012 writes (none).

## Limitations

One seed and 2,000 updates. PDB-derived cohort selection bias persists despite sequence-only model inputs. Duplicate TRAIN sequences retain sample weight. Identity bootstrap does not model within-cluster dependence. Prefix diagnostics cover fixed interior positions, not every residue. Biological quality and foldability were not evaluated.

## Exactly one recommended next experiment

A separately preregistered 10,000-update causal RoPE budget replication with the same capacity, objective and gates.


## Fixed interpretation and independent verification

Fixed before execution: the empirical trigram baseline is worse than unigram on both panels. Beating trigram alone is not evidence of contextual learning. Ordered-context evidence depends on the preregistered shuffle and context-window gates. No gate is changed.

A separate bootstrap implementation reproduced all 86 reported gate/stratum/ablation intervals exactly, using 10,000 resamples. It independently reproduced the gates and classification, six checkpoint hashes, optimizer/scheduler/sampler counters and update-0 fingerprint. Protected contract/data/source hashes still match.

The sealed runner labels its execution-start HEAD as `preparation_commit` in raw results.json; that raw provenance is preserved. This handoff distinguishes the original preparation commit from the deferral/execution-start commit. All scientific outputs are unchanged. E011 and Phase 4C match execution-start snapshots. Phase 4D independently advanced from 43864b3… to f9b42c3… through its owner’s bounded correction oracle v4 commits and further owner edits; E012 made no writes there. Its branch remains unchanged; concurrent working-tree status is recorded in independent_verification.json.

Training telemetry records every update; training_summary.json publishes cadence-50 means, exposure counts, gradient/clipping and AMP telemetry. The processed totals are 128,000 proteins and 23,487,604 residues. Training CE is not used as a success gate.
