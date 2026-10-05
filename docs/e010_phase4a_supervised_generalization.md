# E010 Phase 4A preparation

E010 Phase 3 is closed as `passed_shared_capacity_selection`, selecting the 12,844,352-parameter Large model. Phase 4A is prepared as a bounded, non-authorizing supervised generalization pilot. Existing E010 artifacts were preserved. At the time of this preparation note, training and Phase 4B preparation had not been performed. Subsequent v2/v3, multicorruption, and real-denoiser Phase 4B experiments are now complete; see [the forensic closeout](e010_phase4b_forensic_closeout.md). This preparation document remains a historical contract, not the current project status. Production sampler integration remains a separate decision.

## Data and fixed corruptions

The plan reads only `data/full/splits/train.parquet` and `data/full/splits/validation.parquet`, requesting only `sample_id`, `length`, `path`, and `split`. It does not open the prospective/test split. Final selections are identity-disjoint:

| Length stratum | Training identities | Development identities |
|---|---:|---:|
| 20–64 | 410 | 64 |
| 65–128 | 410 | 64 |
| 129–256 | 410 | 64 |
| 257–384 | 409 | 64 |
| 385–500 | 409 | 64 |
| **Total** | **2,048** | **320** |

Metadata-ranked candidates with a source identity, length, mask, or finiteness mismatch were skipped during read-only validation; 44 candidate rows were rejected before the requested counts were filled. All 32 Phase 2 reference corruptions were independently reconstructed and hash-matched. The deterministic cache contains one fixed-timestamp, hash-pinned corruption for every selected identity (2,368 total); every archive and target/corruption/mask tensor hash was rechecked.

## Batch smoke and future training schedule

The length-500 Large-model smoke included forward, backward, finite-gradient checks, and one AdamW step. Batch 18 passed; batch 19 failed allocation, making **18 the largest supported microbatch** in the search. At batch 18, peak memory was 14,095 MiB allocated / 15,454 MiB reserved. The prepared schedule accumulates one microbatch from each of the five length strata before an optimizer step: nominal effective batch 90, 23 optimizer updates per exposure epoch, and 1,150 planned updates over 50 epochs. The final partial step is 68 examples because of the 409/410 per-stratum counts.

The fresh model uses the exact Phase 3 Large architecture and a new deterministic initialization seed (41041). The update-0 checkpoint contains fresh model weights, empty AdamW state, no scheduler, Python/NumPy/Torch CPU/CUDA RNG state, sampler RNG state, the fully materialized balanced schedule, per-identity exposure state, and verified input/cache pins. Its SHA-256 is `1abce7d09b24edd365d35bbf9a9c7bc7ae61c3e38a5950c88bfeade29a744b3a`. The retained schedule plans 102,400 total exposures, exactly 50 for each training identity. **No optimizer step was executed.**

The planned loss is a mean of per-structure masked coordinate MSE plus the Phase 3 `1e-5` Cartesian residual penalty. Per-structure normalization gives every identity equal weight regardless of length. Evaluation is planned after 5, 10, 20, 35, and 50 exposures; checkpoint selection uses development mean aligned RMSE only, with the earlier boundary winning exact ties.

## Corrupted-input development baseline

The 320 development structures are evaluated using their fixed corrupted coordinates before training:

| Metric | Corrupted-input baseline |
|---|---:|
| Mean aligned RMSE | 1.1810 Å (95% identity-bootstrap CI 1.1556–1.2070) |
| Median / maximum aligned RMSE | 1.1052 / 1.7765 Å |
| Error-versus-length slope | 0.001481 Å/residue |
| Mean i+1 / i+2 / i+3 distance RMSE | 0.9709 / 0.9707 / 0.9707 Å |
| Chirality inversion rate | 13.60% |

Stratum-specific mean RMSE and bootstrap intervals, per-identity baseline values, geometry telemetry, and collapse checks are recorded in the baseline JSON. It shows no coordinate or diversity collapse. Paired trained-versus-corrupted changes will be produced only if a separate training execution is authorized.

The predeclared gates remain: at least 30% overall development RMSE reduction; at least 20% reduction in every stratum with no stratum worsening; no slope increase; finite outputs; no increase in chirality inversion rate; at least 20% improvement in mean i+1/i+2/i+3 distance RMSE; and no coordinate or diversity collapse. Bootstrap intervals use development identities as the sampling unit (10,000 paired resamples, 95%).

## Prepared artifacts

- [Plan and selected identities](../reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v1/phase4a_plan.json)
- [Ranked candidate pool](../reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v1/ranked_candidate_pool.json)
- [Read-only input validation](../reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v1/input_validation.json)
- [Length-500 batch smoke](../reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v1/batch_size_smoke.json)
- [Deterministic corruption cache manifest](../reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v1/corruption_cache_manifest.json)
- [Cache hash validation](../reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v1/cache_validation.json)
- [Development corrupted-input baseline](../reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v1/development_corrupted_baseline.json)
- [Exact update-0 resume manifest](../reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v1/resume_manifest.json)
- [Training and evaluation protocol](../reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v1/training_protocol.json)
- [Phase 4A config](../configs/e010_phase4a_supervised_generalization.yaml) · [preparation runner](../scripts/prepare_e010_phase4a.py)

All preparation artifacts are non-authorizing. No training or pilot work starts automatically.
