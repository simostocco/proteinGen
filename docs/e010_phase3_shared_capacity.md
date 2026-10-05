# E010 Phase 3: shared-capacity experiment

This experiment is non-authorizing. Existing E010 Phase 1/2 artifacts were left unchanged. The two new variants used the exact Phase 2 v1 32-identity training panel and corruption archives, Phase 2 coordinate-only objective, deterministic uniform identity schedule, AdamW semantics, Phase 2 `_metrics` evaluator, and acceptance gates. No held-out data, priors, geometry losses, fixed bonds, identity embeddings, mixture-of-experts, sequence features, altered corruption, or sampling were introduced. No supervised pilot was prepared or run.

An initial Medium execution was not retained: its end-of-run exposure check detected that a shuffle-order bug could unbalance epoch exposures. The schedule was corrected to reshuffle before choosing the first identity of each epoch, and the retained Medium and Large runs both verify exactly 200 exposures per identity. The rejected attempt produced no saved training result or weights.

## CUDA smokes

Both length-500 forward/backward smokes had finite outputs and gradients and remained below the memory limits.

| Variant | Parameters | Peak allocated | Peak reserved | Status |
|---|---:|---:|---:|---|
| Medium (width 320, 4 blocks, 8 heads, 48 vector channels) | 5,070,128 | 506 MiB | 594 MiB | Passed |
| Large (width 416, 6 blocks, 8 heads, 64 vector channels) | 12,844,352 | 849 MiB | 996 MiB | Passed |

The configured ceilings were 6,144 MiB allocated and 7,680 MiB reserved.

## Training results and selection

Both models trained for 6,400 updates. Each identity was exposed exactly 200 times at the final boundary. Medium improved the Phase 2 mean RMSE but missed the median gate narrowly. Large passed every gate and is therefore selected by the predeclared rule as the smallest passing variant.

| Variant | Mean RMSE | Median RMSE | Maximum RMSE | RMSE/length slope | Finite | Gate result |
|---|---:|---:|---:|---:|---|---|
| Medium | 0.15885 Å | 0.15479 Å | 0.29271 Å | 0.0004035 Å/residue | Yes | Failed: median >0.15 Å |
| Large | 0.13556 Å | 0.13223 Å | 0.24775 Å | 0.0003606 Å/residue | Yes | Passed |

Phase 2 v2 at update 6,400 had mean/median/maximum RMSE 0.28534/0.25555/0.51292 Å and a 0.0007388 Å/residue slope. Relative to that mean, Medium improved by 0.12649 Å or 0.03912 Å per million added parameters; Large improved by 0.14978 Å or 0.01361 Å per million added parameters. This efficiency measure divides mean RMSE reduction by added parameter count over the 1,836,640-parameter Phase 2 model.

Mean RMSE by length stratum at update 6,400:

| Stratum | Medium | Large |
|---|---:|---:|
| 1–64 | 0.0745 Å | 0.0702 Å |
| 65–128 | 0.1226 Å | 0.0965 Å |
| 129–256 | 0.1498 Å | 0.1230 Å |
| 257–384 | 0.1830 Å | 0.1563 Å |
| 385–500 | 0.2505 Å | 0.2198 Å |

The complete per-identity trajectories, each length-stratum trajectory, local i+1/i+2/i+3 distance telemetry, chirality inversion counts, update-window gradient and optimizer-step norms, clipping fractions, runtime, and peak training memory are in the variant result JSONs. At update 6,400, runtime was 158.9 seconds for Medium and 251.7 seconds for Large. Peak training memory was 559,364,096 allocated / 878,706,688 reserved bytes for Medium and 960,722,432 / 1,417,674,752 bytes for Large.

## Adjudication

Large passes all gates, so it is the selected Phase 3 capacity model. This result allows preparation of a separate held-out supervised-refinement pilot under the stated rule; the pilot was not prepared or run automatically. The Phase 3 artifacts remain non-authorizing.

## Files

- [CUDA smoke results](../reports/experiments/E010_global_equivariant_expressivity/phase3_shared_capacity_v1/cuda_smokes.json)
- [Medium training results and trajectories](../reports/experiments/E010_global_equivariant_expressivity/phase3_shared_capacity_v1/medium_training_result.json)
- [Large training results and trajectories](../reports/experiments/E010_global_equivariant_expressivity/phase3_shared_capacity_v1/large_training_result.json)
- [Selection adjudication](../reports/experiments/E010_global_equivariant_expressivity/phase3_shared_capacity_v1/selection_adjudication.json)
- [Phase 3 config](../configs/e010_phase3_shared_capacity.yaml) · [runner](../scripts/run_e010_phase3_shared_capacity.py)
