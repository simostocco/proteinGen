# E010 Phase 2

## v1 closeout

The immutable v1 record at `reports/experiments/E010_global_equivariant_expressivity/v1/` contains a passing free-coordinate oracle and a passing global-model expressivity test. The model reached its predeclared aligned-RMSE gate at update 100. The result closes as `passed_single_structure_expressivity`; it does not establish held-out generalization. Its geometry telemetry remains descriptive.

## Phase 2 protocol

Phase 2 reuses the E010 global Cartesian equivariant model and coordinate-only objective. It measures forward/backward CUDA time and peak allocated/reserved memory at lengths 64, 128, 256, 384, and 500, then jointly trains on 32 fixed corruption/target pairs sampled from the existing train split. The panel contains 6, 6, 7, 6, and 7 examples in strata 1–64, 65–128, 129–256, 257–384, and 385–500. Selection is deterministic by SHA-256 rank. Prior development, prospective, validation, test, and holdout panel identities, plus the E010 v1 identity, are excluded.

Each corruption follows the exact E009 fixed-corruption construction: calibrated independent noise centered by coordinate axis plus a centered cumulative drift normalized to 0.12 sigma RMS. Targets and corruptions are centered and hash-pinned. Phase 2 uses no fixed bond constraints, E009 geometry prior, local geometry losses, sampling, or prospective data.

Training is single-example joint sampling across the 32 pairs with AdamW at 3e-4, zero weight decay, 1e-5 residual L2, normalized coordinate MSE, and gradient clipping above L2 norm 5. Evaluation boundaries are updates 0, 100, 250, 500, 1000, 1500, and 2000. The final fixed-boundary acceptance criteria are mean aligned RMSE <= 0.20 Å, median <= 0.15 Å, max <= 0.50 Å, zero non-finite predictions, and length-regression slope <= 0.0005 Å/residue. Chirality inversions and i+1/i+2/i+3 local-distance RMSE are descriptive only. These are training-panel fit results, not held-out evidence.

If Phase 2 passes, a separate supervised pilot may be prepared. Its inputs must be frozen E007 denoiser predictions from corrupted real structures, paired with their known real targets. The pilot is not automatically executed. No Bayesian prior is reintroduced until a coordinate model shows held-out improvement; any later fitted prior is restricted to a weak-regularizer comparison against a coordinate-only control.

## Results

The 32 fixed pairs and protocol are in `reports/experiments/E010_global_equivariant_expressivity/phase2_v1_prepared/`. The panel is train-only and has no overlap with the recorded prior development/prospective/validation/test/holdout identities. All five v1 artifact hashes were rechecked after Phase 2 and are recorded in `phase2_adjudication.json`.

Forward/backward lifecycle measurements ran on an NVIDIA RTX A4000, batch size 1, after two warmups, over five measured updates per length:

| Length | Forward ms | Backward ms | Combined ms | Peak allocated MiB | Peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 64 | 6.80 | 9.53 | 16.33 | 32.37 | 38.00 |
| 128 | 6.69 | 10.09 | 16.78 | 63.04 | 74.00 |
| 256 | 6.91 | 8.82 | 15.73 | 126.64 | 154.00 |
| 384 | 7.73 | 9.13 | 16.86 | 229.94 | 258.00 |
| 500 | 11.37 | 12.88 | 24.25 | 355.08 | 428.00 |

Joint training ran for all 2,000 updates with the unchanged 1.84M-parameter global Cartesian model and coordinate-only objective. Per-structure results at each fixed boundary are in `training_result.json`. At the final boundary:

- Mean / median / maximum aligned RMSE: 0.526 / 0.461 / 1.023 Å (limits: 0.20 / 0.15 / 0.50 Å).
- Non-finite predictions: 0 (limit: 0).
- Linear error-vs-length slope: 0.001804 Å/residue (limit: 0.0005 Å/residue).
- Descriptive geometry telemetry: 570 total chirality inversions; mean i+1/i+2/i+3 distance RMSE 0.344/0.366/0.385 Å.

Phase 2 therefore failed four of five acceptance gates. This measures fit on the training panel and is not held-out improvement evidence. The conditional supervised pilot was not prepared or executed, and no Bayesian prior was introduced. v1 remains closed as `passed_single_structure_expressivity`, with its passing artifact files unchanged.
