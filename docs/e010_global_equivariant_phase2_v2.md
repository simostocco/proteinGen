# E010 Phase 2 v2 continuation

This continuation is non-authorizing. Phase 2 v1 remains `failed_under_predeclared_2000_update_gate`; no v1 artifact was edited or relabeled.

## Exact resume and execution

The original update-2000 file was weights-only. The v2 runner replayed the original 2,000 updates with the original model, loss, AdamW settings, fixed pairs, and identity schedule. Replay reproduced every saved v1 boundary metric within 1e-6 (maximum delta 0) and matched every saved model tensor bit-for-bit. Only after that check did it serialize the complete AdamW state, constant-scheduler state (`none`), Python/NumPy/Torch CPU/CUDA RNG states, v1 schedule state, exposure counts, and loss history into the update-2000 checkpoint.

Before update 2001, the code reloaded that checkpoint and checked its file and embedded state hashes. It also checked the source checkpoint, panel protocol, all 32 corruption archives, and each target/corruption/mask byte hash. All four v2 checkpoints were independently rechecked after retrieval; each full optimizer/RNG state hash matches its manifest. The v2 run used the unchanged 1.84M-parameter model, AdamW at 3e-4 with no scheduler, and the unchanged coordinate-only objective. After update 2000, a deterministic least-exposed-first schedule with SHA-256 tie order balanced cumulative exposures exactly at the requested boundaries.

## Boundary results

| Update | Exposure min / mean / max | Mean / median / max aligned RMSE (Å) | RMSE-length slope (Å/residue) | Non-finite | Chirality inversions | Last-20% loss slope |
|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 62 / 62.5 / 64 | 0.526 / 0.461 / 1.023 | 0.0018042 | 0 | 570 | -9.352e-5 |
| 3200 | 100 / 100 / 100 | 0.530 / 0.532 / 0.868 | 0.0013598 | 0 | 479 | 7.923e-6 |
| 4800 | 150 / 150 / 150 | 0.350 / 0.371 / 0.525 | 0.0006958 | 0 | 229 | -2.184e-5 |
| 6400 | 200 / 200 / 200 | 0.285 / 0.256 / 0.513 | 0.0007388 | 0 | 154 | -1.052e-5 |

Per-identity trajectories, per-stratum metrics at every boundary, gradient and optimizer-step norms, clipping fractions, and i+1/i+2/i+3 RMSE are in `continuation_result.json`. Final mean RMSE by stratum (20–64, 65–128, 129–256, 257–384, 385–500) was 0.137, 0.192, 0.256, 0.422, and 0.405 Å. Final mean local-distance RMSE was 0.228/0.248/0.240 Å for i+1/i+2/i+3; chirality counts are descriptive.

Mean aligned RMSE by length stratum at each boundary (Å):

| Update | 20–64 | 65–128 | 129–256 | 257–384 | 385–500 |
|---:|---:|---:|---:|---:|---:|
| 2000 | 0.192 | 0.335 | 0.453 | 0.682 | 0.914 |
| 3200 | 0.261 | 0.380 | 0.493 | 0.674 | 0.803 |
| 4800 | 0.194 | 0.267 | 0.349 | 0.445 | 0.472 |
| 6400 | 0.137 | 0.192 | 0.256 | 0.422 | 0.405 |

At 6400, the mean, median, maximum, and length-slope gates still fail; finiteness passes. Errors declined from updates 3200 to 6400, so the record follows the continued-improvement branch. A linear projection from 150 to 200 exposures estimates the mean, median, and maximum RMSE gates at about 267, 246, and 254 total exposures (roughly 8,528, 7,861, and 8,115 total updates). The length-slope gate worsened over that interval, so the latest linear trend gives no finite exposure estimate for passing all gates. Treat these as simple extrapolations, not a guarantee.

Only the shortest stratum meets all three RMSE summary limits. Isolated length overfits were not run because the joint results continued improving rather than showing a material plateau. No geometry term, prior, sequence input, sampling, held-out data, or supervised pilot was introduced.

Here “no sequence input” means no amino-acid tokens or embeddings were added. The model’s pre-existing positional index features remain unchanged as part of the exact architecture.

## Artifacts

- Protocol/config: `configs/e010_global_equivariant_phase2_v2.yaml`
- Runner: `scripts/run_e010_phase2_v2.py`
- Exact replay and source checks: `reports/experiments/E010_global_equivariant_expressivity/phase2_v2_continuation/exact_replay_audit.json` and `input_audit.json`
- Full trajectories and diagnostics: `continuation_result.json`
- Checkpoint hashes and internal state hashes: `checkpoint_manifest.json`
- Adjudication and exposure projection: `continuation_adjudication.json`
