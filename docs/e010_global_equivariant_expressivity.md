# E010 global equivariant expressivity diagnostic

E010 is an isolated, bounded and non-authorizing diagnostic. It reads the E009 v4 length-64 corruption cache without calling E009's cache creation or run handlers. The config pins the cache bytes, target/coarse/mask tensor bytes, sample ID, and the source file containing E009's exact aligned-RMSE evaluator. Validation fails closed if any pin differs. No E009 artifact is written.

The versioned protocol is at `reports/experiments/E010_global_equivariant_expressivity/v1/protocol.json`. E010 writes `oracle_result.json`; it writes `model_result.json` only after the oracle gate passes; and it writes `report.json` after oracle failure or after model adjudication. Outputs have `authorizes_downstream: false`. No checkpoint or resume path is implemented because each stage is at most 1000 updates and publishes only a completed atomic result.

## Symmetry and subsequent status

The implemented relative-vector/scalar architecture is O(3)-equivariant and, including translation, E(3)-equivariant: reflections are included. Proper SO(3)/SE(3) transforms are a subset, not a handedness guarantee. There is no orientation-sensitive pseudoscalar or handedness-preference feature. Ordinary pairwise distances are reflection-invariant. These limitations are distinct from correct reflection detection by the evaluator. The later real-domain adaptation and diagnostics are summarized in [the Phase 4B closeout](e010_phase4b_forensic_closeout.md).

## Predeclared stages and interpretation

Stage 1 learns a single masked Cartesian residual of shape `[1,N,3]` directly in the target frame. It minimizes masked coordinate MSE only. It has no neural network, geometry prior, fixed bond lengths, posterior sample, or learned prior. Evaluations occur at updates 0, 1, 10, 50, 100, 250, 500 and 1000. Aligned RMSE <= 0.01 Å passes. A failure publishes `evaluation_or_optimization_pipeline_failure` and prevents model-overfit execution.

Stage 2 is a fresh 1.8M-parameter global positional equivariant residual model. Scalar node states use normalized sequence index, N- and C-terminal distances, and eight sinusoidal Fourier frequencies. Four blocks use global masked all-residue attention with pair-distance and sequence-separation logits. Vector states update through scalar gates applied to normalized relative coordinate vectors. Scalar and vector residual connections and equivariant normalizations preserve rigid-transform behavior. The output is `x_corrupt + masked_delta`; adjacent distances are unconstrained. The only objective is normalized coordinate MSE plus a 1e-5 residual L2 term. No fitted geometry prior or geometry loss is used.

The model stage uses AdamW at 3e-4, no weight decay, and clips the normalized total gradient only above a predeclared L2 norm of 5.0. It evaluates at updates 0, 10, 50, 100, 250, 500 and 1000, with early success at aligned RMSE <= 0.10 Å. It records aligned and unaligned errors, per-residue errors and index slope, descriptive i+1/i+2/i+3 distance errors and chirality inversions, finite status, gradient and optimizer-step norms, clip activation, runtime, peak memory, parameter count, and rigid-transform equivariance error.

The report follows the fixed interpretation table: oracle fail means pipeline/evaluator/optimization failure; oracle pass with model fail means remaining representation/architecture capacity failure; both pass means E009's local message-passing representation/objective were inadequate; model pass with poor descriptive geometry means a learned prior may later be considered only as a weak regularizer.

## Commands

Run the first three commands in order before either scientific stage. CUDA smoke is bounded to one length-64 optimizer step and checks finite loss/gradients, the parameter-count range, and maximum rigid-transform error <= 1e-4 Å. Execute the oracle only when separately authorized; execute model-overfit only after the oracle result passes. The implementation itself does not run either scientific stage.

```bash
PYTHONPATH=src:. python scripts/run_e010_global_equivariant.py --config configs/e010_global_equivariant_expressivity_v1.yaml --plan-only
PYTHONPATH=src:. python scripts/run_e010_global_equivariant.py --config configs/e010_global_equivariant_expressivity_v1.yaml --validate-only
PYTHONPATH=src:. python scripts/run_e010_global_equivariant.py --config configs/e010_global_equivariant_expressivity_v1.yaml --cuda-smoke
PYTHONPATH=src:. python scripts/run_e010_global_equivariant.py --config configs/e010_global_equivariant_expressivity_v1.yaml --oracle
PYTHONPATH=src:. python scripts/run_e010_global_equivariant.py --config configs/e010_global_equivariant_expressivity_v1.yaml --model-overfit
```
