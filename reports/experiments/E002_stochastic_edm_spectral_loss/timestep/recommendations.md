# Timestep Diagnostic Recommendations

Weights evaluated: `ema`.
Terminal q(x_T) empirically close to N(0,I): `True`.
Terminal q(x_T) theoretically close to N(0,I): `True`.
Histogram L1 distance q(x_T) vs N(0,I): `0.011648068173796978`.
High-timestep model epsilon MSE mean: `6.84591e-05`.
High-timestep model x0 RMSE Angstrom mean: `9.76525`.

## High-Timestep Baseline Comparison

- `oracle` epsilon MSE at t=499: `8.49666e-17`
- `model` epsilon MSE at t=499: `4.02096e-11`
- `zero` epsilon MSE at t=499: `2.3362e-10`
- `noisy_input` epsilon MSE at t=499: `1.20353e-09`

## Decision Notes

- Continue the same training only if repeated diagnostics show high-timestep errors improving.
- Consider zero-terminal-SNR or schedule changes if q(x_T) is not close to N(0,I).
- Consider centered/standardized normalization if scale-only normalization leaves a large terminal shift.
- Consider v-prediction if epsilon-to-x0 amplification dominates high-timestep failures.
- Consider min-SNR loss weighting if timestep performance is strongly unbalanced.
- Treat x0 clipping/dynamic thresholding only as a stabilizer, not proof of valid generation.
