# E009 Bayesian geometry-prior refiner

The runnable vertical slice is separate from E008. It includes CPU streaming prior fitting, an artifact-gated length-500 CUDA smoke, and a fixed-corruption length-64 isolated overfit. The length-500 overfit, 32-structure overfit, and pilot are not implemented or exposed by the runner.

## Model and objective

The refiner is a seven-block, width-128 invariant-message/equivariant-vector graph network with 1,173,776 trainable parameters. It has sequence edges at offsets ±1/±2/±3 and up to 24 nearest spatial edges. Radial pair-distance features and normalized sequence offsets condition invariant edge messages. Coordinate corrections are sums of relative unit vectors; posterior scales are invariant scalars bounded to [0.03, 3.0] Å. Sampling is reparameterized as `X = μ(C) + σ(C) ⊙ ε`.

The learned prior has a three-component Gaussian mixture on Cα bond lengths, a four-component Gaussian mixture on `logit(angle/π)` with the exact logistic Jacobian, and a six-component von Mises mixture on signed pseudo-dihedrals. Deterministic streaming EM writes mixture weights, means, scales/concentrations, convergence history, training and development likelihoods, empirical quantiles, length/method counts, correlations, and source hashes. Geometry extraction starts a new fragment at every masked residue or chain boundary.

The configured objective is

`L = w_coord L_coord + w_geom L_geom + w_pair L_pair + w_contact L_contact + w_rg L_rg + w_chiral L_chiral + w_KL L_KL`.

`L_coord` is normalized diagonal-Gaussian coordinate NLL. `L_geom` is the mean of separate sampled bond, angle, and torsion prior NLLs. `L_pair` is a long-range Gaussian distance NLL; `L_contact` is Bernoulli NLL for the target 8 Å contact map; `L_rg` is Gaussian NLL on radius of gyration; `L_chiral` is a signed scalar-triple-product likelihood; `L_KL` is analytic KL to an isotropic unit Gaussian centered on the coarse input. No adjacent-distance target is fixed.

## Artifacts and status

Configured fresh publication locations:

- Prior: `reports/experiments/E009_bayesian_geometry_refiner/prior_v1/prior.json` and its `prior.sha256` sidecar.
- CUDA smoke: `reports/experiments/E009_bayesian_geometry_refiner/cuda_smoke_v1/smoke_result.json`.
- Fixed length-64 input: `reports/experiments/E009_bayesian_geometry_refiner/fixed_overfit_corruptions.npz`.
- Length-64 overfit: `reports/experiments/E009_bayesian_geometry_refiner/overfit_length_64_v1/result.json` and exact `resume.pt` state.

This paragraph originally described implementation-time status. Scientific stages have since run; completed artifacts take precedence over that historical status. In particular, `reports/experiments/E009_bayesian_geometry_refiner/objective_diagnostic_v1/paired_result.json` records the 1,000-update paired objective diagnostic on CPU (coordinate-only aligned RMSE approximately 0.5561 Å, comparison arm approximately 0.6212 Å). This bounded overfit does not establish held-out generalization. Prior fitting uses the 150,568 E008 training identities and 6,275 held-out development identities. Prospective identity values are never decoded from the split record or selected for payload reads. The fixed corruption picks the lexicographically first eligible length-64 training identity and verifies its source hash and regenerated tensor on reuse.

## Commands

Run stages in order and stop if a stage fails:

```bash
python scripts/run_e009_bayesian_refiner.py --config configs/e009_bayesian_refiner.yaml --fit-prior
python scripts/run_e009_bayesian_refiner.py --config configs/e009_bayesian_refiner.yaml --cuda-smoke
python scripts/run_e009_bayesian_refiner.py --config configs/e009_bayesian_refiner.yaml --overfit-length-64
```

An interrupted length-64 stage resumes only from an atomic, hash-compatible checkpoint:

```bash
python scripts/run_e009_bayesian_refiner.py --config configs/e009_bayesian_refiner.yaml --overfit-length-64 --resume
```

`--plan-only` validates pinned input hashes and the complete config, calculates the parameter count without constructing the model, does not initialize CUDA, and creates no output. The three stage handlers publish independently and reject prior/stage overwrite or incompatible partial state. The smoke records runtime, CUDA peak allocation/reservation, process peak RSS, equivariance, invariant scales, gradients, optimizer mutation, cleanup, and before/after source hashes. The overfit records metrics at updates 0, 10, 50, 100, 250, 500, and 1000, saves update-1000 resume state before gate adjudication, and stops on non-finite losses or gradients.
