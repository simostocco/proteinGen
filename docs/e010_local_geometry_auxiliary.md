# E010 frozen local-geometry auxiliary diagnostic

Reviewed 2026-10-05. Classification: **L1 — clean auxiliary resolution on the
prescribed frozen panels**. This is not yet evidence of sustained training benefit.
Condition-weight tuning remains closed.

The source baseline was `938fb5ab338b878e1e4a4647437262c7ebc57148`.
The Phase 4B update-1092 checkpoint SHA256 is
`f5211cbc1be5092175ce15b9761a4efba761d242310287cd6b1e04df6a6744ef`.
The same cached five training and five development identities and conditions
50/250/450 were reused; development did not influence calibration or selection.
See [the condition-weight record](e010_condition_weight_counterfactual.md) for
these identities and the preserved sample-manifest hash.

The only variable was lambda in `L_cart + lambda * L_local`, where Cartesian
loss remains equal-protein masked xyz-component MSE plus `1e-5` residual-component
MSE, without alignment. Local loss is the arithmetic mean of endpoint-masked
distance-error MSEs at offsets 1/2/3, independently normalized per protein.
Historical equal condition weights were fixed throughout.

Training-only calibration gave `s = ||g_cart||/||g_local|| = 0.13438815304368323`.
The pre-registered kappa values were 0, 0.25, 0.50, 0.75, 1.00 and lambda=kappa*s.
The smallest nonzero candidate passing Cartesian descent, >=80% Cartesian
retention and non-harmful mean-local derivative was **kappa=0.50,
lambda=0.06719407652184162**. Selection was locked before development evaluation.

| Quantity | Cartesian control | Selected auxiliary |
|---|---:|---:|
| Training Cartesian derivative | -0.260010 | -0.306763 |
| Cartesian retention | 100% | 117.98% |
| Training mean-local derivative | +0.136074 | -0.147686 |
| Development mean-local derivative | +0.137008 | -0.140530 |
| Training finite alpha=0.01 Cartesian delta, Å² | -0.002595886 | -0.003062900 |
| Training finite local MSE delta, Å² | +0.001361934 | -0.001476812 |
| Development finite Cartesian delta, Å² | -0.001751415 | -0.001968916 |
| Development finite local MSE delta, Å² | +0.001369079 | -0.001408219 |

All three aggregate local-offset derivatives were negative on both panels.
The selected common Adam direction improved local metrics in all three
conditions, but condition-50 Cartesian error still increased. The two shortest
identities in each panel still had predicted local harm. These panels cannot
establish a population-level length law.

`blocks.0` Adam local harm decreased but remained positive; `blocks.3` provided
stronger protective cancellation; `blocks.4` changed to beneficial. The selected
total raw-gradient/local cosine remained negative (-0.267697). Resolution is
specific to the saved-history Adam direction, not a claim that the original raw
objectives are aligned. Alpha=0.01 independent-copy checks confirmed the changes.
No optimizer step, training, architecture change or source checkpoint mutation
occurred. Full raw artifacts remain outside Git in the preserved audit collection.

The next experiment is one short matched Cartesian-control versus fixed-lambda
auxiliary continuation, from identical model and Adam state. Training must test
the full fixed evaluation population; the frozen five-protein panels are not
the primary endpoint. Chirality-aware representation remains a separate track.
