# V16C float64 Jacobian-chain correction

V16A and V16B are immutable. A new `e010_sequential_teacher_v16c` lineage preserves all scientific equations, thresholds, solver settings, eligibility, K_max16 and s_max0.10 Å. No optimization or teacher panel is permitted during validation.

The one-step scale is `current.new_tensor(0.10)`. Eligibility stays Boolean; its arithmetic multiplier is explicitly converted to current.dtype/device. The quartet chain uses those float64 objects, eliminating Boolean-times-Python-float promotion to float32. NumPy ball inputs are explicitly converted to float64.

An operation-wide opt-in `Float64Trace` checks all floating Torch inputs/outputs, including nested local representation, signed quartet geometry, Kabsch, objective and backward operators. Integers and Boolean masks remain unchanged. The disabled tracer has no production cost.

Tracing additionally found PyTorch determinant-backward temporary 0/1 tensors constructed in the default floating dtype. Their values are exact, but V16C scopes the default to float64 during jac/cjac differentiation and restores it in finally. Teacher workers are isolated single-threaded processes; the scope changes neither package environment nor persistent default settings. Kabsch/SVD/determinant equations are unchanged. Independent derivative validation uses the same scoped precision policy.

Exact historical sinusoidal z and cosine directions at lengths12/32/500 are recovered using V16B hashes and stored comparisons. Check value parity before derivatives: coordinates, every term, q/turn/bonds, assessability/eligibility, safety classifications and every constraint. Compare full-Jacobian Jd, direct autograd JVP, and reverse scalar family contractions to unchanged tolerance1e-12+1e-10*abs(Jd).

Primary centered finite difference is fixed h=1e-4 in dimensionless z units, with unchanged tolerance1e-8+1e-5*abs(Jd). Diagnostic h=3e-5 and historical h=1e-6 cannot choose acceptance. Nine distinct length×phase checks are evaluated. To preserve the historical nine slots at each length (phase×old h), map all three old epsilon slots for a direction to the same new primary h, explicitly recording duplicate references rather than claiming independent repeats.

Both audit and focused-test runner install fail-fast optimizer guards. Old tests requiring real optimization are excluded; no solver execution is necessary to validate dtype, values or derivatives. Publish only after independent numerical reproduction and protected historical hash verification. No CUDA or neural training.
