# E010 V14 SLSQP benchmark

**SQP-E — solver resource/scaling failure.**
The scientific panel was not launched. No scientific feasibility interpretation is made.

SLSQP uses dense QP storage. The frozen length-500 resource preflight requires 11.088218 GiB for its installed mandatory workspace and constraint normals alone. The preregistered allocation budget is 8.302388 GiB, after reserving RAM for the framework/OS. The resource gate refused the dense allocation.
Full length-500 geometry, constraints and derivative plumbing were validated without reducing K or length.

| Preflight length | Physical convergence | SLSQP success | Iterations | Wall seconds | Peak RSS GiB |
|---|---|---|---:|---:|---:|
| 12 | False | True | 144 | 2.427 | 0.682 |
| 32 | False | True | 313 | 14.247 | 0.710 |

Fixed SLSQP configuration: CPU float64, one arithmetic thread, maxiter=2000, ftol=1e-12, analytic objective and vector constraint Jacobians. No additional constraint scaling. SLSQP does not use the historical exact Hessian-vector products. Its success flag is telemetry only.
All protected hashes and independently reconstructed preflight certificates passed. Historical V11/V12/V13 remain unchanged. No CUDA, neural training or follow-on solver was launched.
