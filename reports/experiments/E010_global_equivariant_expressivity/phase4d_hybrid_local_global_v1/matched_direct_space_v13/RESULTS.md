# E010 V13 matched 2000-iteration direct-space repeat

Classification: **MATCH-C**. Physical convergence **6/60**, V11 **4/60**.
Material-shadow cases **51**, V11 **55/60**. Complete feasibility certified: **False**.

| Condition | V11 local gain % | V13 local gain % | Aligned change % | Chiral change % | Convergence |
|---|---:|---:|---:|---:|---:|
| 50 | 19.155202 | 19.169373 | -4.328701 | -3.475220 | 3/20 |
| 250 | 8.358575 | 8.385916 | -0.149087 | -0.004633 | 1/20 |
| 450 | 8.486524 | 8.500692 | 0.137313 | -0.003925 | 2/20 |

The only settings change is maxiter=1000 to 2000. Original V11 solver, Hessian-vector products, initialization, inputs, constraints and V9B physical gates are unchanged.
Every example ran exactly once from zero. All metrics and certificates independently reproduced from private saved states without optimization. Historical solver-history prefixes match exactly.
No example is filtered; nonconverged endpoint metrics are descriptive and do not certify full-panel feasibility.
See result.json and B records for numerical distributions, paired comparisons, safety and trajectory telemetry.
No CUDA, neural training, changed objective, solver or geometric budget, or automatic follow-on.
