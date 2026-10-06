# E010 V13 matched 2000-iteration direct-space repeat

Reuse the exact committed V11 implementation, cached inputs, deterministic CPU
float64 setup, direct correction variables, constraints, Hessian-vector products,
callback, physical certificate, and shadow protocol. The sole numerical change is
`solver.maxiter: 1000 -> 2000`. No new solver implementation or model is introduced.
K remains 8 and the per-step physical radius remains 0.04 Angstrom.

Copy the committed V11 execution settings and replace only maxiter. Assert equality
of every other field, physical configuration and physical limit against the V11
contract. Pin all historical tracked files, exact V11 private state hashes, the
frozen input hashes, and new execution/report/test sources before running any
scientific example. Validate zero parity, baseline feasibility, derivatives and
HVPs using the unchanged V11 validation function for all 60 inputs. No new
synthetic solver run is needed: the mathematical implementation is unchanged.

Run all 60 examples exactly once from zero, with the historical four CPU workers
and one arithmetic thread per worker. The unchanged V11 attempt marker prevents
restart or outcome-based reruns. Serialize final variables and states before
optional reporting, exactly as V11 did. The physical callback may terminate a
solve before 2000 only upon satisfying the unchanged frozen contract. An exception
is recorded, never retried with different settings.

Verify the saved histories and physical screens through the V11 stopping point
match the historical V11 values exactly, excluding elapsed runtime. For the four
historically converged cases, also require identical final coordinates, metrics,
iterations and certificates. Do not rerun a scientific case to fix a mismatch.

After execution, independently reconstruct all metrics and physical certificates
from saved states without optimization. Aggregate every example, by condition and
all five length strata, including P0-P8 trajectories and correction directions.
Report convergence, shadows, iterations, stationarity, complementarity, dual
negativity, barriers, trust radii, safety and gains side by side with V11. Record
paired changes rather than requiring nonconverged endpoints to be identical.

Classification is frozen before execution. MATCH-A requires all 60 physical
certificates, condition-450 gain >=5%, no new inversions or per-example failures,
preserved assessability and all historical safety gates. MATCH-B is incomplete
convergence with a material improvement: at least 6 additional passes or at least
6 fewer material-shadow cases. This descriptive materiality repeats the V11
ten-percent-of-panel rule and does not alter a scientific acceptance threshold.
MATCH-C is below that improvement rule. Any implementation, input-integrity,
execution or reproduction failure is MATCH-D. Only MATCH-A certifies complete
K=8/s_max=0.04 feasibility.

No solver change, neural training, CUDA, enlarged radius, new objective,
convergence relaxation or follow-on experiment. Historical V11/V12 remain
immutable. Publish versioned implementation/protocol/tests and small textual
result records only; private NPZ states, datasets and checkpoints are not committed.
