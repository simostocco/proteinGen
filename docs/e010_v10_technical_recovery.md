# V10 authorized technical recovery

The original failure record at f90ea80e57bd1239d5d21049bbd11df8b512be60 is
immutable. Apply only its validated one-line empty-active-set telemetry guard.
This changes neither physical quantities nor optimization: an empty set of
linearized constraints has zero reported cone violation.

Exactly these missing panel indices may be replayed, once each, from zero:
0, 5, 15, 16, 17, 18, 24, 29, 30, 32, 37, 47, 51, 53, 55, 56, 59.
No optimizer is invoked for any of the 43 saved examples. Saved-state control
checks use indices 8, 9, 13, 42 and compare original/patched certificates,
coordinate hashes, objective and metrics. Iteration/status metadata and its
file hash must remain unchanged; this check does not rerun their solvers.

Load the original V10 configuration and committed V9B physical contract
directly, without modification. Keep K8/.04 Å, local-only final objective,
float64 CPU, exact constraints, Hessian products, zero initialization,
trust-constr settings and iteration cap, eligibility, inputs and shadow steps.
Each replay starts with a persistent once-only invocation marker. No retries.

All records go into `strict_scientific_v10/technical_recovery`, with explicit
per-state provenance. The 43 JSON records are byte-identical copies; their
original coordinate archives are only read. The 17 new states are labelled
**technical recovery replay states**, never serialized originals. Persist
variables, multipliers and every coordinate state with hashes. Coordinate
archives remain untracked.

Independently reconstruct all 60 trajectories and certificates and aggregate
using the unchanged V10 reporting/gating implementation. Do not overwrite the
original partial result. Retain V10-D whenever the frozen physical contract
does not support scientific adjudication. Stop after publication; no neural
training, CUDA or subsequent intervention.
