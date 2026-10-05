# E008 geometry-native decoder prototype

E008 is a single bounded decoder experiment after E007 Phase 3I.5. It freezes
the existing `v_only` update-500 coordinate generator and learns only a small
internal-coordinate decoder. The decoder represents periodic angle corrections
as normalized sine/cosine pairs and reconstructs the trace by differentiable
forward kinematics at 3.8 Å per adjacent Cα bond. No adjacent-distance loss is
used.

The decoder uses 72,004 parameters with the checked-in width-48, three-block
configuration. Its scalar inputs are centered local distances, angle and signed
pseudo-torsion sine/cosine values, a signed local triple product, length/position
conditioning, and per-residue RBF summaries of pair distances in four sequence
separation bands. These inputs are SE(3)-invariant; the output is aligned to the
coarse input by a proper Kabsch rotation. Padding must be a contiguous false
suffix and is returned as exact zeros. The model supports lengths 20–500.

Training examples are real sidecar Cα traces with zero-mean synthetic coordinate
noise. Noise scale is estimated from already-published generator output; the
training set does not launch diffusion sampling. The pilot first checks 32 real
structure round trips, overfits 32 fixed structures across five length strata,
then trains for 2,000 length-balanced decoder updates. Evaluation uses disjoint
development and prospective identity-30 clusters plus 100 newly sampled outputs
from the frozen update-500 prior (20 at each of 64, 128, 256, 384, 500).

## Automatic prototype manifest

The authorized E006 coordinate sidecars provide source paths and hashes, but do
not provide whole-structure conformation labels or experimental method. The
manifest builder reads only those authorized local sidecars and the configured
local processed metadata Parquet file pinned by the configured SHA-256. Missing
methods and labels remain `unknown`. `globular` and `idp` are retained only
when the local metadata row supplies that value, `label_source`, and
`label_authoritative=true`; no remote lookup or coordinate-based biological
label inference is performed. NMR is represented
only in `experimental_method`.

`selection_class=prototype_structured` is an independent decoder-feasibility
screen: sequence length 20–500, complete C-alpha mask, continuous chain, all
adjacent distances in [3.6, 4.0] Å, mean radial distance divided by sqrt(length)
at most 2.5 Å, and long-range (sequence separation at least 8) sub-8 Å contact
fraction at least 0.005. The same criteria apply to NMR, X-ray and cryo-EM.
Known IDPs, ambiguous labels, unresolved traces, chain breaks, and geometry
failures are excluded with reasons. This selection is not a definitive
biological globular annotation. The deterministic CSV contains exactly
`sample_id,experimental_method,conformation_class,selection_class,label_source,source_path,source_sha256,exclusion_reason`;
its adjacent summary JSON publishes the rules, counts, exclusions and hashes.

Build the manifest without constructing or running a model:

```bash
python scripts/run_e008_geometry_native_decoder.py --config configs/e008_geometry_native_decoder.yaml --build-manifest
```

Inspect the exact selected identities, length strata, conformation classes,
experimental methods, and disjoint train/development/prospective split before
any pilot workload:

```bash
python scripts/run_e008_geometry_native_decoder.py --config configs/e008_geometry_native_decoder.yaml --validate
```

The optional `--structure-class-manifest` override accepts this same eight
column schema and requires provenance for authoritative conformation labels.

## Commands

### Tiny-overfit gate protocol correction (2026-09-28)

The initial E008 YAML omitted the tiny-overfit RMSE threshold while the runner
read `milestones.tiny_overfit_coordinate_rmse_angstrom_max`. The earlier E008
design did not specify this numeric criterion under another field name. Before
any adjudication, the corrected protocol therefore defines near-memorization as
mean aligned coordinate RMSE <= 0.5 Å over the same 32 fixed corrupted/clean
structures used for optimization. This is a geometry-scale criterion (well
below a Cα bond length and small relative to the configured 0.15–2.0 Å input
noise); it was set without using the observed run result. The original config
is intentionally retained as the schema-regression fixture. The corrected
fresh-run configuration is `configs/e008_geometry_native_decoder_restart_v2.yaml`.

The failed attempt executed 500 tiny-overfit optimizer updates and zero pilot
updates. Its final gate metric was computed in process but was not persisted:
the exception occurred before the update-500 checkpoint and there is no
result/report record. The architecture and tiny-overfit outcome remain
unadjudicated. The last atomic checkpoint is update 400, so it cannot resume the
completed 500-update boundary and must not be used to infer a result.

CPU contract and parameter-count preflight:

```bash
python scripts/run_e008_geometry_native_decoder.py --config configs/e008_geometry_native_decoder.yaml --plan-only
```

Longest-length CUDA forward/backward memory smoke (not run during implementation):

```bash
python scripts/run_e008_geometry_native_decoder.py --config configs/e008_geometry_native_decoder.yaml --cuda-smoke
```

Tiny-overfit milestone plus the bounded 2,000-update pilot and 100-output
evaluation (not run during implementation):

```bash
python scripts/run_e008_geometry_native_decoder.py --config configs/e008_geometry_native_decoder.yaml --pilot
```

The pilot consumes the generated manifest configured in the YAML:

```bash
python scripts/run_e008_geometry_native_decoder.py --config configs/e008_geometry_native_decoder.yaml --pilot
```

### Failure-mode diagnostic (non-authorizing)

The corrected-config run's 500-update record is preserved in
`decoder_pilot_v2`: checkpoint
`fa88f9a30f028545eb08be3f96e60efa85e4474a8cfe80f138abdae9d05e3cad`, metrics
`ff5e36473983d6b39e3ca6c2174ca4b3d2b656b81946bf63ca2977b217ac823d`, and
`tiny_overfit_result.json` with aligned RMSE 3.591038 Å (gate 0.5 Å, failed).
The pinned config, panel, run manifest and class manifest are identified in
`configs/e008_failure_diagnostic_v1.yaml`. Source evidence is unchanged.

Across the complete 500-row curve, means of the first and last 20 updates show
total loss 7.17→5.91; coordinate 4.71→4.09; i+2 0.348→0.290; i+3 0.513→0.439;
long-range pair 3.19→2.35; radius 0.968→0.215; and contact 0.0852→0.0826.
These losses improve noisily, especially radius and long-range terms. Internal
is effectively flat (0.212→0.222) and chirality flat/slightly worse
(0.0283→0.0340). None shows sustained divergence. Because updates sampled
different structures, this is not a per-structure convergence trace.

Diagnostic outputs go to
`reports/experiments/E008_geometry_native/failure_diagnostic_v1/`. The CLI
checks required config fields and pinned hashes before CUDA checks, keeps each
target/corruption fixed, freezes the generator, and writes independent
non-authorizing JSON results.

```bash
python scripts/diagnose_e008_failure.py --config configs/e008_failure_diagnostic_v1.yaml --plan-only
python scripts/diagnose_e008_failure.py --config configs/e008_failure_diagnostic_v1.yaml --evaluate-existing
python scripts/diagnose_e008_failure.py --config configs/e008_failure_diagnostic_v1.yaml --overfit-length-64
python scripts/diagnose_e008_failure.py --config configs/e008_failure_diagnostic_v1.yaml --overfit-length-500
```

The GPU diagnostics are intentionally not run during implementation. Estimated
runtime is 10–30 minutes per mode, hardware dependent.

Run and publish the overfit gate independently from the 2,000-update pilot:

```bash
python scripts/run_e008_geometry_native_decoder.py --config configs/e008_geometry_native_decoder_restart_v2.yaml --tiny-overfit-only
```

Only after `tiny_overfit_result.json` reports `passed`, resume from its
integrity-checked atomic checkpoint to execute the bounded pilot:

```bash
python scripts/run_e008_geometry_native_decoder.py --config configs/e008_geometry_native_decoder_restart_v2.yaml --resume
```

The configured runtime estimate is 3–6 GPU hours, based on the existing
160-output E007 sampling replication runtime and the decoder update workload.
The smoke reports actual maximum-length GPU memory and latency before the
bounded pilot is launched.

## Go/no-go

The pilot reports all declared local, global, diversity, chirality, displacement
and sampler-overhead measurements in its report. The decision is `go` only if
all checked criteria pass. If local geometry passes but global safeguards fail,
the next allowed decoder change is global pair/contact conditioning. A `no_go`
result ends decoder refinement and moves the research question to diffusion over
`[sin(theta), cos(theta), sin(phi), cos(phi)]`; it does not authorize further
Cartesian auxiliary-loss or sampler-unrolled work.

The unchanged numeric criteria are evaluated on the identity-disjoint prospective
`prototype_structured` feasibility cohort. The resulting decision makes no claim
that those structures carry authoritative globular labels.
