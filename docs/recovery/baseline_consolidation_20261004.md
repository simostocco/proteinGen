# Research baseline consolidation — 2026-10-04–05

The starting branch was `main`, HEAD
`b9d08d1b6cc24bf203c9ac629c7fe29d8198ec32`, upstream `origin/main`, remote
`https://github.com/simostocco/proteinGen.git`. Original index: 375 additions and
four modifications (`docs/experiment_timeline.md`, `pyproject.toml`,
`src/protein_distance_diffusion/data/__init__.py`, and
`src/protein_distance_diffusion/models/embeddings.py`). Original unstaged change:
`reports/experiments/E004_symmetric_triangle_multiplication/ARCHITECTURE.md`.
Original untracked entries: `PLAN.md`, `environment/`, `external_models/`.

Original binary-capable staged/unstaged patches, status, complete path inventories
and SHA-256 checksums were preserved outside Git under
`/home/simostocco/proteingen_audit_artifacts/20261002/consolidation/`.
No stash, reset, history rewrite or scientific training was used.

## Change review and retained work

All 380 staged/unstaged file entries were inventoried with structured parsing,
module definitions, experiment configuration keys and per-file purpose/action.
The inventory is external, not a package dependency. Original grouping comprises
118 configs, 81 scripts, 83 source entries, 86 test entries, ten documentation
entries, pyproject, and the E004 documentation addendum. The implementation spans
sequence/geometry readiness and E005–E010 experiment contracts, data pipelines,
models, evaluators and lifecycle infrastructure. It is retained as one coherent
baseline rather than discarded or split across dependencies.

The existing dtype-aware embedding change and data-package exports are retained.
The pytest root import path and Ruff third-party model exclusion are retained.
The E004 architecture addendum records repairability/MDS limitations and the
Distance-AF dry-run interface; it is included without changing measured results.
The local relocation note is preserved as
`e007_phase3i2_relocation_record.md`. Environment specifications are historical
E007 Phase 4B locks, not claims about the current environment. Acquired third-party
weights/source remain local under ignored `external_models/`; no weights or raw
scientific diagnostics are added.

## Initial complete-suite failure ledger

The first complete run: 1,596 passed, five failed, 13 skipped; six multiprocessing
deprecation warnings; 903.78 seconds. The previous 67-test focused pass was not
used as a substitute.

| Test | Classification/root cause | Repair |
|---|---|---|
| `test_near_500_tiny_model_execution_is_bounded_and_masked` | Smoke consistency check omitted repeated float32 centering used by the actual v-target implementation | Match clean/noise centering operations exactly; retain tolerance |
| `test_leakage_refuses_forward_before_model_and_publishes_atomically` | Same smoke check failed before intended leakage guard was exercised | Same implementation repair; leakage assertion retained |
| `test_reconciled_v2_contract_is_accepted_and_read_only` | Test assumed a historical 115-update staging run had not published | Verify immutable preparation pins read-only, independent of completed lifecycle state |
| `test_plan_only_dispatch_does_not_enter_execution` | Test asserted a real staging directory must exist | Isolate paths in a temporary root and assert plan-only creates neither staging nor final |
| `test_fresh_state_saves_pinned_metadata_from_validated_real_contract` | Execution contract loader only resolved staging metadata after publication moved it to final | Resolve final versus staging explicitly and reject ambiguous coexistence |

Additional lint-discovered runtime fixes:

- Phase 4A v1 passed undefined `ckpt` to boundary publication; use the committed
  `latest.pt` path.
- Exact-batch CUDA smoke reporting used undefined `identities`; reporting now
  uses the validated identity/seed selection through a shared helper.
- Unused bindings/imports, ambiguous names, late-bound closure warnings, import
  ordering and formatting were repaired without changing objective coefficients
  or architecture. Intentional project imports following CLI path bootstrap have
  narrowly explained E402 annotations.

Initial lint: 1,208 findings in 34 files. Initial format: 36 files required
formatting. Two historical configs retain terminal blank lines because their
exact bytes are pinned; `.gitattributes` scopes the whitespace exception to those
files, without changing test assertions or recorded hashes. Removing E007's
blank line briefly caused four provenance tests to fail; restoring its original
bytes resolved all four: `test_plan_is_model_free_and_refuses_existing_output`,
`test_plan_reads_existing_staging_without_mutation`,
`test_pinned_normalization_hash_scale_and_production_contract`, and
`test_v2_plan_pins_v1_panels_and_preserves_completed_evidence`. Historical source byte pins are not rewritten to match
repaired/formatted code; fresh execution requires a reviewed contract. A subsequent
full run also exposed `test_historical_execution_and_repaired_reporting_hashes_share_runner_path`,
which incorrectly assumed the live runner would forever equal the historical
reporting-repair hash. Its exact reviewed source is now a non-executed `.py.txt`
fixture, verified against the unchanged historical SHA. A negative regression
confirms unreviewed runner bytes still fail validation. No expected hash was
updated to match current code.

## Scientific regressions and documentation

Publication now rejects identity collisions, inconsistent sequence/residue or
coordinate counts, and NaN/Inf coordinates before writing an archive. The worker
records explicit rejection reasons instead of silently overwriting distinct
identities. Existing v2 exclusion/physical-alias/reallocation tests remain intact.
New small tests cover malformed authorized source arrays and publication safety.

E010 tests explicitly exercise reflection-inclusive E(3) behavior. Additional
Phase 4B evaluator tests verify local i+1/i+2/i+3 errors against unaligned distances,
proper rotation/translation, reflected chirality and degenerate coverage.
Future metrics expose chirality eligible counts/assessability and block an
unassessable chirality gate. Existing zero-eligible numeric sentinel is retained
for compatibility and no longer suffices for authorization. Historical artifacts
retain their original interpretation; see the closeout's coverage caveat.

The incompatible historical training/development squared-error field is relabeled
and explained for future reports. This is a reporting correction, not a change
to training loss or historical results.

E000's finalized timestamp/count now follow its calibrated summary
(2026-08-31, heuristic 12/375). E009's completed objective diagnostic supersedes
its implementation-time “not run” statement. Phase 4A preparation status is
explicitly historical. E010 symmetry terminology and the concise
[Phase 4B forensic closeout](../e010_phase4b_forensic_closeout.md) preserve training,
data, gradient and finite-displacement evidence without committing raw audit data.

## Quality commands and portability limits

Established commands are `pytest -q`, `ruff check .`, and
`ruff format --check .` (README and pyproject). There is no configured type checker
or GitHub Actions workflow. Tests were run with `PYTHONDONTWRITEBYTECODE=1`,
`OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` and pytest's cache provider disabled;
this controls incidental writes and CPU concurrency, not assertions.

Some research tests require local ignored experiment/data artifacts; a source-only
checkout is not yet a self-contained reproduction bundle. Existing unavailable/opt-in integration checks remain explicitly skipped by their
original conditions. The environment snapshots and raw audit provenance distinguish
historical setup from currently installed package versions. Final validation and
pushed SHA are recorded in the external consolidation report and Git history.

No condition weights, structural objective or architecture were changed during
consolidation. Its planned next step was the predeclared frozen condition-weight
counterfactual, prepared only after the baseline was pushed and verified.

Subsequent closeout: both frozen counterfactuals are complete; CF1 was W2 and
CF2 was X4 under the pre-registered Cartesian-retention gate. Weight-only tuning
is closed and CF2 should not be trained. See the
[completed record](../e010_condition_weight_counterfactual.md). The next question
is a frozen local-auxiliary coefficient diagnostic with equal condition weights,
not another weighting or a training run. The audited baseline commit remains
`058ac3e747d9dc28d1a3e9d0f449627de24cf2be`.

## Final validation

Complete suite: **1,620 passed, 13 skipped, zero failures**, six existing multiprocessing
deprecation warnings, 952.84 seconds. `ruff check .` and `ruff format --check .`
passed (366 formatted files); all 351 Python files under src/scripts/tests parsed
without generating bytecode. Working/index whitespace checks passed. The 13 existing skips are 11 CUDA checks, one unavailable local mmCIF fixture,
and one opt-in live RCSB integration check.
GitHub reports zero configured Actions workflows. Both diagnostic checkpoint
SHA-256 values still match the closeout record. No scientific training was launched.
