# Sequence-Geometry Codesign Readiness Plan

## Current Contract

The current processed manifests and NPZ files store canonical one-letter
sequences. NPZ files also contain sequence tokens, residue IDs, residue masks,
C-alpha coordinates, distance matrices, and JSON metadata. The current schema
does not explicitly guarantee whether each sequence is ATOM-derived,
SEQRES-derived, or otherwise inferred. That provenance must be audited against
the raw structure before codesign model implementation.

`scripts/audit_sequence_data_readiness.py` is a read-only, staged audit. It
checks processed NPZ and manifest agreement, inventories raw mmCIF chain/model
sequence evidence, measures unknown and modified residue use, audits NMR model
consistency, summarizes exact duplicates, and checks train/validation overlap.
It writes only to a new report directory and never repairs source data.

## Staged Audit

The corpus contains 223,709 raw mmCIF files (about 72 GB), 506,919 merged
processed rows, 270,785 training rows, and 25,257 validation rows. Another
210,877 processed rows are outside train and validation. Because several rows
can refer to different chains or models from one source structure, raw parsing
is scheduled by unique `source_file`, not manifest row.

The audit has three explicit modes:

- `manifest-only` streams all Parquet rows and inspects corresponding NPZ
  metadata without opening a raw mmCIF. It establishes counts, sequence and
  matrix agreement, duplicates, policy distributions, excluded rows, and split
  leakage before any raw parsing is approved.
- `raw-pilot` deterministically selects unique sources with stratification over
  split membership, length, method, model, chain multiplicity, trimming,
  sequence alphabet, repeated source use, and known missing-C-alpha failures.
  Requested and achieved coverage are both reported because some strata may be
  sparse.
- `raw-full` enumerates the raw corpus only after the first two stages pass. It
  stores source path and SHA-256 identity in workflow state and supports
  `--resume` without reparsing completed unchanged inputs. Parsed structures
  are released after each source. The default `verbose-v3` profile retains the
  historical per-source JSON layout; production full audits use the compact
  layout below.

`--resume` and `--restart` are mutually exclusive. Input paths, required
manifest columns, state databases, and output-path separation are validated
before output creation or deletion. A changed source hash creates a new pending
source version. Partial protocols report completed, failed, and pending counts
plus the recovery instruction.

Each mode writes a protocol and separate manifest summary, alphabet,
duplicate/leakage, ambiguous-pairing, NPZ-mismatch, policy, and acceptance
outputs. Raw modes additionally write sequence provenance,
SEQRES/ATOM/matrix alignments, residue exceptions, NMR consistency, failures,
and stratum coverage. `seqres_sequence`, `atom_sequence`, and
`matrix_sequence` remain separate evidence fields; disagreement is diagnostic
and is not silently treated as corruption.

The gate order is strict: review the complete manifest-only report, then the
250-source pilot and its sparse/failure strata, and only then authorize the
deferred full raw audit. None of these audit stages migrates data or authorizes
model implementation.

### Manifest Analysis Performance And Recovery

The first real manifest-only attempt indexed all 802,961 rows successfully but
stalled before publishing `manifest_summary.json`. The original split-status
query materialized distinct train and validation IDs and then reported `SCAN t
LEFT-JOIN` and `SCAN v LEFT-JOIN` inside the processed-row scan. That plan is
effectively quadratic in processed and split rows. It was replaced by indexed
membership probes. Leakage is now computed by intersecting grouped unique keys,
including a composite PDB/chain/model index, rather than matching every pair of
member rows.

Analysis indexes cover manifest kind plus sample ID, matrix path, source file,
recomputed and stored sequence hashes, PDB ID, chain ID, model ID, cluster ID,
split-group ID, and the PDB/chain/model tuple. Index creation is an explicit
timed stage followed by `ANALYZE`.

Duplicate, leakage, and ambiguous-pairing files contain one row per diagnostic
key, never one row per matching pair. By default each table emits at most
10,000 groups and each group retains at most five deterministic sample-ID
examples. `--max-diagnostic-groups` and `--max-examples-per-group` configure
these bounds. Companion summary JSON files preserve total group/member counts
and truncation flags.

Every major stage is recorded in `stage_state` and announced on stdout. During
long SQL statements, `sequence_readiness_protocol.partial.json` receives a
heartbeat with the current stage, elapsed time, and SQLite virtual-machine step
count. NPZ metadata validation reads the distance-array header without
decompressing the matrix payload, checkpoints its last processed manifest row,
and resumes idempotently. Ctrl+C marks the active stage interrupted, preserves
SQLite state, and does not publish `sequence_readiness_protocol.json`.

Stored sequence-hash coverage is reported independently for processed, train,
and validation manifests. A missing `sequence_hash` column is `unavailable`,
not a mismatch. Available hashes use the versioned `sha256_utf8_v1` rule and
report checked, matched, and mismatched counts explicitly. Model statistics are
derived from the manifest's actual `model_number` column. A processed population
containing only model 1 describes retention policy, not raw-model availability:
the raw audit enumerates every `_atom_site.pdbx_PDB_model_num` present in each
selected NMR mmCIF.

Acceptance reporting is criterion-specific. Manifest-resolvable criteria can
pass or fail independently, while SEQRES/ATOM/matrix provenance, alternate
locations, modified residues, missing-residue alignment, and NMR cross-model
consistency remain `pending_raw_audit` until raw evidence is inspected.

### Compact Raw-Audit Storage

`--storage-profile compact-v1` publishes typed, dictionary-encoded,
Zstandard-compressed Parquet under `tables/<logical-table>/part-NNNNNN.parquet`.
The default `--sources-per-partition 1000` bounds the 223,709-source corpus at
224 partitions per logical table. It never creates a file or directory per
source. `source_identity` stores each path, SHA-256, and parse outcome once;
other tables use a stable `source_id`. Logical tables cover protein chain/model
evidence, matrix-pair alignments, physical candidates, residue-ID conventions,
NMR consistency, modified residues, missing residue/C-alpha outcomes,
nonpolymer summaries, blocking failures, and residue-token counts. Full residue
payloads are retained only where the scientific decision needs them.

`--state-dir` places `audit_state.sqlite` and its SQLite WAL/SHM files on Linux,
while `--output-dir` contains only published scientific artifacts. Both
resolved paths and the storage profile are pinned in `run_config.json` and the
completed protocol, so an incompatible cross-filesystem resume is refused.
Each partition is written to a temporary file on the artifact filesystem,
renamed, hashed, row-counted, and given an atomic commit marker before SQLite
records it. Resume validates every marker and Parquet file and reconciles a
partition finalized immediately before an interrupted SQLite commit. Missing,
corrupt, duplicate, overlapping, or incompatible source-range partitions stop
the run. Ctrl+C leaves an incomplete heartbeat protocol and reusable state.

`storage_projection.json` records pilot bytes by logical table, bytes per
source, projected corpus size, expected partition count, current free space,
and the safety decision. A compact `raw-full` run requires both a passing exact
v3 equivalence report and free space satisfying `free >= 2 * projected
remaining bytes + 20 GiB`; capacity is checked again at every partition. The
explicit `--unsafe-skip-disk-check` escape hatch is recorded prominently and
is not part of the approved commands.

`SequenceReadinessArtifactReader` exposes the same logical alignment and
candidate records from historical `verbose-v3` and new `compact-v1` audits.
Summary-only reporting and the `sequence_geometry_pairing_v1` builder use this
reader, so the completed v3 pilot remains readable without migration.

The first compact pilot produced correct scientific payloads and exact count
contracts, but its initial equivalence reader omitted `source_sha256` while
resolving chain/model `source_id` values and compared compact candidate aliases
and native Booleans directly with verbose-v3 fields and string Booleans. The
keyed read-only diagnosis in
`reports/compact_v1_existing_equivalence_after_fix.json` confirms zero logical
field, row-association, or primary-key differences after reader normalization.
The failed pilot and its state remain immutable historical artifacts.

### Raw Provenance Semantics

The first 250-source raw pilot exposed a parser defect: it grouped arbitrary
`_atom_site` residues by author chain without first restricting them to the
declared protein-polymer entity. Waters, ions, ligands, glycans, and nucleic
acids therefore entered protein unknown-token and missing-residue counts. The
v2 parser establishes protein entities through `_entity_poly`, maps label
chains through `_struct_asym`, and joins declared positions and coordinates by
`_pdbx_poly_seq_scheme.asym_id`, `_atom_site.label_asym_id`, entity ID, label
sequence ID, author sequence ID, and insertion code. Nonpolymer components are
reported separately and never contribute to sequence acceptance. The mapping
rule is versioned as `protein_distance_diffusion_default_v1`; it preserves the
preprocessing rule `MSE -> MET -> M`.

The audit reproduces preprocessing model-1 selection for matrix provenance,
terminal C-alpha trimming, and deterministic C-alpha alternate-location
selection (blank altloc first, then finite occupancy, altloc A, lexical altloc,
and source-row order; duplicate blank or duplicate named altlocs are
ambiguous). Raw NMR inspection still enumerates every coordinate model and
summarizes consistency once per source and label chain.

Matrix alignment is classified as `exact`, `valid_terminal_trim`,
`valid_residue_id_selection`, `internal_gap`, `ambiguous_subsequence`,
`sequence_mismatch`, or `unavailable`. Terminal trims require a contiguous
polymer interval plus matching retained label boundaries and trim counts.
Ordered author residue IDs and insertion codes resolve repeated sequence
substrings; string containment alone cannot. Where raw selected C-alpha
coordinates and the physical NPZ matrix are available, pairwise distances are
required to agree within `1e-4` Angstrom.

Raw-pilot criteria use `pilot_passed`, `pilot_failed`, or `informational` with
nonzero evidence counts. A bounded pilot always retains the separate
`corpus_wide_status: pending_full_raw_audit`. Coverage reports distinguish the
unique selected-source denominator from overlapping stratum memberships and
from deterministic fill sources; sums over achieved strata are not source
counts.

The interrupted `reports/sequence_data_readiness_manifest` index is complete
and can be resumed in place; it does not require rebuilding or a new output
directory:

```bash
PYTHONPATH=src python scripts/audit_sequence_data_readiness.py \
  --audit-mode manifest-only \
  --processed-manifest data/full/processed_recovery/merged_manifest.parquet \
  --train-manifest data/full/splits_recovered_all_structures/train.parquet \
  --validation-manifest data/full/splits_recovered_all_structures/validation.parquet \
  --output-dir reports/sequence_data_readiness_manifest \
  --resume
```

### Commands

Complete manifest-only audit:

```bash
PYTHONPATH=src python scripts/audit_sequence_data_readiness.py \
  --audit-mode manifest-only \
  --processed-manifest data/full/processed_recovery/merged_manifest.parquet \
  --train-manifest data/full/splits_recovered_all_structures/train.parquet \
  --validation-manifest data/full/splits_recovered_all_structures/validation.parquet \
  --output-dir reports/sequence_data_readiness_manifest
```

The original `reports/sequence_data_readiness_raw_pilot_250` run is preserved as
historical diagnostic evidence. Its partitions predate entity-aware polymer
selection and cannot be resumed into a corrected report. The corrected parser
must use a fresh versioned directory.

Deterministic 250-source raw pilot v2:

```bash
PYTHONPATH=src python scripts/audit_sequence_data_readiness.py \
  --audit-mode raw-pilot \
  --raw-dir data/full/raw/mmcif \
  --processed-manifest data/full/processed_recovery/merged_manifest.parquet \
  --train-manifest data/full/splits_recovered_all_structures/train.parquet \
  --validation-manifest data/full/splits_recovered_all_structures/validation.parquet \
  --preprocess-state-db data/full/processed/preprocess_state.sqlite \
  --preprocess-state-db data/full/processed_recovery/preprocess_state.sqlite \
  --max-source-files 250 \
  --samples-per-stratum 4 \
  --pilot-seed 4004 \
  --checkpoint-frequency 25 \
  --output-dir reports/sequence_data_readiness_raw_pilot_250_v2
```

Resume the same pilot after interruption:

```bash
PYTHONPATH=src python scripts/audit_sequence_data_readiness.py \
  --audit-mode raw-pilot \
  --raw-dir data/full/raw/mmcif \
  --processed-manifest data/full/processed_recovery/merged_manifest.parquet \
  --train-manifest data/full/splits_recovered_all_structures/train.parquet \
  --validation-manifest data/full/splits_recovered_all_structures/validation.parquet \
  --preprocess-state-db data/full/processed/preprocess_state.sqlite \
  --preprocess-state-db data/full/processed_recovery/preprocess_state.sqlite \
  --max-source-files 250 \
  --samples-per-stratum 4 \
  --pilot-seed 4004 \
  --checkpoint-frequency 25 \
  --output-dir reports/sequence_data_readiness_raw_pilot_250_v2 \
  --resume
```

The production full audit must use the guarded `compact-v1` command in
"Compact Equivalence And Deferred Production Run" below. Omitting
`--storage-profile` deliberately retains legacy `verbose-v3` behavior for
historical compatibility, but that one-file-per-source layout is unsuitable
for the 223,709-source corpus.

### Bounded Provenance Forensics

The v2 pilot's blocking rows require a separate bounded forensic pass before
acceptance semantics are revised. `scripts/diagnose_sequence_provenance_failures.py`
processes only those rows plus deterministic passing controls. It enumerates
all protein label-asym candidates linked to the requested author chain and
scores author residue IDs with insertion codes, label sequence IDs, zero-based
positions, and one-based positions independently. All candidate scores are
retained; minimum coordinate error alone never authorizes reassignment.

The tool verifies that the NPZ `distance_matrix` is the physical C-alpha matrix
reconstructed from its own stored coordinates before comparing raw
coordinates. It derives a serialization tolerance from known-good pilot rows,
checks raw size/mtime against both preprocessing state databases, and records
that historical preprocessing did not store raw SHA-256 values. Its SQLite
checkpoint is resumable, while Ctrl+C leaves only an incomplete protocol.
Pilot frequencies remain clustered observations from stratified sources, not
corpus prevalence estimates.

Run the bounded forensic analysis only in a new output directory:

```bash
PYTHONPATH=src python -u scripts/diagnose_sequence_provenance_failures.py \
  --audit-dir reports/sequence_data_readiness_raw_pilot_250_v2 \
  --preprocess-state-db data/full/processed/preprocess_state.sqlite \
  --preprocess-state-db data/full/processed_recovery/preprocess_state.sqlite \
  --expected-failure-count 197 \
  --passing-controls-per-method 1 \
  --output-dir reports/sequence_data_readiness_raw_pilot_250_v2_forensics
```

Resume an interrupted run by repeating the command with `--resume`.

### Corrected Pilot V3

The completed forensic pass established that 118 blocking rows came from the
audit's label-chain-first lookup, 16 from an audit residue-ID convention error,
and 12 from stale legacy trim metadata. The first 146 rows reconstruct their
stored distance matrices exactly. Historical source SHA-256 values were not
stored; the shared `state_size_mtime_match_sha_unavailable` result is therefore
a separate provenance qualifier, not a biological alignment classification.

Pilot v3 uses author-chain-first candidate enumeration, retains label/auth and
positional residue namespaces separately, and emits strict and practical
training eligibility. It preserves every competing candidate. A missing raw
C-alpha is missing evidence, while a failed coordinate reconstruction is
contradictory evidence. The four exact-coordinate cases `1EJ7_L`, `3B5K_A`,
`3B5K_B`, and `5OTY_A` remain provenance-incomplete but may be reported as
`conditionally_verified_pair`; they are not strict verified pairs.

The prior audit protocol pins the exact 250-source cohort, and the prior
forensic table pins the 197-row transition accounting. The resulting pilot
fractions are stratified diagnostic evidence and must not be reported as
corpus prevalence.

Prepare the corrected deterministic pilot in a fresh directory with:

```bash
conda activate proteingen
PYTHONPATH=src python -u scripts/audit_sequence_data_readiness.py \
  --audit-mode raw-pilot \
  --raw-dir data/full/raw/mmcif \
  --processed-manifest data/full/processed_recovery/merged_manifest.parquet \
  --train-manifest data/full/splits_recovered_all_structures/train.parquet \
  --validation-manifest data/full/splits_recovered_all_structures/validation.parquet \
  --preprocess-state-db data/full/processed/preprocess_state.sqlite \
  --preprocess-state-db data/full/processed_recovery/preprocess_state.sqlite \
  --max-source-files 250 \
  --samples-per-stratum 4 \
  --pilot-seed 4004 \
  --checkpoint-frequency 25 \
  --prior-audit-dir reports/sequence_data_readiness_raw_pilot_250_v2 \
  --prior-forensics-dir reports/sequence_data_readiness_raw_pilot_250_v2_forensics \
  --output-dir reports/sequence_data_readiness_raw_pilot_250_v3
```

This command has not been run. Its additional outputs are
`v2_forensic_v3_transition.csv`, `strict_training_eligibility.csv`,
`practical_training_eligibility.csv`, `unresolved_cases.csv`, and
`candidate_resolution_evidence.parquet`, alongside the corrected acceptance
and protocol JSON files.

After a completed v3 run, derived reporting can be regenerated without raw
parsing, NPZ reads, or manifest indexing:

```bash
conda activate proteingen
PYTHONPATH=src python -u scripts/audit_sequence_data_readiness.py \
  --summary-only \
  --output-dir reports/sequence_data_readiness_raw_pilot_250_v3
```

Summary-only mode verifies the completed 686-pair and 197-case transition
contracts before writing. It hashes the preserved alignment, candidate,
forensic, raw-provenance, residue-token, and NMR evidence tables before and
after regeneration. The report distinguishes failure of the unfiltered cohort
(11 unresolved pairs) from usability after deterministic filtering (675
eligible pairs). Zero strict archival pairs means historical raw SHA-256 values
were not retained; it does not mean that zero sequence-geometry pairs are
valid. These stratified 250-source results do not estimate corpus-wide
prevalence.

### Compact Equivalence And Deferred Production Run

Create the compact 250-source pilot from the exact v3 cohort. The Linux state
directory and log remain on the native filesystem; scientific artifacts are
published to the Windows-mounted audit directory in bounded partitions.

```bash
conda activate proteingen
mkdir -p logs
nohup env PYTHONPATH=src python -u scripts/audit_sequence_data_readiness.py \
  --audit-mode raw-pilot \
  --storage-profile compact-v1 \
  --state-dir reports/sequence_data_readiness_raw_pilot_250_compact_v1_v2_state \
  --sources-per-partition 1000 \
  --raw-dir data/full/raw/mmcif \
  --processed-manifest data/full/processed_recovery/merged_manifest.parquet \
  --train-manifest data/full/splits_recovered_all_structures/train.parquet \
  --validation-manifest data/full/splits_recovered_all_structures/validation.parquet \
  --preprocess-state-db data/full/processed/preprocess_state.sqlite \
  --preprocess-state-db data/full/processed_recovery/preprocess_state.sqlite \
  --max-source-files 250 \
  --samples-per-stratum 4 \
  --pilot-seed 4004 \
  --checkpoint-frequency 25 \
  --prior-audit-dir reports/sequence_data_readiness_raw_pilot_250_v2 \
  --prior-forensics-dir reports/sequence_data_readiness_raw_pilot_250_v2_forensics \
  --equivalence-reference-dir reports/sequence_data_readiness_raw_pilot_250_v3 \
  --output-dir "/mnt/d/Users/Simone Stocco/proteinGen_audits/sequence_data_readiness_raw_pilot_250_compact_v1_v2" \
  > logs/sequence_data_readiness_raw_pilot_250_compact_v1_v2.log 2>&1 &
```

Revalidate logical equivalence later without parsing raw structures or reading
NPZ files:

```bash
conda activate proteingen
PYTHONPATH=src python -u scripts/audit_sequence_data_readiness.py \
  --equivalence-only \
  --equivalence-reference-dir reports/sequence_data_readiness_raw_pilot_250_v3 \
  --output-dir "/mnt/d/Users/Simone Stocco/proteinGen_audits/sequence_data_readiness_raw_pilot_250_compact_v1" \
  --equivalence-report reports/compact_v1_existing_equivalence_after_fix.json
```

Only after `compact_equivalence_report.json` and `storage_projection.json` both
pass review, start the deferred full audit with:

```bash
conda activate proteingen
mkdir -p logs
nohup env PYTHONPATH=src python -u scripts/audit_sequence_data_readiness.py \
  --audit-mode raw-full \
  --storage-profile compact-v1 \
  --state-dir reports/sequence_data_readiness_raw_full_compact_v1_state \
  --sources-per-partition 1000 \
  --raw-dir data/full/raw/mmcif \
  --processed-manifest data/full/processed_recovery/merged_manifest.parquet \
  --train-manifest data/full/splits_recovered_all_structures/train.parquet \
  --validation-manifest data/full/splits_recovered_all_structures/validation.parquet \
  --preprocess-state-db data/full/processed/preprocess_state.sqlite \
  --preprocess-state-db data/full/processed_recovery/preprocess_state.sqlite \
  --checkpoint-frequency 1000 \
  --storage-projection "/mnt/d/Users/Simone Stocco/proteinGen_audits/sequence_data_readiness_raw_pilot_250_compact_v1_v2/storage_projection.json" \
  --compact-equivalence-report "/mnt/d/Users/Simone Stocco/proteinGen_audits/sequence_data_readiness_raw_pilot_250_compact_v1_v2/compact_equivalence_report.json" \
  --output-dir "/mnt/d/Users/Simone Stocco/proteinGen_audits/sequence_data_readiness_raw_full_compact_v1" \
  > logs/sequence_data_readiness_raw_full_compact_v1.log 2>&1 &
```

Resume either run by repeating its exact command with `--resume`. Do not use
`--restart` on a recoverable run; restart intentionally replaces both its
published output and its dedicated state directory.

## Proposed Sample Schema

The future versioned schema must include `sample_id`, `pdb_id`, `chain_id`,
`model_id`, `sequence`, `residue_tokens`, `residue_numbers`, `insertion_codes`,
`residue_mask`, `matrix_path`, `requested_length`, `actual_length`,
`sequence_source`, `source_structure_sha256`, `matrix_sha256`,
`sequence_sha256`, `parser_config_sha256`, and `schema_version`.

`sequence_source` is one of `ATOM-derived`, `SEQRES-derived`, or `unknown`.
`requested_length` is nullable for native structures and required for generated
evaluation cohorts. Residue arrays retain author numbering and insertion codes
while canonical ordering follows label sequence IDs.

Migration must write a new dataset root. It must compare every proposed record
with the current immutable manifest/NPZ pair, quarantine ambiguities, preserve
input/output hashes, and rebuild splits only after acceptance checks pass.

## Acceptance Gate

- Sequence length, residue arrays, mask, coordinate count, and both matrix
  dimensions agree exactly.
- Sequences contain only the 20 canonical one-letter amino acids after an
  explicit, versioned mapping. Initially only `MSE -> MET` is permitted.
- Internal missing C-alpha positions are rejected. Terminal trimming remains
  explicit and bounded by its configured policy.
- Alternate locations are selected deterministically and the selected altloc
  and occupancy are retained.
- Residue numbers and insertion codes remain separate fields.
- Every matrix maps unambiguously to one PDB entry, chain, model, and sequence.
- NMR uses the declared deterministic model and all models have a consistent
  residue sequence.
- Exact sequence, configured sequence-identity cluster, PDB ID, and sample ID
  have zero train/validation leakage.
- Unknown sequence provenance, unknown residues, duplicate identities,
  malformed masks, or missing provenance hashes fail acceptance.

## First Experiment

Train the first codesign model on real PDB sequence-geometry pairs, with
controlled corrupted-real geometry as a distinct conditioning regime. Keep
separate sequence and pair/geometry branches, connect them through recurrent
bidirectional feedback, and test an optional learned geometry gate. Conditioning
dropout must support sequence-only, gated, and forced-conditioning inference
from one controlled model family.

Use E004 N=64 and N=128 as primary generated-geometry cohorts. Include only
preregistered selected N=256 proposals as a stress cohort; defer N=384 and
N=500. The primary designability endpoint is unrestrained AlphaFold folding.
Distance-AF may be a secondary held-out-restraint experiment because fitting
supplied restraints is partly circular.

No model implementation or migration should begin until the audit and dataset
acceptance gate pass.

## Versioned Pairing Layer

`sequence_geometry_pairing_v1` is an immutable derived view over the existing
processed manifest, split manifests, NPZ samples, completed raw audit, and
normalization metadata. The builder writes `all_pairs.parquet`, policy-filtered
train and validation manifests, excluded rows with explicit reasons, compact
summaries, a strict 20-residue `PAD`/`MASK` vocabulary, schema and protocol
JSON, and an input-hash ledger. Residue identifiers, insertion codes, selected
alternate locations, and masks remain referenced from audit/NPZ evidence rather
than being duplicated as large Parquet strings. Every manifest row records
`sequence_geometry_pairing_v1` and the evidence and vocabulary versions.

The `strict` policy accepts only audit rows with complete archival provenance.
The practical policy accepts `verified_sequence_geometry_pair`,
`unique_sequence_pair_coordinate_unavailable`, and
`residue_identity_provenance_incomplete`; it deterministically excludes
unresolved candidates, demonstrated mismatches, and missing or non-unique
matrix associations. Missing historical source SHA evidence lowers archival
provenance strength but is not a sequence mismatch. `all_with_status` retains
every row for diagnosis while unresolved rows remain ineligible for training.
The corrected 250-source pilot supports 675 practical pairs and excludes 11
unresolved pairs from 686 total; it remains a bounded diagnostic and does not
estimate corpus-wide prevalence.

Matrix association and raw candidate enumeration are separate contracts.
Matrix association is unique only when one processed `sample_id` owns one
`matrix_path` and that path belongs to no other sample. The audit's historical
`raw_match_count` is retained as `raw_polymer_candidate_count`: it counts
enumerated raw polymer candidates and never controls matrix ownership.
`raw_physical_candidate_count` deduplicates candidate evidence by source,
model, entity, label chain, and author chain, while
`candidate_evidence_row_count` separately counts tested residue-ID
conventions. Multiple physical candidates are acceptable when exactly one
author-linked physical candidate is uniquely supported and no second
author-linked candidate has strong evidence.

Candidate admissibility is defined before evidence strength: only physical
candidates with normalized `author_chain_match=true` can represent a manifest
chain. Strong same-label or same-sequence candidates outside that domain remain
compact rejected-decoy evidence with `author_chain_mismatch`; their coordinate
disagreement does not contradict the selected pair. The manifest and protocol
therefore distinguish all physical candidates, author-linked candidates,
non-author-linked candidates, all-domain strong candidates, and strong
author-linked candidates. Selection fails closed only when the author-linked
domain is empty for an eligible pair or contains multiple strongly supported
physical candidates.

Pairing eligibility and derived-dataset membership are separate axes.
`pairing_eligible` records whether sequence and geometry evidence supports the
pair. `selected_by_eligibility_policy` records the requested strict/practical
policy decision. `eligible_for_training` additionally requires an original
`train` or `validation` assignment. A valid pair from the original excluded
split remains pairing-eligible but receives `original_split_excluded` and
`dataset_membership_status=excluded`; it is not relabeled as an unresolved
pair. Pairing failures retain their scientific reason, and an unresolved row
from the original excluded split carries both `original_split_excluded` and
`unresolved_ambiguity` in deterministic order. Consequently every row in
`excluded_pairs.parquet` has at least one explicit reason.

The corrected pilot contract is 675 pairing-eligible and 11 pairing-ineligible
rows. Original splits contain 422 train, 36 validation, and 228 excluded rows.
The derived practical dataset contains 415 eligible train rows, 35 eligible
validation rows, and a 236-row excluded union. The disjoint diagnostic tables
separate the 11 pairing-ineligible rows from the 225 pairing-eligible rows
excluded only by their original split. The existing
`data/derived/sequence_geometry_pairing_v1_pilot_preview` is retained as
historical evidence; corrected preview publication uses a new `_v2` directory.

Completed v3 candidate evidence stores Booleans as strings, so the pairing
builder parses only native Booleans, integer `0`/`1`, or exact case-insensitive
`true`/`false` and `1`/`0` strings. It rejects whitespace-padded or otherwise
unrecognized values. All nine candidate Boolean fields are required in this
schema. Future audit output writes them as non-null native Parquet Booleans and
adds `physical_candidate_id`, `evidence_row_index`, and
`residue_id_convention`. The legacy `candidate_index` identifies an evidence
row/convention and is never part of physical-candidate identity. Multiple
strong conventions for one physical candidate are supporting evidence, not
candidate ambiguity.

Eligibility reconciliation is field-aware. The transition table supplies
final statuses for its bounded 197-case cohort; practical and strict tables
define eligibility membership; the unresolved summary defines exclusion
membership and only the columns it actually contains. Alignment
`training_eligibility` is normalized to `practical_training_eligibility` for
comparison. An absent column, empty cell, `None`, pandas missing value, or NaN
means that source did not supply evidence; only two explicit non-null values
that disagree constitute a contradiction.

`SequenceGeometryDataset` supports sequence-only, real-geometry conditioned,
deterministic conditioning-dropout, and explicitly configured corrupted-real
inputs without modifying stored NPZ files. The null geometry is a zero matrix
with a false pair mask. Corruptions preserve symmetry, a zero diagonal, and
pair-mask exclusion. This gives later sequence-only, forced-conditioned,
learned-gate, and recurrent-feedback experiments one stable input contract.

Pilot-derived preview (diagnostic only):

Validate every audited pilot row without publishing a derived dataset:

```bash
PYTHONPATH=src python scripts/build_sequence_geometry_pairing_manifest.py \
  --processed-manifest data/full/processed_recovery/merged_manifest.parquet \
  --train-manifest data/full/splits_recovered_all_structures/train.parquet \
  --validation-manifest data/full/splits_recovered_all_structures/validation.parquet \
  --audit-dir reports/sequence_data_readiness_raw_pilot_250_v3 \
  --normalization-file data/full/processed_recovery/normalization_train.json \
  --eligibility-policy practical \
  --allow-pilot-evidence \
  --validate-only \
  --expected-total 686 \
  --expected-eligible 675 \
  --expected-excluded 11 \
  --validation-report \
    reports/sequence_data_readiness_raw_pilot_250_v3/pairing_builder_validation.json
```

After validation succeeds, build the pilot-derived preview:

```bash
PYTHONPATH=src python scripts/build_sequence_geometry_pairing_manifest.py \
  --processed-manifest data/full/processed_recovery/merged_manifest.parquet \
  --train-manifest data/full/splits_recovered_all_structures/train.parquet \
  --validation-manifest data/full/splits_recovered_all_structures/validation.parquet \
  --audit-dir reports/sequence_data_readiness_raw_pilot_250_v3 \
  --normalization-file data/full/processed_recovery/normalization_train.json \
  --eligibility-policy practical \
  --allow-pilot-evidence \
  --output-dir data/derived/sequence_geometry_pairing_v1_pilot_preview_v2
```

The deferred corpus-wide raw audit is the `raw-full` command in the Commands
section above. After that audit completes and its hashes and report semantics
pass validation, build the definitive manifest with:

```bash
PYTHONPATH=src python scripts/build_sequence_geometry_pairing_manifest.py \
  --processed-manifest data/full/processed_recovery/merged_manifest.parquet \
  --train-manifest data/full/splits_recovered_all_structures/train.parquet \
  --validation-manifest data/full/splits_recovered_all_structures/validation.parquet \
  --audit-dir reports/sequence_data_readiness_raw_full \
  --normalization-file data/full/processed_recovery/normalization_train.json \
  --eligibility-policy practical \
  --output-dir data/derived/sequence_geometry_pairing_v1
```

### Bounded full-corpus pairing

The original pairing implementation is suitable for pilots but not for the
506,919-pair corpus. It materializes all three manifests, all audit alignments,
and the multi-million-row candidate-evidence table as pandas frames. It then
copies candidate and eligibility groups into dictionaries, constructs one
Python dictionary per output row, concatenates all per-sample frames, and
creates several filtered output-frame copies. Object-backed strings and merge
temporaries make peak memory a multiple of the serialized input size; this is
the source of the WSL termination seen before a validation report could be
written.

Completed `raw-full` compact-v1 audits use a bounded path. PyArrow scanners
read at most 4,096 input rows per batch, essential records and indexes live in
a WAL-protected SQLite state database, and candidate evidence is materialized
only for the current sample batch. Matrix files are SHA-256 hashed sequentially
without loading distance arrays. Validation failures retain aggregate counts
and at most 100 representative sample IDs per reason. Heartbeats record stage,
progress, elapsed time, current RSS, and peak RSS; the default 4,096 MiB ceiling
stops after a durable checkpoint and can be resumed with `--resume`.

Publication uses the same state and writes zstd-compressed Parquet partitions
atomically for `all_pairs`, eligible train and validation pairs, excluded
pairs, pairing-ineligible pairs, and pairing-eligible rows excluded by the
original split. Completed partitions are hash- and row-count-validated on
resume and before final directory publication. No full output list or frame is
created. The definitive 506,919/501,797/5,122 total/eligible/excluded contract
is enforced from the verified audit attestation and remains available as
explicit validation expectations.

Run the full validation with a local Linux state directory:

```bash
PYTHONPATH=src python -u scripts/build_sequence_geometry_pairing_manifest.py \
  --processed-manifest data/full/processed_recovery/merged_manifest.parquet \
  --train-manifest data/full/splits_recovered_all_structures/train.parquet \
  --validation-manifest data/full/splits_recovered_all_structures/validation.parquet \
  --audit-dir "/mnt/d/Users/Simone Stocco/proteinGen_audits/sequence_data_readiness_raw_full_compact_v1" \
  --normalization-file data/full/processed_recovery/normalization_train.json \
  --eligibility-policy practical \
  --validate-only \
  --expected-total 506919 \
  --expected-eligible 501797 \
  --expected-excluded 5122 \
  --validation-report reports/sequence_geometry_pairing_v1_full_validation_v2.json \
  --state-dir reports/sequence_geometry_pairing_v1_full_validation_v2_state \
  --batch-size 4096 \
  --max-memory-mib 4096 \
  --checkpoint-frequency 1000 \
  --maximum-failure-examples 100
```

After an interruption, repeat the same command with `--resume`. The validator
does not repeat completed input batches or matrix hashes. A definitive build
uses a separate new state directory and adds `--output-dir`; it must be started
only after the validation report passes.
