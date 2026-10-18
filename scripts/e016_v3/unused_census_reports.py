"""Compact census reporting, not a new training data view."""
import json,collections,subprocess,shutil,hashlib
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
D=Path('/mnt/d/Simone/proteinGen_data');R=D/'e016_unused_train_census_20261018_v1';P=D/'pdb_supervised_split_50id80cov_20261016_v1'
def load(n):return json.load(open(R/n))
def save(n,x):(R/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def md(n,x):(R/n).write_text(x.rstrip()+'\n')
c=load('UNUSED_SEQUENCE_CLASSIFICATION.json');cov=load('STRUCTURAL_OBSERVATION_COVERAGE.json');mech=load('IDENTITY_EXCLUSION_MECHANISMS.json');comp=load('UNUSED_VS_JOINT_POPULATION_COMPARISON.json');ready=load('POTENTIALLY_RECOVERABLE_GEOMETRY.json');raw=load('BOUNDED_RAW_SOURCE_CONFIRMATION.json');integ=load('CENSUS_SOURCE_INTEGRITY.json');run=load('CENSUS_EXECUTION.json')
cats=[('A_NO_LINKED_OBSERVATION','No linked observation'),('B_NO_OBSERVED_CA','Linked observations, no observed Cα'),('C_PARTIAL_CA_QUALITY_OR_CONTINUITY_FAILURE','Mapped/conformer-usable partial Cα; quality or continuity admission fails'),('D_MAPPING_OR_CONFORMER_PROBLEM','No observation passes mapping/conformer checks'),('E_COMPLETE_CA_OTHER_QUALIFICATION_FAILURE','Mapped/conformer-usable complete Cα; other complete-view requirements fail'),('F_POLICY_READY_GEOMETRY_OMITTED','Verified local geometry and accepted quality; skipped by complete-Cα view predicate'),('G_OTHER_UNRESOLVED','Other / cause unresolved')]
for k,label in cats:c['primary_categories'].setdefault(k,0)
save('UNUSED_SEQUENCE_CLASSIFICATION.json',c)
ids=pq.read_table(R/'UNUSED_IDENTITY_CLASSIFICATION.parquet').to_pylist();dec={r['sequence_id']:r for r in pq.read_table(R/'UNUSED_IDENTITY_ADMISSION_DECISIONS.parquet').to_pylist()}
for r in ids:r['admission_mechanism']=dec[r['sequence_id']]['mechanism'];r['admission_mechanism_observation_id']=dec[r['sequence_id']]['observation_id']
pq.write_table(pa.Table.from_pylist(ids),R/'UNUSED_IDENTITY_CLASSIFICATION.parquet',compression='zstd')
obs=pq.read_table(R/'UNUSED_OBSERVATION_EVIDENCE.parquet').to_pylist();allCA=sum(r['max_coverage']==1 for r in ids);target_obs=[r for r in obs if r['trustworthy'] and r['valid_d']];qualityfail=[r for r in target_obs if not r['quality']]
ready['quality_excluded_but_mapped_verified_geometry']=dict(identities=len({r['sequence_hash'] for r in qualityfail}),observations=len(qualityfail),targets={k:sum(r[k] for r in qualityfail) for k in ['valid_d','valid_a','valid_t']},warning='NOT immediately admissible: requires an explicit quality-policy decision, not missing-coordinate completion.')
ready['policy_ready_means']='Passes mapping/conformer, existing method-resolution and valid-local-target checks; still excluded by the actual partial builder complete_CA skip. Admission requires an explicit future view-definition correction.'
save('POTENTIALLY_RECOVERABLE_GEOMETRY.json',ready)
aa=load('SEQUENCE_ONLY_TRAINING_ELIGIBILITY.json');aa['forward_smoke_status']=load('AA_ONLY_FORWARD_SMOKE_TEST.json')['status'];aa['forward_smoke_cases']=25;aa['AA_ready_is_data_eligibility_not_current_J1_sampler_admission']=True;aa['eligible_targets_by_relative_position_decile']={str(k):sum(sum(min(9,int(10*t/r['length']))==k for t in range(r['length'])) for r in ids) for k in range(10)};save('SEQUENCE_ONLY_TRAINING_ELIGIBILITY.json',aa)
# Bin counts are observation-weighted only when explicitly labelled.
cov['verified_target_count_bins_identity']={f'{lo}-{hi}':sum(lo<=r['verified_distance_targets']<=hi for r in ids) for lo,hi in [(0,0),(1,49),(50,199),(200,999),(1000,10**12)]};cov['geometry_targets_by_length_band']={f'{lo}-{hi}':{k:sum(r[k] for r in ids if lo<=r['length']<=hi) for k in ['verified_distance_targets','verified_angle_targets','verified_torsion_targets','AA_missing_CA_targets']} for lo,hi in [(20,64),(65,128),(129,256),(257,384),(385,500)]};save('STRUCTURAL_OBSERVATION_COVERAGE.json',cov)
# Observable composition changes, no unmeasured family-diversity claim.
diff={a:comp['unused']['identity_first_composition'][a]-comp['joint_union']['identity_first_composition'][a] for a in comp['unused']['identity_first_composition']};comp['identity_first_composition_difference']=diff;comp['composition_total_variation']=sum(abs(v) for v in diff.values())/2;comp['identity_union_expansion_percent']=67615/50604*100;save('UNUSED_VS_JOINT_POPULATION_COMPARISON.json',comp)
table='| Primary identity category | Unique TRAIN identities |\n|---|---:|\n'+'\n'.join(f'| {label} | {c["primary_categories"][k]:,} |' for k,label in cats)+'\n| **Total** | **67,615** |'
mechanisms='| Strongest-source admission mechanism | Identities |\n|---|---:|\n'+'\n'.join(f'| `{k}` | {n:,} |' for k,n in mech['counts'].items())
examples=[]
for k,label in cats:
 for r in c['examples'].get(k,[])[:2]:examples.append(f'- **{label}**: {r["example_entry_id"]}, sequence `{r["sequence_id"]}`, observation `{r["example_observation_id"]}`; original BC reason `{r["example_BC_reason"]}`. Census reasons `{r["example_exclusion_reasons"]}`. Exact cache location: `{r["example_coordinate_ref"]}`.')
md('DATASET_VIEW_POLICY_TRACE.md',f'''# Existing view admission and exclusion trace

This is a TRAIN-only read-only census. All rule references are relative to `source/`, pinned to the recorded source commit. No observations were admitted and no sampling or model settings changed.

## Original complete view

`scripts/e016_v3/build_unified_catalogue.py` verifies a full canonical deposited entity string and labelled atom correspondence. It requires complete Cα; unambiguous chain mapping; no ambiguous altloc/tie; SOLUTION NMR or recorded resolution ≤3 Å for X-ray / ≤4 Å for the other admitted structural method; exact source/model/auth-chain/full-sequence/float32-Cα matching to a fully unbroken historical C–N sidecar. Absence of a matching sidecar records `PEPTIDE_CONTINUITY_UNASSESSED_NO_MATCHING_FULL_SIDECAR`; it does **not** establish a physical chain break.

`scripts/e016_v3/finish_unified_intake.py` supplies authoritative v3 overlays. Full admission requires verified mapping, complete Cα, method/resolution eligibility, unambiguous selected backbone conformer and every C–N continuity link. Older same-source mappings remain as explicitly superseded evidence. v3 resolutions use `resolutions_angstrom` (list); the census verifier reads this field rather than treating the absent legacy scalar as unknown.

`scripts/e016_v3/audit_atomic_identity_and_actions.py` additionally quarantines mismatched observed atom tokens and excludes actions outside the original distance/angle/torsion transformed-density support. No censoring or repair is performed in this census.

## Partial view

`scripts/e016_v3/prepare_partial_view.py` first executes **`if row['complete_CA']: continue`**. This occurs before partial mapping/quality/continuity assessment. Remaining observations require exact canonical parent ownership, verified label correspondence, existing method-resolution criteria, unambiguous author/model/altloc selection, and at least one valid local target backed by verified C–N continuity. Legacy continuity comes only from source/model/chain/residue/coordinate/conformer-matched sidecars; the frozen recovery journals were reused, not rebuilt. Missing Cα is not interpolated.

`partial_geometry.derive` permits distance with one valid consecutive link, angle with two, torsion with three; it never bridges a gap. All originally complete qualified observations remain available separately.

## The demonstrated omission

**10CX**, full length 179, sequence `14a631e66be5f8878076bc5cbe49afb2d5e25b280a2aeb203f3afab6fc8bfca0`, observation `483ffbd0d00ee2982a5c8cc23e810e9fdd70b5fd9700b475067251d2d54a6bfd`, has all 179 Cα, accepted 0.9-Å X-ray metadata and two failed source C–N continuity links. Its ordinary full view correctly fails the all-links requirement. Its partial view is never considered because Cα coverage is complete. Existing masked local semantics nevertheless yield **176 distance / 173 angle / 172 torsion** targets in three verified linked segments, longest 175 residues. The unchanged `derive` library independently reproduces these counts. This is a narrow view-definition gap, not permission to connect the two broken links or silently admit the observation.

## J1 sampling and the absent AA-only route

`j1_sequence_first.Sampler` draws its AA identities only from `base.jointids['TRAIN']` or `PartialData.ids['TRAIN']`; their union is 50,604. The unused identities are not merely unlucky stochastic samples: they are outside these sampler populations. Observation-first multiplicity is not used.

`scaled_supervised.Data.target(..., joint=False)` already represents a known AA prefix with all coordinate validity bits false and no geometry target. But `partial_continuation.PartialData.target` routes an unknown observation to `base.target(..., joint=True)`; `PartialData.batch` also expects a geometry vector. A future explicit AA-only branch must call `joint=False`, retain true N/t/AA labels and masks, emit all-false geometry-target masks and exclude the absent geometry from the loss. The optional geometry tensor may need a masked numerical placeholder for batching; it must never become a label. This production change has **not** been made.

The 25 CPU forward checks retain sequence tokens, produce finite AA likelihoods and exactly identical logits when all masked coordinate placeholders are NaN. Physical pair validity and geometry loss are zero; no suffix residues or future coordinates enter the encoder. The learned input computation remains the original bidirectional-within-literal-prefix computation.

## Exclusive classification and mechanism counts

Identity classification uses strongest evidence: policy-quality local-target omission first; then a trustworthy complete-Cα observation; then any trustworthy nonzero partial observation; then all-zero Cα; then only mapping/conformer-failing observations; no-link last. Mapping/conformer trust includes canonical position, model, finite coordinate, altloc and author-number checks, and excludes atomic-identity quarantine/superseded views. “Trustworthy” does not certify C–N continuity or physical fold quality.

The separate mechanism table chooses one source ranked by policy readiness, mapping/conformer trust, accepted quality, verified local targets, complete flag, observed count, and stable inventory order. It describes a first actionable admission blocker, not an exclusive diagnosis of physical biology. Other observations can have additional blockers.

{mechanisms}

No identity is classified as biologically disordered, intrinsically unstructured or lacking all deposited coordinates. Unknown metadata or unmatched evidence is kept unknown. The 596 `COMPLETE_CA_FLAG_CONTRADICTION` observations all already have `QUARANTINE_ATOMIC_IDENTITY`: their default qualification flag differs from the raw mask; this is not 596 newly demonstrated independent mask bugs.

## Representative exact evidence

{chr(10).join(examples)}

The per-identity and per-observation Parquet tables retain stable sequence/entity/observation/source IDs and exact original cache paths/row numbers. Multi-label reason counts overlap and must not be summed as disjoint identities.
''')
md('FINAL_UNUSED_SEQUENCE_CENSUS_REPORT.md',f'''# Unused E016 TRAIN sequence census

**Completed read-only. All 67,615 identities have linked observations and observed Cα atoms. Their absence from the structural views is an eligibility/evidence/view-definition result, not evidence that they lack experimental coordinates.**

## Reconciliation

The sealed sequence assignments contain 118,219 exact-unique canonical TRAIN sequences. The complete view has 21,475 identities / 63,315 observations. The frozen partial view has 32,144 TRAIN identities / 118,461 observations, with 3,015 identities overlapping the complete view. Thus 21,475 + 32,144 − 3,015 = 50,604; the exact set complement is **67,615**. Full strings, not only hashes, were checked for uniqueness and equality against verified deposited entity records. All unused identities have at least one verified LM-eligible entity occurrence.

## Mutually exclusive result

{table}

There are **342,713 linked observations** across these identities. Every one contains at least one cached observed Cα. Source/entity/hash ownership was checked for every observation. A total of **11,917** unused identities have a complete Cα mask in at least one record, including mapping/conformer-rejected records; 8,908 are in the primary trustworthy-complete category, with 10CX separated into the one clear omission category. “Complete Cα” does not imply complete N/C/O backbone information, verified peptide continuity, acceptable experimental quality or a valid full-chain training observation.

## Why admission failed

{mechanisms}

These source-selection mechanisms are another disjoint identity breakdown of the same 67,615, not extra populations. The largest blockers are missing indexed quality metadata, unmatched verified C–N evidence and mapping/conformer restrictions. Multiple blockers often coexist. Original BC reasons and overlapping observation-level checks are separately preserved in `EXCLUSION_REASONS_BY_OBSERVATION.json`; for example, 339,388 records lack matched verified continuity evidence and 146,500 fail method/resolution eligibility. Those numbers are overlapping observations, not proteins.

The original legacy catalogue sources its resolution lookup from the historical processed manifest. It does not extract resolution anew from every raw mmCIF. Bounded checks of {len(raw['records'])} representative local sources found all compared Cα position counts consistent with their cache. In particular, **103L (1.9 Å) and 10DC (2.08 Å)** contain resolution in the original CIF while the unified legacy metadata records null. Conversely 10BV actually reports 3.05 Å and would remain outside the original X-ray cutoff even after recovering metadata. Thus the number eventually recoverable by metadata repair is **not established** by the 24,585 identity-level missing-metadata classifications. No raw archive sweep or repair was performed.

## Existing geometry and honest recoverability

275,573 observations pass the census mapping/conformer checks and contain **51,248,228 observed Cα positions**, **8,524,299 known-AA positions with missing Cα**, and 425,372 consecutive observed-position runs. These are observation-multiplicity counts. A Cα run is not automatically a verified C–N segment.

Preserved authoritative continuity supports **274,594 distance / 270,875 angle / 267,170 torsion targets** across **175 identities / 3,033 observations**. There are {cov['verified_C_N_segments_total_with_at_least_one_link']:,} verified linked segments (at least two residues), longest {cov['longest_verified_C_N_segment']}. Of these, **174 identities / 3,032 observations** fail the existing method-resolution policy, so these targets are not immediately admitted geometry. The remaining **one** is 10CX: **176 / 173 / 172** targets recoverable through an explicit view-definition correction using unchanged masks and action semantics. No correction was applied.

The larger **50,822,856 / 50,397,855 / 49,973,389** counts of adjacent-CA candidate distance/angle/torsion windows lack the required accepted continuity/quality evidence. They are an upper candidate inventory, **not usable training targets**. The census does not assert their physical continuity or universal recoverability. Existing hashes of all 477 referenced cache artifacts were verified ({integ['coordinate_bytes_hashed']:,} bytes), and the original relation-edge evidence and accepted split remain available without the retired prefilters.

## AA-only readiness and pipeline limitation

**All 67,615** unused identities are data-ready for next-AA supervision: exact full deposited canonical strings, only the 20-AA vocabulary, true lengths 20–500, inherited TRAIN membership, no fragment renumbering. They supply **16,625,418 distinct identity-position labels**, including the BOS-conditioned first-residue target. Data eligibility does not imply current J1 loader admission.

The model's existing all-missing geometric masks support this representation: **25 bounded CPU inference checks passed**, with finite AA predictions, no physical geometric pair contribution and zero geometry loss. Masked NaN coordinate placeholders had exactly zero logit effect. No optimizer was constructed. The current J1 sampler/PartialData batch path needs an explicit AA-only route before it can use this population; see `DATASET_VIEW_POLICY_TRACE.md`. No production code, training mixture, normalization or model parameters were changed.

## Diversity comparison

| Population | Identities | Mean length | Median length | Length 385–500 |
|---|---:|---:|---:|---:|
| Existing joint-view union | 50,604 | 195.74 | 175 | 3,808 |
| Unused | 67,615 | 245.88 | 232 | 10,161 |

The unused identity counts by length are **2,856 / 9,459 / 26,280 / 18,859 / 10,161** in 20–64 / 65–128 / 129–256 / 257–384 / 385–500. The AA-only identity pool would expand by **133.6%**, and available distinct labels by **167.8%** relative to the present structural-view union. All length ranges already exist; the important gain is denser coverage, especially of longer sequences. Identity-first amino-acid composition total-variation distance is {comp['composition_total_variation']:.4f}; full frequencies are in the comparison JSON. This supports observable structural-selection bias in length/composition, without establishing its biological cause. Exact uniqueness is not evolutionary-family independence.

## Recommended next decision, not executed

Safely enabling these AA labels does not require a split change or invented geometry. First implement/test the explicit AA-only loader route. For a bounded future sampling pilot, use disjoint **40% complete identities / 30% partial-only identities / 30% currently unused AA-only identities**, with identity-first sampling and resumable position coverage. Preserve the independent 32 qualified complete-geometry decisions per 64 AA decisions. Partial-only means 29,129 identities; the 3,015 overlapping partial/complete identities must not be double-counted as new proteins. This deliberately limits the new all-missing-context input shift; it is an engineering proposal, not a measured optimum.

Uniform AA sampling over all 118,219 identities is the later population-proportional alternative: **18.17% complete / 24.64% partial-only / 57.19% currently unused**. Consider that larger all-missing fraction only after frozen complete/partial VALIDATION and AA-only robustness checks establish its effect. Keep structural and AA exposure denominators separate. Do not admit unverified geometry or expand a quality cutoff merely to enlarge the data.

## Verification, resource use and limits

The main projected-cache census took {run['elapsed_seconds']:.1f} seconds and peaked at {run['peak_RSS_bytes']/2**30:.2f} GiB RSS. Additional integrity checks and bounded raw confirmations are recorded separately. This evidence root uses roughly 50 MiB of compact outputs plus a small source worktree, with more than 17 GiB free on D:; the 10-GiB reserve was preserved. One report-packaging path error was corrected; its failed attempt log is retained. No training, optimizer update, TEST evaluation, dataset rebuild, homology screen, archive-wide parsing, additional deletion or recreation of retired prefilters occurred.

The classification exhaustively explains current view exclusion. It does not fully establish the physical quality or recoverability of every unassessed peptide link. Raw confirmations are bounded, not an estimate of archive-wide repair yield. Full raw source hash refresh was limited to representative entries; existing sealed cache hashes were rechecked across all referenced coordinate artifacts. The original recorded raw-source hash provenance remains the broad source basis.

Detailed evidence: `TRAIN_IDENTITY_SET_RECONCILIATION.json`, `UNUSED_IDENTITY_CLASSIFICATION.parquet`, `UNUSED_OBSERVATION_EVIDENCE.parquet`, `IDENTITY_EXCLUSION_MECHANISMS.json`, `DATASET_VIEW_POLICY_TRACE.md`, `CENSUS_SOURCE_INTEGRITY.json`, `CENSUS_CORRECTNESS_CHECKS.json`, `AA_ONLY_FORWARD_SMOKE_TEST.json`, and `PROVENANCE_SHA256.json`.

**Ready for review. No new training or data admission is authorized or performed by this census.**
''')
# Decision record belongs only to this new read-only evidence root.
md('EXPERIMENT_DECISIONS.md','''# Census decisions

- Scientific question: distinguish missing structural information from eligibility, provenance and view restrictions for the exact unused TRAIN complement.
- Use sealed downstream sequence, observation, entity and partial-view tables. Retired prefilter shards are neither required nor regenerated.
- Read existing coordinate masks, canonical mappings and conformer metadata in projected batches. Preserve Cα adjacency counts separately from verified C–N targets.
- Reuse frozen continuity recovery journals; inspect only representative raw files to check resolution metadata and Cα presence. Unknown source quality is not fabricated as a pass or a biological failure.
- Distinguish the complete-Cα-with-chain-break gap (10CX) from legitimately complete qualified chains. Count recoverable local targets with unchanged derive semantics; do not admit the example.
- Check all-missing geometry through inference only. Production model, sampler and normalization are unchanged.
- Record unknown legacy resolution and missing matched C–N evidence as metadata/evidence restrictions; neither proves absence of deposited coordinates.
- Prefer a future bounded AA-only-mixture pilot rather than immediately making 57.19% of AA contexts entirely geometry-unavailable. No sampler change is made here.
''')
md('CHANGELOG.md','''# Census changes

Added standalone read-only scripts for set reconciliation, projected coordinate census, source/ownership/hash verification, AA-only forward diagnostics, unchanged geometry-library checks and compact reports. No production package files, model weights, optimizer configuration, dataset membership or historical evidence were changed. A report path concatenation error in the first verification attempt was fixed; the attempt log remains as evidence.
''')
save('RESOURCE_REPORT.json',dict(main_census=run,additional_integrity_elapsed_seconds=integ['additional_elapsed_seconds'],CPU_forward_elapsed_seconds=load('AA_ONLY_FORWARD_SMOKE_TEST.json')['elapsed_seconds'],new_optimizer_updates=0,GPU_training_jobs_started=0,free_D_bytes=shutil.disk_usage(R).free,new_evidence_file_bytes=sum(p.stat().st_size for p in R.iterdir() if p.is_file()),reserve_bytes=10*2**30,retired_artifacts_recreated=0,data_deleted_bytes=0))
print('reports written')
