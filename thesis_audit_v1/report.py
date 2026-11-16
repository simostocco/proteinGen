"""Assemble evidence-backed reports from the completed census, without training."""
from pathlib import Path
import collections, hashlib, json, shutil, sqlite3, subprocess, sys
from audit import REPO,HIST,SIDE,PAIR,FROZEN,files,rows,sha
def finalize(out):
    def read(n):return json.loads((out/n).read_text())
    def write(n,s):(out/n).write_text(s)
    def save(n,d):write(n,json.dumps(d,indent=2,sort_keys=True)+'\n')
    d=read('DUPLICATION_REPORT.json');t=read('TERMINI_AND_ORDER_AUDIT.json');p=read('PROVENANCE_SHA256.json');s=d['populations']
    db=sqlite3.connect(f'file:{out}/audit.sqlite?mode=ro',uri=True)
    collisions=db.execute('SELECT count(*) FROM (SELECT h FROM records GROUP BY h HAVING count(DISTINCT seq)>1)').fetchone()[0]
    assert collisions==0
    d['sha256_collision_groups_checked']=collisions
    d['e012_retained_sequences_with_multiple_full_polymers_in_panel']=sum(len({m['full_hash'] for m in t['mapping_examples'] if m.get('retained_hash')==g['hash']})>1 for g in d['structural_panel'])
    d['terminal_metadata_e012_train']={k:db.execute(q).fetchone()[0] for k,q in {
        'n_terminal_trimmed_records':"SELECT count(*) FROM records WHERE pop='e012_train' AND tn>0",
        'c_terminal_trimmed_records':"SELECT count(*) FROM records WHERE pop='e012_train' AND tc>0",
        'either_terminal_trimmed_records':"SELECT count(*) FROM records WHERE pop='e012_train' AND tn+tc>0",
        'unknown_terminal_metadata_records':"SELECT count(*) FROM records WHERE pop='e012_train' AND (tn IS NULL OR tc IS NULL)",
        'declared_trimmed_n_residues':"SELECT sum(tn) FROM records WHERE pop='e012_train'",
        'declared_trimmed_c_residues':"SELECT sum(tc) FROM records WHERE pop='e012_train'"}.items()}
    homroot=REPO/'reports/experiments/E007_matrix_sequence_cogeneration/split_homology_audit_v1_publication_correction_v1'
    homreport=json.loads((homroot/'report.json').read_text());homology={}
    sequence_identity={r[0]:(r[1],r[2]) for r in db.execute("SELECT id,h,substr(pop,6) FROM records WHERE pop IN ('e012_train','e012_validation')")}
    accepted_train_ids=set()
    epanels=json.loads((HIST/'reports/experiments/E012_causal_rope_sequence/pilot_v1/panels.json').read_text())
    for label in ['identity_30','identity_50','identity_70','identity_90']:
        q=homroot/f'mmseqs/{label}/cluster_assignments.parquet'
        clusters=collections.defaultdict(collections.Counter);val_ids={}
        for r in rows(q,['sample_id','split','cluster_id','sequence_hash']):
            assert sequence_identity[r['sample_id']]==(r['sequence_hash'],r['split'])
            if label=='identity_30' and r['split']=='train':accepted_train_ids.add(r['sample_id'])
            clusters[r['cluster_id']][r['split']]+=1
            if r['split']=='validation':val_ids[r['sample_id']]=r['cluster_id']
        mixed={c for c,v in clusters.items() if v['train'] and v['validation']}
        affected={sid for sid,c in val_ids.items() if c in mixed}
        assert len(affected)==homreport['homology_thresholds'][label]['affected_validation_samples']
        assert len(mixed)==homreport['homology_thresholds'][label]['cross_split_clusters']
        homology[label]={'assignment_path':str(q),'assignment_sha256':sha(q),'mixed_cluster_count':len(mixed),'validation_records_affected':len(affected),'validation_denominator':len(val_ids),'clean_validation_records':len(val_ids)-len(affected),'e012_fixed_panels_affected':{name:len(set(ids)&affected) for name,ids in epanels.items()},'identity_and_sequence_match_e012':True,'scope':'existing reclustering of E007 accepted 230132 TRAIN / 23161 VALIDATION; no MMseqs rerun, no new alignments'}
        additional_sources=files(q)
        for src in additional_sources:p['inputs'].append({'path':str(src),'sha256':sha(src),'bytes':src.stat().st_size})
    d['historical_homology_reclustering']=homology;save('DUPLICATION_REPORT.json',d)
    needed={m['sample_id']:m for m in t['mapping_examples']};raw_seen=set()
    for r in rows(SIDE/'eligibility_manifest.parquet',['sample_id','source_path','source_sha256','npz_path','npz_sha256']):
        if r['sample_id'] not in needed:continue
        needed[r['sample_id']]['source_path']=r['source_path'];needed[r['sample_id']]['recorded_source_sha256']=r['source_sha256'];needed[r['sample_id']]['historical_npz_path']=r['npz_path'];needed[r['sample_id']]['historical_npz_sha256']=r['npz_sha256']
        if r['source_path'] not in raw_seen:
            q=Path(r['source_path']);digest=sha(q);p['inputs'].append({'path':str(q),'sha256':digest,'bytes':q.stat().st_size,'expected_sha256':r['source_sha256'],'expected_match':digest==r['source_sha256']});raw_seen.add(str(q))
    save('TERMINI_AND_ORDER_AUDIT.json',t)
    additional=[REPO/'data/full/processed/preprocess_summary.json',REPO/'data/full/processed_recovery/preprocess_summary.json',REPO/'.e015_worktree/configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml']
    # E015 uses the sealed E014 population. Recover its artifact content and compare identities.
    e14=REPO/'e014_artifacts'
    population_candidates=[]
    for q in sorted((e14/'population').glob('*.parquet')):
        if q.stat().st_size<300*1024**2:population_candidates.append(q)
    population_evidence=[]
    for q in population_candidates:
        import pyarrow.parquet as pq
        pf=pq.ParquetFile(q);names=pf.schema_arrow.names
        info={'path':str(q),'rows':pf.metadata.num_rows,'columns':names,'sha256':sha(q)}
        if 'sample_id' in names:
            ids={r['sample_id'] for r in rows(q,['sample_id'])}
            known={r[0] for r in db.execute("SELECT id FROM records WHERE pop='e012_train'")}
            info.update(unique_sample_ids=len(ids),outside_e006_train=len(ids-known))
            if q.name=='train.parquet':
                digest=hashlib.sha256(json.dumps(sorted(ids),sort_keys=True,separators=(',',':')).encode()).hexdigest()
                info.update(e007_accepted_train_identity_difference=len(ids^accepted_train_ids),sorted_train_identity_sha256=digest)
                assert digest=='d3f2ee345d0e4e9a27d0136aad1eca9dfe87821a3a82d90aa4c3194bd928a3f9'
                assert ids==accepted_train_ids
        population_evidence.append(info);additional.append(q)
    additional += [e14/'population/population.json',REPO/'.e015_worktree/src/protein_distance_diffusion/training/e015_runtime.py',REPO/'.e014_worktree/src/protein_distance_diffusion/data/e014_population.py',homroot/'report.json',homroot/'protocol.json',HIST/'reports/experiments/E012_causal_rope_sequence/pilot_v1/panels.json']
    population_protocol=json.loads((e14/'population/population.json').read_text())
    for source in population_protocol['source_hashes']:
        q=Path(source)
        if q.is_file() and q.stat().st_size<100*1024**2:additional.append(q)
    for q in additional:
        if q.is_file():p['inputs'].append({'path':str(q),'sha256':sha(q),'bytes':q.stat().st_size,'expected_sha256':None,'expected_match':None})
    p['audit_code']=[{'path':str(q),'sha256':sha(q)} for q in sorted((REPO/'thesis_audit_v1').glob('*.py'))]
    save('PROVENANCE_SHA256.json',p);save('E015_POPULATION_EVIDENCE.json',population_evidence)
    accepted=d['e007_coordinate_acceptance'];a=s['e012_train'];v=s['e012_validation'];classes=d['pair_class_counts'];mapped=[m for m in t['mapping_examples'] if 'full_hash' in m]
    def table(data):return '\n'.join(f'| {name} | {val.get("records")} | {val.get("unique_retained_sequences","—")} |' for name,val in data.items())
    write('DATASET_LINEAGE.md',f'''# Historical lineage — audit v1

VERIFIED denotes this run's source/artifact observations; historical protocol claims remain identifiable. INFERRED denotes interpretation. UNKNOWN denotes unavailable or unmeasured evidence. No training or structure prediction was run.

## Authoritative population definitions

| Population | Records | Exact unique retained sequences |
| --- | ---: | ---: |
{table(s)}

Counts are rows, not independent proteins or folds. Checksums and original absolute paths are in PROVENANCE_SHA256.json; source-code snapshots are current files at recorded repository heads, not proof that every historical sample used that exact revision.

## Flow

```mermaid
flowchart TD
  A[Deposited PDB entry: experimental method and resolution] --> B[Polymer entity: deposited construct sequence]
  B --> C[Chain: label asym and author chain IDs]
  C --> D[Model 1 atoms and selected alternate conformers]
  D --> E[Baseline strict CA and residue filtering]
  D --> F[Recovery: insertion codes and terminal CA trimming]
  E --> G[Merged 506919 chain records]
  F --> G
  G --> H[Within-entry exact retained-sequence collapse: 338481]
  H --> I[Exact unique sequence MMseqs clusters; PDB and cluster grouping]
  I --> J[TRAIN all structures: 270785]
  I --> K[VALIDATION representatives: 25257]
  I --> L[TEST representatives: 25066]
  J --> M[Pairing eligible TRAIN: 268330]
  K --> N[Pairing eligible VALIDATION: 25075]
  M --> O[E006 provenance and stereochemical checks: TRAIN 231743]
  N --> P[E006 checks: VALIDATION 23307]
  O --> Q[E011 to E012 sequence-only projection: 231743 rows]
  P --> R[E012 VALIDATION: 23307 rows]
  O --> S[E007 complete CA and uninterrupted peptide links: 230132]
  S --> T[E014/E015 sealed geometry population; see identity evidence]
```

VERIFIED from recovery summary: strict baseline 151364 samples plus 355555 newly recovered samples = 506919. Recovery candidate selection explicitly revisits baseline `missing_calpha` and `insertion_codes` rejections. Its 71772 reused samples are processing evidence, not an additional contribution to that sum. Historical summaries mix file and chain rejection events; their counts must not be subtracted from a single hypothetical chain denominator.

Split workflow: 506919 to 338481 removes 168438 within-PDB exact retained-sequence copies. 146872 exact retained sequences are clustered using MMseqs2 15-6f452+ds-2, 30% identity, 80% coverage, cov-mode 0, seed 42, with PDB/cluster group protection. Stored splits: TRAIN 270785; VALIDATION and TEST all-structure views 33848 each. Canonical validation/test files keep 25257/25066 representatives. These are actual manifest definitions, not names inferred in isolation.

Pairing uses the original TRAIN and representative VALIDATION memberships, excludes 2455 TRAIN and 182 VALIDATION pairing-ineligible rows, yielding 268330+25075=293405. Pairing protocol classifies 501797 of 506919 globally eligible, but most globally eligible records are outside these selected memberships. It does not create a new random split.

E006 excludes 38355 of 293405 selected pairing rows, yielding 255050 = 231743 TRAIN + 23307 VALIDATION. Exact per-split exclusions are recomputed from eligibility partitions in DUPLICATION_REPORT.json. The named categories are identity mapping, CA anchor ambiguity, residue namespace, source/NPZ coordinate, stereochemical contradiction and unresolved modification.

E007 adds complete CA and uninterrupted C–N connectivity requirements. Recomputed combinations: `{json.dumps(accepted,sort_keys=True)}`. Thus 1611 TRAIN and 146 VALIDATION records are excluded, yielding 230132 and 23161. E012 does **not** apply this extra E007 filter; it projects the E006 population unchanged. E015's config points to E014's frozen population and its hash; E015_POPULATION_EVIDENCE.json records recovered artifacts rather than assuming equivalence from an experiment name.

## Source/version limitations

Current mmCIF source uses numeric label positions, while baseline rejection summaries include insertion-code rejection, showing the parser evolved. The historical source code revision for each preprocessing event is UNKNOWN unless an artifact explicitly records it. Present source behavior and immutable sample evidence are reported separately. Dataset protocol SHA256, schema, vocabulary, normalization, shard inventory and every 744 sidecar shard are hashed and checked against the sealed E006 protocol. Historical models and the certified 20M corpus are not opened or rehashed.
''')
    write('DATASET_AUDIT.md',f'''# proteinGen thesis PDB sequence audit v1

## Findings

1. **VERIFIED:** 231743 historical E012 TRAIN rows represent {a['unique_retained_sequences']} exact retained sequences. There are {a['excess_duplicate_records']} excess duplicate records ({100*a['excess_duplicate_records']/a['records']:.2f}% of rows). Maximum multiplicity is {a['max_multiplicity']}. Exact groups across entries: {a['sequences_across_entries']}; groups with within-entry repetition: {a['sequences_repeated_within_entry']}. These quantities overlap conceptually and are not fold counts. Historical TRAIN kept alternate entries after within-entry sequence collapse; duplicate abundance therefore largely reflects repeated deposition and retained constructs. Counts are recomputed from actual strings, with no SHA256 collisions among distinct strings in the census.
2. **VERIFIED, bounded panel:** {len(d['structural_panel'])} duplicated sequence groups, {len(t['mapping_examples'])} selected records, {len(d['structural_pairs'])} pairs; classifications `{json.dumps(classes,sort_keys=True)}`. RMSD/contact thresholds and every pair are in DUPLICATION_REPORT.json. This panel measures differences in deposited CA conformations, not thermodynamic states or fold families. The population-wide number of distinct folds is UNKNOWN. The first four records per group cap and hash-ranked stratification prevent interpreting these fractions as population prevalence; high-multiplicity groups remain group-level inconclusive unless heterogeneity is witnessed.
3. **VERIFIED:** {d['e012_retained_sequences_with_multiple_full_polymers_in_panel']} sampled retained-sequence groups map to multiple full-polymer hashes. Corresponding pairs are conservatively inconclusive for structural correspondence. Full-polymer and retained-sequence equality are distinct. Entire-population prevalence of trimming-induced equivalence is UNKNOWN because all raw entries were not reparsed. Declared TRAIN terminal metadata: `{json.dumps(d['terminal_metadata_e012_train'],sort_keys=True)}`. Null fields mean UNKNOWN, not zero trimming. Baseline `original_chain_length` is not necessarily full entity length.
4. **VERIFIED:** E012 reads the preserved sequence strings in order; BOS input is `[BOS]+tokens[:-1]`, target is `tokens`, no EOS target. Requested length is the retained sequence length and generation runs exactly that many canonical steps. Padded positions are excluded from loss. Author IDs are never lexicographically sorted by the current mmCIF parser. {t['denominators']['n_to_c_mapped']}/{t['denominators']['authoritatively_mapped']} successfully mapped panel records increase in authoritative label order. This supports correct order for audited records and code, not an unconditional all-history proof. Source-derived zero-based sidecar IDs must not be mistaken for author residue numbers.
5. **VERIFIED selection:** methods restricted to X-ray, EM and solution NMR; X-ray <=3 A and EM <=4 A; retained length 20–500; strict baseline rejects missing CA; recovery accepts only terminal trimming <=25% and rejects internal unresolved CA. E006 adds provenance/modification/stereochemical filters. E007/E015 additionally require complete connected CA views. **INFERRED:** this favors short, canonically mappable, coordinate-resolved regions and experimental constructs; it undersamples internally unresolved chains and loses terminal context. It cannot estimate the proportion of biological proteins that fold.
6. **Feasible design:** preserve full deposited polymers and evidence masks; exact-unique PDB-only sequence LM; justified alternate structures in a separate linked table; positive/unlabeled compatibility task with context and exposure-aware evaluation. Proposed schema and experiments make choices explicit. No new scientific dataset is materialized or silently substituted.
7. **UNKNOWN:** independent-chain thermodynamic stability, equilibrium folding free energy, cooperative folding, kinetic accessibility, concentration/condition dependence and native-state uniqueness. PDB presence alone supplies none of those labels. Missing coordinates are not proven IDRs, and shuffles are statistical controls, not experimentally demonstrated nonfolders.

## Duplicate weighting and leakage

Uniform record sampling gives sequence probability m/231743. Inverse-multiplicity weight 1/m gives total sequence weight 1 and total TRAIN weight {a['effective_inverse_multiplicity_total_weight']}. The entropy-effective sequence count under historical uniform record exposure is {a['entropy_effective_sequences_under_uniform_record_sampling']:.2f}, compared with {a['unique_retained_sequences']} exact groups. Kish effective *records* under inverse multiplicity are {a['kish_effective_records_with_inverse_multiplicity']:.2f}; this is a weighting diagnostic, not independent fold count. Per-sequence probabilities and weights are exported in TSVs. Length/method/resolution/chain/trim concentration tables are in JSON; chain identifiers are naming context, not biological assembly labels.

VERIFIED cross-split intersections (sample, retained sequence, PDB, stored homology cluster): `{json.dumps(d['leakage_e012_train_validation'],sort_keys=True)}`. E006 projection mismatches with E012: {d['sidecar_sequence_projection_mismatches']}. This verifies stored-group separation. It does not prove all train/validation pairs are below 30% identity: greedy representative clustering can leave cross-cluster homology, and observed truncations can obscure full-polymer relationships. No MMseqs computation was rerun.

## E012 order-learning evidence

Historical E012 was a scratch-initialized causal RoPE sequence LM with fixed length conditioning and equal protein-record loss; no sequence deduplication or 1/m correction in the execution path. Existing 2000-update result is CSEQ-C: ordered-prefix sensitivity and beyond-eight-residue benefit on both fixed 2048-record panels, but frozen unigram-margin and short-stratum gates fail; independent bigram interval includes zero. This is source/report evidence, not a new model evaluation. Prefix shuffling preserves composition at matched targets. Baselines use TRAIN only, but record multiplicity biases their counts. Panel records need not be sequence-independent; bootstrap uncertainty should be reassessed at exact-sequence/homology-cluster level in a future version.

The 20M UniRef50 warm-start system has its own frozen protocol and model lineage. It is external pretraining and cannot be presented as PDB-only evidence. Its artifacts and corpus remain unchanged; no training was launched.

## Reproducibility and scope

Read DATASET_LINEAGE.md for exact filters and denominators, TERMINI_AND_ORDER_AUDIT.json for full deposited/canonical strings and residue mappings in the panel, DISORDER_LABEL_EVIDENCE.md for biological evidence boundaries, and PROPOSED_EXPERIMENTS.md for review choices. PROVENANCE_SHA256.json lists original paths, observed byte hashes and declared protocol hashes separately. RUN_METRICS.json reports resource use; test_results.txt records 18 isolated CPU tests including demonstrated unresolved terminal-monomer loss, legacy SEQRES omissions and lost insertion codes. Historical artifacts were read only. All outputs/caches are on D:.

Reproduce using the copied scripts in `reproduce/` with a fresh D: output directory: `python -B audit.py --output <fresh-D-root>` then `python -B audit.py --output <fresh-D-root> --finalize`. Use the recorded proteingen environment; `python -B test_audit.py` runs fixture tests on D:. No network, GPU, inference or training is needed. The raw source mappings are a bounded panel, not a deposited-PDB-wide census. Assembly, mutation, fragment and unobserved metadata are retained where available; ligand/oligomer dependence and experimentally curated disorder are not inferred from those fields.
''')
    with (out/'DATASET_AUDIT.md').open('a') as f:
        f.write('\n## Verified later homology leakage\n\nThe later published E007 reclustering contradicts a blanket claim of homology-safe historical validation. Recomputed assignments: '+json.dumps(homology,sort_keys=True)+'\n\nAt 30% identity / 80% coverage, 1578 of 23161 coordinate-accepted validation records fall in mixed TRAIN/VALIDATION clusters; the clean geometry validation population has 21583 records. These are homology-cluster links, not proof of identical sequence or thermodynamic equivalence. E012 retains its original 23307 validation rows; panel overlap counts above quantify known affected records within the reclustered accepted subset. The remaining 146 coordinate-rejected E012 validation rows were outside this reclustering and their homology status is UNKNOWN. Existing results are unchanged but must be described with this limitation.\n')
    with (out/'DATASET_LINEAGE.md').open('a') as f:
        f.write('\nLater homology publication correction reuses existing MMseqs calculations and excludes 1578 mixed-cluster validation records from 23161, leaving 21583. E014/E015 select fixed 500 development and 2000 prospective validation records from this clean pool; their full TRAIN remains 230132. E012 historical validation is unchanged. The four threshold assignment tables and panel intersections are recomputed in DUPLICATION_REPORT.json.\n')
    write('DISORDER_LABEL_EVIDENCE.md','''# Disorder and native-state evidence

| Topic | VERIFIED pipeline/artifact behavior | Biological interpretation |
| --- | --- | --- |
| Resolved regions | Requires canonical sequence/coordinate correspondence; E007 complete CA and peptide connectivity | Observable model under experimental conditions; independent stability UNKNOWN |
| Unresolved termini | Recovery trims up to 25% combined termini; baseline can have older observed-only definitions | Missing density may reflect flexibility, tag, processing or experimental limitations; IDR UNKNOWN |
| Unresolved internal region | Scheme-aware mmCIF rejects internal absent CA; fallback/legacy topology may omit positions | Strong selection toward resolved contiguous regions; no direct disorder classification |
| Flexible regions | No curated disorder evidence variable; model/altloc selection reduces representation | B factors and absent atoms are condition-dependent observations, not definitive IDR labels |
| NMR ensemble | mmCIF accepts SOLUTION NMR but selects model 1 | One model does not establish unique conformation or resolve ensemble uncertainty |
| Fold on binding | No verified explicit exclusion by biological dependence | Binding-dependent ordering remains eligible if coordinates satisfy filters |
| Cofactors/oligomers | Chains extracted; no independent-chain stability test; panel assembly records preserved | Assembly/ligand presence is context evidence, not proof of essential dependence |
| Engineered constructs/tags | Deposited entity strings and mutation/fragment/sequence-difference annotations retained in audit panel; historic sequence follows retained coordinates | Tags and truncations can survive or be lost depending on visibility; native gene sequence equality UNKNOWN |

New schema uses `disorder_evidence.status = known / absent / unknown`. `absent` requires explicit supporting experimental evidence within stated residue range and conditions; no database annotation defaults to unknown. Missing coordinates go into resolvability masks only. A positive compatibility observation refers to the deposited experimental construct in its context. Unobserved proteins are unlabeled, not negatives. A PDB-only cohort cannot identify unbiased biological prevalence or train a fully calibrated binary folder/nonfolder classifier without an explicit observation/selection model and suitable unlabeled population.

Thermodynamic evidence missing from this audit: reversible equilibrium unfolding/refolding measurements, folding free energies across specified conditions, calorimetric/van't Hoff cooperativity, concentration dependence, cofactor/binding titrations, aggregation controls, ensemble populations and exclusion of alternative minima. Experimental NMR/kinetic/spectroscopic evidence can constrain dynamics but must be evaluated separately. Neither deposited structure nor low aligned RMSD proves a unique thermodynamic native state. Shuffled sequences have no certified nonfolding label.

Authoritative annotation semantics: wwPDB says polymer sequence represents the experimental sample including tags and unmodeled residues ([processing procedures](https://www.wwpdb.org/documentation/procedure)). Canonical sequence positions and author numbering are different namespaces ([sequence scheme](https://mmcif.wwpdb.org/dictionaries/mmcif_ma.dic/Items/_pdbx_poly_seq_scheme.seq_id.html), [author numbering](https://mmcif.wwpdb.org/dictionaries/mmcif_nmr-star.dic/Items/_pdbx_poly_seq_scheme.auth_seq_num.html)). These sources define metadata semantics; they do not certify historical labels. Source inspection shows no curated disorder/stability labels used in the audited path.
''')
    fields={
    'schema_version':{'const':'pdb_only_research_v1'},'sample_id':{'type':'string'},'pdb_id':{'type':'string'},'entity_id':{'type':'string'},'label_asym_id':{'type':'string'},'auth_asym_id':{'type':'string'},'model_number':{'type':'integer','minimum':1},
    'deposited_full_sequence':{'type':'string'},'canonical_sequence':{'type':'string'},'observed_coordinate_sequence':{'type':'string'},
    'deposited_sequence_sha256':{'type':'string'},'canonical_sequence_sha256':{'type':'string'},'observed_sequence_sha256':{'type':'string'},
    'residue_mapping':{'type':'array','items':{'type':'object','required':['label_seq_id','auth_seq_id','insertion_code','deposited_monomer','canonical_residue','coordinate_index','ca_observed','canonicalization_evidence'],'properties':{'label_seq_id':{'type':'integer','minimum':1},'auth_seq_id':{'type':['string','null']},'insertion_code':{'type':'string'},'deposited_monomer':{'type':'string'},'canonical_residue':{'type':['string','null']},'coordinate_index':{'type':['integer','null'],'minimum':0},'ca_observed':{'type':'boolean'},'canonicalization_evidence':{'type':'object'}}}},
    'missing_residue_mask':{'type':'array','items':{'type':'boolean'}},'missing_ca_mask':{'type':'array','items':{'type':'boolean'}},
    'terminal_trimming':{'type':'object','required':['n_count','c_count','reference_sequence','policy','applied'],'properties':{'n_count':{'type':'integer','minimum':0},'c_count':{'type':'integer','minimum':0},'reference_sequence':{'const':'deposited_full_polymer'},'policy':{'type':'string'},'applied':{'type':'boolean'}}},
    'experimental_method':{'type':['string','null']},'resolution_angstrom':{'type':['number','null']},
    'structural_context':{'type':'object','required':['assembly_ids','assembly_evidence','ligands','partner_chains','dependence_evidence','construct_metadata'],'properties':{'assembly_ids':{'type':'array'},'assembly_evidence':{'type':['object','null']},'ligands':{'type':'array'},'partner_chains':{'type':'array'},'dependence_evidence':{'type':['object','null']},'construct_metadata':{'type':'object'}}},
    'disorder_evidence':{'type':'object','required':['status','evidence','conditions','residue_ranges'],'properties':{'status':{'enum':['known','absent','unknown']},'evidence':{'type':'array'},'conditions':{'type':['object','null']},'residue_ranges':{'type':'array'}}},
    'resolvability':{'type':'object','required':['observed_fraction','ca_fraction','internal_missing_ranges','alternate_conformers','ensemble_model_count','peptide_continuity','selection_reason']},
    'sequence_multiplicity':{'type':'object','required':['full_polymer','observed_sequence','pdb_entries','population_version']},'homology_cluster_id':{'type':['string','null']},'split':{'enum':['train','validation','test','unassigned']},
    'provenance':{'type':'object','required':['source_path','source_sha256','deposition_revision','parser_version','parser_sha256','config_sha256','canonicalization_version','dataset_manifest_sha256','split_manifest_sha256'],'properties':{'source_path':{'type':'string'},'source_sha256':{'type':'string'},'deposition_revision':{'type':['string','null']},'parser_version':{'type':'string'},'parser_sha256':{'type':'string'},'config_sha256':{'type':'string'},'canonicalization_version':{'type':'string'},'dataset_manifest_sha256':{'type':'string'},'split_manifest_sha256':{'type':['string','null']}}}}
    save('PROPOSED_PDB_ONLY_SCHEMA.json',{'$schema':'https://json-schema.org/draft/2020-12/schema','$id':'urn:proteingen:pdb-only-research:v1','title':'Proposed polymer/chain observation; no thermodynamic folding label','type':'object','required':list(fields),'properties':fields,'additionalProperties':False,'$comment':'Design only. Cross-field invariants tested in contracts.py: mapping covers full polymer in canonical order; mask lengths equal full length; absent coordinate indices null; modified monomers retained even if no canonicalization is justified; no gaps create peptide adjacency. Full hashes define LM dedup; structure observations retain model/altloc/context identity. Cluster null only before assignment; leakage disjointness required before release.'})
    write('PROPOSED_EXPERIMENTS.md','''# Versioned PDB-only research designs — no launch authorization

1. **Exact-unique language model dataset:** unit is full deposited canonical construct sequence, deduplicated by exact string/hash, with source links to all experimental observations. Retain tags/mutations in provenance; do not silently strip them. Unknown/ambiguous monomers require an explicit reject-versus-unknown-token policy. Full sequence preserves unresolved regions; a separately named observed-sequence ablation can measure selection effects. Assign all linked entries and homology groups jointly to TRAIN/VALIDATION/TEST; freeze entity-aware grouping, thresholds, seed and manifests. Compare deduplicated training with repeated-record and 1/m weighting controls at matched tokens/exposures, architecture and compute. Use independent full-polymer/homology-cluster bootstrap and held-out likelihood/order sensitivity across lengths. Prespecify EOS-trained variable-length LM versus historical length-conditioned fixed-length no-EOS LM as different objectives.

2. **Structure observations dataset:** preserve entity, chain, model, atom alternate-conformer choices, residue mapping, missing masks and assembly/ligand context. Full sequence is the reference; unobserved coordinates remain masked and internal gaps never create bonds. Collapse only demonstrated redundant observations with aligned-residue coverage, proper-rotation CA RMSD, distance/contact criteria and matched context. Retain well-supported alternate conformations and NMR ensembles; record ambiguous alignments as inconclusive. Choose thresholds and ensemble weighting through sensitivity analysis, not a universal fold label. Keep sequence-linked alternate conformations within one split.

3. **Positive/unlabeled structural-order compatibility:** a positive is an experimentally observed, sufficiently resolved conformation of a specified deposited construct under recorded context, with explicit completeness/heterogeneity criteria. Sequences lacking that evidence are unlabeled. A PDB-only archive is a selected cohort: incompletely resolved chains can supply limited within-archive unlabeled examples, but absence of PDB coverage cannot be interpreted as biological negative evidence. Model sequence-to-observability compatibility, optionally conditioned on requested experimental context, without predicting coordinates. Keep resolvability distinct from experimentally curated disorder and from stability. Prevent resolution, sample IDs, source metadata or coordinate-derived target summaries from entering a sequence-only predictor; evaluate whether length/composition/construct bias explains performance. Artificial shuffles are perturbation controls with unknown biological labels. Calibration and class-prior identification require external unbiased sampling or explicitly stated sensitivity assumptions.

## Required release checks

Check numeric label orientation, insertion/negative author IDs, unsupported terminal monomers, full entity coverage, model/altloc correspondence, missing CA and peptide adjacency, exact-string and PDB/group leakage, cross-cluster full-polymer similarity, assembly/context grouping, and representative-split selection. Tests in reproduce/test_audit.py verify current source and proposed invariant checks. No historical parser is patched. The unsupported terminal scheme omission and legacy PDB policy omissions are documented; complete residue preservation is an unambiguous invariant, but canonicalizing unknown chemistry or changing scientific inclusion criteria requires review before a new parser/dataset is released.

## Questions requiring scientific review

Choose full construct versus curated native sequence, tag policy, unsupported monomer handling, length range, allowed methods/resolution, whether to retain NMR model ensembles, observability threshold, context dependence, homology split coverage, and similarity thresholds. Audit every raw entity before claiming whole-population trimming-induced duplicates or no source-order failures. Existing sidecars must be reused rather than recreated; certify the new dataset under a new version/path after these choices.

## Handoff

Historical E007/E015/E012 remain paused. No training, GPU work, ColabFold, ProteinMPNN, Foldseek, expensive inference, deletion or historical replacement occurred. The E012 reservoir and certified corpus remain untouched. Review DATASET_AUDIT.md and UNKNOWN findings before defining thesis claims. Await further instructions.
''')
    rep=out/'reproduce';rep.mkdir(exist_ok=True)
    for q in (REPO/'thesis_audit_v1').glob('*.py'):shutil.copyfile(q,rep/q.name)
    shutil.copyfile(REPO/'thesis_audit_v1/test_results.txt',out/'test_results.txt')
    save('TEST_RESULTS.json',{'tests':18,'passed':18,'scope':'CPU synthetic mmCIF and legacy PDB parser and mapping contracts; existing source behavior including demonstrated terminal monomer omission and legacy SEQRES/insertion loss; BOS/shift/padding/loss; no model inference','log':'test_results.txt'})
    processes=subprocess.check_output(['ps','-eo','pid,stat,comm,args'],text=True)
    matches=[line for line in processes.splitlines() if any(x in line for x in ['train_e007','train_e015','run_e012_causal_rope.py','run_e012_large_corpus','train_e012']) and not any(x in line for x in ['audit.py','report.py','codex'])]
    save('SAFETY_STATUS.json',{'training_started':False,'historical_files_written':False,'reservoir_deleted':False,'gpu_used':False,'local_matching_training_processes':matches,'scope':'local Linux process observation; no claim about uninspected remote hosts','frozen_e012_status':json.loads((FROZEN/'status.json').read_text()),'frozen_e012_pause_verification':json.loads((FROZEN/'pause_verification.json').read_text())})
    assert all(r.get('expected_match') is not False for r in p['inputs'])
    assert 'Ran 18 tests' in (out/'test_results.txt').read_text() and '\nOK\n' in (out/'test_results.txt').read_text()
    import resource
    from importlib.metadata import version
    versions={name:version(name) for name in ['numpy','pyarrow','gemmi','matplotlib','MDAnalysis','torch','pytest']}
    save('FINALIZATION_METRICS.json',{'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'homology_calculations_reused':True,'training_launched':False,'python':sys.version,'dependency_versions':versions,'interpreter':sys.executable})
    save('OUTPUT_SHA256.json',{str(q.relative_to(out)):sha(q) for q in sorted(out.rglob('*')) if q.is_file() and q.name!='OUTPUT_SHA256.json' and q.suffix not in ['.sqlite'] and 'mplconfig' not in q.parts})
    db.close();print('Reports assembled',out)
