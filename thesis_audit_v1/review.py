"""Final compact evidence tables and consistency checks; no source datasets written."""
from pathlib import Path
import collections,json,shutil
import numpy as np
from audit import REPO,SIDE,sha,rows
def review(out):
    d=json.loads((out/'DUPLICATION_REPORT.json').read_text());t=json.loads((out/'TERMINI_AND_ORDER_AUDIT.json').read_text());run=json.loads((out/'RUN_METRICS.json').read_text())
    mapped=[m for m in t['mapping_examples'] if 'full_hash' in m];pairs=[r for r in d['structural_pairs'] if 'rmsd_angstrom' in r]
    assert all(m['internal_label_gaps']==0 and m['n_to_c'] and m['source_sha256_verified'] for m in mapped)
    assert d['populations']['e012_train']['unique_retained_sequences']==87930
    assert d['sidecar_sequence_projection_mismatches']==0
    assert d['e007_coordinate_acceptance']=={'train':{'accepted':230132,'chain_break':1611},'validation':{'accepted':23161,'chain_break':146}}
    # Panel is illustrative; do not turn hash-ranked sampling into a prevalence estimate.
    summary={'aligned_pair_count':len(pairs),'rmsd_quantiles_angstrom':{str(q):float(np.quantile([r['rmsd_angstrom'] for r in pairs],q)) for q in [0,.25,.5,.75,.9,1]},
      'method_counts_selected_records':dict(collections.Counter(m['method'] for m in mapped)),
      'group_class_counts':dict(collections.Counter(g['classification'] for g in d['structural_panel'])),
      'pairs_with_differing_full_polymers':sum(r.get('full_polymer_equal') is False for r in d['structural_pairs']),
      'maximum_nmr_model_count_in_sources':max(m['model_count'] for m in mapped if m['method']=='SOLUTION NMR'),
      'largest_rmsd_examples':sorted(pairs,key=lambda r:-r['rmsd_angstrom'])[:5]}
    d['structural_panel_summary']=summary;(out/'DUPLICATION_REPORT.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
    lookup={m['sample_id']:m for m in mapped};cropping=[]
    for g in d['structural_panel']:
        ms=[lookup[sid] for sid in g['sample_ids'] if sid in lookup]
        if len({m['full_hash'] for m in ms})>1:
            cropping.append([{k:m[k] for k in ['sample_id','entity_id','full_length','coordinate_length','n_terminal_missing','c_terminal_missing','full_hash','retained_hash','source_path']} for m in ms])
    (out/'SPECIFIC_EXAMPLES.json').write_text(json.dumps({'crop_induced_equivalence_examples':cropping,'correct_order_examples':[{k:m[k] for k in ['sample_id','source_path','label_seq_ids','auth_seq_ids','insertion_codes','n_to_c']} for m in mapped[:3]],'structural_coordinate_difference_examples':summary['largest_rmsd_examples'],'synthetic_parser_defects':'test_unsupported_terminal_scheme_residue_can_disappear; test_legacy_seqres_missing_positions_not_checked; test_legacy_insertion_code_lost_from_returned_ids'},indent=2,sort_keys=True)+'\n')
    section=f'''\n## Specific examples and measured limits\n\nAll {len(mapped)} mapped panel records have strictly increasing label positions and no internal label gaps. {t['denominators']['full_polymer_differs_from_retained']}/{len(mapped)} differ from the full canonical deposited polymer. Panel methods: {summary['method_counts_selected_records']}; these are panel denominators, not archive prevalence. Original NMR source files include up to {summary['maximum_nmr_model_count_in_sources']} models, but historical samples and this comparison use only model 1.\n\nRetained-sequence duplication can mask distinct constructs: `7z53_W` and `9qex_E` both retain 56 residues, while their full polymers contain 59 and 60 residues respectively (3 versus 4 missing C-terminal positions). Their full hashes differ. Such records are not interchangeable full-polymer training examples. `6zk9_b` and `6zkc_b` retain 95/124 full-polymer residues with 28 N-terminal and 1 C-terminal missing positions. These observations are not experimental disorder labels. SPECIFIC_EXAMPLES.json provides authoritative mappings and source paths.\n\nAmong {len(pairs)} aligned pairs, RMSD median is {summary['rmsd_quantiles_angstrom']['0.5']:.3f} A, 90th percentile {summary['rmsd_quantiles_angstrom']['0.9']:.3f} A, and maximum {summary['rmsd_quantiles_angstrom']['1']:.3f} A. `7bl2_9P1` versus `7bl4_9` has RMSD 3.954 A, distance RMSE 2.125 A, and contact Jaccard 0.728 over 338 mapped residues. This demonstrates coordinate variation within an identical full-polymer sequence, but remains inconclusive under the prespecified heterogeneous threshold. Zero heterogeneous classifications do not establish absence of heterogeneous structures in the population.\n\nRaw mmCIF expansion dominates memory: measured census peak RSS was {run['peak_rss_mib']:.1f} MiB ({run['peak_rss_mib']/1024:.2f} GiB); runtime {run['elapsed_seconds']/60:.2f} minutes. Manifest batches are 256 rows and SQLite cache 32 MiB, but these do not cap Gemmi's in-memory parsing of an individual large entry. This is an explicit implementation limitation; a lower hard memory limit would require streaming atom-table extraction or treating oversized entries as inconclusive in a new audit version. No GPU/model inference was used.\n\nHistorical stored TRAIN weights sum to {d['populations']['e012_train']['sum_existing_sample_weights']:.6f} after E006 filtering, because their multiplicities were defined in the earlier structural population. E012 drops those weights during projection and samples records uniformly. A new 1/m weighting must recompute multiplicity within its own frozen population; reusing old structural weights is not equivalent.\n'''
    path=out/'DATASET_AUDIT.md';text=path.read_text();marker='\n## Specific examples and measured limits';text=text.split(marker)[0];path.write_text(text+section)
    # Verify primary outputs, scientific denominators, schema syntax and hash checks.
    required=['DATASET_AUDIT.md','DATASET_LINEAGE.md','DUPLICATION_REPORT.json','TERMINI_AND_ORDER_AUDIT.json','DISORDER_LABEL_EVIDENCE.md','PROPOSED_PDB_ONLY_SCHEMA.json','PROPOSED_EXPERIMENTS.md','test_results.txt']
    for name in required:assert (out/name).stat().st_size>0
    prov=json.loads((out/'PROVENANCE_SHA256.json').read_text())
    bypath={r['path']:r for r in prov['inputs']};protocol=json.loads((SIDE/'protocol.json').read_text())
    import pyarrow.parquet as pq
    for shard in protocol['shards']:
        q=SIDE/shard['path'];entry=bypath.get(str(q))
        if entry is None:
            entry={'path':str(q),'sha256':sha(q),'bytes':q.stat().st_size};prov['inputs'].append(entry);bypath[str(q)]=entry
        entry.update(expected_sha256=shard['sha256'],expected_match=entry['sha256']==shard['sha256'])
        assert entry['expected_match'] and pq.ParquetFile(q).metadata.num_rows==shard['row_count']
    excluded=collections.Counter((r['split'],r['exclusion_reason']) for r in rows(SIDE/'exclusions.parquet',['split','exclusion_reason']))
    expected=collections.Counter({(r['split'],r['reason']):r['records'] for r in d['e006_exclusions'] if not r['eligible']})
    assert excluded==expected
    import yaml
    sealed=yaml.safe_load((REPO/'configs/e007_split_homology_audit_v1.yaml').read_text())['dataset']
    for name,key in [('protocol.json','protocol_sha256'),('schema.json','schema_sha256'),('vocabulary.json','vocabulary_sha256'),('normalization.json','normalization_sha256'),('shard_hashes.sha256','shard_inventory_sha256')]:
        entry=bypath[str(SIDE/name)];entry.update(expected_sha256=sealed[key],expected_match=entry['sha256']==sealed[key]);assert entry['expected_match']
    d['all_e006_protocol_shards_hash_verified']=len(protocol['shards']);d['e006_exclusion_artifacts_match_eligibility_partitions']=True
    (out/'DUPLICATION_REPORT.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
    assert all(r.get('expected_match') is not False for r in prov['inputs'])
    schema=json.loads((out/'PROPOSED_PDB_ONLY_SCHEMA.json').read_text())
    assert schema['$schema']=='https://json-schema.org/draft/2020-12/schema'
    assert set(schema['required'])==set(schema['properties'])
    assert schema['properties']['disorder_evidence']['properties']['status']['enum']==['known','absent','unknown']
    for q in (REPO/'thesis_audit_v1').glob('*.py'):shutil.copyfile(q,out/'reproduce'/q.name)
    prov['audit_code']=[{'path':str(q),'sha256':sha(q)} for q in sorted((REPO/'thesis_audit_v1').glob('*.py'))];(out/'PROVENANCE_SHA256.json').write_text(json.dumps(prov,indent=2,sort_keys=True)+'\n')
    (out/'AUDIT_VALIDATION.json').write_text(json.dumps({'status':'passed','required_outputs':required,'all_mapped_panel_records_n_to_c':True,'all_mapped_panel_records_no_internal_gaps':True,'sidecar_projection_exact_match':True,'schema_checks':'JSON syntax, required/properties agreement, evidence enum verified; full draft meta-schema validation not run (jsonschema unavailable)','historical_counts_reconciled':True,'sidecar_shard_hash_mismatches':0,'tests_passed':18},indent=2)+'\n')
    (out/'OUTPUT_SHA256.json').write_text(json.dumps({str(q.relative_to(out)):sha(q) for q in sorted(out.rglob('*')) if q.is_file() and q.name!='OUTPUT_SHA256.json' and q.suffix!='.sqlite' and 'mplconfig' not in q.parts},indent=2,sort_keys=True)+'\n')
    print('Final review passed',summary)
