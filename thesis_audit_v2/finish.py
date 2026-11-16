"""Complete publication from frozen v2 views; reuse v1 anchors without new inference."""
from pathlib import Path
import collections,json,os,resource,shutil,subprocess,sys,time
from core import hash_sequence,future_bos_annotation
from run import REPO,HIST,V1,FROZEN,save,sha,jsonrows,gz_writer
def finish(out):
    if (out/'REPORT_COMPLETE.json').exists():raise RuntimeError('v2 publication already complete')
    start=time.time();membership=json.loads((out/'E012_VALIDATION_MEMBERSHIP_V2.json').read_text());frozen=json.loads((out/'FROZEN_PANEL_VIEWS_V2.json').read_text());metrics=json.loads((out/'EXISTING_PREDICTION_REAGGREGATION_V2.json').read_text())
    assert sha(out/'E012_VALIDATION_MEMBERSHIP_V2.json')==frozen['membership_sha256']
    assert sha(out/'PROPOSED_EVALUATION_PROTOCOL_V2.md')==metrics['frozen_protocol_sha256']
    v1_index=json.loads((V1/'OUTPUT_SHA256.json').read_text())
    for n in ['TERMINI_AND_ORDER_AUDIT.json','DATASET_AUDIT.md','DATASET_LINEAGE.md']:
        assert sha(V1/n)==v1_index[n]
    mappings=json.loads((V1/'TERMINI_AND_ORDER_AUDIT.json').read_text())['mapping_examples'];wanted={m['sample_id'] for m in mappings};selected={}
    for r in jsonrows(out/'population_records.jsonl.gz'):
        if r['sample_id'] in wanted:selected[r['sample_id']]=r
    cached=json.loads((out/'tests/actual_problematic_examples.json').read_text());fixtures={};supplement=[]
    for m in mappings:
        if not m.get('source_sha256_verified'):raise ValueError('v1 coordinate anchor lacks verified source identity')
        sid=m['sample_id'];r=selected[sid];full=m['canonical_full_sequence'];labels=m['label_seq_ids'];retained=''.join(full[p-1] for p in labels)
        assert hash_sequence(full)==m['full_hash'] and hash_sequence(retained)==m['retained_hash']==r['retained_sha256']
        assert len(retained)==m['coordinate_length'] and all(a<b for a,b in zip(labels,labels[1:]))
        if r.get('full_sha256') is not None:assert r['full_sha256']==m['full_hash']
        if r.get('source_sha256') is not None:assert r['source_sha256']==m['recorded_source_sha256']
        info={'sample_id':sid,'evidence_source':'immutable v1 exact sequence/CA coordinate-anchor mapping; no raw reparse','source_path':m['source_path'],'source_sha256':m['recorded_source_sha256'],'canonical_complete_sequence':full,'canonical_complete_sha256':m['full_hash'],
             'deposited_one_letter_lexeme':m['deposited_full_sequence'],'historical_retained_sequence':retained,'retained_label_seq_ids':labels,'retained_auth_seq_ids':m['auth_seq_ids'],'retained_insertion_codes':m['insertion_codes'],'entity_id':m['entity_id'],'auth_asym_id':m['auth_asym_id'],'label_asym_id':m['label_asym_id'],
             'retained_correspondence_status':'VERIFIED_V1_AUTHORITATIVE_ANCHOR','positions_before_retained_interval':min(labels)-1,'positions_after_retained_interval':len(full)-max(labels),
             'generic_v2_cache_full_status':r['full_status'],'generic_v2_cache_mapping_status':r['mapping_status'],'full_model_missing_masks':'use cached verified observation where available; otherwise UNKNOWN, not inferred from retained interval',
             'full_v2_search_membership':'unchanged; supplement does not retroactively alter frozen search inputs or LM eligibility',
             'disorder_evidence':{'status':'unknown','evidence':[]},**future_bos_annotation(m['full_hash'],m['retained_hash'],'polymer_N_terminus' if min(labels)==1 else 'fragment_start')}
        supplement.append(info)
        if sid in {'7z53_W','9qex_E','6zk9_b','6zkc_b'}:fixtures[sid]={**info,'generic_v2_cached_observation':cached.get(sid)}
    with gz_writer(out/'V1_AUTHORITATIVE_MAPPING_SUPPLEMENT_V2.jsonl.gz') as f:
        for r in sorted(supplement,key=lambda r:r['sample_id']):f.write((json.dumps(r,separators=(',',':'))+'\n').encode())
    save(out,'tests/actual_examples_evidence_v2.json',fixtures)
    supplement_stats={'v1_authoritative_anchor_records':len(supplement),'generic_cache_full_unknown_but_v1_full_entity_known':sum(r['generic_v2_cache_full_status']=='UNKNOWN' for r in supplement),'generic_cache_retained_mapping_unknown_but_v1_anchor_known':sum(r['generic_v2_cache_mapping_status']=='UNKNOWN' for r in supplement),'source_sha_and_retained_hash_consistent':True,'frozen_search_input_policy_not_amended':True,'counts_not_injected_into_strict_scheme_lm_eligibility':True}
    save(out,'MAPPING_SUPPLEMENT_COUNTS_V2.json',supplement_stats)
    (out/'POLYMER_MAPPING_SUPPLEMENT_V2.md').write_text(f'''# Authoritative v1 mapping supplement — v2

VERIFIED: {len(supplement)} v1 records provide independently checked full-entity strings and sequence/CA-anchor correspondences. v2 reuses their immutable hashes, confirms exact retained identities and source hashes against the generic cache where available, and preserves every mapping in V1_AUTHORITATIVE_MAPPING_SUPPLEMENT_V2.jsonl.gz. No raw structure was reparsed.

The generic historical cache does not contain usable author-linked candidate evidence for every v1 record: {supplement_stats['generic_cache_full_unknown_but_v1_full_entity_known']}/{len(supplement)} have full-status UNKNOWN under v2's strict cached-scheme procedure despite full entity evidence in v1; {supplement_stats['generic_cache_retained_mapping_unknown_but_v1_anchor_known']}/{len(supplement)} have retained-mapping UNKNOWN in that cache but verified retained anchors in v1. These are different evidence scopes, not repaired cache rows. Their direct evidence is retained in a separate linked supplement; it does not retroactively alter frozen search FASTAs, panels or the 90249 strict-scheme LM candidate count.

Actual regression cases: `9qex_E` is missing from the usable generic author-linked cache candidates, while v1 verifies its 60-position full polymer and 56-position retained interval. `7z53_W` has 59 versus56 positions. `6zk9_b` and `6zkc_b` have124 versus95 positions; v1 verifies retained labels29..123, while generic v2 full-mask evidence independently records28 N-terminal and1 C-terminal missing CA positions. No weak or substring mapping is promoted into coordinate correspondence. Retained interval endpoints alone are not treated as an exhaustive full-model missing-coordinate mask when that mask is unavailable.

All future retained views are explicitly tagged fragment_start unless the retained string equals the full deposited canonical string; the separate starts_at_deposited_N_terminus flag distinguishes N-terminal fragments from internal starts. The failed first regression log is preserved as tests/test_results.txt because it exposed an incorrect assumption of identical coverage between the generic cache and v1's authoritative subset. Final regressions use both sources with their proper evidence semantics, in tests/focused_test_results_v2.txt. Generic masks or unseen author IDs remain UNKNOWN where missing.

Approval needed before extending the strict full-polymer target population: adopt a frozen policy for integrating entity-sequence/coordinate-anchor evidence when complete legacy scheme evidence is absent, then re-screen the changed target universe under a new version. Current conditional views remain assessed against their original frozen inputs; the entire-E012 full scope stays UNKNOWN.
''')
    tests=out/'tests';env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(tests));cmd=[sys.executable,'-B',str(REPO/'thesis_audit_v2/test_v2.py'),'--root',str(out)]
    with (tests/'focused_test_results_v2.txt').open('w') as log:run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
    if run.returncode:raise RuntimeError('focused v2 tests failed; no publication marker')
    save(out,'FOCUSED_TEST_RESULTS.json',{'status':'passed','tests':20,'command':cmd,'final_log':'tests/focused_test_results_v2.txt','final_log_sha256':sha(tests/'focused_test_results_v2.txt'),'first_log_preserved':'tests/test_results.txt','actual_examples':sorted(fixtures),'no_training_or_inference':True})
    prov=json.loads((out/'PROVENANCE_V2.json').read_text())
    for p in (REPO/'thesis_audit_v2').glob('*.py'):prov.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'kind':'v2_source_snapshot'})
    for e in metrics['experiments'].values():
        p=Path(e['prediction_path']);assert sha(p)==e['prediction_sha256'];prov.append({'path':str(p),'sha256':e['prediction_sha256'],'bytes':p.stat().st_size,'kind':'reused_existing_per_identity_prediction'})
    protection=json.loads((out/'PRETRAINING_PROTECTION_V2.json').read_text())
    for path,expected in protection['input_hashes'].items():
        p=Path(path);assert sha(p)==expected;prov.append({'path':path,'sha256':expected,'bytes':p.stat().st_size,'kind':'frozen_pretraining_certificate_linkage'})
    for p in [Path(frozen['original_panels_path']),V1/'TERMINI_AND_ORDER_AUDIT.json',V1/'OUTPUT_SHA256.json',out/'population_records.jsonl.gz']:
        prov.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
    save(out,'PROVENANCE_PUBLICATION_V2.json',prov)
    rep=out/'reproduce'
    for p in (REPO/'thesis_audit_v2').glob('*.py'):
        target=rep/p.name
        if target.exists():
            backup=out/'publication_attempt_001_tests_failed/reproduce';backup.mkdir(parents=True,exist_ok=True);target.rename(backup/p.name)
        shutil.copyfile(p,target)
    status=json.loads((FROZEN/'status.json').read_text())
    processes=subprocess.check_output(['ps','-eo','pid,stat,comm,args'],text=True)
    active=[line for line in processes.splitlines() if any(x in line for x in ['train_e007','train_e015','run_e012_causal_rope.py train','run_e012_large_corpus.py train']) and not any(x in line for x in ['codex','run.py','finish.py'])]
    save(out,'SAFETY_AND_RESOURCE_V2.json',{'training_launched':False,'model_inference':False,'folding_designability_evaluation':False,'historical_artifacts_written':False,'raw_mmcif_reparse_count':0,'e012_frozen_state':status['state'],'local_matching_training_processes':active,'scope':'local Linux observation; remote hosts not inspected','prepare_resumed_peak_rss_mib':json.loads((out/'DATASET_COUNTS_V2.json').read_text())['prepare_peak_rss_mib'],'publication_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'first_prepare_resource_stop_preserved':True})
    (out/'REPRODUCE_V2.md').write_text('''# Reproduce v2 data methodology

Use the recorded proteingen Python environment and a new D: root. The source paths are explicit in reproduce/run.py. Run `python -B run.py prepare --output <new-D-root>`, then `python -B run.py search --output <new-D-root>`, then `python -B run.py report --output <new-D-root>`. The first generic-cache regression assumption intentionally remains documented in the preserved first log; run `python -B run.py finalize --output <new-D-root>` to integrate immutable v1 authoritative-anchor evidence and publish the final focused tests. No training or inference occurs. All original source/cache hashes must match. The prepare stage resumes completed v2 observation partitions if a memory-guard stop occurs, preserving the partial population rather than replacing historical data.

The direct v1 supplement does not change strict-scheme eligibility, search membership or previously frozen evaluation views. Any new target universe needs a separately approved policy/version. Tests can be repeated with `python -B test_v2.py --root <published-D-root>`; direct runs print results and do not write historical files. Logs and temporary files must be redirected/configured on D:.
''')
    save(out,'REPORT_COMPLETE.json',{'status':'complete with explicit UNKNOWN scopes','membership_sha256':sha(out/'E012_VALIDATION_MEMBERSHIP_V2.json'),'protocol_sha256':metrics['frozen_protocol_sha256'],'tests_passed':20,'v1_supplement':supplement_stats,'known_retained_validation_relationships':membership['known_retained_mixed_validation'],'known_combined_validation_relationships':membership['known_combined_mixed_validation'],'entire_e012_homology_scope_fully_established':False,'elapsed_seconds':time.time()-start,'no_training_or_inference':True})
    outputs={}
    for p in sorted(out.rglob('*')):
        if not p.is_file() or p.name=='OUTPUT_SHA256.json':continue
        if 'mmseqs_search' in p.parts and 'tmp' in p.parts:continue
        outputs[str(p.relative_to(out))]=sha(p)
    save(out,'OUTPUT_SHA256.json',outputs)
    print('v2 final publication complete',membership['panel_view_counts'],'supplement',supplement_stats,flush=True)
