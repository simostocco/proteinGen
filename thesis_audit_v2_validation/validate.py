"""Append-only v2 integrity, frequency-weight and clean-reproduction verification."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import argparse,ast,collections,gzip,hashlib,json,resource,shutil,subprocess,sys
import pyarrow as pa
import pyarrow.parquet as pq
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    if p.exists():raise RuntimeError('append-only validation artifact exists: '+str(p))
    p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def main(out):
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    index=json.loads((out/'OUTPUT_SHA256.json').read_text());mismatches=[n for n,h in index.items() if sha(out/n)!=h]
    if mismatches:raise RuntimeError('published v2 checksum mismatch: '+str(mismatches))
    v1=Path('/mnt/d/Simone/proteinGen_data/thesis_pdb_sequence_audit_v1');v1index=json.loads((v1/'OUTPUT_SHA256.json').read_text());assert all(sha(v1/n)==h for n,h in v1index.items())
    counts=json.loads((out/'DATASET_COUNTS_V2.json').read_text());freq=collections.defaultdict(collections.Counter);original=collections.Counter()
    with gzip.open(out/'population_records.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);s=r['split'];original[s]+=1;freq[('retained',s)][r['retained_sha256']]+=1
            if r['full_lm_eligible']:freq[('strict_full_20_500',s)][r['full_sha256']]+=1
    assert dict(original)==counts['population_counts'];assert len(freq[('retained','train')])==87930 and sum(freq[('retained','train')].values())==231743
    for split in original:
        assert len(freq[('strict_full_20_500',split)])==counts['full_lm_unique_by_historical_split'][split]
        assert sum(freq[('strict_full_20_500',split)].values())==counts['counts'][split]['lm_eligible_records']
    weight_path=out/'CURRENT_HISTORICAL_SPLIT_WEIGHTS_V2.parquet';writer=None;batch=[];summaries={}
    for (rep,s),counter in sorted(freq.items()):
        summaries[rep+'_'+s]={'record_denominator':sum(counter.values()),'unique_sequence_denominator':len(counter),'total_inverse_multiplicity_record_weight':len(counter),'historical_E006_weights_used':False}
        for h,m in sorted(counter.items()):
            batch.append({'representation':rep,'historical_split':s,'sequence_sha256':h,'current_multiplicity':m,'current_per_record_unique_sequence_weight':1/m,'current_uniform_record_sequence_probability':m/sum(counter.values())})
            if len(batch)==256:
                table=pa.Table.from_pylist(batch)
                if writer is None:writer=pq.ParquetWriter(weight_path,table.schema)
                writer.write_table(table);batch=[]
    if batch:writer.write_table(pa.Table.from_pylist(batch))
    if writer:writer.close()
    save(out/'CURRENT_WEIGHT_DENOMINATORS_V2.json',summaries)
    # Preserve published snapshots. Correct fresh-run orchestration in a separately versioned bundle.
    clean=out/'reproduce_clean_v2_1';clean.mkdir(exist_ok=False)
    for p in (out/'reproduce').glob('*.py'):shutil.copyfile(p,clean/p.name)
    p=clean/'report.py';text=p.read_text();needle="    reproduce=out/'reproduce';reproduce.mkdir(exist_ok=True)"
    assert needle in text
    text=text.replace(needle,"    from finish import finish\n    finish(out)\n    return\n"+needle)
    p.write_text(text)
    p=clean/'finish.py';text=p.read_text()
    text=text.replace("(REPO/'thesis_audit_v2').glob('*.py')", "Path(__file__).resolve().parent.glob('*.py')")
    text=text.replace("str(REPO/'thesis_audit_v2/test_v2.py')", "str(Path(__file__).resolve().parent/'test_v2.py')")
    text=text.replace("rep=out/'reproduce'", "rep=out/'reproduce';rep.mkdir(exist_ok=True)")
    text=text.replace("'first_log_preserved':'tests/test_results.txt'", "'first_log_preserved':'tests/test_results.txt' if (tests/'test_results.txt').exists() else None")
    text=text.replace("'first_prepare_resource_stop_preserved':True", "'first_prepare_resource_stop_preserved':(out/'PREPARE_RESOURCE_STOP.json').exists()")
    p.write_text(text)
    for p in clean.glob('*.py'):ast.parse(p.read_text())
    env=os.environ.copy();env['TMPDIR']=str(out/'tests')
    log=out/'tests/clean_reproduction_bundle_tests_v2_1.txt'
    with log.open('w') as f:result=subprocess.run([sys.executable,'-B',str(clean/'test_v2.py'),'--root',str(out)],stdout=f,stderr=subprocess.STDOUT,env=env)
    assert result.returncode==0
    readme=out/'REPRODUCE_FINAL_V2.md'
    readme.write_text('''# Clean v2 reproduction entrypoint

Use `reproduce_clean_v2_1/run.py` with the recorded proteingen Python environment and a fresh D: output root. Run `python -B run.py prepare --output <fresh-D-root>`, `python -B run.py search --output <fresh-D-root>`, then `python -B run.py report --output <fresh-D-root>`. The final report stage now creates the immutable-v1 mapping supplement before running actual-example tests, and publishes only after tests pass. It avoids the first orchestration's regression-failure/recovery sequence. No new scientific screening or canonicalization policy was introduced by this workflow correction.

The original successful publication, first failing regression log and all original source snapshots remain immutable. The clean bundle was syntax checked and its 20 focused tests rerun against the published v2 evidence. The entire cache/search pipeline was not launched a second time to test orchestration. Resource bounds, datasets, homology policy, frozen views and existing-prediction estimands are unchanged. Fresh-run timing/stop history naturally differs from this run's preserved attempt history.

CURRENT_HISTORICAL_SPLIT_WEIGHTS_V2.parquet recomputes exact retained and strict-full multiplicities independently within each original split. These weights reproduce uniform unique-sequence exposure within that represented population; no E006 weights are used. They are diagnostics for historical memberships, not a certified new homology-separated TRAIN release. A new split requires new per-split multiplicities. CURRENT_WEIGHT_DENOMINATORS_V2.json gives every denominator. Inverse full-identity weighting of existing retained-fragment predictions is not full-polymer likelihood.

For biological interpretation, read SCIENTIFIC_CORRECTION_PLAN.md together with POLYMER_MAPPING_SUPPLEMENT_V2.md and EVALUATION_EVIDENCE_V2.md. Strict entire-E012 full-polymer validation and TEST release remain blocked. Training and inference remain unauthorized.
''')
    save(out/'APPEND_ONLY_VALIDATION_V2.json',{'status':'passed','original_v2_output_files_hash_checked':len(index),'v1_immutable_output_files_hash_checked':len(v1index),'v1_unchanged':True,'v2_published_artifacts_unchanged':True,'frequency_counts_match_independent_candidate_manifest':True,'clean_reproduction_tests':20,'clean_reproduction_full_pipeline_rerun':False,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'no_training_or_inference':True})
    extra=[weight_path,out/'CURRENT_WEIGHT_DENOMINATORS_V2.json',readme,out/'APPEND_ONLY_VALIDATION_V2.json',log,*clean.glob('*.py')]
    save(out/'ADDITIONAL_OUTPUT_SHA256_V2.json',{str(p.relative_to(out)):sha(p) for p in extra})
    print('Append-only validation passed; v1/v2 unchanged; split weights',summaries,flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);r=a.parse_args();assert str(r.output.resolve()).startswith('/mnt/d/');main(r.output)
