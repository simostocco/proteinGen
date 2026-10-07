"""Extract fixed 100k/1M replay inputs after the committed 31M cursor, on D: only."""
from pathlib import Path
import gzip, hashlib, importlib.util, json, os, time, sys, types
import pyarrow as pa
import pyarrow.parquet as pq

BASE=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2')
ROOT=BASE/'v2c_optimization'
for key in ['TMP','TEMP','TMPDIR','XDG_CACHE_HOME']:os.environ[key]=str(ROOT/'tmp')
import tempfile
tempfile.tempdir=str(ROOT/'tmp')
spec=importlib.util.spec_from_file_location('legacy',Path(__file__).with_name('prepare_e012_sequence_corpus_v2b.py'))
if os.name=='nt':
    # The replay extractor uses only the unchanged FASTA parser, not resource APIs.
    sys.modules.setdefault('resource',types.SimpleNamespace(RUSAGE_SELF=0,RUSAGE_CHILDREN=1,
        getrusage=lambda *_:types.SimpleNamespace(ru_maxrss=0)))
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)

def main():
    root=ROOT/'replays';root.mkdir(exist_ok=True)
    complete=root/'replay_1000000.complete.json'
    if complete.exists():raise RuntimeError('Verified replay already exists')
    state=json.loads((ROOT/'audit/authoritative_state.json').read_text())
    assert state['raw']==31_000_000 and state['offset']==15_459_089_980
    path=root/'replay_1000000.parquet.partial'
    schema=pa.schema([('off',pa.int64()),('header',pa.string()),('seq',pa.string()),('malformed',pa.bool_())])
    count=0;batch=[];canonical=0;start=time.perf_counter();first=None;last=None
    with (BASE/'raw/uniref50.fasta.gz').open('rb',buffering=8*1024**2) as source,\
         gzip.GzipFile(fileobj=source,mode='rb') as reader,\
         pq.ParquetWriter(path,schema,compression='zstd') as writer:
        for off,header,seq,malformed in legacy.fasta_records(reader,state['offset']):
            if off==state['skip_completed_offset']:continue
            if first is None:first=off
            batch.append({'off':off,'header':header,'seq':seq,'malformed':malformed})
            canonical+=int(not malformed and 20<=len(seq)<=500 and set(seq)<=legacy.ALPHABET)
            count+=1;last=off
            if len(batch)==16384:
                writer.write_table(pa.Table.from_pylist(batch,schema=schema));batch=[]
            if count%100_000==0:
                elapsed=time.perf_counter()-start
                progress={'stage':'replay_extraction','raw':state['raw']+count,
                    'percent':100*(state['raw']+count)/legacy.RAW_COUNT,'records':count,
                    'raw_per_second':count/elapsed,'canonical_per_second':canonical/elapsed,
                    'elapsed_seconds':elapsed,'eta_seconds':(1_000_000-count)*elapsed/count,
                    'scientific_continuation':False,'training_launched':False}
                print(json.dumps(progress),flush=True)
                with (ROOT/'logs/replay_extraction.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(progress)+'\n')
            if count==1_000_000:break
        if batch:writer.write_table(pa.Table.from_pylist(batch,schema=schema))
    assert count==1_000_000
    target=path.with_suffix('');path.replace(target)
    manifest={'baseline_raw':state['raw'],'baseline_offset':state['offset'],'skip_completed_offset':state['skip_completed_offset'],
        'first_replayed_offset':first,'last_replayed_offset':last,'records':count,'canonical':canonical,
        'sha256':legacy.digest(target),'bytes':target.stat().st_size,'wall_seconds':time.perf_counter()-start,
        'cases':[100_000,1_000_000],'scientific_continuation_launched':False,'training_launched':False}
    legacy.save(complete,manifest)
    profile='''# V2B per-record database profile and V2C redesign

The original DB/WAL/SHM and 30M JSON mirror are preserved byte-for-byte on D:.
SQLite commits the dedup index, reservoir and progress cursor before saving its
JSON mirror. The committed SQL cursor is 31M; the mirror is 30M. V2C uses only the
validated committed 31M baseline and never replays the 30M-to-31M interval.

For each canonical record V2B computes SHA256(sequence) and queries
SELECT off FROM exact_hashes WHERE h=?. On a positive hash it compares the actual
sequence, via candidate lookup or a seek into the immutable raw gzip. For a novel
sequence it inserts into exact_hashes and, when its frozen priority is eligible,
into candidates. These insertions maintain the seen-hash B-tree, candidate primary
key index and priority index. Duplicates update multiplicity and lexicographic
representative ID. At a 1M raw boundary it trims to 24M, commits SQLite, then saves
the JSON mirror. Indexed reads/writes are typically 4KB random operations; observed
CPU utilization was very low. A nonblocking stack sample located the continuation
in candidate insertion and later cap trimming.

V2C uses a 128MiB Bloom filter seeded from every committed seen hash. A definite
negative skips the SQLite lookup. Positives always perform full-string exact
verification. New hashes immediately enter the Bloom/current-interval map;
abandoned/uncommitted Bloom entries can only create false positives, never false
duplicates. A small delta index is bulk-written in sorted hash order.

The large baseline is immutable. Candidate payloads append sequentially on D:.
Compact priority keys are sorted in bounded batches; exact binary partitioning of
the sorted base/delta determines the same (priority, hash, offset) prefix. The
large baseline is never updated for each tail record. This removes remaining
per-record maintenance of large random B-trees, rather than only skipping reads.

All generated files, logs, checkpoints, caches, replays and test fixtures are on D:.
There is no RAM filesystem or scratch on C:, Linux home, /tmp or /var/tmp.
Scientific continuation is gated on 100k and 1M equivalence/resume benchmarks and
at least 5x measured speedup. Setup/export/seed time is reported separately.
No optimizer, model evaluation, training or generation is launched.
'''
    (ROOT/'reports/OPERATION_PROFILE.md').write_text(profile,encoding='utf-8')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
