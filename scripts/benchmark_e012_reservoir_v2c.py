"""D:-only fixed replay benchmarking. Never resumes the scientific source."""
from pathlib import Path
from contextlib import closing
import argparse, copy, hashlib, importlib.util, json, os, shutil, sqlite3, struct, sys, time
import numpy as np
import pyarrow.parquet as pq

ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')
spec=importlib.util.spec_from_file_location('v2c',Path(__file__).with_name('e012_reservoir_v2c.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.configure_storage(ROOT)
AA=m.AA

def inputs(n):
    count=0
    for batch in pq.ParquetFile(ROOT/'replays/replay_1000000.parquet').iter_batches(batch_size=16384):
        for row in batch.to_pylist():
            if count==n:return
            yield row;count+=1
    assert count==n

def copy_baseline(case):
    dest=case/'baseline';dest.mkdir(exist_ok=True)
    for name in ['reservoir.sqlite','reservoir.sqlite-wal']:
        src=ROOT/'preservation'/name;out=dest/name
        if not out.exists():
            partial=out.with_suffix(out.suffix+'.partial')
            shutil.copyfile(src,partial);partial.replace(out)
        assert out.stat().st_size==src.stat().st_size
    return dest/'reservoir.sqlite'

def baseline_db(path):
    db=sqlite3.connect(path)
    db.execute('PRAGMA cache_size=-7340032')
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA wal_autocheckpoint=65536')
    return db

def action_digest(digest,h,off,p,action,source_id):
    digest.update(h+struct.pack('<q',off)+p+action.encode()+b'\0'+source_id.encode()+b'\0')

def run(n,mode):
    assert (ROOT/'continuation_manifest.json').exists(),'31M consistency validation must finish first'
    case=ROOT/'benchmarks'/f'{mode}_{n}';case.mkdir(parents=True,exist_ok=True)
    if (case/'result.json').exists():raise RuntimeError('Completed benchmark already exists')
    state=copy.deepcopy(json.loads((ROOT/'audit/authoritative_state.json').read_text()))
    assert state['raw']==31_000_000
    io=m.Progress(case,state['raw'],n)
    setup_start=time.perf_counter()
    changes=hashlib.sha256();decisions=hashlib.sha256();canonical=novel=duplicates=accepted=queries=0
    new_keys=[];update_events=[];seq_cache={}
    base_sorted=ROOT/'exports/base_priority_sorted.bin'
    if not base_sorted.exists():m.sorted_keys(ROOT/'exports/candidate_keys.bin',base_sorted)
    pool=m.TieredPool(base_sorted,case)
    cutoff=pool.cutoff
    if mode=='v2b':
        path=copy_baseline(case);db=baseline_db(path);base=db;delta=None;membership=None
    else:
        path=ROOT/'audit/reservoir.sqlite'
        base=sqlite3.connect('file:'+path.as_posix()+'?mode=ro',uri=True)
        base.execute('PRAGMA query_only=ON');base.execute('PRAGMA cache_size=-262144')
        delta=m.open_delta(case/'delta.sqlite');db=delta
        bloom=m.Bloom();bloom.seed_file(ROOT/'exports/seen.bin')
        def seq_at(h,off):
            if off in seq_cache:return seq_cache[off]
            row=base.execute('SELECT seq FROM candidates WHERE h=? AND off=?',(h,off)).fetchone()
            if row:return row[0]
            # Exact verification for a discarded representative seeks immutable raw.
            import gzip
            parser_spec=importlib.util.spec_from_file_location('native',Path(__file__).with_name('run_e012_v2b_native.py'))
            native=importlib.util.module_from_spec(parser_spec);parser_spec.loader.exec_module(native)
            legacy=native.load()
            source=ROOT.parent/'raw/uniref50.fasta.gz'
            with source.open('rb',buffering=8*1024**2) as f,gzip.GzipFile(fileobj=f,mode='rb') as gz:
                return next(legacy.fasta_records(gz,off))[2]
        membership=m.ExactMembership(base,delta,bloom,seq_at)
        m.seed_committed_delta(bloom,delta)
    setup_seconds=time.perf_counter()-setup_start
    process_start=time.perf_counter();last_off=None
    payload=(case/'candidate_delta.jsonl').open('w',encoding='utf-8') if mode=='v2c' else None
    # No global cap/cutoff update before the legacy 1M raw boundary.
    for i,row in enumerate(inputs(n),1):
        off=row['off'];seq=row['seq'];sid=row['header'].split()[0];last_off=off
        if row['malformed'] or not 20<=len(seq)<=500 or not set(seq)<=AA:
            io.emit(state['raw']+i,canonical,m.CAP+accepted,queries);continue
        canonical+=1;h=m.sequence_hash(seq);p=m.priority(seq)
        prior=None
        if mode=='v2b':
            queries+=1
            for prior_off, in db.execute('SELECT off FROM exact_hashes WHERE h=?',(h,)):
                queries+=1
                previous=db.execute('SELECT seq FROM candidates WHERE h=? AND off=?',(h,prior_off)).fetchone()
                if previous and previous[0]==seq:prior=prior_off;break
                if not previous:
                    import gzip,types
                    if os.name=='nt':sys.modules.setdefault('resource',types.SimpleNamespace(RUSAGE_SELF=0,RUSAGE_CHILDREN=1,getrusage=lambda *_:types.SimpleNamespace(ru_maxrss=0)))
                    ps=importlib.util.spec_from_file_location('legacy_parser',Path(__file__).with_name('prepare_e012_sequence_corpus_v2b.py'))
                    legacy=importlib.util.module_from_spec(ps);ps.loader.exec_module(legacy)
                    with (ROOT.parent/'raw/uniref50.fasta.gz').open('rb',buffering=8*1024**2) as f,gzip.GzipFile(fileobj=f,mode='rb') as gz:
                        previous=(next(legacy.fasta_records(gz,prior_off))[2],)
                    if previous[0]==seq:prior=prior_off;break
        else:
            prior=membership.find(h,seq);queries=membership.queries
        if prior is not None:
            duplicates+=1
            if mode=='v2b':db.execute('UPDATE candidates SET multiplicity=multiplicity+1,source_id=min(source_id,?) WHERE h=? AND off=?',(sid,h,prior))
            else:
                delta.execute('INSERT INTO updates VALUES(?,?,1,?) ON CONFLICT(h,off) DO UPDATE SET multiplicity=multiplicity+1,source_id=min(source_id,excluded.source_id)',(h,prior,sid))
            update_events.append((h.hex(),prior,sid))
            action_digest(changes,h,prior,p,'duplicate',sid)
        else:
            novel+=1
            if mode=='v2b':db.execute('INSERT INTO exact_hashes VALUES(?,?)',(h,off))
            else:membership.add(h,off,seq)
            eligible=cutoff is None or (p,h,off)<=cutoff
            action_digest(decisions,h,off,p,'accept' if eligible else 'reject',sid)
            action_digest(changes,h,off,p,'novel',sid)
            if eligible:
                accepted+=1
                if mode=='v2b':db.execute('INSERT INTO candidates VALUES(?,?,?,?,?,1,0,1)',(h,off,p,seq,sid))
                else:
                    seq_cache[off]=seq
                    payload.write(json.dumps({'h':h.hex(),'off':off,'p':p.hex(),'seq':seq,'source_id':sid,'multiplicity':1,'historical':0,'external':1})+'\n')
                new_keys.append((p,h,off,0,len(seq)))
        io.emit(state['raw']+i,canonical,m.CAP+accepted,queries)
    processing_seconds=time.perf_counter()-process_start
    if payload:
        payload.flush();os.fsync(payload.fileno());payload.close()
    new=np.array(new_keys,dtype=m.KEY_DTYPE);new.sort(order=['p','h','off'])
    take_base,take_new,boundary=m.partition_prefix(pool.base,new,m.CAP)
    rejection_hash=hashlib.sha256()
    if mode=='v2b':
        excess=accepted
        if excess:
            actual_boundary=db.execute('SELECT p,h,off FROM candidates ORDER BY p DESC,h DESC,off DESC LIMIT 1 OFFSET ?',(excess-1,)).fetchone()
            # The first rejected key is the successor of the exact selected cutoff.
            assert actual_boundary>boundary
            removed=list(db.execute('SELECT p,h,off,rowid FROM candidates WHERE (p,h,off)>=(?,?,?) ORDER BY p,h,off',actual_boundary))
            assert len(removed)==excess
            for p,h,off,_ in removed:rejection_hash.update(p+h+struct.pack('<q',off))
            # Verify the exact rejected set against immutable baseline + accepted delta.
            expected=np.concatenate([pool.base[take_base:],new[take_new:]])
            expected.sort(order=['p','h','off'])
            eh=hashlib.sha256()
            for key in expected:eh.update(m.fixed(key['p'])+m.fixed(key['h'])+struct.pack('<q',int(key['off'])))
            assert eh.hexdigest()==rejection_hash.hexdigest()
            db.executemany('DELETE FROM candidates WHERE rowid=?',((r,) for r in sorted(x[3] for x in removed)))
    else:
        membership.commit_hashes()
        pool.pending=[tuple(x) for x in new]
        pool.select_interval()
        assert pool.base_count==take_base and len(pool.delta)==take_new and pool.cutoff==boundary
        expected=np.concatenate([pool.base[take_base:],new[take_new:]])
        expected.sort(order=['p','h','off'])
        for key in expected:rejection_hash.update(m.fixed(key['p'])+m.fixed(key['h'])+struct.pack('<q',int(key['off'])))
    checkpoint={'raw':state['raw']+n,'offset':last_off,'skip_completed_offset':last_off,
        'stage':'benchmark_only','canonical_delta':canonical,'unique_delta':novel,'duplicates_delta':duplicates,
        'reservoir_count':m.CAP,'source_baseline_raw':state['raw'],'not_scientific_continuation':True}
    if mode=='v2c':
        pool.write_checkpoint(case/'selected_pool',{'baseline_raw':state['raw'],'seed':12014})
        m.commit_continuation(db,case,checkpoint,{'baseline_raw':state['raw'],'seed':12014,'records':n},
            [case/'candidate_delta.jsonl',case/'selected_pool/delta_priority.bin',case/'selected_pool/pool.json'])
        recovered=m.resume_continuation(db,{'baseline_raw':state['raw'],'seed':12014,'records':n})
        assert recovered['state']==checkpoint
    else:
        db.execute("INSERT OR REPLACE INTO progress VALUES('benchmark_checkpoint',?)",(json.dumps(checkpoint),))
        db.commit()
    checkpoint_seconds=time.perf_counter()-process_start-processing_seconds
    result={'mode':mode,'records':n,'canonical':canonical,'novel':novel,'duplicates':duplicates,'accepted_before_trim':accepted,
        'selected_count':m.CAP,'selected_base_prefix':take_base,'selected_delta_prefix':take_new,
        'selected_cutoff':[boundary[0].hex(),boundary[1].hex(),boundary[2]],
        'exact_rejected_keyset_sha256':rejection_hash.hexdigest(),'per_record_changes_sha256':changes.hexdigest(),
        'per_record_priority_decisions_sha256':decisions.hexdigest(),'multiplicity_update_events':update_events,
        'setup_seconds':setup_seconds,'processing_seconds':processing_seconds,'checkpoint_seconds':checkpoint_seconds,
        'raw_per_second_processing':n/processing_seconds,'raw_per_second_including_checkpoint':n/(processing_seconds+checkpoint_seconds),
        'sqlite_membership_queries':queries,'checkpoint':checkpoint,'scientific_continuation_launched':False,'training_launched':False}
    io.emit(state['raw']+n,canonical,m.CAP,queries,force=True)
    db.close()
    if mode!='v2b':base.close()
    pool.close();m.atomic_json(case/'result.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['v2b','v2c']);p.add_argument('records',type=int,choices=[100000,1000000])
    a=p.parse_args();run(a.records,a.mode)
