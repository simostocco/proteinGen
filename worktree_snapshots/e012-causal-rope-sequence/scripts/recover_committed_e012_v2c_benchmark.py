"""Independently certify the committed V2B 1M replay whose result mirror was lost.

No source records or SQL mutations are replayed. Full selected keys and dedup keys
are compared exactly; full payload fingerprints compare every retained row field.
Timing is an explicit pre-shutdown lower bound, never including downtime.
"""
from pathlib import Path
from contextlib import closing
import hashlib, importlib.util, json, os, sqlite3, struct, time
import numpy as np
import pyarrow.parquet as pq

ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')
SCRIPTS=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v2c',SCRIPTS/'e012_reservoir_v2c.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.configure_storage(ROOT)
CASE=ROOT/'benchmarks/v2b_1000000'

def fingerprint(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).digest()

def compare_merge(cursor,base,delta,fields,total,label):
    i=j=n=0;start=last=time.time()
    for batch in iter(lambda:cursor.fetchmany(100_000),[]):
        if fields==['h','off']:
            actual=np.array(batch,dtype=m.SEEN_DTYPE)
        else:actual=np.array([(p,h,off,0,length) for p,h,off,length in batch],dtype=m.KEY_DTYPE)
        a=actual[fields];b=base[fields];d=delta[fields]
        stop_b=int(np.searchsorted(b,a[-1],side='right'));stop_d=int(np.searchsorted(d,a[-1],side='right'))
        expected=np.concatenate([b[i:stop_b],d[j:stop_d]])
        expected.sort(order=fields)
        assert len(expected)==len(a) and np.array_equal(expected,a),(label,n)
        i,j=stop_b,stop_d;n+=len(a)
        if time.time()-last>=60:
            print(json.dumps({'phase':label,'verified_rows':n,'target':total,'seconds':time.time()-start}),flush=True);last=time.time()
    assert n==total and i==len(base) and j==len(delta),(label,n,i,j)
    print(json.dumps({'phase':label,'verified_rows':n,'passed':True}),flush=True)

def main():
    if (CASE/'result.json').exists():return
    evidence=m.configure_storage(CASE/'shutdown_recovery')
    path=CASE/'baseline/reservoir.sqlite'
    # This is the disposable private benchmark clone. A writable connection lets
    # SQLite publish its rebuilt WAL index and checkpoint on close; query_only
    # prohibits scientific row changes. Original V2B files are never opened here.
    with closing(sqlite3.connect(path)) as db:
        db.execute('PRAGMA query_only=ON');db.execute('PRAGMA cache_size=-131072')
        checkpoint=json.loads(db.execute("SELECT v FROM progress WHERE k='benchmark_checkpoint'").fetchone()[0])
        print('Committed 32M benchmark SQL checkpoint recovered; comparing frozen replay',flush=True)
        assert checkpoint['raw']==32_000_000 and checkpoint['reservoir_count']==m.CAP
        assert checkpoint['unique_delta']==checkpoint['canonical_delta']==996_217 and checkpoint['duplicates_delta']==0
        baseline=json.loads((ROOT/'audit/authoritative_state.json').read_text())
        assert json.loads(db.execute("SELECT v FROM progress WHERE k='checkpoint'").fetchone()[0])==baseline
        base_seen=np.memmap(ROOT/'exports/seen_hash_sorted.bin',dtype=m.SEEN_DTYPE,mode='r')
        base=np.memmap(ROOT/'exports/base_priority_sorted.bin',dtype=m.KEY_DTYPE,mode='r')
        cutoff=m.key_tuple(base[-1]);changes=hashlib.sha256();decisions=hashlib.sha256()
        seen=[];accepted=[];canonical=0;last_off=None
        for batch in pq.ParquetFile(ROOT/'replays/replay_1000000.parquet').iter_batches():
            for row in batch.to_pylist():
                seq=row['seq'];off=row['off'];last_off=off
                if row['malformed'] or not 20<=len(seq)<=500 or not set(seq)<=m.AA:continue
                canonical+=1;h=m.sequence_hash(seq);p=m.priority(seq);sid=row['header'].split()[0]
                seen.append((h,off));eligible=(p,h,off)<=cutoff
                decisions.update(h+struct.pack('<q',off)+p+('accept' if eligible else 'reject').encode()+b'\0'+sid.encode()+b'\0')
                changes.update(h+struct.pack('<q',off)+p+b'novel\0'+sid.encode()+b'\0')
                if eligible:accepted.append((p,h,off,0,len(seq)))
        assert last_off==checkpoint['offset']==checkpoint['skip_completed_offset']
        novel=np.array(seen,dtype=m.SEEN_DTYPE);novel.sort(order=['h','off']);del seen
        assert canonical==len(novel)==996_217
        assert not np.any(novel['h'][1:]==novel['h'][:-1])
        positions=np.searchsorted(base_seen['h'],novel['h']);valid=positions<len(base_seen)
        assert not np.any(base_seen['h'][positions[valid]]==novel['h'][valid]),'Replay includes a baseline hash; require full collision recovery'
        new=np.array(accepted,dtype=m.KEY_DTYPE);del accepted;new.sort(order=['p','h','off'])
        take_base,take_new,boundary=m.partition_prefix(base,new,m.CAP)
        compare_merge(db.execute('SELECT h,off FROM exact_hashes ORDER BY h,off'),base_seen,novel,['h','off'],baseline['unique']+len(novel),'dedup_keys')
        base_seen._mmap.close();del base_seen,novel
        compare_merge(db.execute('SELECT p,h,off,length(seq) FROM candidates ORDER BY p,h,off'),base[:take_base],new[:take_new],['p','h','off','length'],m.CAP,'selected_keys')
        rejected=np.concatenate([base[take_base:],new[take_new:]]);rejected.sort(order=['p','h','off'])
        rejection=hashlib.sha256()
        for key in rejected:rejection.update(m.fixed(key['p'])+m.fixed(key['h'])+struct.pack('<q',int(key['off'])))
        accepted_count=len(new);del rejected,new;base._mmap.close();del base
        union_spec=importlib.util.spec_from_file_location('union',SCRIPTS/'union_e012_reservoir_v2c.py')
        union=importlib.util.module_from_spec(union_spec);union_spec.loader.exec_module(union)
        expected_file=evidence/'expected_payload_fingerprints.bin';actual_file=evidence/'actual_payload_fingerprints.bin'
        def selected(row):return (bytes.fromhex(row['p']),bytes.fromhex(row['h']),row['off'])<=boundary
        n=0;last=time.time()
        with expected_file.open('wb',buffering=8*1024**2) as out:
            for row in union.base_rows(ROOT/'exports/candidate_payload.bin'):
                if selected(row):out.write(fingerprint(row));n+=1
                if time.time()-last>=60:print(json.dumps({'phase':'expected_payloads','rows':n}),flush=True);last=time.time()
            for batch in pq.ParquetFile(ROOT/'replays/replay_1000000.parquet').iter_batches():
                for row in batch.to_pylist():
                    seq=row['seq']
                    if row['malformed'] or not 20<=len(seq)<=500 or not set(seq)<=m.AA:continue
                    item={'h':m.sequence_hash(seq).hex(),'off':row['off'],'p':m.priority(seq).hex(),'seq':seq,
                        'source_id':row['header'].split()[0],'multiplicity':1,'historical':0,'external':1}
                    if selected(item):out.write(fingerprint(item));n+=1
            out.flush();os.fsync(out.fileno())
        assert n==m.CAP
        n=0;last=time.time()
        with actual_file.open('wb',buffering=8*1024**2) as out:
            for h,off,p,seq,sid,mult,hist,ext in db.execute('SELECT h,off,p,seq,source_id,multiplicity,historical,external FROM candidates NOT INDEXED'):
                assert m.sequence_hash(seq)==h and m.priority(seq)==p
                item={'h':h.hex(),'off':off,'p':p.hex(),'seq':seq,'source_id':sid,'multiplicity':mult,'historical':hist,'external':ext}
                out.write(fingerprint(item));n+=1
                if time.time()-last>=60:print(json.dumps({'phase':'actual_payloads','rows':n}),flush=True);last=time.time()
            out.flush();os.fsync(out.fileno())
        assert n==m.CAP
        arrays=[np.memmap(p,dtype='S32',mode='r+') for p in [expected_file,actual_file]]
        for data in arrays:data.sort();data.flush()
        for start in range(0,m.CAP,100_000):assert np.array_equal(arrays[0][start:start+100_000],arrays[1][start:start+100_000]),start
        for data in arrays:data._mmap.close()
        rows=[json.loads(line) for line in (CASE/'progress.jsonl').read_text().splitlines()]
        first,last=rows[0],rows[-1];assert first['raw']==31_000_001 and last['raw']==32_000_000
        processing_lower_bound=last['elapsed_seconds']-first['elapsed_seconds']
        certificate={'committed_checkpoint':checkpoint,'all_dedup_keys_equal':True,'all_selected_keys_equal':True,
            'all_payload_fields_equal':True,'payload_count':m.CAP,'payload_comparison':'Sorted SHA256 of canonical complete row; exact key comparisons separately',
            'expected_payload_sha256':m.sha_file(expected_file),'actual_payload_sha256':m.sha_file(actual_file),
            'processing_seconds_lower_bound':processing_lower_bound,'end_to_end_seconds_lower_bound':last['elapsed_seconds'],
            'shutdown_downtime_counted':False,'source_scan_repeated':False,'sql_read_only':True,'training_launched':False}
        m.atomic_json(evidence/'COMMITTED_BENCHMARK_RECOVERED.json',certificate)
        result={'mode':'v2b','records':1_000_000,'canonical':canonical,'novel':canonical,'duplicates':0,'accepted_before_trim':accepted_count,
            'selected_count':m.CAP,'selected_base_prefix':take_base,'selected_delta_prefix':take_new,
            'selected_cutoff':[boundary[0].hex(),boundary[1].hex(),boundary[2]],'exact_rejected_keyset_sha256':rejection.hexdigest(),
            'per_record_changes_sha256':changes.hexdigest(),'per_record_priority_decisions_sha256':decisions.hexdigest(),
            'multiplicity_update_events':[],'setup_seconds':first['elapsed_seconds'],'processing_seconds':processing_lower_bound,'checkpoint_seconds':0,
            'end_to_end_seconds':last['elapsed_seconds'],'timing_is_lower_bound':True,'checkpoint_seconds_not_recoverable':True,
            'raw_per_second_processing':1_000_000/processing_lower_bound,'raw_per_second_including_checkpoint':1_000_000/processing_lower_bound,
            'sqlite_membership_queries':last['database_queries'],'checkpoint':checkpoint,'scientific_continuation_launched':False,'training_launched':False,
            'recovery_certificate_sha256':m.sha_file(evidence/'COMMITTED_BENCHMARK_RECOVERED.json')}
        m.atomic_json(CASE/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
