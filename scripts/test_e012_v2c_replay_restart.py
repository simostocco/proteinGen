"""Actual D:-only 1M replay interruption/resume and exact metadata comparison."""
from pathlib import Path
import copy,importlib.util,itertools,json,os,sqlite3,subprocess,sys,time
import numpy as np
import pyarrow.parquet as pq
spec=importlib.util.spec_from_file_location('v2c',Path(__file__).with_name('e012_reservoir_v2c.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization');m.configure_storage(ROOT)

def checkpoint(case):
    with sqlite3.connect(ROOT/'replay_restart'/case/'delta.sqlite') as db:
        return json.loads(db.execute("SELECT v FROM progress WHERE k='continuation_checkpoint'").fetchone()[0])

def payloads(cp):
    for path in sorted(cp['outputs']):
        if Path(path).name=='candidates.jsonl':
            with m.require_d(path).open() as f:
                for line in f:yield json.loads(line)

def main():
    start=time.time();runner=Path(__file__).with_name('continue_e012_reservoir_v2c.py')
    while not (ROOT/'replay_restart/uninterrupted/REPLAY_COMPLETE.json').exists():time.sleep(30)
    resumed=ROOT/'replay_restart/resumed';resumed.mkdir(exist_ok=True)
    attempts=[]
    for extra,expected in [(['--crash-at','31400000'],73),(['--crash-before-at','31450000'],74),([],0)]:
        t=time.time()
        with (ROOT/'logs/restart_replay.log').open('ab') as log:
            completed=subprocess.run([sys.executable,'-B',str(runner),'--replay-case','resumed']+extra,cwd=ROOT,stdout=log,stderr=log)
        attempts.append({'extra':extra,'exit':completed.returncode,'wall_seconds':time.time()-t})
        assert completed.returncode==expected,attempts
        if extra:
            cp=checkpoint('resumed');assert cp['state']['raw']==31_400_000
            mirror=json.loads((resumed/'continuation_checkpoint.json').read_text())
            assert mirror['state']['raw']==31_300_000,'JSON mirror must retain deliberate 100k lag after injected post-commit crash'
    a=checkpoint('uninterrupted');b=checkpoint('resumed')
    fields=['raw','offset','skip_completed_offset','canonical','length_valid','malformed','unique','duplicates','raw_stats','valid_stats','historical_matches','reservoir_count','stage','admission_cutoff']
    same={k:a['state'][k]==b['state'][k] for k in fields};assert all(same.values()),same
    assert a['state']['raw']==32_000_000
    pa=json.loads((Path(a['state']['pool_root'])/'pool.json').read_text());pb=json.loads((Path(b['state']['pool_root'])/'pool.json').read_text())
    assert (pa['base_count'],pa['delta_count'],pa['cutoff'])==(pb['base_count'],pb['delta_count'],pb['cutoff'])
    f52=ROOT/'benchmarks/v2c_1000000';original=json.loads((f52/'result.json').read_text())
    assert pa['base_count']==original['selected_base_prefix'] and pa['delta_count']==original['selected_delta_prefix'] and pa['cutoff']==original['selected_cutoff']
    arrays=[np.memmap(Path(cp['state']['pool_root'])/'delta_priority.bin',dtype=m.KEY_DTYPE,mode='r') for cp in [a,b]]
    reference=np.memmap(f52/'selected_pool/delta_priority.bin',dtype=m.KEY_DTYPE,mode='r')
    for data in arrays:
        assert len(data)==len(reference)
        for field in ['p','h','off','length']:assert np.array_equal(data[field],reference[field]),field
    count=0
    with (f52/'candidate_delta.jsonl').open() as f:
        for rows in itertools.zip_longest(payloads(a),payloads(b),(json.loads(line) for line in f)):
            assert all(row is not None for row in rows) and rows[0]==rows[1]==rows[2],f'Exact payload discrepancy at {count}'
            count+=1
    assert count==original['accepted_before_trim']
    # Independent historical V2B bookkeeping reference on the same frozen interval.
    ns=importlib.util.spec_from_file_location('native',Path(__file__).with_name('run_e012_v2b_native.py'))
    native=importlib.util.module_from_spec(ns);ns.loader.exec_module(native);legacy=native.load()
    old=copy.deepcopy(json.loads((ROOT/'audit/authoritative_state.json').read_text()))
    hist={row['sequence'] for batch in pq.ParquetFile(m.historical_train_file(ROOT,legacy.REPO)).iter_batches(columns=['sequence']) for row in batch.to_pylist()}
    rs,vs=legacy.Stats(old['raw_stats']),legacy.Stats(old['valid_stats'])
    for batch in pq.ParquetFile(ROOT/'replays/replay_1000000.parquet').iter_batches():
        for row in batch.to_pylist():
            seq=row['seq'];old['raw']+=1;rs.add(seq)
            if row['malformed']:old['malformed']+=1
            elif 20<=len(seq)<=500:
                old['length_valid']+=1
                if set(seq)<=m.AA:
                    old['canonical']+=1;vs.add(seq)
                    if seq in hist:
                        item=old['historical_matches'].setdefault(seq,[row['off'],row['header'].split()[0],0]);item[1]=min(item[1],row['header'].split()[0]);item[2]+=1
    old['unique']+=original['novel'];old['duplicates']+=original['duplicates'];old['raw_stats']=rs.checkpoint();old['valid_stats']=vs.checkpoint()
    # JSON normalizes integer histogram keys, as both durable implementations do.
    old=json.loads(json.dumps(old))
    for k in ['raw','canonical','length_valid','malformed','unique','duplicates','raw_stats','valid_stats','historical_matches']:assert old[k]==a['state'][k],k
    report={'passed':True,'records':1_000_000,'baseline_raw':31_000_000,'frozen_replay_sha256':m.sha_file(ROOT/'replays/replay_1000000.parquet'),
        'exact_selected_keys_match_f52':True,'all_payload_fields_equal':True,'payloads_compared':count,
        'historical_bookkeeping_equal_to_v2b':True,'state_fields_equal':same,'attempts':attempts,
        'post_commit_pre_mirror_recovery':True,'uncommitted_interval_recovery':True,
        'uninterrupted_end_mtime':(ROOT/'replay_restart/uninterrupted/REPLAY_COMPLETE.json').stat().st_mtime,
        'uninterrupted_scan_seconds':a['state']['scan_seconds'],'wall_seconds':time.time()-start,'training_launched':False}
    m.atomic_json(ROOT/'reports/RESTART_REPLAY.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
