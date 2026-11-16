"""Recover shutdown evidence, finish missing gates, and reuse the sealed 32M replay.

Run with native Windows Python. Generated evidence and all child cwd/scratch are D:.
Original V2B and replay artifacts are preserved. No training entry point is called.
"""
from pathlib import Path
from contextlib import contextmanager, closing
import argparse, copy, importlib.util, json, os, shutil, sqlite3, subprocess, sys, time

ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')
SCRIPTS=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v2c',SCRIPTS/'e012_reservoir_v2c.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@contextmanager
def read_db(path):
    with closing(sqlite3.connect('file:'+path.as_posix()+'?mode=ro',uri=True)) as db:
        yield db

def validate_replay(case='uninterrupted'):
    work=ROOT/'replay_restart'/case
    print('Validating sealed replay '+case,flush=True)
    with read_db(work/'delta.sqlite') as db:
        cp=json.loads(db.execute("SELECT v FROM progress WHERE k='continuation_checkpoint'").fetchone()[0])
        assert cp==json.loads((work/'continuation_checkpoint.json').read_text())
        for name,meta in cp['outputs'].items():
            p=m.require_d(name)
            assert p.stat().st_size==meta['bytes'] and m.sha_file(p)==meta['sha256'],name
        assert db.execute('PRAGMA quick_check').fetchall()==[('ok',)]
        count=db.execute('SELECT count(*) FROM exact_hashes').fetchone()[0]
        a=json.loads((ROOT/'audit/authoritative_state.json').read_text())
        assert count==cp['state']['unique']-a['unique']==996_217
        assert db.execute('SELECT count(*) FROM updates').fetchone()[0]==0
    state=cp['state'];replay=json.loads((ROOT/'replays/replay_1000000.complete.json').read_text())
    assert state['raw']==32_000_000 and state['stage']=='replay_complete'
    assert state['offset']==state['skip_completed_offset']==replay['last_replayed_offset']
    assert cp['inputs']['replay_sha256']==replay['sha256']==m.sha_file(ROOT/'replays/replay_1000000.parquet')
    assert cp['inputs']['baseline_export_sha256']==m.sha_file(ROOT/'exports/export.complete.json')
    pool=json.loads((Path(state['pool_root'])/'pool.json').read_text())
    assert pool['base_count']+pool['delta_count']==state['reservoir_count']==24_000_000
    return cp

def inspect():
    import psutil
    for p in psutil.process_iter(['pid','cmdline']):
        args=p.info['cmdline'] or []
        if any('python' in str(a).lower() for a in args[:1]):
            try:
                print(json.dumps({'pid':p.pid,'argv':args,'rss':p.memory_info().rss,'io':p.io_counters()._asdict(),'cpu':p.cpu_times()._asdict()}),flush=True)
                spy=ROOT/'tooling/e012_pyspy.exe'
                if p.pid!=os.getpid() and spy.exists() and any(str(a).endswith('recover_e012_v2c.py') for a in args):
                    sample=subprocess.run([str(spy),'dump','--pid',str(p.pid),'--nonblocking'],capture_output=True,text=True,timeout=20)
                    print(sample.stdout,flush=True)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass

def warm_wal():
    import hashlib
    path=ROOT/'benchmarks/v2b_1000000/baseline/reservoir.sqlite-wal'
    digest=hashlib.sha256();start=time.time();last=start;count=0
    with path.open('rb',buffering=8*1024**2) as f:
        while block:=f.read(8*1024**2):
            digest.update(block);count+=len(block)
            if time.time()-last>=30:
                print(json.dumps({'phase':'read_only_wal_warm','bytes_read':count,'bytes_total':path.stat().st_size,'seconds':time.time()-start}),flush=True);last=time.time()
    result={'path':str(path),'bytes':count,'sha256':digest.hexdigest(),'wall_seconds':time.time()-start,'read_only':True}
    m.atomic_json(ROOT/'reports/INTERRUPTED_WAL_SHA256.json',result);print(json.dumps(result),flush=True)

def warm_ahead():
    folder=ROOT/'benchmarks/v2b_1000000/baseline'
    wal=folder/'reservoir.sqlite-wal';shm=folder/'reservoir.sqlite-shm'
    start=time.time();last_start=-1
    with wal.open('rb',buffering=8*1024**2) as f:
        for _ in range(12):
            # WAL-index blocks each describe 4096 frames (first has 34 fewer).
            cursor=max(0,(shm.stat().st_size//32768*4096-34)*4120+32-64*1024**2)
            end=min(wal.stat().st_size,cursor+2*1024**3)
            if cursor<=last_start:time.sleep(2);continue
            f.seek(cursor);remaining=end-cursor
            while remaining:
                block=f.read(min(8*1024**2,remaining));assert block;remaining-=len(block)
            print(json.dumps({'phase':'read_only_wal_read_ahead','from_byte':cursor,'through_byte':end,'elapsed':time.time()-start}),flush=True)
            last_start=cursor
            if end==wal.stat().st_size:break
            time.sleep(1)

def child(stage,args):
    if (ROOT/'STOP_CONTINUATION').exists():raise RuntimeError('STOP_CONTINUATION requested')
    m.atomic_json(ROOT/'reports/RECOVERY_SUPERVISOR_STATUS.json',{'phase':stage,'pid':os.getpid(),'started_unix':time.time(),'training_launched':False})
    start=time.time()
    with (ROOT/'logs'/('recovery_'+stage+'.log')).open('ab') as log:
        command=[sys.executable,'-B',str(SCRIPTS/args[0])]+args[1:]
        if stage=='certify':
            command=['C:/Windows/System32/wsl.exe','--distribution','Ubuntu','--cd',str(ROOT).replace('\\','/').replace('D:','/mnt/d'),
                '--exec','/home/simostocco/miniforge3/envs/proteingen/bin/python','-B',
                '/home/simostocco/proteinGen-causal-rope/scripts/complete_e012_v2c_pipeline.py','certify']
        subprocess.run(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,check=True)
    m.atomic_json(ROOT/'reports'/('recovery_timing_'+stage+'.json'),{'stage':stage,'wall_seconds':time.time()-start,'started_unix':start,'ended_unix':time.time(),'training_launched':False})

def promote(cp):
    work=m.configure_storage(ROOT/'continuation')
    if (work/'delta.sqlite').exists():
        # An existing full-continuation SQL checkpoint always takes precedence.
        with read_db(work/'delta.sqlite') as db:
            row=db.execute("SELECT v FROM progress WHERE k='continuation_checkpoint'").fetchone()
            if row:
                current=json.loads(row[0]);m.resume_continuation(db,current['inputs'])
                assert current['state']['raw']>=cp['state']['raw']
                return
        raise RuntimeError('Uncertified continuation DB exists; preserve for inspection')
    gen=work/'generations'/('promotion_32000000_'+str(time.time_ns()));gen.mkdir(parents=True)
    state=copy.deepcopy(cp['state']);outputs=[]
    for old in cp['outputs']:
        src=Path(old)
        if src.name=='candidates.jsonl':dest=gen/src.parent.name/src.name
        else:dest=gen/'pool'/src.name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        with dest.open('r+b') as handle:handle.flush();os.fsync(handle.fileno())
        assert m.sha_file(dest)==cp['outputs'][old]['sha256'];outputs.append(dest)
    state['pool_root']=str(gen/'pool');state['stage']='external'
    inputs={'baseline_raw':31_000_000,'seed':12014,'gate_sha256':m.sha_file(ROOT/'V2C_VERDICT.json'),
        'baseline_export_sha256':m.sha_file(ROOT/'exports/export.complete.json')}
    # Backup includes every committed dedup/multiplicity row, without modifying replay.
    partial=work/'delta.sqlite.partial'
    with read_db(ROOT/'replay_restart/uninterrupted/delta.sqlite') as src,closing(sqlite3.connect(partial)) as dst:src.backup(dst)
    db=m.open_delta(partial)
    try:
        m.commit_continuation(db,work,state,inputs,outputs)
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:db.close()
    os.replace(partial,work/'delta.sqlite')
    m.atomic_json(work/'PROMOTED_32M_REPLAY.json',{'source_checkpoint_sha256':m.sha_file(ROOT/'replay_restart/uninterrupted/continuation_checkpoint.json'),
        'checkpoint_sha256':m.sha_file(work/'continuation_checkpoint.json'),'raw':state['raw'],'offset':state['offset'],
        'skip_completed_offset':state['skip_completed_offset'],'replayed_committed_records':0,'training_launched':False})

def run():
    # A sealed full-source boundary outranks all replay/checkpoint baselines.
    full=ROOT/'continuation/EXTERNAL_TAIL_COMPLETE.json'
    if full.exists():
        gate=json.loads((ROOT/'V2C_VERDICT.json').read_text())
        assert gate['verdict']=='V2C-GO' and all(gate['gates'].values())
        assert gate['report_sha256']==m.sha_file(ROOT/'reports/BENCHMARK_EQUIVALENCE.json')
        assert gate['restart_report_sha256']==m.sha_file(ROOT/'reports/RESTART_REPLAY.json')
        assert gate['tests_sha256']==m.sha_file(ROOT/'reports/TESTS.json')
        with read_db(ROOT/'continuation/delta.sqlite') as db:
            row=db.execute("SELECT v FROM progress WHERE k='continuation_checkpoint'").fetchone();assert row
            saved=json.loads(row[0]);cp=m.resume_continuation(db,saved['inputs'])
            assert db.execute('PRAGMA quick_check').fetchall()==[('ok',)]
            assert cp['state']['raw']==m.RAW_COUNT and cp['state']['stage']=='historical'
            baseline=json.loads((ROOT/'audit/authoritative_state.json').read_text())
            assert db.execute('SELECT count(*) FROM exact_hashes').fetchone()[0]==cp['state']['unique']-baseline['unique']
        assert cp==json.loads((ROOT/'continuation/continuation_checkpoint.json').read_text())
        cert=json.loads(full.read_text());assert cert['state']==cp['state']
        assert cert['checkpoint_sha256']==m.sha_file(ROOT/'continuation/continuation_checkpoint.json')
        m.atomic_json(ROOT/'reports/COMPLETED_SOURCE_RECOVERY.json',{'raw':m.RAW_COUNT,'stage':'historical','source_rescan':False,
            'checkpoint_sha256':cert['checkpoint_sha256'],'remaining_source_records':0,'training_launched':False})
        downstream()
        return
    cp=validate_replay()
    m.atomic_json(ROOT/'reports/SHUTDOWN_RECOVERY.json',{'head':subprocess.check_output(['C:/Windows/System32/wsl.exe','--distribution','Ubuntu','--cd','/home/simostocco/proteinGen-causal-rope','--exec','git','rev-parse','HEAD'],text=True).strip(),
        'selected_checkpoint':str(ROOT/'replay_restart/uninterrupted/delta.sqlite'),'highest_valid_v2c_raw':32_000_000,
        'remaining_source_records':m.RAW_COUNT-32_000_000,'full_continuation_previously_started':(ROOT/'continuation').exists(),
        'recovery_source_sha256':{name:m.sha_file(SCRIPTS/name) for name in ['recover_e012_v2c.py','recover_committed_e012_v2c_benchmark.py','certify_e012_v2c_gate.py','report_benchmarks_e012_v2c.py']},
        'modified_tracked_reports_preserved':True,'training_launched':False})
    baseline=ROOT/'benchmarks/v2b_1000000'
    if not (baseline/'result.json').exists():
        # Phase-1 native SQL inspection established the committed 32M benchmark.
        # The child independently verifies it; do not reopen/rebuild the 15GB WAL
        # in this parent as well. A missing commit fails closed without rerunning.
        child('recover_committed_v2b_1m',['recover_committed_e012_v2c_benchmark.py'])
    child('nine_gates',['certify_e012_v2c_gate.py'])
    gate=json.loads((ROOT/'V2C_VERDICT.json').read_text());assert gate['verdict']=='V2C-GO' and all(gate['gates'].values())
    spec=importlib.util.spec_from_file_location('supervisor',SCRIPTS/'supervise_e012_v2c.py')
    supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
    supervisor.cleanup_certified_clones()
    promote(cp)
    child('external',['continue_e012_reservoir_v2c.py'])
    downstream()

def downstream():
    child('historical_union',['union_e012_reservoir_v2c.py'])
    for stage in ['reservoir-finalize','screen','verify-protected','final','certify','diversity','handoff']:
        child(stage,['complete_e012_v2c_pipeline.py',stage])
    m.atomic_json(ROOT/'reports/RECOVERY_SUPERVISOR_STATUS.json',{'phase':'frozen_corpus_ready','training_launched':False,'ended_unix':time.time()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inspect','run','validate','warm-wal','warm-ahead','stop-stale-reader']);p.add_argument('--detach',action='store_true');a=p.parse_args()
    assert os.name=='nt','Use native Windows Python for preserved Windows checkpoint paths'
    m.configure_storage(ROOT)
    if a.action=='inspect':inspect()
    elif a.action=='stop-stale-reader':
        import psutil
        proc=psutil.Process(26756);args=proc.cmdline()
        assert any(str(x).endswith('recover_e012_v2c.py') for x in args) and args[-1]=='run'
        assert not proc.children(),'Refuse to interrupt a validation or continuation child'
        proc.terminate();proc.wait(20);print('Stopped only redundant pre-child reader PID 26756',flush=True)
    elif a.action=='warm-wal':warm_wal()
    elif a.action=='warm-ahead':warm_ahead()
    elif a.action=='validate':validate_replay();print('32M sealed checkpoint validated',flush=True)
    elif a.detach:
        with (ROOT/'logs/recovery_supervisor.log').open('ab') as log:
            proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'run'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
        print(json.dumps({'recovery_supervisor_pid':proc.pid,'root':str(ROOT),'training_launched':False}),flush=True)
    else:
        try:run()
        except BaseException as exc:
            m.atomic_json(ROOT/'reports/RECOVERY_SUPERVISOR_STATUS.json',{'phase':'blocked','error':repr(exc),'timestamp_unix':time.time(),'training_launched':False})
            raise
