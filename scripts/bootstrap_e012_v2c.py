"""Freeze and inspect E012 V2B without changing original corpus files. All output D:."""
from pathlib import Path
from contextlib import closing
import argparse, hashlib, json, os, shutil, sqlite3, time, sys

ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2')
OUT=ROOT/'v2c_optimization'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb',buffering=8*1024**2) as f:
        while block:=f.read(8*1024**2): h.update(block)
    return h.hexdigest()

def write_json(path,value):
    partial=path.with_suffix(path.suffix+'.partial')
    with partial.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    partial.replace(path)

def main():
    for name in ['preservation','audit','tmp','logs','benchmarks','tooling','reports']:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    for key in ['TEMP','TMP','TMPDIR']:
        os.environ[key]=str(OUT/'tmp')
    import tempfile
    tempfile.tempdir=str(OUT/'tmp')
    import psutil
    if '--inspect-bench' in sys.argv:
        for proc in psutil.process_iter(['pid','cmdline']):
            args=proc.info['cmdline'] or []
            if any(x.endswith('benchmark_e012_reservoir_v2c.py') for x in args):
                first=proc.io_counters();cpu=proc.cpu_times();time.sleep(5)
                now=proc.io_counters();cpu2=proc.cpu_times()
                print(json.dumps({'pid':proc.pid,'argv':args,'rss':proc.memory_info().rss,
                    'available_ram':psutil.virtual_memory().available,'read_5s':now.read_bytes-first.read_bytes,
                    'write_5s':now.write_bytes-first.write_bytes,'cpu_5s':cpu2.user+cpu2.system-cpu.user-cpu.system}),flush=True)
                exe=OUT/'tooling/e012_pyspy.exe'
                if exe.exists():
                    import subprocess
                    sampled=subprocess.run([str(exe),'dump','--pid',str(proc.pid),'--nonblocking'],capture_output=True,text=True,timeout=30)
                    print(sampled.stdout,flush=True)
        return
    if '--stop-export' in sys.argv:
        for proc in psutil.process_iter(['pid','cmdline']):
            args=proc.info['cmdline'] or []
            if any(x.endswith('inspect_export_e012_v2c.py') for x in args):
                proc.terminate();proc.wait(timeout=30)
                print('Stopped only V2C snapshot validator',flush=True)
        return
    status=json.loads((ROOT/'stats/build_status.json').read_text())
    pid=status.get('native_pid')
    assert not pid or not psutil.pid_exists(pid), 'Scientific worker must be stopped'
    selected=[ROOT/'filtered/reservoir.sqlite',ROOT/'filtered/reservoir.sqlite-wal',
        ROOT/'filtered/reservoir.sqlite-shm',ROOT/'tmp/filter_checkpoint.json']
    selected=[p for p in selected if p.exists()]
    free=shutil.disk_usage(ROOT).free
    amount=sum(p.stat().st_size for p in selected)
    assert amount+5_000_000_000 < free*0.7
    frozen=OUT/'preservation/FROZEN_FILES.json'
    if not frozen.exists():
        records=[]
        for source in selected:
            dest=OUT/'preservation'/source.name
            before=source.stat()
            start=time.perf_counter()
            if not dest.exists():
                partial=dest.with_suffix(dest.suffix+'.partial')
                h=hashlib.sha256()
                with source.open('rb',buffering=8*1024**2) as reader,partial.open('wb',buffering=8*1024**2) as writer:
                    while block:=reader.read(8*1024**2):
                        h.update(block); writer.write(block)
                    writer.flush(); os.fsync(writer.fileno())
                digest=h.hexdigest()
                partial.replace(dest)
            else:
                digest=sha(source)
            assert sha(dest)==digest, 'Snapshot differs from original'
            after=source.stat()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            item={'original':str(source),'snapshot':str(dest),'bytes':before.st_size,'sha256':digest,
                'original_mtime_ns':before.st_mtime_ns,'copy_and_verify_seconds':time.perf_counter()-start}
            records.append(item)
            print(json.dumps(item),flush=True)
        write_json(frozen,{'original_files_unchanged':True,'free_before_bytes':free,'snapshot_bytes':amount,'files':records})
    manifest=json.loads(frozen.read_text())
    # Audit links refer only to the independent preserved copies, never originals.
    for name in ['reservoir.sqlite','reservoir.sqlite-wal']:
        source=OUT/'preservation'/name
        dest=OUT/'audit'/name
        if source.exists() and not dest.exists(): os.link(source,dest)
    dbpath=OUT/'audit/reservoir.sqlite'
    with closing(sqlite3.connect('file:'+dbpath.as_posix()+'?mode=ro',uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        state=json.loads(db.execute("SELECT v FROM progress WHERE k='checkpoint'").fetchone()[0])
        schema=[dict(zip(['type','name','table','rootpage','sql'],row)) for row in db.execute(
            "SELECT type,name,tbl_name,rootpage,sql FROM sqlite_master ORDER BY name")]
        metadata={'page_size':db.execute('PRAGMA page_size').fetchone()[0],
            'page_count':db.execute('PRAGMA page_count').fetchone()[0],
            'journal_mode':db.execute('PRAGMA journal_mode').fetchone()[0]}
    mirror=json.loads((OUT/'preservation/filter_checkpoint.json').read_text())
    keys=['raw','offset','stage','canonical','unique','duplicates','reservoir_count']
    audit={'requested_checkpoint':{'raw':30000000,'offset':15234569728,'stage':'external','canonical':23719152,'unique':23719152},
        'json_mirror':{k:mirror.get(k) for k in keys},'authoritative_sql':{k:state.get(k) for k in keys},
        'schema':schema,'database_metadata':metadata,'originals_opened_by_sqlite':False,
        'scientific_continuation_launched':False,'training_launched':False}
    write_json(OUT/'reports/checkpoint_audit.json',audit)
    write_json(OUT/'audit/authoritative_state.json',state)
    for item in manifest['files']:
        assert sha(Path(item['snapshot']))==item['sha256'], 'Read-only audit changed a preserved file'
    print(json.dumps(audit,indent=2),flush=True)
    # Relocate only temporary artifacts produced by the current V2B task.
    for name in ['e012_pyspy_release.json','e012_pyspy_0_4_1.whl','e012_pyspy.exe',
                 'e012_uniref50.release_note','e012_RELEASE.metalink']:
        source=Path(r'\\wsl.localhost\Ubuntu\tmp')/name
        dest=OUT/'tooling'/name
        if source.exists():
            digest=sha(source)
            if not dest.exists(): shutil.copyfile(source,dest)
            assert sha(dest)==digest
            source.unlink()
    write_json(OUT/'reports/storage_policy.json',{'root':str(OUT),'all_new_generated_artifacts_on_D':True,
        'temporary_directory':str(OUT/'tmp'),'original_reservoir_deleted_or_regenerated':False,
        'historical_data_untouched':True})

if __name__=='__main__': main()
