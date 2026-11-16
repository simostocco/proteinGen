"""One-minute D:-only recovery heartbeat; observes workers without controlling them."""
from pathlib import Path
import importlib.util,json,os,subprocess,sys,time
import psutil

ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')
spec=importlib.util.spec_from_file_location('v2c',Path(__file__).with_name('e012_reservoir_v2c.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.configure_storage(ROOT)

def tail_json(path):
    if not path.exists():return None
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-65536));lines=f.read().splitlines()
    for line in reversed(lines):
        try:return json.loads(line)
        except (ValueError,UnicodeDecodeError):continue

def main():
    started=time.time()
    while True:
        status=json.loads((ROOT/'reports/RECOVERY_SUPERVISOR_STATUS.json').read_text())
        source=tail_json(ROOT/'continuation/progress.jsonl') or {}
        full=ROOT/'continuation/EXTERNAL_TAIL_COMPLETE.json'
        raw=m.RAW_COUNT if full.exists() else source.get('raw')
        workers=[]
        for proc in psutil.process_iter(['pid','cmdline']):
            args=proc.info['cmdline'] or []
            if any(str(x).endswith(name) for x in args for name in ['complete_e012_v2c_pipeline.py','continue_e012_reservoir_v2c.py','union_e012_reservoir_v2c.py','recover_e012_v2c.py']):
                try:
                    memory=proc.memory_info()
                    workers.append({'pid':proc.pid,'rss_bytes':memory.rss,'peak_rss_bytes':getattr(memory,'peak_wset',memory.rss),'argv':args})
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        row={'timestamp_unix':time.time(),'phase':status['phase'],'raw':raw,'raw_target':m.RAW_COUNT,
            'percent':100*raw/m.RAW_COUNT if raw else None,'remaining_raw_records':m.RAW_COUNT-raw if raw else None,
            'source_recent_records_s':0 if full.exists() else source.get('raw_per_second_recent'),
            'source_cumulative_records_s':source.get('raw_per_second'),'reservoir_count':source.get('reservoir_count'),
            'bloom_negatives':source.get('bloom_negatives'),'bloom_positives':source.get('bloom_positives'),
            'bloom_negative_fraction':source.get('bloom_negative_fraction'),'bloom_positive_fraction':source.get('bloom_positive_fraction'),
            'sqlite_fallback_queries':source.get('database_queries'),'native_worker_rss_bytes':sum(w['rss_bytes'] for w in workers),
            'ram_scope':'Native E012 Python workers only; Linux MMseqs RSS is recorded by GNU time in each batch command log',
            'd_free_bytes':__import__('shutil').disk_usage(ROOT).free,'source_elapsed_seconds':source.get('elapsed_seconds'),
            'source_eta_seconds':0 if full.exists() else source.get('eta_seconds'),'monitor_elapsed_seconds':time.time()-started,
            'workers':workers,'training_launched':False}
        for stage in ['screen','verify-protected']:
            event=tail_json(ROOT/'logs'/('recovery_'+stage+'.log'))
            if event and event.get('stage') in ['screen','verify_protected']:
                row[stage]=event
        m.atomic_json(ROOT/'reports/RECOVERY_HEARTBEAT.json',row)
        with (ROOT/'reports/RECOVERY_HEARTBEAT.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
        print(json.dumps(row),flush=True)
        if status['phase']=='frozen_corpus_ready':return
        if status['phase']=='blocked' and not psutil.pid_exists(status.get('pid',-1)):return
        time.sleep(60)

if __name__=='__main__':
    assert os.name=='nt'
    if '--detach' in sys.argv:
        with (ROOT/'logs/recovery_heartbeat.log').open('ab') as log:
            proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve())],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
        print(json.dumps({'heartbeat_pid':proc.pid,'interval_seconds':60,'root':str(ROOT)}),flush=True)
    else:main()
