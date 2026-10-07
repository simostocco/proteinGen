"""D:-only unattended gate supervisor for the authorized V2C continuation.

Wait for the already running 1M baseline; fail closed if it exits without a result.
Every launch and completion is recorded on D:. No training entry point is used.
"""
from pathlib import Path
import importlib.util,json,os,subprocess,sys,time
import psutil
spec=importlib.util.spec_from_file_location('v2c',Path(__file__).with_name('e012_reservoir_v2c.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')
m.configure_storage(ROOT)

def cleanup_certified_clones():
    verdict=json.loads((ROOT/'V2C_VERDICT.json').read_text());assert verdict['verdict']=='V2C-GO'
    for n in [100_000,1_000_000]:
        folder=ROOT/'benchmarks'/f'v2b_{n}/baseline'
        paths=list(folder.glob('*'));allowed={'reservoir.sqlite','reservoir.sqlite-wal','reservoir.sqlite-shm'}
        if not paths:continue
        assert all(p.name in allowed and p.is_file() and p.stat().st_nlink==1 for p in paths)
        files={str(p):{'sha256':m.sha_file(p),'bytes':p.stat().st_size} for p in paths}
        cert=ROOT/'reports'/f'private_clone_cleanup_{n}.json'
        m.atomic_json(cert,{'verdict_sha256':m.sha_file(ROOT/'V2C_VERDICT.json'),'replay_result_sha256':m.sha_file(folder.parent/'result.json'),
            'derivable_from_immutable_preserved_baseline_and_replay':True,'files':files})
        for path in paths:
            assert path.resolve().is_relative_to(ROOT/'benchmarks');path.unlink()

def main(pid):
    status=ROOT/'reports/SUPERVISOR_STATUS.json'
    try:
        while not (ROOT/'benchmarks/v2b_1000000/result.json').exists():
            if (ROOT/'STOP_CONTINUATION').exists():raise RuntimeError('D: STOP_CONTINUATION requested')
            proc=psutil.Process(pid)
            assert any(a.endswith('benchmark_e012_reservoir_v2c.py') for a in proc.cmdline()),'Baseline PID ownership mismatch'
            row={'phase':'waiting_for_1m_v2b','pid':os.getpid(),'baseline_pid':pid,
                'baseline_alive':proc.is_running(),'timestamp_unix':time.time(),'training_launched':False,'scientific_continuation_launched':False}
            progress=ROOT/'benchmarks/v2b_1000000/progress.jsonl'
            if progress.exists():
                lines=progress.read_text().splitlines()
                for line in reversed(lines):
                    try:row['baseline_progress']=json.loads(line);break
                    except json.JSONDecodeError:continue
            m.atomic_json(status,row);time.sleep(30)
        while not (ROOT/'reports/RESTART_REPLAY.json').exists():
            if (ROOT/'STOP_CONTINUATION').exists():raise RuntimeError('D: STOP_CONTINUATION requested')
            m.atomic_json(status,{'phase':'waiting_for_actual_restart_replay','pid':os.getpid(),'training_launched':False,'scientific_continuation_launched':False})
            time.sleep(30)
        stages=[('gate',['certify_e012_v2c_gate.py']),('external',['continue_e012_reservoir_v2c.py']),('historical_union',['union_e012_reservoir_v2c.py'])]
        stages.extend((s,['complete_e012_v2c_pipeline.py',s]) for s in ['reservoir-finalize','screen','verify-protected','final','certify','diversity','handoff'])
        for stage,args in stages:
            script=args[0]
            if (ROOT/'STOP_CONTINUATION').exists():raise RuntimeError('D: STOP_CONTINUATION requested')
            # The continuation command also independently checks the two-case >=5x gate.
            if stage!='gate':
                assert json.loads((ROOT/'V2C_VERDICT.json').read_text())['verdict']=='V2C-GO','Frozen gate did not pass'
            if stage=='external':cleanup_certified_clones()
            m.atomic_json(status,{'phase':'running','script':script,'pid':os.getpid(),
                'timestamp_unix':time.time(),'training_launched':False,'scientific_continuation_launched':stage!='gate'})
            started=time.time()
            command=[sys.executable,'-B',str(Path(__file__).with_name(script))]+args[1:]
            if stage=='certify':
                command=['C:/Windows/System32/wsl.exe','--distribution','Ubuntu','--cd',str(ROOT).replace('\\','/').replace('D:','/mnt/d'),
                    '--exec','/home/simostocco/miniforge3/envs/proteingen/bin/python','-B',
                    '/home/simostocco/proteinGen-causal-rope/scripts/complete_e012_v2c_pipeline.py','certify']
            with (ROOT/'logs'/f'{stage}.log').open('ab') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=log,check=True)
            timing=ROOT/'reports'/f'timing_{stage}.json'
            if not timing.exists():m.atomic_json(timing,{'stage':stage,'wall_seconds':time.time()-started,'started_unix':started,'ended_unix':time.time(),'training_launched':False})
        m.atomic_json(status,{'phase':'frozen_corpus_ready','pid':os.getpid(),
            'timestamp_unix':time.time(),'training_launched':False,'scientific_continuation_launched':True,'corpus_ready':True})
    except BaseException as exc:
        m.atomic_json(status,{'phase':'blocked','error':repr(exc),'pid':os.getpid(),'timestamp_unix':time.time(),'training_launched':False})
        if not (ROOT/'V2C_VERDICT.json').exists():
            m.atomic_json(ROOT/'V2C_VERDICT.json',{'verdict':'V2C-NOGO','discrepancy':repr(exc),'training_launched':False,'timestamp_unix':time.time()})
        raise

if __name__=='__main__':
    if '--detach' in sys.argv:
        log=(ROOT/'logs/supervisor.log').open('ab')
        child=subprocess.Popen([sys.executable,'-B',__file__,sys.argv[-1]],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
            creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
        print(json.dumps({'supervisor_pid':child.pid,'data_root':str(ROOT),'training_launched':False}));log.close()
    else:main(int(sys.argv[-1]))
