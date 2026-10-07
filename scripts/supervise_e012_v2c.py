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
        for script in ['report_benchmarks_e012_v2c.py','continue_e012_reservoir_v2c.py','union_e012_reservoir_v2c.py']:
            if (ROOT/'STOP_CONTINUATION').exists():raise RuntimeError('D: STOP_CONTINUATION requested')
            # The continuation command also independently checks the two-case >=5x gate.
            if script!='report_benchmarks_e012_v2c.py':
                assert (ROOT/'BENCHMARK_GATE_PASSED.json').exists(),'Equivalence/speedup gate did not pass'
            m.atomic_json(status,{'phase':'running','script':script,'pid':os.getpid(),
                'timestamp_unix':time.time(),'training_launched':False,'scientific_continuation_launched':script!='report_benchmarks_e012_v2c.py'})
            with (ROOT/'logs'/f'{script}.log').open('ab') as log:
                subprocess.run([sys.executable,'-B',str(Path(__file__).with_name(script))],cwd=ROOT,stdout=log,stderr=log,check=True)
        m.atomic_json(status,{'phase':'candidate_union_complete_protected_screen_pending','pid':os.getpid(),
            'timestamp_unix':time.time(),'training_launched':False,'scientific_continuation_launched':True,'corpus_ready':False})
    except BaseException as exc:
        m.atomic_json(status,{'phase':'blocked','error':repr(exc),'pid':os.getpid(),'timestamp_unix':time.time(),'training_launched':False})
        raise

if __name__=='__main__':
    if '--detach' in sys.argv:
        log=(ROOT/'logs/supervisor.log').open('ab')
        child=subprocess.Popen([sys.executable,'-B',__file__,sys.argv[-1]],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
            creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
        print(json.dumps({'supervisor_pid':child.pid,'data_root':str(ROOT),'training_launched':False}));log.close()
    else:main(int(sys.argv[-1]))
